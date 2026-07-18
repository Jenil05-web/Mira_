"""
pipeline/ambient.py
====================
MIRA Ambient Consult Mode — Phase 1 (Batch mode)

One tap to start, one tap to end. MIRA transcribes the whole visit,
extracts clinical entities, surfaces relevant guidelines, and drafts
a SOAP note for doctor review.

New pieces (everything else reuses existing MIRA architecture):
  - AmbientConsultState  (parallel to MIRAState, not a replacement)
  - agent0_entity_extractor  (new — extracts symptoms/meds/history)
  - agent_note_synthesizer   (repurposed report writer → SOAP format)
  - agent_ambient_critic     (same hallucination guard, new prompt)

Reused as-is:
  - Whisper transcription (same engine.transcribe_voice_query path)
  - CONDITION_VOCAB matching (via pipeline.tools)
  - VectorStore / guideline search (Agent 2 equivalent)
  - Human approve/reject gate (same UX pattern)
"""

import json
import logging
import time
import uuid
from typing import Optional

from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class AmbientConsultState:
    """
    Mutable state object for one ambient consult session.
    Not a TypedDict (unlike MIRAState) because it lives in Streamlit
    session_state rather than a LangGraph checkpoint.
    """
    def __init__(self, session_id: str = "", user_id: str = "anon",
                 hospital_id: str = "demo"):
        self.session_id        = session_id or str(uuid.uuid4())
        self.user_id           = user_id
        self.hospital_id       = hospital_id
        self.patient_id        = None           # nullable — walk-ins have no MRN
        self.consent_confirmed = False
        self.input_mode        = "ambient_voice"

        # Transcript
        self.raw_transcript    = ""
        self.spoken_language   = ""

        # Entity extraction (Agent 0)
        self.structured_entities: dict = {}    # {symptoms, duration, meds, allergies, history}
        self.matched_conditions: list  = []

        # Guideline surfacing (reused Agent 2)
        self.retrieved_guidelines = ""

        # Note synthesis (Agent 3b)
        self.draft_note  = ""
        self.final_note  = ""
        self.doctor_edits = ""

        # Safety / hallucination guard (reused Agent 4)
        self.safety_flags: list = []
        self.approved    = False

        # Timing
        self.started_at  = None
        self.ended_at    = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ─────────────────────────────────────────────────────────────────────────────
# AMBIENT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AmbientConsultEngine:
    """
    Thin wrapper around the existing MIRAEngineProd that adds
    the ambient-mode processing chain.
    Usage:
        ambient = AmbientConsultEngine(mira_engine)
        state = ambient.new_session(user_id, hospital_id)
        # ... doctor does consult, audio is captured in UI ...
        state = ambient.process_consult(state, audio_bytes_or_transcript)
        # state.draft_note is now ready for review
        state = ambient.approve(state)
    """

    def __init__(self, engine):
        """
        engine: the existing MIRAEngineProd singleton — we reuse its
                llm, openai_client, vector_store, and audit logger.
        """
        self.engine       = engine
        self.llm          = engine.llm
        self.openai_client = engine.openai_client
        self.vector_store  = engine.vector_store
        self.audit         = engine.audit

    def new_session(self, user_id: str, hospital_id: str,
                    patient_id: Optional[str] = None) -> AmbientConsultState:
        state = AmbientConsultState(user_id=user_id, hospital_id=hospital_id)
        state.patient_id = patient_id
        state.consent_confirmed = True          # caller must enforce consent gate in UI
        state.started_at = time.time()
        logger.info(f"Ambient session started: {state.session_id}")
        self.audit.log_tool_call("ambient_consult_start", state.session_id, 0, 0, True)
        return state

    # ── Step 1: transcribe (reuses existing Whisper path) ─────────────────
    def transcribe(self, state: AmbientConsultState,
                   audio_bytes: bytes) -> AmbientConsultState:
        try:
            result = self.engine.transcribe_voice_query(audio_bytes)
            state.raw_transcript  = result.get("text", "")
            state.spoken_language = result.get("spoken_language", "english")
        except Exception as e:
            logger.error(f"Ambient transcription failed: {e}")
            state.raw_transcript = ""
        return state

    # ── Step 2: entity extraction (Agent 0) ───────────────────────────────
    def agent0_entity_extractor(self, state: AmbientConsultState) -> AmbientConsultState:
        if not state.raw_transcript.strip():
            state.structured_entities = {}
            return state

        prompt = f"""You are a clinical information extractor. 
Read the following doctor-patient consultation transcript and extract structured clinical information.

TRANSCRIPT:
{state.raw_transcript}

Return ONLY a valid JSON object with these exact keys (use empty list [] if nothing found):
{{
  "chief_complaint": "one-sentence summary of why patient came in",
  "symptoms": ["list of symptoms mentioned"],
  "symptom_duration": "how long symptoms have been present",
  "severity": "mild/moderate/severe if mentioned, else null",
  "patient_history": ["relevant past medical history mentioned"],
  "current_medications": ["medications patient is currently taking"],
  "allergies": ["drug or other allergies mentioned"],
  "vitals_mentioned": ["any vital signs mentioned verbally"],
  "doctor_observations": ["clinical observations the doctor states"],
  "patient_age_gender": "age and gender if mentioned, else null"
}}

Return ONLY the JSON. No markdown, no explanation."""

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            content = response.content.replace("```json", "").replace("```", "").strip()
            state.structured_entities = json.loads(content)
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            state.structured_entities = {
                "chief_complaint": "Unable to extract — see raw transcript",
                "symptoms": [], "symptom_duration": "", "severity": None,
                "patient_history": [], "current_medications": [], "allergies": [],
                "vitals_mentioned": [], "doctor_observations": [],
                "patient_age_gender": None
            }

        # Match extracted symptoms against CONDITION_VOCAB
        try:
            from pipeline.tools import CONDITION_VOCAB
            all_text = " ".join([
                " ".join(state.structured_entities.get("symptoms", [])),
                state.structured_entities.get("chief_complaint", ""),
                " ".join(state.structured_entities.get("doctor_observations", [])),
            ]).lower()
            matched = [
                cond for cond, meta in CONDITION_VOCAB.items()
                if any(alias in all_text for alias in meta.get("aliases", [cond]))
            ]
            state.matched_conditions = matched
        except Exception:
            state.matched_conditions = []

        return state

    # ── Step 3: guideline surfacing (reuses VectorStore / Agent 2) ────────
    def agent_guideline_search(self, state: AmbientConsultState) -> AmbientConsultState:
        entities = state.structured_entities
        symptoms = ", ".join(entities.get("symptoms", []))
        complaint = entities.get("chief_complaint", "")
        conditions = ", ".join(state.matched_conditions)

        search_query = f"{complaint}. Symptoms: {symptoms}. Conditions: {conditions}".strip(". ")
        if not search_query:
            state.retrieved_guidelines = ""
            return state

        try:
            state.retrieved_guidelines = self.vector_store.search(search_query, k=4)
        except Exception as e:
            logger.error(f"Ambient guideline search failed: {e}")
            state.retrieved_guidelines = ""

        return state

    # ── Step 4: SOAP note synthesis (Agent 3b) ────────────────────────────
    def agent_note_synthesizer(self, state: AmbientConsultState) -> AmbientConsultState:
        entities   = state.structured_entities
        transcript = state.raw_transcript
        guidelines = state.retrieved_guidelines or "No specific guidelines retrieved."

        # Build patient context string
        patient_ctx = ""
        if state.patient_id:
            patient_ctx = f"Patient ID / MRN: {state.patient_id}\n"
        elif entities.get("patient_age_gender"):
            patient_ctx = f"Patient: {entities['patient_age_gender']}\n"
        else:
            patient_ctx = "Patient: Walk-in (no MRN on record)\n"

        prompt = f"""You are an expert clinical documentation specialist.
Generate a structured SOAP clinical note from the following consultation data.

{patient_ctx}

EXTRACTED CLINICAL ENTITIES:
Chief Complaint: {entities.get('chief_complaint', 'Not specified')}
Symptoms: {', '.join(entities.get('symptoms', []))}
Duration: {entities.get('symptom_duration', 'Not specified')}
Severity: {entities.get('severity', 'Not specified')}
Patient History: {', '.join(entities.get('patient_history', []))}
Current Medications: {', '.join(entities.get('current_medications', []))}
Allergies: {', '.join(entities.get('allergies', []))}
Vitals Mentioned: {', '.join(entities.get('vitals_mentioned', []))}
Doctor Observations: {', '.join(entities.get('doctor_observations', []))}

RELEVANT CLINICAL GUIDELINES:
{guidelines}

FULL TRANSCRIPT (for reference):
{transcript[:3000]}

Generate a professional SOAP note in the following format:

**SUBJECTIVE**
[Patient's reported complaints, history, and symptoms in their own words]

**OBJECTIVE**
[Physical examination findings, vital signs, and clinical observations]

**ASSESSMENT**
[Clinical assessment — likely diagnosis or differential, with reasoning grounded in what was said]

**PLAN**
[Recommended investigations, treatments, medications, follow-up actions]

**GUIDELINES REFERENCED**
[Briefly note which guidelines informed the Assessment/Plan, if any]

CRITICAL RULES:
- Only include information that was actually mentioned in the transcript or extracted entities.
- Do NOT invent symptoms, findings, or history that were not stated.
- If information for any section is absent, write "Not documented during this encounter."
- Keep the tone clinical and concise — this will be filed as a medical record."""

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            state.draft_note = response.content.strip()
        except Exception as e:
            logger.error(f"SOAP note synthesis failed: {e}")
            state.draft_note = "Note synthesis failed. Please review the transcript manually."

        return state

    # ── Step 5: hallucination / mismatch guard (reused Agent 4 pattern) ───
    def agent_ambient_critic(self, state: AmbientConsultState) -> AmbientConsultState:
        if not state.draft_note or not state.raw_transcript:
            state.safety_flags = ["no_content"]
            return state

        prompt = f"""You are a clinical documentation safety reviewer.

Your job: verify that every factual claim in the SOAP NOTE below is traceable 
to something actually said in the CONSULTATION TRANSCRIPT.

SOAP NOTE:
{state.draft_note}

CONSULTATION TRANSCRIPT:
{state.raw_transcript[:3000]}

Check for:
1. Symptoms or findings mentioned in the note but NOT in the transcript
2. Medications mentioned in the note but NOT in the transcript  
3. Diagnoses stated with certainty that are not supported by the transcript
4. Patient demographics that contradict what was said

Return a JSON object with:
{{
  "flags": ["list of specific issues found, e.g. 'symptom_invented: chest pain not mentioned'"],
  "safe_to_file": true/false,
  "reviewer_note": "one sentence summary for the doctor"
}}

If the note accurately reflects the transcript, return:
{{"flags": [], "safe_to_file": true, "reviewer_note": "Note accurately reflects the consultation."}}

Return ONLY the JSON."""

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            content = response.content.replace("```json", "").replace("```", "").strip()
            review = json.loads(content)
            state.safety_flags = review.get("flags", [])
            if not review.get("safe_to_file", True):
                state.safety_flags.append("hallucination_detected")
        except Exception as e:
            logger.error(f"Ambient critic failed: {e}")
            state.safety_flags = ["critic_error"]

        return state

    # ── Main entry point: run full batch pipeline ──────────────────────────
    def process_consult(self, state: AmbientConsultState,
                        audio_bytes: Optional[bytes] = None,
                        transcript_text: Optional[str] = None) -> AmbientConsultState:
        """
        Run the full post-consult pipeline:
        transcribe → entity extract → guideline search → SOAP note → critic

        Pass either audio_bytes (raw WAV/MP3) OR a pre-made transcript_text.
        """
        state.ended_at = time.time()

        # 1. Transcribe (or accept pre-made text)
        if audio_bytes:
            state = self.transcribe(state, audio_bytes)
        elif transcript_text:
            state.raw_transcript = transcript_text

        if not state.raw_transcript.strip():
            state.draft_note = "No transcript content to process."
            return state

        # 2. Entity extraction
        state = self.agent0_entity_extractor(state)

        # 3. Guideline search
        state = self.agent_guideline_search(state)

        # 4. SOAP note synthesis
        state = self.agent_note_synthesizer(state)

        # 5. Critic / hallucination guard
        state = self.agent_ambient_critic(state)

        self.audit.log_tool_call(
            "ambient_consult_processed", state.session_id,
            int((state.ended_at - state.started_at) * 1000), 1,
            True
        )
        return state

    def approve(self, state: AmbientConsultState,
                doctor_edits: str = "") -> AmbientConsultState:
        """Doctor signs off. Merges edits into final_note."""
        if doctor_edits.strip():
            state.doctor_edits = doctor_edits
            state.final_note   = doctor_edits
        else:
            state.final_note = state.draft_note
        state.approved = True
        self.audit.log_tool_call("ambient_consult_approved", state.session_id, 0, 1, True)
        return state
