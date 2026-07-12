"""
mira_pipeline_prod.py
======================
MIRA Production Pipeline — ENHANCED.

DROP-IN REPLACEMENT for the previous mira_pipeline_prod.py. Same public API
(get_engine, MIRAEngineProd, run_until_review, submit_human_decision,
new_thread, get_current_state) — your Streamlit app does not need to change
its calling conventions, though a couple of new optional helpers are added
(see "NEW PUBLIC API" near the bottom).

WHAT WAS FIXED IN THIS VERSION
-------------------------------
1. REVISION NOT WORKING
   The old code computed a revised report on "reject" but never wrote it
   back into the LangGraph checkpoint (`graph.update_state`). Any later
   read of state (e.g. a page rerun calling get_current_state) saw the
   stale pre-revision report. Fixed: every revision path now persists via
   `graph.update_state(...)` and returns the checkpoint's own values, so
   what you see is guaranteed to be what's stored.

2. "NA, no patient found" -> THEN A RANDOM UNRELATED PATIENT REPORT
   The old SQL agent silently swapped in an unrelated generic "abnormal
   patients" query when a specifically-requested patient ID had no rows,
   and then wrote a full report as if that unrelated patient was the one
   asked about. Fixed: we now (a) explicitly check whether the patient
   exists at all, (b) never blend an unrelated patient into a
   single-patient answer, and (c) if we do have to broaden a *list* query,
   we tag that fact in the state so the report says so honestly instead
   of pretending it's an exact match.

3. SINGLE RESULT FOR PLURAL QUESTIONS ("which patients show signs of AKI")
   `should_retry_sql` always returned "ok", so the retry edge in the graph
   was dead code. Fixed: a real intent parser detects list-style questions
   ("patients", "which patients", "list", "how many", multiple IDs) and a
   working retry loop broadens the query (up to MAX_SQL_RETRIES times)
   until enough distinct patients are found or retries are exhausted —
   and if genuinely only one patient qualifies, the report says so
   explicitly instead of silently acting like nothing was wrong.

4. APPROVE FINALIZING A FABRICATED REPORT WHEN THERE WAS NO DATA
   The MemorySaver-lost-state recovery path trusted a frontend-supplied
   `paused_state` blindly, even if it belonged to a *different* question
   (a classic stale-cache bug). Fixed: that recovery path now checks the
   `clinical_question` actually matches before reusing anything, and the
   no-data guard runs before any report can be marked approved.

5. NO CLINICAL VOCABULARY / "COMMON SENSE"
   Added a CONDITION_VOCAB dictionary (AKI, sepsis, hyperkalemia,
   hypokalemia, anemia, hyperglycemia, hypoglycemia, hypoxia, liver
   injury, thrombocytopenia, leukocytosis) that gets folded into both the
   SQL-generation prompt and the semantic search query whenever the
   question mentions (or implies) one of these, so vague, real-world
   phrasing ("signs of AKI", "is this patient septic?") maps to the right
   labs without the user having to spell out lab names.

6. HALLUCINATION / MISMATCH GUARD
   The critic agent now cross-checks patient IDs mentioned in the final
   report against patient IDs actually present in the SQL results, and
   flags (rather than silently ships) any mismatch.

7. "START NEW AUDIT" SUPPORT
   Added `start_new_audit()` — returns a brand-new thread config and does
   not reuse any cached per-thread state, so a "New Audit" button in your
   UI has a clean, single-call way to reset. (See NEW PUBLIC API below —
   this is a backend hook; wiring an actual button is a one-line change in
   your Streamlit file, shown in the docstring for that method.)

8. PERSISTENT CHECKPOINTS
   `config_manager.get_checkpoint_config()` already returns a proper
   Postgres connection string whenever Supabase is configured, but it was
   never actually consulted — every deployment silently ran on
   MemorySaver, meaning every audit (including in-progress revisions) was
   lost on any restart. Fixed: `_build_checkpointer()` now uses
   PostgresSaver whenever Supabase is available, falling back to
   MemorySaver with a clear log warning otherwise.

9. MULTILINGUAL VOICE INPUT (English / Hindi / Gujarati)
   Added `transcribe_voice_query()` using Whisper. Non-English audio is
   translated to English via Whisper's /translations endpoint (everything
   downstream — SQL generation, condition vocabulary, patient-ID
   extraction — only understands English medical terms), while the
   verbatim transcript in the original language is always preserved
   in `original_transcript` for the audit trail. New MIRAState fields:
   `input_mode`, `spoken_language`, `original_transcript`.

10. GUIDELINE SEARCH CACHING (cost control)
   Added a small in-memory cache in `VectorStore.search()` so a "Request
   revision" loop — which often re-runs a very similar guideline search —
   doesn't re-pay for the same embedding + search repeatedly in one
   session. Deliberately simple (single-instance, in-memory); revisit with
   a shared cache once you're running more than one instance.

11. AUTO-DETECT VOICE LANGUAGE WAS SILENTLY BROKEN
   `transcribe_voice_query(spoken_language="auto")` called Whisper's
   transcription endpoint without `response_format="verbose_json"`, so
   the response never actually carried a `language` field. That made
   `detected_lang` fall back to `"en"` on every single call — regardless
   of what was actually spoken — which meant the `!= "en"` check that
   should trigger translation NEVER fired. Hindi/Gujarati recordings were
   silently passed straight into the SQL/condition-vocabulary agents in
   their original language (which those agents don't understand), and the
   UI would always report "English" as the detected language no matter
   what was said. Fixed: the transcription call now requests
   `response_format="verbose_json"` so `language` is actually populated,
   and the comparison uses Whisper's real output format — the full
   language name in lowercase English (e.g. "english", "hindi",
   "gujarati"), not an ISO code, which is what Whisper actually returns.

BACKWARDS COMPATIBLE:
  If no Supabase credentials are set, the pipeline behaves exactly like
  before — SQLite + FAISS + MemorySaver.

INSTALL (production extras beyond base requirements):
  pip install sqlalchemy psycopg2-binary langgraph-checkpoint-postgres
  # ^ langgraph-checkpoint-postgres is REQUIRED for persistent checkpoints
  #   once Supabase is configured (see _build_checkpointer below). Without
  #   it, the app falls back to MemorySaver and logs a warning — it will
  #   still run, but every audit session is lost on restart/redeploy.
"""

import io
import json
import logging
import os
import pickle
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Generator, Optional, TypedDict

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from config_manager import ConfigManager
from db_adapter import DBAdapter, create_adapter
from fhir_adapter import FHIRAdapter
from audit_logger import AuditLogger
from trend_agent import TrendAgent


logger = logging.getLogger(__name__)

MAX_SQL_RETRIES = 2          # how many times we'll broaden a list query
MIN_LIST_PATIENTS = 2        # below this, a "plural" question is treated as under-served

# NEW — voice-input languages. Whisper's /translations endpoint always
# outputs English regardless of spoken language, which is exactly what we
# want here: the SQL/condition-vocabulary agents downstream only understand
# English medical terms, so non-English speech is translated on the way in
# rather than requiring the rest of the pipeline to be multilingual-aware.
VOICE_LANGUAGE_CODES = {"english": "en", "hindi": "hi", "gujarati": "gu"}


# ══════════════════════════════════════════════════════════════════════════
# CLINICAL VOCABULARY — maps common real-world phrasing to lab hints
# ══════════════════════════════════════════════════════════════════════════

CONDITION_VOCAB: dict[str, dict] = {
    "aki": {
        "aliases": ["aki", "acute kidney injury", "kidney injury", "renal failure", "renal injury"],
        "sql_hint": (
            "For acute kidney injury (AKI): look at creatinine "
            "(d.label ILIKE '%creatinine%') where valuenum is elevated above "
            "ref_range_upper, and/or BUN (d.label ILIKE '%urea nitrogen%' OR "
            "d.label ILIKE '%bun%'). A rising creatinine trend is the strongest signal."
        ),
        "search_terms": "acute kidney injury creatinine elevation diagnostic criteria staging",
    },
    "sepsis": {
        "aliases": ["sepsis", "septic", "septicemia"],
        "sql_hint": (
            "For sepsis concerns: look at lactate (d.label ILIKE '%lactate%') elevated "
            "above ref_range_upper, WBC (d.label ILIKE '%white blood cell%' OR "
            "d.label ILIKE '%wbc%') abnormal high or low, and any temperature/vitals labs present."
        ),
        "search_terms": "sepsis criteria lactate elevated white blood cell abnormal",
    },
    "hyperkalemia": {
        "aliases": ["hyperkalemia", "high potassium", "elevated potassium"],
        "sql_hint": "For hyperkalemia: d.label ILIKE '%potassium%' AND l.valuenum > l.ref_range_upper.",
        "search_terms": "hyperkalemia elevated potassium management",
    },
    "hypokalemia": {
        "aliases": ["hypokalemia", "low potassium"],
        "sql_hint": "For hypokalemia: d.label ILIKE '%potassium%' AND l.valuenum < l.ref_range_lower.",
        "search_terms": "hypokalemia low potassium management",
    },
    "anemia": {
        "aliases": ["anemia", "anaemia", "low hemoglobin", "low hematocrit"],
        "sql_hint": (
            "For anemia: d.label ILIKE '%hemoglobin%' OR d.label ILIKE '%hematocrit%', "
            "with valuenum < ref_range_lower."
        ),
        "search_terms": "anemia low hemoglobin hematocrit workup",
    },
    "hyperglycemia": {
        "aliases": ["hyperglycemia", "high glucose", "high blood sugar"],
        "sql_hint": "For hyperglycemia: d.label ILIKE '%glucose%' AND l.valuenum > l.ref_range_upper.",
        "search_terms": "hyperglycemia elevated glucose management",
    },
    "hypoglycemia": {
        "aliases": ["hypoglycemia", "low glucose", "low blood sugar"],
        "sql_hint": "For hypoglycemia: d.label ILIKE '%glucose%' AND l.valuenum < l.ref_range_lower.",
        "search_terms": "hypoglycemia low glucose management",
    },
    "hypoxia": {
        "aliases": ["hypoxia", "hypoxemia", "low oxygen"],
        "sql_hint": "For hypoxia: d.label ILIKE '%oxygen%' OR d.label ILIKE '%pao2%' OR d.label ILIKE '%sao2%', with low valuenum.",
        "search_terms": "hypoxia low oxygen saturation management",
    },
    "liver injury": {
        "aliases": ["liver injury", "hepatic injury", "liver failure", "hepatic failure"],
        "sql_hint": (
            "For liver injury: d.label ILIKE '%bilirubin%' OR d.label ILIKE '%alt%' OR "
            "d.label ILIKE '%ast%', with valuenum > ref_range_upper."
        ),
        "search_terms": "liver injury elevated bilirubin transaminase",
    },
    "thrombocytopenia": {
        "aliases": ["thrombocytopenia", "low platelets"],
        "sql_hint": "For thrombocytopenia: d.label ILIKE '%platelet%' AND l.valuenum < l.ref_range_lower.",
        "search_terms": "thrombocytopenia low platelet count causes",
    },
    "leukocytosis": {
        "aliases": ["leukocytosis", "high white blood cell", "high wbc"],
        "sql_hint": "For leukocytosis: d.label ILIKE '%white blood cell%' AND l.valuenum > l.ref_range_upper.",
        "search_terms": "leukocytosis elevated white blood cell count causes",
    },
}

LIST_INTENT_PHRASES = [
    "which patients", "list of patients", "list patients", "all patients",
    "how many patients", "every patient", "show me patients", "show patients",
    "patients who", "patients with", "patients show", "patients showing",
]


# ══════════════════════════════════════════════════════════════════════════
# INTENT PARSER — turns free-form questions into structured hints
# ══════════════════════════════════════════════════════════════════════════

def parse_clinical_intent(question: str) -> dict:
    """
    Lightweight NLU: extracts patient ID(s), list-vs-single intent, and any
    recognized clinical conditions, so downstream agents don't need the
    user to spell out exact lab names or phrasing.
    """
    q = (question or "").lower()

    # ---- Patient ID extraction (supports single or multiple IDs) ----
    id_block_match = re.search(
        r"patients?\s*(?:id[s]?)?\s*[:#]?\s*((?:\d{2,}\s*(?:,|and|&)?\s*)+)", q
    )
    patient_ids: list[int] = []
    if id_block_match:
        patient_ids = [int(x) for x in re.findall(r"\d{2,}", id_block_match.group(1))]

    # ---- List vs single-patient intent ----
    has_plural_word = bool(re.search(r"\bpatients\b", q))
    has_list_phrase = any(p in q for p in LIST_INTENT_PHRASES)
    is_list = (has_list_phrase or (has_plural_word and len(patient_ids) != 1) or len(patient_ids) > 1)
    # If exactly one ID was named explicitly, that overrides "list" framing —
    # the user wants that one patient, even if they said "patient's" etc.
    if len(patient_ids) == 1 and not has_list_phrase:
        is_list = False

    # ---- Condition vocabulary matching ----
    matched_conditions = []
    for key, entry in CONDITION_VOCAB.items():
        if any(alias in q for alias in entry["aliases"]):
            matched_conditions.append(key)

    return {
        "raw_question": question,
        "patient_ids": patient_ids,
        "is_list": is_list,
        "conditions": matched_conditions,
    }


def _condition_sql_hints(conditions: list[str]) -> str:
    if not conditions:
        return ""
    lines = [CONDITION_VOCAB[c]["sql_hint"] for c in conditions if c in CONDITION_VOCAB]
    return "\n".join(lines)


def _condition_search_terms(conditions: list[str]) -> str:
    if not conditions:
        return ""
    return " ".join(CONDITION_VOCAB[c]["search_terms"] for c in conditions if c in CONDITION_VOCAB)


def _distinct_patient_ids(sql_result_json: str) -> set:
    try:
        rows = json.loads(sql_result_json).get("rows", [])
        return {r["subject_id"] for r in rows if r.get("subject_id") is not None}
    except Exception:
        return set()


# ══════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════

class MIRAState(TypedDict):
    clinical_question:  str
    sql_query_used:     str
    sql_result:         str
    sql_retry_count:    int
    sql_error:          str
    search_query_used:  str
    guidelines:         str
    trend_data:         str
    clinical_reasoning: str
    final_report:       str
    safety_flags:       list[str]
    approved:           bool
    human_decision:     str
    human_feedback:     str
    # Production additions
    user_id:            str
    hospital_id:        str
    session_id:         str
    # NEW — intent + data-quality tracking
    query_intent:        dict
    data_status:         str   # "ok" | "patient_not_found" | "no_data" | "broadened_query"
    requested_patient_ids: list
    found_patient_ids:     list
    missing_patient_ids:   list
    # NEW — voice-input provenance (for audit trail transparency: a
    # clinician or hospital reviewer should always be able to see the
    # original spoken wording, not just whatever it was translated into)
    input_mode:           str   # "text" | "voice"
    spoken_language:      str   # "english" | "hindi" | "gujarati" | ""
    original_transcript:  str   # raw transcript before any translation


def make_initial_state(clinical_question: str,
                       user_id: str = "anon",
                       hospital_id: str = "demo",
                       session_id: str = "",
                       input_mode: str = "text",
                       spoken_language: str = "",
                       original_transcript: str = "") -> MIRAState:
    return {
        "clinical_question": clinical_question,
        "sql_query_used": "", "sql_result": "", "sql_retry_count": 0, "sql_error": "",
        "search_query_used": "", "guidelines": "", "trend_data": "",
        "clinical_reasoning": "",
        "final_report": "", "safety_flags": [], "approved": False,
        "human_decision": "", "human_feedback": "",
        "user_id": user_id,
        "hospital_id": hospital_id,
        "session_id": session_id or str(uuid.uuid4()),
        "query_intent": {},
        "data_status": "",
        "requested_patient_ids": [],
        "found_patient_ids": [],
        "missing_patient_ids": [],
        "input_mode": input_mode,
        "spoken_language": spoken_language,
        "original_transcript": original_transcript,
    }


# ══════════════════════════════════════════════════════════════════════════
# VECTOR STORE — pgvector or FAISS, resolved at runtime
# ══════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Unified vector search interface.
    Uses pgvector (Supabase) if configured, FAISS otherwise.
    Both return the same result shape.
    """

    def __init__(self, cfg: ConfigManager, openai_client: OpenAI):
        self._client = openai_client
        self._embedding_model = "text-embedding-3-small"
        vs_cfg = cfg.get_vector_store_config()
        self._type = vs_cfg["type"]
        # NEW — cost control: a "Request revision" loop re-runs
        # agent2_semantic_crossref, which often produces the same or a very
        # similar search query as before. Caching avoids paying for a fresh
        # embedding + search on every revision round on the same session.
        # Deliberately small and in-memory — fine for a single-instance demo;
        # revisit with a shared cache (e.g. Redis) once running multi-instance.
        self._search_cache: dict[tuple, list[dict]] = {}
        self._search_cache_max = 200

        if self._type == "pgvector":
            from sqlalchemy import create_engine, text
            self._engine = create_engine(vs_cfg["connection_string"])
            self._faiss_index = None
            self._metadata = None
            logger.info("VectorStore: using Supabase pgvector")
        else:
            # FAISS fallback
            idx_path = vs_cfg.get("index_path", "./mira_data/medical_faiss.index")
            meta_path = vs_cfg.get("metadata_path", "./mira_data/faiss_metadata.pkl")
            self._faiss_index = faiss.read_index(str(idx_path))
            with open(meta_path, "rb") as f:
                self._metadata = pickle.load(f)
            self._engine = None
            logger.info("VectorStore: using local FAISS")

    def _embed(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(
            model=self._embedding_model, input=[text]
        )
        return np.array([resp.data[0].embedding], dtype=np.float32)

    def search(self, query: str, k: int = 3,
               hospital_id: str = "global") -> list[dict]:
        cache_key = (query.strip().lower(), k, hospital_id)
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._type == "pgvector":
            results = self._search_pgvector(query, k, hospital_id)
        else:
            results = self._search_faiss(query, k)

        if len(self._search_cache) >= self._search_cache_max:
            self._search_cache.pop(next(iter(self._search_cache)))  # evict oldest
        self._search_cache[cache_key] = results
        return results

    def _search_pgvector(self, query: str, k: int, hospital_id: str) -> list[dict]:
        from sqlalchemy import text
        vec = self._embed(query)[0].tolist()
        vec_str = "[" + ",".join(str(x) for x in vec) + "]"
        with self._engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT source, topic, content,
                       1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM mira_embeddings
                WHERE hospital_id = :hid OR hospital_id = 'global'
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :k
            """), {"vec": vec_str, "hid": hospital_id, "k": k}).fetchall()
        return [
            {"source": r.source, "topic": r.topic, "text": r.content,
             "rank": i + 1, "relevance_score": round(float(r.similarity), 4)}
            for i, r in enumerate(rows)
        ]

    def _search_faiss(self, query: str, k: int) -> list[dict]:
        vec = self._embed(query)
        distances, indices = self._faiss_index.search(vec, k)
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            chunk = self._metadata[idx].copy()
            chunk["rank"] = rank + 1
            chunk["relevance_score"] = round(1 / (1 + float(dist)), 4)
            results.append(chunk)
        return results


# ══════════════════════════════════════════════════════════════════════════
# PRODUCTION ENGINE
# ══════════════════════════════════════════════════════════════════════════

class MIRAEngineProd:
    """
    Production MIRA engine.
    Accepts any hospital_id — each gets its own data source config.
    All agent calls are HIPAA-audited automatically.
    """

    def __init__(self, cfg: Optional[ConfigManager] = None):
        self.cfg = cfg or ConfigManager()
        os.environ["OPENAI_API_KEY"] = self.cfg.openai_api_key

        # ── OpenAI clients ───────────────────────────────────────────────
        self.openai_client = OpenAI(api_key=self.cfg.openai_api_key)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

        # ── Audit logger ─────────────────────────────────────────────────
        audit_cfg = self.cfg.get_audit_config()
        self.audit = AuditLogger(
            connection_string=audit_cfg["connection_string"],
            enabled=audit_cfg["enabled"],
        )

        # ── Vector store (pgvector or FAISS) ────────────────────────────
        self.vector_store = VectorStore(self.cfg, self.openai_client)

        # ── Checkpointer (PostgresSaver or MemorySaver) ──────────────────
        # FIX: config_manager.get_checkpoint_config() already returns a
        # proper Postgres connection string whenever Supabase is
        # configured, but this was never actually consulted — every
        # deployment silently ran on MemorySaver, meaning every audit
        # session (including in-progress revisions) was lost on any
        # Render restart or redeploy. Now we use PostgresSaver whenever
        # Supabase is configured, and fall back to MemorySaver only for
        # local dev with no Supabase set up.
        self.checkpointer = self._build_checkpointer()

        # ── Per-hospital data adapters (lazy, cached) ─────────────────
        self._adapters: dict[str, object] = {}

        # ── Trend agent (shared, uses adapter connection) ────────────────
        self._trend_agents: dict[str, TrendAgent] = {}

        # ── Build tools + graph ───────────────────────────────────────────
        self._build_tools()
        self._build_graph()

    def _build_checkpointer(self):
        ckpt_cfg = self.cfg.get_checkpoint_config()
        self._ckpt_cfg = ckpt_cfg  # saved for auto-reconnect
        if ckpt_cfg.get("type") == "postgres":
            conn_string = ckpt_cfg["connection_string"]

            # Supabase's connection pooler (PgBouncer in transaction mode)
            # is incompatible with PostgresSaver — it blocks prepared
            # statements, DDL, and pipeline mode, all of which the library
            # requires.  Detect pooler URLs and skip straight to MemorySaver.
            if "pooler.supabase.com" in conn_string or ":6543/" in conn_string:
                logger.info(
                    "Checkpointer: MemorySaver (Supabase pooler detected — "
                    "PostgresSaver requires a direct connection on port 5432, "
                    "not the PgBouncer pooler on port 6543)"
                )
                return MemorySaver()

            # Non-pooler Postgres URL — try PostgresSaver normally.
            try:
                from langgraph.checkpoint.postgres import PostgresSaver
                saver_ctx = PostgresSaver.from_conn_string(conn_string)
                saver = saver_ctx.__enter__()
                saver.setup()
                logger.info("Checkpointer: PostgresSaver (direct Postgres)")
                return saver
            except ImportError:
                logger.warning(
                    "'langgraph-checkpoint-postgres' is not installed. "
                    "Falling back to MemorySaver."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize PostgresSaver ({e}). "
                    "Falling back to MemorySaver."
                )
        logger.info("Checkpointer: MemorySaver")
        return MemorySaver()

    @staticmethod
    def _is_conn_error(exc: Exception) -> bool:
        """Return True if the exception looks like a stale/dead Postgres connection."""
        msg = str(exc).lower()
        return any(phrase in msg for phrase in (
            "connection is closed",
            "connection already closed",
            "server closed the connection",
            "connection was reset",
            "broken pipe",
            "connection timed out",
            "ssl connection has been closed",
        ))

    def _reconnect_checkpointer(self):
        """Tear down the stale checkpointer, build a fresh one, and
        recompile the graph so every subsequent call uses the new
        connection. This is only needed for single-connection
        PostgresSaver — a pooled checkpointer heals itself."""
        logger.warning("Postgres connection lost — rebuilding checkpointer and graph")
        try:
            # Try to close the old connection cleanly (ignore errors)
            old = getattr(self, 'checkpointer', None)
            if old and hasattr(old, 'conn') and old.conn and not old.conn.closed:
                try:
                    old.conn.close()
                except Exception:
                    pass
        except Exception:
            pass
        self.checkpointer = self._build_checkpointer()
        self._build_graph()

    # ── Data adapter per hospital ────────────────────────────────────────
    def _get_adapter(self, hospital_id: str):
        if hospital_id not in self._adapters:
            data_cfg = self.cfg.get_data_source(hospital_id)
            self._adapters[hospital_id] = create_adapter(data_cfg)
        return self._adapters[hospital_id]

    def _get_trend_agent(self, hospital_id: str) -> Optional[TrendAgent]:
        if hospital_id not in self._trend_agents:
            adapter = self._get_adapter(hospital_id)
            if isinstance(adapter, DBAdapter):
                try:
                    raw_conn = sqlite3.connect(
                        adapter.connection_string.replace("sqlite:///", ""),
                        check_same_thread=False
                    ) if "sqlite" in adapter.connection_string else adapter.engine.raw_connection()
                    self._trend_agents[hospital_id] = TrendAgent(raw_conn)
                except Exception:
                    return None
            else:
                return None
        return self._trend_agents.get(hospital_id)

    # ── LangChain tools ──────────────────────────────────────────────────
    def _build_tools(self):
        engine = self

        @tool
        def sql_query(query: str, hospital_id: str = "default") -> str:
            """
            Execute a SQL SELECT query against the hospital's patient database.
            Tables vary by hospital — always reference the schema description
            provided in your system prompt for column and table names.
            Returns JSON string of results, or an error with a schema hint.
            """
            start = time.monotonic()
            try:
                adapter = engine._get_adapter(hospital_id)
                result = adapter.run_query(query)
                duration = int((time.monotonic() - start) * 1000)
                rows = json.loads(result).get("rows", [])
                engine.audit.log_tool_call(
                    "sql_query", "", duration, len(rows), True
                )
                return result
            except Exception as e:
                duration = int((time.monotonic() - start) * 1000)
                engine.audit.log_tool_call("sql_query", "", duration, 0, False, str(e))
                return json.dumps({"error": str(e)})

        @tool
        def vector_search(query: str, k: int = 3,
                          hospital_id: str = "global") -> str:
            """
            Search the medical knowledge base using semantic similarity.
            Returns top-k most relevant clinical guideline chunks with citations.
            """
            start = time.monotonic()
            try:
                results = engine.vector_store.search(query, k, hospital_id)
                duration = int((time.monotonic() - start) * 1000)
                engine.audit.log_tool_call(
                    "vector_search", "", duration, len(results), True
                )
                return json.dumps({"guidelines": results}, default=str)
            except Exception as e:
                duration = int((time.monotonic() - start) * 1000)
                engine.audit.log_tool_call(
                    "vector_search", "", duration, 0, False, str(e)
                )
                return json.dumps({"error": str(e)})

        self.sql_query_tool = sql_query
        self.vector_search_tool = vector_search

    # ── Agent helpers ────────────────────────────────────────────────────
    def _schema_for(self, hospital_id: str) -> str:
        try:
            return self._get_adapter(hospital_id).get_schema_description()
        except Exception:
            schema_path = Path("./mira_data/db_schema.txt")
            return schema_path.read_text() if schema_path.exists() else ""

    def _build_guideline_text(self, guidelines_json: str) -> str:
        try:
            text = ""
            for g in json.loads(guidelines_json).get("guidelines", []):
                text += f"\n[{g['source']}] {g['topic']}:\n{g['text']}\n"
            return text
        except Exception:
            return guidelines_json

    # ── Patient existence check (adapter-aware — works for any schema) ────
    def _check_patients_exist(self, patient_ids: list[int], hospital_id: str) -> tuple[list, list]:
        """Returns (found_ids, missing_ids) via the adapter's concept-mapped
        lookup — no hardcoded table/column names."""
        adapter = self._get_adapter(hospital_id)
        return adapter.check_patients_exist(patient_ids)

    # ── Sub-handler: one or more SPECIFIC patient IDs requested ────────────
    def _handle_specific_patient_query(self, state: MIRAState, intent: dict,
                                        hospital_id: str) -> MIRAState:
        """Adapter-aware: calls adapter.get_patient_labs() which builds the
        correct query for any DB schema, or fetches via FHIR — and always
        returns canonical {"rows": [...]} format."""
        patient_ids = intent["patient_ids"]
        found_ids, missing_ids = self._check_patients_exist(patient_ids, hospital_id)

        if not found_ids:
            empty_result = json.dumps({
                "rows": [],
                "requested_patient_ids": patient_ids,
                "found_patient_ids": [],
                "missing_patient_ids": missing_ids,
            })
            self.audit.log_agent_run("data_extractor", state.get("session_id", ""),
                                     0, True, rows_returned=0,
                                     user_id=state.get("user_id", ""), hospital_id=hospital_id)
            return {**state,
                    "sql_query_used": f"-- patient existence check for {patient_ids} --",
                    "sql_result": empty_result,
                    "sql_error": "", "sql_retry_count": state.get("sql_retry_count", 0),
                    "data_status": "patient_not_found",
                    "requested_patient_ids": patient_ids,
                    "found_patient_ids": [],
                    "missing_patient_ids": missing_ids}

        adapter = self._get_adapter(hospital_id)
        start = time.monotonic()
        result = adapter.get_patient_labs(found_ids, limit=200)
        parsed = json.loads(result)
        rows = parsed.get("rows", [])
        duration = int((time.monotonic() - start) * 1000)

        data_status = "ok" if rows else "no_data"
        enriched = json.dumps({
            "rows": rows,
            "requested_patient_ids": patient_ids,
            "found_patient_ids": found_ids,
            "missing_patient_ids": missing_ids,
        }, default=str)

        source_label = "fhir" if isinstance(adapter, FHIRAdapter) else "sql"
        self.audit.log_agent_run("data_extractor", state.get("session_id", ""),
                                 duration, True, rows_returned=len(rows),
                                 user_id=state.get("user_id", ""), hospital_id=hospital_id)
        self.audit.log_data_access(source_label, "lab_observations", len(rows),
                                   state.get("session_id", ""),
                                   state.get("user_id", ""), hospital_id)

        return {**state,
                "sql_query_used": f"-- adapter.get_patient_labs({found_ids}) [{source_label}] --",
                "sql_result": enriched,
                "sql_error": "", "sql_retry_count": state.get("sql_retry_count", 0),
                "data_status": data_status,
                "requested_patient_ids": patient_ids,
                "found_patient_ids": found_ids,
                "missing_patient_ids": missing_ids}

    # ── Sub-handler: general / list-style question ────────────────────────
    def _handle_general_query(self, state: MIRAState, intent: dict,
                               hospital_id: str, schema: str) -> MIRAState:
        """Adapter-aware general-query handler.
        • FHIR path  → calls adapter.get_broad_abnormal_labs() directly
                        (no SQL generation, no LLM step for data extraction).
        • SQL path   → uses LLM to generate schema-agnostic SQL via the
                        adapter's concept-mapped prompt instructions, then
                        falls back to adapter.get_broad_abnormal_labs().
        Either path lands in the same canonical {"rows": [...]} shape."""
        adapter = self._get_adapter(hospital_id)
        question = intent["raw_question"]
        retry_count = state.get("sql_retry_count", 0)
        conditions = intent.get("conditions", [])
        start = time.monotonic()
        is_fhir = isinstance(adapter, FHIRAdapter)

        # ── FHIR path (no SQL at all) ──────────────────────────────────
        if is_fhir:
            result = adapter.get_broad_abnormal_labs(limit=60)
            parsed = json.loads(result)
            rows = parsed.get("rows", [])
            duration = int((time.monotonic() - start) * 1000)
            distinct_patients = {r.get("subject_id") for r in rows
                                 if r.get("subject_id") is not None}
            data_status = "ok" if rows else "no_data"
            if intent["is_list"] and len(distinct_patients) < MIN_LIST_PATIENTS:
                data_status = "list_insufficient"

            enriched = json.dumps({
                "rows": rows, "used_fallback_broad_query": False,
            }, default=str)

            self.audit.log_agent_run("data_extractor", state.get("session_id", ""),
                                     duration, True, rows_returned=len(rows),
                                     user_id=state.get("user_id", ""),
                                     hospital_id=hospital_id)
            self.audit.log_data_access("fhir", "lab_observations", len(rows),
                                       state.get("session_id", ""),
                                       state.get("user_id", ""), hospital_id)

            return {**state,
                    "sql_query_used": "-- adapter.get_broad_abnormal_labs() [fhir] --",
                    "sql_result": enriched,
                    "sql_error": "", "sql_retry_count": retry_count,
                    "data_status": data_status,
                    "requested_patient_ids": [],
                    "found_patient_ids": list(distinct_patients),
                    "missing_patient_ids": []}

        # ── SQL path (LLM-generated SQL with concept-mapped prompt) ────
        condition_hints = _condition_sql_hints(conditions)

        # Dynamic SQL instructions built from the adapter's concept map
        # so the LLM uses the *real* table/column names for this hospital.
        sql_instructions = adapter.get_sql_prompt_instructions()

        broaden_note = ""
        if retry_count > 0:
            broaden_note = (
                f"\nRETRY {retry_count}: your previous query returned too few distinct "
                "patients for this plural/list question. Remove restrictive filters, "
                "widen the abnormal threshold, and increase the patient pool so that "
                "at least 10 different subject_ids can appear."
            )

        system_prompt = f"""You are a medical SQL expert. Write a single valid PostgreSQL SELECT query.
Always be flexible and common-sense in understanding the question, including vague or
colloquially-phrased clinical questions.

DATABASE SCHEMA:
{schema}

VOCABULARY:
- "critical"/"severe" -> valuenum > ref_range_upper * 1.5 OR valuenum < ref_range_lower * 0.5 OR flag IS NOT NULL
- "abnormal" -> valuenum > ref_range_upper OR valuenum < ref_range_lower OR flag IS NOT NULL
- "high"/"elevated" -> valuenum > ref_range_upper
- "low" -> valuenum < ref_range_lower
{sql_instructions}
{condition_hints}
- CRITICAL: When asked for a LIST of patients ("which patients...", "patients with...",
  "how many patients..."), your query MUST return multiple subject_ids. Use IN, GROUP BY,
  or subqueries to ensure data from at least 10 different patients is returned whenever
  the underlying data supports it. Never write a query that can return only one subject_id
  for a plural question.
- LIMIT 60 minimum when returning lists.
- IMPORTANT: Always alias the output columns to the canonical names shown in the SELECT
  instruction above (subject_id, gender, age, lab_name, valuenum, valueuom,
  ref_range_lower, ref_range_upper, flag, charttime). Downstream agents rely on
  these exact names.
- Return ONLY raw SQL, no markdown, no explanation.{broaden_note}"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Clinical question: {question}")
        ])
        raw_sql = response.content.strip().replace("```sql", "").replace("```", "").strip()
        result = self.sql_query_tool.invoke({"query": raw_sql, "hospital_id": hospital_id})
        parsed = json.loads(result)
        rows = parsed.get("rows", [])
        used_fallback = False

        if "error" in parsed or len(rows) == 0:
            logger.info("Generated SQL returned 0 rows or error; using adapter broad fallback.")
            result = adapter.get_broad_abnormal_labs(limit=60)
            parsed = json.loads(result)
            rows = parsed.get("rows", [])
            raw_sql = "-- adapter.get_broad_abnormal_labs() [sql fallback] --"
            used_fallback = True

        duration = int((time.monotonic() - start) * 1000)
        distinct_patients = {r.get("subject_id") for r in rows if r.get("subject_id") is not None}

        data_status = "broadened_query" if used_fallback else "ok"
        if intent["is_list"] and len(distinct_patients) < MIN_LIST_PATIENTS:
            data_status = "list_insufficient"
        if len(rows) == 0:
            data_status = "no_data"

        enriched = json.dumps({
            "rows": rows,
            "used_fallback_broad_query": used_fallback,
        }, default=str)

        self.audit.log_agent_run("data_extractor", state.get("session_id", ""),
                                 duration, True, rows_returned=len(rows),
                                 user_id=state.get("user_id", ""),
                                 hospital_id=hospital_id)
        self.audit.log_data_access("sql", "lab_observations", len(rows),
                                   state.get("session_id", ""),
                                   state.get("user_id", ""), hospital_id)

        return {**state, "sql_query_used": raw_sql, "sql_result": enriched,
                "sql_error": "", "sql_retry_count": retry_count,
                "data_status": data_status,
                "requested_patient_ids": [], "found_patient_ids": list(distinct_patients),
                "missing_patient_ids": []}

    # ── Agent 1 — SQL Data Extractor (dispatcher) ────────────────────────
    def agent1_sql_extractor(self, state: MIRAState) -> MIRAState:
        hospital_id = state.get("hospital_id", "default")
        question = state.get("clinical_question", "") or state.get("question", "") or ""

        # Preserve intent across retries so we don't re-parse every loop.
        intent = state.get("query_intent") or parse_clinical_intent(question)
        state = {**state, "query_intent": intent}

        if intent["patient_ids"] and state.get("sql_retry_count", 0) == 0:
            return self._handle_specific_patient_query(state, intent, hospital_id)

        schema = self._schema_for(hospital_id)
        return self._handle_general_query(state, intent, hospital_id, schema)

    def should_retry_sql(self, state: MIRAState) -> str:
        # FIX: this used to be a stub that always returned "ok", so the
        # retry edge in the graph was dead code. Now it actually broadens
        # list-style queries that came back with too few distinct patients.
        intent = state.get("query_intent", {})
        retry_count = state.get("sql_retry_count", 0)
        if not intent.get("is_list"):
            return "ok"
        if state.get("data_status") != "list_insufficient":
            return "ok"
        if retry_count >= MAX_SQL_RETRIES:
            return "ok"
        return "retry"

    # (called by the "retry" edge before looping back into sql_extractor)
    def _increment_retry(self, state: MIRAState) -> MIRAState:
        return {**state, "sql_retry_count": state.get("sql_retry_count", 0) + 1}

    # ── Agent 1.5 — Trend Check ──────────────────────────────────────────
    def agent_trend_check(self, state: MIRAState) -> MIRAState:
        sql_result = state.get("sql_result", "")
        if not sql_result:
            return {**state, "trend_data": ""}

        hospital_id = state.get("hospital_id", "default")
        trend_agent = self._get_trend_agent(hospital_id)
        if not trend_agent:
            return {**state, "trend_data": ""}

        try:
            rows = json.loads(sql_result).get("rows", [])
            subject_id, lab_name = None, None
            for row in rows:
                if "subject_id" in row and row["subject_id"] is not None:
                    subject_id = row["subject_id"]
                for key in ("label", "lab_name"):
                    if key in row and row[key]:
                        lab_name = row[key]
                if subject_id and lab_name:
                    break

            if not subject_id or not lab_name:
                return {**state, "trend_data": ""}

            trend_result = trend_agent.analyze_patient_lab(int(subject_id), str(lab_name))
            if trend_result.get("trend") == "insufficient_data":
                return {**state, "trend_data": ""}
            return {**state, "trend_data": json.dumps(trend_result, default=str)}
        except Exception:
            return {**state, "trend_data": ""}

    # ── Agent 2 — Semantic Cross-Ref ─────────────────────────────────────
    def agent2_semantic_crossref(self, state: MIRAState) -> MIRAState:
        sql_context = state.get("sql_result", "") or "No patient data retrieved."
        hospital_id = state.get("hospital_id", "global")
        question = state.get("clinical_question", "") or state.get("question", "") or ""
        intent = state.get("query_intent", {})
        condition_terms = _condition_search_terms(intent.get("conditions", []))

        response = self.llm.invoke([
            SystemMessage(content=(
                "Extract the SPECIFIC lab test names and their abnormal direction "
                "(high/low) from the patient data. Write a semantic search query using "
                "those EXACT lab names. Return ONLY the search query."
            )),
            HumanMessage(content=f"Question: {question}\n"
                                  f"Patient data: {sql_context[:1000]}")
        ])
        search_query = response.content.strip()
        if condition_terms:
            search_query = f"{search_query} {condition_terms}".strip()

        guidelines_result = self.vector_search_tool.invoke({
            "query": search_query, "k": 3, "hospital_id": hospital_id
        })

        try:
            parsed = json.loads(guidelines_result)
            top_score = parsed.get("guidelines", [{}])[0].get("relevance_score", 0)
            if top_score < 0.3:
                parsed["low_relevance_warning"] = (
                    "Retrieved guidelines may not match this specific finding — "
                    "treat as general context only."
                )
                guidelines_result = json.dumps(parsed, default=str)
        except Exception:
            pass

        return {**state, "search_query_used": search_query,
                "guidelines": guidelines_result}

    # ── Agent 3 — Clinical Reasoning ─────────────────────────────────────
    def agent3_clinical_reasoning(self, state: MIRAState) -> MIRAState:
        sql_result = state.get("sql_result", "No patient data.")
        guideline_text = self._build_guideline_text(state.get("guidelines", ""))
        intent = state.get("query_intent", {})
        data_status = state.get("data_status", "")

        try:
            parsed_sql = json.loads(sql_result)
            rows = parsed_sql.get("rows", [])
        except Exception:
            parsed_sql, rows = {}, []

        unique_patients = {r["subject_id"] for r in rows if r.get("subject_id") is not None}

        # ---- Build an honest, specific status note for the model ----
        status_note = ""
        if data_status == "patient_not_found":
            missing = state.get("missing_patient_ids") or parsed_sql.get("missing_patient_ids", [])
            status_note = (
                f"\n\nDATA STATUS: The requested patient ID(s) {missing} were NOT found in the "
                "system. Do not invent or substitute a different patient. State plainly that "
                "no matching patient exists and suggest the user double-check the ID."
            )
        elif data_status == "no_data":
            status_note = (
                "\n\nDATA STATUS: The patient(s)/criteria in question exist but no matching lab "
                "data was found. State plainly that no relevant lab results are on record — do "
                "not fabricate findings."
            )
        elif data_status == "broadened_query":
            status_note = (
                "\n\nDATA STATUS: No results matched the specific criteria in the question, so "
                "the system broadened the search to general abnormal findings across patients. "
                "You MUST clearly disclose this to the reader at the top of the report (e.g. "
                "'No exact match for your query; showing general abnormal findings instead') — "
                "never present broadened results as if they were an exact match."
            )
        elif data_status == "list_insufficient" and intent.get("is_list"):
            status_note = (
                f"\n\nDATA STATUS: This is a list-style question, but only {len(unique_patients)} "
                "patient(s) matched after broadening the search as far as reasonably possible. "
                "State the actual count plainly — do not imply a larger cohort than what was found."
            )

        if intent.get("is_list") and len(unique_patients) == 1 and data_status not in (
            "patient_not_found", "no_data"
        ):
            status_note += (
                "\n\nNOTE: The question asked about multiple patients, but only ONE patient "
                "actually matched the criteria. Say this explicitly, e.g. 'Only one patient "
                "matched this criteria: Patient [ID].' Do not treat it as a data error."
            )
        elif len(unique_patients) == 1:
            status_note += (
                "\n\nNOTE: Only ONE patient matched the criteria. "
                "The analysis below refers to a single patient case."
            )
        elif len(unique_patients) == 0 and data_status not in ("patient_not_found",):
            status_note += "\n\nWARNING: No patients found matching the query criteria."

        trend_context = ""
        try:
            td = state.get("trend_data", "")
            if td:
                trend_parsed = json.loads(td)
                trend_context = f"\n\nLAB TRAJECTORY:\n{trend_parsed.get('summary', '')}"
        except Exception:
            pass

        relevance_warning = ""
        try:
            if json.loads(state.get("guidelines", "{}")).get("low_relevance_warning"):
                relevance_warning = (
                    "\n\nNOTE: Retrieved guidelines may not directly match these findings. "
                    "State plainly if no matching guideline exists — do not force-fit."
                )
        except Exception:
            pass

        feedback_context = ""
        if state.get("human_decision") == "reject" and state.get("human_feedback"):
            feedback_context = (
                f"\n\nCLINICIAN FEEDBACK (you MUST revise the report to address this, and the "
                f"revision must be substantively different from the previous draft):\n"
                f"{state['human_feedback']}"
            )

        system_prompt = f"""You are an expert clinical AI assistant. Write a professional, well-structured clinical report.

Format your response EXACTLY as:
## Patient Summary
[Clear, concise patient demographics and chief findings]

## Identified Concerns
[Bullet points of abnormal findings with values]

## Clinical Guideline Context
[Relevant guidelines or clinical pearls]

## Recommended Actions
[Specific, actionable recommendations]

IMPORTANT GUIDELINES:
- Use proper medical terminology, correct grammar, and a professional tone throughout.
- Ground every number, lab value, and patient ID in the provided data — never hallucinate.
- Interpret the clinician's question with common sense even if phrased informally or
  incompletely; don't demand the user restate things in a rigid format.
- For list queries: if multiple patients, format as "1. Patient ID [ID], [demographics], [key finding]"
  etc., one per line.
- If only ONE patient matched a plural question, say so explicitly rather than silently
  presenting a single case as if it were what was asked for.
- If NO data or the requested patient was not found: clearly state that fact as the FIRST
  line of the Patient Summary, in plain language, with no other sections speculating beyond it.
- Use markdown formatting for readability.
- Be concise but complete.{status_note}{trend_context}{relevance_warning}{feedback_context}"""

        question = state.get("clinical_question", "") or state.get("question", "") or ""

        start = time.monotonic()
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {question}\n\n"
                                  f"PATIENT DATA:\n{sql_result[:2000]}\n\n"
                                  f"GUIDELINES:\n{guideline_text[:2000]}")
        ])
        reasoning = response.content.strip()
        duration = int((time.monotonic() - start) * 1000)

        self.audit.log_agent_run("clinical_reasoning", state.get("session_id", ""),
                                 duration, True, user_id=state.get("user_id", ""),
                                 hospital_id=state.get("hospital_id", ""))

        return {**state, "clinical_reasoning": reasoning,
                "human_decision": "", "human_feedback": ""}

    def stream_clinical_reasoning(self, state: MIRAState) -> Generator[str, None, None]:
        """Streaming version for Streamlit st.write_stream."""
        sql_result = state.get("sql_result", "No patient data.")
        guideline_text = self._build_guideline_text(state.get("guidelines", ""))

        trend_context = ""
        try:
            td = state.get("trend_data", "")
            if td:
                trend_parsed = json.loads(td)
                trend_context = f"\n\nLAB TRAJECTORY:\n{trend_parsed.get('summary', '')}"
        except Exception:
            pass

        feedback_context = ""
        if state.get("human_decision") == "reject" and state.get("human_feedback"):
            feedback_context = f"\n\nCLINICIAN FEEDBACK:\n{state['human_feedback']}"

        system_prompt = f"""You are an expert clinical AI assistant. Synthesize data into:
## Patient Summary
## Identified Concerns
## Clinical Guideline Context
## Recommended Actions
Ground every claim in the data or a cited guideline. Never hallucinate values.{trend_context}{feedback_context}"""

        question = state.get("clinical_question", "") or state.get("question", "") or ""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {question}\n\n"
                                  f"PATIENT DATA:\n{sql_result[:2000]}\n\n"
                                  f"GUIDELINES:\n{guideline_text[:2000]}")
        ]
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

    # ── Human review (pause point) ───────────────────────────────────────
    def human_review_node(self, state: MIRAState) -> MIRAState:
        return state

    def route_after_human_review(self, state: MIRAState) -> str:
        if state.get("human_decision") == "reject":
            return "revise"
        return "proceed"

    # ── Agent 4 — Critic & Safety ────────────────────────────────────────
    def agent4_critic_safety(self, state: MIRAState) -> MIRAState:
        reasoning  = state.get("clinical_reasoning", "")
        sql_result = state.get("sql_result", "")
        guidelines = state.get("guidelines", "")

        system_prompt = """You are a medical AI safety critic AND formatter.
Your job: validate the draft report and ensure it's properly formatted as a professional clinical document.

Validation checks:
1. ALL values/numbers must appear in the patient data (no hallucinations)
2. Recommendations must be clinically sound (not contradicting guidelines)
3. Must have all four sections: Patient Summary, Identified Concerns, Clinical Guideline Context, Recommended Actions

Be PERMISSIVE: Don't flag missing tests, incomplete analysis for unrequested items, or minor guideline gaps.

Formatting fixes:
- Ensure proper markdown structure with ## headers
- Use bullet points for lists
- Make sure patient IDs and values are clearly stated
- Ensure professional, grammatically correct tone
- If report mentions single patient, keep that note

CRITICAL: final_report must be the complete, formatted clinical report.
NEVER put critique text or meta-comments in final_report.
Respond ONLY in JSON:
{"approved": true/false, "safety_flags": [], "corrections": "...", "final_report": "..."}"""

        start = time.monotonic()
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"PATIENT DATA (ground truth):\n{sql_result[:1500]}\n\n"
                                  f"GUIDELINES:\n{guidelines[:1000]}\n\n"
                                  f"DRAFT:\n{reasoning}")
        ])
        duration = int((time.monotonic() - start) * 1000)

        try:
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            critic_output = json.loads(raw)
        except Exception:
            critic_output = {"approved": True, "safety_flags": [], "final_report": reasoning}

        final_report = critic_output.get("final_report", reasoning)

        # Ensure final_report is not empty and is properly formatted
        if not final_report or len(final_report.strip()) < 50:
            final_report = reasoning

        # Remove critique artifacts if they snuck in
        critique_markers = ["the analysis contains", "needs to be revised",
                            "the analysis does not", "hallucinated value",
                            "issues that need addressing", "critique:", "issues:"]
        if any(m in final_report.lower()[:500] for m in critique_markers):
            final_report = reasoning

        approved = critic_output.get("approved", True)
        safety_flags = list(critic_output.get("safety_flags", []))

        # NEW — hallucination / mismatch guard: flag (don't silently ship)
        # any patient ID mentioned in the report that never appeared in
        # the actual SQL results.
        try:
            data_ids = {str(pid) for pid in _distinct_patient_ids(sql_result)}
            mentioned_ids = set(re.findall(r"[Pp]atient(?:\s*ID)?\s*[:#]?\s*(\d{2,})", final_report))
            unexplained = mentioned_ids - data_ids
            if unexplained and data_ids:
                safety_flags.append(f"id_mismatch:{','.join(sorted(unexplained))}")
                approved = False
        except Exception:
            pass

        self.audit.log_agent_run("critic_safety", state.get("session_id", ""),
                                 duration, True, user_id=state.get("user_id", ""),
                                 hospital_id=state.get("hospital_id", ""))
        self.audit.log_report_finalized(
            state.get("user_id", ""), state.get("session_id", ""),
            approved, safety_flags, state.get("hospital_id", "")
        )

        return {**state, "final_report": final_report,
                "safety_flags": safety_flags, "approved": approved}

    # ── Build graph ───────────────────────────────────────────────────────
    def _build_graph(self):
        builder = StateGraph(MIRAState)

        builder.add_node("sql_extractor",      self.agent1_sql_extractor)
        builder.add_node("retry_bump",         self._increment_retry)
        builder.add_node("trend_check",        self.agent_trend_check)
        builder.add_node("semantic_crossref",  self.agent2_semantic_crossref)
        builder.add_node("clinical_reasoning", self.agent3_clinical_reasoning)
        builder.add_node("human_review",       self.human_review_node)
        builder.add_node("critic_safety",      self.agent4_critic_safety)

        builder.set_entry_point("sql_extractor")

        builder.add_conditional_edges(
            "sql_extractor", self.should_retry_sql,
            {"retry": "retry_bump", "ok": "trend_check"}
        )
        builder.add_edge("retry_bump", "sql_extractor")
        builder.add_edge("trend_check",        "semantic_crossref")
        builder.add_edge("semantic_crossref",  "clinical_reasoning")
        builder.add_edge("clinical_reasoning", "human_review")
        builder.add_conditional_edges(
            "human_review", self.route_after_human_review,
            {"proceed": "critic_safety", "revise": "clinical_reasoning"}
        )
        builder.add_edge("critic_safety", END)

        self.graph = builder.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["human_review"]
        )

    # ── Public API (called by streamlit_app_prod.py) ─────────────────────
    def new_thread(self) -> dict:
        return {"configurable": {"thread_id": str(uuid.uuid4())}}

    def transcribe_voice_query(self, audio_bytes: bytes, filename: str,
                               spoken_language: str = "english",
                               user_id: str = "", hospital_id: str = "") -> dict:
        """
        Transcribes a recorded clinical question, with support for auto-detection
        and explicit English, Hindi, and Gujarati input.

        When spoken_language="auto", Whisper auto-detects the language.
        For non-English audio we use Whisper's /translations endpoint,
        which always outputs English text regardless of the spoken
        language — this is deliberate: everything downstream (SQL
        generation, the condition vocabulary, patient-ID extraction) only
        understands English medical terms, so translation happens once,
        here, rather than requiring every agent to be multilingual-aware.

        Returns:
            {
              "clinical_question": <English text to feed the pipeline>,
              "original_transcript": <verbatim transcript in the spoken
                                       language, kept for audit trail>,
              "spoken_language": <detected or specified language>,
              "detected_language": <language name if auto-detected>,
              "error": <str, only present on failure>,
            }

        The original_transcript is always preserved, even when translated,
        so a clinician or auditor can later see exactly what was said —
        never just the machine-translated version.
        """
        spoken_language = (spoken_language or "english").lower()
        is_auto_detect = spoken_language == "auto"
        start = time.monotonic()

        # Determine MIME type from filename for the Whisper API
        _ext = (filename or "query.webm").rsplit(".", 1)[-1].lower()
        _mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4",
                     "ogg": "audio/ogg", "webm": "audio/webm", "flac": "audio/flac"}
        _mime = _mime_map.get(_ext, "audio/webm")
        _fname = filename or "query.webm"

        def _make_audio_file():
            """Return a fresh file-tuple for each Whisper API call.
            Each call consumes the BytesIO stream, so we must create a new one."""
            return (_fname, io.BytesIO(audio_bytes), _mime)

        try:

            if is_auto_detect:
                # Auto-detect: transcribe without language constraint,
                # then auto-translate to English if needed.
                # FIX: the default response_format ("json") does NOT include
                # a `language` field at all, so the old code's
                # `getattr(verbatim_resp, "language", None)` always fell back
                # to "en" — meaning detected_language was reported as English
                # for every recording, and the `!= "en"` check below never
                # triggered translation. Hindi/Gujarati audio was silently
                # sent downstream untranslated, breaking the whole feature.
                # response_format="verbose_json" is required to get `language`
                # back at all. Note Whisper returns the FULL language name in
                # lowercase English (e.g. "english", "hindi", "gujarati"), not
                # an ISO code — so we compare against that, not "en".
                verbatim_resp = self.openai_client.audio.transcriptions.create(
                    model="whisper-1", file=_make_audio_file(), response_format="verbose_json"
                )
                detected_lang = (getattr(verbatim_resp, "language", None) or "english").lower()
                original_text = verbatim_resp.text.strip()

                # If Whisper detected non-English, translate to English
                if detected_lang not in ("english", "en"):
                    translated_resp = self.openai_client.audio.translations.create(
                        model="whisper-1", file=_make_audio_file()
                    )
                    english_text = translated_resp.text.strip()
                else:
                    english_text = original_text

                result = {
                    "clinical_question": english_text,
                    "original_transcript": original_text,
                    "spoken_language": "auto",
                    "detected_language": detected_lang,
                }
            elif spoken_language == "english":
                resp = self.openai_client.audio.transcriptions.create(
                    model="whisper-1", file=_make_audio_file(), language="en"
                )
                text = resp.text.strip()
                result = {"clinical_question": text, "original_transcript": text,
                         "spoken_language": spoken_language}
            else:
                # Explicit non-English language: get verbatim + translated
                lang_code = VOICE_LANGUAGE_CODES.get(spoken_language, "en")
                # Get the verbatim transcript (for the audit trail) AND the
                # English translation (for the pipeline) — two calls, but
                # each is cheap, and correctness/auditability here matters
                # more than saving one Whisper call.
                verbatim_resp = self.openai_client.audio.transcriptions.create(
                    model="whisper-1", file=_make_audio_file(), language=lang_code
                )
                translated_resp = self.openai_client.audio.translations.create(
                    model="whisper-1", file=_make_audio_file()
                )
                result = {
                    "clinical_question": translated_resp.text.strip(),
                    "original_transcript": verbatim_resp.text.strip(),
                    "spoken_language": spoken_language,
                }

            duration = int((time.monotonic() - start) * 1000)
            self.audit.log_tool_call("voice_transcription", "", duration, 1, True)
            return result

        except Exception as e:
            duration = int((time.monotonic() - start) * 1000)
            self.audit.log_tool_call("voice_transcription", "", duration, 0, False, str(e))
            return {"clinical_question": "", "original_transcript": "",
                    "spoken_language": spoken_language, "error": str(e)}

    def run_until_review(self, clinical_question: str, config: dict,
                         user_id: str = "anon", hospital_id: str = "demo",
                         session_id: str = "",
                         input_mode: str = "text",
                         spoken_language: str = "",
                         original_transcript: str = "") -> MIRAState:
        initial = make_initial_state(clinical_question, user_id, hospital_id, session_id,
                                     input_mode=input_mode, spoken_language=spoken_language,
                                     original_transcript=original_transcript)
        self.audit.log_query(user_id, hospital_id, session_id,
                             config["configurable"]["thread_id"],
                             clinical_question, len(clinical_question))
        try:
            return self.graph.invoke(initial, config)
        except Exception as exc:
            if self._is_conn_error(exc):
                self._reconnect_checkpointer()
                return self.graph.invoke(initial, config)
            raise

    def submit_human_decision(self, config: dict, decision: str,
                               feedback: str = "",
                               user_id: str = "",
                               hospital_id: str = "",
                               clinical_question: str = "",
                               paused_state: dict = None) -> MIRAState:
        thread_id = config["configurable"]["thread_id"]

        try:
            snapshot = self.graph.get_state(config)
            current_state = dict(snapshot.values or {})
        except Exception as exc:
            if self._is_conn_error(exc):
                self._reconnect_checkpointer()
                try:
                    snapshot = self.graph.get_state(config)
                    current_state = dict(snapshot.values or {})
                except Exception:
                    current_state = {}
            else:
                current_state = {}

        # MemorySaver lost state (e.g. a process restart). FIX: only trust
        # a frontend-supplied paused_state if it actually matches the
        # question being submitted — previously this reused whatever the
        # UI happened to have cached, even from an unrelated prior query,
        # which produced the "same random patient every time" bug.
        if not current_state.get("clinical_question"):
            same_question = (
                paused_state
                and clinical_question
                and paused_state.get("clinical_question") == clinical_question
            )
            if same_question and paused_state.get("clinical_reasoning"):
                current_state = dict(paused_state)
                # Persist recovered state into a fresh checkpoint so future
                # reads are consistent.
                try:
                    self.graph.update_state(config, current_state)
                except Exception:
                    pass
            else:
                return {
                    **(paused_state or make_initial_state(clinical_question, user_id, hospital_id)),
                    "human_decision": decision,
                    "human_feedback": feedback,
                    "final_report": (
                        "Session state could not be recovered for this question. "
                        "Please start a new audit and resubmit the query rather than "
                        "approving — no report has been generated to finalize."
                    ),
                    "approved": False,
                    "safety_flags": ["session_state_lost"],
                }

        # Prevent finalizing reports with no/insufficient data
        sql_result = current_state.get("sql_result", "")
        try:
            sql_data = json.loads(sql_result) if sql_result else {}
            has_data = len(sql_data.get("rows", [])) > 0 and "error" not in sql_data
        except Exception:
            has_data = False

        if not has_data and decision == "approve":
            no_data_state = {
                **current_state,
                "approved": False,
                "final_report": (
                    "No valid patient data was retrieved for this question, so there is "
                    "nothing to finalize. Please start a new audit with a different or "
                    "corrected query (for example, double-check the patient ID or "
                    "broaden the criteria)."
                ),
                "safety_flags": list(set(current_state.get("safety_flags", []) + ["no_data"])),
            }
            try:
                self.graph.update_state(config, no_data_state)
            except Exception:
                pass
            return no_data_state

        # If rejecting, regenerate the report from feedback and PERSIST it.
        # FIX: previously the revised report was computed and returned but
        # never written back into the LangGraph checkpoint, so any later
        # read of state (e.g. a page rerun) still showed the stale report.
        if decision == "reject" and feedback:
            revised_state = {**current_state,
                             "human_decision": "reject",
                             "human_feedback": feedback}
            revised_state = self.agent3_clinical_reasoning(revised_state)
            revised_state["human_decision"] = "reject"
            revised_state["human_feedback"] = feedback
            revised_state = self.agent4_critic_safety(revised_state)
            revised_state["human_decision"] = ""
            revised_state["human_feedback"] = ""

            try:
                self.graph.update_state(config, revised_state)
            except Exception as exc:
                if self._is_conn_error(exc):
                    self._reconnect_checkpointer()
                    self.graph.update_state(config, revised_state)
                else:
                    raise
            self.audit.log_human_review(user_id, thread_id, decision, True, hospital_id)

            # Return exactly what's now persisted, so caller and checkpoint
            # can never disagree.
            try:
                return self.graph.get_state(config).values
            except Exception:
                return revised_state

        try:
            self.graph.update_state(config, {
                "human_decision": decision,
                "human_feedback": feedback,
            })
        except Exception as exc:
            if self._is_conn_error(exc):
                self._reconnect_checkpointer()
                self.graph.update_state(config, {
                    "human_decision": decision,
                    "human_feedback": feedback,
                })
            else:
                raise
        self.audit.log_human_review(user_id, thread_id, decision,
                                    bool(feedback), hospital_id)
        try:
            return self.graph.invoke(None, config)
        except Exception as exc:
            if self._is_conn_error(exc):
                self._reconnect_checkpointer()
                return self.graph.invoke(None, config)
            raise

    def get_current_state(self, config: dict) -> MIRAState:
        try:
            return self.graph.get_state(config).values
        except Exception as exc:
            if self._is_conn_error(exc):
                self._reconnect_checkpointer()
                return self.graph.get_state(config).values
            raise

    # ══════════════════════════════════════════════════════════════════
    # NEW PUBLIC API
    # ══════════════════════════════════════════════════════════════════

    def start_new_audit(self) -> dict:
        """
        Returns a brand-new thread config, guaranteed not to reuse any
        cached state from a previous question.
        """
        return self.new_thread()

    def run_triage(self, hospital_id: str, limit: int = 50) -> list[dict]:
        """
        Polls the adapter for abnormal labs across the hospital,
        uses an LLM to evaluate severity and group by patient,
        and returns a ranked list of the most critical patients.
        """
        adapter = self._get_adapter(hospital_id)
        # 1. Fetch raw abnormal labs
        raw_json = adapter.get_broad_abnormal_labs(limit=limit)
        parsed = json.loads(raw_json)
        rows = parsed.get("rows", [])
        if not rows:
            return []
            
        # 2. Use LLM to triage
        prompt = f"""You are a clinical triage AI.
Review these recent abnormal labs across the hospital:
{json.dumps(rows)}

Identify up to the 5 most critical patients based on these labs. For each, provide:
1. "subject_id": The patient ID
2. "severity_score": 1-10 (10 being most critical)
3. "reason": A 1-sentence explanation of why they are critical.
4. "labs": A summary of their key abnormal labs.

Return ONLY a valid JSON array of objects with these exact keys. No markdown."""
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            content = response.content.replace("```json", "").replace("```", "").strip()
            triage_data = json.loads(content)
            # Sort by severity descending
            triage_data.sort(key=lambda x: x.get("severity_score", 0), reverse=True)
            return triage_data
        except Exception as e:
            logger.error(f"Triage failed: {e}")
            return []


# ══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ══════════════════════════════════════════════════════════════════════════

_engine: Optional[MIRAEngineProd] = None


def get_engine(cfg: Optional[ConfigManager] = None) -> MIRAEngineProd:
    global _engine
    if _engine is None:
        _engine = MIRAEngineProd(cfg or ConfigManager())
    return _engine


if __name__ == "__main__":
    print("🏥 MIRA Production Pipeline — smoke test\n")
    engine = get_engine()
    cfg_obj = ConfigManager()
    print(cfg_obj.describe())

    cfg = engine.new_thread()
    question = "which patients show signs of AKI"
    print(f"\nQuery: {question}")
    paused = engine.run_until_review(question, cfg, user_id="dev", hospital_id="demo")
    print(f"\n🛑 Paused. Data status: {paused.get('data_status')}")
    print(f"Preview:\n{paused['clinical_reasoning'][:400]}...")

    print("\n--- Testing revision path ---")
    revised = engine.submit_human_decision(
        cfg, "reject", feedback="Please double check for creatinine trend and be more specific.",
        user_id="dev", hospital_id="demo", clinical_question=question
    )
    print(f"Revised report differs from original: {revised['final_report'] != paused['clinical_reasoning']}")

    final = engine.submit_human_decision(cfg, "approve", user_id="dev", hospital_id="demo",
                                          clinical_question=question)
    print(f"\n✅ Final report ({len(final['final_report'])} chars)")
    print(f"Approved: {final['approved']}")
    print(f"Safety flags: {final['safety_flags'] or 'None'}")