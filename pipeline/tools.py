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

from core.config import ConfigManager
from adapters.db import DBAdapter, create_adapter
from adapters.fhir import FHIRAdapter
from core.audit import AuditLogger
from pipeline.agents.trend import TrendAgent


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
    "nausea": {
        "aliases": ["nausea", "nauseous", "feeling sick"],
        "sql_hint": "For nausea: typically a clinical symptom, check for any associated electrolyte imbalances like sodium or potassium or signs of dehydration.",
        "search_terms": "nausea differential diagnosis clinical evaluation",
    },
    "vomiting": {
        "aliases": ["vomiting", "emesis", "throwing up", "puking"],
        "sql_hint": "For vomiting: check for hypokalemia (d.label ILIKE '%potassium%') and metabolic alkalosis (d.label ILIKE '%bicarbonate%').",
        "search_terms": "vomiting emesis hypokalemia dehydration management",
    },
    "headache": {
        "aliases": ["headache", "migraine", "head pain"],
        "sql_hint": "For headache: check for any abnormal vital signs, particularly elevated blood pressure.",
        "search_terms": "headache migraine tension secondary causes evaluation",
    },
    "dizziness": {
        "aliases": ["dizziness", "dizzy", "lightheaded", "vertigo"],
        "sql_hint": "For dizziness: check for anemia (low hemoglobin/hematocrit) or hypoglycemia (low glucose).",
        "search_terms": "dizziness vertigo lightheadedness orthostatic evaluation",
    },
    "red eyes": {
        "aliases": ["red eyes", "red eye", "bloodshot eyes", "conjunctivitis"],
        "sql_hint": "For red eyes: mostly a clinical finding, check for systemic signs of inflammation (WBC, CRP).",
        "search_terms": "red eye conjunctivitis differential diagnosis management",
    },
    "fever": {
        "aliases": ["fever", "feverish", "high temperature", "pyrexia"],
        "sql_hint": "For fever: check for signs of infection such as elevated WBC (d.label ILIKE '%white blood cell%') or lactate.",
        "search_terms": "fever pyrexia infection evaluation causes",
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