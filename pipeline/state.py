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