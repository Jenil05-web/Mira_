import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
import math
import os
import pickle
import re
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

VOICE_LANGUAGE_CODES = {"english": "en", "hindi": "hi", "gujarati": "gu"}

# ── Retry / backoff constants ─────────────────────────────────────────────
_LLM_MAX_ATTEMPTS  = 3        # total attempts (1 original + 2 retries)
_LLM_BASE_DELAY    = 1.0      # seconds before first retry
_LLM_MAX_DELAY     = 30.0     # cap on exponential back-off
_DB_QUERY_TIMEOUT  = 20       # seconds before a DB query is abandoned
_CB_FAILURE_THRESH = 3        # consecutive failures before circuit opens
_CB_RESET_AFTER    = 120      # seconds after which a tripped circuit re-tries


from pipeline.state import *
from pipeline.tools import *
from pipeline.tools import _condition_sql_hints, _condition_search_terms, _distinct_patient_ids


# ══════════════════════════════════════════════════════════════════════════
# PER-HOSPITAL CIRCUIT BREAKER
# ══════════════════════════════════════════════════════════════════════════

class _CircuitBreaker:
    """
    Simple per-hospital circuit breaker.
    States: CLOSED (normal) → OPEN (stop sending) → HALF-OPEN (one probe).
    A hospital's circuit opens after _CB_FAILURE_THRESH consecutive failures
    and auto-resets after _CB_RESET_AFTER seconds.
    """
    def __init__(self):
        self._failures: dict[str, int]   = {}
        self._opened_at: dict[str, float] = {}

    def record_success(self, hospital_id: str) -> None:
        self._failures.pop(hospital_id, None)
        self._opened_at.pop(hospital_id, None)

    def record_failure(self, hospital_id: str) -> None:
        self._failures[hospital_id] = self._failures.get(hospital_id, 0) + 1
        if self._failures[hospital_id] >= _CB_FAILURE_THRESH:
            if hospital_id not in self._opened_at:
                self._opened_at[hospital_id] = time.monotonic()
                logger.error(
                    f"Circuit breaker OPENED for hospital '{hospital_id}' after "
                    f"{_CB_FAILURE_THRESH} consecutive failures. Will retry after "
                    f"{_CB_RESET_AFTER}s."
                )

    def is_open(self, hospital_id: str) -> bool:
        opened = self._opened_at.get(hospital_id)
        if opened is None:
            return False
        if time.monotonic() - opened > _CB_RESET_AFTER:
            logger.info(f"Circuit breaker HALF-OPEN probe for '{hospital_id}'.")
            return False   # allow one probe
        return True

    def degraded_response(self, hospital_id: str) -> str:
        return json.dumps({
            "error": (
                f"Hospital '{hospital_id}' database is temporarily unavailable "
                f"(circuit open after repeated failures). "
                f"Please retry in {_CB_RESET_AFTER} seconds."
            ),
            "rows": [],
        })


_circuit_breaker = _CircuitBreaker()
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

    # ── LLM retry helper ────────────────────────────────────────────────
    def _llm_with_retry(self, messages: list, caller: str = "llm"):
        """
        Invoke self.llm with exponential back-off on transient OpenAI errors
        (rate limits, timeouts, connection resets). Raises on the final attempt.
        """
        import openai  # import here to avoid polluting the top-level namespace
        _transient = (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )
        delay = _LLM_BASE_DELAY
        for attempt in range(1, _LLM_MAX_ATTEMPTS + 1):
            try:
                return self.llm.invoke(messages)
            except _transient as exc:
                if attempt == _LLM_MAX_ATTEMPTS:
                    logger.error(f"{caller}: OpenAI transient error after "
                                 f"{attempt} attempts: {exc}")
                    raise
                jitter = delay * (0.5 + 0.5 * (hash(str(exc)) % 100) / 100)
                logger.warning(
                    f"{caller}: transient OpenAI error (attempt {attempt}), "
                    f"retrying in {jitter:.1f}s — {exc}"
                )
                time.sleep(jitter)
                delay = min(delay * 2, _LLM_MAX_DELAY)

    # ── Whisper retry helper ───────────────────────────────────────────────
    def _whisper_with_retry(self, call_fn, caller: str = "whisper"):
        """
        Run a Whisper API call (passed as a zero-argument lambda) with
        exponential back-off on transient OpenAI errors, identical policy
        to _llm_with_retry.
        """
        import openai
        _transient = (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )
        delay = _LLM_BASE_DELAY
        for attempt in range(1, _LLM_MAX_ATTEMPTS + 1):
            try:
                return call_fn()
            except _transient as exc:
                if attempt == _LLM_MAX_ATTEMPTS:
                    logger.error(f"{caller}: Whisper transient error after "
                                 f"{attempt} attempts: {exc}")
                    raise
                jitter = delay * (0.5 + 0.5 * (hash(str(exc)) % 100) / 100)
                logger.warning(
                    f"{caller}: transient Whisper error (attempt {attempt}), "
                    f"retrying in {jitter:.1f}s — {exc}"
                )
                time.sleep(jitter)
                delay = min(delay * 2, _LLM_MAX_DELAY)

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
        """
        Returns the adapter for a hospital.  On first load:
          1. Validates the config is not obviously malformed (fail-fast).
          2. Tests the actual DB connection and raises a clear error if it fails
             rather than propagating opaque exceptions three agents deep.
          3. Checks the circuit breaker — if the hospital is tripped, raises
             immediately instead of spending time on a doomed connection.
        """
        if hospital_id in self._adapters:
            return self._adapters[hospital_id]

        # ── Circuit breaker check ────────────────────────────────────────
        if _circuit_breaker.is_open(hospital_id):
            raise RuntimeError(
                f"Hospital '{hospital_id}' circuit breaker is open — "
                "connection has failed repeatedly. Retry in a few minutes."
            )

        # ── Config validation (fail fast) ────────────────────────────────
        data_cfg = self.cfg.get_data_source(hospital_id)
        source_type = data_cfg.get("type", "")
        if source_type == "fhir":
            fhir_url = data_cfg.get("base_url", "").strip()
            if not fhir_url:
                raise ValueError(
                    f"Hospital '{hospital_id}' FHIR config is missing 'base_url'. "
                    "Fix the hospital configuration before running queries."
                )
        elif source_type in ("sql", "postgres", "postgresql", "sqlite", ""):
            conn_str = data_cfg.get("connection_string", "").strip()
            if not conn_str:
                raise ValueError(
                    f"Hospital '{hospital_id}' SQL config is missing "
                    "'connection_string'. Fix the hospital configuration."
                )

        # ── Attempt to build adapter + test connectivity ──────────────────
        try:
            adapter = create_adapter(data_cfg)
            # For SQL adapters: run a cheap sanity-check query immediately
            # so we surface a bad connection string NOW, not mid-query.
            if isinstance(adapter, DBAdapter):
                try:
                    with adapter.engine.connect() as _conn:
                        _conn.execute(__import__("sqlalchemy").text("SELECT 1"))
                except Exception as conn_exc:
                    _circuit_breaker.record_failure(hospital_id)
                    raise ConnectionError(
                        f"Hospital '{hospital_id}' database connection failed at startup: "
                        f"{conn_exc}. Check the connection string and DB availability."
                    ) from conn_exc
            self._adapters[hospital_id] = adapter
            _circuit_breaker.record_success(hospital_id)
            return adapter
        except (ValueError, ConnectionError):
            raise   # already descriptive — let it propagate
        except Exception as exc:
            _circuit_breaker.record_failure(hospital_id)
            raise RuntimeError(
                f"Failed to initialise adapter for hospital '{hospital_id}': {exc}"
            ) from exc

    def _get_trend_agent(self, hospital_id: str) -> Optional[TrendAgent]:
        if hospital_id not in self._trend_agents:
            adapter = self._get_adapter(hospital_id)
            if isinstance(adapter, DBAdapter):
                try:
                    self._trend_agents[hospital_id] = TrendAgent(adapter)
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

        response = self._llm_with_retry([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Clinical question: {question}")
        ], caller="agent1_sql_gen")
        raw_sql = response.content.strip().replace("```sql", "").replace("```", "").strip()
        result = self.sql_query_tool.invoke({"query": raw_sql, "hospital_id": hospital_id})
        parsed = json.loads(result)
        rows = parsed.get("rows", [])
        used_fallback = False

        # ── Hard stop on security rejections — never fall back to broad query ──
        # If the safety gate rejected the SQL (error starts with "Rejected:"),
        # return immediately with a security_rejected status. Do NOT silently
        # hand this off to get_broad_abnormal_labs — that would mean an
        # injection attempt always gets data through the fallback path.
        gate_error = parsed.get("error", "")
        if gate_error.startswith("Rejected:"):
            duration = int((time.monotonic() - start) * 1000)
            self.audit.log_agent_run(
                "data_extractor", state.get("session_id", ""),
                duration, False, rows_returned=0,
                user_id=state.get("user_id", ""), hospital_id=hospital_id,
            )
            rejection_msg = json.dumps({
                "rows": [],
                "error": gate_error,
                "security_rejection": True,
            })
            return {**state,
                    "sql_query_used": raw_sql,
                    "sql_result": rejection_msg,
                    "sql_error": gate_error,
                    "sql_retry_count": retry_count,
                    "data_status": "security_rejected",
                    "requested_patient_ids": [],
                    "found_patient_ids": [],
                    "missing_patient_ids": []}

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

        response = self._llm_with_retry([
            SystemMessage(content=(
                "Extract the SPECIFIC lab test names and their abnormal direction "
                "(high/low) from the patient data. Write a semantic search query using "
                "those EXACT lab names. Return ONLY the search query."
            )),
            HumanMessage(content=f"Question: {question}\n"
                                  f"Patient data: {sql_context[:1000]}")
        ], caller="agent2_semantic")
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
        response = self._llm_with_retry([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {question}\n\n"
                                  f"PATIENT DATA:\n{sql_result[:2000]}\n\n"
                                  f"GUIDELINES:\n{guideline_text[:2000]}")
        ], caller="clinical_reasoning")
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
        response = self._llm_with_retry([
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                "PATIENT DATA (ground truth):\n"
                f"{sql_result[:1500]}\n\n"
                "GUIDELINES:\n"
                f"{guidelines[:1000]}\n\n"
                "DRAFT:\n"
                f"{reasoning}"
            ))
        ], caller="agent4_critic")
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
                verbatim_resp = self._whisper_with_retry(
                    lambda: self.openai_client.audio.transcriptions.create(
                        model="whisper-1", file=_make_audio_file(), response_format="verbose_json"
                    ), caller="whisper_verbatim"
                )
                detected_lang = (getattr(verbatim_resp, "language", None) or "english").lower()
                original_text = verbatim_resp.text.strip()

                # If Whisper detected non-English, translate to English
                if detected_lang not in ("english", "en"):
                    translated_resp = self._whisper_with_retry(
                        lambda: self.openai_client.audio.translations.create(
                            model="whisper-1", file=_make_audio_file()
                        ), caller="whisper_translate_auto"
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
                resp = self._whisper_with_retry(
                    lambda: self.openai_client.audio.transcriptions.create(
                        model="whisper-1", file=_make_audio_file(), language="en"
                    ), caller="whisper_en"
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
                verbatim_resp = self._whisper_with_retry(
                    lambda: self.openai_client.audio.transcriptions.create(
                        model="whisper-1", file=_make_audio_file(), language=lang_code
                    ), caller="whisper_verbatim_lang"
                )
                translated_resp = self._whisper_with_retry(
                    lambda: self.openai_client.audio.translations.create(
                        model="whisper-1", file=_make_audio_file()
                    ), caller="whisper_translate_lang"
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

    # ── Question-level injection guard ───────────────────────────────────
    # Detects SQL injection embedded directly in the clinical question string
    # (e.g. "show me patient 1; DELETE FROM patients"). This is a separate
    # attack vector from the SQL gate in run_query (which catches harmful
    # LLM-generated SQL). Here we catch it BEFORE the LLM sees the input.
    _QUESTION_INJECTION = re.compile(
        r";\s*(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER|TRUNCATE|REPLACE|MERGE|"
        r"EXEC|EXECUTE|GRANT|REVOKE)\b",
        re.IGNORECASE,
    )

    def _check_question_injection(self, question: str) -> str | None:
        """Returns a rejection reason string if question looks injected, else None."""
        m = self._QUESTION_INJECTION.search(question)
        if m:
            return f"Rejected: forbidden SQL keyword '{m.group(1)}' detected in question."
        return None

    def run_until_review(self, clinical_question: str, config: dict,
                         user_id: str = "anon", hospital_id: str = "demo",
                         session_id: str = "",
                         input_mode: str = "text",
                         spoken_language: str = "",
                         original_transcript: str = "") -> MIRAState:

        # ── Hard gate: reject questions that contain SQL injection patterns ──
        rejection = self._check_question_injection(clinical_question)
        if rejection:
            self.audit.log_query(user_id, hospital_id, session_id,
                                 config["configurable"]["thread_id"],
                                 clinical_question, len(clinical_question))
            rejection_result = json.dumps({
                "rows": [], "error": rejection, "security_rejection": True
            })
            return {
                **make_initial_state(clinical_question, user_id, hospital_id,
                                     session_id, input_mode=input_mode),
                "sql_result":  rejection_result,
                "sql_error":   rejection,
                "data_status": "security_rejected",
                "clinical_reasoning": (
                    "This question was blocked before processing: it contains a "
                    "SQL injection pattern that is not permitted."
                ),
                "safety_flags": ["question_injection_blocked"],
            }

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

        # ── Idempotency guard ─────────────────────────────────────────────────
        # A doctor double-clicking "Approve" (or the UI retrying a POST) must
        # not re-run the critic agent or double-log the audit event.
        # If the state already shows a finalised decision that matches the
        # incoming one, return immediately.
        existing_decision = current_state.get("human_decision", "")
        already_approved  = current_state.get("approved", False)
        if decision == "approve" and already_approved:
            logger.info(
                f"submit_human_decision: duplicate 'approve' on thread {thread_id} — "
                "returning cached final state (idempotent)."
            )
            return current_state
        if decision == "reject" and existing_decision == "reject" and not feedback:
            # Reject with no new feedback on an already-rejected state — no-op.
            logger.info(
                f"submit_human_decision: duplicate 'reject' (no new feedback) on "
                f"thread {thread_id} — returning cached state (idempotent)."
            )
            return current_state

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
            response = self._llm_with_retry(
            [SystemMessage(content=prompt)],
            caller="ambient_soap"
        )
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
