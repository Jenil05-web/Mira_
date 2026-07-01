"""
mira_pipeline_prod.py
======================
MIRA Production Pipeline.

DROP-IN REPLACEMENT for mira_pipeline.py with:
  - ConfigManager    → secrets from GCP / env / .env
  - DBAdapter        → schema-agnostic SQL (SQLite dev → Supabase prod)
  - FHIRAdapter      → plug-and-play hospital EHR connection
  - AuditLogger      → HIPAA append-only trail on every agent call
  - PostgresSaver    → persistent LangGraph checkpoints (Supabase)
  - pgvector search  → replaces local FAISS when Supabase is configured
  - MemorySaver      → automatic fallback when Supabase not configured

BACKWARDS COMPATIBLE:
  If no Supabase credentials are set, the pipeline behaves exactly like
  the original mira_pipeline.py — SQLite + FAISS + MemorySaver.
  Adding Supabase credentials upgrades all three automatically.

INSTALL (production extras beyond base requirements):
  pip install sqlalchemy psycopg2-binary langgraph-checkpoint-postgres
"""

import json
import logging
import os
import pickle
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
from audit_logger import AuditLogger
from Mira_project.trend_agent import TrendAgent

logger = logging.getLogger(__name__)


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


def make_initial_state(clinical_question: str,
                       user_id: str = "anon",
                       hospital_id: str = "demo",
                       session_id: str = "") -> MIRAState:
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
        if self._type == "pgvector":
            return self._search_pgvector(query, k, hospital_id)
        return self._search_faiss(query, k)

    def _search_pgvector(self, query: str, k: int,
                         hospital_id: str) -> list[dict]:
        from sqlalchemy import text
        vec = self._embed(query)[0].tolist()
        with self._engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT source, topic, content,
                       1 - (embedding <=> :vec::vector) AS similarity
                FROM mira_embeddings
                WHERE hospital_id = :hid OR hospital_id = 'global'
                ORDER BY embedding <=> :vec::vector
                LIMIT :k
            """), {"vec": str(vec), "hid": hospital_id, "k": k}).fetchall()
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
        ck_cfg = self.cfg.get_checkpoint_config()
        if ck_cfg["type"] == "postgres":
            try:
                from langgraph.checkpoint.postgres import PostgresSaver
                self.checkpointer = PostgresSaver.from_conn_string(
                    ck_cfg["connection_string"]
                )
                logger.info("Checkpointer: PostgresSaver (Supabase)")
            except Exception as e:
                logger.warning(f"PostgresSaver failed ({e}), falling back to MemorySaver")
                self.checkpointer = MemorySaver()
        else:
            self.checkpointer = MemorySaver()
            logger.info("Checkpointer: MemorySaver (dev)")

        # ── Per-hospital data adapters (lazy, cached) ─────────────────
        self._adapters: dict[str, object] = {}

        # ── Trend agent (shared, uses adapter connection) ────────────────
        self._trend_agents: dict[str, TrendAgent] = {}

        # ── Build tools + graph ───────────────────────────────────────────
        self._build_tools()
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

    # ── Agent 1 — SQL Data Extractor ─────────────────────────────────────
    def agent1_sql_extractor(self, state: MIRAState) -> MIRAState:
        hospital_id = state.get("hospital_id", "default")
        retry_count = state.get("sql_retry_count", 0)
        previous_error = state.get("sql_error", "")
        error_context = (
            f"\nYour previous SQL failed: {previous_error}\nFix it."
            if previous_error else ""
        )

        schema = self._schema_for(hospital_id)

        system_prompt = f"""You are a medical SQL expert. Write a single valid SQL SELECT query.

DATABASE SCHEMA:
{schema}

VOCABULARY:
- "critical"/"severe" → valuenum > ref_range_upper * 1.5 OR valuenum < ref_range_lower * 0.5 OR flag IS NOT NULL
- "abnormal" → valuenum > ref_range_upper OR valuenum < ref_range_lower OR flag IS NOT NULL
- "high"/"elevated" → valuenum > ref_range_upper
- "low" → valuenum < ref_range_lower
- For lab names: use LIKE '%name%' not exact match
- Always include value, valuenum, valueuom, charttime, ref_range_lower, ref_range_upper, flag in SELECT
- Always JOIN patient demographics so age/gender are available
- Guard NULL ref ranges: (ref_range_upper IS NOT NULL AND valuenum > ref_range_upper)
- ORDER BY severity when relevant; LIMIT 25
- Return ONLY raw SQL, no markdown{error_context}"""

        start = time.monotonic()
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Clinical question: {state['clinical_question']}")
        ])
        raw_sql = response.content.strip().replace("```sql", "").replace("```", "").strip()

        result = self.sql_query_tool.invoke({"query": raw_sql, "hospital_id": hospital_id})
        parsed = json.loads(result)
        duration = int((time.monotonic() - start) * 1000)

        if "error" in parsed:
            self.audit.log_agent_run("sql_extractor", state.get("session_id", ""),
                                     duration, False, error=parsed["error"],
                                     user_id=state.get("user_id", ""),
                                     hospital_id=hospital_id)
            return {**state, "sql_query_used": raw_sql, "sql_result": "",
                    "sql_error": parsed["error"], "sql_retry_count": retry_count + 1}

        rows = parsed.get("rows", [])
        if len(rows) == 0 and retry_count < 3:
            return {
                **state, "sql_query_used": raw_sql, "sql_result": "",
                "sql_error": (
                    "Query returned 0 rows. Filter likely too strict. "
                    "Use LIKE for lab names and compare valuenum against ref ranges "
                    "instead of relying on the flag column."
                ),
                "sql_retry_count": retry_count + 1
            }

        self.audit.log_agent_run("sql_extractor", state.get("session_id", ""),
                                 duration, True, rows_returned=len(rows),
                                 user_id=state.get("user_id", ""),
                                 hospital_id=hospital_id)
        self.audit.log_data_access("sql", "lab_observations", len(rows),
                                   state.get("session_id", ""),
                                   state.get("user_id", ""), hospital_id)

        return {**state, "sql_query_used": raw_sql, "sql_result": result,
                "sql_error": "", "sql_retry_count": retry_count}

    def should_retry_sql(self, state: MIRAState) -> str:
        if state.get("sql_error") and state.get("sql_retry_count", 0) < 3:
            return "retry"
        return "ok"

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

        response = self.llm.invoke([
            SystemMessage(content=(
                "Extract the SPECIFIC lab test names and their abnormal direction "
                "(high/low) from the patient data. Write a semantic search query using "
                "those EXACT lab names. Return ONLY the search query."
            )),
            HumanMessage(content=f"Question: {state['clinical_question']}\n"
                                  f"Patient data: {sql_context[:1000]}")
        ])
        search_query = response.content.strip()

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

        trend_context = ""
        try:
            td = state.get("trend_data", "")
            if td:
                trend_parsed = json.loads(td)
                trend_context = (
                    f"\n\nLAB TRAJECTORY:\n{trend_parsed.get('summary', '')}"
                )
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
                f"\n\nCLINICIAN FEEDBACK (revise accordingly):\n{state['human_feedback']}"
            )

        system_prompt = f"""You are an expert clinical AI assistant. Synthesize data into:
## Patient Summary
## Identified Concerns
## Clinical Guideline Context
## Recommended Actions
Ground every claim in the data or a cited guideline. Never hallucinate values.
Only use numbers/IDs/results that appear verbatim in the patient data below.
If a lab trajectory is provided, weight it heavily — worsening trend > single abnormal value.
If zero rows returned after retries, say "no patients matched this threshold" not "data unavailable."{trend_context}{relevance_warning}{feedback_context}"""

        start = time.monotonic()
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {state['clinical_question']}\n\n"
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

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {state['clinical_question']}\n\n"
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

        system_prompt = """You are a medical AI safety critic AND editor.
Check for: hallucinated values not in patient data, guideline mismatch,
missing recommendations, dangerous omissions.

CRITICAL: final_report must ALWAYS be a complete corrected clinical report
in the format (Patient Summary / Identified Concerns / Clinical Guideline Context /
Recommended Actions). NEVER put critique text or complaint lists into final_report.
Put your review notes in "corrections" only.

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

        critique_markers = ["the analysis contains", "needs to be revised",
                            "the analysis does not", "hallucinated value",
                            "issues that need addressing"]
        if any(m in final_report.lower()[:300] for m in critique_markers):
            final_report = reasoning

        approved = critic_output.get("approved", True)
        safety_flags = critic_output.get("safety_flags", [])

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
        builder.add_node("trend_check",        self.agent_trend_check)
        builder.add_node("semantic_crossref",  self.agent2_semantic_crossref)
        builder.add_node("clinical_reasoning", self.agent3_clinical_reasoning)
        builder.add_node("human_review",       self.human_review_node)
        builder.add_node("critic_safety",      self.agent4_critic_safety)

        builder.set_entry_point("sql_extractor")

        builder.add_conditional_edges(
            "sql_extractor", self.should_retry_sql,
            {"retry": "sql_extractor", "ok": "trend_check"}
        )
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

    def run_until_review(self, clinical_question: str, config: dict,
                         user_id: str = "anon", hospital_id: str = "demo",
                         session_id: str = "") -> MIRAState:
        initial = make_initial_state(clinical_question, user_id, hospital_id, session_id)
        self.audit.log_query(user_id, hospital_id, session_id,
                             config["configurable"]["thread_id"],
                             clinical_question, len(clinical_question))
        return self.graph.invoke(initial, config)

    def submit_human_decision(self, config: dict, decision: str,
                               feedback: str = "",
                               user_id: str = "",
                               hospital_id: str = "") -> MIRAState:
        thread_id = config["configurable"]["thread_id"]
        self.graph.update_state(config, {
            "human_decision": decision,
            "human_feedback": feedback
        })
        self.audit.log_human_review(user_id, thread_id, decision,
                                    bool(feedback), hospital_id)
        return self.graph.invoke(None, config)

    def get_current_state(self, config: dict) -> MIRAState:
        return self.graph.get_state(config).values


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
    question = "Which patients have abnormal lab results?"
    print(f"\nQuery: {question}")
    paused = engine.run_until_review(question, cfg, user_id="dev", hospital_id="demo")
    print(f"\n🛑 Paused. Preview:\n{paused['clinical_reasoning'][:400]}...")
    final = engine.submit_human_decision(cfg, "approve", user_id="dev", hospital_id="demo")
    print(f"\n✅ Final report ({len(final['final_report'])} chars)")
    print(f"Approved: {final['approved']}")
    print(f"Safety flags: {final['safety_flags'] or 'None'}")