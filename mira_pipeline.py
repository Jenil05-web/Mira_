"""
mira_pipeline.py
=================
Core logic for MIRA — Multi-Agent Clinical Audit & Real-Time Triage System.

This file contains NO UI code. It is the backend brain:
  - Data layer connections (SQLite + FAISS)
  - Tool definitions (sql_query, vector_search)
  - The 4 agent functions
  - The LangGraph pipeline (with Human-in-the-Loop interrupt + streaming)

Import this module from streamlit_app.py to power the UI.
"""

import os
import json
import sqlite3
import pickle
import uuid
from pathlib import Path
from typing import TypedDict, Generator

import numpy as np
import pandas as pd
import faiss
from openai import OpenAI
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()  # reads .env in the project root into os.environ


# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

class Config:
    """Central config. Set OPENAI_API_KEY in a .env file — never hardcode it here."""
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    DB_PATH        = Path("./mira_data/mimic.db")
    FAISS_PATH     = Path("./mira_data/medical_faiss.index")
    META_PATH      = Path("./mira_data/faiss_metadata.pkl")
    SCHEMA_PATH    = Path("./mira_data/db_schema.txt")
    LLM_MODEL      = "gpt-4o"
    EMBEDDING_MODEL = "text-embedding-3-small"
    MAX_SQL_RETRIES = 3

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Create a .env file in the project "
                "root with:\n  OPENAI_API_KEY=sk-your-key-here"
            )


# ══════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ══════════════════════════════════════════════════════════════════════════

class MIRAState(TypedDict):
    # Input
    clinical_question: str

    # Agent 1 — SQL Data Extractor
    sql_query_used: str
    sql_result: str
    sql_retry_count: int
    sql_error: str

    # Agent 2 — Semantic Cross-Ref
    search_query_used: str
    guidelines: str

    # Agent 3 — Clinical Reasoning
    clinical_reasoning: str

    # Agent 4 — Critic & Safety
    final_report: str
    safety_flags: list[str]
    approved: bool

    # Human-in-the-Loop
    human_decision: str   # "approve" / "reject" / ""
    human_feedback: str


def make_initial_state(clinical_question: str) -> MIRAState:
    """Factory for a clean starting state."""
    return {
        "clinical_question": clinical_question,
        "sql_query_used": "", "sql_result": "", "sql_retry_count": 0, "sql_error": "",
        "search_query_used": "", "guidelines": "", "clinical_reasoning": "",
        "final_report": "", "safety_flags": [], "approved": False,
        "human_decision": "", "human_feedback": ""
    }


# ══════════════════════════════════════════════════════════════════════════
# MIRA ENGINE — wraps data layer, tools, agents, and the compiled graph
# ══════════════════════════════════════════════════════════════════════════

class MIRAEngine:
    """
    Owns all connections (SQLite, FAISS, OpenAI client, LLM) and exposes
    the compiled LangGraph pipeline. Instantiate once per app session.
    """

    def __init__(self, config: Config = Config()):
        self.config = config
        self.config.validate()
        self._connect_data_layer()
        self._build_tools()
        self._build_llm()
        self._build_graph()

    # ── Data layer ───────────────────────────────────────────────────────
    def _connect_data_layer(self):
        if not self.config.DB_PATH.exists():
            raise FileNotFoundError(
                f"SQLite DB not found at {self.config.DB_PATH}. "
                "Run notebook 01_data_setup.ipynb first."
            )
        self.conn = sqlite3.connect(self.config.DB_PATH, check_same_thread=False)
        self.db_schema = self.config.SCHEMA_PATH.read_text()

        self.faiss_index = faiss.read_index(str(self.config.FAISS_PATH))
        with open(self.config.META_PATH, "rb") as f:
            self.guidelines_metadata = pickle.load(f)

        self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        response = self.openai_client.embeddings.create(
            model=self.config.EMBEDDING_MODEL, input=texts
        )
        return np.array([r.embedding for r in response.data], dtype=np.float32)

    # ── Tools ────────────────────────────────────────────────────────────
    def _build_tools(self):
        engine = self  # closure capture

        @tool
        def sql_query(query: str) -> str:
            """
            Execute a SQL SELECT query against the MIMIC-IV patient database.
            Tables: patients, admissions, labevents, d_labitems, diagnoses_icd.
            Always JOIN d_labitems ON labevents.itemid = d_labitems.itemid for readable lab names.
            Use LIKE '%name%' for lab name matching, not exact equality — labels vary
            (e.g. 'Creatinine' vs 'Creatinine, Serum'). Always include value/valuenum/valueuom/
            charttime/ref_range_lower/ref_range_upper/flag columns when the question is about lab
            results. Do not over-filter on flag = 'abnormal' since it is frequently NULL even for
            clinically abnormal values — compare against ref_range_lower/upper instead.
            Returns a JSON string of results, or an error message.
            """
            try:
                df = pd.read_sql_query(query, engine.conn)
                if df.empty:
                    return json.dumps({"result": "No data found."})
                return json.dumps({"rows": df.head(20).to_dict(orient="records")}, default=str)
            except Exception as e:
                return json.dumps({"error": str(e), "hint": "Check table/column names against the schema."})

        @tool
        def vector_search(query: str, k: int = 3) -> str:
            """
            Search the medical knowledge base using semantic similarity.
            Returns top-k most relevant clinical guideline chunks with source citations.
            """
            try:
                query_vec = engine._get_embeddings([query])
                distances, indices = engine.faiss_index.search(query_vec, k)
                results = []
                for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                    chunk = engine.guidelines_metadata[idx].copy()
                    chunk["rank"] = rank + 1
                    chunk["relevance_score"] = round(1 / (1 + float(dist)), 4)
                    results.append(chunk)
                return json.dumps({"guidelines": results}, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})

        self.sql_query_tool = sql_query
        self.vector_search_tool = vector_search

    # ── LLM ──────────────────────────────────────────────────────────────
    def _build_llm(self):
        os.environ["OPENAI_API_KEY"] = self.config.OPENAI_API_KEY
        self.llm = ChatOpenAI(model=self.config.LLM_MODEL, temperature=0, streaming=True)

    # ── Agent 1 — SQL Data Extractor ────────────────────────────────────
    def agent1_sql_extractor(self, state: MIRAState) -> MIRAState:
        retry_count = state.get("sql_retry_count", 0)
        previous_error = state.get("sql_error", "")
        error_context = f"\nYour previous SQL failed: {previous_error}\nFix it." if previous_error else ""

        system_prompt = f"""You are a medical SQL expert. Write a single valid SQLite SELECT query.
DATABASE SCHEMA:
{self.db_schema}

VOCABULARY — map vague clinical phrasing to concrete SQL logic:
- "critical" / "severe" / "dangerous" lab results → valuenum > ref_range_upper * 1.5
  OR valuenum < ref_range_lower * 0.5 (far outside normal range), OR flag IS NOT NULL.
  Combine with OR so you don't miss real cases just because flag is NULL.
- "abnormal" lab results → valuenum > ref_range_upper OR valuenum < ref_range_lower
  OR flag IS NOT NULL.
- "high" / "elevated" → valuenum > ref_range_upper.
- "low" → valuenum < ref_range_lower.
- If the question names a specific lab (creatinine, glucose, sodium, potassium, etc.),
  filter d_labitems.label LIKE '%name%' (case-insensitive substring, never exact match —
  labels vary like 'Creatinine, Serum').
- If the question does NOT name a specific lab and just says "lab results" or "labs",
  do NOT filter by label at all — search across all lab types.
- If the question mentions a named patient, you cannot filter by name (no name column exists);
  state in a comment that patient names aren't in this dataset, only subject_id.

RULES:
- Always JOIN d_labitems ON labevents.itemid = d_labitems.itemid
- ALWAYS include labevents.value, labevents.valuenum, labevents.valueuom, labevents.charttime,
  labevents.ref_range_lower, labevents.ref_range_upper, and labevents.flag in the SELECT —
  never select only patient demographic columns when the question is about lab results.
- ref_range_lower/upper can be NULL for some lab types — guard with
  (ref_range_upper IS NOT NULL AND valuenum > ref_range_upper) to avoid SQL errors on NULL comparisons.
- Always also JOIN patients ON labevents.subject_id = patients.subject_id so gender/age are available.
- Order results by how far valuenum deviates from the reference range when the question is about
  severity (critical/severe), so the most extreme cases appear first.
- LIMIT to 25 rows unless the question asks for a count or aggregate.
- Return ONLY raw SQL, no markdown
{error_context}"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Clinical question: {state['clinical_question']}")
        ])
        raw_sql = response.content.strip().replace("```sql", "").replace("```", "").strip()

        result = self.sql_query_tool.invoke({"query": raw_sql})
        parsed = json.loads(result)

        if "error" in parsed:
            return {**state, "sql_query_used": raw_sql, "sql_result": "",
                    "sql_error": parsed["error"], "sql_retry_count": retry_count + 1}

        rows = parsed.get("rows", [])
        if len(rows) == 0 and retry_count < self.config.MAX_SQL_RETRIES:
            # Successfully ran but returned nothing — likely over-filtered.
            # Treat as a soft error so Agent 1 retries with looser logic.
            return {
                **state, "sql_query_used": raw_sql, "sql_result": "",
                "sql_error": (
                    "Query ran successfully but returned 0 rows. Your filter is "
                    "likely too strict (e.g. exact label match, or flag = 'abnormal' "
                    "when flag is often NULL). Loosen the WHERE clause — use LIKE for "
                    "lab names and compare valuenum against ref_range bounds instead "
                    "of relying on the flag column."
                ),
                "sql_retry_count": retry_count + 1
            }

        return {**state, "sql_query_used": raw_sql, "sql_result": result,
                "sql_error": "", "sql_retry_count": retry_count}

    def should_retry_sql(self, state: MIRAState) -> str:
        if state.get("sql_error") and state.get("sql_retry_count", 0) < self.config.MAX_SQL_RETRIES:
            return "retry"
        return "ok"

    # ── Agent 2 — Semantic Cross-Ref ────────────────────────────────────
    def agent2_semantic_crossref(self, state: MIRAState) -> MIRAState:
        sql_context = state.get("sql_result", "") or "No patient data retrieved."

        response = self.llm.invoke([
            SystemMessage(content="Write a 1-2 sentence semantic search query for clinical "
                                   "guidelines based on the patient data. Return ONLY the query."),
            HumanMessage(content=f"Question: {state['clinical_question']}\nPatient data: {sql_context[:1000]}")
        ])
        search_query = response.content.strip()

        guidelines_result = self.vector_search_tool.invoke({"query": search_query, "k": 3})

        return {**state, "search_query_used": search_query, "guidelines": guidelines_result}

    # ── Agent 3 — Clinical Reasoning ────────────────────────────────────
    def _build_guideline_text(self, guidelines_json: str) -> str:
        try:
            text = ""
            for g in json.loads(guidelines_json).get("guidelines", []):
                text += f"\n[{g['source']}] {g['topic']}:\n{g['text']}\n"
            return text
        except Exception:
            return guidelines_json

    def agent3_clinical_reasoning(self, state: MIRAState) -> MIRAState:
        sql_result = state.get("sql_result", "No patient data.")
        guideline_text = self._build_guideline_text(state.get("guidelines", "No guidelines."))

        feedback_context = ""
        if state.get("human_decision") == "reject" and state.get("human_feedback"):
            feedback_context = (
                f"\n\nIMPORTANT — A clinician reviewed your previous analysis and rejected it "
                f"with this feedback:\n{state['human_feedback']}\nRevise your analysis accordingly."
            )

        system_prompt = f"""You are an expert clinical AI assistant. Synthesize patient data + guidelines into:
## Patient Summary
## Identified Concerns
## Clinical Guideline Context
## Recommended Actions
Ground every claim in the data or a cited guideline. Never hallucinate values.
If the patient data shows zero rows or an error after retries, say plainly that no patients
matched this specific threshold in the available dataset — do NOT claim the data type itself
is "not provided" if the schema includes that column; the dataset has lab values, the query
just didn't find a match this time.{feedback_context}"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {state['clinical_question']}\n\n"
                                  f"PATIENT DATA:\n{sql_result[:2000]}\n\n"
                                  f"GUIDELINES:\n{guideline_text[:2000]}")
        ])
        reasoning = response.content.strip()

        return {**state, "clinical_reasoning": reasoning, "human_decision": "", "human_feedback": ""}

    def stream_clinical_reasoning(self, state: MIRAState) -> Generator[str, None, str]:
        """
        Generator version of Agent 3 for UI streaming.
        Yields tokens one at a time; returns (via StopIteration.value pattern)
        the full text at the end. Used by the Streamlit layer with st.write_stream.
        """
        sql_result = state.get("sql_result", "No patient data.")
        guideline_text = self._build_guideline_text(state.get("guidelines", "No guidelines."))

        feedback_context = ""
        if state.get("human_decision") == "reject" and state.get("human_feedback"):
            feedback_context = (
                f"\n\nIMPORTANT — A clinician rejected your previous analysis with this feedback:\n"
                f"{state['human_feedback']}\nRevise accordingly."
            )

        system_prompt = f"""You are an expert clinical AI assistant. Synthesize patient data + guidelines into:
## Patient Summary
## Identified Concerns
## Clinical Guideline Context
## Recommended Actions
Ground every claim in the data or a cited guideline. Never hallucinate values.
If the patient data shows zero rows or an error, say plainly that no patients matched this
specific threshold — do NOT claim the data type itself is "not provided" if the schema includes
that column.{feedback_context}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {state['clinical_question']}\n\n"
                                  f"PATIENT DATA:\n{sql_result[:2000]}\n\n"
                                  f"GUIDELINES:\n{guideline_text[:2000]}")
        ]

        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

    # ── Human Review Node (pause point) ─────────────────────────────────
    def human_review_node(self, state: MIRAState) -> MIRAState:
        """No-op node. Pausing happens via interrupt_before at compile time."""
        return state

    def route_after_human_review(self, state: MIRAState) -> str:
        if state.get("human_decision") == "reject":
            return "revise"
        return "proceed"

    # ── Agent 4 — Critic & Safety ───────────────────────────────────────
    def agent4_critic_safety(self, state: MIRAState) -> MIRAState:
        reasoning  = state.get("clinical_reasoning", "")
        sql_result = state.get("sql_result", "")
        guidelines = state.get("guidelines", "")

        system_prompt = """You are a medical AI safety critic. Check for hallucinated values,
guideline grounding, dangerous recommendations, and completeness.
Respond ONLY in JSON: {"approved": true/false, "safety_flags": [], "final_report": "..."}"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"PATIENT DATA:\n{sql_result[:1500]}\n\n"
                                  f"GUIDELINES:\n{guidelines[:1000]}\n\n"
                                  f"ANALYSIS:\n{reasoning}")
        ])

        try:
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            critic_output = json.loads(raw)
        except Exception:
            critic_output = {"approved": True, "safety_flags": [], "final_report": reasoning}

        return {
            **state,
            "final_report": critic_output.get("final_report", reasoning),
            "safety_flags": critic_output.get("safety_flags", []),
            "approved": critic_output.get("approved", True),
        }

    # ── Build & compile the LangGraph ───────────────────────────────────
    def _build_graph(self):
        builder = StateGraph(MIRAState)

        builder.add_node("sql_extractor",      self.agent1_sql_extractor)
        builder.add_node("semantic_crossref",  self.agent2_semantic_crossref)
        builder.add_node("clinical_reasoning", self.agent3_clinical_reasoning)
        builder.add_node("human_review",       self.human_review_node)
        builder.add_node("critic_safety",      self.agent4_critic_safety)

        builder.set_entry_point("sql_extractor")

        builder.add_conditional_edges(
            "sql_extractor", self.should_retry_sql,
            {"retry": "sql_extractor", "ok": "semantic_crossref"}
        )
        builder.add_edge("semantic_crossref", "clinical_reasoning")
        builder.add_edge("clinical_reasoning", "human_review")
        builder.add_conditional_edges(
            "human_review", self.route_after_human_review,
            {"proceed": "critic_safety", "revise": "clinical_reasoning"}
        )
        builder.add_edge("critic_safety", END)

        self.checkpointer = MemorySaver()
        self.graph = builder.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["human_review"]
        )

    # ── Public API for the UI layer ─────────────────────────────────────
    def new_thread(self) -> dict:
        """Returns a fresh LangGraph config with a unique thread_id."""
        return {"configurable": {"thread_id": str(uuid.uuid4())}}

    def run_until_review(self, clinical_question: str, config: dict) -> MIRAState:
        """Runs Agents 1 → 2 → 3, pauses before human_review. Returns the paused state."""
        initial_state = make_initial_state(clinical_question)
        return self.graph.invoke(initial_state, config)

    def submit_human_decision(self, config: dict, decision: str, feedback: str = "") -> MIRAState:
        """
        Submits the doctor's decision and resumes the graph.
        decision: "approve" or "reject"
        Returns the final state (either final_report if approved, or a fresh
        paused state at human_review again if rejected and revised).
        """
        self.graph.update_state(config, {"human_decision": decision, "human_feedback": feedback})
        return self.graph.invoke(None, config)

    def get_current_state(self, config: dict) -> MIRAState:
        """Fetch the latest checkpointed state for a thread (for UI refresh)."""
        snapshot = self.graph.get_state(config)
        return snapshot.values


# ══════════════════════════════════════════════════════════════════════════
# Module-level singleton helper (optional convenience for simple scripts)
# ══════════════════════════════════════════════════════════════════════════

_engine_instance: MIRAEngine | None = None


def get_engine(config: Config = None) -> MIRAEngine:
    """Lazily creates and caches a single MIRAEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = MIRAEngine(config or Config())
    return _engine_instance


if __name__ == "__main__":
    # Quick CLI smoke test — run `python mira_pipeline.py` to sanity check
    engine = get_engine()
    cfg = engine.new_thread()
    question = "Which patients have abnormal lab results? Summarize the key concerns."
    print(f"🏥 Running MIRA for: {question}\n")
    paused_state = engine.run_until_review(question, cfg)
    print("🛑 Paused for human review. Agent 3's draft:\n")
    print(paused_state["clinical_reasoning"])
    print("\n▶️  Auto-approving for smoke test...")
    final_state = engine.submit_human_decision(cfg, "approve")
    print("\n✅ FINAL REPORT:\n")
    print(final_state["final_report"])