"""
mira_pipeline.py
=================
Core logic for MIRA — Multi-Agent Clinical Audit & Real-Time Triage System.

This file contains NO UI code. It is the backend brain:
  - Data layer connections (SQLite + FAISS)
  - Tool definitions (sql_query, vector_search)
  - The 4 core agents + optional TrendAgent (Agent 1b)
  - The LangGraph pipeline (with Human-in-the-Loop interrupt + streaming)

Import this module from streamlit_app.py to power the UI.

CHANGELOG — TrendAgent integration:
  - MIRAState gains two new fields: trend_result (dict) and trend_summary (str)
  - MIRAEngine._connect_data_layer() also instantiates a TrendAgent
  - New method: agent1b_trend_analysis() — runs after Agent 1 succeeds,
    extracts the primary lab name + first subject_id from the SQL result,
    and populates state["trend_result"] / state["trend_summary"].
  - agent3_clinical_reasoning() and stream_clinical_reasoning() now include
    trend_summary as a third data source in the prompt.
  - New graph node "trend_analysis" sits between sql_extractor and
    semantic_crossref. It is skipped gracefully if SQL returned no rows.
  - Public helper get_trend_result() lets the UI surface trend data separately.
"""

import os
import json
import re
import sqlite3
import pickle
import uuid
from pathlib import Path
from typing import TypedDict, Generator, Optional

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

# ── Suppress benign NumPy warnings 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# ── TrendAgent lives in trend_agent.py alongside this file ───────────────
from trend_agent import TrendAgent

load_dotenv()  # reads .env in the project root into os.environ

# BASE DIRECTORY — for absolute paths


BASE_DIR = Path(__file__).parent

# CONFIG


class Config:
    OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY")
    DB_PATH         = BASE_DIR / "mira_data" / "mimic.db"
    FAISS_PATH      = BASE_DIR / "mira_data" / "medical_faiss.index"
    META_PATH       = BASE_DIR / "mira_data" / "faiss_metadata.pkl"
    SCHEMA_PATH     = BASE_DIR / "mira_data" / "db_schema.txt"
    LLM_MODEL       = "gpt-4o"
    EMBEDDING_MODEL = "text-embedding-3-small"
    MAX_SQL_RETRIES = 3

    @classmethod 
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Create a .env file in the project "
                "root with:\n  OPENAI_API_KEY=sk-your-key-here"
            )
        # Ensure data directory exists
        if not cls.DB_PATH.parent.exists():
            raise FileNotFoundError(
                f"Data directory not found at {cls.DB_PATH.parent}. "
                "Run notebook 01_data_setup.ipynb first."
            )

# STATE DEFINITION


class MIRAState(TypedDict):
    # Input
    clinical_question: str

    # Agent 1 — SQL Data Extractor
    sql_query_used: str
    sql_result: str
    sql_retry_count: int
    sql_error: str

    # Agent 1b — Lab Trajectory Analysis (TrendAgent)
    # trend_result holds the full dict returned by TrendAgent.analyze_patient_lab()
    # trend_summary is the human-readable string passed into Agent 3's prompt
    trend_result: dict
    trend_summary: str

    # Agent 2 — Semantic Cross-Ref
    search_query_used: str
    guidelines: str

    # Agent 3 — Clinical Reasoning
    clinical_analysis: str

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
        "trend_result": {}, "trend_summary": "",
        "search_query_used": "", "guidelines": "", "clinical_analysis": "",
        "final_report": "", "safety_flags": [], "approved": False,
        "human_decision": "", "human_feedback": ""
    }



# MIRA ENGINE — wraps data layer, tools, agents, and the compiled graph


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

    # ── Data layer 
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

        # ── TrendAgent shares the same SQLite connection ─────────────────
        self.trend_agent = TrendAgent(self.conn)

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        response = self.openai_client.embeddings.create(
            model=self.config.EMBEDDING_MODEL, input=texts
        )
        return np.array([r.embedding for r in response.data], dtype=np.float32)

    # ── Tools 
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

    # ── Agent 1 — SQL Data Extractor ─────────────────────────────────────
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

    # ── Agent 1b — Lab Trajectory Analysis (TrendAgent) 
    def agent1b_trend_analysis(self, state: MIRAState) -> MIRAState:
        """
        Runs after Agent 1 succeeds. Extracts the primary lab name and the
        first subject_id from the SQL result rows, then runs TrendAgent to
        detect worsening trajectories across that patient's history.

        Gracefully skips (returns empty trend fields) when:
          - SQL returned no rows
          - No recognisable lab name can be inferred from the question
          - TrendAgent itself returns insufficient_data
        """
        sql_result = state.get("sql_result", "")
        if not sql_result:
            return {**state, "trend_result": {}, "trend_summary": ""}

        # ── 1. Pull subject_id from first SQL row 
        subject_id: Optional[int] = None
        try:
            rows = json.loads(sql_result).get("rows", [])
            if rows:
                subject_id = rows[0].get("subject_id")
                if subject_id is not None:
                    subject_id = int(subject_id)
        except Exception:
            pass

        if subject_id is None:
            return {**state, "trend_result": {}, "trend_summary": ""}

        # ── 2. Infer the primary lab name from the clinical question ──────
        # Common lab keywords — extend this list as needed.
        LAB_KEYWORDS = [
            "creatinine", "glucose", "sodium", "potassium", "hemoglobin",
            "hematocrit", "platelet", "wbc", "white blood cell", "bilirubin",
            "albumin", "lactate", "troponin", "inr", "pt", "ptt", "bicarbonate",
            "chloride", "magnesium", "phosphate", "calcium", "urea", "bun",
        ]
        question_lower = state["clinical_question"].lower()
        lab_name: Optional[str] = None
        for kw in LAB_KEYWORDS:
            if kw in question_lower:
                lab_name = kw
                break

        # Fallback: try to pull the label column from the first SQL row
        if lab_name is None and rows:
            raw_label = rows[0].get("label", "")
            if raw_label:
                # Use the first word of the label as a fuzzy search key
                lab_name = raw_label.split(",")[0].split(" ")[0].lower()

        if not lab_name:
            return {**state, "trend_result": {}, "trend_summary": ""}

        # ── 3. Run TrendAgent 
        try:
            trend_result = self.trend_agent.analyze_patient_lab(
                subject_id=subject_id,
                lab_name=lab_name,
            )
        except Exception as exc:
            trend_result = {
                "source": "trend_agent",
                "trend": "error",
                "summary": f"TrendAgent raised an exception: {exc}",
            }

        trend_summary = trend_result.get("summary", "")

        return {**state, "trend_result": trend_result, "trend_summary": trend_summary}

    # ── Agent 2 — Semantic Cross-Ref 
    def agent2_semantic_crossref(self, state: MIRAState) -> MIRAState:
        sql_context = state.get("sql_result", "") or "No patient data retrieved."

        response = self.llm.invoke([
            SystemMessage(content="Extract the SPECIFIC lab test names and their abnormal "
                                   "direction (high/low) from the patient data below. Write a "
                                   "semantic search query using those EXACT lab names — e.g. "
                                   "'low platelet count thrombocytopenia management' not a generic "
                                   "phrase. If multiple labs are abnormal, name the most critical one. "
                                   "Return ONLY the search query, nothing else."),
            HumanMessage(content=f"Question: {state['clinical_question']}\nPatient data: {sql_context[:1000]}")
        ])
        search_query = response.content.strip()

        guidelines_result = self.vector_search_tool.invoke({"query": search_query, "k": 3})

        # Relevance guard
        try:
            parsed = json.loads(guidelines_result)
            top_score = parsed.get("guidelines", [{}])[0].get("relevance_score", 0)
            if top_score < 0.3:
                parsed["low_relevance_warning"] = (
                    "Retrieved guidelines may not match this specific finding — "
                    "treat as general context only, do not present as directly applicable."
                )
                guidelines_result = json.dumps(parsed, default=str)
        except Exception:
            pass

        return {**state, "search_query_used": search_query, "guidelines": guidelines_result}

    # ── Agent 3 — Clinical Reasoning ─────────────────────────────────────
    def _build_guideline_text(self, guidelines_json: str) -> str:
        try:
            text = ""
            for g in json.loads(guidelines_json).get("guidelines", []):
                text += f"\n[{g['source']}] {g['topic']}:\n{g['text']}\n"
            return text
        except Exception:
            return guidelines_json

    def _build_trend_block(self, state: MIRAState) -> str:
        """
        Formats the TrendAgent output into a clearly labelled prompt block.
        Returns an empty string when no trend data is available so the prompt
        stays clean for generic questions that have no single-lab focus.
        """
        trend_summary = state.get("trend_summary", "")
        if not trend_summary:
            return ""

        trend_result = state.get("trend_result", {})
        trend_direction = trend_result.get("trend", "unknown")
        pct_change = trend_result.get("pct_change")
        slope = trend_result.get("slope_per_hour")
        crossed_high = trend_result.get("crossed_critical_high", False)
        crossed_low  = trend_result.get("crossed_critical_low", False)

        lines = ["\nLAB TRAJECTORY ANALYSIS (TrendAgent):"]
        lines.append(f"  Direction : {trend_direction.upper()}")
        if pct_change is not None:
            lines.append(f"  % change  : {pct_change:+.1f}%")
        if slope is not None:
            lines.append(f"  Slope     : {slope} units/hour")
        if crossed_high:
            lines.append("  ⚠ Crossed ABOVE upper reference bound during this period.")
        if crossed_low:
            lines.append("  ⚠ Crossed BELOW lower reference bound during this period.")
        lines.append(f"  Summary   : {trend_summary}")

        return "\n".join(lines)

    def agent3_clinical_reasoning(self, state: MIRAState) -> MIRAState:
        sql_result    = state.get("sql_result", "No patient data.")
        guideline_text = self._build_guideline_text(state.get("guidelines", "No guidelines."))
        trend_block   = self._build_trend_block(state)

        relevance_warning = ""
        try:
            if json.loads(state.get("guidelines", "{}")).get("low_relevance_warning"):
                relevance_warning = (
                    "\n\nNOTE: The retrieved guidelines may NOT directly match the lab findings "
                    "below. Do not force-fit them. State explicitly that no closely matching "
                    "guideline was found for this specific finding, and recommend clinical "
                    "correlation instead of citing an unrelated guideline as if it applies."
                )
        except Exception:
            pass

        feedback_context = ""
        if state.get("human_decision") == "reject" and state.get("human_feedback"):
            feedback_context = (
                f"\n\nIMPORTANT — A clinician reviewed your previous analysis and rejected it "
                f"with this feedback:\n{state['human_feedback']}\nRevise your analysis accordingly."
            )

        # Instruct Agent 3 to weave in trajectory data when present
        trend_instruction = ""
        if trend_block:
            trend_instruction = (
                "\n\nA Lab Trajectory Analysis is provided below as a THIRD data source. "
                "Integrate it into 'Identified Concerns' and 'Recommended Actions'. "
                "If the trajectory is WORSENING, highlight it as a priority clinical signal — "
                "a deteriorating trend carries more urgency than a single abnormal snapshot."
            )

        system_prompt = f"""You are an expert clinical AI assistant. Synthesize patient data + guidelines into:
## Patient Summary
## Identified Concerns
## Clinical Guideline Context
## Recommended Actions
Ground every claim in the data or a cited guideline. Never hallucinate values — only state
numbers, subject IDs, or lab results that appear verbatim in the patient data below.
If the patient data shows zero rows or an error after retries, say plainly that no patients
matched this specific threshold in the available dataset — do NOT claim the data type itself
is "not provided" if the schema includes that column; the dataset has lab values, the query
just didn't find a match this time.{relevance_warning}{feedback_context}{trend_instruction}"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Question: {state['clinical_question']}\n\n"
                f"PATIENT DATA:\n{sql_result[:2000]}\n\n"
                f"GUIDELINES:\n{guideline_text[:2000]}"
                + (f"\n{trend_block}" if trend_block else "")
            ))
        ])
        reasoning = response.content.strip()

        return {**state, "clinical_analysis": reasoning, "human_decision": "", "human_feedback": ""}

    def stream_clinical_reasoning(self, state: MIRAState) -> Generator[str, None, str]:
        """
        Generator version of Agent 3 for UI streaming.
        Yields tokens one at a time. Used by the Streamlit layer with st.write_stream.
        """
        sql_result    = state.get("sql_result", "No patient data.")
        guideline_text = self._build_guideline_text(state.get("guidelines", "No guidelines."))
        trend_block   = self._build_trend_block(state)

        feedback_context = ""
        if state.get("human_decision") == "reject" and state.get("human_feedback"):
            feedback_context = (
                f"\n\nIMPORTANT — A clinician rejected your previous analysis with this feedback:\n"
                f"{state['human_feedback']}\nRevise accordingly."
            )

        trend_instruction = ""
        if trend_block:
            trend_instruction = (
                "\n\nA Lab Trajectory Analysis is provided as a THIRD data source. "
                "Integrate it into 'Identified Concerns' and 'Recommended Actions'. "
                "If the trajectory is WORSENING, highlight it as a priority clinical signal."
            )

        system_prompt = f"""You are an expert clinical AI assistant. Synthesize patient data + guidelines into:
## Patient Summary
## Identified Concerns
## Clinical Guideline Context
## Recommended Actions
Ground every claim in the data or a cited guideline. Never hallucinate values.
If the patient data shows zero rows or an error, say plainly that no patients matched this
specific threshold — do NOT claim the data type itself is "not provided" if the schema includes
that column.{feedback_context}{trend_instruction}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Question: {state['clinical_question']}\n\n"
                f"PATIENT DATA:\n{sql_result[:2000]}\n\n"
                f"GUIDELINES:\n{guideline_text[:2000]}"
                + (f"\n{trend_block}" if trend_block else "")
            ))
        ]

        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

    # ── Human Review Node (pause point) ──────────────────────────────────
    def human_review_node(self, state: MIRAState) -> MIRAState:
        """No-op node. Pausing happens via interrupt_before at compile time."""
        return state

    def route_after_human_review(self, state: MIRAState) -> str:
        if state.get("human_decision") == "reject":
            return "revise"
        return "proceed"

    # ── Agent 4 — Critic & Safety ─────────────────────────────────────────
    def agent4_critic_safety(self, state: MIRAState) -> MIRAState:
        reasoning  = state.get("clinical_analysis", "")
        sql_result = state.get("sql_result", "")
        guidelines = state.get("guidelines", "")

        system_prompt = """You are a medical AI safety critic AND editor.

Check the analysis for: hallucinated values not present in the patient data,
guideline mismatch (guidelines retrieved don't match the actual lab findings),
missing recommendations, and dangerous omissions.

CRITICAL — "final_report" must ALWAYS be a complete, corrected, patient-facing
clinical report using the same format as the original (Patient Summary /
Identified Concerns / Clinical Guideline Context / Recommended Actions).
NEVER put your critique, complaint list, or meta-commentary about what's wrong
into final_report — that field is read directly by the clinician, not by you
explaining yourself. If you find issues:
  - Remove any value not present in the actual patient data below.
  - If the guidelines don't match the findings, state plainly that no
    specific guideline was matched for this finding, instead of citing
    an unrelated one.
  - Add a concrete recommendation (e.g. "recommend repeat testing and
    clinical correlation") rather than leaving it vague.
  - If any finding is severely abnormal, explicitly state it warrants
    prompt clinical attention.
  - If a Lab Trajectory Analysis was provided and shows WORSENING, verify
    the final report reflects the urgency of a deteriorating trend, not
    just a point-in-time snapshot.

Put your reasoning about what you changed in "corrections" (for logs only,
clinician never sees this field). Respond ONLY in JSON:
{"approved": true/false, "safety_flags": [], "corrections": "...", "final_report": "..."}"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"PATIENT DATA (ground truth — only use values present here):\n{sql_result[:1500]}\n\n"
                f"GUIDELINES RETRIEVED:\n{guidelines[:1000]}\n\n"
                f"LAB TRAJECTORY SUMMARY:\n{state.get('trend_summary', 'Not available.')}\n\n"
                f"DRAFT ANALYSIS TO REVIEW AND CORRECT:\n{reasoning}"
            ))
        ])

        try:
            raw = response.content.strip().replace("```json", "").replace("```", "").strip()
            critic_output = json.loads(raw)
        except Exception:
            critic_output = {"approved": True, "safety_flags": [], "final_report": reasoning}

        final_report = critic_output.get("final_report", reasoning)

        # Safety net: if the model still returned a critique-shaped report, fall back.
        critique_markers = ["hallucinated value", "the analysis contains", "needs to be revised",
                             "the analysis does not", "issues that need addressing"]
        if any(marker in final_report.lower()[:300] for marker in critique_markers):
            final_report = reasoning

        return {
            **state,
            "final_report": final_report,
            "safety_flags": critic_output.get("safety_flags", []),
            "approved": critic_output.get("approved", True),
        }

    # ── Build & compile the LangGraph ─────────────────────────────────────
    def _build_graph(self):
        builder = StateGraph(MIRAState)

        builder.add_node("sql_extractor",      self.agent1_sql_extractor)
        builder.add_node("trend_analysis",     self.agent1b_trend_analysis)   # ← NEW
        builder.add_node("semantic_crossref",  self.agent2_semantic_crossref)
        builder.add_node("clinical_reasoning", self.agent3_clinical_reasoning)
        builder.add_node("human_review",       self.human_review_node)
        builder.add_node("critic_safety",      self.agent4_critic_safety)

        builder.set_entry_point("sql_extractor")

        builder.add_conditional_edges(
            "sql_extractor", self.should_retry_sql,
            {"retry": "sql_extractor", "ok": "trend_analysis"}      # ← was "semantic_crossref"
        )
        builder.add_edge("trend_analysis",     "semantic_crossref")  # ← NEW edge
        builder.add_edge("semantic_crossref",  "clinical_reasoning")
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

    # ── Public API for the UI layer ───────────────────────────────────────
    def new_thread(self) -> dict:
        """Returns a fresh LangGraph config with a unique thread_id."""
        return {"configurable": {"thread_id": str(uuid.uuid4())}}

    def run_until_review(self, clinical_question: str, config: dict) -> MIRAState:
        """Runs Agents 1 → 1b → 2 → 3, pauses before human_review. Returns the paused state."""
        initial_state = make_initial_state(clinical_question)
        return self.graph.invoke(initial_state, config)

    def submit_human_decision(self, config: dict, decision: str, feedback: str = "") -> MIRAState:
        """
        Submits the doctor's decision and resumes the graph.
        decision: "approve" or "reject"
        Returns the final state.
        """
        self.graph.update_state(config, {"human_decision": decision, "human_feedback": feedback})
        return self.graph.invoke(None, config)

    def get_current_state(self, config: dict) -> MIRAState:
        """Fetch the latest checkpointed state for a thread (for UI refresh)."""
        snapshot = self.graph.get_state(config)
        return snapshot.values

    def get_trend_result(self, config: dict) -> dict:
        """
        Convenience helper for the UI: returns the TrendAgent result dict
        from the current thread's checkpointed state.

        Usage in streamlit_app.py:
            trend = engine.get_trend_result(st.session_state.thread_config)
            if trend.get("trend") == "worsening":
                st.warning(trend["summary"])
        """
        state = self.get_current_state(config)
        return state.get("trend_result", {})

    def run_trend_for_patient(self, subject_id: int, lab_name: str) -> dict:
        """
        Direct TrendAgent call — bypasses the full pipeline.
        Useful for the UI to run ad-hoc trajectory lookups.

        Usage in streamlit_app.py:
            result = engine.run_trend_for_patient(10027602, "creatinine")
            st.write(result["summary"])
        """
        return self.trend_agent.analyze_patient_lab(subject_id=subject_id, lab_name=lab_name)

    def find_worsening_patients(self, lab_name: str, min_readings: int = 2, limit: int = 10) -> dict:
        """
        Exposes TrendAgent.find_worsening_patients() at the engine level.
        Lets the UI run a "who is getting worse?" query without touching TrendAgent directly.
        """
        return self.trend_agent.find_worsening_patients(
            lab_name=lab_name, min_readings=min_readings, limit=limit
        )



# Module-level singleton helper (optional convenience for simple scripts)


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
    question = "Which patients have abnormal creatinine results? Summarize the key concerns."
    print(f"🏥 Running MIRA for: {question}\n")
    paused_state = engine.run_until_review(question, cfg)

    print("📈 Trend Analysis Result:")
    print(paused_state.get("trend_summary") or "  (no trend data — insufficient readings or lab not detected)")

    print("\n🛑 Paused for human review. Agent 3's draft:\n")
    print(paused_state["clinical_analysis"])

    print("\n▶️  Auto-approving for smoke test...")
    final_state = engine.submit_human_decision(cfg, "approve")
    print("\n✅ FINAL REPORT:\n")
    print(final_state["final_report"])