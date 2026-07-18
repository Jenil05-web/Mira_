"""
db_adapter.py
==============
MIRA Production — Database Adapter

WHAT THIS SOLVES:
  Some hospitals (smaller clinics, research institutions) don't yet have a
  live FHIR endpoint but have a direct database (PostgreSQL, MySQL, SQL Server,
  SQLite). This adapter auto-discovers their schema, maps it to MIRA's
  expected concepts, and generates the schema string for Agent 1.

  It's the fallback when FHIR isn't available. In production:
    - Large hospital with Epic/Cerner → use FHIRAdapter
    - Hospital with direct DB access → use DBAdapter
    - Dev/research (MIMIC-IV) → use DBAdapter with SQLite

  The DBAdapter also powers MIMIC-IV in your current dev setup,
  so migrating your existing notebooks to this is zero-effort.

SUPPORTED DATABASES (all free tiers available):
  SQLite      → sqlite:///path/to/db.sqlite3  (dev/MIMIC-IV)
  PostgreSQL  → postgresql://user:pass@host:5432/dbname  (Supabase free tier)
  MySQL       → mysql+pymysql://user:pass@host:3306/dbname
  SQL Server  → mssql+pyodbc://user:pass@host/dbname

INSTALL:
  pip install sqlalchemy
  pip install psycopg2-binary  # for PostgreSQL / Supabase
  pip install pymysql          # for MySQL (optional)

Supabase free tier: https://supabase.com — 500MB PostgreSQL, no credit card.
"""

import json
import logging
import re
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, inspect, text

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# CONCEPT MAP — maps hospital column names → MIRA's expected concepts
# ══════════════════════════════════════════════════════════════════════════

# Hospitals use wildly different column names for the same concept.
# This map lets MIRA auto-detect the right column by fuzzy matching.
# Format: concept → list of possible column names (case-insensitive)

CONCEPT_MAP = {
    "patient_id":    ["subject_id", "patient_id", "pat_id", "mrn", "pid",
                      "patientid", "patient_mrn", "member_id"],
    "gender":        ["gender", "sex", "patient_sex", "pat_sex"],
    "age":           ["age", "anchor_age", "patient_age", "age_at_admission"],
    "admit_time":    ["admittime", "admit_time", "admission_date", "admitted_dt",
                      "encounter_start", "visit_start"],
    "discharge_time":["dischtime", "discharge_time", "discharge_date",
                      "encounter_end", "visit_end"],
    "diagnosis":     ["diagnosis", "admitting_diagnosis", "primary_dx",
                      "diagnosis_text", "chief_complaint"],
    "icd_code":      ["icd_code", "icd9_code", "icd10_code", "dx_code",
                      "diagnosis_code", "code"],
    "lab_item_id":   ["itemid", "item_id", "lab_item_id", "test_id", "order_id"],
    "lab_name":      ["label", "lab_name", "test_name", "item_name",
                      "observation_name", "analyte"],
    "lab_value":     ["value", "result", "result_value", "lab_value",
                      "test_result", "valuestring"],
    "lab_valuenum":  ["valuenum", "value_num", "numeric_result", "result_numeric",
                      "valuenumeric"],
    "lab_unit":      ["valueuom", "unit", "uom", "units", "result_unit",
                      "value_unit"],
    "lab_flag":      ["flag", "abnormal_flag", "result_flag", "interpretation",
                      "critical_flag"],
    "ref_lower":     ["ref_range_lower", "reference_low", "ref_low",
                      "normal_low", "lower_limit"],
    "ref_upper":     ["ref_range_upper", "reference_high", "ref_high",
                      "normal_high", "upper_limit"],
    "chart_time":    ["charttime", "chart_time", "result_time", "observation_time",
                      "collected_dt", "reported_dt"],
}

# MIRA's core tables — concept-level names (not actual table names)
CORE_TABLE_CONCEPTS = {
    "patients":   ["patients", "patient", "pat", "members", "persons", "demographics"],
    "admissions": ["admissions", "admission", "encounters", "encounter",
                   "visits", "hospital_stays"],
    "labevents":  ["labevents", "lab_events", "lab_results", "laboratory",
                   "observations", "results", "test_results"],
    "lab_dict":   ["d_labitems", "lab_items", "lab_dictionary", "test_catalog",
                   "lab_master", "item_master"],
    "diagnoses":  ["diagnoses_icd", "diagnoses", "diagnosis", "dx", "icd_codes"],
}


# ══════════════════════════════════════════════════════════════════════════
# DB ADAPTER
# ══════════════════════════════════════════════════════════════════════════

class DBAdapter:
    """
    Schema-agnostic database adapter. Connects to any SQLAlchemy-supported
    database, auto-discovers the schema, maps it to MIRA concepts, and
    exposes the same query interface as FHIRAdapter.
    """

    def __init__(self, connection_string: str):
        """
        connection_string examples:
          SQLite    : sqlite:///./mira_data/mimic.db
          Supabase  : postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres
          PostgreSQL: postgresql://user:pass@localhost:5432/hospital_db
          MySQL     : mysql+pymysql://user:pass@host:3306/db
        """
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)
        self._schema_cache: Optional[dict] = None
        self._table_map: Optional[dict] = None

    # ── Schema discovery ────────────────────────────────────────────────
    def discover_schema(self, force_refresh: bool = False) -> dict:
        """
        Introspects the database schema and returns:
          - all table names
          - columns per table (with types and sample values)
          - mapped MIRA concept → actual table/column name
        """
        if self._schema_cache and not force_refresh:
            return self._schema_cache

        inspector = inspect(self.engine)
        tables = {}

        for table_name in inspector.get_table_names():
            columns = []
            for col in inspector.get_columns(table_name):
                columns.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                })
            # Sample one row for context
            sample = self._sample_row(table_name)
            tables[table_name] = {"columns": columns, "sample_row": sample}

        self._schema_cache = tables
        self._table_map = self._map_tables_to_concepts(tables)
        return tables

    def _sample_row(self, table_name: str) -> Optional[dict]:
        try:
            with self.engine.connect() as conn:
                row = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 1")).fetchone()
                if row:
                    return dict(row._mapping)
        except Exception:
            pass
        return None

    def _map_tables_to_concepts(self, tables: dict) -> dict:
        """
        Maps each MIRA concept (patients, labevents, etc.) to the actual
        table name and column names in this hospital's database.
        """
        table_map = {}
        table_names_lower = {t.lower(): t for t in tables}

        for concept, aliases in CORE_TABLE_CONCEPTS.items():
            for alias in aliases:
                if alias.lower() in table_names_lower:
                    real_name = table_names_lower[alias.lower()]
                    cols = {c["name"].lower(): c["name"]
                            for c in tables[real_name]["columns"]}
                    table_map[concept] = {
                        "table": real_name,
                        "columns": self._map_columns_to_concepts(cols),
                    }
                    break

        return table_map

    def _map_columns_to_concepts(self, cols_lower: dict) -> dict:
        """Maps MIRA column concepts to actual column names in this table."""
        mapped = {}
        for concept, aliases in CONCEPT_MAP.items():
            for alias in aliases:
                if alias.lower() in cols_lower:
                    mapped[concept] = cols_lower[alias.lower()]
                    break
        return mapped

    # ── Query execution ──────────────────────────────────────────────────
    # ── SQL Safety Gate ──────────────────────────────────────────────────
    _BLOCKED_KEYWORDS = re.compile(
        r"\b(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER|TRUNCATE|REPLACE|MERGE|"
        r"EXEC|EXECUTE|GRANT|REVOKE|CALL)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _validate_select_only(sql: str) -> str | None:
        """
        Returns an error string if the SQL is unsafe, else None.
        Rules:
          1. Must be a single statement (no semicolons except a trailing one).
          2. Must start with SELECT (after stripping comments/whitespace).
          3. Must not contain any write/DDL keyword anywhere in the statement.
        """
        # Strip leading/trailing whitespace and an optional trailing semicolon
        cleaned = sql.strip().rstrip(";").strip()

        # Rule 1 — no multiple statements
        if ";" in cleaned:
            return "Rejected: multiple statements are not allowed."

        # Rule 2 — must start with SELECT
        first_word = cleaned.split()[0].upper() if cleaned.split() else ""
        if first_word != "SELECT":
            return f"Rejected: only SELECT statements are allowed (got '{first_word}')."

        # Rule 3 — no destructive keywords anywhere in the body
        match = DBAdapter._BLOCKED_KEYWORDS.search(cleaned)
        if match:
            return f"Rejected: forbidden keyword '{match.group()}' found in query."

        return None  # safe

    def run_query(self, sql: str) -> str:
        """
        Execute raw SQL — used by Agent 1 the same way as the original
        sql_query tool. Returns JSON string matching MIRA's expected shape.
        Applies a safety gate: only single SELECT statements are permitted.
        """
        safety_error = self._validate_select_only(sql)
        if safety_error:
            return json.dumps({
                "error": safety_error,
                "hint": "Only read-only SELECT queries are permitted.",
            })

        try:
            df = pd.read_sql_query(sql, self.engine)
            if df.empty:
                return json.dumps({"result": "No data found."})
            return json.dumps(
                {"rows": df.head(25).to_dict(orient="records"),
                 "total_returned": len(df)},
                default=str
            )
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "hint": "Check table/column names against the schema.",
            })

    # ── Schema string for Agent 1 prompt ────────────────────────────────
    def get_schema_description(self) -> str:
        """
        Returns a schema string for Agent 1's system prompt — in the same
        format as db_schema.txt but auto-generated from the live database.
        This means it always reflects the actual schema regardless of
        which hospital's database is connected.
        """
        schema = self.discover_schema()
        lines = [f"DATABASE SCHEMA ({self._detect_db_type()}):\n"]

        for table_name, info in schema.items():
            lines.append(f"TABLE: {table_name}")
            for col in info["columns"]:
                lines.append(f"  - {col['name']} ({col['type']})")
            if info.get("sample_row"):
                # Only show first 4 fields of sample row to keep prompt concise
                sample_preview = dict(list(info["sample_row"].items())[:4])
                lines.append(f"  Sample: {sample_preview}")
            lines.append("")

        if self._table_map:
            lines.append("\nMIRA CONCEPT MAPPING (auto-detected):")
            for concept, info in self._table_map.items():
                col_map = info.get("columns", {})
                lines.append(f"  {concept} → table '{info['table']}'")
                for c_concept, c_col in col_map.items():
                    lines.append(f"    {c_concept} → column '{c_col}'")
            lines.append("")

        return "\n".join(lines)

    def _detect_db_type(self) -> str:
        dialect = self.engine.dialect.name
        return {"sqlite": "SQLite", "postgresql": "PostgreSQL",
                "mysql": "MySQL", "mssql": "SQL Server"}.get(dialect, dialect)

    # ── Convenience: mapped queries ──────────────────────────────────────
    def get_abnormal_labs_query(self) -> str:
        """
        Auto-generates the 'find abnormal labs' SQL for this hospital's
        specific schema — so Agent 1 gets a correct starting query rather
        than guessing column names from scratch.
        """
        if not self._table_map:
            self.discover_schema()

        lab_info = self._table_map.get("labevents", {})
        dict_info = self._table_map.get("lab_dict", {})
        patient_info = self._table_map.get("patients", {})

        if not lab_info:
            return "-- Could not auto-detect lab events table"

        lab_table = lab_info["table"]
        lab_cols = lab_info.get("columns", {})
        patient_table = patient_info.get("table", "patients") if patient_info else "patients"
        patient_cols = patient_info.get("columns", {}) if patient_info else {}

        subject_col = lab_cols.get("patient_id", "subject_id")
        value_col = lab_cols.get("lab_valuenum", "valuenum")
        ref_upper_col = lab_cols.get("ref_upper", "ref_range_upper")
        ref_lower_col = lab_cols.get("ref_lower", "ref_range_lower")
        flag_col = lab_cols.get("lab_flag", "flag")
        time_col = lab_cols.get("chart_time", "charttime")

        if dict_info:
            dict_table = dict_info["table"]
            dict_cols = dict_info.get("columns", {})
            item_id_col = lab_cols.get("lab_item_id", "itemid")
            dict_item_id = list(dict_cols.values())[0] if dict_cols else "itemid"
            label_col = dict_cols.get("lab_name", "label")
            name_select = f"d.{label_col} AS lab_name"
            join_clause = f"JOIN {dict_table} d ON l.{item_id_col} = d.{dict_item_id}"
        else:
            name_select = f"'unknown' AS lab_name"
            join_clause = ""

        p_subject = patient_cols.get("patient_id", "subject_id") if patient_cols else "subject_id"
        gender_col = patient_cols.get("gender", "gender") if patient_cols else "gender"
        age_col = patient_cols.get("age", "anchor_age") if patient_cols else "anchor_age"

        return f"""SELECT
    p.{p_subject} AS subject_id,
    p.{gender_col} AS gender,
    p.{age_col} AS age,
    {name_select},
    l.{value_col} AS valuenum,
    l.{ref_lower_col} AS ref_range_lower,
    l.{ref_upper_col} AS ref_range_upper,
    l.{flag_col} AS flag,
    l.{time_col} AS charttime
FROM {lab_table} l
JOIN {patient_table} p ON l.{subject_col} = p.{p_subject}
{join_clause}
WHERE (
    ({ref_upper_col} IS NOT NULL AND {value_col} > {ref_upper_col} * 1.2)
    OR ({ref_lower_col} IS NOT NULL AND {value_col} < {ref_lower_col} * 0.8)
    OR {flag_col} IS NOT NULL
)
AND {value_col} IS NOT NULL
ORDER BY {time_col} DESC
LIMIT 25"""

    # ── Canonical-format interface (used by mira_pipeline_prod.py) ────────
    # These methods let the pipeline call any adapter — DB or FHIR —
    # through the same API and always get {"rows": [...]} back in one
    # canonical shape: subject_id, gender, age, lab_name, valuenum,
    # valueuom, ref_range_lower, ref_range_upper, flag, charttime.

    def check_patients_exist(self, patient_ids: list) -> tuple[list, list]:
        """Schema-agnostic patient existence check.  Returns (found, missing)."""
        if not self._table_map:
            self.discover_schema()

        patient_info = self._table_map.get("patients", {})
        if not patient_info:
            return [], list(patient_ids)

        table = patient_info["table"]
        id_col = patient_info.get("columns", {}).get("patient_id", "subject_id")

        found, missing = [], []
        for pid in patient_ids:
            try:
                with self.engine.connect() as conn:
                    row = conn.execute(
                        text(f"SELECT {id_col} FROM {table} WHERE {id_col} = :pid LIMIT 1"),
                        {"pid": pid},
                    ).fetchone()
                    (found if row else missing).append(pid)
            except Exception:
                missing.append(pid)
        return found, missing

    def _canonical_select_parts(self) -> tuple:
        """Internal helper — returns (select_clause, from_join_clause)
        built entirely from the concept map."""
        if not self._table_map:
            self.discover_schema()

        lab  = self._table_map.get("labevents", {})
        pat  = self._table_map.get("patients", {})
        dct  = self._table_map.get("lab_dict", {})
        if not lab or not pat:
            return None, None

        lt, lc = lab["table"], lab.get("columns", {})
        pt, pc = pat["table"], pat.get("columns", {})

        l_pid  = lc.get("patient_id", "subject_id")
        l_val  = lc.get("lab_valuenum", "valuenum")
        l_uom  = lc.get("lab_unit", "valueuom")
        l_rlo  = lc.get("ref_lower", "ref_range_lower")
        l_rhi  = lc.get("ref_upper", "ref_range_upper")
        l_flag = lc.get("lab_flag", "flag")
        l_time = lc.get("chart_time", "charttime")
        p_pid  = pc.get("patient_id", "subject_id")
        p_gen  = pc.get("gender", "gender")
        p_age  = pc.get("age", "anchor_age")

        if dct:
            dt, dc = dct["table"], dct.get("columns", {})
            l_item = lc.get("lab_item_id", "itemid")
            d_item = dc.get("lab_item_id", list(dc.values())[0] if dc else "itemid")
            d_label = dc.get("lab_name", "label")
            name_sel  = f"d.{d_label} AS lab_name"
            dict_join = f"JOIN {dt} d ON l.{l_item} = d.{d_item}"
        else:
            l_name = lc.get("lab_name")
            name_sel  = f"l.{l_name} AS lab_name" if l_name else "'unknown' AS lab_name"
            dict_join = ""

        select = (f"p.{p_pid} AS subject_id, p.{p_gen} AS gender, p.{p_age} AS age, "
                  f"{name_sel}, l.{l_val} AS valuenum, l.{l_uom} AS valueuom, "
                  f"l.{l_rlo} AS ref_range_lower, l.{l_rhi} AS ref_range_upper, "
                  f"l.{l_flag} AS flag, l.{l_time} AS charttime")

        from_join = (f"FROM {lt} l\n"
                     f"JOIN {pt} p ON l.{l_pid} = p.{p_pid}\n"
                     f"{dict_join}")

        return select, from_join

    def get_patient_labs(self, patient_ids: list, limit: int = 200) -> str:
        """Schema-agnostic lab query for specific patients.
        Returns canonical JSON ``{"rows": [...], "total_returned": N}``."""
        sel, fj = self._canonical_select_parts()
        if sel is None:
            return json.dumps({"rows": [], "error": "Could not map lab/patient tables"})

        lc  = self._table_map["labevents"].get("columns", {})
        pc  = self._table_map["patients"].get("columns", {})
        l_val  = lc.get("lab_valuenum", "valuenum")
        l_time = lc.get("chart_time", "charttime")
        p_pid  = pc.get("patient_id", "subject_id")

        ids_csv = ",".join(str(int(p)) for p in patient_ids)
        sql = (f"SELECT {sel}\n{fj}\n"
               f"WHERE p.{p_pid} IN ({ids_csv})\n"
               f"  AND l.{l_val} IS NOT NULL\n"
               f"ORDER BY p.{p_pid}, l.{l_time} DESC\n"
               f"LIMIT {int(limit)}")
        return self.run_query(sql)

    def get_broad_abnormal_labs(self, limit: int = 60) -> str:
        """Schema-agnostic broad abnormal-labs query.  Returns canonical JSON."""
        sql = self.get_abnormal_labs_query()
        if sql.startswith("--"):
            return json.dumps({"rows": [], "error": sql})
        sql = re.sub(r"LIMIT\s+\d+", f"LIMIT {int(limit)}", sql, flags=re.IGNORECASE)
        return self.run_query(sql)

    def get_sql_prompt_instructions(self) -> str:
        """Generate dynamic LLM-SQL prompt instructions from the concept map
        so the LLM writes correct SQL for *this* hospital's schema."""
        if not self._table_map:
            self.discover_schema()

        lab  = self._table_map.get("labevents", {})
        pat  = self._table_map.get("patients", {})
        dct  = self._table_map.get("lab_dict", {})
        if not lab or not pat:
            return ""

        lt, lc = lab["table"], lab.get("columns", {})
        pt, pc = pat["table"], pat.get("columns", {})

        l_pid  = lc.get("patient_id", "subject_id")
        l_val  = lc.get("lab_valuenum", "valuenum")
        l_uom  = lc.get("lab_unit", "valueuom")
        l_rlo  = lc.get("ref_lower", "ref_range_lower")
        l_rhi  = lc.get("ref_upper", "ref_range_upper")
        l_flag = lc.get("lab_flag", "flag")
        l_time = lc.get("chart_time", "charttime")
        p_pid  = pc.get("patient_id", "subject_id")
        p_gen  = pc.get("gender", "gender")
        p_age  = pc.get("age", "anchor_age")

        lines = [f"- Use table aliases: {lt} l, {pt} p"]

        sel_cols = (f"p.{p_pid} AS subject_id, p.{p_gen} AS gender, p.{p_age} AS age, ")

        if dct:
            dt, dc = dct["table"], dct.get("columns", {})
            l_item  = lc.get("lab_item_id", "itemid")
            d_item  = dc.get("lab_item_id", list(dc.values())[0] if dc else "itemid")
            d_label = dc.get("lab_name", "label")
            lines[0] += f", {dt} d"
            lines.append(f"- Always JOIN {dt} d ON l.{l_item} = d.{d_item}")
            lines.append(f"- Always JOIN {pt} p ON l.{l_pid} = p.{p_pid}")
            sel_cols += f"d.{d_label} AS lab_name, "
            lines.append(f"- For lab names use LIKE e.g. d.{d_label} ILIKE '%creatinine%'")
        else:
            l_name = lc.get("lab_name")
            lines.append(f"- Always JOIN {pt} p ON l.{l_pid} = p.{p_pid}")
            if l_name:
                sel_cols += f"l.{l_name} AS lab_name, "

        sel_cols += (f"l.{l_val} AS valuenum, l.{l_uom} AS valueuom, "
                     f"l.{l_rlo} AS ref_range_lower, l.{l_rhi} AS ref_range_upper, "
                     f"l.{l_flag} AS flag, l.{l_time} AS charttime")

        lines.append(f"- ALWAYS include in SELECT: {sel_cols}")
        lines.append(f"- WHERE l.{l_val} IS NOT NULL always")
        return "\n".join(lines)

    def results_to_sql_format(self, rows: list) -> str:
        """Wrap rows in canonical MIRA JSON.  Matches FHIRAdapter.results_to_sql_format()."""
        return json.dumps({"rows": rows, "total_returned": len(rows)}, default=str)


# ══════════════════════════════════════════════════════════════════════════
# ADAPTER FACTORY — picks FHIR or DB based on connection config
# ══════════════════════════════════════════════════════════════════════════

def create_adapter(config: dict):
    """
    Factory function. Config dict from ConfigManager determines which adapter:
      {"type": "fhir", "base_url": "...", "auth_mode": "open"}
      {"type": "db",   "connection_string": "sqlite:///..."}

    Usage in mira_pipeline.py:
        adapter = create_adapter(config_manager.get_data_source())
        schema_str = adapter.get_schema_description()
    """
    from adapters.fhir import FHIRAdapter, FHIRAuthHandler

    adapter_type = config.get("type", "db")

    if adapter_type == "fhir":
        auth = FHIRAuthHandler(
            mode=config.get("auth_mode", "open"),
            token=config.get("token", ""),
            client_id=config.get("client_id", ""),
            client_secret=config.get("client_secret", ""),
            token_url=config.get("token_url", ""),
        )
        return FHIRAdapter(base_url=config["base_url"], auth=auth)

    elif adapter_type == "db":
        return DBAdapter(connection_string=config["connection_string"])

    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}. Use 'fhir' or 'db'.")


# ══════════════════════════════════════════════════════════════════════════
# CLI smoke test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    db_path = Path("./mira_data/mimic.db")
    if not db_path.exists():
        print(f"MIMIC DB not found at {db_path}. Run 01_data_setup.ipynb first.")
        sys.exit(1)

    print("Testing DBAdapter against MIMIC-IV SQLite...\n")
    adapter = DBAdapter(f"sqlite:///{db_path}")

    print("1. Discovered tables:")
    schema = adapter.discover_schema()
    for t, info in schema.items():
        cols = [c["name"] for c in info["columns"]]
        print(f"   {t}: {cols}")

    print("\n2. Concept mapping:")
    for concept, info in (adapter._table_map or {}).items():
        print(f"   {concept} → {info['table']}")

    print("\n3. Auto-generated abnormal labs query:")
    print(adapter.get_abnormal_labs_query())

    print("\n4. Execute that query:")
    result = adapter.run_query(adapter.get_abnormal_labs_query())
    rows = json.loads(result).get("rows", [])
    print(f"   Returned {len(rows)} rows")
    if rows:
        print(f"   First row: {list(rows[0].items())[:4]}")