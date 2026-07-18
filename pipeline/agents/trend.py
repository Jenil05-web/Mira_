"""
trend_agent.py
================
Optional 5th agent: Lab Trajectory Analysis.

WHY THIS EXISTS:
  Your current agents look at a single abnormal lab value at one point in
  time. They miss the more clinically meaningful signal: a value that's
  WORSENING over repeated tests (creatinine climbing 1.1 -> 1.8 -> 2.6 mg/dL
  over 3 days is a much stronger AKI signal than one high reading alone).

  This agent pulls a patient's full history for a given lab and detects
  directional trends, rate of change, and whether a value crossed a
  reference bound during the trend.

SCHEMA-AGNOSTIC:
  TrendAgent now accepts a DBAdapter instance and derives all table/column
  names from its concept map, so it works against any hospital's schema —
  not just MIMIC-IV's labevents/d_labitems layout.
"""

import json
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text


class TrendAgent:
    """
    Detects directional trends in a patient's repeated lab measurements.
    Fully schema-agnostic: uses the DBAdapter concept map for all names.
    """

    def __init__(self, adapter):
        """
        Args:
            adapter: a DBAdapter instance (has .engine and ._table_map).
        """
        self.adapter = adapter
        # Ensure schema is discovered
        if not adapter._table_map:
            adapter.discover_schema()
        self._map = adapter._table_map

    # ── Internal helpers ────────────────────────────────────────────────────

    def _schema_names(self):
        """Return (lab_table, lab_cols, dict_table, dict_cols, pat_pid_col)."""
        lab  = self._map.get("labevents", {})
        dct  = self._map.get("lab_dict", {})
        pat  = self._map.get("patients", {})

        lt   = lab.get("table", "labevents")
        lc   = lab.get("columns", {})
        dt   = dct.get("table", "d_labitems") if dct else None
        dc   = dct.get("columns", {}) if dct else {}
        pc   = pat.get("columns", {}) if pat else {}

        return lt, lc, dt, dc, pc

    def _read_df(self, sql: str, params: dict) -> pd.DataFrame:
        """Execute a SQL query via the adapter's SQLAlchemy engine."""
        with self.adapter.engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn, params=params)

    # ── Public API ──────────────────────────────────────────────────────────

    def analyze_patient_lab(self, subject_id: int, lab_name: str) -> dict:
        """
        Pulls all historical readings for a given lab + patient, computes
        trend direction, slope, and whether it crossed reference bounds.
        """
        lt, lc, dt, dc, pc = self._schema_names()

        l_pid   = lc.get("patient_id",    "subject_id")
        l_val   = lc.get("lab_valuenum",  "valuenum")
        l_uom   = lc.get("lab_unit",      "valueuom")
        l_rlo   = lc.get("ref_lower",     "ref_range_lower")
        l_rhi   = lc.get("ref_upper",     "ref_range_upper")
        l_time  = lc.get("chart_time",    "charttime")
        l_item  = lc.get("lab_item_id",   "itemid")

        if dt:
            d_item  = dc.get("lab_item_id", list(dc.values())[0] if dc else "itemid")
            d_label = dc.get("lab_name",  "label")
            join_clause = f"JOIN {dt} d ON l.{l_item} = d.{d_item}"
            name_col    = f"d.{d_label}"
            name_filter = f"AND {name_col} ILIKE :lab_pattern"
        else:
            join_clause = ""
            name_col    = "NULL"
            name_filter = ""

        sql = f"""
            SELECT l.{l_time}    AS charttime,
                   l.{l_val}     AS valuenum,
                   l.{l_uom}     AS valueuom,
                   l.{l_rlo}     AS ref_range_lower,
                   l.{l_rhi}     AS ref_range_upper,
                   {name_col}    AS label
            FROM {lt} l
            {join_clause}
            WHERE l.{l_pid} = :subject_id
              {name_filter}
              AND l.{l_val} IS NOT NULL
            ORDER BY l.{l_time} ASC
        """
        params = {"subject_id": subject_id, "lab_pattern": f"%{lab_name}%"}

        try:
            df = self._read_df(sql, params)
        except Exception as e:
            return {
                "source": "trend_agent",
                "subject_id": subject_id,
                "lab_name": lab_name,
                "trend": "insufficient_data",
                "summary": f"Query error: {e}",
            }

        if df.empty:
            return {
                "source": "trend_agent",
                "subject_id": subject_id,
                "lab_name": lab_name,
                "trend": "insufficient_data",
                "summary": f"No historical readings found for {lab_name} for patient {subject_id}.",
            }

        if len(df) < 2:
            single = df.iloc[0]
            return {
                "source": "trend_agent",
                "subject_id": subject_id,
                "lab_name": lab_name,
                "readings": df.to_dict(orient="records"),
                "trend": "insufficient_data",
                "summary": (
                    f"Only one {lab_name} reading on record "
                    f"({single['valuenum']} {single.get('valueuom', '')}). "
                    "Cannot establish a trend from a single data point."
                ),
            }

        return self._compute_trend(df, lab_name, subject_id)

    def _compute_trend(self, df: pd.DataFrame, lab_name: str,
                       subject_id: Optional[int] = None) -> dict:
        values = df["valuenum"].values
        times  = pd.to_datetime(df["charttime"])

        hours_elapsed = (times - times.iloc[0]).dt.total_seconds() / 3600
        slope = np.polyfit(hours_elapsed, values, 1)[0] if len(values) >= 2 else 0

        first_val, last_val = values[0], values[-1]
        pct_change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0

        ref_upper = df["ref_range_upper"].dropna().iloc[-1] if df["ref_range_upper"].notna().any() else None
        ref_lower = df["ref_range_lower"].dropna().iloc[-1] if df["ref_range_lower"].notna().any() else None

        crossed_critical_high = ref_upper is not None and last_val > ref_upper  and first_val <= ref_upper
        crossed_critical_low  = ref_lower is not None and last_val < ref_lower  and first_val >= ref_lower

        if abs(pct_change) < 5:
            trend = "stable"
        elif pct_change > 0:
            trend = "worsening" if (ref_upper is not None and last_val > ref_upper) else "rising"
        else:
            trend = "worsening" if (ref_lower is not None and last_val < ref_lower) else "falling"

        unit = df["valueuom"].dropna().iloc[-1] if df["valueuom"].notna().any() else ""

        summary_parts = [
            f"{lab_name.title()} trend across {len(df)} readings: "
            f"{first_val:.2f} -> {last_val:.2f} {unit} "
            f"({pct_change:+.1f}% change over {hours_elapsed.iloc[-1]:.0f} hours)."
        ]
        if crossed_critical_high:
            summary_parts.append(f"Crossed ABOVE the upper reference bound ({ref_upper}) during this period.")
        if crossed_critical_low:
            summary_parts.append(f"Crossed BELOW the lower reference bound ({ref_lower}) during this period.")
        if trend == "worsening":
            summary_parts.append("This trajectory indicates a WORSENING clinical trend, not just an isolated abnormal value.")
        elif trend == "stable":
            summary_parts.append("Values have remained relatively stable across readings.")

        return {
            "source": "trend_agent",
            "subject_id": subject_id,
            "lab_name": lab_name,
            "readings": df.to_dict(orient="records"),
            "trend": trend,
            "slope_per_hour": round(float(slope), 4),
            "pct_change": round(float(pct_change), 2),
            "crossed_critical_high": bool(crossed_critical_high),
            "crossed_critical_low": bool(crossed_critical_low),
            "summary": " ".join(summary_parts),
        }

    def analyze_as_tool_output(self, subject_id: int, lab_name: str) -> str:
        """JSON string version — drop-in alongside sql_query/vector_search outputs."""
        return json.dumps(self.analyze_patient_lab(subject_id, lab_name), default=str)

    def find_worsening_patients(self, lab_name: str, min_readings: int = 2,
                                limit: int = 10) -> dict:
        """
        Scans all patients with repeated readings of a given lab and returns
        those showing a worsening trajectory.
        """
        lt, lc, dt, dc, _ = self._schema_names()

        l_pid   = lc.get("patient_id",   "subject_id")
        l_val   = lc.get("lab_valuenum", "valuenum")
        l_uom   = lc.get("lab_unit",     "valueuom")
        l_rlo   = lc.get("ref_lower",    "ref_range_lower")
        l_rhi   = lc.get("ref_upper",    "ref_range_upper")
        l_time  = lc.get("chart_time",   "charttime")
        l_item  = lc.get("lab_item_id",  "itemid")

        if dt:
            d_item  = dc.get("lab_item_id", list(dc.values())[0] if dc else "itemid")
            d_label = dc.get("lab_name",  "label")
            join_clause = f"JOIN {dt} d ON l.{l_item} = d.{d_item}"
            name_col    = f"d.{d_label}"
            name_filter = f"AND {name_col} ILIKE :lab_pattern"
        else:
            join_clause = ""
            name_col    = "NULL"
            name_filter = ""

        sql = f"""
            SELECT l.{l_pid}  AS subject_id,
                   l.{l_time} AS charttime,
                   l.{l_val}  AS valuenum,
                   l.{l_uom}  AS valueuom,
                   l.{l_rlo}  AS ref_range_lower,
                   l.{l_rhi}  AS ref_range_upper,
                   {name_col} AS label
            FROM {lt} l
            {join_clause}
            WHERE l.{l_val} IS NOT NULL
              {name_filter}
            ORDER BY l.{l_pid}, l.{l_time} ASC
        """
        params = {"lab_pattern": f"%{lab_name}%"}

        try:
            df = self._read_df(sql, params)
        except Exception as e:
            return {"source": "trend_agent", "lab_name": lab_name,
                    "worsening_patients": [], "error": str(e)}

        if df.empty:
            return {"source": "trend_agent", "lab_name": lab_name, "worsening_patients": []}

        results = []
        for pid, group in df.groupby("subject_id"):
            if len(group) < min_readings:
                continue
            trend_result = self._compute_trend(group.reset_index(drop=True), lab_name, int(pid))
            if trend_result["trend"] == "worsening":
                results.append(trend_result)

        results.sort(key=lambda r: abs(r["pct_change"]), reverse=True)
        return {
            "source": "trend_agent",
            "lab_name": lab_name,
            "worsening_patients": results[:limit],
            "total_worsening_found": len(results),
        }