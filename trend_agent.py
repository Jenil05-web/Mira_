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
  critical threshold during the trend — independent of any ML framework.
  Pure pandas/numpy — no GPU, no extra heavy dependency, fits tabular
  labevents data exactly as it exists in MIMIC-IV.

ISOLATED FROM mira_pipeline.py — does not run unless you explicitly wire
it into Agent 1's output or call it directly.

HOW TO WIRE IT IN (your decision, not automatic):
    from trend_agent import TrendAgent
    trend_agent = TrendAgent(conn)  # same sqlite3 connection as your engine
    trend_result = trend_agent.analyze_patient_lab(subject_id=10027602, lab_name="creatinine")
    # then pass trend_result["summary"] into Agent 3's prompt as a third
    # data source, alongside sql_result and guideline_text
"""

import json
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd


class TrendAgent:
    """
    Detects directional trends in a patient's repeated lab measurements
    over time. Pure statistical — no ML model required.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def analyze_patient_lab(self, subject_id: int, lab_name: str) -> dict:
        """
        Pulls all historical readings for a given lab + patient, computes
        trend direction, slope, and whether it crossed reference bounds.
        """
        query = """
            SELECT l.charttime, l.valuenum, l.valueuom,
                   l.ref_range_lower, l.ref_range_upper, d.label
            FROM labevents l
            JOIN d_labitems d ON l.itemid = d.itemid
            WHERE l.subject_id = ?
              AND d.label LIKE ?
              AND l.valuenum IS NOT NULL
            ORDER BY l.charttime ASC
        """
        df = pd.read_sql_query(query, self.conn, params=(subject_id, f"%{lab_name}%"))

        if df.empty:
            return {
                "source": "trend_agent",
                "subject_id": subject_id,
                "lab_name": lab_name,
                "trend": "insufficient_data",
                "summary": f"No historical readings found for {lab_name} for this patient."
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
                    f"Cannot establish a trend from a single data point."
                )
            }

        return self._compute_trend(df, lab_name, subject_id)

    def _compute_trend(self, df: pd.DataFrame, lab_name: str, subject_id: Optional[int] = None) -> dict:
        values = df["valuenum"].values
        times = pd.to_datetime(df["charttime"])

        hours_elapsed = (times - times.iloc[0]).dt.total_seconds() / 3600
        slope = np.polyfit(hours_elapsed, values, 1)[0] if len(values) >= 2 else 0

        first_val, last_val = values[0], values[-1]
        pct_change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0

        ref_upper = df["ref_range_upper"].dropna().iloc[-1] if df["ref_range_upper"].notna().any() else None
        ref_lower = df["ref_range_lower"].dropna().iloc[-1] if df["ref_range_lower"].notna().any() else None

        crossed_critical_high = ref_upper is not None and last_val > ref_upper and first_val <= ref_upper
        crossed_critical_low  = ref_lower is not None and last_val < ref_lower and first_val >= ref_lower

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
            "summary": " ".join(summary_parts)
        }

    def analyze_as_tool_output(self, subject_id: int, lab_name: str) -> str:
        """JSON string version — drop-in alongside sql_query/vector_search outputs."""
        return json.dumps(self.analyze_patient_lab(subject_id, lab_name), default=str)

    def find_worsening_patients(self, lab_name: str, min_readings: int = 2, limit: int = 10) -> dict:
        """
        Scans all patients with repeated readings of a given lab and returns
        those showing a worsening trajectory — useful for "who is getting
        worse" style triage questions rather than single-patient lookup.
        """
        query = """
            SELECT l.subject_id, l.charttime, l.valuenum, l.valueuom,
                   l.ref_range_lower, l.ref_range_upper, d.label
            FROM labevents l
            JOIN d_labitems d ON l.itemid = d.itemid
            WHERE d.label LIKE ? AND l.valuenum IS NOT NULL
            ORDER BY l.subject_id, l.charttime ASC
        """
        df = pd.read_sql_query(query, self.conn, params=(f"%{lab_name}%",))
        if df.empty:
            return {"source": "trend_agent", "lab_name": lab_name, "worsening_patients": []}

        results = []
        for subject_id, group in df.groupby("subject_id"):
            if len(group) < min_readings:
                continue
            trend_result = self._compute_trend(group.reset_index(drop=True), lab_name, int(subject_id))
            if trend_result["trend"] == "worsening":
                results.append(trend_result)

        results.sort(key=lambda r: abs(r["pct_change"]), reverse=True)
        return {
            "source": "trend_agent",
            "lab_name": lab_name,
            "worsening_patients": results[:limit],
            "total_worsening_found": len(results)
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    db_path = Path("./mira_data/mimic.db")
    if not db_path.exists():
        print(f"DB not found at {db_path}. Run 01_data_setup.ipynb first.")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    agent = TrendAgent(conn)

    lab = sys.argv[1] if len(sys.argv) > 1 else "creatinine"
    print(f"Scanning for worsening '{lab}' trends across all patients...\n")
    result = agent.find_worsening_patients(lab)
    print(json.dumps(result, indent=2, default=str)[:3000])