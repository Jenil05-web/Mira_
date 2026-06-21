"""
streamlit_app.py
=================
MIRA — Multi-Agent Clinical Audit & Real-Time Triage System
Frontend UI layer. All clinical logic lives in mira_pipeline.py.

Changes from previous version:
  - Sidebar removed; layout is now full-width two-column
  - Primary button text color fixed (white text on dark button)
  - Lab value charts rendered after diagnosis using st.altair_chart
  - Fill rail and processing strip animations stabilized for Streamlit
  - Trend analysis panel surfaced when available

Run with: streamlit run streamlit_app.py
"""

import json
import re
import time
import streamlit as st
import markdown as md_lib

from mira_pipeline import get_engine, Config


# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MIRA — Clinical Audit Console",
    page_icon="◍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS & CSS
# ══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --paper:      #FAFAF8;
    --surface:    #FFFFFF;
    --ink:        #1C2B33;
    --slate:      #5C7A89;
    --slate-2:    #8FA3AE;
    --teal:       #2D8C7F;
    --teal-tint:  #EAF4F2;
    --teal-deep:  #1F6358;
    --clay:       #C9501F;
    --clay-tint:  #FCEEE6;
    --line:       #E7E9E4;
    --line-soft:  #F0F1ED;
    --gold:       #B98A2E;
    --gold-tint:  #FBF3E4;
}

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}
.stApp { background: var(--paper); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1360px; }
#MainMenu, footer { visibility: hidden; }

/* hide collapsed sidebar toggle entirely */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── HEADER ── */
.mira-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 28px;
}
.mira-header-left { display: flex; align-items: center; gap: 14px; }
.mira-mark {
    width: 40px; height: 40px;
    border-radius: 10px;
    background: var(--ink);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Newsreader', serif;
    font-weight: 600; font-size: 19px; color: #FAFAF8;
    flex-shrink: 0;
    letter-spacing: -0.5px;
}
.mira-title {
    font-family: 'Newsreader', serif;
    font-size: 22px; font-weight: 500;
    color: var(--ink); letter-spacing: 0.1px; line-height: 1.1;
}
.mira-subtitle { font-size: 12.5px; color: var(--slate); margin-top: 2px; }

/* ── LIVE PILL ── */
.live-pill {
    display: flex; align-items: center; gap: 8px;
    background: var(--teal-tint);
    border: 1px solid rgba(45,140,127,0.2);
    border-radius: 100px;
    padding: 6px 13px 6px 10px;
}
.live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--teal); flex-shrink: 0;
    animation: pulse-dot 2.4s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { box-shadow: 0 0 0 0 rgba(45,140,127,0.5); }
    50%      { box-shadow: 0 0 0 5px rgba(45,140,127,0); }
}
.live-pill-text { font-size: 12px; font-weight: 500; color: var(--teal-deep); }

/* ── FILL RAIL ── */
.fill-rail-wrap { margin-bottom: 26px; }
.fill-rail-track {
    position: relative; height: 2px;
    background: var(--line); border-radius: 2px;
    overflow: hidden; margin-bottom: 13px;
}
.fill-rail-progress {
    position: absolute; top: 0; left: 0; height: 100%;
    background: linear-gradient(90deg, var(--teal-deep), var(--teal));
    border-radius: 2px;
    transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
}

/* animated shimmer on the active progress bar */
.fill-rail-progress.is-running::after {
    content: '';
    position: absolute; top: 0; right: -60px;
    width: 60px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
    animation: rail-shimmer 1.5s ease-in-out infinite;
}
@keyframes rail-shimmer {
    0%   { transform: translateX(-60px); }
    100% { transform: translateX(400px); }
}

.fill-rail-labels { display: flex; justify-content: space-between; }
.fill-rail-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--slate-2);
    display: flex; align-items: center; gap: 5px;
}
.fill-rail-label.is-done   { color: var(--slate); }
.fill-rail-label.is-active { color: var(--teal-deep); font-weight: 600; }
.fill-rail-label.is-wait   { color: var(--gold); font-weight: 600; }

.rail-node {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--line); flex-shrink: 0;
}
.rail-node.is-done   { background: var(--teal-deep); }
.rail-node.is-active {
    background: var(--teal);
    animation: node-breathe 1.8s ease-in-out infinite;
}
.rail-node.is-wait   { background: var(--gold); }
@keyframes node-breathe {
    0%,100% { box-shadow: 0 0 0 0 rgba(45,140,127,0.5); }
    50%      { box-shadow: 0 0 0 4px rgba(45,140,127,0); }
}

/* ── PROCESSING STRIP ── */
.processing-strip {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 15px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px; margin-top: 12px;
}
.processing-text { font-size: 12.5px; color: var(--slate); font-weight: 500; white-space: nowrap; }
.shimmer-bar {
    flex: 1; height: 2px;
    background: var(--line-soft);
    border-radius: 2px; overflow: hidden; position: relative;
}
.shimmer-bar::after {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 35%; height: 100%;
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
    animation: shimmer-pass 1.5s ease-in-out infinite;
}
@keyframes shimmer-pass {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(400%); }
}

/* ── PANELS ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
}
.panel-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--slate-2);
    margin-bottom: 12px; font-weight: 500;
    display: flex; align-items: center; gap: 6px;
}
.eyebrow-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--teal); }

/* ── TREND PANEL ── */
.trend-panel {
    background: var(--gold-tint);
    border: 1px solid rgba(185,138,46,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.trend-panel.worsening {
    background: var(--clay-tint);
    border-color: rgba(201,80,31,0.25);
}
.trend-panel.stable {
    background: var(--teal-tint);
    border-color: rgba(45,140,127,0.2);
}
.trend-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 0.07em;
    text-transform: uppercase; font-weight: 600;
    color: var(--gold); margin-bottom: 5px;
}
.trend-panel.worsening .trend-label { color: var(--clay); }
.trend-panel.stable .trend-label { color: var(--teal-deep); }
.trend-summary { font-size: 13px; color: var(--ink); line-height: 1.6; }

/* ── REPORT TYPOGRAPHY ── */
.report-surface h2 {
    font-family: 'Newsreader', serif;
    font-size: 16.5px; font-weight: 600; color: var(--ink);
    margin-top: 20px; margin-bottom: 8px;
    padding-top: 16px; border-top: 1px solid var(--line-soft);
}
.report-surface h2:first-child { margin-top: 0; padding-top: 0; border-top: none; }
.report-surface p, .report-surface li {
    font-size: 14px; line-height: 1.72; color: var(--ink);
}
.report-surface strong { color: var(--ink); font-weight: 600; }
.report-surface code {
    font-family: 'IBM Plex Mono', monospace;
    background: var(--line-soft); padding: 1px 5px;
    border-radius: 4px; font-size: 12.5px; color: var(--teal-deep);
}

/* ── BANNERS ── */
.banner {
    display: flex; align-items: center; gap: 9px;
    border-radius: 10px; padding: 12px 15px;
    margin-bottom: 14px; font-size: 13px; font-weight: 500;
}
.banner-approved {
    background: var(--teal-tint);
    border: 1px solid rgba(45,140,127,0.22);
    color: var(--teal-deep);
}
.banner-flagged {
    background: var(--clay-tint);
    border: 1px solid rgba(201,80,31,0.22);
    color: var(--clay);
}

/* ── MONO DATA BLOCKS ── */
.mono-block {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px; line-height: 1.55;
    background: var(--line-soft); border: 1px solid var(--line);
    border-radius: 8px; padding: 12px 14px;
    color: var(--slate); white-space: pre-wrap; overflow-x: auto;
}

/* ── BUTTONS ── */
/* secondary / default button */
.stButton > button {
    border-radius: 9px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 0.52rem 1.1rem !important;
    border: 1px solid var(--line) !important;
    background: var(--surface) !important;
    color: var(--ink) !important;
    transition: border-color 0.15s, background 0.15s !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    border-color: var(--slate-2) !important;
    background: var(--line-soft) !important;
}

/* primary button — override Streamlit's default which can let text go transparent */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"],
div[data-testid="stButton"] > button[kind="primary"] {
    background: var(--ink) !important;
    border: 1px solid var(--ink) !important;
    color: #FFFFFF !important;
}
.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span {
    color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #0F1A1F !important;
    border-color: #0F1A1F !important;
    color: #FFFFFF !important;
}

/* ── INPUTS ── */
.stTextArea textarea, .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    color: var(--ink) !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px var(--teal-tint) !important;
}
.stTextArea textarea::placeholder { color: var(--slate-2) !important; }

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    font-size: 12.5px !important;
    color: var(--slate) !important;
}

/* ── EXAMPLE CHIPS ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.chip {
    display: inline-flex; align-items: center;
    background: var(--line-soft); border: 1px solid var(--line);
    border-radius: 100px; padding: 5px 12px;
    font-size: 11.5px; color: var(--slate);
}

/* ── EMPTY STATE ── */
.empty-state { text-align: center; padding: 80px 30px; }
.empty-state .mark {
    width: 50px; height: 50px; border-radius: 13px;
    background: var(--line-soft);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 16px auto;
    font-family: 'Newsreader', serif;
    font-size: 21px; color: var(--slate-2);
}
.empty-state .heading {
    font-family: 'Newsreader', serif;
    font-size: 17px; font-weight: 500; color: var(--ink); margin-bottom: 8px;
}
.empty-state .sub {
    font-size: 13px; color: var(--slate);
    max-width: 360px; margin: 0 auto; line-height: 1.7;
}

/* ── CHART AREA ── */
.chart-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--slate-2);
    margin: 20px 0 10px 0; font-weight: 500;
}

/* ── GLOBAL TEXT OVERRIDES ── */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span { color: var(--ink) !important; }
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    color: var(--ink) !important;
    font-family: 'Newsreader', serif !important;
}
div[data-testid="stMarkdownContainer"] strong { color: var(--ink) !important; font-weight: 600 !important; }
div[data-testid="stMarkdownContainer"] code {
    color: var(--teal-deep) !important; background: var(--line-soft) !important;
}
hr { border-color: var(--line) !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "engine": None,
        "engine_error": None,
        "thread_config": None,
        "stage": "idle",           # idle | running | awaiting_review | complete
        "pipeline_step": 0,        # 0-4 tracks which strip to show while running
        "paused_state": None,
        "final_state": None,
        "show_feedback_box": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()

if st.session_state.engine is None and st.session_state.engine_error is None:
    try:
        st.session_state.engine = get_engine()
    except Exception as e:
        st.session_state.engine_error = str(e)


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="mira-header">
    <div class="mira-header-left">
        <div class="mira-mark">M</div>
        <div>
            <div class="mira-title">MIRA Clinical Audit Console</div>
            <div class="mira-subtitle">Patient data · Clinical guidelines · Safety review</div>
        </div>
    </div>
    <div class="live-pill">
        <div class="live-dot"></div>
        <div class="live-pill-text">System active</div>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.engine_error:
    st.error(
        f"**MIRA could not connect to its data layer.**\n\n"
        f"`{st.session_state.engine_error}`\n\n"
        f"Run `01_data_setup.ipynb` first to build the SQLite database and FAISS index.",
        icon="⚠️",
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

STAGE_RAIL = [
    ("sql",    "Querying records"),
    ("search", "Matching guidelines"),
    ("reason", "Drafting analysis"),
    ("review", "Your review"),
    ("critic", "Safety check"),
]

STAGE_INDEX = {
    "idle": -1,
    "running": 0,       # generic running — step controlled by pipeline_step
    "awaiting_review": 3,
    "complete": 5,
}


def render_fill_rail(stage: str, active_step: int = 0):
    """
    Renders the 5-node fill rail.
    active_step (0-4) is used when stage == 'running' to advance the node highlight
    without a full page re-render.
    """
    if stage == "idle":
        current_idx, fill_pct, running = -1, 0, False
    elif stage == "running":
        current_idx = active_step
        fill_pct = (active_step / len(STAGE_RAIL)) * 100 + (100 / len(STAGE_RAIL)) * 0.4
        running = True
    elif stage == "awaiting_review":
        current_idx, fill_pct, running = 3, 70, False
    elif stage == "complete":
        current_idx, fill_pct, running = 5, 100, False
    else:
        current_idx, fill_pct, running = -1, 0, False

    shimmer_cls = "is-running" if running else ""
    labels_html = ""
    for i, (_, label) in enumerate(STAGE_RAIL):
        if stage == "awaiting_review" and i == 3:
            node_cls, label_cls = "is-wait", "is-wait"
        elif i < current_idx:
            node_cls, label_cls = "is-done", "is-done"
        elif i == current_idx:
            node_cls, label_cls = "is-active", "is-active"
        else:
            node_cls, label_cls = "", ""

        labels_html += (
            f'<div class="fill-rail-label {label_cls}">'
            f'<span class="rail-node {node_cls}"></span>{label}</div>'
        )

    st.markdown(f"""
    <div class="fill-rail-wrap">
        <div class="fill-rail-track">
            <div class="fill-rail-progress {shimmer_cls}" style="width:{fill_pct}%;"></div>
        </div>
        <div class="fill-rail-labels">{labels_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_processing_strip(message: str):
    st.markdown(f"""
    <div class="processing-strip">
        <span class="processing-text">{message}</span>
        <div class="shimmer-bar"></div>
    </div>
    """, unsafe_allow_html=True)


def render_report_panel(markdown_text: str):
    html_body = md_lib.markdown(markdown_text, extensions=["extra"])
    st.markdown(
        f'<div class="panel report-surface">{html_body}</div>',
        unsafe_allow_html=True,
    )


def render_trend_panel(trend_result: dict):
    """Renders a colour-coded trajectory summary panel."""
    if not trend_result:
        return
    trend = trend_result.get("trend", "")
    summary = trend_result.get("summary", "")
    if not summary or trend == "insufficient_data":
        return

    panel_cls = ""
    label_text = "Lab Trajectory"
    if trend == "worsening":
        panel_cls, label_text = "worsening", "⚠ Worsening trajectory"
    elif trend in ("stable",):
        panel_cls, label_text = "stable", "✓ Stable trajectory"
    elif trend in ("rising", "falling"):
        label_text = f"Trajectory: {trend.title()}"

    pct = trend_result.get("pct_change")
    slope = trend_result.get("slope_per_hour")
    meta = ""
    if pct is not None:
        meta += f" &nbsp;·&nbsp; {pct:+.1f}%"
    if slope is not None:
        meta += f" &nbsp;·&nbsp; {slope:+.4f} units/hr"

    st.markdown(f"""
    <div class="trend-panel {panel_cls}">
        <div class="trend-label">{label_text}{meta}</div>
        <div class="trend-summary">{summary}</div>
    </div>
    """, unsafe_allow_html=True)


def extract_lab_chart_data(state: dict):
    """
    Pulls charting data from the SQL result rows.
    Returns a list of dicts with keys: label, valuenum, ref_lower, ref_upper, charttime.
    Only rows with numeric valuenum are included.
    """
    sql_result = state.get("sql_result", "")
    if not sql_result:
        return []
    try:
        rows = json.loads(sql_result).get("rows", [])
    except Exception:
        return []

    chart_rows = []
    for r in rows:
        try:
            val = float(r.get("valuenum") or 0)
            if val == 0:
                continue
            chart_rows.append({
                "label":    str(r.get("label") or r.get("d_label") or "Lab"),
                "valuenum": val,
                "ref_lower": float(r.get("ref_range_lower") or 0) or None,
                "ref_upper": float(r.get("ref_range_upper") or 0) or None,
                "charttime": str(r.get("charttime") or ""),
                "subject_id": str(r.get("subject_id") or ""),
            })
        except (TypeError, ValueError):
            continue
    return chart_rows


def render_lab_charts(state: dict):
    """
    Renders two complementary charts from the SQL result:
    1. Horizontal bar chart — lab value vs reference range (deviation view)
    2. Scatter/dot plot per patient showing raw values coloured by flag status
    Uses st.altair_chart which is bundled with Streamlit.
    """
    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        return

    chart_rows = extract_lab_chart_data(state)
    if not chart_rows:
        return

    df = pd.DataFrame(chart_rows)

    # ── Compute deviation ratio (how far past the reference bound) ──────
    def deviation(row):
        if row["ref_upper"] and row["valuenum"] > row["ref_upper"]:
            return round((row["valuenum"] - row["ref_upper"]) / row["ref_upper"] * 100, 1)
        if row["ref_lower"] and row["valuenum"] < row["ref_lower"]:
            return round((row["ref_lower"] - row["valuenum"]) / row["ref_lower"] * 100, 1)
        return 0.0

    df["deviation_pct"] = df.apply(deviation, axis=1)
    df["status"] = df.apply(
        lambda r: "Critical"
        if r["ref_upper"] and r["valuenum"] > r["ref_upper"] * 1.5
        else ("Abnormal" if r["deviation_pct"] > 0 else "Normal"),
        axis=1,
    )

    # ── Cap to 15 most deviated rows for readability ─────────────────────
    df_plot = df.nlargest(15, "deviation_pct").copy()
    # Truncate long labels
    df_plot["label_short"] = df_plot["label"].str[:28]
    df_plot["tooltip_label"] = (
        "Subject " + df_plot["subject_id"] + " · " + df_plot["label"]
    )

    color_scale = alt.Scale(
        domain=["Critical", "Abnormal", "Normal"],
        range=["#C9501F", "#B98A2E", "#2D8C7F"],
    )

    st.markdown('<div class="chart-eyebrow">Lab values — deviation from reference range</div>', unsafe_allow_html=True)

    # ── Chart 1: Horizontal bar — deviation % ────────────────────────────
    bar_chart = (
        alt.Chart(df_plot)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "deviation_pct:Q",
                title="% deviation beyond reference bound",
                axis=alt.Axis(grid=True, gridColor="#F0F1ED", labelColor="#8FA3AE", titleColor="#5C7A89"),
            ),
            y=alt.Y(
                "label_short:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelColor="#1C2B33", labelFont="IBM Plex Mono", labelFontSize=11),
            ),
            color=alt.Color("status:N", scale=color_scale, legend=alt.Legend(
                title=None, orient="top", labelColor="#5C7A89", labelFontSize=11,
            )),
            tooltip=[
                alt.Tooltip("tooltip_label:N", title="Patient · Lab"),
                alt.Tooltip("valuenum:Q", title="Value", format=".2f"),
                alt.Tooltip("ref_upper:Q", title="Ref upper", format=".2f"),
                alt.Tooltip("ref_lower:Q", title="Ref lower", format=".2f"),
                alt.Tooltip("deviation_pct:Q", title="% over bound", format=".1f"),
            ],
        )
        .properties(height=max(180, len(df_plot) * 28), background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(domainColor="#E7E9E4", tickColor="#E7E9E4")
    )

    st.altair_chart(bar_chart, use_container_width=True)

    # ── Chart 2: Dot strip — actual values per lab, coloured by status ───
    if len(df) > 1:
        st.markdown('<div class="chart-eyebrow" style="margin-top:18px;">Raw values by lab type</div>', unsafe_allow_html=True)

        strip_df = df.copy()
        strip_df["label_short"] = strip_df["label"].str[:28]

        strip_chart = (
            alt.Chart(strip_df)
            .mark_circle(size=68, opacity=0.82)
            .encode(
                x=alt.X(
                    "valuenum:Q",
                    title="Measured value",
                    axis=alt.Axis(grid=True, gridColor="#F0F1ED", labelColor="#8FA3AE", titleColor="#5C7A89"),
                ),
                y=alt.Y(
                    "label_short:N",
                    title=None,
                    axis=alt.Axis(labelColor="#1C2B33", labelFont="IBM Plex Mono", labelFontSize=11),
                ),
                color=alt.Color("status:N", scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("label:N", title="Lab"),
                    alt.Tooltip("subject_id:N", title="Subject"),
                    alt.Tooltip("valuenum:Q", title="Value", format=".2f"),
                    alt.Tooltip("charttime:N", title="Time"),
                ],
            )
            .properties(height=max(140, strip_df["label_short"].nunique() * 30), background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(domainColor="#E7E9E4", tickColor="#E7E9E4")
        )

        st.altair_chart(strip_chart, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT — two columns, no sidebar
# ══════════════════════════════════════════════════════════════════════════

left_col, right_col = st.columns([1, 1.6], gap="large")

EXAMPLE_QUESTIONS = [
    "Which patients have critical lab results?",
    "Find patients with abnormal creatinine — signs of AKI?",
    "Are there signs of sepsis in recent admissions?",
]

# ── LEFT COLUMN — query input + rail ─────────────────────────────────────
with left_col:
    st.markdown(
        '<div class="panel-eyebrow"><span class="eyebrow-dot"></span>CLINICAL QUERY</div>',
        unsafe_allow_html=True,
    )

    is_locked = st.session_state.stage in ["running", "awaiting_review"]

    question = st.text_area(
        label="Clinical query",
        placeholder="e.g. Which patients show signs of acute kidney injury based on their latest labs?",
        height=108,
        key="question_input",
        label_visibility="collapsed",
        disabled=is_locked,
    )

    st.markdown(
        '<div class="chip-row">' +
        "".join(f'<span class="chip">{q}</span>' for q in EXAMPLE_QUESTIONS) +
        '</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    run_disabled = is_locked or not question.strip()
    run_clicked = st.button(
        "Run clinical audit",
        type="primary",
        use_container_width=True,
        disabled=run_disabled,
    )

    st.write("")

    # Fill rail — reflects current stage
    render_fill_rail(
        stage=st.session_state.stage,
        active_step=st.session_state.pipeline_step,
    )

    # Data sources expander — shown once we have a state
    active_state = st.session_state.final_state or st.session_state.paused_state
    if active_state:
        with st.expander("View data sources used"):
            st.markdown('<div class="panel-eyebrow">SQL QUERY EXECUTED</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="mono-block">{active_state.get("sql_query_used", "—")}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="panel-eyebrow" style="margin-top:14px;">GUIDELINE SEARCH QUERY</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="mono-block">{active_state.get("search_query_used", "—")}</div>',
                unsafe_allow_html=True,
            )
            try:
                guidelines = json.loads(active_state.get("guidelines", "{}")).get("guidelines", [])
                if guidelines:
                    st.markdown('<div class="panel-eyebrow" style="margin-top:14px;">GUIDELINES RETRIEVED</div>', unsafe_allow_html=True)
                    for g in guidelines:
                        st.markdown(
                            f'<div style="font-size:12.5px; color:#5C7A89; padding:3px 0;">'
                            f'<strong style="color:#1C2B33;">{g["source"]}</strong> — {g["topic"]}</div>',
                            unsafe_allow_html=True,
                        )
            except Exception:
                pass


# ── RIGHT COLUMN — results area ───────────────────────────────────────────
with right_col:

    # ── IDLE ─────────────────────────────────────────────────────────────
    if st.session_state.stage == "idle":
        st.markdown("""
        <div class="empty-state">
            <div class="mark">◍</div>
            <div class="heading">No audit running</div>
            <div class="sub">Enter a clinical question on the left. MIRA queries the patient
            database, cross-references medical guidelines, and prepares a report for your review
            before anything is finalised.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── AWAITING HUMAN REVIEW ─────────────────────────────────────────────
    elif st.session_state.stage == "awaiting_review":
        state = st.session_state.paused_state

        st.markdown(
            '<div class="panel-eyebrow" style="color:#B98A2E;">'
            '<span class="eyebrow-dot" style="background:#B98A2E;"></span>'
            'DRAFT — AWAITING YOUR REVIEW</div>',
            unsafe_allow_html=True,
        )

        # Trend panel (if TrendAgent ran)
        render_trend_panel(state.get("trend_result", {}))

        # Draft report
        render_report_panel(state["clinical_reasoning"])

        # Lab charts
        render_lab_charts(state)

        st.write("")
        approve_col, reject_col = st.columns(2)
        with approve_col:
            approve_clicked = st.button("Approve and finalise", type="primary", use_container_width=True)
        with reject_col:
            reject_clicked = st.button("Request revision", use_container_width=True)

        if reject_clicked:
            st.session_state.show_feedback_box = True

        if st.session_state.show_feedback_box:
            st.write("")
            feedback_text = st.text_area(
                "What should the analysis address that it's currently missing?",
                placeholder="e.g. Be more specific about units, or add urgency levels for each finding.",
                key="feedback_box",
            )
            submit_feedback = st.button("Send revision request", type="primary")
            if submit_feedback:
                ph = st.empty()
                with ph.container():
                    render_processing_strip("Sending feedback to the clinical reasoning agent")
                final = st.session_state.engine.submit_human_decision(
                    st.session_state.thread_config, "reject", feedback_text
                )
                ph.empty()
                st.session_state.paused_state = final
                st.session_state.show_feedback_box = False
                st.rerun()

        if approve_clicked:
            ph = st.empty()
            with ph.container():
                render_processing_strip("Running final safety check")
            final = st.session_state.engine.submit_human_decision(
                st.session_state.thread_config, "approve"
            )
            ph.empty()
            st.session_state.final_state = final
            st.session_state.stage = "complete"
            st.rerun()

    # ── COMPLETE ──────────────────────────────────────────────────────────
    elif st.session_state.stage == "complete":
        state = st.session_state.final_state

        if state.get("approved"):
            st.markdown(
                '<div class="banner banner-approved">'
                '<span>✓</span>'
                'Cleared by safety review — every claim is grounded in retrieved data</div>',
                unsafe_allow_html=True,
            )
        else:
            flags = ", ".join(state.get("safety_flags", [])) or "Review recommended before clinical use"
            st.markdown(
                f'<div class="banner banner-flagged">'
                f'<span>▲</span>{flags}</div>',
                unsafe_allow_html=True,
            )

        # Trend panel
        render_trend_panel(state.get("trend_result", {}))

        st.markdown(
            '<div class="panel-eyebrow"><span class="eyebrow-dot"></span>FINAL CLINICAL REPORT</div>',
            unsafe_allow_html=True,
        )
        render_report_panel(state.get("final_report", "No report generated."))

        # Lab charts
        render_lab_charts(state)

        st.write("")
        if st.button("Start a new audit"):
            st.session_state.stage = "idle"
            st.session_state.pipeline_step = 0
            st.session_state.paused_state = None
            st.session_state.final_state = None
            st.session_state.thread_config = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# RUN HANDLER
# ══════════════════════════════════════════════════════════════════════════

if run_clicked and question.strip():
    st.session_state.thread_config = st.session_state.engine.new_thread()
    st.session_state.stage = "running"
    st.session_state.pipeline_step = 0

    # Each strip message also advances pipeline_step so the fill rail moves
    steps = [
        (0, "Querying patient database"),
        (1, "Cross-referencing clinical guidelines"),
        (2, "Analysing lab trajectories"),
        (2, "Drafting clinical analysis"),
    ]

    with right_col:
        progress_ph = st.empty()

        for step_idx, message in steps:
            st.session_state.pipeline_step = step_idx
            with progress_ph.container():
                # Re-render the fill rail inside the placeholder so it advances
                render_fill_rail(stage="running", active_step=step_idx)
                render_processing_strip(message)
            time.sleep(0.3)   # brief pause so the user sees each step

        # Actually run the pipeline (blocks until Agent 3 finishes)
        paused = st.session_state.engine.run_until_review(
            question.strip(), st.session_state.thread_config
        )
        progress_ph.empty()

    st.session_state.paused_state = paused
    st.session_state.stage = "awaiting_review"
    st.session_state.pipeline_step = 3
    st.rerun()