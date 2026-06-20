"""
streamlit_app.py
=================
MIRA — Multi-Agent Clinical Audit & Real-Time Triage System
Frontend UI layer. All clinical logic lives in mira_pipeline.py.

Design: light clinical-instrument console. Warm paper background,
serif headings, single teal accent reserved for "live" states, clay-red
reserved strictly for safety flags. A horizontal fill-rail replaces a
numbered stepper; a breathing dot replaces a spinner for ambient status.

Run with: streamlit run streamlit_app.py
"""

import json
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
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — TOKENS & CSS
# ══════════════════════════════════════════════════════════════════════════
#
# Color:
#   --paper      #FAFAF8   page background, warm not stark
#   --surface    #FFFFFF   card / panel surface
#   --ink        #1C2B33   primary text, near-black slate
#   --slate      #5C7A89   secondary text, instrument blue-grey
#   --slate-2    #8FA3AE   tertiary text / placeholders
#   --teal       #2D8C7F   the one accent — reserved for "live" / active states
#   --teal-tint  #EAF4F2   teal background tint
#   --clay       #C9501F   safety-flag red, rationed strictly
#   --clay-tint  #FCEEE6   clay background tint
#   --line       #E7E9E4   hairline border
#   --line-soft  #F0F1ED   even softer divider
#
# Type:
#   Newsreader   — headings, serif, clinical-journal register
#   Inter        — UI body text
#   IBM Plex Mono — data, SQL, identifiers, monospace numerals
#
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

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--paper);
}

header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1240px; }
#MainMenu, footer { visibility: hidden; }

/* HEADER */
.mira-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 22px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 30px;
}
.mira-header-left { display: flex; align-items: center; gap: 16px; }
.mira-mark {
    width: 42px; height: 42px;
    border-radius: 11px;
    background: var(--ink);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Newsreader', serif;
    font-weight: 600; font-size: 20px; color: var(--paper);
    flex-shrink: 0;
}
.mira-title {
    font-family: 'Newsreader', serif;
    font-size: 23px; font-weight: 500;
    color: var(--ink);
    letter-spacing: 0.1px;
    line-height: 1.1;
}
.mira-subtitle {
    font-size: 13px; color: var(--slate);
    margin-top: 2px;
    font-weight: 400;
}

.live-pill {
    display: flex; align-items: center; gap: 8px;
    background: var(--teal-tint);
    border: 1px solid rgba(45,140,127,0.18);
    border-radius: 100px;
    padding: 7px 14px 7px 11px;
}
.live-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--teal);
    position: relative;
    flex-shrink: 0;
}
.live-dot::before {
    content: '';
    position: absolute;
    top: -4px; left: -4px;
    width: 15px; height: 15px;
    border-radius: 50%;
    background: var(--teal);
    opacity: 0.35;
    animation: breathe 2.2s ease-in-out infinite;
}
@keyframes breathe {
    0%   { transform: scale(0.6); opacity: 0.45; }
    50%  { transform: scale(1.25); opacity: 0.08; }
    100% { transform: scale(0.6); opacity: 0.45; }
}
.live-pill-text {
    font-size: 12.5px; font-weight: 500;
    color: var(--teal-deep);
    letter-spacing: 0.01em;
}

/* SIGNATURE ELEMENT — fill rail */
.fill-rail-wrap { margin-bottom: 28px; }
.fill-rail-track {
    position: relative;
    height: 3px;
    background: var(--line);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 14px;
}
.fill-rail-progress {
    position: absolute;
    top: 0; left: 0; height: 100%;
    background: linear-gradient(90deg, var(--teal-deep), var(--teal));
    border-radius: 3px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.fill-rail-labels { display: flex; justify-content: space-between; }
.fill-rail-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.03em;
    color: var(--slate-2);
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: color 0.4s ease;
}
.fill-rail-label.is-active { color: var(--teal-deep); font-weight: 600; }
.fill-rail-label.is-done { color: var(--slate); }
.fill-rail-label.is-waiting { color: var(--gold); font-weight: 600; }
.fill-rail-icon { width: 6px; height: 6px; border-radius: 50%; background: var(--line); flex-shrink: 0; }
.fill-rail-icon.is-active { background: var(--teal); position: relative; }
.fill-rail-icon.is-active::before {
    content: '';
    position: absolute; top: -3px; left: -3px;
    width: 12px; height: 12px; border-radius: 50%;
    background: var(--teal); opacity: 0.3;
    animation: breathe 1.6s ease-in-out infinite;
}
.fill-rail-icon.is-done { background: var(--teal-deep); }
.fill-rail-icon.is-waiting { background: var(--gold); }

/* PROCESSING STRIP — shimmer, not a spinner */
.processing-strip {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 13px 16px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    margin-top: 14px;
}
.processing-text { font-size: 12.5px; color: var(--slate); font-weight: 500; }
.shimmer-bar {
    position: relative;
    flex: 1;
    height: 2px;
    background: var(--line-soft);
    border-radius: 2px;
    overflow: hidden;
}
.shimmer-bar::after {
    content: '';
    position: absolute;
    top: 0; left: -40%;
    width: 40%; height: 100%;
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
    animation: shimmer 1.4s ease-in-out infinite;
}
@keyframes shimmer {
    0%   { left: -40%; }
    100% { left: 100%; }
}

/* PANELS */
.panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 24px 26px;
    margin-bottom: 18px;
}
.panel-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--slate-2);
    margin-bottom: 12px;
    font-weight: 500;
}
.panel-eyebrow.with-dot { display: flex; align-items: center; gap: 7px; }
.eyebrow-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--teal); }

/* REPORT TYPOGRAPHY */
.report-surface h2 {
    font-family: 'Newsreader', serif;
    font-size: 17px; font-weight: 600;
    color: var(--ink);
    margin-top: 22px; margin-bottom: 9px;
    padding-top: 18px;
    border-top: 1px solid var(--line-soft);
}
.report-surface h2:first-child { margin-top: 0; padding-top: 0; border-top: none; }
.report-surface p, .report-surface li { font-size: 14.5px; line-height: 1.7; color: var(--ink); }
.report-surface strong { color: var(--ink); font-weight: 600; }
.report-surface code {
    font-family: 'IBM Plex Mono', monospace;
    background: var(--line-soft);
    padding: 1px 6px; border-radius: 4px;
    font-size: 13px; color: var(--teal-deep);
}

/* BANNERS */
.banner {
    display: flex; align-items: center; gap: 10px;
    border-radius: 10px;
    padding: 13px 16px;
    margin-bottom: 16px;
    font-size: 13.5px; font-weight: 500;
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
.banner-icon { font-size: 14px; flex-shrink: 0; }

/* MONO DATA BLOCKS */
.mono-block {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    line-height: 1.55;
    background: var(--line-soft);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 13px 15px;
    color: var(--slate);
    white-space: pre-wrap;
    overflow-x: auto;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
.sidebar-section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--slate-2);
    margin-bottom: 10px;
    font-weight: 500;
}
.history-item {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 9px 0;
    border-bottom: 1px solid var(--line-soft);
    font-size: 12.5px;
    color: var(--slate);
}
.history-item:last-child { border-bottom: none; }
.history-dot { flex-shrink: 0; margin-top: 4px; font-size: 9px; }
.history-dot.ok { color: var(--teal); }
.history-dot.flag { color: var(--clay); }

/* BUTTONS */
.stButton button {
    border-radius: 9px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 0.55rem 1.1rem !important;
    border: 1px solid var(--line) !important;
    background: var(--surface) !important;
    color: var(--ink) !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}
.stButton button:hover {
    border-color: var(--slate-2) !important;
    background: var(--line-soft) !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    background: var(--ink) !important;
    border: 1px solid var(--ink) !important;
    color: var(--paper) !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #0F1A1F !important;
    border-color: #0F1A1F !important;
}

/* INPUTS */
.stTextArea textarea, .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    color: var(--ink) !important;
    border-radius: 10px !important;
    font-size: 14.5px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px var(--teal-tint) !important;
}
.stTextArea textarea::placeholder { color: var(--slate-2) !important; }

/* EXAMPLE CHIPS */
.chip-row { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }
.chip {
    display: inline-flex; align-items: center;
    background: var(--line-soft);
    border: 1px solid var(--line);
    border-radius: 100px;
    padding: 6px 13px;
    font-size: 12px;
    color: var(--slate);
}

/* EMPTY STATE */
.empty-state { text-align: center; padding: 90px 30px; }
.empty-state .mark {
    width: 52px; height: 52px;
    border-radius: 14px;
    background: var(--line-soft);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 18px auto;
    font-family: 'Newsreader', serif;
    font-size: 22px; color: var(--slate-2);
}
.empty-state .heading {
    font-family: 'Newsreader', serif;
    font-size: 18px; font-weight: 500;
    color: var(--ink); margin-bottom: 8px;
}
.empty-state .sub {
    font-size: 13.5px; color: var(--slate);
    max-width: 380px; margin: 0 auto;
    line-height: 1.65;
}

/* EXPANDER */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    color: var(--slate) !important;
}

/* FORCE TEXT COLOR — Streamlit's native markdown elements don't inherit
   our --ink variable correctly in all themes, so we set it explicitly. */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span {
    color: var(--ink) !important;
}
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    color: var(--ink) !important;
    font-family: 'Newsreader', serif !important;
}
div[data-testid="stMarkdownContainer"] strong { color: var(--ink) !important; font-weight: 600 !important; }
div[data-testid="stMarkdownContainer"] code {
    color: var(--teal-deep) !important;
    background: var(--line-soft) !important;
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
        "stage": "idle",
        "paused_state": None,
        "final_state": None,
        "history": [],
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
            <div class="mira-subtitle">Cross-referencing live patient data against medical guidelines</div>
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
        icon="⚠️"
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-section-title">Recent audits</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            '<div style="font-size:12.5px; color:#8FA3AE; line-height:1.6;">'
            'Audits you run this session will appear here.</div>',
            unsafe_allow_html=True
        )
    else:
        items_html = ""
        for entry in reversed(st.session_state.history):
            label = entry["question"][:50] + ("…" if len(entry["question"]) > 50 else "")
            dot_class = "ok" if entry["approved"] else "flag"
            symbol = "●" if entry["approved"] else "▲"
            items_html += (
                f'<div class="history-item">'
                f'<span class="history-dot {dot_class}">{symbol}</span>'
                f'<span>{label}</span></div>'
            )
        st.markdown(items_html, unsafe_allow_html=True)

    st.markdown(
        '<div style="height:1px; background:#E7E9E4; margin:22px 0 18px 0;"></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section-title">About this review</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12.5px; color:#5C7A89; line-height:1.7;">'
        'Every finding is grounded in a live database query and cross-checked '
        'against clinical guidelines. A report only reaches you after passing '
        'an automated safety check — and nothing is finalized without your review.'
        '</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════
# FILL RAIL RENDERER
# ══════════════════════════════════════════════════════════════════════════

def render_fill_rail(stage: str):
    nodes = [
        ("sql", "Querying records"),
        ("search", "Matching guidelines"),
        ("reason", "Drafting analysis"),
        ("review", "Your review"),
        ("critic", "Safety check"),
    ]

    stage_index_map = {
        "idle": -1,
        "running_sql": 0,
        "running_search": 1,
        "running_reasoning": 2,
        "awaiting_review": 3,
        "revising": 2,
        "running_critic": 4,
        "complete": 5,
    }
    current_idx = stage_index_map.get(stage, -1)
    total = len(nodes)

    if stage == "idle":
        fill_pct = 0
    elif stage == "complete":
        fill_pct = 100
    else:
        fill_pct = ((current_idx) / total) * 100 + (100 / total) * 0.5

    labels_html = ""
    for i, (key, label) in enumerate(nodes):
        if stage == "awaiting_review" and key == "review":
            cls, icon_cls = "is-waiting", "is-waiting"
        elif i < current_idx:
            cls, icon_cls = "is-done", "is-done"
        elif i == current_idx:
            cls, icon_cls = "is-active", "is-active"
        else:
            cls, icon_cls = "", ""

        labels_html += (
            f'<div class="fill-rail-label {cls}">'
            f'<span class="fill-rail-icon {icon_cls}"></span>{label}</div>'
        )

    st.markdown(f"""
    <div class="fill-rail-wrap">
        <div class="fill-rail-track">
            <div class="fill-rail-progress" style="width:{fill_pct}%;"></div>
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
    """
    Renders agent-generated markdown as a single real HTML block inside
    the styled panel, instead of splitting raw-HTML divs and st.markdown
    calls (which don't visually nest and caused the invisible-text bug).
    """
    html_body = md_lib.markdown(markdown_text, extensions=["extra"])
    st.markdown(
        f'<div class="panel report-surface">{html_body}</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════

left_col, right_col = st.columns([1, 1.55], gap="large")

EXAMPLE_QUESTIONS = [
    "Which patients have critical lab results?",
    "Find patients with abnormal creatinine — signs of AKI?",
    "Are there signs of sepsis in recent admissions?",
]

with left_col:
    st.markdown('<div class="panel-eyebrow with-dot"><span class="eyebrow-dot"></span>CLINICAL QUERY</div>', unsafe_allow_html=True)

    question = st.text_area(
        label="Clinical query",
        placeholder="e.g. Which patients show signs of acute kidney injury based on their latest labs?",
        height=104,
        key="question_input",
        label_visibility="collapsed",
        disabled=(st.session_state.stage in ["running", "awaiting_review"])
    )

    chips_html = '<div class="chip-row">' + "".join(
        f'<span class="chip">{q}</span>' for q in EXAMPLE_QUESTIONS
    ) + '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)
    st.write("")

    run_disabled = st.session_state.stage in ["running", "awaiting_review"] or not question.strip()
    run_clicked = st.button(
        "Run clinical audit",
        type="primary",
        use_container_width=True,
        disabled=run_disabled
    )

    st.write("")
    render_fill_rail(
        "running_sql" if st.session_state.stage == "running" else
        "awaiting_review" if st.session_state.stage == "awaiting_review" else
        "complete" if st.session_state.stage == "complete" else
        "idle"
    )

    if st.session_state.paused_state or st.session_state.final_state:
        active_state = st.session_state.final_state or st.session_state.paused_state
        with st.expander("View data sources used"):
            st.markdown('<div class="panel-eyebrow">SQL QUERY EXECUTED</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="mono-block">{active_state.get("sql_query_used", "—")}</div>', unsafe_allow_html=True)

            st.markdown('<div class="panel-eyebrow" style="margin-top:16px;">GUIDELINE SEARCH QUERY</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="mono-block">{active_state.get("search_query_used", "—")}</div>', unsafe_allow_html=True)

            try:
                guidelines = json.loads(active_state.get("guidelines", "{}")).get("guidelines", [])
                if guidelines:
                    st.markdown('<div class="panel-eyebrow" style="margin-top:16px;">GUIDELINES RETRIEVED</div>', unsafe_allow_html=True)
                    for g in guidelines:
                        st.markdown(
                            f'<div style="font-size:13px; color:#5C7A89; padding:4px 0;">'
                            f'<strong style="color:#1C2B33;">{g["source"]}</strong> — {g["topic"]}</div>',
                            unsafe_allow_html=True
                        )
            except Exception:
                pass


with right_col:

    if st.session_state.stage == "idle":
        st.markdown("""
        <div class="empty-state">
            <div class="mark">◍</div>
            <div class="heading">No audit running</div>
            <div class="sub">Enter a clinical question on the left. MIRA queries the patient
            database, cross-references medical guidelines, and prepares a report for your review
            before anything is finalized.</div>
        </div>
        """, unsafe_allow_html=True)

    elif st.session_state.stage == "awaiting_review":
        state = st.session_state.paused_state

        st.markdown(
            '<div class="panel-eyebrow with-dot" style="color:#B98A2E;">'
            '<span class="eyebrow-dot" style="background:#B98A2E;"></span>'
            'DRAFT — AWAITING YOUR REVIEW</div>',
            unsafe_allow_html=True
        )

        render_report_panel(state["clinical_reasoning"])

        approve_col, reject_col = st.columns(2)
        with approve_col:
            approve_clicked = st.button("Approve and finalize", type="primary", use_container_width=True)
        with reject_col:
            reject_clicked = st.button("Request revision", use_container_width=True)

        if reject_clicked:
            st.session_state.show_feedback_box = True

        if st.session_state.show_feedback_box:
            st.write("")
            feedback_text = st.text_area(
                "What should the analysis address that it's currently missing?",
                placeholder="e.g. Be more specific about units, or add an urgency level for each finding.",
                key="feedback_box",
                label_visibility="visible"
            )
            submit_feedback = st.button("Send revision request", type="primary")
            if submit_feedback:
                feedback_placeholder = st.empty()
                with feedback_placeholder.container():
                    render_processing_strip("Sending feedback to the clinical reasoning agent")
                final = st.session_state.engine.submit_human_decision(
                    st.session_state.thread_config, "reject", feedback_text
                )
                feedback_placeholder.empty()
                st.session_state.paused_state = final
                st.session_state.show_feedback_box = False
                st.rerun()

        if approve_clicked:
            approve_placeholder = st.empty()
            with approve_placeholder.container():
                render_processing_strip("Running final safety check")
            final = st.session_state.engine.submit_human_decision(
                st.session_state.thread_config, "approve"
            )
            approve_placeholder.empty()
            st.session_state.final_state = final
            st.session_state.stage = "complete"
            st.session_state.history.append({
                "question": final["clinical_question"],
                "approved": final.get("approved", True)
            })
            st.rerun()

    elif st.session_state.stage == "complete":
        state = st.session_state.final_state

        if state.get("approved"):
            st.markdown(
                '<div class="banner banner-approved">'
                '<span class="banner-icon">✓</span>'
                'Cleared by safety review — every claim is grounded in retrieved data</div>',
                unsafe_allow_html=True
            )
        else:
            flags = ", ".join(state.get("safety_flags", [])) or "Review recommended before clinical use"
            st.markdown(
                f'<div class="banner banner-flagged">'
                f'<span class="banner-icon">▲</span>'
                f'Flagged by safety review — {flags}</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="panel-eyebrow with-dot"><span class="eyebrow-dot"></span>FINAL CLINICAL REPORT</div>', unsafe_allow_html=True)
        render_report_panel(state.get("final_report", "No report generated."))

        if st.button("Start a new audit"):
            st.session_state.stage = "idle"
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

    with right_col:
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            render_processing_strip("Querying patient database")
            time.sleep(0.35)
            render_processing_strip("Cross-referencing clinical guidelines")
            st.markdown('</div>', unsafe_allow_html=True)

        paused = st.session_state.engine.run_until_review(
            question.strip(), st.session_state.thread_config
        )
        progress_placeholder.empty()

    st.session_state.paused_state = paused
    st.session_state.stage = "awaiting_review"
    st.rerun()