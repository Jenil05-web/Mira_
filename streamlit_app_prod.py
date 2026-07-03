"""
streamlit_app_prod.py
======================
MIRA Production — Clinical Audit Console (Production Build)

Differences from streamlit_app.py (dev):
  - Auth gate: JWT login required before any clinical data is visible
  - Hospital context: every query tagged with user's hospital_id
  - Admin panel: audit log viewer + stats (admin role only)
  - Audit trail: every action logged via AuditLogger
  - Uses mira_pipeline_prod.py instead of mira_pipeline.py

Run with: streamlit run streamlit_app_prod.py
"""

import json
import time
import uuid

import markdown as md_lib
import streamlit as st

from auth import AuthManager, require_auth, Role
from config_manager import ConfigManager
from audit_logger import AuditLogger
from mira_pipeline_prod import get_engine

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
# DESIGN SYSTEM (same tokens as dev, with admin panel additions)
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
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
#MainMenu, footer { visibility: hidden; }

/* HEADER */
.mira-header {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 22px; border-bottom: 1px solid var(--line); margin-bottom: 30px;
}
.mira-header-left { display: flex; align-items: center; gap: 16px; }
.mira-mark {
    width: 42px; height: 42px; border-radius: 11px; background: var(--ink);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Newsreader', serif; font-weight: 600; font-size: 20px; color: var(--paper);
}
.mira-title { font-family: 'Newsreader', serif; font-size: 23px; font-weight: 500; color: var(--ink); line-height: 1.1; }
.mira-subtitle { font-size: 13px; color: var(--slate); margin-top: 2px; }
.header-right { display: flex; align-items: center; gap: 12px; }

.live-pill {
    display: flex; align-items: center; gap: 8px;
    background: var(--teal-tint); border: 1px solid rgba(45,140,127,0.18);
    border-radius: 100px; padding: 7px 14px 7px 11px;
}
.live-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--teal);
    position: relative; flex-shrink: 0;
}
.live-dot::before {
    content: ''; position: absolute; top: -4px; left: -4px;
    width: 15px; height: 15px; border-radius: 50%;
    background: var(--teal); opacity: 0.35;
    animation: breathe 2.2s ease-in-out infinite;
}
@keyframes breathe {
    0%   { transform: scale(0.6); opacity: 0.45; }
    50%  { transform: scale(1.25); opacity: 0.08; }
    100% { transform: scale(0.6); opacity: 0.45; }
}
.live-pill-text { font-size: 12.5px; font-weight: 500; color: var(--teal-deep); }

/* USER BADGE */
.user-badge {
    display: flex; align-items: center; gap: 8px;
    background: var(--line-soft); border: 1px solid var(--line);
    border-radius: 100px; padding: 6px 14px 6px 10px;
    font-size: 12.5px; color: var(--slate);
}
.role-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.role-dot.clinician { background: var(--teal); }
.role-dot.admin { background: var(--gold); }

/* FILL RAIL */
.fill-rail-wrap { margin-bottom: 28px; }
.fill-rail-track {
    position: relative; height: 3px; background: var(--line);
    border-radius: 3px; overflow: hidden; margin-bottom: 14px;
}
.fill-rail-progress {
    position: absolute; top: 0; left: 0; height: 100%;
    background: linear-gradient(90deg, var(--teal-deep), var(--teal));
    border-radius: 3px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.fill-rail-labels { display: flex; justify-content: space-between; }
.fill-rail-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;
    letter-spacing: 0.03em; color: var(--slate-2); text-transform: uppercase;
    display: flex; align-items: center; gap: 6px; transition: color 0.4s ease;
}
.fill-rail-label.is-active { color: var(--teal-deep); font-weight: 600; }
.fill-rail-label.is-done { color: var(--slate); }
.fill-rail-label.is-waiting { color: var(--gold); font-weight: 600; }
.fill-rail-icon { width: 6px; height: 6px; border-radius: 50%; background: var(--line); flex-shrink: 0; }
.fill-rail-icon.is-active { background: var(--teal); }
.fill-rail-icon.is-done { background: var(--teal-deep); }
.fill-rail-icon.is-waiting { background: var(--gold); }

/* PROCESSING STRIP */
.processing-strip {
    display: flex; align-items: center; gap: 11px; padding: 13px 16px;
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 10px; margin-top: 14px;
}
.processing-text { font-size: 12.5px; color: var(--slate); font-weight: 500; }
.shimmer-bar {
    position: relative; flex: 1; height: 2px;
    background: var(--line-soft); border-radius: 2px; overflow: hidden;
}
.shimmer-bar::after {
    content: ''; position: absolute; top: 0; left: -40%; width: 40%; height: 100%;
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
    animation: shimmer 1.4s ease-in-out infinite;
}
@keyframes shimmer { 0% { left: -40%; } 100% { left: 100%; } }

/* PANELS */
.panel {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 14px; padding: 24px 26px; margin-bottom: 18px;
}
.panel-eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;
    letter-spacing: 0.09em; text-transform: uppercase; color: var(--slate-2);
    margin-bottom: 12px; font-weight: 500;
}
.panel-eyebrow.with-dot { display: flex; align-items: center; gap: 7px; }
.eyebrow-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--teal); }

/* REPORT TYPOGRAPHY */
.report-surface h2 {
    font-family: 'Newsreader', serif; font-size: 17px; font-weight: 600; color: var(--ink);
    margin-top: 22px; margin-bottom: 9px; padding-top: 18px; border-top: 1px solid var(--line-soft);
}
.report-surface h2:first-child { margin-top: 0; padding-top: 0; border-top: none; }
.report-surface p, .report-surface li { font-size: 14.5px; line-height: 1.7; color: var(--ink); }
.report-surface strong { color: var(--ink); font-weight: 600; }

/* FORCE TEXT COLOR */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li { color: var(--ink) !important; }
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    color: var(--ink) !important; font-family: 'Newsreader', serif !important;
}
div[data-testid="stMarkdownContainer"] strong { color: var(--ink) !important; font-weight: 600 !important; }

/* BANNERS */
.banner {
    display: flex; align-items: center; gap: 10px; border-radius: 10px;
    padding: 13px 16px; margin-bottom: 16px; font-size: 13.5px; font-weight: 500;
}
.banner-approved { background: var(--teal-tint); border: 1px solid rgba(45,140,127,0.22); color: var(--teal-deep); }
.banner-flagged  { background: var(--clay-tint); border: 1px solid rgba(201,80,31,0.22); color: var(--clay); }

/* MONO BLOCK */
.mono-block {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; line-height: 1.55;
    background: var(--line-soft); border: 1px solid var(--line); border-radius: 8px;
    padding: 13px 15px; color: var(--slate); white-space: pre-wrap; overflow-x: auto;
}

/* ADMIN TABLE */
.audit-row {
    display: flex; align-items: center; gap: 12px; padding: 9px 0;
    border-bottom: 1px solid var(--line-soft); font-size: 12.5px;
}
.audit-row:last-child { border-bottom: none; }
.audit-badge {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    padding: 3px 8px; border-radius: 100px; font-weight: 600;
}
.badge-query    { background: #EAF4F2; color: #1F6358; }
.badge-agent    { background: #FBF3E4; color: #B98A2E; }
.badge-review   { background: #EAF4F2; color: #1F6358; }
.badge-error    { background: #FCEEE6; color: #C9501F; }
.badge-login    { background: var(--line-soft); color: var(--slate); }

/* STAT CARDS */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 24px; }
.stat-card {
    background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
    padding: 18px 20px;
}
.stat-value { font-family: 'Newsreader', serif; font-size: 28px; font-weight: 500; color: var(--ink); }
.stat-label { font-size: 12px; color: var(--slate); margin-top: 4px; }

/* BUTTONS */
.stButton button {
    border-radius: 9px !important; font-weight: 500 !important; font-size: 14px !important;
    padding: 0.55rem 1.1rem !important; border: 1px solid var(--line) !important;
    background: var(--surface) !important; color: var(--ink) !important;
    transition: all 0.15s ease !important; box-shadow: none !important;
}
.stButton button:hover { border-color: var(--slate-2) !important; background: var(--line-soft) !important; }
.stButton button p, .stButton button span, .stButton button div { color: inherit !important; }
div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stButton"] button[kind="primary"] p,
div[data-testid="stButton"] button[kind="primary"] span {
    background: var(--ink) !important; border: 1px solid var(--ink) !important; color: var(--paper) !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover { background: #0F1A1F !important; }

/* INPUTS */
.stTextArea textarea, .stTextInput input {
    background: var(--surface) !important; border: 1px solid var(--line) !important;
    color: var(--ink) !important; border-radius: 10px !important; font-size: 14.5px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--teal) !important; box-shadow: 0 0 0 3px var(--teal-tint) !important;
}

/* CHIPS */
.chip-row { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }
.chip {
    display: inline-flex; align-items: center; background: var(--line-soft);
    border: 1px solid var(--line); border-radius: 100px; padding: 6px 13px;
    font-size: 12px; color: var(--slate);
}

/* EMPTY STATE */
.empty-state { text-align: center; padding: 90px 30px; }
.empty-state .mark {
    width: 52px; height: 52px; border-radius: 14px; background: var(--line-soft);
    display: flex; align-items: center; justify-content: center; margin: 0 auto 18px auto;
    font-family: 'Newsreader', serif; font-size: 22px; color: var(--slate-2);
}
.empty-state .heading { font-family: 'Newsreader', serif; font-size: 18px; font-weight: 500; color: var(--ink); margin-bottom: 8px; }
.empty-state .sub { font-size: 13.5px; color: var(--slate); max-width: 380px; margin: 0 auto; line-height: 1.65; }

hr { border-color: var(--line) !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# INITIALISE SERVICES (cached — one instance per app)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_services():
    cfg = ConfigManager()
    auth = AuthManager(
        jwt_secret=cfg.get("MIRA_JWT_SECRET", "dev_secret_change_in_production"),
        expire_minutes=int(cfg.get("MIRA_ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
    )
    audit_cfg = cfg.get_audit_config()
    audit = AuditLogger(audit_cfg["connection_string"], audit_cfg["enabled"])
    engine = get_engine(cfg)
    return cfg, auth, audit, engine


cfg, auth_manager, audit_logger, engine = load_services()


# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "auth_token": "",
        "auth_user": None,
        "session_id": str(uuid.uuid4()),
        "thread_config": None,
        "stage": "idle",
        "paused_state": None,
        "final_state": None,
        "show_feedback_box": False,
        "active_tab": "audit",   # "audit" | "admin"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


# ══════════════════════════════════════════════════════════════════════════
# AUTH GATE — must pass before anything clinical renders
# ══════════════════════════════════════════════════════════════════════════

user = auth_manager.get_user_from_token(st.session_state.auth_token)

if not user:
    # ── Login screen ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="max-width:400px; margin:80px auto 0 auto; text-align:center;">
        <div style="font-family:'Newsreader',serif; font-size:32px; font-weight:500;
                    color:#1C2B33; margin-bottom:6px;">MIRA</div>
        <div style="font-size:13.5px; color:#5C7A89; margin-bottom:36px;">
            Multi-Agent Clinical Audit & Real-Time Triage System
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 1.4, 1])
    with col_m:
        st.markdown('<div style="background:#fff;border:1px solid #E7E9E4;border-radius:16px;padding:32px 28px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:16px;font-weight:600;color:#1C2B33;margin-bottom:20px;">Sign in</div>', unsafe_allow_html=True)

        email    = st.text_input("Email", placeholder="clinician@hospital.com", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        st.markdown("")

        if st.button("Sign in", type="primary", use_container_width=True):
            login_user = auth_manager.login(email, password)
            if login_user:
                st.session_state.auth_token = auth_manager.create_token(login_user)
                st.session_state.auth_user  = login_user
                audit_logger.log_login(login_user.user_id, login_user.hospital_id,
                                       st.session_state.session_id)
                st.rerun()
            else:
                st.error("Invalid email or password.")

        st.markdown("")
        st.caption("Dev: clinician@mira.dev / mira_clinician_2024")
        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# HEADER (only shown when authenticated)
# ══════════════════════════════════════════════════════════════════════════

role_dot_class = "admin" if user.role == Role.ADMIN else "clinician"

st.markdown(f"""
<div class="mira-header">
    <div class="mira-header-left">
        <div class="mira-mark">M</div>
        <div>
            <div class="mira-title">MIRA Clinical Audit Console</div>
            <div class="mira-subtitle">Cross-referencing live patient data against medical guidelines</div>
        </div>
    </div>
    <div class="header-right">
        <div class="live-pill">
            <div class="live-dot"></div>
            <div class="live-pill-text">System active</div>
        </div>
        <div class="user-badge">
            <span class="role-dot {role_dot_class}"></span>
            {user.display_name} &nbsp;·&nbsp; {user.role}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB NAV — Audit Console | Admin (admin role only)
# ══════════════════════════════════════════════════════════════════════════

if user.can("view_audit_log"):
    tab_audit, tab_admin = st.tabs(["Clinical Audit", "Admin"])
else:
    tab_audit = st.container()
    tab_admin = None


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def render_fill_rail(stage: str):
    nodes = [
        ("sql",    "Querying records"),
        ("search", "Matching guidelines"),
        ("reason", "Drafting analysis"),
        ("review", "Your review"),
        ("critic", "Safety check"),
    ]
    stage_index_map = {
        "idle": -1, "running_sql": 0, "running_search": 1,
        "running_reasoning": 2, "awaiting_review": 3, "complete": 5,
    }
    current_idx = stage_index_map.get(stage, -1)
    total = len(nodes)
    fill_pct = 0 if stage == "idle" else (100 if stage == "complete" else
               ((current_idx / total) * 100 + (100 / total) * 0.5))

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
        labels_html += (f'<div class="fill-rail-label {cls}">'
                        f'<span class="fill-rail-icon {icon_cls}"></span>{label}</div>')

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
    </div>""", unsafe_allow_html=True)


def render_report_panel(markdown_text: str):
    html_body = md_lib.markdown(markdown_text, extensions=["extra"])
    st.markdown(f'<div class="panel report-surface">{html_body}</div>',
                unsafe_allow_html=True)


def render_trend_chart(trend_json: str):
    if not trend_json:
        return False
    try:
        trend = json.loads(trend_json)
    except Exception:
        return False
    readings = trend.get("readings", [])
    if len(readings) < 2:
        return False

    labels    = [r.get("charttime", "")[:16] for r in readings]
    values    = [r.get("valuenum") for r in readings]
    unit      = readings[-1].get("valueuom", "") or ""
    lab_name  = trend.get("lab_name", "Lab value").title()
    trend_dir = trend.get("trend", "stable")
    ref_upper = readings[-1].get("ref_range_upper")
    ref_lower = readings[-1].get("ref_range_lower")

    color_map = {"worsening": "#C9501F", "rising": "#B98A2E",
                 "falling": "#B98A2E", "stable": "#2D8C7F", "improving": "#2D8C7F"}
    line_color = color_map.get(trend_dir, "#2D8C7F")
    badge = {"worsening": "Worsening", "rising": "Rising", "falling": "Falling",
             "stable": "Stable", "improving": "Improving"}.get(trend_dir, trend_dir.title())

    html = f"""
    <div style="font-family:'Inter',sans-serif;background:#fff;border:1px solid #E7E9E4;
                border-radius:14px;padding:22px 24px;margin-bottom:18px;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10.5px;
                        letter-spacing:0.09em;text-transform:uppercase;color:#8FA3AE;font-weight:500;">
                LAB TRAJECTORY — {lab_name}
            </div>
            <div style="display:flex;align-items:center;gap:6px;background:{line_color}1A;
                        border:1px solid {line_color}40;border-radius:100px;padding:4px 12px;">
                <span style="width:6px;height:6px;border-radius:50%;background:{line_color};"></span>
                <span style="font-size:12px;font-weight:500;color:{line_color};">{badge}</span>
            </div>
        </div>
        <div style="position:relative;height:220px;">
            <canvas id="tc_{id(trend_json)}"></canvas>
        </div>
        <div style="display:flex;gap:16px;margin-top:14px;font-size:12px;color:#5C7A89;">
            <span><span style="display:inline-block;width:10px;height:2px;background:{line_color};
                  margin-right:5px;vertical-align:middle;"></span>{lab_name} ({unit})</span>
            <span><span style="display:inline-block;width:10px;height:0;border-top:1px dashed #8FA3AE;
                  margin-right:5px;vertical-align:middle;"></span>Reference range</span>
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    new Chart(document.getElementById('tc_{id(trend_json)}'), {{
        type: 'line',
        data: {{
            labels: {json.dumps(labels)},
            datasets: [{{
                label: '{lab_name}',
                data: {json.dumps(values)},
                borderColor: '{line_color}',
                backgroundColor: '{line_color}15',
                borderWidth: 2, pointRadius: 4,
                pointBackgroundColor: '{line_color}',
                tension: 0.25, fill: true
            }}]
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
                x: {{ ticks: {{ color: '#8FA3AE', font: {{ size: 11 }} }}, grid: {{ display: false }} }},
                y: {{ ticks: {{ color: '#8FA3AE', font: {{ size: 11 }} }}, grid: {{ color: '#F0F1ED' }} }}
            }}
        }}
    }});
    </script>"""

    st.components.v1.html(html, height=340)
    return True


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — CLINICAL AUDIT CONSOLE
# ══════════════════════════════════════════════════════════════════════════

EXAMPLE_QUESTIONS = [
    "Which patients have critical lab results?",
    "Find patients with abnormal creatinine — signs of AKI?",
    "Are there signs of sepsis in recent admissions?",
]

with tab_audit:
    left_col, right_col = st.columns([1, 1.55], gap="large")

    with left_col:
        st.markdown('<div class="panel-eyebrow with-dot"><span class="eyebrow-dot"></span>CLINICAL QUERY</div>',
                    unsafe_allow_html=True)

        question = st.text_area(
            label="Clinical query",
            placeholder="e.g. Which patients show signs of AKI based on their latest labs?",
            height=104, key="question_input", label_visibility="collapsed",
            disabled=(st.session_state.stage in ["running", "awaiting_review"])
        )

        st.markdown('<div class="chip-row">' +
                    "".join(f'<span class="chip">{q}</span>' for q in EXAMPLE_QUESTIONS) +
                    '</div>', unsafe_allow_html=True)
        st.write("")

        run_disabled = st.session_state.stage in ["running", "awaiting_review"] or not question.strip()
        run_clicked = st.button("Run clinical audit", type="primary",
                                use_container_width=True, disabled=run_disabled)

        st.write("")
        render_fill_rail(
            "running_sql" if st.session_state.stage == "running" else
            "awaiting_review" if st.session_state.stage == "awaiting_review" else
            "complete" if st.session_state.stage == "complete" else "idle"
        )

        if st.session_state.paused_state or st.session_state.final_state:
            active = st.session_state.final_state or st.session_state.paused_state
            with st.expander("View data sources used"):
                st.markdown('<div class="panel-eyebrow">SQL QUERY EXECUTED</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="mono-block">{active.get("sql_query_used","—")}</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-eyebrow" style="margin-top:16px;">GUIDELINE SEARCH QUERY</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="mono-block">{active.get("search_query_used","—")}</div>', unsafe_allow_html=True)

                trend_json = active.get("trend_data", "")
                if trend_json:
                    try:
                        tp = json.loads(trend_json)
                        st.markdown('<div class="panel-eyebrow" style="margin-top:16px;">LAB TRAJECTORY</div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="font-size:13px;color:#5C7A89;">{tp.get("summary","")}</div>', unsafe_allow_html=True)
                    except Exception:
                        pass

                try:
                    guidelines = json.loads(active.get("guidelines", "{}")).get("guidelines", [])
                    if guidelines:
                        st.markdown('<div class="panel-eyebrow" style="margin-top:16px;">GUIDELINES RETRIEVED</div>', unsafe_allow_html=True)
                        for g in guidelines:
                            st.markdown(f'<div style="font-size:13px;color:#5C7A89;padding:4px 0;">'
                                        f'<strong style="color:#1C2B33;">{g["source"]}</strong> — {g["topic"]}</div>',
                                        unsafe_allow_html=True)
                except Exception:
                    pass

        # Sign-out button
        st.write("")
        if st.button("Sign out", use_container_width=True):
            audit_logger.log_logout(user.user_id, st.session_state.session_id)
            st.session_state.auth_token = ""
            st.session_state.auth_user  = None
            st.session_state.stage      = "idle"
            st.session_state.paused_state = None
            st.session_state.final_state  = None
            st.rerun()

    with right_col:
        if st.session_state.stage == "idle":
            st.markdown("""
            <div class="empty-state">
                <div class="mark">◍</div>
                <div class="heading">No audit running</div>
                <div class="sub">Enter a clinical question on the left. MIRA queries the patient
                database, cross-references medical guidelines, and prepares a report for your
                review before anything is finalized.</div>
            </div>""", unsafe_allow_html=True)

        elif st.session_state.stage == "awaiting_review":
            state = st.session_state.paused_state
            st.markdown(
                '<div class="panel-eyebrow with-dot" style="color:#B98A2E;">'
                '<span class="eyebrow-dot" style="background:#B98A2E;"></span>'
                'DRAFT — AWAITING YOUR REVIEW</div>', unsafe_allow_html=True
            )
            render_trend_chart(state.get("trend_data", ""))
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
                    placeholder="e.g. Be more specific about units, add urgency level.",
                    key="feedback_box", label_visibility="visible"
                )
                if st.button("Send revision request", type="primary"):
                    ph = st.empty()
                    with ph.container():
                        render_processing_strip("Sending feedback to reasoning agent")
                    final = engine.submit_human_decision(
                        st.session_state.thread_config, "reject", feedback_text,
                        user_id=user.user_id, hospital_id=user.hospital_id
                    )
                    ph.empty()
                    st.session_state.paused_state = final
                    st.session_state.show_feedback_box = False
                    st.rerun()

            if approve_clicked:
                ph = st.empty()
                with ph.container():
                    render_processing_strip("Running final safety check")
                final = engine.submit_human_decision(
                    st.session_state.thread_config, "approve",
                    user_id=user.user_id, hospital_id=user.hospital_id
                )
                ph.empty()
                st.session_state.final_state = final
                st.session_state.stage = "complete"
                st.rerun()

        elif st.session_state.stage == "complete":
            state = st.session_state.final_state
            if state.get("approved"):
                st.markdown('<div class="banner banner-approved"><span>✓</span>'
                            'Cleared by safety review — all claims grounded in retrieved data</div>',
                            unsafe_allow_html=True)
            else:
                flags = ", ".join(state.get("safety_flags", [])) or "Review recommended"
                st.markdown(f'<div class="banner banner-flagged"><span>▲</span>Flagged — {flags}</div>',
                            unsafe_allow_html=True)

            st.markdown('<div class="panel-eyebrow with-dot"><span class="eyebrow-dot"></span>FINAL CLINICAL REPORT</div>',
                        unsafe_allow_html=True)
            render_trend_chart(state.get("trend_data", ""))
            render_report_panel(state.get("final_report", "No report generated."))

            if st.button("Start a new audit"):
                st.session_state.stage = "idle"
                st.session_state.paused_state = None
                st.session_state.final_state  = None
                st.session_state.thread_config = None
                st.rerun()

    # ── Run handler ───────────────────────────────────────────────────────
    if run_clicked and question.strip():
        st.session_state.thread_config = engine.new_thread()
        st.session_state.stage = "running"

        with right_col:
            ph = st.empty()
            with ph.container():
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                render_processing_strip("Querying patient database")
                time.sleep(0.3)
                render_processing_strip("Cross-referencing clinical guidelines")
                st.markdown('</div>', unsafe_allow_html=True)

            paused = engine.run_until_review(
                question.strip(),
                st.session_state.thread_config,
                user_id=user.user_id,
                hospital_id=user.hospital_id,
                session_id=st.session_state.session_id,
            )
            ph.empty()

        st.session_state.paused_state = paused
        st.session_state.stage = "awaiting_review"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — ADMIN PANEL (admin role only)
# ══════════════════════════════════════════════════════════════════════════

if tab_admin and user.can("view_audit_log"):
    with tab_admin:
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        # ── Stats row ─────────────────────────────────────────────────────
        stats = audit_logger.get_stats(hospital_id=user.hospital_id)
        if stats:
            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-value">{int(stats.get('total_queries') or 0)}</div>
                    <div class="stat-label">Total audits run</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{int(stats.get('unique_users') or 0)}</div>
                    <div class="stat-label">Active clinicians</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{int(stats.get('total_reviews') or 0)}</div>
                    <div class="stat-label">Reports reviewed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{int(stats.get('avg_agent_ms') or 0)}ms</div>
                    <div class="stat-label">Avg agent time</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Audit log table ───────────────────────────────────────────────
        st.markdown('<div class="panel-eyebrow with-dot"><span class="eyebrow-dot"></span>AUDIT LOG</div>',
                    unsafe_allow_html=True)

        filter_col, _ = st.columns([1, 3])
        with filter_col:
            event_filter = st.selectbox(
                "Filter by event", label_visibility="collapsed",
                options=["All events", "query_submitted", "agent_run",
                         "human_review", "report_finalized", "error"]
            )

        logs = audit_logger.get_recent_logs(
            limit=50,
            event_type="" if event_filter == "All events" else event_filter
        )

        badge_class_map = {
            "query_submitted": "badge-query",
            "agent_run":       "badge-agent",
            "human_review":    "badge-review",
            "report_finalized":"badge-review",
            "error":           "badge-error",
            "user_login":      "badge-login",
            "user_logout":     "badge-login",
        }

        if logs:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            rows_html = ""
            for log in logs:
                badge_cls = badge_class_map.get(log["event_type"], "badge-login")
                ts = str(log.get("timestamp", ""))[:19].replace("T", " ")
                detail = log.get("action_detail") or log.get("agent_name") or "—"
                success_icon = "✓" if log.get("success") else "✗"
                success_color = "#2D8C7F" if log.get("success") else "#C9501F"
                rows_html += f"""
                <div class="audit-row">
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                                 color:#8FA3AE;min-width:140px;">{ts}</span>
                    <span class="audit-badge {badge_cls}">{log['event_type']}</span>
                    <span style="font-size:12.5px;color:#5C7A89;flex:1;">{detail}</span>
                    <span style="font-size:12px;color:#8FA3AE;">{log.get('user_id','—')}</span>
                    <span style="font-size:13px;color:{success_color};">{success_icon}</span>
                </div>"""
            st.markdown(rows_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("No audit events recorded yet.")