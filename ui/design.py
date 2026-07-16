import streamlit as st
import json
import time
import markdown as md_lib
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
.banner-info     { background: var(--gold-tint); border: 1px solid rgba(185,138,46,0.25); color: var(--gold); }

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

/* VOICE / MIC PANEL */
.mic-panel {
    background: linear-gradient(145deg, var(--surface) 0%, var(--teal-tint) 130%);
    border: 1px solid var(--line); border-radius: 14px;
    padding: 18px 20px; margin-top: 4px;
}
.mic-panel-head { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.mic-icon-badge {
    width: 34px; height: 34px; border-radius: 10px; background: var(--ink);
    display: flex; align-items: center; justify-content: center; font-size: 15px;
    flex-shrink: 0;
}
.mic-panel-title { font-size: 13.5px; font-weight: 600; color: var(--ink); }
.mic-panel-sub { font-size: 12px; color: var(--slate); margin-top: 1px; }
.mic-result-chip {
    display: inline-flex; align-items: center; gap: 7px; margin-top: 12px;
    background: var(--teal-tint); border: 1px solid rgba(45,140,127,0.22);
    border-radius: 100px; padding: 6px 13px 6px 11px; font-size: 12px; color: var(--teal-deep);
}
.mic-result-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--teal); flex-shrink: 0; }
.mic-transcript-box {
    margin-top: 8px; font-size: 12.5px; color: var(--slate); line-height: 1.6;
    background: var(--line-soft); border: 1px solid var(--line); border-radius: 8px;
    padding: 10px 13px;
}

hr { border-color: var(--line) !important; }
</style>
"""

def apply_design():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


