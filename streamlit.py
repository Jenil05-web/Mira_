"""
streamlit_app.py
=================
MIRA — Multi-Agent Clinical Audit & Real-Time Triage System
Frontend UI layer. All clinical logic lives in mira_pipeline.py.

Design: precision clinical instrument. Deep navy-charcoal foundation,
warm paper accents for content panels, a single emerald green reserved
for "live" and confirmed states, amber for patient-review holds,
and clay-rose reserved strictly for safety flags.

Typography trio: Instrument Serif for editorial headings (clinical journal),
Geist for all UI chrome (modern, legible), IBM Plex Mono for data identifiers.

Signature element: the "pulse rail" — a 2px horizontal track that fills
left-to-right as stages complete, with a soft glow on the active stage node.

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
    page_title="MIRA — Clinical Audit",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════
#
#  PALETTE
#  -------
#  --void        #0D1117   page foundation — deep charcoal, not pure black
#  --surface-0   #111820   sidebar / recessed panels
#  --surface-1   #161E28   card background
#  --surface-2   #1D2733   raised card / input background
#  --surface-3   #243040   hover / highlighted surface
#  --rim         #2A3A4A   hairline border, standard
#  --rim-2       #334455   slightly stronger border
#  --paper       #F5F2EC   warm paper — main content area
#  --paper-dim   #EDE9E2   subtler paper for nested panels
#  --ink         #1A2535   primary text on paper
#  --ink-2       #3D5060   secondary text on paper
#  --ink-3       #6B8090   tertiary / placeholder on paper
#  --ghost       #8FA8B8   muted text on dark surfaces
#  --ghost-2     #C8D8E0   readable text on dark surfaces
#  --white       #F0EDE8   max legibility text on dark
#
#  ACCENTS (rationed)
#  --emerald     #10B981   live / approved — one true action color
#  --emerald-dim #0A7D58   emerald pressed / deep
#  --emerald-bg  rgba(16,185,129,0.08)   tint on dark
#  --emerald-bg2 #EBF9F4   tint on paper
#  --amber       #F59E0B   review hold — yield state
#  --amber-dim   #B37308   amber deep
#  --amber-bg    rgba(245,158,11,0.10)   tint on dark
#  --clay        #E05C3A   safety flag — danger state
#  --clay-dim    #A83E23   clay deep
#  --clay-bg     rgba(224,92,58,0.10)    tint on dark
#  --clay-bg2    #FDF0EC   tint on paper
#
#  TYPOGRAPHY
#  --serif       'Instrument Serif'   clinical-journal register
#  --sans        'Geist'              clean instrument UI
#  --mono        'IBM Plex Mono'      data / identifiers / SQL
#
# ══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500;600&family=Geist:wght@300;400;500;600&display=swap');

/* ── TOKENS ── */
:root {
    --void:        #0D1117;
    --surface-0:   #111820;
    --surface-1:   #161E28;
    --surface-2:   #1D2733;
    --surface-3:   #243040;
    --rim:         #2A3A4A;
    --rim-2:       #334455;
    --paper:       #F5F2EC;
    --paper-dim:   #EDE9E2;
    --paper-rim:   #DDD8D0;
    --ink:         #1A2535;
    --ink-2:       #3D5060;
    --ink-3:       #6B8090;
    --ghost:       #8FA8B8;
    --ghost-2:     #C8D8E0;
    --white:       #F0EDE8;
    --emerald:     #10B981;
    --emerald-dim: #0A7D58;
    --emerald-bg:  rgba(16,185,129,0.08);
    --emerald-bg2: #EBF9F4;
    --emerald-rim: rgba(16,185,129,0.22);
    --amber:       #F59E0B;
    --amber-dim:   #B37308;
    --amber-bg:    rgba(245,158,11,0.10);
    --amber-rim:   rgba(245,158,11,0.28);
    --clay:        #E05C3A;
    --clay-dim:    #A83E23;
    --clay-bg:     rgba(224,92,58,0.10);
    --clay-bg2:    #FDF0EC;
    --clay-rim:    rgba(224,92,58,0.25);
}

/* ── RESET ── */
html, body, [class*="css"] {
    font-family: 'Geist', 'Inter', system-ui, sans-serif;
    color: var(--ghost-2);
}

.stApp { background: var(--void); }
header[data-testid="stHeader"] { background: transparent; border-bottom: none; }
.block-container { padding-top: 0 !important; max-width: 1320px; padding-left: 2rem !important; padding-right: 2rem !important; }
#MainMenu, footer { visibility: hidden; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--surface-0) !important;
    border-right: 1px solid var(--rim) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2.2rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

/* ── PAGE MASTHEAD ── */
.mira-masthead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 22px 0 22px 0;
    border-bottom: 1px solid var(--rim);
    margin-bottom: 28px;
}
.mira-brand { display: flex; align-items: center; gap: 14px; }
.mira-hex {
    width: 40px; height: 40px;
    background: var(--emerald);
    clip-path: polygon(50% 0%, 93% 25%, 93% 75%, 50% 100%, 7% 75%, 7% 25%);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    position: relative;
}
.mira-hex::after {
    content: '⬡';
    font-size: 22px;
    color: var(--void);
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    line-height: 1;
}
.mira-wordmark {
    font-family: 'Instrument Serif', serif;
    font-size: 21px;
    color: var(--white);
    letter-spacing: 0.04em;
    line-height: 1;
}
.mira-tagline {
    font-size: 11.5px;
    color: var(--ghost);
    letter-spacing: 0.01em;
    margin-top: 3px;
    font-weight: 400;
}

/* ── STATUS PILL ── */
.status-pill {
    display: flex; align-items: center; gap: 7px;
    padding: 6px 13px 6px 10px;
    border-radius: 100px;
    border: 1px solid var(--emerald-rim);
    background: var(--emerald-bg);
}
.status-pip {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--emerald);
    flex-shrink: 0;
    position: relative;
}
.status-pip::before {
    content: '';
    position: absolute;
    top: -4px; left: -4px;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: var(--emerald);
    opacity: 0;
    animation: pulse-ring 2.4s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
}
@keyframes pulse-ring {
    0%   { transform: scale(0.5); opacity: 0.5; }
    60%  { transform: scale(1.3); opacity: 0; }
    100% { transform: scale(1.3); opacity: 0; }
}
.status-text {
    font-size: 11.5px; font-weight: 500;
    color: var(--emerald);
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

/* ── PAPER ZONE — content right column ── */
.paper-zone {
    background: var(--paper);
    border-radius: 16px;
    border: 1px solid var(--paper-rim);
    overflow: hidden;
    min-height: 560px;
}

/* ── QUERY PANEL — left column ── */
.query-panel {
    background: var(--surface-1);
    border: 1px solid var(--rim);
    border-radius: 14px;
    padding: 22px 20px;
}

/* ── EYEBROW LABEL ── */
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ghost);
    font-weight: 500;
    margin-bottom: 13px;
    display: flex;
    align-items: center;
    gap: 7px;
}
.eyebrow-pip {
    width: 4px; height: 4px;
    border-radius: 50%;
    background: var(--emerald);
    flex-shrink: 0;
}
.eyebrow-amber { color: var(--amber); }
.eyebrow-pip-amber { background: var(--amber); }

/* ── PULSE RAIL ── */
.pulse-rail-wrap { margin-top: 22px; }
.pulse-rail-track {
    height: 2px;
    background: var(--rim);
    border-radius: 2px;
    position: relative;
    overflow: visible;
    margin-bottom: 18px;
}
.pulse-rail-fill {
    position: absolute;
    top: 0; left: 0; height: 100%;
    background: linear-gradient(90deg, var(--emerald-dim), var(--emerald));
    border-radius: 2px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 6px rgba(16,185,129,0.4);
}
.pulse-rail-nodes { display: flex; justify-content: space-between; align-items: flex-start; }
.pulse-node {
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    flex: 1;
    position: relative;
}
.pulse-node-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--rim-2);
    border: 1px solid var(--rim-2);
    flex-shrink: 0;
    margin-top: -3px;
    transition: all 0.4s ease;
    position: relative;
}
.pulse-node-dot.is-done {
    background: var(--emerald-dim);
    border-color: var(--emerald);
}
.pulse-node-dot.is-active {
    background: var(--emerald);
    border-color: var(--emerald);
    box-shadow: 0 0 10px rgba(16,185,129,0.6);
}
.pulse-node-dot.is-active::before {
    content: '';
    position: absolute;
    top: -5px; left: -5px;
    width: 18px; height: 18px;
    border-radius: 50%;
    background: var(--emerald);
    opacity: 0;
    animation: pulse-ring 1.8s ease-in-out infinite;
}
.pulse-node-dot.is-hold {
    background: var(--amber);
    border-color: var(--amber);
    box-shadow: 0 0 8px rgba(245,158,11,0.5);
}
.pulse-node-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--ghost);
    text-align: center;
    transition: color 0.4s ease;
    white-space: nowrap;
}
.pulse-node-label.is-active { color: var(--emerald); font-weight: 600; }
.pulse-node-label.is-done { color: var(--ghost-2); }
.pulse-node-label.is-hold { color: var(--amber); font-weight: 600; }

/* ── PROCESSING CARD ── */
.proc-card {
    background: var(--surface-2);
    border: 1px solid var(--rim);
    border-radius: 12px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 14px;
    position: relative;
    overflow: hidden;
}
.proc-card::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 1px;
    background: linear-gradient(90deg, transparent, var(--emerald), transparent);
    animation: scan-line 2s ease-in-out infinite;
}
@keyframes scan-line {
    0%   { left: -100%; opacity: 0; }
    20%  { opacity: 1; }
    80%  { opacity: 1; }
    100% { left: 100%; opacity: 0; }
}
.proc-icon {
    width: 28px; height: 28px;
    border-radius: 8px;
    background: var(--emerald-bg);
    border: 1px solid var(--emerald-rim);
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
    flex-shrink: 0;
}
.proc-text {
    font-size: 12.5px; font-weight: 500;
    color: var(--ghost-2);
    flex: 1;
}
.proc-wave {
    display: flex; gap: 3px; align-items: center;
}
.proc-wave span {
    display: block;
    width: 3px; height: 3px;
    border-radius: 50%;
    background: var(--emerald);
    animation: wave-dot 1.2s ease-in-out infinite;
}
.proc-wave span:nth-child(2) { animation-delay: 0.15s; }
.proc-wave span:nth-child(3) { animation-delay: 0.30s; }
.proc-wave span:nth-child(4) { animation-delay: 0.45s; }
@keyframes wave-dot {
    0%, 80%, 100% { transform: scaleY(1); opacity: 0.4; }
    40%           { transform: scaleY(2.2); opacity: 1; }
}

/* ── EXAMPLE CHIPS ── */
.chip-scroll { display: flex; flex-direction: column; gap: 5px; margin-top: 10px; }
.chip-item {
    display: block;
    padding: 8px 12px;
    background: var(--surface-2);
    border: 1px solid var(--rim);
    border-radius: 8px;
    font-size: 11.5px;
    color: var(--ghost);
    cursor: default;
    transition: all 0.15s ease;
    line-height: 1.4;
}
.chip-item:hover {
    background: var(--surface-3);
    border-color: var(--rim-2);
    color: var(--ghost-2);
}

/* ── SIDEBAR ELEMENTS ── */
.sidebar-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ghost);
    font-weight: 500;
    margin-bottom: 10px;
}
.audit-item {
    display: flex; align-items: flex-start; gap: 9px;
    padding: 10px 0;
    border-bottom: 1px solid var(--rim);
    cursor: default;
}
.audit-item:last-child { border-bottom: none; }
.audit-badge {
    width: 20px; height: 20px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px;
    flex-shrink: 0;
    margin-top: 1px;
}
.audit-badge-ok { background: var(--emerald-bg); color: var(--emerald); border: 1px solid var(--emerald-rim); }
.audit-badge-flag { background: var(--clay-bg); color: var(--clay); border: 1px solid var(--clay-rim); }
.audit-text { font-size: 11.5px; color: var(--ghost-2); line-height: 1.45; }
.audit-meta { font-size: 10px; color: var(--ghost); margin-top: 2px; }
.sidebar-info {
    font-size: 11.5px; color: var(--ghost);
    line-height: 1.65;
    padding: 12px 14px;
    background: var(--surface-2);
    border-radius: 10px;
    border: 1px solid var(--rim);
    margin-top: 4px;
}

/* ── PAPER CONTENT ── */
.paper-header {
    padding: 24px 28px 18px 28px;
    border-bottom: 1px solid var(--paper-rim);
}
.paper-body { padding: 26px 28px; }

/* ── REPORT TYPOGRAPHY (rendered on paper) ── */
.report-render {
    font-family: 'Geist', sans-serif;
    color: var(--ink);
}
.report-render h1, .report-render h2 {
    font-family: 'Instrument Serif', serif;
    font-weight: 400;
    color: var(--ink);
    margin-top: 26px;
    margin-bottom: 10px;
    padding-top: 20px;
    border-top: 1px solid var(--paper-rim);
}
.report-render h1 { font-size: 20px; }
.report-render h2 { font-size: 17px; }
.report-render h1:first-child, .report-render h2:first-child {
    margin-top: 0; padding-top: 0; border-top: none;
}
.report-render h3 {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--ink-2);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 20px; margin-bottom: 6px;
}
.report-render p, .report-render li {
    font-size: 14.5px; line-height: 1.72;
    color: var(--ink);
}
.report-render li { margin-bottom: 5px; }
.report-render strong { font-weight: 600; color: var(--ink); }
.report-render em { font-style: italic; color: var(--ink-2); }
.report-render code {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    background: var(--paper-dim);
    border: 1px solid var(--paper-rim);
    border-radius: 4px;
    padding: 1px 6px;
    color: var(--emerald-dim);
}
.report-render table {
    width: 100%; border-collapse: collapse;
    margin: 16px 0; font-size: 13.5px;
}
.report-render th {
    text-align: left; padding: 9px 12px;
    background: var(--paper-dim);
    border: 1px solid var(--paper-rim);
    font-weight: 600; color: var(--ink-2);
    font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.05em;
}
.report-render td {
    padding: 8px 12px;
    border: 1px solid var(--paper-rim);
    color: var(--ink);
}
.report-render tr:nth-child(even) td { background: var(--paper-dim); }

/* ── BANNERS ── */
.verdict-banner {
    display: flex; align-items: center; gap: 11px;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 20px;
    font-size: 13px; font-weight: 500;
    line-height: 1.45;
}
.verdict-approved {
    background: var(--emerald-bg2);
    border: 1px solid rgba(16,185,129,0.25);
    color: var(--emerald-dim);
}
.verdict-flagged {
    background: var(--clay-bg2);
    border: 1px solid rgba(224,92,58,0.25);
    color: var(--clay-dim);
}
.verdict-icon { font-size: 16px; flex-shrink: 0; }

/* ── REVIEW ACTIONS ── */
.review-action-row {
    display: flex; gap: 10px; margin-top: 22px;
    padding-top: 18px;
    border-top: 1px solid var(--paper-rim);
}

/* ── DATA SOURCE PANEL ── */
.data-panel {
    background: var(--surface-1);
    border: 1px solid var(--rim);
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 10px;
}
.data-panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ghost);
    font-weight: 500;
    margin-bottom: 8px;
}
.mono-display {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    line-height: 1.6;
    color: var(--ghost-2);
    white-space: pre-wrap;
    overflow-x: auto;
    background: var(--void);
    border: 1px solid var(--rim);
    border-radius: 7px;
    padding: 10px 13px;
}
.guideline-row {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 7px 0;
    border-bottom: 1px solid var(--rim);
    font-size: 12px;
}
.guideline-row:last-child { border-bottom: none; }
.guideline-source {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    color: var(--emerald);
    flex-shrink: 0;
    padding-top: 1px;
}
.guideline-topic { color: var(--ghost-2); line-height: 1.45; }

/* ── EMPTY STATE ── */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 460px;
    padding: 40px 30px;
    text-align: center;
}
.empty-hex {
    width: 56px; height: 56px;
    background: var(--paper-dim);
    clip-path: polygon(50% 0%, 93% 25%, 93% 75%, 50% 100%, 7% 75%, 7% 25%);
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 22px;
    position: relative;
}
.empty-hex::after {
    content: '⬡';
    font-size: 26px;
    color: var(--ink-3);
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    line-height: 1;
}
.empty-heading {
    font-family: 'Instrument Serif', serif;
    font-size: 22px; font-weight: 400;
    color: var(--ink); margin-bottom: 10px;
}
.empty-sub {
    font-size: 13.5px; color: var(--ink-3);
    max-width: 360px; margin: 0 auto;
    line-height: 1.7;
}
.empty-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 10px; margin-top: 30px; max-width: 480px;
}
.empty-card {
    background: var(--paper-dim);
    border: 1px solid var(--paper-rim);
    border-radius: 10px;
    padding: 12px 14px;
    text-align: left;
    font-size: 12px; color: var(--ink-3);
    line-height: 1.5;
}
.empty-card-icon { font-size: 15px; margin-bottom: 6px; display: block; }
.empty-card-label { font-weight: 600; color: var(--ink-2); font-size: 12px; }

/* ── BUTTONS ── */
.stButton button {
    font-family: 'Geist', sans-serif !important;
    border-radius: 9px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 0.52rem 1.1rem !important;
    border: 1px solid var(--rim-2) !important;
    background: var(--surface-2) !important;
    color: var(--ghost-2) !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    letter-spacing: 0.01em !important;
}
.stButton button:hover {
    border-color: var(--emerald-rim) !important;
    background: var(--surface-3) !important;
    color: var(--white) !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    background: var(--emerald-dim) !important;
    border: 1px solid var(--emerald) !important;
    color: #fff !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: var(--emerald) !important;
}

/* ── INPUTS ── */
.stTextArea textarea, .stTextInput input {
    font-family: 'Geist', sans-serif !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--rim) !important;
    color: var(--white) !important;
    border-radius: 10px !important;
    font-size: 14px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--emerald) !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.12) !important;
}
.stTextArea textarea::placeholder { color: var(--ghost) !important; }
div[data-testid="stTextArea"] label,
div[data-testid="stTextInput"] label { color: var(--ghost) !important; }

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--rim) !important;
    border-radius: 9px !important;
    color: var(--ghost) !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
.streamlit-expanderContent {
    background: var(--surface-1) !important;
    border: 1px solid var(--rim) !important;
    border-top: none !important;
    border-radius: 0 0 9px 9px !important;
}

/* ── MARKDOWN OVERRIDES (dark zone) ── */
.stMarkdown p, .stMarkdown li, .stMarkdown span,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span {
    color: var(--ghost-2) !important;
}
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    color: var(--white) !important;
    font-family: 'Instrument Serif', serif !important;
}
div[data-testid="stMarkdownContainer"] strong { color: var(--white) !important; }
div[data-testid="stMarkdownContainer"] code {
    color: var(--emerald) !important;
    background: var(--emerald-bg) !important;
}

hr { border-color: var(--rim) !important; }
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
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

if st.session_state.engine is None and st.session_state.engine_error is None:
    try:
        st.session_state.engine = get_engine()
    except Exception as e:
        st.session_state.engine_error = str(e)


# ══════════════════════════════════════════════════════════════════════════
# MASTHEAD
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="mira-masthead">
    <div class="mira-brand">
        <div class="mira-hex"></div>
        <div>
            <div class="mira-wordmark">MIRA</div>
            <div class="mira-tagline">Multi-agent clinical audit &amp; triage</div>
        </div>
    </div>
    <div class="status-pill">
        <div class="status-pip"></div>
        <div class="status-text">System active</div>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.engine_error:
    st.error(
        f"**Engine connection failed.**\n\n"
        f"`{st.session_state.engine_error}`\n\n"
        f"Run `01_data_setup.ipynb` first to build the SQLite database and FAISS index.",
        icon="⚠️"
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-label">Recent audits</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            '<div style="font-size:12px; color:#8FA8B8; line-height:1.65; padding:10px 0;">'
            'Audits you run this session will appear here.</div>',
            unsafe_allow_html=True
        )
    else:
        items_html = ""
        for entry in reversed(st.session_state.history):
            label = entry["question"][:52] + ("…" if len(entry["question"]) > 52 else "")
            ok = entry["approved"]
            badge_cls = "audit-badge-ok" if ok else "audit-badge-flag"
            symbol = "✓" if ok else "▲"
            items_html += (
                f'<div class="audit-item">'
                f'<span class="audit-badge {badge_cls}">{symbol}</span>'
                f'<div><div class="audit-text">{label}</div></div></div>'
            )
        st.markdown(items_html, unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:var(--rim);margin:20px 0 16px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">About this system</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-info">'
        'Every finding is grounded in a live database query and cross-checked against '
        'medical guidelines. Reports pass an automated safety review before reaching you — '
        'and nothing is finalized without your sign-off.'
        '</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════
# PULSE RAIL RENDERER
# ══════════════════════════════════════════════════════════════════════════

def render_pulse_rail(stage: str):
    nodes = [
        ("sql",    "Query DB"),
        ("search", "Guidelines"),
        ("reason", "Analysis"),
        ("review", "Your review"),
        ("check",  "Safety check"),
    ]

    stage_index = {
        "idle": -1,
        "running_sql": 0,
        "running_search": 1,
        "running_reasoning": 2,
        "awaiting_review": 3,
        "revising": 2,
        "running_critic": 4,
        "complete": 5,
    }
    idx = stage_index.get(stage, -1)
    total = len(nodes)

    if stage == "idle":
        fill_pct = 0
    elif stage == "complete":
        fill_pct = 100
    else:
        fill_pct = (idx / total) * 100 + (100 / total) * 0.45

    nodes_html = ""
    for i, (key, label) in enumerate(nodes):
        if stage == "awaiting_review" and key == "review":
            dot_cls = label_cls = "is-hold"
        elif i < idx:
            dot_cls = label_cls = "is-done"
        elif i == idx:
            dot_cls = label_cls = "is-active"
        else:
            dot_cls = label_cls = ""

        nodes_html += (
            f'<div class="pulse-node">'
            f'<div class="pulse-node-dot {dot_cls}"></div>'
            f'<div class="pulse-node-label {label_cls}">{label}</div>'
            f'</div>'
        )

    st.markdown(f"""
    <div class="pulse-rail-wrap">
        <div class="pulse-rail-track">
            <div class="pulse-rail-fill" style="width:{fill_pct}%;"></div>
        </div>
        <div class="pulse-rail-nodes">{nodes_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_proc_card(icon: str, message: str):
    st.markdown(f"""
    <div class="proc-card">
        <div class="proc-icon">{icon}</div>
        <span class="proc-text">{message}</span>
        <div class="proc-wave">
            <span></span><span></span><span></span><span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_report(markdown_text: str, on_paper: bool = True):
    html_body = md_lib.markdown(markdown_text, extensions=["extra", "tables"])
    wrapper_cls = "report-render"
    if on_paper:
        st.markdown(
            f'<div class="{wrapper_cls}">{html_body}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="{wrapper_cls}">{html_body}</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════

EXAMPLES = [
    "Which patients have critical lab results?",
    "Find patients with abnormal creatinine — signs of AKI?",
    "Are there signs of sepsis in recent admissions?",
]

left_col, right_col = st.columns([1, 1.7], gap="large")

# ── LEFT COLUMN — Query input ──────────────────────────────────────────

with left_col:
    st.markdown('<div class="query-panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="eyebrow"><span class="eyebrow-pip"></span>Clinical query</div>',
        unsafe_allow_html=True
    )

    question = st.text_area(
        label="Clinical query",
        placeholder="e.g. Which patients show signs of acute kidney injury based on their latest labs?",
        height=110,
        key="question_input",
        label_visibility="collapsed",
        disabled=(st.session_state.stage in ["running", "awaiting_review"])
    )

    # Example chips
    chips_html = '<div class="chip-scroll">'
    for ex in EXAMPLES:
        chips_html += f'<div class="chip-item">{ex}</div>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    run_disabled = (
        st.session_state.stage in ["running", "awaiting_review"]
        or not question.strip()
    )
    run_clicked = st.button(
        "Run clinical audit →",
        type="primary",
        use_container_width=True,
        disabled=run_disabled
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Pulse rail
    rail_stage = (
        "running_sql" if st.session_state.stage == "running" else
        "awaiting_review" if st.session_state.stage == "awaiting_review" else
        "complete" if st.session_state.stage == "complete" else
        "idle"
    )
    render_pulse_rail(rail_stage)

    # Data sources expander
    if st.session_state.paused_state or st.session_state.final_state:
        active = st.session_state.final_state or st.session_state.paused_state
        with st.expander("View data sources"):
            st.markdown('<div class="data-panel-label">SQL query executed</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="mono-display">{active.get("sql_query_used", "—")}</div>',
                unsafe_allow_html=True
            )
            st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="data-panel-label">Guideline search query</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="mono-display">{active.get("search_query_used", "—")}</div>',
                unsafe_allow_html=True
            )
            try:
                guidelines = json.loads(active.get("guidelines", "{}")).get("guidelines", [])
                if guidelines:
                    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="data-panel-label">Guidelines retrieved</div>', unsafe_allow_html=True)
                    rows_html = ""
                    for g in guidelines:
                        rows_html += (
                            f'<div class="guideline-row">'
                            f'<span class="guideline-source">{g["source"]}</span>'
                            f'<span class="guideline-topic">{g["topic"]}</span>'
                            f'</div>'
                        )
                    st.markdown(rows_html, unsafe_allow_html=True)
            except Exception:
                pass


# ── RIGHT COLUMN — Result area ─────────────────────────────────────────

with right_col:

    # ── IDLE ──────────────────────────────────────────────────────────
    if st.session_state.stage == "idle":
        st.markdown("""
        <div style="background:var(--paper); border-radius:16px; border:1px solid var(--paper-rim); min-height:560px;">
          <div class="empty-state">
            <div class="empty-hex"></div>
            <div class="empty-heading">No audit running</div>
            <div class="empty-sub">
              Enter a clinical question on the left. MIRA queries the patient database,
              cross-references medical guidelines, and prepares a report for your
              review before anything is finalized.
            </div>
            <div class="empty-grid">
              <div class="empty-card">
                <span class="empty-card-icon">🔬</span>
                <span class="empty-card-label">Lab results</span><br>
                Abnormal values, critical ranges
              </div>
              <div class="empty-card">
                <span class="empty-card-icon">🫀</span>
                <span class="empty-card-label">Triage flags</span><br>
                Sepsis, AKI, acute conditions
              </div>
              <div class="empty-card">
                <span class="empty-card-icon">📋</span>
                <span class="empty-card-label">Guideline check</span><br>
                Cross-reference protocols
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── AWAITING REVIEW ───────────────────────────────────────────────
    elif st.session_state.stage == "awaiting_review":
        state = st.session_state.paused_state

        st.markdown(
            '<div style="background:var(--paper); border-radius:16px; border:1px solid var(--paper-rim);">',
            unsafe_allow_html=True
        )

        # Paper header — amber hold indicator
        st.markdown("""
        <div class="paper-header">
          <div class="eyebrow eyebrow-amber">
            <span class="eyebrow-pip eyebrow-pip-amber"></span>
            Draft report &mdash; awaiting your review
          </div>
          <div style="font-family:'Instrument Serif',serif; font-size:19px; color:var(--ink); margin-top:4px;">
            Review before finalizing
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="paper-body">', unsafe_allow_html=True)
        render_report(state["clinical_reasoning"], on_paper=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Action row — rendered inside the paper zone
        st.markdown('<div style="padding:0 28px 24px 28px; border-top:1px solid var(--paper-rim);">', unsafe_allow_html=True)
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

        approve_col, reject_col = st.columns([1, 1])
        with approve_col:
            approve_clicked = st.button("Approve & finalize", type="primary", use_container_width=True)
        with reject_col:
            reject_clicked = st.button("Request revision", use_container_width=True)

        if reject_clicked:
            st.session_state.show_feedback_box = True

        if st.session_state.show_feedback_box:
            st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
            feedback_text = st.text_area(
                "What should the analysis address?",
                placeholder="e.g. Be more specific about units, or add urgency level per finding.",
                key="feedback_box",
            )
            submit_feedback = st.button("Send revision request", type="primary")
            if submit_feedback and feedback_text.strip():
                fb_slot = st.empty()
                with fb_slot.container():
                    render_proc_card("↻", "Sending feedback to the clinical reasoning agent")
                final = st.session_state.engine.submit_human_decision(
                    st.session_state.thread_config, "reject", feedback_text
                )
                fb_slot.empty()
                st.session_state.paused_state = final
                st.session_state.show_feedback_box = False
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if approve_clicked:
            approve_slot = st.empty()
            with approve_slot.container():
                render_proc_card("🛡", "Running final safety check")
            final = st.session_state.engine.submit_human_decision(
                st.session_state.thread_config, "approve"
            )
            approve_slot.empty()
            st.session_state.final_state = final
            st.session_state.stage = "complete"
            st.session_state.history.append({
                "question": final["clinical_question"],
                "approved": final.get("approved", True)
            })
            st.rerun()

    # ── COMPLETE ──────────────────────────────────────────────────────
    elif st.session_state.stage == "complete":
        state = st.session_state.final_state
        approved = state.get("approved", True)

        st.markdown(
            '<div style="background:var(--paper); border-radius:16px; border:1px solid var(--paper-rim);">',
            unsafe_allow_html=True
        )

        st.markdown('<div class="paper-header">', unsafe_allow_html=True)

        if approved:
            st.markdown("""
            <div class="verdict-banner verdict-approved">
                <span class="verdict-icon">✓</span>
                <span>Cleared by safety review — every claim is grounded in retrieved data.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            flags = ", ".join(state.get("safety_flags", [])) or "Review recommended before clinical use"
            st.markdown(f"""
            <div class="verdict-banner verdict-flagged">
                <span class="verdict-icon">▲</span>
                <span>Flagged by safety review — {flags}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            '<div class="eyebrow"><span class="eyebrow-pip"></span>Final clinical report</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)  # end paper-header

        st.markdown('<div class="paper-body">', unsafe_allow_html=True)
        render_report(state.get("final_report", "No report generated."), on_paper=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="padding:0 28px 22px 28px;">', unsafe_allow_html=True)
        if st.button("← Start a new audit"):
            st.session_state.stage = "idle"
            st.session_state.paused_state = None
            st.session_state.final_state = None
            st.session_state.thread_config = None
            st.rerun()
        st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# RUN HANDLER
# ══════════════════════════════════════════════════════════════════════════

if run_clicked and question.strip():
    st.session_state.thread_config = st.session_state.engine.new_thread()
    st.session_state.stage = "running"

    with right_col:
        proc_slot = st.empty()
        with proc_slot.container():
            st.markdown(
                '<div style="background:var(--paper);border-radius:16px;border:1px solid var(--paper-rim);padding:24px 28px;min-height:180px;">',
                unsafe_allow_html=True
            )
            render_proc_card("⬡", "Querying patient database")
            time.sleep(0.4)
            render_proc_card("⊕", "Cross-referencing clinical guidelines")
            st.markdown('</div>', unsafe_allow_html=True)

        paused = st.session_state.engine.run_until_review(
            question.strip(), st.session_state.thread_config
        )
        proc_slot.empty()

    st.session_state.paused_state = paused
    st.session_state.stage = "awaiting_review"
    st.rerun()