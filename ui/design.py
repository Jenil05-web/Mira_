import streamlit as st
import json
import time
import markdown as md_lib

# ══════════════════════════════════════════════════════════════════════════
# MIRA — PREMIUM DESIGN SYSTEM v2
# Glassmorphism · Micro-animations · Smooth transitions · Elegant typography
# ══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,300;6..72,400;6..72,500;6..72,600&family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ── TOKENS ─────────────────────────────────────────────────────────── */
:root {
    --paper:       #F8F9FA;
    --surface:     #FFFFFF;
    --glass:       rgba(255,255,255,0.72);
    --ink:         #0F1C24;
    --ink-soft:    #1C2B33;
    --slate:       #4A6778;
    --slate-2:     #7A96A4;
    --teal:        #1E8A7A;
    --teal-light:  #2DA896;
    --teal-tint:   #E6F5F3;
    --teal-deep:   #145E52;
    --teal-glow:   rgba(30,138,122,0.18);
    --clay:        #C04B22;
    --clay-tint:   #FCEEE6;
    --line:        #E2E6E9;
    --line-soft:   #EDF0F2;
    --gold:        #A97C2B;
    --gold-tint:   #FAF0DE;
    --shadow-sm:   0 1px 3px rgba(15,28,36,0.06), 0 1px 2px rgba(15,28,36,0.04);
    --shadow-md:   0 4px 12px rgba(15,28,36,0.08), 0 2px 4px rgba(15,28,36,0.04);
    --shadow-lg:   0 12px 32px rgba(15,28,36,0.10), 0 4px 8px rgba(15,28,36,0.06);
    --radius-sm:   8px;
    --radius-md:   12px;
    --radius-lg:   16px;
    --ease-out:    cubic-bezier(0.16, 1, 0.3, 1);
    --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* ── BASE ────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--paper); }
.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 40% at 20% 0%, rgba(30,138,122,0.055) 0%, transparent 70%),
        radial-gradient(ellipse 60% 30% at 80% 100%, rgba(21,94,82,0.04) 0%, transparent 60%);
    pointer-events: none; z-index: 0;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1300px; position: relative; z-index: 1; }
#MainMenu, footer { visibility: hidden; }

/* ── HEADER ──────────────────────────────────────────────────────────── */
.mira-header {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 20px; border-bottom: 1px solid var(--line); margin-bottom: 28px;
}
.mira-header-left { display: flex; align-items: center; gap: 14px; }
.mira-mark {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(145deg, #0F1C24 0%, #1C3040 100%);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Newsreader', serif; font-weight: 600; font-size: 21px; color: #F8F9FA;
    box-shadow: var(--shadow-md), 0 0 0 1px rgba(255,255,255,0.08) inset;
    transition: transform 0.3s var(--ease-spring), box-shadow 0.3s ease;
}
.mira-mark:hover {
    transform: scale(1.06) rotate(-2deg);
    box-shadow: var(--shadow-lg), 0 0 0 1px rgba(255,255,255,0.1) inset;
}
.mira-title {
    font-family: 'Newsreader', serif; font-size: 22px; font-weight: 500;
    color: var(--ink); line-height: 1.1; letter-spacing: -0.01em;
}
.mira-subtitle { font-size: 12.5px; color: var(--slate-2); margin-top: 2px; letter-spacing: 0.01em; }
.header-right { display: flex; align-items: center; gap: 10px; }

/* ── LIVE PILL ──────────────────────────────────────────────────────── */
.live-pill {
    display: flex; align-items: center; gap: 8px;
    background: var(--teal-tint); border: 1px solid rgba(30,138,122,0.2);
    border-radius: 100px; padding: 7px 15px 7px 11px;
    box-shadow: 0 0 0 3px rgba(30,138,122,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.live-pill:hover { transform: translateY(-1px); box-shadow: 0 0 0 4px rgba(30,138,122,0.10); }
.live-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--teal);
    position: relative; flex-shrink: 0;
}
.live-dot::before {
    content: ''; position: absolute; top: -4px; left: -4px;
    width: 15px; height: 15px; border-radius: 50%;
    background: var(--teal); opacity: 0.3;
    animation: breathe 2.4s ease-in-out infinite;
}
@keyframes breathe {
    0%   { transform: scale(0.5); opacity: 0.5; }
    50%  { transform: scale(1.3); opacity: 0.05; }
    100% { transform: scale(0.5); opacity: 0.5; }
}
.live-pill-text { font-size: 12px; font-weight: 600; color: var(--teal-deep); letter-spacing: 0.01em; }

/* ── USER BADGE ──────────────────────────────────────────────────────── */
.user-badge {
    display: flex; align-items: center; gap: 8px;
    background: var(--glass); backdrop-filter: blur(8px);
    border: 1px solid var(--line); border-radius: 100px;
    padding: 6px 14px 6px 10px; font-size: 12.5px; color: var(--slate);
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}
.user-badge:hover { border-color: var(--slate-2); transform: translateY(-1px); }
.role-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.role-dot.clinician { background: var(--teal); }
.role-dot.admin { background: var(--gold); }

/* ── MIRA LOADER (main loading animation) ───────────────────────────── */
.mira-loader-wrap {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 80px 40px; gap: 32px;
}
.mira-loader-logo {
    width: 72px; height: 72px; border-radius: 20px;
    background: linear-gradient(145deg, #0F1C24, #1E8A7A);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Newsreader', serif; font-weight: 600; font-size: 32px; color: #F8F9FA;
    box-shadow: 0 0 0 0 rgba(30,138,122,0.5);
    animation: logo-pulse 2s ease-in-out infinite;
    position: relative;
}
.mira-loader-logo::after {
    content: '';
    position: absolute; inset: -8px; border-radius: 28px;
    background: linear-gradient(135deg, rgba(30,138,122,0.3), rgba(14,94,82,0.1));
    animation: ring-spin 3s linear infinite;
    border: 1.5px solid rgba(30,138,122,0.35);
}
@keyframes logo-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(30,138,122,0.4), 0 8px 24px rgba(30,138,122,0.2); transform: scale(1); }
    50%       { box-shadow: 0 0 0 12px rgba(30,138,122,0), 0 8px 32px rgba(30,138,122,0.35); transform: scale(1.04); }
}
@keyframes ring-spin {
    0%   { transform: rotate(0deg) scale(1); opacity: 0.8; }
    50%  { transform: rotate(180deg) scale(1.05); opacity: 0.4; }
    100% { transform: rotate(360deg) scale(1); opacity: 0.8; }
}
.mira-loader-label {
    font-family: 'Newsreader', serif; font-size: 20px; font-weight: 500;
    color: var(--ink); text-align: center; letter-spacing: -0.01em;
}
.mira-loader-sub {
    font-size: 13px; color: var(--slate-2); text-align: center;
    margin-top: -22px; letter-spacing: 0.01em;
}
.mira-loader-steps {
    display: flex; flex-direction: column; gap: 10px; width: 100%; max-width: 320px;
}
.mira-loader-step {
    display: flex; align-items: center; gap: 12px;
    padding: 11px 16px; border-radius: var(--radius-md);
    background: var(--surface); border: 1px solid var(--line);
    box-shadow: var(--shadow-sm);
    animation: step-in 0.4s var(--ease-out) both;
}
.mira-loader-step:nth-child(1) { animation-delay: 0.0s; }
.mira-loader-step:nth-child(2) { animation-delay: 0.15s; }
.mira-loader-step:nth-child(3) { animation-delay: 0.3s; }
.mira-loader-step:nth-child(4) { animation-delay: 0.45s; }
@keyframes step-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.loader-step-icon {
    width: 28px; height: 28px; border-radius: 8px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 13px;
}
.loader-step-icon.active { background: var(--teal-tint); }
.loader-step-icon.waiting { background: var(--line-soft); }
.loader-step-icon.done { background: #E6F5EF; }
.loader-step-text { font-size: 13px; font-weight: 500; color: var(--ink); }
.loader-step-status { font-size: 11px; color: var(--slate-2); margin-top: 1px; }
.loader-step-active .loader-step-text { color: var(--teal-deep); }
.loader-step-active .loader-step-status { color: var(--teal); }

/* Bouncing dots for active step */
.dot-bounce {
    display: inline-flex; gap: 3px; align-items: center; height: 14px;
}
.dot-bounce span {
    width: 4px; height: 4px; border-radius: 50%; background: var(--teal);
    animation: bounce-dot 1.2s ease-in-out infinite;
}
.dot-bounce span:nth-child(2) { animation-delay: 0.2s; }
.dot-bounce span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce-dot {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40%            { transform: translateY(-5px); opacity: 1; }
}

/* ── FILL RAIL ───────────────────────────────────────────────────────── */
.fill-rail-wrap { margin-bottom: 28px; }
.fill-rail-track {
    position: relative; height: 2px; background: var(--line);
    border-radius: 3px; overflow: hidden; margin-bottom: 16px;
}
.fill-rail-progress {
    position: absolute; top: 0; left: 0; height: 100%;
    background: linear-gradient(90deg, var(--teal-deep), var(--teal-light));
    border-radius: 3px; transition: width 0.8s var(--ease-out);
    box-shadow: 0 0 8px rgba(30,138,122,0.4);
}
.fill-rail-labels { display: flex; justify-content: space-between; }
.fill-rail-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 0.06em; color: var(--slate-2); text-transform: uppercase;
    display: flex; align-items: center; gap: 5px;
    transition: color 0.4s ease, font-weight 0.2s ease;
}
.fill-rail-label.is-active { color: var(--teal-deep); font-weight: 700; }
.fill-rail-label.is-done { color: var(--teal); }
.fill-rail-label.is-waiting { color: var(--gold); font-weight: 700; }
.fill-rail-icon { width: 5px; height: 5px; border-radius: 50%; background: var(--line); flex-shrink: 0; transition: background 0.4s ease; }
.fill-rail-icon.is-active { background: var(--teal); box-shadow: 0 0 4px var(--teal-glow); }
.fill-rail-icon.is-done { background: var(--teal-deep); }
.fill-rail-icon.is-waiting { background: var(--gold); }

/* ── PROCESSING STRIP ────────────────────────────────────────────────── */
.processing-strip {
    display: flex; align-items: center; gap: 12px; padding: 14px 18px;
    background: linear-gradient(90deg, var(--teal-tint) 0%, rgba(230,245,243,0.4) 100%);
    border: 1px solid rgba(30,138,122,0.2); border-radius: var(--radius-md);
    margin: 14px 0; box-shadow: 0 2px 8px rgba(30,138,122,0.08);
    animation: strip-in 0.35s var(--ease-out);
}
@keyframes strip-in {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: translateY(0); }
}
.processing-strip-icon {
    width: 32px; height: 32px; border-radius: 10px;
    background: linear-gradient(145deg, #0F1C24, #1E8A7A);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
    animation: icon-breathe 2s ease-in-out infinite;
}
@keyframes icon-breathe {
    0%, 100% { opacity: 1; } 50% { opacity: 0.7; }
}
.processing-text { font-size: 13px; color: var(--teal-deep); font-weight: 600; flex: 1; }
.processing-sub  { font-size: 11.5px; color: var(--slate-2); margin-top: 1px; }
.shimmer-bar {
    position: relative; flex: 1; height: 2px;
    background: rgba(30,138,122,0.15); border-radius: 2px; overflow: hidden; min-width: 60px;
}
.shimmer-bar::after {
    content: ''; position: absolute; top: 0; left: -40%; width: 40%; height: 100%;
    background: linear-gradient(90deg, transparent, var(--teal-light), transparent);
    animation: shimmer 1.3s ease-in-out infinite;
}
@keyframes shimmer { 0% { left: -40%; } 100% { left: 100%; } }

/* ── PANELS ──────────────────────────────────────────────────────────── */
.panel {
    background: var(--glass); backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.8); border-bottom-color: var(--line);
    border-radius: var(--radius-lg); padding: 22px 24px; margin-bottom: 16px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.3s ease, transform 0.2s var(--ease-spring);
}
.panel:hover { box-shadow: var(--shadow-md); }
.panel-eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--slate-2);
    margin-bottom: 12px; font-weight: 600;
}
.panel-eyebrow.with-dot { display: flex; align-items: center; gap: 7px; }
.eyebrow-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--teal); }

/* ── REPORT TYPOGRAPHY ───────────────────────────────────────────────── */
.report-surface h2 {
    font-family: 'Newsreader', serif; font-size: 17px; font-weight: 600; color: var(--ink);
    margin-top: 24px; margin-bottom: 10px;
    padding-top: 20px; border-top: 1px solid var(--line-soft);
    letter-spacing: -0.01em;
}
.report-surface h2:first-child { margin-top: 0; padding-top: 0; border-top: none; }
.report-surface p, .report-surface li {
    font-size: 14.5px; line-height: 1.75; color: var(--ink-soft);
}
.report-surface strong { color: var(--ink); font-weight: 600; }

/* ── FORCE TEXT COLOR ────────────────────────────────────────────────── */
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

/* ── BANNERS ─────────────────────────────────────────────────────────── */
.banner {
    display: flex; align-items: center; gap: 10px; border-radius: var(--radius-md);
    padding: 13px 16px; margin-bottom: 16px; font-size: 13px; font-weight: 500;
    animation: banner-in 0.4s var(--ease-out);
}
@keyframes banner-in {
    from { opacity: 0; transform: translateY(-6px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.banner-approved { background: var(--teal-tint); border: 1px solid rgba(30,138,122,0.22); color: var(--teal-deep); box-shadow: 0 2px 8px rgba(30,138,122,0.1); }
.banner-flagged  { background: var(--clay-tint); border: 1px solid rgba(192,75,34,0.2); color: var(--clay); }
.banner-info     { background: var(--gold-tint); border: 1px solid rgba(169,124,43,0.2); color: var(--gold); }

/* ── MONO BLOCK ──────────────────────────────────────────────────────── */
.mono-block {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; line-height: 1.6;
    background: var(--line-soft); border: 1px solid var(--line); border-radius: var(--radius-sm);
    padding: 14px 16px; color: var(--slate); white-space: pre-wrap; overflow-x: auto;
}

/* ── BUTTONS ─────────────────────────────────────────────────────────── */
.stButton button {
    border-radius: 10px !important; font-weight: 500 !important; font-size: 13.5px !important;
    padding: 0.58rem 1.2rem !important; border: 1px solid var(--line) !important;
    background: var(--surface) !important; color: var(--ink) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s var(--ease-out) !important;
    position: relative !important; overflow: hidden !important;
}
.stButton button::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.5), transparent);
    opacity: 1; pointer-events: none;
}
.stButton button:hover {
    border-color: var(--slate-2) !important;
    background: var(--line-soft) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}
.stButton button:active { transform: translateY(0px) !important; box-shadow: var(--shadow-sm) !important; }
.stButton button p, .stButton button span, .stButton button div { color: inherit !important; }

/* Primary button */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #0F1C24 0%, #1C3040 100%) !important;
    border: 1px solid #0F1C24 !important; color: #F8F9FA !important;
    box-shadow: 0 4px 14px rgba(15,28,36,0.25), 0 1px 3px rgba(15,28,36,0.15) !important;
}
div[data-testid="stButton"] button[kind="primary"] p,
div[data-testid="stButton"] button[kind="primary"] span { color: #F8F9FA !important; }
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1C3040 0%, #0F1C24 100%) !important;
    box-shadow: 0 6px 20px rgba(15,28,36,0.3), 0 2px 6px rgba(15,28,36,0.15) !important;
    transform: translateY(-2px) !important;
}

/* ── INPUTS ──────────────────────────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    background: var(--surface) !important; border: 1px solid var(--line) !important;
    color: var(--ink) !important; border-radius: var(--radius-md) !important;
    font-size: 14.5px !important; box-shadow: var(--shadow-sm) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px var(--teal-glow), var(--shadow-sm) !important;
    outline: none !important;
}

/* ── CHIPS ───────────────────────────────────────────────────────────── */
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.chip {
    display: inline-flex; align-items: center;
    background: var(--glass); backdrop-filter: blur(6px);
    border: 1px solid var(--line); border-radius: 100px;
    padding: 5px 12px; font-size: 11.5px; color: var(--slate);
    transition: all 0.2s ease; box-shadow: var(--shadow-sm);
}
.chip:hover { border-color: var(--teal); color: var(--teal-deep); transform: translateY(-1px); }

/* ── EMPTY STATE ─────────────────────────────────────────────────────── */
.empty-state { text-align: center; padding: 80px 30px; }
.empty-state .mark {
    width: 56px; height: 56px; border-radius: 16px;
    background: linear-gradient(145deg, var(--line-soft), var(--surface));
    border: 1px solid var(--line);
    display: flex; align-items: center; justify-content: center; margin: 0 auto 20px auto;
    font-family: 'Newsreader', serif; font-size: 24px; color: var(--slate-2);
    box-shadow: var(--shadow-sm);
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-5px); }
}
.empty-state .heading {
    font-family: 'Newsreader', serif; font-size: 20px; font-weight: 500;
    color: var(--ink); margin-bottom: 10px; letter-spacing: -0.01em;
}
.empty-state .sub {
    font-size: 13.5px; color: var(--slate-2); max-width: 360px;
    margin: 0 auto; line-height: 1.7;
}

/* ── ADMIN TABLE ─────────────────────────────────────────────────────── */
.audit-row {
    display: flex; align-items: center; gap: 12px; padding: 10px 0;
    border-bottom: 1px solid var(--line-soft); font-size: 12.5px;
    transition: background 0.15s ease; border-radius: 4px;
}
.audit-row:last-child { border-bottom: none; }
.audit-badge {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    padding: 3px 9px; border-radius: 100px; font-weight: 600; letter-spacing: 0.02em;
}
.badge-query  { background: var(--teal-tint); color: var(--teal-deep); }
.badge-agent  { background: var(--gold-tint); color: var(--gold); }
.badge-review { background: var(--teal-tint); color: var(--teal-deep); }
.badge-error  { background: var(--clay-tint); color: var(--clay); }
.badge-login  { background: var(--line-soft); color: var(--slate); }

/* ── STAT CARDS ──────────────────────────────────────────────────────── */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
.stat-card {
    background: var(--glass); backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.8); border-bottom-color: var(--line);
    border-radius: var(--radius-md); padding: 18px 20px;
    box-shadow: var(--shadow-sm);
    transition: all 0.25s var(--ease-out);
}
.stat-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.stat-value {
    font-family: 'Newsreader', serif; font-size: 30px; font-weight: 500;
    color: var(--ink); letter-spacing: -0.02em;
}
.stat-label { font-size: 11.5px; color: var(--slate-2); margin-top: 4px; letter-spacing: 0.01em; }

/* ── VOICE / MIC PANEL ───────────────────────────────────────────────── */
.mic-panel {
    background: linear-gradient(145deg, var(--surface) 0%, var(--teal-tint) 150%);
    border: 1px solid rgba(30,138,122,0.18); border-radius: var(--radius-lg);
    padding: 18px 20px; margin-top: 4px; box-shadow: var(--shadow-sm);
    transition: box-shadow 0.3s ease;
}
.mic-panel:hover { box-shadow: var(--shadow-md); }
.mic-panel-head { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.mic-icon-badge {
    width: 36px; height: 36px; border-radius: 11px;
    background: linear-gradient(145deg, #0F1C24, #1E8A7A);
    display: flex; align-items: center; justify-content: center; font-size: 16px;
    flex-shrink: 0; box-shadow: 0 4px 10px rgba(30,138,122,0.25);
}
.mic-panel-title { font-size: 13.5px; font-weight: 600; color: var(--ink); }
.mic-panel-sub { font-size: 12px; color: var(--slate-2); margin-top: 1px; }
.mic-result-chip {
    display: inline-flex; align-items: center; gap: 7px; margin-top: 12px;
    background: var(--teal-tint); border: 1px solid rgba(30,138,122,0.22);
    border-radius: 100px; padding: 6px 13px 6px 11px; font-size: 12px; color: var(--teal-deep);
    font-weight: 500;
}
.mic-result-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--teal); flex-shrink: 0; }
.mic-transcript-box {
    margin-top: 8px; font-size: 13px; color: var(--slate); line-height: 1.65;
    background: rgba(255,255,255,0.6); backdrop-filter: blur(4px);
    border: 1px solid var(--line); border-radius: var(--radius-sm);
    padding: 11px 14px; font-style: italic;
}

/* ── TRANSITIONS for page elements ───────────────────────────────────── */
[data-testid="stVerticalBlock"] > div {
    animation: fade-up 0.3s var(--ease-out) both;
}
@keyframes fade-up {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── EXPANDER ────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-size: 13px !important; font-weight: 500 !important;
    color: var(--slate) !important;
    border-radius: var(--radius-md) !important;
    transition: background 0.2s ease !important;
}
.streamlit-expanderHeader:hover { color: var(--ink) !important; }
.streamlit-expanderContent { border-top: none !important; }

/* ── SELECT / RADIO ──────────────────────────────────────────────────── */
.stSelectbox > div > div, .stMultiSelect > div > div {
    border-radius: var(--radius-md) !important;
    border-color: var(--line) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px var(--teal-glow) !important;
}

hr { border-color: var(--line) !important; margin: 20px 0 !important; }

/* ── SIDEBAR ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--glass) !important; backdrop-filter: blur(16px) !important;
    border-right: 1px solid var(--line) !important;
}
/* ── GLOBAL STREAMLIT LOADER OVERLAY ─────────────────────────────────── */
/* Hides the default Streamlit running man and injects a MIRA pulsing logo */
[data-testid="stStatusWidget"] {
    visibility: hidden;
}
[data-testid="stStatusWidget"]::before {
    content: '◍';
    visibility: visible;
    position: fixed;
    top: 24px;
    right: 24px;
    width: 44px; height: 44px;
    background: linear-gradient(145deg, #0F1C24, #1E8A7A);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-family: 'Newsreader', serif; font-size: 22px;
    box-shadow: 0 4px 14px rgba(30,138,122,0.3), 0 0 0 1px rgba(255,255,255,0.1) inset;
    animation: corner-pulse 1.8s ease-in-out infinite;
    z-index: 999999;
}
@keyframes corner-pulse {
    0%, 100% { transform: scale(1); box-shadow: 0 4px 14px rgba(30,138,122,0.3); }
    50% { transform: scale(0.92); box-shadow: 0 2px 8px rgba(30,138,122,0.15); opacity: 0.85; }
}
</style>
"""

def apply_design():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_mira_loader(step: int = 0):
    """
    Premium MIRA loading animation shown during the 'running' stage.
    step: 0=starting, 1=querying DB, 2=cross-referencing guidelines, 3=drafting report
    """
    steps = [
        ("🔍", "Querying patient database",    "Searching lab records & clinical data"),
        ("📚", "Cross-referencing guidelines", "Matching against medical knowledge base"),
        ("🧠", "Drafting clinical report",      "Synthesizing findings with AI reasoning"),
        ("✅", "Safety check",                 "Validating report before your review"),
    ]
    steps_html = ""
    for i, (icon, label, sub) in enumerate(steps):
        if i < step:
            cls = "done"; status_html = '<span style="color:var(--teal);font-size:11px;">✓ Done</span>'
        elif i == step:
            cls = "active loader-step-active"
            status_html = '<div class="dot-bounce"><span></span><span></span><span></span></div>'
        else:
            cls = "waiting"; status_html = '<span style="color:var(--slate-2);font-size:11px;">Waiting</span>'

        icon_cls = "active" if i == step else ("done" if i < step else "waiting")
        steps_html += f"""
<div class="mira-loader-step">
    <div class="loader-step-icon {icon_cls}">{icon}</div>
    <div style="flex:1;">
        <div class="loader-step-text">{label}</div>
        <div class="loader-step-status">{sub}</div>
    </div>
    {status_html}
</div>"""

    st.markdown(f"""
<div class="mira-loader-wrap">
    <div class="mira-loader-logo">◍</div>
    <div>
        <div class="mira-loader-label">MIRA is working…</div>
        <div class="mira-loader-sub">Your report will be ready for review shortly</div>
    </div>
    <div class="mira-loader-steps">
        {steps_html}
    </div>
</div>""", unsafe_allow_html=True)
