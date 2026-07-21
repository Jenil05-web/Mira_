import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

__doc__ = """
streamlit_app_prod.py
======================
MIRA Production — Clinical Audit Console (Production Build)

Differences from streamlit_app.py (dev):
  - Auth gate: JWT login required before any clinical data is visible
  - Hospital context: every query tagged with user's hospital_id
  - Admin panel: audit log viewer + stats (admin role only)
  - Audit trail: every action logged via AuditLogger
  - Uses mira_pipeline_prod.py instead of mira_pipeline.py

FIXES IN THIS VERSION (aligned with the enhanced mira_pipeline_prod.py)
-------------------------------------------------------------------------
1. REVISION REQUEST WAS BROKEN AT THE UI LAYER
   The old "Send revision request" button called
   `submit_human_decision(..., "approve", ...)` — the wrong decision — and
   never passed the clinician's typed feedback at all, so even a perfectly
   correct backend had nothing to revise. Fixed: it now sends
   `"reject"` with `feedback=feedback_text`, which matches what the
   pipeline expects.

2. THE ORIGINAL QUESTION WAS SILENTLY LOST
   `pending_question` was cleared right after the initial run, but was
   still the value used later when calling `submit_human_decision(...,
   clinical_question=...)` on approve/reject — so it was always sent as
   `""`. This matters a lot with the new pipeline, which uses
   `clinical_question` to validate stale-session recovery. Fixed: the
   question is now kept in a separate, durable `current_question` field
   that persists for the life of the audit.

3. NO WAY TO ABORT / RESET MID-AUDIT
   Previously "Start a new audit" only appeared after a report was fully
   finalized. Fixed: a "New Audit" reset is now available from the left
   panel at every stage (idle, running, awaiting review, complete), and
   uses the new `engine.start_new_audit()` helper.

4. NO VISIBILITY INTO DATA-QUALITY SIGNALS
   The pipeline now reports `data_status` (ok / patient_not_found /
   no_data / broadened_query / list_insufficient), which patient IDs
   were requested vs. found vs. missing, and richer safety flags
   (including `id_mismatch:...`). None of this reached the UI before.
   Fixed: a status banner now surfaces these plainly, and the "View data
   sources" panel shows requested/found/missing patient IDs and detected
   query intent (list vs. single patient, recognized conditions).

5. SAFETY FLAGS WERE SHOWN AS RAW, UNEXPLAINED STRINGS
   Fixed: flags are now mapped to short, human-readable explanations.

6. MULTILINGUAL VOICE INPUT (English / Hindi / Gujarati) — AUTO-DETECT
   A "dictate your query" panel using `engine.transcribe_voice_query()`.
   Spoken language is auto-detected server-side (no manual selector) and
   non-English speech is translated to English before reaching the
   pipeline. The verbatim transcript in the original language, plus the
   detected language, is preserved and shown in "View data sources".

7. SESSION-STATE WIDGET MUTATION BUG FIXED
   Streamlit forbids writing to `st.session_state.question_input` after
   the `question_input` widget has already been instantiated in the same
   script run. Directly assigning to it inside the transcribe handler
   (which runs *after* the text_area was created higher up on the page)
   raised StreamlitAPIException. Fixed: the transcribed text is now
   staged in a separate `_pending_question_text` key, which is drained
   into `question_input` at the very top of the script — before the
   widget is created — then the page reruns cleanly.

Run with: streamlit run streamlit_app_prod.py
"""

import json
import time
import uuid

import markdown as md_lib
import streamlit as st

from core.auth import AuthManager, require_auth, Role
from core.config import ConfigManager
from core.audit import AuditLogger
from pipeline.engine import get_engine
from pipeline.ambient import AmbientConsultEngine

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MIRA — Clinical Audit Console",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from ui.design import *
apply_design()
from ui.components import *
from ui.components import _lang_label
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
# SAFETY FLAG EXPLANATIONS — human-readable, shown instead of raw codes
# ══════════════════════════════════════════════════════════════════════════

def explain_safety_flag(flag: str) -> str:
    if flag == "no_data":
        return "No patient data was available to finalize."
    if flag == "session_state_lost":
        return "The audit session expired before this could be finalized."
    if flag.startswith("id_mismatch:"):
        ids = flag.split(":", 1)[1]
        return f"Report referenced patient ID(s) not present in the retrieved data: {ids}"
    return flag.replace("_", " ").capitalize()


def status_banner_html(state: dict) -> str:
    """Builds an honest, plain-language banner for data-quality states
    surfaced by the pipeline (patient_not_found / no_data / broadened_query
    / list_insufficient). Returns '' if nothing needs flagging."""
    status = (state or {}).get("data_status", "")
    if status == "patient_not_found":
        missing = state.get("missing_patient_ids") or []
        ids = ", ".join(str(i) for i in missing) if missing else "the requested ID"
        return (f'<div class="banner banner-flagged"><span>▲</span>'
                f'Patient {ids} was not found in the system. No unrelated patient '
                f'data has been substituted.</div>')
    if status == "no_data":
        return ('<div class="banner banner-flagged"><span>▲</span>'
                'No matching lab data was found for this query.</div>')
    if status == "broadened_query":
        return ('<div class="banner banner-info"><span>ℹ</span>'
                'No exact match for your query — showing general abnormal findings '
                'across patients instead. This is disclosed in the report below.</div>')
    if status == "list_insufficient":
        return ('<div class="banner banner-info"><span>ℹ</span>'
                'This looked like a multi-patient question, but only a limited number '
                'of patients matched even after broadening the search.</div>')
    return ""


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
        "active_tab": "audit",
        "pending_question": "",
        "current_question": "",
        "voice_original_transcript": "",
        "voice_spoken_language": "",
        "current_input_mode": "text",
        "_pending_question_text": None,
        # Ambient Consult Mode
        "ambient_state": None,
        "ambient_stage": "idle",    # idle | processing | review | done
        "ambient_audio_bytes": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_audit_session():
    """Fully resets to a clean slate — used by the always-available
    'New Audit' button and by 'Start a new audit' after completion.
    Uses engine.start_new_audit() so no stale thread/state is reused."""
    st.session_state.thread_config = engine.start_new_audit()
    st.session_state.stage = "idle"
    st.session_state.paused_state = None
    st.session_state.final_state = None
    st.session_state.pending_question = ""
    st.session_state.current_question = ""
    st.session_state.show_feedback_box = False
    st.session_state.pop("feedback_box", None)
    st.session_state.voice_original_transcript = ""
    st.session_state.voice_spoken_language = ""
    st.session_state.current_input_mode = "text"
    st.session_state.pop("question_input", None)
    st.session_state._pending_question_text = None


# Call to initialise session state
init_session()


# Clear stale thread config on cold start (Render spin-down fix)
if st.session_state.get("thread_config") and st.session_state.stage == "idle":
    st.session_state.thread_config = None


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
is_running = st.session_state.stage == "running" or st.session_state.get("_ambient_processing")
mark_cls = "mira-mark is-thinking" if is_running else "mira-mark"

st.markdown(f"""
<div class="mira-header">
    <div class="mira-header-left">
        <div class="{mark_cls}">M</div>
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
    tab_audit, tab_triage, tab_ambient, tab_admin = st.tabs(
        ["Clinical Audit", "Live Triage Dashboard", "Ambient Consult", "Admin"])
else:
    tab_audit, tab_triage, tab_ambient = st.tabs(
        ["Clinical Audit", "Live Triage Dashboard", "Ambient Consult"])
    tab_admin = None


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

        # FIX: drain any transcribed text staged by the voice-input handler
        # into the question_input widget's state *before* the widget below
        # is instantiated. Writing to st.session_state.question_input after
        # the widget exists raises StreamlitAPIException — this runs first.
        if st.session_state.get("_pending_question_text") is not None:
            st.session_state.question_input = st.session_state._pending_question_text
            st.session_state._pending_question_text = None

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

        # ── Voice input — auto-detected language (English / Hindi / Gujarati) ──
        # Spoken language is auto-detected server-side; whatever isn't
        # English is translated before reaching the pipeline, since the SQL
        # and condition-vocabulary agents only understand English medical
        # terms. The verbatim transcript in the original language is always
        # kept and shown for audit purposes.
        _controls_disabled = st.session_state.stage in ["running", "awaiting_review"]

        with st.expander("🎙️  Dictate your query — language is detected automatically"):
            st.markdown(f"""
            <div class="mic-panel">
                <div class="mic-panel-head">
                    <div class="mic-icon-badge">🎙️</div>
                    <div>
                        <div class="mic-panel-title">Speak your clinical question</div>
                        <div class="mic-panel-sub">English, Hindi, or Gujarati — no need to pick a language</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")

            audio_bytes, audio_filename = None, "query.webm"
            try:
                audio_value = st.audio_input("Record your question", key="voice_recorder",
                                             disabled=_controls_disabled,
                                             label_visibility="collapsed")
                if audio_value is not None:
                    audio_bytes = audio_value.getvalue()
                    # st.audio_input records in webm format — the filename
                    # extension must match so Whisper decodes it correctly.
                    audio_filename = "query.webm"
            except AttributeError:
                # Older Streamlit versions don't have st.audio_input — fall
                # back to a file upload so this still works everywhere.
                uploaded_audio = st.file_uploader(
                    "Upload a short audio clip (wav / mp3 / m4a / ogg / webm)",
                    type=["wav", "mp3", "m4a", "ogg", "webm"], key="voice_uploader",
                    disabled=_controls_disabled,
                )
                if uploaded_audio is not None:
                    audio_bytes = uploaded_audio.getvalue()
                    audio_filename = uploaded_audio.name

            transcribe_clicked = st.button(
                "✦ Transcribe and use as query", use_container_width=True, type="primary",
                disabled=(audio_bytes is None or _controls_disabled),
            )

            if transcribe_clicked and audio_bytes:
                ph = st.empty()
                with ph.container():
                    render_processing_strip("Detecting language and transcribing")
                try:
                    result = engine.transcribe_voice_query(
                        audio_bytes, audio_filename, spoken_language="auto",
                        user_id=user.user_id, hospital_id=user.hospital_id,
                    )
                except Exception as exc:
                    result = {"error": str(exc)}
                ph.empty()
                if "error" in result:
                    st.error(f"Transcription failed: {result['error']}")
                elif not result.get("clinical_question", "").strip():
                    st.warning("No speech detected - please try recording again.")
                else:
                    # FIX: stage the value instead of writing directly to
                    # st.session_state.question_input here — the widget
                    # already exists this run. It gets drained at the top
                    # of the script on the rerun triggered below.
                    st.session_state._pending_question_text = result["clinical_question"]
                    st.session_state.voice_original_transcript = result.get("original_transcript", "")
                    st.session_state.voice_spoken_language = result.get("detected_language") or result.get("spoken_language", "")
                    st.session_state.current_input_mode = "voice"
                    st.toast("Transcription complete", icon="🎙️")
                    st.rerun()

            if (st.session_state.get("voice_original_transcript")
                    and st.session_state.get("current_input_mode") == "voice"):
                spoken = _lang_label(st.session_state.voice_spoken_language) or "Detected language"
                st.markdown(f"""
                <div class="mic-result-chip">
                    <span class="mic-result-dot"></span>Detected: {spoken}
                </div>
                <div class="mic-transcript-box">
                    \u201c{st.session_state.voice_original_transcript}\u201d
                </div>
                """, unsafe_allow_html=True)

        st.write("")

        run_disabled = st.session_state.stage in ["running", "awaiting_review"] or not question.strip()
        run_col, new_audit_col = st.columns([2, 1])
        with run_col:
            run_clicked = st.button("Run clinical audit", type="primary",
                                    use_container_width=True, disabled=run_disabled)
        with new_audit_col:
            # FIX / NEW FEATURE: previously only available after a report was
            # fully finalized. Now available at every stage — lets the
            # clinician abort a stuck or unwanted audit and start clean
            # without waiting for it to finish.
            new_audit_clicked = st.button("New audit", use_container_width=True)

        if new_audit_clicked:
            reset_audit_session()
            st.toast("New audit started", icon="✨")
            st.rerun()

        st.write("")
        render_fill_rail(
            "running_sql" if st.session_state.stage == "running" else
            "awaiting_review" if st.session_state.stage == "awaiting_review" else
            "complete" if st.session_state.stage == "complete" else "idle"
        )

        if st.session_state.paused_state or st.session_state.final_state:
            active = st.session_state.final_state or st.session_state.paused_state
            render_data_sources_expander(active)

        # Sign-out button
        st.write("")
        if st.button("Sign out", use_container_width=True):
            audit_logger.log_logout(user.user_id, st.session_state.session_id)
            st.session_state.auth_token = ""
            st.session_state.auth_user  = None
            reset_audit_session()
            st.rerun()

    with right_col:
        if st.session_state.stage == "idle":
            st.markdown("""
            <div class="empty-state">
                <div class="mark is-alive">M</div>
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

            banner = status_banner_html(state)
            if banner:
                st.markdown(banner, unsafe_allow_html=True)

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
                send_col, cancel_col = st.columns(2)
                with send_col:
                    send_clicked = st.button(
                        "Send revision request", type="primary",
                        disabled=not feedback_text.strip()
                    )
                with cancel_col:
                    cancel_clicked = st.button("Cancel")

                if cancel_clicked:
                    st.session_state.show_feedback_box = False
                    st.session_state.pop("feedback_box", None)
                    st.rerun()

                if send_clicked:
                    ph = st.empty()
                    with ph.container():
                        render_processing_strip("Sending feedback to reasoning agent")
                    # FIX: this used to send decision="approve" and never
                    # passed the feedback text at all, so nothing was ever
                    # actually revised. It now sends "reject" + the
                    # clinician's feedback, and uses the durable
                    # current_question rather than the already-cleared
                    # pending_question.
                    revised = engine.submit_human_decision(
                        st.session_state.thread_config, "reject",
                        feedback=feedback_text.strip(),
                        user_id=user.user_id, hospital_id=user.hospital_id,
                        clinical_question=st.session_state.get("current_question") or "",
                        paused_state=st.session_state.paused_state,
                    )
                    ph.empty()
                    st.session_state.paused_state = revised
                    st.session_state.show_feedback_box = False
                    st.session_state.pop("feedback_box", None)
                    st.rerun()

            if approve_clicked:
                ph = st.empty()
                with ph.container():
                    render_processing_strip("Running final safety check")
                final = engine.submit_human_decision(
                    st.session_state.thread_config, "approve",
                    user_id=user.user_id, hospital_id=user.hospital_id,
                    clinical_question=st.session_state.get("current_question") or "",
                    paused_state=st.session_state.paused_state,
                )
                ph.empty()
                st.session_state.final_state = final
                st.session_state.stage = "complete"
                st.rerun()

        elif st.session_state.stage == "complete":
            state = st.session_state.final_state

            banner = status_banner_html(state)
            if banner:
                st.markdown(banner, unsafe_allow_html=True)

            if state.get("approved"):
                st.markdown('<div class="banner banner-approved"><svg class="check" viewBox="0 0 10 8" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M1 4L3.5 6.5L9 1" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                            'Cleared by safety review — all claims grounded in retrieved data</div>',
                            unsafe_allow_html=True)
            else:
                flags = state.get("safety_flags", [])
                explained = "; ".join(explain_safety_flag(f) for f in flags) or "Review recommended"
                st.markdown(f'<div class="banner banner-flagged"><span>▲</span>Flagged — {explained}</div>',
                            unsafe_allow_html=True)

            st.markdown('<div class="panel-eyebrow with-dot"><span class="eyebrow-dot"></span>FINAL CLINICAL REPORT</div>',
                        unsafe_allow_html=True)
            render_trend_chart(state.get("trend_data", ""))
            render_report_panel(state.get("final_report", "No report generated."))
            
            # ── Export as Word/PDF ───────────────────────────────────────
            # Style the export buttons: remove hover, set distinct color
            st.markdown("""
            <style>
            div[data-testid="stDownloadButton"] button {
                background-color: #5C7A89 !important;
                color: #FFFFFF !important;
                border: none !important;
                box-shadow: none !important;
                transition: none !important;
            }
            div[data-testid="stDownloadButton"] button:hover,
            div[data-testid="stDownloadButton"] button:active,
            div[data-testid="stDownloadButton"] button:focus {
                background-color: #5C7A89 !important;
                color: #FFFFFF !important;
                border: none !important;
                box-shadow: none !important;
                transform: none !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # MS Word can seamlessly open HTML files saved as .doc
            html_report = f"""<html><head><meta charset="utf-8"></head>
            <body><h2>MIRA Clinical Report</h2>
            {md_lib.markdown(state.get("final_report", "No report generated."))}
            <br><p><small>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            </body></html>"""
            
            def _create_simple_pdf(text_content):
                """Minimal, zero-dependency PDF generator for demo purposes."""
                lines = text_content.replace('\r', '').split('\n')
                pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
                stream = b"BT\n/F1 10 Tf\n10 750 Td\n"
                for line in lines[:55]:  # fit to 1 page
                    clean = ''.join(c for c in line if 32 <= ord(c) <= 126)
                    clean = clean.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
                    stream += f"({clean}) Tj\n0 -12 Td\n".encode('ascii')
                stream += b"ET"
                pdf += f"5 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode('ascii')
                pdf += stream
                pdf += b"\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000224 00000 n \n0000000311 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n405\n%%EOF\n"
                return pdf

            pdf_report = _create_simple_pdf(f"MIRA Clinical Report\n\n{state.get('final_report', '')}\n\nGenerated: {time.strftime('%Y-%m-%d')}")
            
            export_col1, export_col2, export_col3, _ = st.columns([1.2, 1.2, 1.2, 1.5])
            with export_col1:
                st.download_button("Export as Word (.doc)", data=html_report, 
                                   file_name=f"MIRA_Report_{int(time.time())}.doc", 
                                   mime="application/msword")
            with export_col2:
                st.download_button("Export as PDF (.pdf)", data=pdf_report, 
                                   file_name=f"MIRA_Report_{int(time.time())}.pdf", 
                                   mime="application/pdf")
            with export_col3:
                if st.button("Start a new audit"):
                    reset_audit_session()
                    st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # RUN HANDLER
    # ══════════════════════════════════════════════════════════════════════

    if run_clicked and question.strip() and st.session_state.stage == "idle":
        st.session_state.pending_question = question.strip()
        # FIX: keep the original question in a field that survives the
        # rest of the audit lifecycle (pending_question gets cleared below
        # once the run starts).
        st.session_state.current_question = question.strip()
        st.session_state.stage = "running"
        st.rerun()

    if st.session_state.stage == "running" and st.session_state.get("pending_question"):
        with right_col:
            ph = st.empty()
            with ph.container():
                render_mira_loader(1)
                render_skeleton_panel()

            st.session_state.thread_config = engine.start_new_audit()
            paused = engine.run_until_review(
                st.session_state.pending_question,
                st.session_state.thread_config,
                user_id=user.user_id,
                hospital_id=user.hospital_id,
                session_id=st.session_state.session_id,
                input_mode=st.session_state.get("current_input_mode", "text"),
                spoken_language=st.session_state.get("voice_spoken_language", ""),
                original_transcript=st.session_state.get("voice_original_transcript", ""),
            )
            ph.empty()

        st.session_state.pending_question = ""
        st.session_state.paused_state = paused
        st.session_state.stage = "awaiting_review"
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE TRIAGE DASHBOARD (Multi-patient & Monitoring)
# ══════════════════════════════════════════════════════════════════════════

if tab_triage:
    with tab_triage:
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("<h3>Proactive Triage Dashboard</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color:#5C7A89;'>Automatically scans all available patient data to surface critical risks, ranked by severity.</p>", unsafe_allow_html=True)
        with col2:
            auto_refresh = st.toggle("Enable Live Monitoring (Auto-scans every 30s)")
            if st.button("Run Manual Scan Now", type="primary", use_container_width=True):
                st.session_state.last_triage = None
        
        # We cache the triage result in session_state so it doesn't run on every UI tweak
        if "last_triage" not in st.session_state:
            st.session_state.last_triage = None
            
        should_run = (st.session_state.last_triage is None) or auto_refresh
        
        if should_run:
            ph = st.empty()
            with ph.container():
                render_mira_loader(2)
            st.session_state.last_triage = engine.run_triage(user.hospital_id, limit=60)
            ph.empty()
            
        triage_data = st.session_state.last_triage
        
        if not triage_data:
            st.info("No critical patients detected at this time.")
        else:
            for patient in triage_data:
                score = patient.get("severity_score", 0)
                color = "#C9501F" if score >= 8 else ("#B98A2E" if score >= 5 else "#2D8C7F")
                st.markdown(f"""
                <div style="background:#fff; border-left: 4px solid {color}; border-top: 1px solid #E7E9E4; border-right: 1px solid #E7E9E4; border-bottom: 1px solid #E7E9E4; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:600; font-size:16px;">Patient ID: {patient.get('subject_id', 'Unknown')}</span>
                        <span style="background:{color}15; color:{color}; padding: 4px 10px; border-radius: 100px; font-weight:600; font-size:14px;">Severity: {score}/10</span>
                    </div>
                    <div style="margin-top: 12px; font-size: 14.5px; color:#1C2B33;">
                        <strong>Risk:</strong> {patient.get('reason', '')}
                    </div>
                    <div style="margin-top: 8px; font-size: 13.5px; color:#5C7A89; background:#F9FAFB; padding: 10px; border-radius: 6px;">
                        <strong>Critical Labs:</strong> {patient.get('labs', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        if auto_refresh:
            time.sleep(30)
            st.session_state.last_triage = None
            st.rerun()




# ══════════════════════════════════════════════════════════════════════════
# TAB — AMBIENT CONSULT MODE
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_ambient_engine():
    return AmbientConsultEngine(engine)

ambient_engine = _get_ambient_engine()

with tab_ambient:
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom: 4px;">
        <div style="font-family:'Newsreader',serif; font-size:24px; font-weight:500; color:#1C2B33;">Ambient Consult Mode</div>
        <div style="font-size:13.5px; color:#5C7A89; margin-top:4px;">
            MIRA listens to the consultation and auto-drafts a structured SOAP note — no typing required.
        </div>
    </div>
    """, unsafe_allow_html=True)

    amb_stage = st.session_state.get("ambient_stage", "idle")

    # ── STAGE: IDLE ──────────────────────────────────────────────────
    if amb_stage == "idle":
        st.markdown("""
        <div style="background:#F0F7FF; border:1px solid #C5D9E8; border-radius:10px; padding:18px 20px; margin:18px 0;">
            <div style="font-size:13.5px; font-weight:600; color:#1C2B33; margin-bottom:6px;">Before you begin</div>
            <div style="font-size:13px; color:#5C7A89; line-height:1.7;">
                This mode auto-transcribes the consultation and drafts a SOAP note.
                The patient must verbally consent before recording starts.
                Audio is processed immediately and not stored after the note is generated.
            </div>
        </div>
        """, unsafe_allow_html=True)

        amb_left, amb_right = st.columns([2, 1])
        with amb_left:
            patient_id_input = st.text_input(
                "Patient MRN / ID (optional)",
                placeholder="Leave blank for walk-in patients with no record",
                key="amb_patient_id"
            )
            consent_cb = st.checkbox(
                "I confirm the patient has verbally consented to AI-assisted note-taking.",
                key="amb_consent"
            )
        with amb_right:
            st.markdown('<div style="height:52px;"></div>', unsafe_allow_html=True)
            if st.button("🎤  Start Consult", type="primary",
                         disabled=not consent_cb, use_container_width=True):
                new_state = ambient_engine.new_session(
                    user_id=user.user_id,
                    hospital_id=user.hospital_id,
                    patient_id=patient_id_input.strip() or None
                )
                st.session_state.ambient_state = new_state
                st.session_state.ambient_stage  = "recording"
                # Reset all working state
                st.session_state._amb_live_transcript_readonly = ""
                st.session_state._amb_mic_key   = 0
                st.session_state._amb_last_size = 0
                st.rerun()

    # ── STAGE: RECORDING ─────────────────────────────────────────────
    elif amb_stage == "recording":
        amb_s   = st.session_state.ambient_state
        mic_key = st.session_state.get("_amb_mic_key", 0)
        start_ts = int(amb_s.started_at or time.time())

        # ── Status bar with live JS timer ────────────────────────────
        st.markdown(f"""
        <div style="background:#FFF5F5; border:1px solid #F5C6C6; border-radius:10px;
             padding:14px 20px; margin:8px 0; display:flex; align-items:center; gap:14px;">
            <span class="record-dot"></span>
            <div style="flex:1;">
                <span style="font-weight:600;color:#C9501F;font-size:14.5px;">
                    Consult live &nbsp;&middot;&nbsp; <span id="amb-timer">00:00</span>
                </span>
                <span style="font-size:12px;color:#8FA3AE;margin-left:16px;">
                    Speak → stop → transcript auto-appears. Repeat as needed.
                </span>
            </div>
        </div>
        <style>@keyframes blink{{50%{{opacity:0}}}}</style>
        <script>(function(){{
            var s={start_ts};
            function f(n){{return String(n).padStart(2,'0')}}
            function t(){{
                var e=Math.floor(Date.now()/1000)-s,m=Math.floor(e/60),sc=e%60;
                var el=document.getElementById('amb-timer');
                if(el)el.innerText=f(m)+':'+f(sc);
            }}
            t();setInterval(t,1000);
        }})();</script>
        """, unsafe_allow_html=True)

        # ── True Live Transcript Component (Web Speech API) ─────────────
        st.markdown(
            '<div style="font-size:14px;font-weight:600;color:#1C2B33;margin:12px 0 6px 0;">'
            '🎤 Live Transcription &nbsp;<span style="font-weight:400;color:#8FA3AE;">'
            '— Speak naturally. Words appear instantly. Zero API cost.</span></div>',
            unsafe_allow_html=True
        )

        from ui.ambient_speech import live_speech_component
        
        # The component renders the UI box and returns the transcript string
        live_txt = live_speech_component(key=f"amb_speech_comp_{mic_key}")
        
        # Always save the latest transcript to session state for "End Consult"
        if live_txt:
            st.session_state._amb_live_transcript_readonly = live_txt

        # ── Footer buttons ────────────────────────────────────────────
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        col_end, col_spacer, col_cancel = st.columns([2, 3, 1])
        with col_end:
            if st.button("⏹  End Consult & Generate SOAP Note", type="primary", use_container_width=True):
                final_transcript = st.session_state.get("_amb_live_transcript_readonly", "").strip()
                st.session_state._amb_final_transcript = final_transcript
                st.session_state._amb_mic_key   = 0
                st.session_state._amb_last_size = 0
                st.session_state.ambient_stage  = "processing"
                st.rerun()
        with col_cancel:
            if st.button("Cancel", use_container_width=True):
                st.session_state.ambient_state  = None
                st.session_state.ambient_stage  = "idle"
                st.session_state._amb_live_transcript_readonly = ""
                st.session_state._amb_mic_key   = 0
                st.session_state._amb_last_size = 0
                st.rerun()

    # ── STAGE: PROCESSING ────────────────────────────────────────────
    elif amb_stage == "processing":
        amb_s = st.session_state.ambient_state
        final_transcript = st.session_state.get("_amb_final_transcript", "").strip()

        if not final_transcript:
            st.warning("No transcript was recorded. Please speak at least one segment before ending the consult.")
            st.session_state.ambient_stage = "recording"
            st.rerun()

        with st.spinner("🧠  Analysing consultation and drafting SOAP note..."):
            amb_s = ambient_engine.process_consult(
                amb_s,
                audio_bytes=None,
                transcript_text=final_transcript
            )
            st.session_state.ambient_state = amb_s
            st.session_state.ambient_stage  = "review"
            st.session_state._amb_live_transcript_readonly = ""
            st.session_state._amb_final_transcript    = ""
        st.rerun()




    # ── STAGE: REVIEW ────────────────────────────────────────────────
    elif amb_stage == "review":
        amb_s = st.session_state.ambient_state

        # Safety flags banner
        flags = amb_s.safety_flags
        if flags and "hallucination_detected" in flags:
            st.markdown(f'<div class="banner banner-flagged"><span>▲</span>Safety review flagged potential issues: {"; ".join(flags)}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="banner banner-approved"><svg class="check" viewBox="0 0 10 8" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M1 4L3.5 6.5L9 1" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>Safety check passed - note reflects consultation content</div>',
                        unsafe_allow_html=True)

        # Extracted entities summary (collapsible)
        entities = amb_s.structured_entities
        if entities:
            with st.expander("View extracted clinical entities", expanded=False):
                ent_cols = st.columns(3)
                fields = [
                    ("Chief Complaint", entities.get("chief_complaint", "—")),
                    ("Symptoms", ", ".join(entities.get("symptoms", [])) or "—"),
                    ("Duration", entities.get("symptom_duration", "—")),
                    ("Medications", ", ".join(entities.get("current_medications", [])) or "—"),
                    ("Allergies", ", ".join(entities.get("allergies", [])) or "—"),
                    ("History", ", ".join(entities.get("patient_history", [])) or "—"),
                ]
                for i, (label, val) in enumerate(fields):
                    with ent_cols[i % 3]:
                        st.markdown(f'<div class="panel-eyebrow">{label}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="font-size:13px;color:#1C2B33;margin-bottom:12px;">{val}</div>', unsafe_allow_html=True)

        # Draft note (editable)
        st.markdown('<div class="panel-eyebrow with-dot"><span class="eyebrow-dot"></span>DRAFT SOAP NOTE — Review and edit before filing</div>',
                    unsafe_allow_html=True)
        edited_note = st.text_area(
            "Draft SOAP Note",
            value=amb_s.draft_note,
            height=480,
            label_visibility="collapsed",
            key="amb_edited_note"
        )

        # Action buttons
        btn_approve, btn_restart, _ = st.columns([1, 1, 2])
        with btn_approve:
            if st.button("✓  Approve & Export Note", type="primary", use_container_width=True):
                amb_s = ambient_engine.approve(amb_s, doctor_edits=edited_note)
                st.session_state.ambient_state = amb_s
                st.session_state.ambient_stage  = "done"
                st.rerun()
        with btn_restart:
            if st.button("New Consult", use_container_width=True):
                st.session_state.ambient_state = None
                st.session_state.ambient_stage  = "idle"
                st.session_state.ambient_audio_bytes = None
                st.rerun()

    # ── STAGE: DONE ──────────────────────────────────────────────────
    elif amb_stage == "done":
        amb_s = st.session_state.ambient_state
        if getattr(amb_s, "final_note", None):
            # Final note is ready
            st.markdown('<div class="banner banner-approved"><svg class="check" viewBox="0 0 10 8" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M1 4L3.5 6.5L9 1" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>SOAP note approved and ready to file</div>',
                        unsafe_allow_html=True)
        st.markdown('<div class="panel-eyebrow with-dot"><span class="eyebrow-dot"></span>FINAL SOAP NOTE</div>',
                    unsafe_allow_html=True)
        render_report_panel(amb_s.final_note)

        # Export
        html_note = f"""<html><head><meta charset="utf-8"></head>
        <body><h2>MIRA SOAP Note</h2>
        {md_lib.markdown(amb_s.final_note)}
        <br><p><small>Patient: {amb_s.patient_id or 'Walk-in'} &nbsp;|&nbsp; Doctor: {user.display_name} &nbsp;|&nbsp; {time.strftime('%Y-%m-%d %H:%M')}</small></p>
        </body></html>"""

        st.markdown("""
        <style>
        div[data-testid="stDownloadButton"] button {
            background-color: #5C7A89 !important; color: #FFFFFF !important;
            border: none !important; transition: none !important;
        }
        div[data-testid="stDownloadButton"] button:hover {
            background-color: #5C7A89 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        exp1, exp2, exp3, _ = st.columns([1.2, 1.2, 1.2, 1.5])
        with exp1:
            st.download_button("Export as Word (.doc)", data=html_note,
                               file_name=f"SOAP_Note_{int(time.time())}.doc",
                               mime="application/msword")
        with exp2:
            plain_text = amb_s.final_note.encode("utf-8")
            st.download_button("Export as Text (.txt)", data=plain_text,
                               file_name=f"SOAP_Note_{int(time.time())}.txt",
                               mime="text/plain")
        with exp3:
            if st.button("Start New Consult", use_container_width=True):
                st.session_state.ambient_state = None
                st.session_state.ambient_stage  = "idle"
                st.session_state.ambient_audio_bytes = None
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# TAB — ADMIN PANEL (admin role only)
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