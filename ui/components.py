import streamlit as st
import json
import time
import markdown as md_lib
# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

# Display labels for auto-detected spoken language codes. Falls back to a
# capitalized raw string for anything not in this map, so we never need to
# widen this list to keep the UI working.
_LANG_LABELS = {
    "english": "English", "en": "English",
    "hindi": "हिन्दी (Hindi)", "hi": "हिन्दी (Hindi)",
    "gujarati": "ગુજરાતી (Gujarati)", "gu": "ગુજરાતી (Gujarati)",
}


def _lang_label(code: str) -> str:
    if not code:
        return ""
    return _LANG_LABELS.get(code.lower(), code.capitalize())


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


def render_skeleton_panel():
    st.markdown("""
    <div class="skeleton-panel">
        <div class="skeleton-line title"></div>
        <div class="skeleton-line"></div>
        <div class="skeleton-line"></div>
        <div class="skeleton-line short"></div>
    </div>
    <div class="skeleton-panel">
        <div class="skeleton-line title"></div>
        <div class="skeleton-line"></div>
        <div class="skeleton-line short"></div>
    </div>
    """, unsafe_allow_html=True)


def render_processing_strip(message: str, sub: str = "Please wait..."):
    st.markdown(f"""
    <div class="processing-strip">
        <div class="processing-strip-icon">M</div>
        <div style="flex:1;">
            <div class="processing-text">{message}</div>
            <div class="processing-sub">{sub}</div>
        </div>
        <div class="shimmer-bar" style="max-width: 30%;"></div>
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


def render_data_sources_expander(active: dict):
    """The 'View data sources' panel — now also surfaces query intent and
    patient existence detail so a clinician can see *why* the pipeline
    made the choices it made, instead of a report appearing from nowhere."""
    with st.expander("View data sources used"):
        if active.get("input_mode") == "voice" and active.get("original_transcript"):
            spoken = _lang_label(active.get("spoken_language", "")) or "auto-detected language"
            st.markdown('<div class="panel-eyebrow">VOICE INPUT</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:13px;color:#5C7A89;">Spoken in <strong style="color:#1C2B33;">'
                f'{spoken}</strong>: \u201c{active["original_transcript"]}\u201d</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="panel-eyebrow" style="margin-top:16px;">SQL QUERY EXECUTED</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mono-block">{active.get("sql_query_used","—")}</div>', unsafe_allow_html=True)

        intent = active.get("query_intent") or {}
        if intent:
            conditions = ", ".join(intent.get("conditions", [])) or "none detected"
            list_flag = "yes" if intent.get("is_list") else "no"
            st.markdown('<div class="panel-eyebrow" style="margin-top:16px;">QUERY INTERPRETATION</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:13px;color:#5C7A89;">'
                f'Interpreted as a multi-patient question: <strong style="color:#1C2B33;">{list_flag}</strong><br>'
                f'Recognized clinical terms: <strong style="color:#1C2B33;">{conditions}</strong></div>',
                unsafe_allow_html=True
            )

        requested = active.get("requested_patient_ids") or []
        found = active.get("found_patient_ids") or []
        missing = active.get("missing_patient_ids") or []
        if requested or found:
            st.markdown('<div class="panel-eyebrow" style="margin-top:16px;">PATIENT MATCH DETAIL</div>', unsafe_allow_html=True)
            lines = []
            if requested:
                lines.append(f"Requested: {', '.join(str(i) for i in requested)}")
            if found:
                lines.append(f"Found: {', '.join(str(i) for i in found)}")
            if missing:
                lines.append(f"Not found: {', '.join(str(i) for i in missing)}")
            st.markdown(f'<div style="font-size:13px;color:#5C7A89;">{"<br>".join(lines)}</div>',
                        unsafe_allow_html=True)

        if active.get("data_status") == "broadened_query":
            st.markdown('<div class="panel-eyebrow" style="margin-top:16px;">NOTE</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:13px;color:#5C7A89;">No results matched the specific '
                'criteria, so the search was broadened to general abnormal findings.</div>',
                unsafe_allow_html=True
            )

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


