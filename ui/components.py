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


def render_connection_status(test_result: dict | None, compact: bool = False):
    """
    Renders a colored DB connection status badge.
    test_result: dict returned by DBAdapter.test_connection(), or None if not yet checked.
    compact: if True renders just an inline dot+text suitable for the header.
    """
    if test_result is None:
        color, dot, label = "#8FA3AE", "●", "Not checked"
    elif test_result.get("ok"):
        latency = test_result.get("latency_ms", 0)
        n_mapped = len(test_result.get("mapped", {}))
        color, dot, label = "#1E8A7A", "●", f"Connected · {latency} ms · {n_mapped} concepts mapped"
    else:
        color, dot, label = "#C9501F", "●", "Connection failed"

    if compact:
        st.markdown(
            f'<span style="color:{color};font-size:10px;margin-right:4px;">{dot}</span>'
            f'<span style="font-size:11px;color:#5C7A89;">{label}</span>',
            unsafe_allow_html=True,
        )
        return

    # Full status card
    border = f"1px solid {color}30"
    bg = f"{color}08"
    st.markdown(
        f'<div style="background:{bg};border:{border};border-radius:10px;padding:12px 16px;'
        f'display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
        f'<span style="color:{color};font-size:14px;">{dot}</span>'
        f'<div style="flex:1;">'
        f'<div style="font-size:13px;font-weight:600;color:#1C2B33;">{label}</div>',
        unsafe_allow_html=True,
    )

    if test_result and test_result.get("ok"):
        mapped = test_result.get("mapped", {})
        unmapped = test_result.get("unmapped", [])
        db_type = test_result.get("db_type", "")
        st.markdown(
            f'<div style="font-size:12px;color:#5C7A89;margin-top:2px;">'
            f'{db_type} · {test_result.get("table_count",0)} tables found</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if mapped:
            cols = st.columns(len(mapped))
            for col, (concept, table) in zip(cols, mapped.items()):
                with col:
                    st.markdown(
                        f'<div style="background:#1E8A7A12;border:1px solid #1E8A7A30;'
                        f'border-radius:6px;padding:6px 10px;text-align:center;">'
                        f'<div style="font-size:10px;color:#1E8A7A;font-weight:600;'
                        f'text-transform:uppercase;letter-spacing:0.05em;">{concept}</div>'
                        f'<div style="font-size:11px;color:#1C2B33;margin-top:2px;">{table}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        if unmapped:
            st.markdown(
                f'<div style="font-size:11px;color:#B98A2E;margin-top:8px;">'
                f'⚠ Not mapped: {", ".join(unmapped)}'
                f'</div>',
                unsafe_allow_html=True,
            )
    elif test_result:
        st.markdown(
            f'<div style="font-size:12px;color:#C9501F;margin-top:4px;">'
            f'{test_result.get("error", "Unknown error")}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


def render_hospital_setup_form(config_manager) -> dict | None:
    """
    Admin-only Hospital DB Setup form.
    Returns the active adapter if one was set up, otherwise None.
    config_manager: a ConfigManager instance.
    """
    st.markdown(
        '<div class="panel-eyebrow with-dot"><span class="eyebrow-dot"></span>'
        'DATA SOURCE SETUP</div>',
        unsafe_allow_html=True,
    )

    # ── Source type tabs ──────────────────────────────────────────────────
    tab_db, tab_file, tab_demo = st.tabs([
        "🔌 Database (PostgreSQL / MySQL)",
        "📂 Upload File (Excel / CSV)",
        "🧪 Demo Mode (Indian Data)",
    ])

    # ── Tab 1: Direct DB ──────────────────────────────────────────────────
    with tab_db:
        st.markdown(
            '<div style="font-size:13px;color:#5C7A89;margin:8px 0 16px;">'
            'Connect to your hospital\'s existing database. '
            'Only <strong>SELECT</strong> access is needed — MIRA never writes data.'
            '</div>',
            unsafe_allow_html=True,
        )

        db_type = st.selectbox(
            "Database type",
            ["PostgreSQL", "MySQL", "SQL Server", "SQLite (dev/local)"],
            index=0, key="setup_db_type",
        )
        placeholder_map = {
            "PostgreSQL":         "postgresql://user:password@host:5432/hospital_db",
            "MySQL":              "mysql+pymysql://user:password@host:3306/hospital_db",
            "SQL Server":         "mssql+pyodbc://user:password@host/hospital_db?driver=ODBC+Driver+17+for+SQL+Server",
            "SQLite (dev/local)": "sqlite:///./mira_data/mimic.db",
        }
        conn_str = st.text_input(
            "Connection string",
            placeholder=placeholder_map.get(db_type, ""),
            type="password", key="setup_conn_str",
            help="Credentials are only used to open the connection — never stored in MIRA logs.",
        )

        col_test, col_save, _ = st.columns([1, 1, 3])
        with col_test:
            if st.button("🔌 Test Connection", use_container_width=True,
                         key="btn_test_conn", disabled=not conn_str):
                with st.spinner("Connecting…"):
                    try:
                        from adapters.db import DBAdapter
                        _a = DBAdapter(conn_str)
                        _r = _a.test_connection()
                        st.session_state["_last_conn_test"] = _r
                        st.session_state["_active_source"] = "db"
                    except Exception as e:
                        st.session_state["_last_conn_test"] = {
                            "ok": False, "error": str(e),
                            "latency_ms": 0, "mapped": {}, "unmapped": [],
                            "table_count": 0, "db_type": "unknown",
                        }

        with col_save:
            last = st.session_state.get("_last_conn_test", {})
            if st.button("💾 Save & Apply", use_container_width=True,
                         key="btn_save_conn",
                         disabled=not (conn_str and last.get("ok"))):
                import os
                env_path = ".env"
                lines = []
                if os.path.exists(env_path):
                    with open(env_path) as f:
                        lines = [l for l in f.readlines()
                                 if not l.startswith("MIRA_DEFAULT_DB_URL")]
                lines.append(f"MIRA_DEFAULT_DB_URL={conn_str}\n")
                with open(env_path, "w") as f:
                    f.writelines(lines)
                st.session_state["_conn_saved"] = True
                st.toast("Connection saved — restart the app to apply.", icon="💾")

        cached = st.session_state.get("_last_conn_test")
        if cached is not None and st.session_state.get("_active_source") == "db":
            render_connection_status(cached)

        if st.session_state.get("_conn_saved"):
            st.markdown(
                '<div class="banner banner-approved">'
                '<svg class="check" viewBox="0 0 10 8" fill="none" xmlns="http://www.w3.org/2000/svg">'
                '<path d="M1 4L3.5 6.5L9 1" stroke="currentColor" stroke-width="2" '
                'stroke-linecap="round" stroke-linejoin="round"/></svg>'
                'Saved to .env. Restart MIRA to connect to the new database.</div>',
                unsafe_allow_html=True,
            )

    # ── Tab 2: Flat File Upload ───────────────────────────────────────────
    with tab_file:
        st.markdown(
            '<div style="font-size:13px;color:#5C7A89;margin:8px 0 12px;">'
            'No database? Upload your patient data directly — Excel, CSV, or Google Sheets. '
            '<strong>Data stays in memory and is never saved to disk by MIRA.</strong>'
            '</div>',
            unsafe_allow_html=True,
        )

        upload_mode = st.radio(
            "Source", ["Upload File", "Google Sheets Link"],
            horizontal=True, key="ff_upload_mode",
        )

        adapter_result = None

        if upload_mode == "Upload File":
            uploaded = st.file_uploader(
                "Upload patient data file",
                type=["xlsx", "xls", "csv", "ods"],
                key="ff_uploader",
                help="Excel sheets: each sheet becomes a table. CSV: one table.",
            )
            if uploaded:
                col_load, _ = st.columns([1, 3])
                with col_load:
                    if st.button("📥 Load File", use_container_width=True, key="btn_load_file"):
                        with st.spinner(f"Loading {uploaded.name}…"):
                            try:
                                from adapters.flat_file import FlatFileAdapter
                                _ff = FlatFileAdapter.from_bytes(uploaded.read(), uploaded.name)
                                st.session_state["_flat_file_adapter"] = _ff
                                st.session_state["_flat_file_test"] = _ff.test_connection()
                                st.session_state["_active_source"] = "flat_file"
                                st.toast(f"Loaded {uploaded.name}", icon="📂")
                            except Exception as e:
                                st.session_state["_flat_file_test"] = {
                                    "ok": False, "error": str(e),
                                    "latency_ms": 0, "mapped": {}, "unmapped": [],
                                    "table_count": 0, "db_type": "unknown",
                                }

        else:  # Google Sheets
            sheets_url = st.text_input(
                "Google Sheets share URL",
                placeholder="https://docs.google.com/spreadsheets/d/...",
                key="ff_sheets_url",
            )
            if sheets_url:
                if st.button("🔗 Load from Google Sheets", key="btn_load_sheets"):
                    with st.spinner("Fetching sheet…"):
                        try:
                            from adapters.flat_file import FlatFileAdapter
                            _ff = FlatFileAdapter.from_google_sheets(sheets_url)
                            st.session_state["_flat_file_adapter"] = _ff
                            st.session_state["_flat_file_test"] = _ff.test_connection()
                            st.session_state["_active_source"] = "flat_file"
                            st.toast("Google Sheet loaded", icon="📊")
                        except Exception as e:
                            st.session_state["_flat_file_test"] = {
                                "ok": False, "error": str(e),
                                "latency_ms": 0, "mapped": {}, "unmapped": [],
                                "table_count": 0, "db_type": "unknown",
                            }

        ff_test = st.session_state.get("_flat_file_test")
        if ff_test:
            render_connection_status(ff_test)
            if ff_test.get("ok"):
                rc = ff_test.get("row_counts", {})
                for tbl, cnt in rc.items():
                    st.caption(f"  {tbl}: {cnt:,} rows")

    # ── Tab 3: Demo Mode ──────────────────────────────────────────────────
    with tab_demo:
        st.markdown(
            '<div style="font-size:13px;color:#5C7A89;margin:8px 0 12px;">'
            'Try MIRA instantly with <strong>synthetic Indian hospital data</strong> '
            '(20 patients, 161 lab results). No database or file needed. '
            'Names and values are fictional — safe for demonstrations.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Show the sample data preview
        import pathlib
        demo_patients = pathlib.Path("mira_data/demo_india_patients.csv")
        demo_labs = pathlib.Path("mira_data/demo_india_labs.csv")

        if demo_patients.exists():
            try:
                import pandas as pd
                df_p = pd.read_csv(demo_patients)
                st.markdown("**Sample patients (first 5):**")
                st.dataframe(df_p[["uhid","patient_name","age","gender",
                                   "provisional_diagnosis","ward"]].head(5),
                             use_container_width=True, hide_index=True)
            except Exception:
                pass

        col_demo, _ = st.columns([1, 3])
        with col_demo:
            if st.button("🧪 Load Demo Data", use_container_width=True, key="btn_load_demo"):
                if not (demo_patients.exists() and demo_labs.exists()):
                    st.error("Demo files not found. Run: python mira_data/generate_demo.py")
                else:
                    with st.spinner("Loading demo data…"):
                        try:
                            from adapters.flat_file import FlatFileAdapter
                            _ff = FlatFileAdapter()
                            import pandas as pd
                            _ff._load_dataframe(pd.read_csv(demo_patients), "patients")
                            _ff._load_dataframe(pd.read_csv(demo_labs), "investigation_results")
                            _ff._source_label = "Indian Demo Data"
                            _ff.discover_schema()
                            st.session_state["_flat_file_adapter"] = _ff
                            st.session_state["_flat_file_test"] = _ff.test_connection()
                            st.session_state["_active_source"] = "flat_file"
                            st.toast("Demo data loaded — try asking: 'Which patients have high creatinine?'", icon="🧪")
                        except Exception as e:
                            st.error(f"Failed to load demo data: {e}")

        if (st.session_state.get("_active_source") == "flat_file"
                and st.session_state.get("_flat_file_test", {}).get("source_label") == "Indian Demo Data"):
            st.markdown(
                '<div class="banner banner-approved">'
                '<svg class="check" viewBox="0 0 10 8" fill="none" xmlns="http://www.w3.org/2000/svg">'
                '<path d="M1 4L3.5 6.5L9 1" stroke="currentColor" stroke-width="2" '
                'stroke-linecap="round" stroke-linejoin="round"/></svg>'
                'Demo data active. Go to Clinical Audit tab and ask a question.</div>',
                unsafe_allow_html=True,
            )

    return st.session_state.get("_flat_file_adapter") or st.session_state.get("_last_conn_test")

    st.markdown(
        '<div style="font-size:13px;color:#5C7A89;margin-bottom:16px;">'
        'Connect MIRA to this hospital\'s patient database. '
        'Only <strong>SELECT</strong> access is required — MIRA never writes to your database.'
        '</div>',
        unsafe_allow_html=True,
    )

    db_type = st.selectbox(
        "Database type",
        ["PostgreSQL", "MySQL", "SQL Server", "SQLite (dev/local)"],
        index=0,
        key="setup_db_type",
    )

    placeholder_map = {
        "PostgreSQL":        "postgresql://user:password@host:5432/hospital_db",
        "MySQL":             "mysql+pymysql://user:password@host:3306/hospital_db",
        "SQL Server":        "mssql+pyodbc://user:password@host/hospital_db?driver=ODBC+Driver+17+for+SQL+Server",
        "SQLite (dev/local)":"sqlite:///./mira_data/mimic.db",
    }

    conn_str = st.text_input(
        "Connection string",
        placeholder=placeholder_map.get(db_type, ""),
        type="password",
        key="setup_conn_str",
        help="Your credentials are used only to open the connection — they are never stored in MIRA logs.",
    )

    col_test, col_save, _ = st.columns([1, 1, 3])

    test_result = None
    with col_test:
        if st.button("🔌 Test Connection", use_container_width=True,
                     key="btn_test_conn", disabled=not conn_str):
            with st.spinner("Connecting…"):
                try:
                    from adapters.db import DBAdapter
                    adapter = DBAdapter(conn_str)
                    test_result = adapter.test_connection()
                    st.session_state["_last_conn_test"] = test_result
                    st.session_state["_last_conn_str_display"] = conn_str[:20] + "***"
                except Exception as e:
                    test_result = {"ok": False, "error": str(e),
                                   "latency_ms": 0, "mapped": {}, "unmapped": [],
                                   "table_count": 0, "db_type": "unknown"}
                    st.session_state["_last_conn_test"] = test_result

    with col_save:
        last = st.session_state.get("_last_conn_test", {})
        if st.button("💾 Save & Apply", use_container_width=True,
                     key="btn_save_conn",
                     disabled=not (conn_str and last.get("ok"))):
            import os
            # Write to .env (dev) — in production this would go to Secret Manager
            env_path = ".env"
            lines = []
            if os.path.exists(env_path):
                with open(env_path) as f:
                    lines = [l for l in f.readlines()
                             if not l.startswith("MIRA_DEFAULT_DB_URL")]
            lines.append(f"MIRA_DEFAULT_DB_URL={conn_str}\n")
            with open(env_path, "w") as f:
                f.writelines(lines)
            st.session_state["_conn_saved"] = True
            st.toast("Connection saved — restart the app to apply.", icon="💾")

    # Show cached result if available
    cached = st.session_state.get("_last_conn_test")
    if cached is not None:
        render_connection_status(cached)

    if st.session_state.get("_conn_saved"):
        st.markdown(
            '<div class="banner banner-approved">'
            '<svg class="check" viewBox="0 0 10 8" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M1 4L3.5 6.5L9 1" stroke="currentColor" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>'
            'Connection string saved to .env. Restart MIRA to connect to the new database.</div>',
            unsafe_allow_html=True,
        )

    return test_result
