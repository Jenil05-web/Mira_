"""
audit_logger.py
================
MIRA Production — HIPAA-Compliant Audit Logger

WHAT THIS SOLVES:
  HIPAA requires that every access to Protected Health Information (PHI)
  is logged with: who accessed it, when, what they did, and from where.
  This is not optional — HIPAA Security Rule §164.312(b).

  This file creates an append-only audit trail for every:
    - User login / logout
    - Clinical query submitted
    - SQL query executed by Agent 1
    - FHIR data accessed
    - Agent decision (which agent ran, what it returned)
    - Human review decision (approve / reject + feedback)
    - Final report generated
    - Any error or anomaly

  "Append-only" means rows are INSERT-only — no UPDATE, no DELETE.
  This satisfies HIPAA's requirement for tamper-evident logs.

STORAGE:
  Dev:  SQLite (same mira.db file, separate table)
  Prod: Supabase PostgreSQL (same free-tier DB, mira_audit_log table)
        Row-Level Security (RLS) ensures only admin role can SELECT from it.

WHAT WE LOG (never log actual PHI values — only metadata):
  ✅ user_id, hospital_id, session_id
  ✅ action type (query / agent_run / review / login)
  ✅ agent name, tool name
  ✅ timestamp, duration_ms
  ✅ success / error flag
  ✅ row count returned (not the actual data)
  ❌ actual patient data (never logged)
  ❌ lab values, names, diagnoses (never logged)
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# AUDIT EVENT TYPES
# ══════════════════════════════════════════════════════════════════════════

class AuditEvent:
    LOGIN          = "user_login"
    LOGOUT         = "user_logout"
    QUERY_SUBMIT   = "query_submitted"
    AGENT_RUN      = "agent_run"
    TOOL_CALL      = "tool_call"
    DATA_ACCESS    = "data_access"
    HUMAN_REVIEW   = "human_review"
    REPORT_FINAL   = "report_finalized"
    ERROR          = "error"
    ADMIN_ACTION   = "admin_action"


# ══════════════════════════════════════════════════════════════════════════
# AUDIT LOGGER
# ══════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Append-only audit trail.
    All writes go through _write() which is INSERT-only — no updates ever.
    """

    def __init__(self, connection_string: str, enabled: bool = True):
        self.connection_string = connection_string
        self.enabled = enabled
        if enabled:
            self.engine = create_engine(connection_string)
            self._ensure_table()

    # ── Table setup ──────────────────────────────────────────────────────
    def _ensure_table(self):
        """Creates the audit log table if it doesn't exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS mira_audit_log (
            id              TEXT PRIMARY KEY,
            timestamp       TEXT NOT NULL,
            event_type      TEXT NOT NULL,
            user_id         TEXT,
            hospital_id     TEXT,
            session_id      TEXT,
            thread_id       TEXT,
            agent_name      TEXT,
            tool_name       TEXT,
            action_detail   TEXT,
            rows_returned   INTEGER,
            duration_ms     INTEGER,
            success         INTEGER NOT NULL DEFAULT 1,
            error_message   TEXT,
            ip_address      TEXT,
            metadata        TEXT
        )
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(ddl))
                conn.commit()
        except Exception as e:
            logger.warning(f"AuditLogger: could not create table: {e}")

    # ── Core write method ────────────────────────────────────────────────
    def _write(self, event_type: str, **kwargs):
        """INSERT a single audit record. Never UPDATE or DELETE."""
        if not self.enabled:
            return

        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "user_id": kwargs.get("user_id"),
            "hospital_id": kwargs.get("hospital_id"),
            "session_id": kwargs.get("session_id"),
            "thread_id": kwargs.get("thread_id"),
            "agent_name": kwargs.get("agent_name"),
            "tool_name": kwargs.get("tool_name"),
            "action_detail": kwargs.get("action_detail"),
            "rows_returned": kwargs.get("rows_returned"),
            "duration_ms": kwargs.get("duration_ms"),
            "success": 1 if kwargs.get("success", True) else 0,
            "error_message": kwargs.get("error_message"),
            "ip_address": kwargs.get("ip_address"),
            "metadata": json.dumps(kwargs.get("metadata", {}), default=str),
        }

        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO mira_audit_log
                    (id, timestamp, event_type, user_id, hospital_id, session_id,
                     thread_id, agent_name, tool_name, action_detail, rows_returned,
                     duration_ms, success, error_message, ip_address, metadata)
                    VALUES
                    (:id, :timestamp, :event_type, :user_id, :hospital_id, :session_id,
                     :thread_id, :agent_name, :tool_name, :action_detail, :rows_returned,
                     :duration_ms, :success, :error_message, :ip_address, :metadata)
                """), record)
                conn.commit()
        except Exception as e:
            # Audit log failure should never crash the application.
            # Log to standard logger as fallback.
            logger.error(f"AuditLogger write failed: {e} | Record: {record}")

    # ── Structured log methods ───────────────────────────────────────────
    def log_login(self, user_id: str, hospital_id: str,
                  session_id: str, ip_address: str = ""):
        self._write(
            AuditEvent.LOGIN,
            user_id=user_id, hospital_id=hospital_id,
            session_id=session_id, ip_address=ip_address,
            action_detail="user_authenticated",
        )

    def log_logout(self, user_id: str, session_id: str):
        self._write(
            AuditEvent.LOGOUT,
            user_id=user_id, session_id=session_id,
            action_detail="user_session_ended",
        )

    def log_query(self, user_id: str, hospital_id: str,
                  session_id: str, thread_id: str,
                  question: str, question_length: int):
        """Log a clinical query submission — not the question text itself (could be PHI)."""
        self._write(
            AuditEvent.QUERY_SUBMIT,
            user_id=user_id, hospital_id=hospital_id,
            session_id=session_id, thread_id=thread_id,
            action_detail="clinical_query_submitted",
            metadata={"question_char_length": question_length},
        )

    def log_agent_run(self, agent_name: str, thread_id: str,
                      duration_ms: int, success: bool,
                      rows_returned: int = 0, error: str = "",
                      user_id: str = "", hospital_id: str = ""):
        self._write(
            AuditEvent.AGENT_RUN,
            agent_name=agent_name, thread_id=thread_id,
            duration_ms=duration_ms, success=success,
            rows_returned=rows_returned,
            error_message=error if not success else None,
            user_id=user_id, hospital_id=hospital_id,
        )

    def log_tool_call(self, tool_name: str, thread_id: str,
                      duration_ms: int, rows_returned: int,
                      success: bool, error: str = "",
                      user_id: str = ""):
        """
        Log that a tool was called and how many rows it returned.
        We log row COUNT, not actual row data — HIPAA safe.
        """
        self._write(
            AuditEvent.TOOL_CALL,
            tool_name=tool_name, thread_id=thread_id,
            duration_ms=duration_ms, rows_returned=rows_returned,
            success=success,
            error_message=error if not success else None,
            user_id=user_id,
        )

    def log_data_access(self, data_source: str, resource_type: str,
                        patient_count: int, thread_id: str,
                        user_id: str = "", hospital_id: str = ""):
        """
        Log that PHI was accessed — what type of data, from which source,
        for how many patients. Never log actual PHI values.
        """
        self._write(
            AuditEvent.DATA_ACCESS,
            user_id=user_id, hospital_id=hospital_id,
            thread_id=thread_id,
            action_detail=f"accessed_{resource_type}_from_{data_source}",
            rows_returned=patient_count,
        )

    def log_human_review(self, user_id: str, thread_id: str,
                         decision: str, had_feedback: bool,
                         hospital_id: str = ""):
        """Log clinician approve/reject decision."""
        self._write(
            AuditEvent.HUMAN_REVIEW,
            user_id=user_id, hospital_id=hospital_id,
            thread_id=thread_id,
            action_detail=f"clinician_decision_{decision}",
            metadata={"had_revision_feedback": had_feedback},
        )

    def log_report_finalized(self, user_id: str, thread_id: str,
                              approved_by_critic: bool, safety_flags: list,
                              hospital_id: str = ""):
        """Log final report generation."""
        self._write(
            AuditEvent.REPORT_FINAL,
            user_id=user_id, hospital_id=hospital_id,
            thread_id=thread_id,
            success=approved_by_critic,
            action_detail="final_report_generated",
            metadata={
                "critic_approved": approved_by_critic,
                "safety_flag_count": len(safety_flags),
                "safety_flags": safety_flags,
            },
        )

    def log_error(self, error: str, thread_id: str = "",
                  user_id: str = "", agent_name: str = ""):
        self._write(
            AuditEvent.ERROR,
            user_id=user_id, thread_id=thread_id,
            agent_name=agent_name, success=False,
            error_message=error[:500],
        )

    # ── Context manager: auto-time + log an operation ────────────────────
    @contextmanager
    def timed_agent(self, agent_name: str, thread_id: str,
                    user_id: str = "", hospital_id: str = ""):
        """
        Usage:
            with audit.timed_agent("sql_extractor", thread_id, user_id):
                result = agent1_sql_extractor(state)
        Automatically logs duration and success/failure.
        """
        start = time.monotonic()
        try:
            yield
            duration = int((time.monotonic() - start) * 1000)
            self.log_agent_run(
                agent_name=agent_name, thread_id=thread_id,
                duration_ms=duration, success=True,
                user_id=user_id, hospital_id=hospital_id,
            )
        except Exception as e:
            duration = int((time.monotonic() - start) * 1000)
            self.log_agent_run(
                agent_name=agent_name, thread_id=thread_id,
                duration_ms=duration, success=False,
                error=str(e), user_id=user_id, hospital_id=hospital_id,
            )
            raise

    # ── Admin: read recent audit records ────────────────────────────────
    def get_recent_logs(self, limit: int = 50,
                        user_id: str = "", event_type: str = "") -> list[dict]:
        """
        Returns recent audit records for the admin dashboard.
        In production, this is only accessible to admin role users.
        """
        if not self.enabled:
            return []

        where_clauses = []
        params = {"limit": limit}

        if user_id:
            where_clauses.append("user_id = :user_id")
            params["user_id"] = user_id
        if event_type:
            where_clauses.append("event_type = :event_type")
            params["event_type"] = event_type

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(f"""
                    SELECT id, timestamp, event_type, user_id, hospital_id,
                           agent_name, tool_name, action_detail, rows_returned,
                           duration_ms, success, error_message
                    FROM mira_audit_log
                    {where_sql}
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """), params).fetchall()
                return [dict(r._mapping) for r in rows]
        except Exception as e:
            logger.error(f"get_recent_logs failed: {e}")
            return []

    def get_stats(self, hospital_id: str = "") -> dict:
        """Returns aggregate statistics for the admin dashboard."""
        if not self.enabled:
            return {}

        params = {}
        where = ""
        if hospital_id:
            where = "WHERE hospital_id = :hospital_id"
            params["hospital_id"] = hospital_id

        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(f"""
                    SELECT
                        COUNT(*) as total_events,
                        COUNT(DISTINCT session_id) as total_sessions,
                        COUNT(DISTINCT user_id) as unique_users,
                        SUM(CASE WHEN event_type='query_submitted' THEN 1 ELSE 0 END) as total_queries,
                        SUM(CASE WHEN event_type='human_review' THEN 1 ELSE 0 END) as total_reviews,
                        SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as total_errors,
                        AVG(CASE WHEN agent_name IS NOT NULL THEN duration_ms END) as avg_agent_ms
                    FROM mira_audit_log {where}
                """), params).fetchone()
                return dict(rows._mapping) if rows else {}
        except Exception as e:
            logger.error(f"get_stats failed: {e}")
            return {}


# ══════════════════════════════════════════════════════════════════════════
# SUPABASE RLS POLICY SQL (run once on your Supabase project)
# ══════════════════════════════════════════════════════════════════════════

SUPABASE_RLS_SQL = """
-- Enable Row Level Security on the audit log
ALTER TABLE mira_audit_log ENABLE ROW LEVEL SECURITY;

-- Only service role (admin) can select audit logs
CREATE POLICY "admin_read_audit" ON mira_audit_log
    FOR SELECT USING (auth.role() = 'service_role');

-- Anyone authenticated can insert (needed for logging from agents)
-- But only the service role can read — even users can't read their own logs
CREATE POLICY "insert_audit" ON mira_audit_log
    FOR INSERT WITH CHECK (true);

-- Nobody can update or delete audit records — ever
-- (no UPDATE/DELETE policies = those operations are blocked by RLS)
"""


if __name__ == "__main__":
    print("Testing AuditLogger with SQLite...\n")
    audit = AuditLogger("sqlite:///./mira_data/mimic.db")

    audit.log_login("user_123", "hospital_abc", "session_xyz", "127.0.0.1")
    audit.log_query("user_123", "hospital_abc", "session_xyz", "thread_001",
                    "Which patients have abnormal creatinine?", 42)
    audit.log_agent_run("sql_extractor", "thread_001", 1240, True, rows_returned=12)
    audit.log_tool_call("sql_query", "thread_001", 850, 12, True)
    audit.log_human_review("user_123", "thread_001", "approve", False)
    audit.log_report_finalized("user_123", "thread_001", True, [])

    logs = audit.get_recent_logs(limit=10)
    print(f"Logged {len(logs)} events:\n")
    for log in logs:
        print(f"  [{log['timestamp'][:19]}] {log['event_type']:25} user={log['user_id']}")

    stats = audit.get_stats()
    print(f"\nStats: {stats}")