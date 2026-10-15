"""
MIRA India — Trust, DPDP, and PHI-safe schema helpers.

Used in sales conversations ("why should we give you our database?")
and at runtime so sample rows never leak names/phones into LLM prompts.
"""

from __future__ import annotations

import re
from typing import Any

PHI_COLUMN_HINTS = {
    "patient_name", "name", "full_name", "pat_name", "given_name",
    "family_name", "first_name", "last_name", "person_name", "p_name",
    "phone", "mobile", "phone_no", "mobile_no", "contact_no",
    "phone_number", "cell_phone", "address", "aadhaar", "aadhar",
    "email", "guardian", "relative", "father_name", "husband_name",
}


TRUST_PITCH = [
    {
        "claim": "Read-only access",
        "detail": "MIRA only runs SELECT queries. INSERT/UPDATE/DELETE/DROP are blocked in code.",
    },
    {
        "claim": "Your database stays yours",
        "detail": "Hospital keeps the source of truth. We connect; we do not copy the full HIS.",
    },
    {
        "claim": "Excel never written to our disk",
        "detail": "Uploaded sheets load into in-memory SQLite for that session only.",
    },
    {
        "claim": "No PHI in audit logs",
        "detail": "Logs store who/when/what-action and row counts — not names, labs, or IDs.",
    },
    {
        "claim": "Clinician sign-off",
        "detail": "Every report pauses for human review before it is finalized.",
    },
    {
        "claim": "Hospital isolation",
        "detail": "A user at Clinic A cannot query Clinic B. JWT is bound to hospital_id.",
    },
]


DPDP_CHECKLIST = [
    "Sign a data processing addendum under DPDP Act 2023 (hospital = Data Fiduciary, MIRA = Data Processor).",
    "Collect purpose limitation in writing: clinical decision support + audit trail only.",
    "Use a hospital-issued read-only account; never a DBA / sa / postgres superuser.",
    "Prefer on-premise or hospital VPC for live HIS. Cloud is optional for demo/Excel only.",
    "Do not train foundation models on hospital patient rows.",
    "Provide audit export on request (who queried, when, success/fail).",
    "Kill-switch: disable hospital_id in config if access must stop immediately.",
]


SALES_OBJECTION_DB_ACCESS = """
We do not take your database.

1. IT creates a read-only SQL user with SELECT on clinical tables only.
2. That user is IP-allowlisted to the MIRA server (or we run MIRA inside your LAN).
3. MIRA auto-maps tables (UHID, labs, admissions) and shows the map on screen — you approve it.
4. Queries are SELECT-only and logged. We store metadata, not patient files.
5. You can revoke the user in 1 minute. MIRA then cannot read anything.

If you have no HIS — export Excel/CSV. We load it in memory, never save the file.
"""


def is_phi_column(name: str) -> bool:
    n = re.sub(r"[^\w]", "_", (name or "").strip().lower()).strip("_")
    if n in PHI_COLUMN_HINTS:
        return True
    return any(h in n for h in ("name", "phone", "mobile", "address", "aadhaar", "aadhar", "email"))


def redact_sample_row(row: dict | None, max_fields: int = 4) -> dict:
    """Return a prompt-safe sample: types/shape only, no identifiers."""
    if not row:
        return {}
    out = {}
    for i, (k, v) in enumerate(row.items()):
        if i >= max_fields:
            break
        if is_phi_column(str(k)):
            out[k] = "[redacted]"
        elif v is None:
            out[k] = None
        else:
            out[k] = _shape(v)
    return out


def _shape(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "number"
    s = str(v)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", s):
        return "date"
    if len(s) > 12:
        return f"text({len(s)} chars)"
    return "text"


def login_trust_html() -> str:
    items = "".join(
        f"<li style='margin:0 0 8px 0;'><strong>{c['claim']}</strong> — {c['detail']}</li>"
        for c in TRUST_PITCH
    )
    return (
        "<div style='text-align:left;font-size:12.5px;color:#5C7A89;line-height:1.45;'>"
        "<div style='font-size:11px;letter-spacing:0.08em;text-transform:uppercase;"
        "color:#8FA3AE;margin:18px 0 8px;'>Why hospitals can trust MIRA</div>"
        f"<ul style='padding-left:18px;margin:0;'>{items}</ul>"
        "<div style='margin-top:10px;font-size:11.5px;'>Aligned with DPDP Act 2023 processor duties. "
        "HIPAA-style audit trail included for NABH / enterprise buyers.</div>"
        "</div>"
    )
