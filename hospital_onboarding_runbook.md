# MIRA — Hospital Onboarding Runbook

> **Who uses this**: MIRA ops/engineering on the day a new hospital contract is signed.
> Run every step in order. Do not skip or reorder.

---

## Phase 0 — Pre-Onboarding Checklist (before touching any credentials)

- [ ] BAA (Business Associate Agreement) **signed and countersigned** — stop here if not done
- [ ] Hospital point-of-contact confirmed (name, email, on-call number)
- [ ] Data source type confirmed: **SQL** (PostgreSQL / MySQL / SQLite) or **FHIR** (R4 endpoint)
- [ ] Network path confirmed: VPN, IP allowlist, or public endpoint?
- [ ] Read-only service account / API key provisioned by the hospital

---

## Phase 1 — Credential Collection

### 1a — SQL hospital
Collect and store in environment / secrets manager (never commit to git):

```
HOSPITAL_<ID>_CONN_STR = "postgresql://readonly_user:password@host:5432/db"
```

Required: the account must have `SELECT` on all clinical tables. Verify:
```sql
-- Run this as the provided account — it must succeed and return rows
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
```

### 1b — FHIR hospital
Collect:
```
HOSPITAL_<ID>_FHIR_URL   = "https://ehr.hospital.org/fhir/R4"
HOSPITAL_<ID>_FHIR_TOKEN = "<bearer_token>"
```

Verify the endpoint is live and returns a valid CapabilityStatement:
```bash
curl -H "Authorization: Bearer <token>" \
     https://ehr.hospital.org/fhir/R4/metadata | jq .fhirVersion
# expected: "4.0.1"
```

---

## Phase 2 — Config Registration

Add the new hospital to `config/hospitals.yaml`:

```yaml
hospitals:
  <hospital_id>:
    type: sql
    connection_string: "postgresql://..."
    name: "City General Hospital"
    enabled: false        # start disabled — flip in Phase 6
```

Then verify the adapter loads and the connection test passes:

```bash
python -c "
from pipeline.engine import get_engine
e = get_engine()
a = e._get_adapter('<hospital_id>')
print('Adapter loaded OK:', type(a).__name__)
"
```

A `ValueError` or `ConnectionError` here means credentials or config are wrong — fix before proceeding.

---

## Phase 3 — Schema Introspection

```bash
python -c "
from pipeline.engine import get_engine
import json
e = get_engine()
a = e._get_adapter('<hospital_id>')
schema = a.discover_schema(force_refresh=True)
print('Tables found:', list(schema.keys()))
print(json.dumps(a._table_map, indent=2))
"
```

**Check:**
- [ ] A `patients` table (or alias) was mapped
- [ ] A `labevents` / `observations` table was mapped
- [ ] `subject_id` / patient ID column was detected
- [ ] `valuenum` / numeric value column was detected

If mappings are missing, update `CORE_TABLE_CONCEPTS` and `CONCEPT_MAP` in `adapters/db.py`.

---

## Phase 4 — Run the Eval Suite Against the Sandbox

> Never run against the production DB with real patient data.

Add hospital-specific test cases to `eval/test_cases.yaml` by duplicating relevant cases with `hospital_id: <hospital_id>`, then run:

```bash
python eval/run_eval.py --fail-fast
```

Expected result: **all cases pass**.

| Failure | Action |
|---|---|
| TC-001 patient not found | Confirm a real patient ID exists in sandbox |
| TC-006/6b injection not blocked | Check safety gate wired in this adapter path |
| TC-007/8 AKI/sepsis vocab miss | Add aliases to `pipeline/tools.py` CONDITION_VOCAB |

---

## Phase 5 — Manual Review (staging only)

Run 10–15 manual queries. For each:

- [ ] Correct patient identified (check against source record)
- [ ] Lab values match (spot-check 3 values against the EHR)
- [ ] Clinical reasoning is coherent (clinician review)
- [ ] Safety flags fire when expected
- [ ] No PHI appears in audit logs (`mira_audit_log`)

Get written sign-off from a clinician reviewer before Phase 6.

---

## Phase 6 — Production Flip

```yaml
# config/hospitals.yaml
hospitals:
  <hospital_id>:
    enabled: true    # was: false
```

```bash
git add config/hospitals.yaml
git commit -m "onboard: enable <hospital_name> (<hospital_id>)"
git push
# trigger Render redeploy
```

Check deployment logs for:
```
Adapter loaded OK for hospital '<hospital_id>'
```

If you see a `ConnectionError`, the prod DB is not reachable — fix network/allowlist before go-live.

---

## Phase 7 — Watch the First 50 Real Queries

```sql
SELECT timestamp, user_id, event_type, action_detail, success, error_message
FROM mira_audit_log
WHERE hospital_id = '<hospital_id>'
ORDER BY timestamp DESC
LIMIT 50;
```

Watch for:
- [ ] `success = false` on any `AGENT_RUN` event
- [ ] `data_status = security_rejected` on legitimate queries (safety gate too aggressive?)
- [ ] `data_status = patient_not_found` more than expected (patient ID format mismatch?)
- [ ] `Circuit breaker OPENED` log line (DB is flaky — escalate to hospital IT)

Set up a Slack/email alert on `error_message IS NOT NULL` during the first week.

---

## Phase 8 — Handoff

- [ ] Update internal wiki with hospital schema map
- [ ] Share eval suite additions with hospital IT
- [ ] Schedule 30-day check-in with clinical contact
- [ ] Archive BAA, credentials handover email, and clinician sign-off doc

---

## Quick-Reference: Key Files

| File | Purpose |
|---|---|
| `adapters/db.py` | SQL schema introspection + query execution |
| `adapters/fhir.py` | FHIR R4 adapter |
| `pipeline/tools.py` | `CONDITION_VOCAB`, `CORE_TABLE_CONCEPTS` |
| `eval/run_eval.py` | Regression suite runner |
| `eval/test_cases.yaml` | Test case definitions |
| `core/audit.py` | HIPAA audit log writer |
| `pipeline/engine.py` | `_get_adapter()` — startup validation + circuit breaker |

---

## Emergency: Roll Back a Hospital

```yaml
hospitals:
  <hospital_id>:
    enabled: false   # immediate kill-switch
```

Deploy the change. The circuit breaker (`_CB_FAILURE_THRESH = 3`) also auto-trips if the hospital's DB fails repeatedly — check logs for `Circuit breaker OPENED for hospital`.
