"""
fhir_adapter.py
================
MIRA Production — FHIR R4 Adapter

The single most important file for making MIRA "plug and play."

WHAT THIS SOLVES:
  Every hospital runs a different EHR (Epic, Cerner, Oracle Health, Meditech).
  Each has a different database schema. Without this file, every new hospital
  sale requires a custom integration project — weeks of work, high cost.

  FHIR (Fast Healthcare Interoperability Resources) is the international
  standard (HL7 FHIR R4) that ALL major EHRs are now legally required to
  expose (US 21st Century Cures Act, 2021). Epic, Cerner, and Oracle Health
  all have FHIR R4 endpoints.

  With this adapter: hospital IT gives us one URL + one OAuth2 credential.
  MIRA connects in minutes. No schema mapping. No custom SQL. Plug and play.

FHIR RESOURCES WE USE:
  Patient      → demographics (id, gender, birthDate)
  Observation  → lab results (the FHIR equivalent of labevents)
  Condition    → diagnoses (the FHIR equivalent of diagnoses_icd)
  Encounter    → hospital visits (the FHIR equivalent of admissions)

SUPPORTED EHR SYSTEMS (via their FHIR R4 endpoints):
  Epic          → https://<hospital>.epic.com/interconnect-fhir-oauth/api/FHIR/R4
  Cerner        → https://fhir-ehr.cerner.com/r4/<tenant-id>
  HAPI FHIR     → http://hapi.fhir.org/baseR4  (open test server, no auth)
  MIMIC-IV FHIR → https://mimic.mit.edu/docs/iv/modules/fhir/ (research)
  Azure Health  → https://<workspace>.azurehealthcareapis.com
  GCP FHIR      → https://healthcare.googleapis.com/v1/projects/.../fhir (free tier)

FREE FHIR SERVERS FOR DEVELOPMENT (no credentials needed):
  HAPI Public  → http://hapi.fhir.org/baseR4
  SMART Health → https://r4.smarthealthit.org
  Use these to develop and test before connecting to a real hospital.

AUTH MODES SUPPORTED:
  1. None (open servers like HAPI public)
  2. Bearer token (pre-obtained)
  3. SMART on FHIR OAuth2 (Epic/Cerner production) — client_credentials flow

INSTALL:
  pip install requests

NO paid dependencies. FHIR is an open standard.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# FHIR AUTH HANDLER
# ══════════════════════════════════════════════════════════════════════════

class FHIRAuthHandler:
    """
    Handles all three auth modes MIRA supports:
      - open: no auth (dev/test servers)
      - bearer: static token (simple integrations)
      - smart_oauth2: SMART on FHIR client_credentials (Epic/Cerner production)
    """

    def __init__(self, mode: str = "open", token: str = "",
                 client_id: str = "", client_secret: str = "",
                 token_url: str = ""):
        self.mode = mode
        self._static_token = token
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._cached_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

    def get_headers(self) -> dict:
        if self.mode == "open":
            return {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json"}

        if self.mode == "bearer":
            return {
                "Authorization": f"Bearer {self._static_token}",
                "Accept": "application/fhir+json",
            }

        if self.mode == "smart_oauth2":
            return {
                "Authorization": f"Bearer {self._get_oauth2_token()}",
                "Accept": "application/fhir+json",
            }

        return {"Accept": "application/fhir+json"}

    def _get_oauth2_token(self) -> str:
        """SMART on FHIR client_credentials flow — auto-refreshes on expiry."""
        if self._cached_token and self._token_expires_at:
            if datetime.utcnow() < self._token_expires_at - timedelta(seconds=60):
                return self._cached_token

        resp = requests.post(
            self._token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "scope": "system/*.read",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        self._cached_token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        return self._cached_token


# ══════════════════════════════════════════════════════════════════════════
# FHIR ADAPTER — main class
# ══════════════════════════════════════════════════════════════════════════

class FHIRAdapter:
    """
    Connects MIRA to any FHIR R4 endpoint.

    Usage:
        # Dev / testing (HAPI public, no auth)
        adapter = FHIRAdapter(base_url="http://hapi.fhir.org/baseR4")

        # Production (Epic with SMART OAuth2)
        adapter = FHIRAdapter(
            base_url="https://hospital.epic.com/interconnect-fhir-oauth/api/FHIR/R4",
            auth=FHIRAuthHandler(
                mode="smart_oauth2",
                client_id="...", client_secret="...",
                token_url="https://hospital.epic.com/.../oauth2/token"
            )
        )

        # Query
        patient = adapter.get_patient("12345")
        labs = adapter.get_observations(patient_id="12345", category="laboratory")
    """

    def __init__(self, base_url: str, auth: Optional[FHIRAuthHandler] = None,
                 timeout: int = 20, max_pages: int = 5):
        self.base_url = base_url.rstrip("/")
        self.auth = auth or FHIRAuthHandler(mode="open")
        self.timeout = timeout
        self.max_pages = max_pages  # FHIR returns paginated bundles

    # ── Core HTTP ────────────────────────────────────────────────────────
    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = requests.get(url, headers=self.auth.get_headers(),
                            params=params or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _get_all_pages(self, path: str, params: dict = None) -> list[dict]:
        """Follows FHIR pagination (Bundle.link[rel=next]) to get all entries."""
        entries = []
        url = f"{self.base_url}/{path.lstrip('/')}"
        page = 0

        while url and page < self.max_pages:
            resp = requests.get(url, headers=self.auth.get_headers(),
                                params=(params if page == 0 else {}),
                                timeout=self.timeout)
            resp.raise_for_status()
            bundle = resp.json()

            for entry in bundle.get("entry", []):
                entries.append(entry.get("resource", {}))

            url = next(
                (link["url"] for link in bundle.get("link", []) if link.get("relation") == "next"),
                None
            )
            page += 1

        return entries

    # ── Capability / health check ────────────────────────────────────────
    def ping(self) -> dict:
        """Fetch the server's CapabilityStatement — confirms connectivity."""
        try:
            cap = self._get("metadata")
            return {
                "connected": True,
                "fhir_version": cap.get("fhirVersion", "unknown"),
                "software": cap.get("software", {}).get("name", "unknown"),
                "resources_supported": [
                    r["type"] for r in cap.get("rest", [{}])[0].get("resource", [])
                ][:10]
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    # ── Patient ──────────────────────────────────────────────────────────
    def get_patient(self, patient_id: str) -> Optional[dict]:
        """Fetch a single patient by FHIR id."""
        try:
            resource = self._get(f"Patient/{patient_id}")
            return self._normalize_patient(resource)
        except Exception as e:
            logger.warning(f"get_patient({patient_id}): {e}")
            return None

    def search_patients(self, family: str = "", given: str = "",
                        birthdate: str = "", limit: int = 10) -> list[dict]:
        """Search patients by name or birthdate."""
        params = {"_count": limit}
        if family:
            params["family"] = family
        if given:
            params["given"] = given
        if birthdate:
            params["birthdate"] = birthdate

        resources = self._get_all_pages("Patient", params)
        return [self._normalize_patient(r) for r in resources if r]

    def _normalize_patient(self, r: dict) -> dict:
        """Flatten FHIR Patient resource to a flat dict matching MIRA's expected shape."""
        names = r.get("name", [{}])
        name = names[0] if names else {}
        return {
            "patient_id": r.get("id", ""),
            "gender": r.get("gender", ""),
            "birth_date": r.get("birthDate", ""),
            "family_name": name.get("family", ""),
            "given_name": " ".join(name.get("given", [])),
            "active": r.get("active", True),
            "_raw": r,
        }

    # ── Observations (lab results) ───────────────────────────────────────
    def get_observations(self, patient_id: str, category: str = "laboratory",
                         code: str = "", date_from: str = "",
                         limit: int = 50) -> list[dict]:
        """
        Fetch lab results (Observations) for a patient.
        category='laboratory' → lab tests (equivalent to labevents)
        category='vital-signs' → vitals (equivalent to chartevents)
        code → specific LOINC code e.g. '2160-0' for Creatinine
        """
        params = {
            "patient": patient_id,
            "category": category,
            "_count": limit,
            "_sort": "-date",
        }
        if code:
            params["code"] = code
        if date_from:
            params["date"] = f"ge{date_from}"

        resources = self._get_all_pages("Observation", params)
        return [self._normalize_observation(r) for r in resources if r]

    def _normalize_observation(self, r: dict) -> dict:
        """Flatten FHIR Observation to a flat dict matching MIRA's sql_result row shape."""
        code_obj = r.get("code", {})
        codings = code_obj.get("coding", [{}])
        code = codings[0] if codings else {}

        value = None
        value_unit = ""
        value_num = None
        ref_lower = None
        ref_upper = None

        if "valueQuantity" in r:
            vq = r["valueQuantity"]
            value = str(vq.get("value", ""))
            value_num = vq.get("value")
            value_unit = vq.get("unit", "")
        elif "valueString" in r:
            value = r["valueString"]
        elif "valueCodeableConcept" in r:
            value = r["valueCodeableConcept"].get("text", "")

        ref_ranges = r.get("referenceRange", [{}])
        if ref_ranges:
            rr = ref_ranges[0]
            ref_lower = rr.get("low", {}).get("value")
            ref_upper = rr.get("high", {}).get("value")

        flag = "abnormal" if r.get("interpretation") else None
        if r.get("interpretation"):
            interp_codes = r["interpretation"][0].get("coding", [{}])
            if interp_codes:
                interp_code = interp_codes[0].get("code", "")
                if interp_code in ("H", "HH", "L", "LL", "A", "AA"):
                    flag = "abnormal"

        return {
            "subject_id": r.get("subject", {}).get("reference", "").replace("Patient/", ""),
            "lab_name": code.get("display") or code_obj.get("text", ""),
            "loinc_code": code.get("code", ""),
            "value": value,
            "valuenum": value_num,
            "valueuom": value_unit,
            "charttime": r.get("effectiveDateTime", r.get("effectivePeriod", {}).get("start", "")),
            "flag": flag,
            "ref_range_lower": ref_lower,
            "ref_range_upper": ref_upper,
            "status": r.get("status", ""),
            "_raw": r,
        }

    def get_abnormal_observations(self, patient_id: str = "") -> list[dict]:
        """
        Returns all abnormal lab observations.
        In FHIR, abnormal = interpretation code H/HH/L/LL/A.
        This is the FHIR equivalent of WHERE flag='abnormal' in your SQL.
        """
        params = {
            "category": "laboratory",
            "_count": 50,
            "_sort": "-date",
        }
        if patient_id:
            params["patient"] = patient_id

        resources = self._get_all_pages("Observation", params)
        normalized = [self._normalize_observation(r) for r in resources if r]
        return [o for o in normalized if o["flag"] == "abnormal"]

    # ── Conditions (diagnoses) ───────────────────────────────────────────
    def get_conditions(self, patient_id: str, limit: int = 20) -> list[dict]:
        """Fetch diagnoses — equivalent to diagnoses_icd table."""
        params = {"patient": patient_id, "_count": limit}
        resources = self._get_all_pages("Condition", params)
        return [self._normalize_condition(r) for r in resources if r]

    def _normalize_condition(self, r: dict) -> dict:
        code_obj = r.get("code", {})
        codings = code_obj.get("coding", [{}])
        code = codings[0] if codings else {}
        return {
            "subject_id": r.get("subject", {}).get("reference", "").replace("Patient/", ""),
            "condition": code_obj.get("text", ""),
            "icd_code": code.get("code", ""),
            "system": code.get("system", ""),
            "clinical_status": r.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", ""),
            "onset": r.get("onsetDateTime", ""),
            "_raw": r,
        }

    # ── Encounters (admissions) ──────────────────────────────────────────
    def get_encounters(self, patient_id: str, limit: int = 10) -> list[dict]:
        """Fetch hospital visits — equivalent to admissions table."""
        params = {"patient": patient_id, "_count": limit, "_sort": "-date"}
        resources = self._get_all_pages("Encounter", params)
        return [self._normalize_encounter(r) for r in resources if r]

    def _normalize_encounter(self, r: dict) -> dict:
        period = r.get("period", {})
        type_list = r.get("type", [{}])
        enc_type = type_list[0].get("text", "") if type_list else ""
        return {
            "subject_id": r.get("subject", {}).get("reference", "").replace("Patient/", ""),
            "encounter_id": r.get("id", ""),
            "status": r.get("status", ""),
            "type": enc_type,
            "class": r.get("class", {}).get("code", ""),
            "admit_time": period.get("start", ""),
            "discharge_time": period.get("end", ""),
            "_raw": r,
        }

    # ── High-level MIRA query methods ────────────────────────────────────
    def get_patient_full_context(self, patient_id: str) -> dict:
        """
        Single call that returns everything Agent 1 needs for one patient.
        Returns the same shape as your SQL queries do — so Agent 3 works
        identically regardless of whether the data came from FHIR or SQL.
        """
        return {
            "patient": self.get_patient(patient_id),
            "observations": self.get_observations(patient_id, limit=30),
            "conditions": self.get_conditions(patient_id),
            "encounters": self.get_encounters(patient_id, limit=5),
        }

    def search_abnormal_labs_across_patients(self, loinc_codes: list[str] = None) -> list[dict]:
        """
        The FHIR equivalent of MIRA's 'find patients with abnormal creatinine' query.
        Searches all Observations with abnormal interpretations.
        loinc_codes: optional filter e.g. ['2160-0'] for creatinine only.

        Note: not all FHIR servers support searching without a patient param.
        HAPI public supports it; Epic may require a patient context.
        """
        params = {
            "category": "laboratory",
            "interpretation": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation|H,"
                              "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation|L,"
                              "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation|A",
            "_count": 50,
            "_sort": "-date",
        }
        if loinc_codes:
            params["code"] = ",".join(f"http://loinc.org|{c}" for c in loinc_codes)

        try:
            resources = self._get_all_pages("Observation", params)
            return [self._normalize_observation(r) for r in resources if r]
        except Exception as e:
            logger.warning(f"search_abnormal_labs_across_patients: {e}")
            return []

    # ── Schema description for Agent 1's system prompt ───────────────────
    def get_schema_description(self) -> str:
        """
        Returns a natural-language description of the FHIR data available —
        injected into Agent 1's system prompt exactly like db_schema.txt.
        The agent uses this to construct FHIR queries instead of SQL.
        """
        cap = self.ping()
        supported = cap.get("resources_supported", [])

        return f"""FHIR R4 DATA SOURCE — {self.base_url}
Server: {cap.get('software', 'unknown')} (FHIR {cap.get('fhir_version', 'R4')})
Supported resources: {', '.join(supported)}

AVAILABLE DATA:

RESOURCE: Patient
  Fields: patient_id, gender, birth_date, family_name, given_name
  Query method: get_patient(patient_id) or search_patients(family, given, birthdate)

RESOURCE: Observation (lab results + vital signs)
  Fields: subject_id, lab_name, loinc_code, value, valuenum, valueuom,
          charttime, flag (abnormal/normal), ref_range_lower, ref_range_upper
  Query method: get_observations(patient_id, category='laboratory')
  For abnormal only: get_abnormal_observations(patient_id)
  Common LOINC codes:
    2160-0  Creatinine
    2345-7  Glucose
    777-3   Platelets
    4544-3  Hematocrit
    2823-3  Potassium
    2951-2  Sodium
    1742-6  ALT
    1920-8  AST
    1975-2  Total Bilirubin

RESOURCE: Condition (diagnoses)
  Fields: subject_id, condition, icd_code, clinical_status, onset
  Query method: get_conditions(patient_id)

RESOURCE: Encounter (hospital visits)
  Fields: subject_id, encounter_id, status, type, class, admit_time, discharge_time
  Query method: get_encounters(patient_id)

NOTE: This is a FHIR endpoint, not a SQL database. Use the Python method names
above as your "query language." The data shape of the results matches what you
would expect from SQL: each result is a list of flat dicts."""

    # ── Utility: convert FHIR results to SQL-compatible JSON string ──────
    def results_to_sql_format(self, resources: list[dict]) -> str:
        """
        Converts FHIR results to the same JSON format that sql_query tool returns.
        This means Agent 3 and Agent 4 work identically with FHIR or SQL data.
        """
        return json.dumps({"rows": resources, "total_returned": len(resources)}, default=str)


# ══════════════════════════════════════════════════════════════════════════
# LOINC CODE LOOKUP — common clinical lab codes
# ══════════════════════════════════════════════════════════════════════════

LOINC_CODES = {
    "creatinine": "2160-0",
    "glucose": "2345-7",
    "platelets": "777-3",
    "hematocrit": "4544-3",
    "hemoglobin": "718-7",
    "potassium": "2823-3",
    "sodium": "2951-2",
    "chloride": "2075-0",
    "bicarbonate": "1963-8",
    "bun": "3094-0",
    "alt": "1742-6",
    "ast": "1920-8",
    "bilirubin": "1975-2",
    "albumin": "1751-7",
    "lactate": "2519-9",
    "wbc": "6690-2",
    "inr": "34714-6",
    "troponin": "42757-5",
    "bnp": "42637-9",
    "procalcitonin": "75241-0",
    "crp": "1988-5",
}


def loinc(lab_name: str) -> Optional[str]:
    """Quick lookup helper: loinc('creatinine') → '2160-0'"""
    return LOINC_CODES.get(lab_name.lower().strip())


# ══════════════════════════════════════════════════════════════════════════
# CLI smoke test — python fhir_adapter.py
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing FHIR adapter against HAPI public server (no auth needed)...\n")

    adapter = FHIRAdapter(base_url="http://hapi.fhir.org/baseR4")

    print("1. Ping / CapabilityStatement:")
    cap = adapter.ping()
    print(f"   Connected: {cap['connected']}")
    print(f"   Server: {cap.get('software', 'N/A')}")
    print(f"   FHIR version: {cap.get('fhir_version', 'N/A')}")
    print(f"   Resources: {cap.get('resources_supported', [])[:5]}")

    print("\n2. Schema description (what Agent 1 sees):")
    print(adapter.get_schema_description()[:600])

    print("\n3. Search for patients named 'Smith':")
    patients = adapter.search_patients(family="Smith", limit=3)
    for p in patients[:2]:
        print(f"   {p['patient_id']}: {p['given_name']} {p['family_name']}, {p['gender']}, DOB {p['birth_date']}")

    if patients:
        pid = patients[0]["patient_id"]
        print(f"\n4. Get lab observations for patient {pid}:")
        labs = adapter.get_observations(pid, category="laboratory", limit=5)
        for lab in labs[:3]:
            print(f"   {lab['lab_name']}: {lab['value']} {lab['valueuom']} (flag: {lab['flag']})")