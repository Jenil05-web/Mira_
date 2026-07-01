"""
config_manager.py
==================
MIRA Production — Configuration & Secrets Manager

WHAT THIS SOLVES:
  Dev: API keys in .env file, paths hardcoded, SQLite.
  Production: keys in GCP Secret Manager, PostgreSQL on Supabase,
  multi-hospital config, no secrets ever in code or files on disk.

PRIORITY ORDER (ConfigManager reads in this order, first found wins):
  1. GCP Secret Manager  (production — fully managed, free tier: 6 secrets)
  2. Environment variables (any deployment, CI/CD, Docker)
  3. .env file (local dev)

GCP SECRET MANAGER FREE TIER:
  - 6 active secret versions free
  - 10,000 access operations/month free
  - No credit card needed for free tier
  - https://cloud.google.com/secret-manager/pricing

SUPABASE FREE TIER:
  - 500MB PostgreSQL database
  - pgvector extension included (replaces FAISS)
  - Built-in auth + RLS (row-level security)
  - https://supabase.com/pricing

INSTALL:
  pip install python-dotenv
  pip install google-cloud-secret-manager  # only needed in GCP production
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Try to load .env if present ──────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════
# SECRET BACKENDS
# ══════════════════════════════════════════════════════════════════════════

class EnvSecretBackend:
    """Reads secrets from environment variables."""

    def get(self, key: str) -> Optional[str]:
        return os.environ.get(key)


class GCPSecretBackend:
    """
    Reads secrets from GCP Secret Manager.
    Only initialized when GCP credentials are available — gracefully
    falls back to env vars if not running in GCP.

    Setup (one-time, free):
      gcloud services enable secretmanager.googleapis.com
      gcloud secrets create OPENAI_API_KEY --data-file=-
      echo -n "sk-..." | gcloud secrets versions add OPENAI_API_KEY --data-file=-
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google.cloud import secretmanager
            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def get(self, key: str) -> Optional[str]:
        try:
            client = self._get_client()
            name = f"projects/{self.project_id}/secrets/{key}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.debug(f"GCP secret '{key}' not found: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════
# CONFIG MANAGER
# ══════════════════════════════════════════════════════════════════════════

class ConfigManager:
    """
    Central configuration for a MIRA deployment.
    One instance per running application.

    Usage:
        cfg = ConfigManager()               # auto-detects environment
        cfg = ConfigManager(gcp_project="my-gcp-project-id")  # GCP prod
    """

    def __init__(self, gcp_project: Optional[str] = None):
        self._backends = [EnvSecretBackend()]
        if gcp_project:
            self._backends.insert(0, GCPSecretBackend(gcp_project))
        elif os.environ.get("GCP_PROJECT_ID"):
            self._backends.insert(0, GCPSecretBackend(os.environ["GCP_PROJECT_ID"]))

        self._cache: dict = {}
        self._validate()

    # ── Secret resolution ────────────────────────────────────────────────
    def get(self, key: str, default: str = "") -> str:
        if key in self._cache:
            return self._cache[key]

        for backend in self._backends:
            val = backend.get(key)
            if val:
                self._cache[key] = val
                return val

        if default:
            return default

        logger.warning(f"Config key '{key}' not found in any backend.")
        return ""

    def require(self, key: str) -> str:
        """Like get() but raises if missing — for keys that must exist."""
        val = self.get(key)
        if not val:
            raise ValueError(
                f"Required config '{key}' not found. "
                f"Set it as an environment variable or in .env"
            )
        return val

    # ── Structured config accessors ──────────────────────────────────────
    @property
    def openai_api_key(self) -> str:
        return self.require("OPENAI_API_KEY")

    @property
    def supabase_url(self) -> str:
        return self.get("SUPABASE_URL")

    @property
    def supabase_anon_key(self) -> str:
        return self.get("SUPABASE_ANON_KEY")

    @property
    def supabase_service_role_key(self) -> str:
        return self.get("SUPABASE_SERVICE_ROLE_KEY")

    @property
    def deployment_env(self) -> str:
        return self.get("MIRA_ENV", "development")

    @property
    def is_production(self) -> bool:
        return self.deployment_env == "production"

    def get_data_source(self, hospital_id: str = "default") -> dict:
        """
        Returns the data source config for a given hospital_id.
        In multi-tenant production each hospital has its own credentials.

        Config keys follow the pattern:
          MIRA_{HOSPITAL_ID}_SOURCE_TYPE   → fhir | db
          MIRA_{HOSPITAL_ID}_FHIR_URL      → FHIR base URL
          MIRA_{HOSPITAL_ID}_FHIR_AUTH     → open | bearer | smart_oauth2
          MIRA_{HOSPITAL_ID}_DB_URL        → SQLAlchemy connection string
        """
        prefix = f"MIRA_{hospital_id.upper()}"

        source_type = self.get(f"{prefix}_SOURCE_TYPE", "db")

        if source_type == "fhir":
            return {
                "type": "fhir",
                "base_url": self.get(f"{prefix}_FHIR_URL",
                                     "http://hapi.fhir.org/baseR4"),
                "auth_mode": self.get(f"{prefix}_FHIR_AUTH", "open"),
                "token": self.get(f"{prefix}_FHIR_TOKEN", ""),
                "client_id": self.get(f"{prefix}_FHIR_CLIENT_ID", ""),
                "client_secret": self.get(f"{prefix}_FHIR_CLIENT_SECRET", ""),
                "token_url": self.get(f"{prefix}_FHIR_TOKEN_URL", ""),
            }
        else:
            db_url = self.get(
                f"{prefix}_DB_URL",
                self.get("DATABASE_URL", "sqlite:///./mira_data/mimic.db")
            )
            return {"type": "db", "connection_string": db_url}

    def get_vector_store_config(self) -> dict:
        """
        Returns vector store config.
        Dev: FAISS (local files)
        Prod: Supabase pgvector (same DB, zero extra cost)
        """
        if self.supabase_url:
            return {
                "type": "pgvector",
                "connection_string": self._supabase_db_url(),
                "table": "mira_embeddings",
                "dimension": 1536,
            }
        return {
            "type": "faiss",
            "index_path": self.get("FAISS_INDEX_PATH", "./mira_data/medical_faiss.index"),
            "metadata_path": self.get("FAISS_META_PATH", "./mira_data/faiss_metadata.pkl"),
        }

    def get_checkpoint_config(self) -> dict:
        """
        LangGraph checkpoint store config.
        Dev: MemorySaver (in-memory, lost on restart)
        Prod: PostgreSQL via Supabase (persistent across restarts)
        """
        if self.supabase_url:
            return {
                "type": "postgres",
                "connection_string": self._supabase_db_url(),
            }
        return {"type": "memory"}

    def _supabase_db_url(self) -> str:
        """Builds a SQLAlchemy-compatible Supabase PostgreSQL URL."""
        raw = self.supabase_url
        if raw.startswith("https://"):
            ref = raw.replace("https://", "").replace(".supabase.co", "")
            password = self.supabase_service_role_key or self.supabase_anon_key
            return f"postgresql://postgres:{password}@db.{ref}.supabase.co:5432/postgres"
        return raw

    def get_audit_config(self) -> dict:
        return {
            "enabled": self.get("AUDIT_LOGGING_ENABLED", "true").lower() == "true",
            "table": "mira_audit_log",
            "connection_string": (
                self._supabase_db_url() if self.supabase_url
                else self.get("DATABASE_URL", "sqlite:///./mira_data/mimic.db")
            ),
        }

    # ── Startup validation ───────────────────────────────────────────────
    def _validate(self):
        """Checks required config on startup so failures are explicit."""
        try:
            _ = self.openai_api_key
        except ValueError as e:
            raise ValueError(
                f"MIRA startup failed: {e}\n"
                f"Create a .env file with:\n  OPENAI_API_KEY=sk-your-key-here"
            )

    def describe(self) -> str:
        """Returns a safe summary of current config (no secret values)."""
        lines = [
            f"MIRA Configuration:",
            f"  Environment     : {self.deployment_env}",
            f"  OpenAI key      : {'✓ set' if self.openai_api_key else '✗ missing'}",
            f"  Supabase        : {'✓ connected' if self.supabase_url else '✗ not configured (using SQLite)'}",
            f"  Vector store    : {self.get_vector_store_config()['type']}",
            f"  Checkpoint store: {self.get_checkpoint_config()['type']}",
            f"  Audit logging   : {self.get_audit_config()['enabled']}",
            f"  GCP backend     : {'✓ active' if len(self._backends) > 1 else '✗ not configured (using env)'}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# .env.example generator — writes safe example for new hospital deployments
# ══════════════════════════════════════════════════════════════════════════

ENV_EXAMPLE = """# MIRA Production Configuration
# Copy this to .env and fill in your values.
# Never commit .env to git.

# ── Required ──────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-your-openai-key-here
MIRA_ENV=development

# ── Supabase (free tier — https://supabase.com) ───────────────────────────
# Get these from your Supabase project dashboard → Settings → API
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# ── GCP Secret Manager (optional, production only) ───────────────────────
# GCP_PROJECT_ID=your-gcp-project-id

# ── Data source (one per hospital deployment) ─────────────────────────────

# Option A: FHIR endpoint (Epic/Cerner/HAPI)
# MIRA_DEFAULT_SOURCE_TYPE=fhir
# MIRA_DEFAULT_FHIR_URL=http://hapi.fhir.org/baseR4
# MIRA_DEFAULT_FHIR_AUTH=open

# Option B: Direct database (MIMIC-IV dev / hospital PostgreSQL)
MIRA_DEFAULT_SOURCE_TYPE=db
MIRA_DEFAULT_DB_URL=sqlite:///./mira_data/mimic.db

# Option C: Production hospital database
# MIRA_DEFAULT_DB_URL=postgresql://user:pass@host:5432/hospital_db

# ── Audit logging ─────────────────────────────────────────────────────────
AUDIT_LOGGING_ENABLED=true

# ── Auth (JWT secrets — generate with: python -c "import secrets; print(secrets.token_hex(32))")
MIRA_JWT_SECRET=your-random-32-char-hex-string
MIRA_JWT_ALGORITHM=HS256
MIRA_ACCESS_TOKEN_EXPIRE_MINUTES=480
"""


if __name__ == "__main__":
    Path(".env.example").write_text(ENV_EXAMPLE)
    print("✅ .env.example written\n")

    cfg = ConfigManager()
    print(cfg.describe())