"""
auth.py
========
MIRA Production — Authentication & Role-Based Access Control

WHAT THIS SOLVES:
  Dev MIRA has no auth — anyone who opens the URL can query any patient.
  Production needs: who is logged in, what role they have, which hospital
  they belong to, and whether they're allowed to do what they're trying to do.

ARCHITECTURE:
  - JWT tokens (stateless — no server-side session storage needed)
  - Two roles:
      clinician → can submit queries, review reports
      admin     → all clinician permissions + configure hospital DB,
                  view audit logs, manage users
  - Hospital isolation — a clinician at Hospital A cannot see Hospital B's data
  - Tokens expire after 8 hours (configurable)

FREE STACK:
  - Supabase Auth handles user management (email/password, SSO) for free
  - JWT validation done locally — no Supabase API call on every request
  - python-jose for JWT (lightweight, no heavy auth framework needed)

INSTALL:
  pip install python-jose[cryptography] passlib[bcrypt]
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning(
        "python-jose / passlib not installed. Auth is disabled. "
        "Run: pip install python-jose[cryptography] passlib[bcrypt]"
    )


# ══════════════════════════════════════════════════════════════════════════
# ROLES
# ══════════════════════════════════════════════════════════════════════════

class Role:
    CLINICIAN = "clinician"
    ADMIN     = "admin"

    ALL = {CLINICIAN, ADMIN}

    PERMISSIONS = {
        CLINICIAN: {
            "submit_query",
            "view_report",
            "review_report",
            "approve_report",
            "reject_report",
        },
        ADMIN: {
            "submit_query",
            "view_report",
            "review_report",
            "approve_report",
            "reject_report",
            "view_audit_log",
            "configure_datasource",
            "manage_users",
            "view_stats",
        },
    }

    @classmethod
    def can(cls, role: str, permission: str) -> bool:
        return permission in cls.PERMISSIONS.get(role, set())


# ══════════════════════════════════════════════════════════════════════════
# USER MODEL (lightweight, no ORM)
# ══════════════════════════════════════════════════════════════════════════

class User:
    def __init__(self, user_id: str, email: str, role: str,
                 hospital_id: str, display_name: str = ""):
        self.user_id = user_id
        self.email = email
        self.role = role
        self.hospital_id = hospital_id
        self.display_name = display_name or email.split("@")[0]

    def can(self, permission: str) -> bool:
        return Role.can(self.role, permission)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "role": self.role,
            "hospital_id": self.hospital_id,
            "display_name": self.display_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "User":
        return cls(
            user_id=d["user_id"], email=d["email"],
            role=d["role"], hospital_id=d["hospital_id"],
            display_name=d.get("display_name", ""),
        )


# ══════════════════════════════════════════════════════════════════════════
# TOKEN MANAGER
# ══════════════════════════════════════════════════════════════════════════

class TokenManager:
    """
    Issues and validates JWT access tokens.
    Secret key comes from ConfigManager (GCP Secret Manager in prod, .env in dev).
    """

    def __init__(self, secret_key: str,
                 algorithm: str = "HS256",
                 expire_minutes: int = 480):
        if not JWT_AVAILABLE:
            raise ImportError("python-jose not installed. "
                              "Run: pip install python-jose[cryptography]")

        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expire_minutes = expire_minutes

    def create_token(self, user: User) -> str:
        """Issue a JWT token for the given user."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user.user_id,
            "email": user.email,
            "role": user.role,
            "hospital_id": user.hospital_id,
            "display_name": user.display_name,
            "iat": now,
            "exp": now + timedelta(minutes=self.expire_minutes),
            "jti": str(uuid.uuid4()),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[User]:
        """
        Validates a JWT token and returns the User if valid.
        Returns None on any validation failure (expired, tampered, etc.).
        """
        try:
            payload = jwt.decode(token, self.secret_key,
                                 algorithms=[self.algorithm])
            return User(
                user_id=payload["sub"],
                email=payload["email"],
                role=payload["role"],
                hospital_id=payload["hospital_id"],
                display_name=payload.get("display_name", ""),
            )
        except JWTError as e:
            logger.warning(f"Token validation failed: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════
# PASSWORD MANAGER
# ══════════════════════════════════════════════════════════════════════════

class PasswordManager:
    """
    Bcrypt password hashing. Never store plain-text passwords.
    In production, Supabase Auth handles this — this is only used
    for local development or if you build your own user management.
    """

    def __init__(self):
        if not JWT_AVAILABLE:
            raise ImportError("passlib not installed. "
                              "Run: pip install passlib[bcrypt]")
        self._ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash(self, password: str) -> str:
        return self._ctx.hash(password)

    def verify(self, plain: str, hashed: str) -> bool:
        return self._ctx.verify(plain, hashed)


# ══════════════════════════════════════════════════════════════════════════
# IN-MEMORY USER STORE (dev only — replace with Supabase in production)
# ══════════════════════════════════════════════════════════════════════════

class DevUserStore:
    """
    Simple in-memory user store for local development.
    In production, Supabase Auth + Supabase DB handles this.

    Default dev users:
      clinician@mira.dev / mira_clinician_2024
      admin@mira.dev     / mira_admin_2024
    """

    def __init__(self):
        if not JWT_AVAILABLE:
            self._users = {}
            self._passwords = {}
            return

        pm = PasswordManager()
        self._users = {
            "clinician@mira.dev": User(
                user_id="dev_clinician_001",
                email="clinician@mira.dev",
                role=Role.CLINICIAN,
                hospital_id="demo_hospital",
                display_name="Dr. Demo Clinician",
            ),
            "admin@mira.dev": User(
                user_id="dev_admin_001",
                email="admin@mira.dev",
                role=Role.ADMIN,
                hospital_id="demo_hospital",
                display_name="MIRA Admin",
            ),
        }
        self._passwords = {
            "clinician@mira.dev": pm.hash("mira_clinician_2024"),
            "admin@mira.dev":     pm.hash("mira_admin_2024"),
        }

    def authenticate(self, email: str, password: str) -> Optional[User]:
        user = self._users.get(email)
        if not user:
            return None
        if not JWT_AVAILABLE:
            return None
        pm = PasswordManager()
        hashed = self._passwords.get(email, "")
        if not pm.verify(password, hashed):
            return None
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        return next((u for u in self._users.values() if u.user_id == user_id), None)


# ══════════════════════════════════════════════════════════════════════════
# AUTH MANAGER — facade used by streamlit_app.py
# ══════════════════════════════════════════════════════════════════════════

class AuthManager:
    """
    Facade for auth operations. Streamlit app only talks to this class.

    Usage in streamlit_app.py:
        auth = AuthManager(jwt_secret=cfg.get("MIRA_JWT_SECRET"))

        # Login form
        user = auth.login(email, password)
        if user:
            st.session_state.token = auth.create_token(user)
            st.session_state.user = user

        # On every page load: verify token
        user = auth.get_user_from_token(st.session_state.get("token", ""))
        if not user:
            show_login_screen()
            st.stop()

        # Permission check
        if not user.can("view_audit_log"):
            st.error("Admin access required.")
            st.stop()
    """

    def __init__(self, jwt_secret: str = "dev_secret_change_in_production",
                 expire_minutes: int = 480):
        self._user_store = DevUserStore()
        if JWT_AVAILABLE and jwt_secret:
            self._tokens = TokenManager(jwt_secret, expire_minutes=expire_minutes)
        else:
            self._tokens = None

    def login(self, email: str, password: str) -> Optional[User]:
        return self._user_store.authenticate(email, password)

    def create_token(self, user: User) -> str:
        if self._tokens:
            return self._tokens.create_token(user)
        return f"dev_token_{user.user_id}"

    def get_user_from_token(self, token: str) -> Optional[User]:
        if not token:
            return None
        if self._tokens:
            return self._tokens.verify_token(token)
        # Dev fallback: return demo user
        if token.startswith("dev_token_"):
            user_id = token.replace("dev_token_", "")
            return self._user_store.get_user(user_id)
        return None

    def is_authenticated(self, token: str) -> bool:
        return self.get_user_from_token(token) is not None


# ══════════════════════════════════════════════════════════════════════════
# STREAMLIT AUTH GATE helper — call at top of every page
# ══════════════════════════════════════════════════════════════════════════

def require_auth(auth_manager: "AuthManager") -> "User":
    """
    Call at the top of streamlit_app.py pages.
    If not authenticated, shows login form and calls st.stop().
    If authenticated, returns the User object.

    Usage:
        from auth import require_auth, AuthManager
        auth = AuthManager(jwt_secret=cfg.get("MIRA_JWT_SECRET"))
        user = require_auth(auth)
    """
    import streamlit as st

    if "auth_token" not in st.session_state:
        st.session_state.auth_token = ""
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None

    user = auth_manager.get_user_from_token(st.session_state.auth_token)

    if not user:
        _render_login_form(auth_manager)
        st.stop()

    return user


def _render_login_form(auth_manager: "AuthManager"):
    import streamlit as st

    st.markdown("""
    <style>
    .login-wrap { max-width: 380px; margin: 80px auto 0 auto; }
    </style>
    <div class="login-wrap">
    """, unsafe_allow_html=True)

    st.markdown("## Sign in to MIRA")
    st.caption("Multi-Agent Clinical Audit & Real-Time Triage System")
    st.markdown("")

    email = st.text_input("Email", placeholder="clinician@hospital.com", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    st.markdown("")

    if st.button("Sign in", type="primary", use_container_width=True):
        user = auth_manager.login(email, password)
        if user:
            st.session_state.auth_token = auth_manager.create_token(user)
            st.session_state.auth_user = user
            st.rerun()
        else:
            st.error("Invalid email or password.")

    st.markdown("")
    st.caption("Dev credentials: clinician@mira.dev / mira_clinician_2024")


# ══════════════════════════════════════════════════════════════════════════
# CLI smoke test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.environ["OPENAI_API_KEY"] = "test"

    print("Testing AuthManager...\n")
    auth = AuthManager(jwt_secret="test_secret_32_chars_minimum_ok")

    user = auth.login("clinician@mira.dev", "mira_clinician_2024")
    if user:
        print(f"✅ Login: {user.display_name} ({user.role})")
        token = auth.create_token(user)
        print(f"✅ Token issued: {token[:40]}...")
        recovered = auth.get_user_from_token(token)
        print(f"✅ Token verified: {recovered.email}")

        print(f"\nPermissions for {user.role}:")
        for perm in ["submit_query", "view_audit_log", "manage_users"]:
            print(f"  {perm}: {user.can(perm)}")
    else:
        print("❌ Login failed")

    admin = auth.login("admin@mira.dev", "mira_admin_2024")
    if admin:
        print(f"\n✅ Admin login: {admin.display_name}")
        print(f"  manage_users: {admin.can('manage_users')}")
        print(f"  view_audit_log: {admin.can('view_audit_log')}")
            