-- ============================================================
-- supabase_setup.sql
-- MIRA Production — Run this ONCE in your Supabase SQL Editor
-- Supabase dashboard → SQL Editor → New Query → paste → Run
-- ============================================================
-- Creates:
--   1. mira_audit_log        HIPAA append-only audit trail
--   2. mira_embeddings       pgvector index (replaces FAISS)
--   3. mira_checkpoints      LangGraph state persistence
--   4. mira_hospitals        Hospital tenant registry
--   5. mira_users            App user registry
--   6. RLS policies          Row-level security for all tables
-- ============================================================


-- ── 0. Enable required extensions ────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;       -- pgvector (replaces FAISS)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- UUID generation


-- ════════════════════════════════════════════════════════════════════════
-- 1. HOSPITAL REGISTRY (multi-tenant)
-- Each hospital that buys MIRA gets one row here.
-- ════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS mira_hospitals (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name             TEXT NOT NULL,
    slug             TEXT NOT NULL UNIQUE,  -- used in env var prefix e.g. MIRA_CITYGENERAL_*
    source_type      TEXT NOT NULL DEFAULT 'db',  -- 'fhir' | 'db'
    fhir_url         TEXT,
    fhir_auth_mode   TEXT DEFAULT 'open',
    db_url_secret    TEXT,  -- name of GCP secret holding the DB URL (never stored plain)
    plan             TEXT NOT NULL DEFAULT 'trial',  -- trial | starter | enterprise
    active           BOOLEAN NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed: demo hospital for development
INSERT INTO mira_hospitals (name, slug, source_type, plan)
VALUES ('Demo Hospital', 'demo', 'db', 'trial')
ON CONFLICT (slug) DO NOTHING;


-- ════════════════════════════════════════════════════════════════════════
-- 2. USER REGISTRY
-- Clinicians and admins — one row per MIRA user.
-- In production, Supabase Auth handles passwords — this table only
-- stores role/hospital assignment linked to the Supabase auth.users.id.
-- ════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS mira_users (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    auth_user_id     TEXT UNIQUE,        -- links to Supabase auth.users.id
    email            TEXT NOT NULL UNIQUE,
    display_name     TEXT,
    role             TEXT NOT NULL DEFAULT 'clinician',  -- 'clinician' | 'admin'
    hospital_id      UUID NOT NULL REFERENCES mira_hospitals(id),
    active           BOOLEAN NOT NULL DEFAULT TRUE,
    last_login_at    TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed: dev users (hospital_id = demo hospital)
INSERT INTO mira_users (email, display_name, role, hospital_id)
SELECT
    'clinician@mira.dev',
    'Dr. Demo Clinician',
    'clinician',
    (SELECT id FROM mira_hospitals WHERE slug = 'demo')
ON CONFLICT (email) DO NOTHING;

INSERT INTO mira_users (email, display_name, role, hospital_id)
SELECT
    'admin@mira.dev',
    'MIRA Admin',
    'admin',
    (SELECT id FROM mira_hospitals WHERE slug = 'demo')
ON CONFLICT (email) DO NOTHING;


-- ════════════════════════════════════════════════════════════════════════
-- 3. AUDIT LOG (HIPAA — append-only, no UPDATE/DELETE ever)
-- ════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS mira_audit_log (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type       TEXT NOT NULL,
    user_id          TEXT,
    hospital_id      TEXT,
    session_id       TEXT,
    thread_id        TEXT,
    agent_name       TEXT,
    tool_name        TEXT,
    action_detail    TEXT,
    rows_returned    INTEGER,
    duration_ms      INTEGER,
    success          BOOLEAN NOT NULL DEFAULT TRUE,
    error_message    TEXT,
    ip_address       TEXT,
    metadata         JSONB DEFAULT '{}'
);

-- Index for fast admin dashboard queries
CREATE INDEX IF NOT EXISTS idx_audit_timestamp
    ON mira_audit_log (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_user
    ON mira_audit_log (user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_hospital
    ON mira_audit_log (hospital_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_event
    ON mira_audit_log (event_type, timestamp DESC);


-- ════════════════════════════════════════════════════════════════════════
-- 4. MEDICAL EMBEDDINGS (replaces local FAISS index)
-- Stores the embedded medical guideline chunks.
-- pgvector cosine search replaces faiss.IndexFlatL2.
-- ════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS mira_embeddings (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source           TEXT NOT NULL,      -- e.g. "Surviving Sepsis Campaign 2021"
    topic            TEXT NOT NULL,      -- e.g. "Sepsis — Hour-1 Bundle"
    content          TEXT NOT NULL,      -- the guideline text chunk
    embedding        vector(1536),       -- OpenAI text-embedding-3-small dimension
    hospital_id      TEXT DEFAULT 'global',  -- 'global' = all hospitals share it
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- IVFFlat index for fast approximate nearest-neighbor search
-- (exact search is fine for <10k vectors; switch to IVFFlat at scale)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
    ON mira_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);

CREATE INDEX IF NOT EXISTS idx_embeddings_hospital
    ON mira_embeddings (hospital_id);


-- ════════════════════════════════════════════════════════════════════════
-- 5. LANGGRAPH CHECKPOINTS (replaces MemorySaver)
-- Persists agent graph state across Streamlit reruns and app restarts.
-- LangGraph's PostgresSaver writes here automatically.
-- ════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id        TEXT NOT NULL,
    checkpoint_ns    TEXT NOT NULL DEFAULT '',
    checkpoint_id    TEXT NOT NULL,
    parent_config    JSONB,
    type             TEXT,
    checkpoint       JSONB NOT NULL,
    metadata         JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id        TEXT NOT NULL,
    checkpoint_ns    TEXT NOT NULL DEFAULT '',
    channel          TEXT NOT NULL,
    version          TEXT NOT NULL,
    type             TEXT NOT NULL,
    blob             BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id        TEXT NOT NULL,
    checkpoint_ns    TEXT NOT NULL DEFAULT '',
    checkpoint_id    TEXT NOT NULL,
    task_id          TEXT NOT NULL,
    idx              INTEGER NOT NULL,
    channel          TEXT NOT NULL,
    type             TEXT,
    blob             BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
    ON checkpoints (thread_id, checkpoint_ns);


-- ════════════════════════════════════════════════════════════════════════
-- 6. ROW LEVEL SECURITY (RLS)
-- Controls who can read/write each table at the database level.
-- This is enforced by Supabase even if application code has a bug.
-- ════════════════════════════════════════════════════════════════════════

-- ── Audit log ────────────────────────────────────────────────────────────
ALTER TABLE mira_audit_log ENABLE ROW LEVEL SECURITY;

-- Service role (backend) can INSERT
CREATE POLICY "service_insert_audit"
    ON mira_audit_log FOR INSERT
    TO service_role
    WITH CHECK (true);

-- Only service role can SELECT (admin dashboard uses service_role key)
CREATE POLICY "service_read_audit"
    ON mira_audit_log FOR SELECT
    TO service_role
    USING (true);

-- Nobody can UPDATE or DELETE audit records — ever
-- (absence of UPDATE/DELETE policies blocks them entirely under RLS)


-- ── Embeddings ───────────────────────────────────────────────────────────
ALTER TABLE mira_embeddings ENABLE ROW LEVEL SECURITY;

-- Service role can do everything (needed for seeding + search)
CREATE POLICY "service_all_embeddings"
    ON mira_embeddings FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);


-- ── Checkpoints ──────────────────────────────────────────────────────────
ALTER TABLE checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE checkpoint_blobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE checkpoint_writes ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_all_checkpoints"
    ON checkpoints FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_checkpoint_blobs"
    ON checkpoint_blobs FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "service_all_checkpoint_writes"
    ON checkpoint_writes FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ── Hospitals ─────────────────────────────────────────────────────────────
ALTER TABLE mira_hospitals ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_all_hospitals"
    ON mira_hospitals FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ── Users ────────────────────────────────────────────────────────────────
ALTER TABLE mira_users ENABLE ROW LEVEL SECURITY;

CREATE POLICY "service_all_users"
    ON mira_users FOR ALL TO service_role USING (true) WITH CHECK (true);


-- ════════════════════════════════════════════════════════════════════════
-- 7. HELPER FUNCTION — pgvector similarity search
-- Called by mira_pipeline_prod.py instead of faiss.index.search()
-- ════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE FUNCTION search_embeddings(
    query_embedding  vector(1536),
    match_count      INTEGER DEFAULT 3,
    hospital_filter  TEXT DEFAULT 'global'
)
RETURNS TABLE (
    id          UUID,
    source      TEXT,
    topic       TEXT,
    content     TEXT,
    similarity  FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        id, source, topic, content,
        1 - (embedding <=> query_embedding) AS similarity
    FROM mira_embeddings
    WHERE hospital_id = hospital_filter OR hospital_id = 'global'
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;


-- ════════════════════════════════════════════════════════════════════════
-- 8. AUDIT STATS VIEW — used by admin dashboard
-- ════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE VIEW mira_audit_stats AS
SELECT
    hospital_id,
    COUNT(*)                                                      AS total_events,
    COUNT(DISTINCT session_id)                                    AS total_sessions,
    COUNT(DISTINCT user_id)                                       AS unique_users,
    SUM(CASE WHEN event_type = 'query_submitted'   THEN 1 END)   AS total_queries,
    SUM(CASE WHEN event_type = 'human_review'      THEN 1 END)   AS total_reviews,
    SUM(CASE WHEN event_type = 'report_finalized'  THEN 1 END)   AS total_reports,
    SUM(CASE WHEN success = FALSE                  THEN 1 END)   AS total_errors,
    ROUND(AVG(duration_ms) FILTER (
        WHERE agent_name IS NOT NULL
    ))                                                            AS avg_agent_ms,
    MAX(timestamp)                                                AS last_activity
FROM mira_audit_log
GROUP BY hospital_id;


-- ════════════════════════════════════════════════════════════════════════
-- DONE — verify by running:
-- SELECT table_name FROM information_schema.tables
-- WHERE table_schema = 'public';
-- ════════════════════════════════════════════════════════════════════════