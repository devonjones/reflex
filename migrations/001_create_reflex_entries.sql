-- Migration 001: Create reflex_entries table
-- Description: Store metadata for all captured entries
-- Date: 2026-01-12

CREATE TABLE IF NOT EXISTS reflex_entries (
    id BIGSERIAL PRIMARY KEY,

    -- Discord metadata
    discord_message_id TEXT UNIQUE NOT NULL,
    discord_channel_id TEXT NOT NULL,
    discord_user_id TEXT NOT NULL,

    -- Classification
    category TEXT NOT NULL CHECK (category IN ('person', 'project', 'idea', 'admin', 'inbox')),
    title TEXT NOT NULL,
    tags TEXT[],

    -- LLM metadata
    llm_confidence REAL NOT NULL CHECK (llm_confidence >= 0.0 AND llm_confidence <= 1.0),
    llm_model TEXT NOT NULL,
    llm_reasoning TEXT,

    -- Status
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),

    -- Timestamps
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Markdown export tracking
    exported_to_git BOOLEAN DEFAULT FALSE,
    git_commit_sha TEXT,
    markdown_path TEXT
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_reflex_category ON reflex_entries(category);
CREATE INDEX IF NOT EXISTS idx_reflex_captured_at ON reflex_entries(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_reflex_discord_channel ON reflex_entries(discord_channel_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_reflex_status ON reflex_entries(status);

-- GIN index for tag searches
CREATE INDEX IF NOT EXISTS idx_reflex_tags ON reflex_entries USING GIN(tags);

-- Trigger to update updated_at on row modification
CREATE OR REPLACE FUNCTION update_reflex_entries_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_reflex_entries_updated_at ON reflex_entries;
CREATE TRIGGER trigger_reflex_entries_updated_at
    BEFORE UPDATE ON reflex_entries
    FOR EACH ROW
    EXECUTE FUNCTION update_reflex_entries_updated_at();

-- Comments for documentation
COMMENT ON TABLE reflex_entries IS 'Metadata for all Reflex knowledge captures';
COMMENT ON COLUMN reflex_entries.category IS 'Classification: person, project, idea, admin, or inbox';
COMMENT ON COLUMN reflex_entries.llm_confidence IS 'LLM confidence score (0.0-1.0)';
COMMENT ON COLUMN reflex_entries.llm_model IS 'Model used for classification (e.g., qwen2.5:7b, gemini-1.5-flash)';
COMMENT ON COLUMN reflex_entries.status IS 'Entry status: active or archived';
