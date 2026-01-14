-- Migration: Add actionable field for digest filtering
-- Version: 1.3.0
-- Date: 2026-01-14

-- Up Migration
ALTER TABLE reflex_entries ADD COLUMN IF NOT EXISTS actionable BOOLEAN DEFAULT FALSE;

COMMENT ON COLUMN reflex_entries.actionable IS 'Whether this entry requires user action (tasks, reminders, etc.). LLM-classified during capture.';

-- Set actionable=true for existing admin/project entries (heuristic migration)
-- The Python migration (migrate_to_1_3_0) handles this per-entry with logging
UPDATE reflex_entries SET actionable = true WHERE category IN ('admin', 'project') AND actionable = false;

-- Down Migration (for rollback)
-- ALTER TABLE reflex_entries DROP COLUMN IF EXISTS actionable;
