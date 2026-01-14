-- Migration: Add next_action_date for digest scheduling
-- Version: 1.2.0
-- Date: 2026-01-14

-- Up Migration
ALTER TABLE reflex_entries ADD COLUMN IF NOT EXISTS next_action_date TIMESTAMPTZ;

COMMENT ON COLUMN reflex_entries.next_action_date IS 'When to next show this entry in digest. NULL = show immediately, future date = snoozed';

-- Down Migration (for rollback)
-- ALTER TABLE reflex_entries DROP COLUMN IF EXISTS next_action_date;
