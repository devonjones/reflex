# Reflex Project Specification

## Overview

Reflex is a personal knowledge capture system ("second brain") that uses Discord as the zero-friction capture interface, with automatic two-tier LLM classification, dual storage (Postgres + DuckDB), markdown export to git, and proactive surfacing via daily/weekly digests.

## Core Innovation

Traditional note-taking systems require cognitive work at capture time (deciding where things go, organizing, tagging). Reflex moves that work to an automated background loop:

1. **Capture** - Drop thought in Discord (5 seconds, zero decisions)
2. **Classify** - LLM determines category automatically
3. **Store** - Postgres metadata + DuckDB full text
4. **Export** - Markdown file committed to git
5. **Surface** - Daily digest shows what matters

The human's job is reduced to a single reliable behavior: capture in Discord. Everything else is automation.

## Architecture

### Single Service: reflex-sync

**Why single service?**
- Human typing speed is ~1 msg/min (not email volume)
- All operations tightly coupled (capture → classify → store → export)
- No need for queue complexity or parallelization

**Components:**
- Discord bot listener (discord.py with gateway connection)
- Intent detector (heuristic + two-tier LLM validation)
- Classifier (two-tier LLM cascade: Qwen → Gemini/Claude)
- Storage writers (Postgres + DuckDB)
- Markdown exporter (async git commit)
- Digest generator (APScheduler: daily 7am, weekly Sunday 4pm)
- Command parser (natural language → structured actions)

### Storage Pattern

**Postgres** (`reflex_entries` table):
```sql
CREATE TABLE reflex_entries (
    id BIGSERIAL PRIMARY KEY,
    discord_message_id TEXT UNIQUE NOT NULL,
    discord_channel_id TEXT NOT NULL,
    discord_user_id TEXT NOT NULL,
    category TEXT NOT NULL,  -- person|project|idea|admin|inbox
    title TEXT NOT NULL,
    tags TEXT[],
    llm_confidence REAL NOT NULL,
    llm_model TEXT NOT NULL,
    llm_reasoning TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    exported_to_git BOOLEAN DEFAULT FALSE,
    git_commit_sha TEXT,
    markdown_path TEXT
);
```

**DuckDB** (`reflex_bodies` table):
```sql
CREATE TABLE reflex_bodies (
    entry_id BIGINT PRIMARY KEY,
    original_message TEXT NOT NULL,
    processed_body TEXT
);
```

**Markdown** (`~/cortex-reflex/` git repo):
```
cortex-reflex/
├── person/
│   └── 2026-01-12-alice-contact.md
├── project/
│   └── 2026-01-12-homelab-upgrade.md
├── idea/
│   └── 2026-01-12-auto-tagging.md
├── admin/
│   └── 2026-01-12-dentist-appt.md
└── inbox/
    └── 2026-01-13-article-rag.md
```

### Categories

- **person**: Information about people (relationships, contacts, notes)
- **project**: Work related to specific projects or goals
- **idea**: Random thoughts, inspirations, future possibilities
- **admin**: Tasks, errands, administrative matters
- **inbox**: External ideas needing review (staging for bookmarks, articles)

## Message Flow

### 1. Intent Detection (Command vs Capture)

```
Message arrives in #reflex
    ↓
Heuristic check
    ├─ Looks like command? (starts with move/tag/archive + references "that"/"last")
    │   ↓
    │   Qwen validation (confidence >= 0.7?)
    │   ├─ Yes, is_command=true → Parse command
    │   └─ No or uncertain → Tier 2 validation
    │       ├─ Gemini/Claude (confidence >= 0.6?)
    │       │   ├─ Yes, is_command=true → Parse command
    │       │   └─ Yes, is_command=false → Classify capture
    │       └─ Still uncertain → Ask user
    │
    └─ Looks like capture? → Skip to classification
```

### 2. Classification (Two-Tier Cascade)

```
Capture message
    ↓
Tier 1: Qwen 7b (local, fast, free)
    ├─ Confidence >= 0.7 → Store immediately
    └─ Confidence < 0.7 → Escalate
        ↓
        Tier 2: Gemini Flash or Claude Haiku (cloud, accurate, paid)
        ├─ Confidence >= 0.6 → Store
        └─ Confidence < 0.6 → Post to Discord asking for manual classification
```

### 3. Storage & Export

```
Classification result
    ↓
Transaction: Store in Postgres + DuckDB
    ↓
Async: Export markdown + git commit
    ↓
Discord confirmation (✅ reaction + thread reply)
```

## LLM Integration

### LiteLLM Proxy

All LLM calls go through litellm proxy at `http://ares.evilsoft:4000/`. This provides:
- Unified API for Ollama (Qwen) and cloud providers (Gemini/Claude)
- Request logging and metrics
- Rate limiting and retries
- No need for separate API clients

### Models

**Tier 1**: `ollama/qwen2.5:7b`
- Local inference on Ares (AMD GPU)
- ~1-2 second latency
- Free (no API costs)
- Good for 90% of classifications

**Tier 2**: `gemini/gemini-1.5-flash` (default) or `claude/claude-3-haiku`
- Cloud API via litellm
- ~500ms latency
- Paid (~$0.10/1M tokens)
- Only for ambiguous cases

### Prompt Templates

**Intent Validation**:
```
Is this a command about existing entries, or a new thought to capture?

**COMMAND**: References existing entries ("that idea", "last note") and wants to move/tag/archive.
**CAPTURE**: Everything else - new thoughts to remember.

Message: "{message}"

Return JSON:
{
  "is_command": true|false,
  "confidence": 0.0-1.0
}
```

**Capture Classification**:
```
Classify this note into one category:

- **person**: Information about a person
- **project**: Work related to a project or goal
- **idea**: Random thought, inspiration, future possibility
- **admin**: Task, errand, administrative matter
- **inbox**: External idea needing review

Note: "{message}"

Return JSON:
{
  "category": "...",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "suggested_tags": ["tag1", "tag2"]
}
```

## Natural Language Commands

### Examples
- "move that idea about Foo into my projects"
- "tag the last note with 'urgent'"
- "archive that inbox item"

### Target Resolution

**Recency-based with keyword fallback**:
1. Query last 10 messages from #reflex channel
2. Filter by category if mentioned (e.g., "that idea" → category=idea)
3. Filter by keywords if mentioned (e.g., "about Foo" → title/body contains "foo")
4. If single match → execute action
5. If multiple matches → ask user to clarify
6. If no matches → "couldn't find that entry"

### Actions
- **move**: Update category, re-export markdown to new folder
- **tag**: Append tags, re-export markdown with updated frontmatter
- **archive**: Set status=archived, move markdown to archive/ folder

## Digests

### Daily (7am)
```markdown
# Daily Digest - 2026-01-12

**Captures**: 5 notes

## By Category
- 💡 Ideas: 2
- 📋 Projects: 1
- 👤 Person: 1
- 📥 Inbox: 1

## Active Projects (with next actions)
1. **Homelab upgrade** - Next: Order new RAM
2. **Reflex system** - Next: Test classification
```

### Weekly (Sunday 4pm)
```markdown
# Weekly Digest - Week of 2026-01-06

**Captures**: 23 notes this week

## By Category
- 💡 Ideas: 8
- 📋 Projects: 6
- 👤 Person: 4
- 📥 Inbox: 3
- 📝 Admin: 2

## Patterns This Week
- You captured 5 ideas about automation
- Project "Homelab upgrade" mentioned 3 times
- 3 inbox items still need review

## Next Week Focus
- Review inbox items
- Progress on "Homelab upgrade"
```

## Metrics

Using `cortex_utils.metrics` patterns:

**Standard metrics**:
- `cortex_errors_total{service="reflex", error_type="llm_error|discord_error|storage_error"}`
- `cortex_processing_duration_seconds{operation="intent_detect|classify|store|export"}`

**Custom metrics**:
- `reflex_captures_total{category="person|project|idea|admin|inbox"}`
- `reflex_classification_duration_seconds{tier="tier1|tier2"}`
- `reflex_llm_confidence{model="qwen|gemini|claude"}`
- `reflex_exports_total{status="success|failed"}`
- `reflex_commands_total{action="move|tag|archive|show"}`

## Error Handling

### Discord Alerter

All exceptions sent to Discord webhook using `cortex_utils.alerter.discord`:

```python
from cortex_utils.alerter.discord import DiscordClient, COLOR_CRITICAL

client = DiscordClient(webhook_url=os.getenv("DISCORD_REFLEX_WEBHOOK"))
try:
    # ... operation ...
except Exception as e:
    client.send_embed(
        title="Reflex Classification Error",
        description=f"Failed to classify message: {e}",
        color=COLOR_CRITICAL,
        fields=[
            {"name": "Message ID", "value": message.id, "inline": True},
            {"name": "Channel", "value": message.channel.name, "inline": True},
        ],
        ping=True
    )
```

### Graceful Degradation

- **Qwen unreachable**: Skip to Tier 2 immediately
- **Tier 2 unreachable**: Store with low confidence, notify in Discord
- **Postgres down**: Log error, don't crash bot
- **Git push fails**: Log error, entry still in DB, retry on next export
- **Discord webhook fails**: Log error, don't crash bot

## Future Enhancements

### Phase 5: MCP Server
- Query tools (search by category, date, tags)
- RAG over markdown files
- Install in `~/.claude/mcp/reflex-server/`

### Phase 6: Email Integration
- Triage rule: self-sent emails → `reflex-capture` queue
- reflex-email-worker service
- Shared classifier code in cortex-utils

### Beyond
- Bidirectional sync (manual markdown edits → DB)
- Vector search over embeddings (semantic similarity)
- Attachment support (images, PDFs)
- Multi-user support (per-channel isolation)
- Web UI for browsing/editing
- Mobile app integration (iOS shortcuts → webhook)

## Reliability Checklist

### Critical
- [ ] Discord bot reconnection on disconnect
- [ ] Transaction handling for DB writes (all-or-nothing)
- [ ] Graceful degradation when LLM/DB unavailable

### High Priority
- [ ] Git push failures don't block capture
- [ ] Concurrent message handling (Discord sends fast)
- [ ] Markdown export conflicts (race conditions)

### Medium Priority
- [ ] APScheduler job failures logged to Discord
- [ ] Command ambiguity resolution (multiple matches)
- [ ] Low confidence handling (user asked for clarification)

### Observability
- [ ] All operations logged with structured logging
- [ ] Prometheus metrics exposed on :8096
- [ ] Discord webhook for critical errors
- [ ] Health check endpoint for monitoring
