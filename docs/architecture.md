# Reflex Architecture

## Overview

Reflex is a personal knowledge capture system ("second brain") that reduces cognitive load by automating the organization and retrieval of thoughts, ideas, and information. It uses Discord as a zero-friction capture interface with automatic LLM-powered classification.

## Design Principles

### 1. Minimize Human Cognitive Load

**Traditional note-taking systems fail because they require cognitive work at capture time:**
- Deciding where a note belongs
- Choosing appropriate tags
- Maintaining organizational structure
- Remembering to review notes

**Reflex moves this work to automated background processes:**
- Capture: Drop thought in Discord (5 seconds, zero decisions)
- Classify: LLM determines category automatically
- Store: Dual storage (Postgres + DuckDB) handles persistence
- Export: Markdown files committed to git automatically
- Surface: Daily/weekly digests show relevant information proactively

The human's job is reduced to a single reliable behavior: **post in #reflex channel**.

### 2. Follow Existing Cortex Patterns

Reflex is part of the Cortex ecosystem and follows established patterns:

- **No cross-service imports**: All shared code in `cortex-utils`
- **Structured logging**: `cortex_utils.logging` for consistent JSON logs
- **Prometheus metrics**: `cortex_utils.metrics` with standard + custom metrics
- **Discord alerting**: `cortex_utils.alerter` for exception notifications
- **No hardcoded values**: All configuration via environment variables
- **LLM configuration**: Models and prompts configurable (not hardcoded)
- **Service structure**: Follows `gmail_sync.py` pattern (metrics server, graceful shutdown)

### 3. Use Existing Infrastructure

**DuckDB via Postmark API:**
- Postmark already has a DuckDB API server that handles write locking
- Reflex uses this API instead of direct DuckDB access
- Avoids duplicating infrastructure
- Future: Move DuckDB API to gateway for shared infrastructure (ticket cortex-xx94)

**LiteLLM Proxy:**
- Unified API for Ollama (local) and cloud providers (Gemini/Claude)
- Running at http://ares.evilsoft:4000
- Provides request logging, rate limiting, retries
- No need for separate API clients per provider

## System Architecture

### Single Service Design

**reflex-sync** is a single service containing all components:
- Discord bot listener (discord.py)
- Intent detector (heuristic + LLM validation)
- Classifier (two-tier LLM cascade)
- Storage writers (Postgres + DuckDB API)
- Markdown exporter (async git operations)
- Digest generator (APScheduler)
- Command parser (natural language → actions)

**Why single service instead of microservices?**
1. **Human typing speed is slow** (~1 msg/min, not email volume)
2. **Operations are tightly coupled** (capture → classify → store → export is one transaction)
3. **No parallelization benefits** at this scale
4. **Simpler deployment** (one container, one set of environment variables)

### Data Flow

```
Discord #reflex channel
    ↓
Message received by bot
    ↓
Heuristic: Command or Capture?
    ├─ Looks like command? → Intent validation (two-tier LLM)
    │   ├─ Confirmed command → Parse command → Execute action
    │   └─ Actually capture → Classification
    │
    └─ Looks like capture? → Classification (two-tier LLM)
        ↓
        Store in Postgres + DuckDB (via API)
        ↓
        Export markdown + git commit (async)
        ↓
        Discord confirmation (✅ + thread reply)
```

## Storage Architecture

### Three-Layer Storage

**1. Postgres (Structured Metadata)**
- Table: `reflex_entries`
- Purpose: Fast queries, filtering, aggregation
- Contains: category, title, tags, timestamps, LLM metadata
- Why: Enables complex queries for digests, command target resolution
- Schema: See `migrations/001_create_reflex_entries.sql`

**2. DuckDB (Full Text Bodies)**
- Accessed via: Postmark DuckDB API (http://cortex-duckdb-api:8081)
- Purpose: Store full message text without bloating Postgres
- Key: `entry_id` (maps to `reflex_entries.id`)
- Why: Better full-text search, follows existing Cortex pattern
- Future: Semantic search with vector embeddings

**3. Markdown Files (Git-Tracked Export)**
- Location: `~/cortex-reflex/` (separate git repo)
- Purpose: Portability, manual editing, RAG for Claude app
- Format: YAML frontmatter + markdown body
- Why: Human-readable, version-controlled, cross-tool compatibility

### Storage Decision Matrix

| Need | Storage | Reason |
|------|---------|--------|
| Fast category filtering | Postgres | Indexed category column |
| Recent captures (last 10) | Postgres | Indexed captured_at DESC |
| Full message text | DuckDB | Keeps Postgres lean, better FTS |
| Manual editing | Markdown | Human-readable files |
| Version history | Git | Track changes over time |
| RAG/semantic search | Markdown + MCP | Future: Claude app integration |

## LLM Integration

### Two-Tier Cascade Pattern

**Why cascading?**
- **90% of messages** are unambiguous → Qwen handles them (free, fast)
- **10% of messages** are ambiguous → Escalate to Gemini/Claude (paid, accurate)
- **Cost effective**: Only pay for cloud API when local model is uncertain

**Implementation:**
```
Message → Qwen 7b (Tier 1)
    ├─ confidence >= 0.7 → Store result
    └─ confidence < 0.7 → Gemini Flash (Tier 2)
        ├─ confidence >= 0.6 → Store result
        └─ confidence < 0.6 → Ask user in Discord
```

### LiteLLM Proxy Integration

**Why proxy instead of direct API calls?**
- Unified API for all models (Ollama, Gemini, Claude)
- Request logging and metrics in one place
- Rate limiting and retry logic handled by proxy
- Easy to swap models without code changes

**Configuration:**
```bash
LITELLM_BASE_URL=http://ares.evilsoft:4000
REFLEX_LLM_TIER1_MODEL=ollama/qwen2.5:7b
REFLEX_LLM_TIER2_MODEL=gemini/gemini-1.5-flash
```

**Client code:**
```python
import httpx

response = httpx.post(
    f"{litellm_base_url}/chat/completions",
    json={
        "model": "ollama/qwen2.5:7b",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }
)
```

### Prompt Design

**Structured JSON output with confidence:**
```json
{
  "category": "idea",
  "confidence": 0.85,
  "reasoning": "Mentions future possibility, not current work",
  "suggested_tags": ["automation", "future"]
}
```

**Why confidence scores?**
- Enable tier escalation decisions
- Provide transparency to user
- Track model performance over time
- Debug classification issues

## Intent Detection

### Problem: Command vs Capture Disambiguation

**Commands** reference existing entries:
- "move that idea about Foo into projects"
- "tag the last note with urgent"
- "archive that inbox item"

**Captures** are new thoughts:
- "Remember to move the desk tomorrow" (not a command!)
- "I need to tag the photos for the album" (not a command!)

**Challenge:** Both contain similar words ("move", "tag") but different intent.

### Three-Stage Detection

**Stage 1: Heuristic (Fast, Free)**
```python
def looks_like_command(message: str) -> bool:
    # Command patterns: action verb + entry reference
    patterns = [
        r'^(move|tag|archive|delete|show)\s+(that|this|the\s+last|recent)',
        r'^(what|show\s+me|find|search)',
    ]
    return any(re.match(p, message.lower()) for p in patterns)
```

**Stage 2: Qwen Validation (1-2 sec, Free)**
- Only if heuristic says "looks like command"
- LLM confirms: is this actually a command?
- Threshold: confidence >= 0.7

**Stage 3: Tier 2 Validation (500ms, Paid)**
- Only if Qwen uncertain (confidence < 0.7)
- Gemini/Claude makes final determination
- Threshold: confidence >= 0.6

**Why three stages?**
- **95% of messages** skip validation entirely (obvious captures)
- **4% of messages** validated by Qwen (free)
- **1% of messages** escalated to Tier 2 (paid)
- Cost: ~$0.10/1M tokens × 1% = virtually free

## Command Execution

### Target Resolution: Recency + Keywords

**Problem:** "move that idea about Foo into projects"
- What does "that idea" refer to?
- Multiple ideas might mention "Foo"

**Solution:**
```sql
-- Query last 10 messages from #reflex
SELECT id, title, original_message, category
FROM reflex_entries
WHERE discord_channel_id = '{channel_id}'
ORDER BY captured_at DESC
LIMIT 10;

-- Filter by category if mentioned ("that idea" → category='idea')
-- Filter by keywords ("about Foo" → title/body contains 'foo')
-- If single match → execute
-- If multiple matches → ask user to clarify
-- If no matches → "couldn't find that entry"
```

**Why recency window?**
- Humans naturally reference recent context
- "that idea" almost always means last few messages
- Bounded query (O(10)) is fast
- No need for complex NLP parsing

### Action Types

| Action | Updates | Markdown Export |
|--------|---------|-----------------|
| **move** | category column | Move to new folder (e.g., idea/ → project/) |
| **tag** | Append to tags[] | Update frontmatter |
| **archive** | status = 'archived' | Move to archive/ folder |
| **show** | Read-only | Reply in Discord with entry details |

## Markdown Export

### File Structure

```
~/cortex-reflex/  (separate git repo)
├── person/
│   └── 2026-01-12-alice-contact.md
├── project/
│   └── 2026-01-12-homelab-upgrade.md
├── idea/
│   └── 2026-01-12-auto-tagging.md
├── admin/
│   └── 2026-01-12-dentist-appt.md
├── inbox/
│   └── 2026-01-13-article-rag.md
└── archive/
    └── 2025-12-01-old-idea.md
```

### File Format

```markdown
---
id: 12345
discord_message_id: "1234567890123456789"
category: idea
tags: [automation, llm]
captured_at: 2026-01-12T14:30:00Z
llm_model: ollama/qwen2.5:7b
llm_confidence: 0.85
status: active
---

# Auto-tagging system for notes

Original thought: "What if we used embeddings to suggest tags?"

## LLM Classification
**Category**: idea
**Confidence**: 0.85
**Reasoning**: Future possibility, not tied to specific project.
```

### Export Strategy

**On every write (not batched):**
1. Store in Postgres + DuckDB
2. Generate markdown file
3. Git add + commit (async, non-blocking)
4. Git push to remote (best-effort)

**Why immediate export?**
- Immediate feedback to user (see commit in git log)
- Git handles many small commits efficiently
- Simpler code (no tracking of pending exports)
- Failures are visible immediately (can retry)

**Git operations are async:**
```python
def export_async(entry: Entry) -> None:
    """Export in background thread."""
    threading.Thread(target=_do_export, args=(entry,)).start()
    # Don't block Discord response
```

### Future: Bidirectional Sync

**Phase 1 (current):** Export-only (DB → markdown)
**Phase 2 (future):** Watch markdown files, sync changes back to DB
- File watcher (inotify) detects edits
- Parse frontmatter → update Postgres
- Enables manual editing workflow

## Observability

### Structured Logging

All logs use `cortex_utils.logging` (structlog):
```json
{
  "event": "Message classified",
  "service": "reflex",
  "logger": "reflex.services.classifier",
  "level": "info",
  "timestamp": "2026-01-13T00:00:00Z",
  "category": "idea",
  "confidence": 0.87,
  "model": "ollama/qwen2.5:7b",
  "discord_message_id": "123456"
}
```

### Prometheus Metrics

**Standard Cortex metrics:**
- `cortex_errors_total{service="reflex", error_type="..."}`
- `cortex_processing_duration_seconds{operation="..."}`

**Custom Reflex metrics:**
- `reflex_captures_total{category="person|project|idea|admin|inbox"}`
- `reflex_classification_duration_seconds{tier="tier1|tier2"}`
- `reflex_llm_confidence{model="qwen|gemini|claude"}`
- `reflex_exports_total{status="success|failed"}`
- `reflex_commands_total{action="move|tag|archive|show"}`

**Metrics server:** Port 8096 (standard for Cortex services)

### Discord Alerting

Critical errors sent to Discord webhook via `cortex_utils.alerter`:
```python
client.send_embed(
    title="Reflex Classification Error",
    description=f"Failed to classify message: {e}",
    color=COLOR_CRITICAL,
    fields=[
        {"name": "Message ID", "value": message.id},
        {"name": "Channel", "value": message.channel.name},
    ],
    ping=True  # @here mention for critical issues
)
```

## Deployment

### Container: cortex-reflex-sync

**Image:** `us-central1-docker.pkg.dev/cortex-gmail/cortex/reflex:latest`

**Runs on:** Hades (10.5.2.21)

**Depends on:**
- Postgres (cortex-postgres container)
- DuckDB API (cortex-duckdb-api container)
- LiteLLM Proxy (ares.evilsoft:4000)
- Discord Gateway (discord.com)

**Volumes:**
- `/app/data/cortex-reflex` - Git repository for markdown export
- SSH keys for git push

**Ports:**
- 8096 - Prometheus metrics

### Environment Variables

See `CLAUDE.md` for full list. Key variables:

```bash
# Discord
DISCORD_BOT_TOKEN=<from-homelab-env>
DISCORD_REFLEX_CHANNEL_ID=1460394971273363629

# Database
POSTGRES_HOST=10.5.2.21
POSTGRES_PASSWORD=cortex

# LiteLLM
LITELLM_BASE_URL=http://ares.evilsoft:4000
REFLEX_LLM_TIER1_MODEL=ollama/qwen2.5:7b
REFLEX_LLM_TIER2_MODEL=gemini/gemini-1.5-flash

# Git
REFLEX_GIT_REPO_PATH=/app/data/cortex-reflex
```

### Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
    CMD python -c "import httpx; httpx.get('http://localhost:8096/health')"
```

## Security Considerations

### Secrets Management

- **Discord bot token**: Stored in ~/HomeLab/.env (not in git)
- **Postgres password**: Retrieved from running container
- **LiteLLM API keys**: Managed by litellm proxy (Gemini/Claude)
- **Git SSH keys**: Mounted as volume (not in image)

### Discord Bot Permissions

**Required:**
- View Channels
- Send Messages
- Read Message History
- Add Reactions

**Privileged Intents:**
- Message Content Intent (required to read message text)

### Data Privacy

**All data is personal:**
- Single-user system (one Discord user ID)
- Private channel (#reflex)
- Data stored on personal infrastructure (Hades)
- No external sharing or analytics

## Scalability

### Current Scale

- **Users:** 1 (single-user system)
- **Messages:** ~10-50 per day (human typing speed)
- **Storage:** ~1MB per 1000 entries (Postgres metadata)
- **LLM calls:** ~10-50 per day (Tier 1), ~1-5 per day (Tier 2)

### Design Limits

**What would break first at 100x scale?**
1. **Git repository size** (10k markdown files = ~50MB, fine)
2. **Postgres queries** (100k entries = <100ms queries with indexes, fine)
3. **Discord rate limits** (50 requests/sec, human speed is 0.02/sec, fine)
4. **LiteLLM proxy** (100 requests/day = trivial load, fine)

**Conclusion:** Current architecture handles 100x growth without changes.

### Future Multi-User Support

If expanding to multiple users:
1. **Partition by user:** `WHERE discord_user_id = ?`
2. **Separate git repos:** `~/cortex-reflex-{user_id}/`
3. **Per-user configuration:** Model preferences, categories, etc.
4. **Shared infrastructure:** Postgres, DuckDB API, LiteLLM proxy

## Future Enhancements

### Phase 5: MCP Server

**Goal:** Query Reflex from Claude app (Desktop/Mobile)

**Architecture:**
```
Claude app
    ↓ MCP protocol
MCP server (Python)
    ↓ SQL queries
Postgres (metadata) + Markdown files (RAG)
    ↓ Results
Claude app
```

**Tools:**
- `search(query, category, tags, date_range)` - Structured search
- `recent(n)` - Last N captures
- `semantic_search(query)` - Vector similarity (requires embeddings)

### Phase 6: Email Integration

**Goal:** Self-sent emails → Reflex captures

**Architecture:**
```
Gmail → Triage worker
    ↓ (detects from=self AND to=self)
reflex-capture queue
    ↓
reflex-email-worker
    ↓ (uses shared classifier)
Reflex storage
```

**Shared code:** Move `ReflexClassifier` to `cortex-utils/reflex/`

### Beyond

- **Bidirectional sync:** Manual markdown edits → DB updates
- **Vector search:** Semantic similarity via embeddings
- **Attachment support:** Images, PDFs attached to Discord messages
- **Web UI:** Browse/edit entries in browser
- **Mobile integration:** iOS shortcuts → webhook → Reflex

## Design Decisions

### Why Discord Instead of CLI/Web UI?

**Pros:**
- Already on phone and desktop
- Zero-friction capture (< 5 seconds)
- Notifications work out of the box
- Rich formatting (embeds, reactions)

**Cons:**
- Requires Discord bot token
- Dependent on Discord uptime
- Less control over UI

**Decision:** Pros outweigh cons for personal use. Can add other interfaces later.

### Why Git for Export Instead of Database Only?

**Pros:**
- Version history for free
- Human-readable files
- Easy to grep/search
- Portable (not locked to Postgres)
- RAG-friendly for Claude app

**Cons:**
- More complex (DB + git operations)
- Git push can fail (network issues)

**Decision:** Git provides too much value to skip. Async export mitigates complexity.

### Why Two-Tier LLM Instead of Always Using Claude?

**Cost analysis (at 50 captures/day):**
- **Always Claude:** 50 × $0.25/1M tokens × 500 tokens = $0.00625/day = $2.28/year
- **Two-tier (90% Qwen):** 5 × $0.00625 = $0.00031/day = $0.11/year

**Quality:**
- Qwen 7b handles unambiguous cases well (tested)
- Claude only needed for edge cases

**Decision:** 20x cost savings justifies complexity of cascade.

### Why Single Service Instead of Microservices?

**Complexity cost:**
- Microservices: 3+ containers, queue management, service mesh
- Single service: 1 container, direct function calls

**Performance:**
- Human typing speed: ~1 msg/min
- No parallelization benefits at this scale

**Decision:** Microservices add complexity without benefits for personal use.

## References

- **Cortex Overview:** `/home/devon/Projects/cortex/docs/cortex-overview.md`
- **Project Spec:** `docs/project-spec.md`
- **LLM Configuration:** `/home/devon/Projects/cortex/docs/llm-configuration.md`
- **Beads Tickets:** `.beads/issues.jsonl` (cortex-19d2 epic)
