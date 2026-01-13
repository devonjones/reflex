# Reflex - Personal Second Brain

Personal knowledge capture system using Discord as interface, with two-tier LLM classification (Qwen → Gemini/Claude), dual storage (Postgres + DuckDB), and markdown export to git.

## Quick Reference

- **Spec**: `docs/project-spec.md`
- **Main service**: `src/reflex/services/bot.py`
- **Discord channel**: #reflex (ID: 1460394971273363629)
- **Markdown export**: `~/cortex-reflex/` (separate git repo)

## Project Structure

```
reflex/
├── src/reflex/
│   ├── services/
│   │   ├── bot.py          # Discord bot (main entry point)
│   │   ├── classifier.py   # LLM cascade (Qwen → Tier 2)
│   │   ├── digest.py       # Daily/weekly summaries
│   │   └── commands.py     # Natural language command parser
│   ├── storage/
│   │   ├── postgres.py     # Metadata storage
│   │   ├── duckdb_client.py # Full text storage
│   │   └── exporter.py     # Markdown export + git
│   └── models/
│       └── entry.py        # Data models
├── docs/
│   └── project-spec.md
├── tests/
├── Dockerfile
└── pyproject.toml
```

## Environment Variables

```bash
# Discord
DISCORD_BOT_TOKEN=<from-homelab-env>
DISCORD_REFLEX_CHANNEL_ID=1460394971273363629
DISCORD_USER_ID=509879976879718411
DISCORD_REFLEX_WEBHOOK=<for-error-notifications>

# Database
POSTGRES_HOST=10.5.2.21
POSTGRES_PORT=5432
POSTGRES_DB=cortex
POSTGRES_USER=cortex
POSTGRES_PASSWORD=<secret>
DUCKDB_PATH=/app/data/reflex.duckdb

# LiteLLM Proxy (Qwen + Gemini/Claude)
LITELLM_BASE_URL=http://ares.evilsoft:4000
REFLEX_LLM_TIER1_MODEL=ollama/qwen2.5:7b
REFLEX_LLM_TIER2_MODEL=gemini/gemini-1.5-flash
REFLEX_LLM_TIER1_CONFIDENCE_THRESHOLD=0.7
REFLEX_LLM_TIER2_CONFIDENCE_THRESHOLD=0.6

# Git export
REFLEX_GIT_REPO_PATH=/app/data/cortex-reflex
REFLEX_GIT_REMOTE=git@github.com:devonjones/cortex-reflex.git

# Metrics
METRICS_PORT=8096

# Digest schedule
REFLEX_DIGEST_DAILY_HOUR=7
REFLEX_DIGEST_WEEKLY_DAY=sunday
REFLEX_DIGEST_WEEKLY_HOUR=16
```

## Key Dependencies

- `discord.py` - Discord bot framework
- `httpx` - HTTP client for litellm
- `psycopg2` - Postgres client
- `duckdb` - DuckDB client
- `APScheduler` - Job scheduling for digests
- `GitPython` - Git operations for markdown export
- `cortex-utils` - Shared logging, metrics, alerter

## Development

```bash
# Setup
uv sync

# Run tests
uv run pytest

# Type check
uv run mypy src/reflex

# Format
uv run black src tests
```

## Git Workflow

**NEVER push directly to main.** All changes must be submitted via a pull request, following this workflow:

1. Create a feature branch
2. Make changes and commit
3. Push the branch and create a PR
4. Wait for CI and Gemini Code Assist review
5. Merge via GitHub after approval

## Deployment

- Docker image: `us-central1-docker.pkg.dev/cortex-gmail/cortex/reflex:latest`
- Runs on Hades alongside other Cortex services
- Portainer stack: `cortex-reflex`

## Coding Style

**Follow existing Cortex patterns:**
- Use `cortex_utils.logging` for structured logging
- Use `cortex_utils.metrics` for Prometheus metrics
- Use `cortex_utils.alerter.discord` for error notifications
- Service structure follows `postmark/services/gmail_sync.py` pattern
- LLM client follows `triage/services/worker.py` OllamaClient pattern

**No hardcoded values:**
- All configuration via environment variables
- No homelab IPs/hostnames in code
- LLM models/prompts configurable

## Integration with Email (Future)

When implementing email-to-self integration:
- Triage worker detects self-sent emails
- Routes to `reflex-capture` queue
- Shared classifier code moves to `cortex-utils/reflex/`
