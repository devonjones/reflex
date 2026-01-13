# Reflex - Personal Second Brain

Zero-friction knowledge capture system using Discord as the interface, with automatic LLM classification and markdown export.

## Features

- **Discord capture**: Drop thoughts in #reflex channel
- **Two-tier LLM classification**: Qwen 7b (local) → Gemini/Claude (cloud) cascade
- **Dual storage**: Postgres metadata + DuckDB full text
- **Markdown export**: Auto-commit to git repo organized by category
- **Natural language commands**: "move that idea about Foo into projects"
- **Digests**: Daily (7am) and weekly (Sunday 4pm) summaries via Discord DM

## Categories

- `person` - Information about people
- `project` - Work related to projects/goals
- `idea` - Random thoughts, inspirations
- `admin` - Tasks, errands, administrative matters
- `inbox` - External ideas needing review (staging area)

## Architecture

See `docs/project-spec.md` for detailed specification.

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
uv run ruff check src tests
```

## Deployment

See CLAUDE.md for deployment instructions.
