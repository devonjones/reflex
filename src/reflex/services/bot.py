"""Discord bot for Reflex knowledge capture."""

import asyncio
import os
import re
from datetime import datetime, timezone
from typing import Optional

import discord
import psycopg2
from cortex_utils.logging import configure_logging, get_logger
from cortex_utils.metrics import ERRORS, PROCESSING_DURATION, start_metrics_server
from discord.ext import commands
from prometheus_client import Counter, Histogram

from reflex.migrations import BOT_VERSION, migrate_entry
from reflex.models.entry import Entry
from reflex.services.classifier import ReflexClassifier
from reflex.storage.exporter import MarkdownExporter
from reflex.storage.postgres import PostgresStorage

# Configure logging
configure_logging("reflex", level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

# Metrics
CAPTURES_TOTAL = Counter(
    "reflex_captures_total",
    "Total number of captures",
    ["category"],
)

CLASSIFICATION_DURATION = Histogram(
    "reflex_classification_duration_seconds",
    "Time spent classifying",
    ["tier"],
)

LLM_CONFIDENCE = Histogram(
    "reflex_llm_confidence",
    "LLM confidence scores",
    ["model"],
    buckets=[0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

EXPORTS_TOTAL = Counter(
    "reflex_exports_total",
    "Total markdown exports",
    ["status"],
)

MIGRATIONS_PENDING = Histogram(
    "reflex_migrations_pending",
    "Number of entries pending migration at startup",
    buckets=[0, 1, 5, 10, 20, 50, 100],
)

MIGRATIONS_COMPLETED = Counter(
    "reflex_migrations_completed_total",
    "Total number of successful migrations",
)

MIGRATIONS_FAILED = Counter(
    "reflex_migrations_failed_total",
    "Total number of failed migrations",
)

MIGRATION_DURATION = Histogram(
    "reflex_migration_duration_seconds",
    "Time spent migrating entries",
)


def looks_like_command(message: str) -> bool:
    """Heuristic check if message looks like a command.

    Args:
        message: Message text

    Returns:
        True if message looks like a command
    """
    patterns = [
        r"^(move|tag|archive|delete|show|update)\s+(that|this|the\s+last|recent)",
        r"^(what|show\s+me|find|search)",
    ]
    return any(re.match(p, message.lower()) for p in patterns)


def generate_title(message: str, max_length: int = 80) -> str:
    """Generate a title from message.

    Args:
        message: Original message
        max_length: Maximum title length

    Returns:
        Title string
    """
    # Use first line or sentence
    first_line = message.split("\n")[0]
    if len(first_line) <= max_length:
        return first_line

    # Truncate to max length
    return first_line[:max_length - 3] + "..."


class ReflexBot(commands.Bot):
    """Discord bot for Reflex."""

    def __init__(self) -> None:
        """Initialize bot."""
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read messages
        super().__init__(command_prefix="!", intents=intents)

        # Configuration from environment
        self.reflex_channel_id = os.getenv("DISCORD_REFLEX_CHANNEL_ID")
        if not self.reflex_channel_id:
            raise ValueError("DISCORD_REFLEX_CHANNEL_ID environment variable not set")

        # Initialize Postgres connection
        postgres_host = os.getenv("POSTGRES_HOST")
        if not postgres_host:
            raise ValueError("POSTGRES_HOST environment variable not set")

        postgres_db = os.getenv("POSTGRES_DB")
        if not postgres_db:
            raise ValueError("POSTGRES_DB environment variable not set")

        postgres_user = os.getenv("POSTGRES_USER")
        if not postgres_user:
            raise ValueError("POSTGRES_USER environment variable not set")

        postgres_password = os.getenv("POSTGRES_PASSWORD")
        if not postgres_password:
            raise ValueError("POSTGRES_PASSWORD environment variable not set")

        try:
            postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
        except ValueError:
            logger.warning(f"Invalid POSTGRES_PORT, using default 5432")
            postgres_port = 5432

        self.pg_conn = psycopg2.connect(
            host=postgres_host,
            port=postgres_port,
            dbname=postgres_db,
            user=postgres_user,
            password=postgres_password,
        )

        # Initialize storage
        duckdb_api_url = os.getenv("DUCKDB_API_URL")
        if not duckdb_api_url:
            raise ValueError("DUCKDB_API_URL environment variable not set")
        self.storage = PostgresStorage(self.pg_conn, duckdb_api_url)

        # Initialize markdown exporter
        git_repo_path = os.getenv("REFLEX_GIT_REPO_PATH")
        git_remote = os.getenv("REFLEX_GIT_REMOTE")
        self.exporter: Optional[MarkdownExporter]
        if git_repo_path:
            self.exporter = MarkdownExporter(git_repo_path, git_remote)
            logger.info(f"Markdown exporter initialized: {git_repo_path}")
        else:
            self.exporter = None
            logger.warning(
                "REFLEX_GIT_REPO_PATH not set, markdown export disabled"
            )

        # Initialize classifier
        litellm_url = os.getenv("LITELLM_BASE_URL")
        if not litellm_url:
            raise ValueError("LITELLM_BASE_URL environment variable not set")

        tier1_model = os.getenv("REFLEX_LLM_TIER1_MODEL")
        if not tier1_model:
            raise ValueError("REFLEX_LLM_TIER1_MODEL environment variable not set")

        tier2_model = os.getenv("REFLEX_LLM_TIER2_MODEL")
        if not tier2_model:
            raise ValueError("REFLEX_LLM_TIER2_MODEL environment variable not set")

        # Parse thresholds with error handling
        try:
            tier1_threshold = float(
                os.getenv("REFLEX_LLM_TIER1_CONFIDENCE_THRESHOLD", "0.7")
            )
        except ValueError:
            logger.warning(
                f"Invalid REFLEX_LLM_TIER1_CONFIDENCE_THRESHOLD, using default 0.7"
            )
            tier1_threshold = 0.7

        try:
            tier2_threshold = float(
                os.getenv("REFLEX_LLM_TIER2_CONFIDENCE_THRESHOLD", "0.6")
            )
        except ValueError:
            logger.warning(
                f"Invalid REFLEX_LLM_TIER2_CONFIDENCE_THRESHOLD, using default 0.6"
            )
            tier2_threshold = 0.6

        self.classifier = ReflexClassifier(
            litellm_url, tier1_model, tier2_model, tier1_threshold, tier2_threshold
        )

        logger.info("ReflexBot initialized")

    async def on_ready(self) -> None:
        """Called when bot is ready."""
        if self.user:
            logger.info(f"Bot ready: {self.user} (ID: {self.user.id}) - version {BOT_VERSION}")
        logger.info(f"Listening on channel ID: {self.reflex_channel_id}")

        # Spawn background migration task
        asyncio.create_task(self.migrate_old_entries())

    async def migrate_old_entries(self) -> None:
        """Background task to migrate old entries to current bot version."""
        logger.info(f"Starting migration check for bot version {BOT_VERSION}")

        try:
            with MIGRATION_DURATION.time():
                # Get entries needing migration
                entries = self.storage.get_entries_needing_migration(BOT_VERSION)
                MIGRATIONS_PENDING.observe(len(entries))

                if not entries:
                    logger.info("No entries need migration")
                    return

                logger.info(f"Found {len(entries)} entries needing migration")

                # Migrate each entry
                for entry in entries:
                    try:
                        migrate_entry(
                            entry,
                            BOT_VERSION,
                            self.storage,
                            self.exporter,
                        )
                        MIGRATIONS_COMPLETED.inc()
                    except Exception as e:
                        logger.error(f"Migration failed for entry {entry.id}: {e}", exc_info=True)
                        MIGRATIONS_FAILED.inc()
                        # Continue with next entry

                logger.info(f"Migration complete: {len(entries)} entries processed")

        except Exception as e:
            logger.error(f"Migration task failed: {e}", exc_info=True)
            ERRORS.labels(service="reflex", error_type="migration").inc()

    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming messages.

        Args:
            message: Discord message
        """
        # Ignore bot's own messages
        if message.author == self.user:
            return

        # Only listen to reflex channel
        if str(message.channel.id) != self.reflex_channel_id:
            return

        logger.info(
            f"Received message: {message.id} from {message.author} in {message.channel}"
        )

        try:
            # Check if it looks like a command
            if looks_like_command(message.content):
                await self.handle_potential_command(message)
            else:
                await self.handle_capture(message)

        except Exception as e:
            logger.error(f"Error handling message {message.id}: {e}", exc_info=True)
            ERRORS.labels(service="reflex", error_type="message_handler").inc()
            await message.add_reaction("❌")
            await message.reply(
                "Sorry, something went wrong. Please try again or contact support."
            )

    async def handle_potential_command(self, message: discord.Message) -> None:
        """Handle potential command (validate intent first).

        Args:
            message: Discord message
        """
        logger.info(f"Message looks like command, validating intent: {message.id}")

        try:
            # Validate intent with two-tier cascade
            is_command, confidence = self.classifier.validate_intent(message.content)

            if is_command and confidence >= self.classifier.tier2_threshold:
                # It's a command
                logger.info(
                    f"Confirmed command (confidence={confidence}): {message.content}"
                )
                await message.reply(
                    f"Command detected (confidence: {confidence:.2f})\n\n"
                    "⚠️ Command execution not yet implemented. Coming in Phase 3!"
                )
            else:
                # False positive - treat as capture
                logger.info(
                    f"Not a command (confidence={confidence}), treating as capture"
                )
                await self.handle_capture(message)

        except Exception as e:
            logger.error(f"Intent validation failed: {e}", exc_info=True)
            ERRORS.labels(service="reflex", error_type="intent_validation").inc()
            # Fall back to treating as capture
            await self.handle_capture(message)

    async def handle_capture(self, message: discord.Message) -> None:
        """Handle message capture (classify and store).

        Args:
            message: Discord message
        """
        logger.info(f"Handling capture: {message.id}")

        # Classify with two-tier cascade
        with PROCESSING_DURATION.labels(queue="reflex", operation="classify").time():
            result = self.classifier.classify(message.content)

        LLM_CONFIDENCE.labels(model=result.model).observe(result.confidence)

        logger.info(
            f"Classified as {result.category} (confidence={result.confidence}, model={result.model})"
        )

        # Check if confidence is too low
        if result.confidence < self.classifier.tier2_threshold:
            logger.warning(
                f"Low confidence ({result.confidence}) even after tier 2, asking user"
            )
            await message.reply(
                f"I'm not sure how to classify this (confidence: {result.confidence:.2f}).\n\n"
                f"Please add a prefix to help me:\n"
                f"- `person:` for information about people\n"
                f"- `project:` for work on projects\n"
                f"- `idea:` for thoughts and inspirations\n"
                f"- `admin:` for tasks and errands\n"
                f"- `inbox:` for external content to review"
            )
            return

        # Generate title
        title = generate_title(message.content)

        # Capture timestamp
        now = datetime.now(timezone.utc)

        # Create entry
        entry = Entry(
            id=None,
            discord_message_id=str(message.id),
            discord_channel_id=str(message.channel.id),
            discord_user_id=str(message.author.id),
            category=result.category,
            title=title,
            tags=result.suggested_tags,
            llm_confidence=result.confidence,
            llm_model=result.model,
            llm_reasoning=result.reasoning,
            status="active",
            captured_at=now,
            updated_at=now,
            exported_to_git=False,
            git_commit_sha=None,
            markdown_path=None,
            original_message=message.content,
            bot_version=BOT_VERSION,
        )

        # Store in database
        with PROCESSING_DURATION.labels(queue="reflex", operation="store").time():
            entry_id = self.storage.store_entry(entry)

        CAPTURES_TOTAL.labels(category=result.category).inc()

        logger.info(f"Stored entry {entry_id}")

        # Update entry with ID
        entry.id = entry_id

        # Export to markdown (async, non-blocking)
        if self.exporter:
            with PROCESSING_DURATION.labels(queue="reflex", operation="export").time():
                self.exporter.export_and_commit_async(entry)
            logger.info(f"Triggered async markdown export for entry {entry_id}")

        # Send confirmation
        await message.add_reaction("✅")
        await message.reply(
            f"**Filed as {result.category}** - {title}\n"
            f"*Confidence: {result.confidence:.2f}*\n"
            f"*Tags: {', '.join(result.suggested_tags) if result.suggested_tags else 'none'}*\n"
            f"*Reasoning: {result.reasoning}*"
        )

    async def on_error(self, event_method: str, *args: object, **kwargs: object) -> None:
        """Handle errors in event handlers.

        Args:
            event_method: Name of event method that errored
        """
        logger.error(f"Error in {event_method}", exc_info=True)
        ERRORS.labels(service="reflex", error_type="discord_event").inc()


def main() -> None:
    """Main entry point."""
    # Start metrics server
    metrics_port = int(os.getenv("METRICS_PORT", "8096"))
    start_metrics_server(port=metrics_port)
    logger.info(f"Metrics server started on port {metrics_port}")

    # Get bot token
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    if not bot_token:
        raise ValueError("DISCORD_BOT_TOKEN environment variable not set")

    # Create and run bot
    bot = ReflexBot()
    bot.run(bot_token)


if __name__ == "__main__":
    main()
