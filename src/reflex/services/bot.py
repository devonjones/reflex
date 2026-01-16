"""Discord bot for Reflex knowledge capture."""

import asyncio
import hmac
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import discord
import psycopg2
from aiohttp import web
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from cortex_utils.logging import configure_logging, get_logger
from cortex_utils.metrics import ERRORS, PROCESSING_DURATION, start_metrics_server
from discord.ext import commands
from prometheus_client import Counter, Histogram

from reflex.migrations import BOT_VERSION, migrate_entry
from reflex.models.entry import Entry
from reflex.services.classifier import ReflexClassifier
from reflex.services.commands import CommandParser, ParsedCommand
from reflex.storage.exporter import MarkdownExporter
from reflex.storage.postgres import PostgresStorage
from reflex.utils.date_parser import parse_snooze_date

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

    # Category constants
    VALID_CATEGORIES = ["person", "project", "idea", "admin", "inbox"]

    # Digest constants
    DEFAULT_DIGEST_HOUR = 7
    DIGEST_CATEGORY_EMOJIS = {
        "person": "👤",
        "project": "📋",
        "idea": "💡",
        "admin": "📌",
        "inbox": "📥",
    }
    DIGEST_CATEGORY_ORDER = ["project", "admin", "person", "idea", "inbox"]
    DIGEST_REACTION_EMOJIS = ["✅", "⏰", "📅", "🕐", "🔁"]
    DIGEST_INFO_TITLE_MAX_LENGTH = 80
    DAY_MAP = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
    DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    DEFAULT_WEEKLY_DIGEST_HOUR = 16
    WEEKLY_DIGEST_MAX_ENTRIES_PER_CATEGORY = 5

    def __init__(self) -> None:
        """Initialize bot."""
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read messages
        super().__init__(command_prefix="!", intents=intents)

        # Configuration from environment
        self.reflex_channel_id = os.getenv("DISCORD_REFLEX_CHANNEL_ID")
        if not self.reflex_channel_id:
            raise ValueError("DISCORD_REFLEX_CHANNEL_ID environment variable not set")

        self.webhook_token = os.getenv("REFLEX_WEBHOOK_TOKEN")
        if not self.webhook_token:
            raise ValueError("REFLEX_WEBHOOK_TOKEN environment variable not set")

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

        # Initialize command parser (use Tier 2 model for better accuracy)
        self.command_parser = CommandParser(litellm_url, tier2_model)

        # Initialize scheduler for digest
        self.scheduler = AsyncIOScheduler(timezone=timezone.utc)

        # Digest schedule configuration (default: 7am daily)
        digest_hour_str = os.getenv("REFLEX_DIGEST_DAILY_HOUR", str(self.DEFAULT_DIGEST_HOUR))
        try:
            digest_hour = int(digest_hour_str)
            if not 0 <= digest_hour <= 23:
                raise ValueError("Hour out of range")
            self.digest_hour = digest_hour
        except ValueError:
            logger.warning(f"Invalid REFLEX_DIGEST_DAILY_HOUR='{digest_hour_str}', using default {self.DEFAULT_DIGEST_HOUR}.")
            self.digest_hour = self.DEFAULT_DIGEST_HOUR

        # Weekly digest schedule (default: Sunday 4pm)
        weekly_day_str = os.getenv("REFLEX_DIGEST_WEEKLY_DAY", "sunday").lower()
        weekly_hour_str = os.getenv("REFLEX_DIGEST_WEEKLY_HOUR", str(self.DEFAULT_WEEKLY_DIGEST_HOUR))

        self.weekly_day = self.DAY_MAP.get(weekly_day_str, 6)  # Default to Sunday
        if weekly_day_str not in self.DAY_MAP:
            logger.warning(f"Invalid REFLEX_DIGEST_WEEKLY_DAY='{weekly_day_str}', using default 'sunday'")

        try:
            weekly_hour = int(weekly_hour_str)
            if not 0 <= weekly_hour <= 23:
                raise ValueError("Hour out of range")
            self.weekly_hour = weekly_hour
        except ValueError:
            logger.warning(f"Invalid REFLEX_DIGEST_WEEKLY_HOUR='{weekly_hour_str}', using default {self.DEFAULT_WEEKLY_DIGEST_HOUR}")
            self.weekly_hour = self.DEFAULT_WEEKLY_DIGEST_HOUR

        # State tracking for snooze prompts: (snooze_prompt_id, user_id) -> (entry_id, digest_message_id)
        self.snooze_pending: dict[tuple[int, int], tuple[int, int]] = {}

        # Track digest message_id -> entry_id for reaction handling
        self.digest_message_to_entry: dict[int, int] = {}

        # Track capture message_id -> entry_id for quick-complete via ✅
        self.capture_message_to_entry: dict[int, int] = {}

        logger.info("ReflexBot initialized")

    async def on_ready(self) -> None:
        """Called when bot is ready."""
        if self.user:
            logger.info(f"Bot ready: {self.user} (ID: {self.user.id}) - version {BOT_VERSION}")
        logger.info(f"Listening on channel ID: {self.reflex_channel_id}")

        # Start scheduler for digests
        if not self.scheduler.running:
            # Schedule daily digest at configured hour
            self.scheduler.add_job(
                self.generate_digest,
                CronTrigger(hour=self.digest_hour, minute=0),
                id="daily_digest",
                name=f"Daily Digest at {self.digest_hour}:00 UTC",
                replace_existing=True,
            )

            # Schedule weekly digest (default: Sunday 4pm)
            self.scheduler.add_job(
                self.generate_weekly_digest,
                CronTrigger(day_of_week=self.weekly_day, hour=self.weekly_hour, minute=0),
                id="weekly_digest",
                name=f"Weekly Digest on {self.DAY_NAMES[self.weekly_day]} at {self.weekly_hour}:00 UTC",
                replace_existing=True,
            )

            self.scheduler.start()
            logger.info(
                f"Scheduler started - daily digest at {self.digest_hour}:00 UTC, "
                f"weekly digest on {self.DAY_NAMES[self.weekly_day]} at {self.weekly_hour}:00 UTC"
            )

        # Spawn background migration task
        asyncio.create_task(self.migrate_old_entries())

    async def migrate_old_entries(self) -> None:
        """Background task to migrate old entries to current bot version."""
        logger.info(f"Starting migration check for bot version {BOT_VERSION}")

        try:
            with MIGRATION_DURATION.time():
                # Get entries needing migration (in thread to avoid blocking event loop)
                entries = await asyncio.to_thread(
                    self.storage.get_entries_needing_migration, BOT_VERSION
                )
                MIGRATIONS_PENDING.observe(len(entries))

                if not entries:
                    logger.info("No entries need migration")
                    return

                logger.info(f"Found {len(entries)} entries needing migration")

                # Migrate each entry
                for entry in entries:
                    try:
                        await asyncio.to_thread(
                            migrate_entry,
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

    async def close(self) -> None:
        """Cleanup on bot shutdown."""
        logger.info("Shutting down ReflexBot")

        # Stop scheduler
        if self.scheduler.running:
            logger.info("Shutting down scheduler...")
            try:
                await asyncio.to_thread(self.scheduler.shutdown)
                logger.info("Scheduler stopped")
            except Exception:
                logger.error("Error during scheduler shutdown, continuing...", exc_info=True)

        # Close components with helper method
        await self._close_component(self.storage, "storage layer")
        await self._close_component(self.command_parser, "command parser")
        await self._close_component(self.pg_conn, "postgres connection")

        # Close parent
        await super().close()

    async def _close_component(self, component: Optional[object], name: str) -> None:
        """Safely close a component, running its sync close method in a thread.

        Args:
            component: The component to close (must have a close() method), or None
            name: Human-readable name for logging
        """
        if not component:
            return
        logger.info(f"Closing {name}...")
        try:
            await asyncio.to_thread(component.close)
            logger.info(f"{name.capitalize()} closed")
        except Exception:
            logger.error(f"Error closing {name}, continuing...", exc_info=True)

    def _truncate_title(self, title: str) -> str:
        """Truncate title if it exceeds max length.

        Args:
            title: Original title

        Returns:
            Truncated title with "..." if needed
        """
        if len(title) > self.DIGEST_INFO_TITLE_MAX_LENGTH:
            return title[:self.DIGEST_INFO_TITLE_MAX_LENGTH - 3] + "..."
        return title

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
            # Check if this is a reply to a snooze prompt
            if message.reference and message.reference.message_id:
                ref_msg_id = message.reference.message_id
                user_id = message.author.id
                key = (ref_msg_id, user_id)

                if key in self.snooze_pending:
                    await self.handle_snooze_reply(message, key)
                    return

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
                # It's a command - parse and execute
                logger.info(
                    f"Confirmed command (confidence={confidence}): {message.content}"
                )
                await self.execute_command(message)
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

    async def execute_command(self, message: discord.Message) -> None:
        """Execute a validated command.

        Args:
            message: Discord message containing the command
        """
        try:
            # Parse command
            parsed = await asyncio.to_thread(
                self.command_parser.parse, message.content
            )

            if not parsed:
                await message.reply("❌ Sorry, I couldn't understand that command.")
                return

            logger.info(
                f"Parsed command: action={parsed.action}, target={parsed.target}, "
                f"keywords={parsed.target_keywords}"
            )

            # Resolve target entry
            target_entry = await self.resolve_target(message, parsed)
            if not target_entry:
                await message.reply(
                    "❌ I couldn't find the entry you're referring to. "
                    "Try being more specific or use keywords from the entry title."
                )
                return

            # Execute action
            if parsed.action == "move":
                await self.execute_move(message, target_entry, parsed)
            elif parsed.action == "tag":
                await self.execute_tag(message, target_entry, parsed)
            elif parsed.action == "archive":
                await self.execute_archive(message, target_entry, parsed)
            elif parsed.action == "show":
                await self.execute_show(message, target_entry, parsed)
            else:
                await message.reply(f"❌ Unknown action: {parsed.action}")

        except Exception as e:
            logger.error(f"Command execution failed: {e}", exc_info=True)
            ERRORS.labels(service="reflex", error_type="command_execution").inc()
            await message.reply(
                "❌ Something went wrong executing that command. Please try again."
            )

    async def resolve_target(
        self, message: discord.Message, parsed: ParsedCommand
    ) -> Optional[Entry]:
        """Resolve which entry the command refers to.

        Strategy:
        1. Query recent entries from this channel (last 10)
        2. Filter by keywords if provided
        3. Return most recent match

        Args:
            message: Discord message
            parsed: ParsedCommand

        Returns:
            Entry or None if not found
        """
        # Get recent entries from this channel
        entries = await asyncio.to_thread(
            self.storage.get_recent_entries,
            str(message.channel.id),
            limit=10,
        )

        if not entries:
            logger.info("No recent entries found in channel")
            return None

        # Filter by keywords if provided
        if parsed.target_keywords:
            filtered = []
            for entry in entries:
                title_lower = entry.title.lower()
                category_lower = entry.category.lower()
                if any(
                    kw.lower() in title_lower or kw.lower() in category_lower
                    for kw in parsed.target_keywords
                ):
                    filtered.append(entry)

            if not filtered:
                logger.info(f"Filtered to 0 entries with keywords: {parsed.target_keywords}")
                return None

            entries = filtered
            logger.info(f"Filtered to {len(filtered)} entries by keywords")

        # Return most recent
        target = entries[0]
        logger.info(f"Resolved target: entry_id={target.id}, title={target.title}")
        return target

    async def execute_move(
        self, message: discord.Message, entry: Entry, parsed: ParsedCommand
    ) -> None:
        """Execute move command (change category).

        Args:
            message: Discord message
            entry: Target entry
            parsed: ParsedCommand
        """
        new_category = parsed.parameters.get("new_category")
        if not new_category:
            await message.reply("❌ No target category specified in move command.")
            return

        # Validate category
        if new_category not in self.VALID_CATEGORIES:
            await message.reply(
                f"❌ Invalid category: {new_category}. "
                f"Valid categories: {', '.join(self.VALID_CATEGORIES)}"
            )
            return

        old_category = entry.category
        entry.category = new_category

        # Update in database
        await asyncio.to_thread(self.storage.update_entry, entry)

        # Re-export markdown (category affects file path)
        if self.exporter:
            await asyncio.to_thread(self.exporter.export_entry, entry)

        await message.reply(
            f"✅ Moved entry from **{old_category}** to **{new_category}**\n\n"
            f"_{entry.title}_"
        )
        logger.info(f"Moved entry {entry.id} from {old_category} to {new_category}")

    async def execute_tag(
        self, message: discord.Message, entry: Entry, parsed: ParsedCommand
    ) -> None:
        """Execute tag command (add tags).

        Args:
            message: Discord message
            entry: Target entry
            parsed: ParsedCommand
        """
        new_tags = parsed.parameters.get("tags", [])
        if not new_tags:
            await message.reply("❌ No tags specified in tag command.")
            return

        # Add new tags (avoid duplicates)
        existing_tags = set(entry.tags or [])
        existing_tags.update(new_tags)
        entry.tags = list(existing_tags)

        # Update in database
        await asyncio.to_thread(self.storage.update_entry, entry)

        # Re-export markdown
        if self.exporter:
            await asyncio.to_thread(self.exporter.export_entry, entry)

        await message.reply(
            f"✅ Added tags: **{', '.join(new_tags)}**\n\n_{entry.title}_"
        )
        logger.info(f"Tagged entry {entry.id} with {new_tags}")

    async def execute_archive(
        self, message: discord.Message, entry: Entry, parsed: ParsedCommand
    ) -> None:
        """Execute archive command (set status=archived).

        Args:
            message: Discord message
            entry: Target entry
            parsed: ParsedCommand
        """
        entry.status = "archived"

        # Update in database
        await asyncio.to_thread(self.storage.update_entry, entry)

        # Re-export markdown (status update reflected in frontmatter)
        if self.exporter:
            await asyncio.to_thread(self.exporter.export_entry, entry)

        await message.reply(f"✅ Archived entry\n\n_{entry.title}_")
        logger.info(f"Archived entry {entry.id}")

    async def execute_show(
        self, message: discord.Message, entry: Entry, parsed: ParsedCommand
    ) -> None:
        """Execute show command (display entry details).

        Args:
            message: Discord message
            entry: Target entry
            parsed: ParsedCommand
        """
        # Format entry details
        tags_str = ", ".join(entry.tags) if entry.tags else "none"
        captured_str = (
            entry.captured_at.strftime('%Y-%m-%d %H:%M')
            if entry.captured_at
            else "unknown"
        )
        details = (
            f"**{entry.title}**\n\n"
            f"Category: {entry.category}\n"
            f"Tags: {tags_str}\n"
            f"Status: {entry.status}\n"
            f"Captured: {captured_str}\n"
            f"ID: {entry.id}"
        )

        await message.reply(details)
        logger.info(f"Showed entry {entry.id}")

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
            actionable=result.actionable,
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

        # Track original message for quick-complete (user clicks ✅ on their own message)
        self.capture_message_to_entry[message.id] = entry_id
        logger.debug(f"Tracking capture message {message.id} -> entry {entry_id}")

    async def on_error(self, event_method: str, *args: object, **kwargs: object) -> None:
        """Handle errors in event handlers.

        Args:
            event_method: Name of event method that errored
        """
        logger.error(f"Error in {event_method}", exc_info=True)
        ERRORS.labels(service="reflex", error_type="discord_event").inc()

    async def on_reaction_add(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """Handle emoji reactions on digest messages and captured user messages.

        Args:
            reaction: Discord reaction
            user: User who added the reaction
        """
        # Ignore bot's own reactions
        if user == self.user:
            return

        # Only handle reactions in reflex channel
        if str(reaction.message.channel.id) != self.reflex_channel_id:
            return

        message_id = reaction.message.id
        emoji = str(reaction.emoji)

        # Check if this is a digest entry message (bot's messages)
        entry_id = self.digest_message_to_entry.get(message_id)
        if entry_id and reaction.message.author == self.user:
            logger.info(
                f"Reaction {emoji} from {user} on digest entry {entry_id} (message {message_id})"
            )

            try:
                # Dispatcher pattern for emoji reactions on digest messages
                handler_map = {
                    "✅": lambda r, u, e: self.handle_archive_entry(r, u, e, source="digest"),
                    "⏰": lambda r, u, e: self.handle_snooze_entry(r, u, e, days=7, label="1 week"),
                    "📅": lambda r, u, e: self.handle_snooze_entry(r, u, e, days=30, label="1 month"),
                    "🕐": lambda r, u, e: self.handle_snooze_entry_custom(r, u, e),
                }

                handler = handler_map.get(emoji)
                if handler:
                    await handler(reaction, user, entry_id)
                else:
                    logger.debug(f"Ignoring unknown emoji: {emoji}")

            except Exception as e:
                logger.error(f"Error handling reaction {emoji}: {e}", exc_info=True)
                ERRORS.labels(service="reflex", error_type="reaction_handler").inc()
            return

        # Check if this is a captured user message (user clicks ✅ on their own message)
        entry_id = self.capture_message_to_entry.get(message_id)
        if entry_id:
            # Only handle ✅ for quick-complete on capture messages
            if emoji != "✅":
                logger.debug(f"Ignoring {emoji} on capture message (only ✅ supported)")
                return

            # Only allow the original message author to quick-complete
            if user.id != reaction.message.author.id:
                logger.debug(
                    f"User {user} tried to complete entry {entry_id} but is not original author"
                )
                return

            logger.info(
                f"Quick-complete: {user} archiving entry {entry_id} (message {message_id})"
            )

            try:
                # Archive the entry using shared handler
                await self.handle_archive_entry(reaction, user, entry_id, source="capture")
            except Exception as e:
                logger.error(f"Error handling quick-complete: {e}", exc_info=True)
                ERRORS.labels(service="reflex", error_type="reaction_handler").inc()
            return

        # Not a tracked message
        logger.debug(f"Ignoring reaction on non-tracked message {message_id}")

    async def handle_archive_entry(
        self, reaction: discord.Reaction, user: discord.User, entry_id: int, source: str = "digest"
    ) -> None:
        """Archive a specific entry from a digest or capture confirmation.

        Args:
            reaction: Discord reaction
            user: User who reacted
            entry_id: Entry ID to archive
            source: Source of the archive request ("digest" or "capture")
        """
        # Update entry to archived
        with self.pg_conn.cursor() as cur:
            cur.execute(
                """
                UPDATE reflex_entries
                SET status = 'archived'
                WHERE id = %s
                """,
                (entry_id,),
            )
        self.pg_conn.commit()

        # Remove from tracking (using pop() to avoid KeyError if user reacts twice quickly)
        message_id = reaction.message.id
        if source == "capture":
            self.capture_message_to_entry.pop(message_id, None)
            logger.info(f"Archived entry {entry_id} via quick-complete reaction")
        else:  # "digest"
            self.digest_message_to_entry.pop(message_id, None)
            logger.info(f"Archived entry {entry_id} via digest reaction")

        await reaction.message.reply(f"{user.mention} - ✅ Archived!")

    async def handle_snooze_entry(
        self, reaction: discord.Reaction, user: discord.User, entry_id: int, days: int, label: str
    ) -> None:
        """Snooze a specific entry for a fixed duration.

        Args:
            reaction: Discord reaction
            user: User who reacted
            entry_id: Entry ID to snooze
            days: Number of days to snooze
            label: Human-readable label (e.g., "1 week", "1 month")
        """
        # Set next_action_date
        snooze_until = datetime.now(timezone.utc) + timedelta(days=days)

        with self.pg_conn.cursor() as cur:
            cur.execute(
                """
                UPDATE reflex_entries
                SET next_action_date = %s
                WHERE id = %s
                """,
                (snooze_until, entry_id),
            )
        self.pg_conn.commit()

        # Remove from tracking (use pop for safety)
        self.digest_message_to_entry.pop(reaction.message.id, None)

        logger.info(f"Snoozed entry {entry_id} for {label} until {snooze_until.date()}")
        await reaction.message.reply(
            f"{user.mention} - Snoozed for {label} "
            f"(remind on **{snooze_until.strftime('%B %d, %Y')}**)"
        )

    async def handle_snooze_entry_custom(
        self, reaction: discord.Reaction, user: discord.User, entry_id: int
    ) -> None:
        """Handle custom snooze request for a specific entry - prompt user for date.

        Args:
            reaction: Discord reaction
            user: User who reacted
            entry_id: Entry ID to snooze
        """
        # Send snooze prompt
        snooze_msg = await reaction.message.reply(
            f"{user.mention} - 🕐 When should I remind you? Reply to this message with:\n"
            "- `tomorrow`\n"
            "- `3d` (3 days)\n"
            "- `2w` (2 weeks)\n"
            "- `next week`\n"
            "- `next monday`\n"
            "- `jan 20` or `2026-01-20`"
        )

        # Store pending snooze state (entry_id + digest message ID for cleanup)
        key = (snooze_msg.id, user.id)
        self.snooze_pending[key] = (entry_id, reaction.message.id)
        logger.info(f"Waiting for snooze date from {user} for entry {entry_id}")

    async def handle_snooze_reply(
        self, message: discord.Message, key: tuple[int, int]
    ) -> None:
        """Handle user's reply with snooze date.

        Args:
            message: User's reply message
            key: (snooze_prompt_message_id, user_id) tuple
        """
        entry_id, digest_message_id = self.snooze_pending[key]
        date_str = message.content.strip()

        # Parse the date
        snooze_until = parse_snooze_date(date_str)

        if snooze_until is None:
            await message.reply(
                f"Sorry, I couldn't parse '{date_str}'. Please try again with a format like:\n"
                "- `tomorrow`\n"
                "- `3d` (3 days)\n"
                "- `1w` (1 week)\n"
                "- `next monday`\n"
                "- `jan 20`"
            )
            return

        # Update entry
        with self.pg_conn.cursor() as cur:
            cur.execute(
                """
                UPDATE reflex_entries
                SET next_action_date = %s
                WHERE id = %s
                """,
                (snooze_until, entry_id),
            )
        self.pg_conn.commit()

        # Remove entry from digest tracking if it still exists (use pop for safety)
        self.digest_message_to_entry.pop(digest_message_id, None)

        # Clear pending state (use pop for safety)
        self.snooze_pending.pop(key, None)

        logger.info(f"Snoozed entry {entry_id} until {snooze_until.date()} for user {message.author}")
        await message.reply(
            f"✅ Got it! I'll remind you on **{snooze_until.strftime('%B %d, %Y')}**"
        )

    async def generate_digest(self) -> None:
        """Generate and send daily digest to Discord channel.

        Sends:
        1. ONE MESSAGE PER ACTIONABLE ENTRY (admin only) with reaction buttons
        2. A single summary message with all info entries (person, idea, inbox, project)
        """
        logger.info("Generating daily digest")

        try:
            # Get channel
            if not self.reflex_channel_id:
                logger.error("DISCORD_REFLEX_CHANNEL_ID not configured")
                return

            channel = self.get_channel(int(self.reflex_channel_id))
            if not channel:
                logger.error(f"Could not find channel {self.reflex_channel_id}")
                return

            # Type narrow to ensure it's a text-based channel
            if not isinstance(channel, (discord.TextChannel, discord.Thread)):
                logger.error(f"Channel {self.reflex_channel_id} is not a text channel")
                return

            # Clear previous message tracking (digest and capture) to prevent memory leaks
            # and acting on stale messages
            self.digest_message_to_entry.clear()
            self.capture_message_to_entry.clear()

            # Query actionable entries (admin only) and info entries (everything else)
            action_rows = await asyncio.to_thread(self.storage.get_digest_entries)
            info_rows = await asyncio.to_thread(self.storage.get_digest_info_entries)

            if not action_rows and not info_rows:
                logger.info("No entries to show in digest")
                await channel.send("## Daily Digest\n\nNo entries today! 🎉")
                return

            # Send header message
            await channel.send(
                f"## Daily Digest - {len(action_rows)} action items, {len(info_rows)} FYI"
            )

            # Send ONE MESSAGE PER ACTIONABLE ENTRY with reactions
            if action_rows:
                await channel.send("**Action Items:**")
                now = datetime.now(timezone.utc)
                for row in action_rows:
                    entry_id, category, title, tags, captured_at, _ = row
                    emoji = self.DIGEST_CATEGORY_EMOJIS.get(category, "📝")

                    # Calculate age
                    age_days = (now - captured_at).days

                    # Build message
                    message_parts = [f"{emoji} **{title}**\n"]
                    if tags:
                        message_parts.append(f"*Tags: {', '.join(tags)}*\n")
                    message_parts.append(f"*Captured: {age_days} days ago*")

                    # Warn if old
                    if age_days > 7:
                        message_parts.append(f"\n📌 Open for {age_days} days. Still relevant?")

                    # Send entry message
                    entry_message = "".join(message_parts)
                    sent_message = await channel.send(entry_message)

                    # Add reaction options to THIS entry
                    for emoji_reaction in ["✅", "⏰", "📅", "🕐"]:
                        await sent_message.add_reaction(emoji_reaction)

                    # Track message_id -> entry_id mapping
                    self.digest_message_to_entry[sent_message.id] = entry_id

                    logger.info(f"Sent digest entry {entry_id} (message ID: {sent_message.id})")

            # Send info entries as a single summary (no reactions)
            if info_rows:
                await channel.send("**FYI - Reference Items:**")

                # Group by category
                by_category: dict[str, list[str]] = defaultdict(list)
                for info_row in info_rows:
                    _, category, title, _, _ = info_row
                    by_category[category].append(title)

                # Build summary message
                summary_parts = []
                for category in self.DIGEST_CATEGORY_ORDER:
                    if category not in by_category:
                        continue

                    emoji = self.DIGEST_CATEGORY_EMOJIS.get(category, "📝")
                    summary_parts.append(f"\n{emoji} **{category.title()}**")
                    for title in by_category[category]:
                        display_title = self._truncate_title(title)
                        summary_parts.append(f"\n  • {display_title}")

                await channel.send("".join(summary_parts))

            logger.info(
                f"Sent digest with {len(action_rows)} action items and {len(info_rows)} info items"
            )

        except Exception as e:
            logger.error(f"Failed to generate digest: {e}", exc_info=True)
            ERRORS.labels(service="reflex", error_type="digest_generation").inc()

    async def generate_weekly_digest(self) -> None:
        """Generate and send weekly digest to Discord channel.

        Aggregates past 7 days of entries, grouped by category with counts.
        Shows overall activity summary without individual message reactions.
        """
        logger.info("Generating weekly digest")

        try:
            # Get channel
            if not self.reflex_channel_id:
                logger.error("DISCORD_REFLEX_CHANNEL_ID not configured")
                return

            channel = self.get_channel(int(self.reflex_channel_id))
            if not channel or not isinstance(channel, (discord.TextChannel, discord.Thread)):
                logger.error(f"Reflex channel {self.reflex_channel_id} not found or not a text channel")
                return

            # Query weekly summary
            by_category = await asyncio.to_thread(self.storage.get_weekly_summary)

            # Calculate total entries
            total_entries = sum(len(entries) for entries in by_category.values())

            # Send header
            await channel.send(
                f"## Weekly Digest - {total_entries} entries captured this week"
            )

            # Send summary by category
            if total_entries == 0:
                await channel.send("_No entries captured this week_")
                return

            summary_parts = []
            for category in self.DIGEST_CATEGORY_ORDER:
                if category not in by_category:
                    continue

                entries = by_category[category]
                emoji = self.DIGEST_CATEGORY_EMOJIS.get(category, "📝")
                summary_parts.append(f"\n{emoji} **{category.title()}** ({len(entries)} entries)")

                # Show up to max entries
                for _, title, _, _ in entries[:self.WEEKLY_DIGEST_MAX_ENTRIES_PER_CATEGORY]:
                    display_title = self._truncate_title(title)
                    summary_parts.append(f"\n  • {display_title}")

                if len(entries) > self.WEEKLY_DIGEST_MAX_ENTRIES_PER_CATEGORY:
                    summary_parts.append(f"\n  _...and {len(entries) - self.WEEKLY_DIGEST_MAX_ENTRIES_PER_CATEGORY} more_")

            await channel.send("".join(summary_parts))

            logger.info(f"Sent weekly digest with {total_entries} entries")

        except Exception as e:
            logger.error(f"Failed to generate weekly digest: {e}", exc_info=True)
            ERRORS.labels(service="reflex", error_type="weekly_digest_generation").inc()


async def webhook_digest_handler(request: web.Request) -> web.Response:
    """Handle /webhook/digest POST requests.

    Args:
        request: HTTP request

    Returns:
        HTTP response
    """
    bot: ReflexBot = request.app["bot"]

    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        logger.warning("Webhook request missing or invalid Authorization header")
        return web.Response(status=401, text="Unauthorized")

    token = auth_header.replace("Bearer ", "")
    if not bot.webhook_token or not hmac.compare_digest(token, bot.webhook_token):
        logger.warning("Webhook request with invalid token")
        return web.Response(status=401, text="Unauthorized")

    # Trigger digest generation
    logger.info("Webhook triggered digest generation")
    asyncio.create_task(bot.generate_digest())

    return web.Response(status=200, text="Digest generation triggered")


async def start_webhook_server(bot: ReflexBot, port: int) -> web.AppRunner:
    """Start HTTP webhook server.

    Args:
        bot: Bot instance
        port: Port to listen on

    Returns:
        AppRunner instance
    """
    app = web.Application()
    app["bot"] = bot

    app.router.add_post("/webhook/digest", webhook_digest_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    logger.info(f"Webhook server started on port {port}")
    return runner


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

    # Get webhook port
    webhook_port = int(os.getenv("REFLEX_WEBHOOK_PORT", "8097"))

    # Create bot
    bot = ReflexBot()

    # Start webhook server in bot's event loop
    async def run_bot() -> None:
        """Run bot with webhook server."""
        async with bot:
            # Start webhook server
            await start_webhook_server(bot, webhook_port)
            # Start bot
            await bot.start(bot_token)

    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
