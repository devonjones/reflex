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
from cortex_utils.logging import configure_logging, get_logger
from cortex_utils.metrics import ERRORS, PROCESSING_DURATION, start_metrics_server
from discord.ext import commands
from prometheus_client import Counter, Histogram

from reflex.migrations import BOT_VERSION, migrate_entry
from reflex.models.entry import Entry
from reflex.services.classifier import ReflexClassifier
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

    # Digest constants
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

        # State tracking for snooze prompts: (snooze_prompt_id, user_id) -> (entry_id, digest_message_id)
        self.snooze_pending: dict[tuple[int, int], tuple[int, int]] = {}

        # Track digest message_id -> entry_id for reaction handling
        self.digest_message_to_entry: dict[int, int] = {}

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
        """Handle emoji reactions on digest entry messages.

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

        # Only handle reactions on bot's messages (digest messages)
        if reaction.message.author != self.user:
            return

        # Check if this is a digest entry message
        message_id = reaction.message.id
        entry_id = self.digest_message_to_entry.get(message_id)
        if not entry_id:
            logger.debug(f"Ignoring reaction on non-digest message {message_id}")
            return

        emoji = str(reaction.emoji)
        logger.info(
            f"Reaction {emoji} from {user} on digest entry {entry_id} (message {message_id})"
        )

        try:
            # Dispatcher pattern for emoji reactions
            handler_map = {
                "✅": lambda r, u, e: self.handle_archive_entry(r, u, e),
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

    async def handle_archive_entry(
        self, reaction: discord.Reaction, user: discord.User, entry_id: int
    ) -> None:
        """Archive a specific entry.

        Args:
            reaction: Discord reaction
            user: User who reacted
            entry_id: Entry ID to archive
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

        # Remove from tracking
        del self.digest_message_to_entry[reaction.message.id]

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

        # Remove from tracking
        del self.digest_message_to_entry[reaction.message.id]

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

        # Remove entry from digest tracking if it still exists
        if digest_message_id in self.digest_message_to_entry:
            del self.digest_message_to_entry[digest_message_id]

        # Clear pending state
        del self.snooze_pending[key]

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

            # Clear previous digest's message tracking to prevent memory leaks
            # and acting on stale messages
            self.digest_message_to_entry.clear()

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
                        # Truncate title if too long
                        if len(title) > self.DIGEST_INFO_TITLE_MAX_LENGTH:
                            display_title = title[:self.DIGEST_INFO_TITLE_MAX_LENGTH - 3] + "..."
                        else:
                            display_title = title
                        summary_parts.append(f"\n  • {display_title}")

                await channel.send("".join(summary_parts))

            logger.info(
                f"Sent digest with {len(action_rows)} action items and {len(info_rows)} info items"
            )

        except Exception as e:
            logger.error(f"Failed to generate digest: {e}", exc_info=True)
            ERRORS.labels(service="reflex", error_type="digest_generation").inc()


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
