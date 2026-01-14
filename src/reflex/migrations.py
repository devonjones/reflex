"""Entry migration system for Reflex.

When the bot version changes, entries may need to be updated to match the new schema
or have fields reprocessed with improved logic. This module handles automatic migrations.
"""

from typing import TYPE_CHECKING, Optional

from cortex_utils.logging import get_logger

from reflex.models.entry import Entry

if TYPE_CHECKING:
    from reflex.storage.exporter import MarkdownExporter
    from reflex.storage.postgres import PostgresStorage

logger = get_logger(__name__)

# Current bot version
BOT_VERSION = "1.1.0"


def migrate_to_1_1_0(entry: Entry) -> None:
    """Migration to version 1.1.0: Add next_action_date field.

    Sets next_action_date to None for all entries, which means they will
    show immediately in the digest.

    Args:
        entry: Entry to migrate
    """
    # next_action_date field will be added to Entry model in Phase 4
    # For now, this is a placeholder migration
    logger.info(f"Migrated entry {entry.id} to 1.1.0 (placeholder)")


def migrate_to_1_2_0(entry: Entry, classifier) -> None:  # type: ignore
    """Migration to version 1.2.0: Re-classify with English-only prompt.

    Re-runs classification on the original message using the updated prompt
    that explicitly requests English-only tags and reasoning.

    Args:
        entry: Entry to migrate
        classifier: ReflexClassifier instance for re-classification
    """
    if entry.original_message is None:
        logger.warning(f"Cannot re-classify entry {entry.id}: no original_message")
        return

    logger.info(f"Re-classifying entry {entry.id} with English-only prompt")
    result = classifier.classify(entry.original_message)

    # Update classification fields
    entry.tags = result.suggested_tags
    entry.llm_reasoning = result.reasoning
    entry.llm_confidence = result.confidence
    entry.llm_model = result.model
    # Don't change category - that would be too disruptive

    logger.info(f"Migrated entry {entry.id} to 1.2.0 (re-classified)")


# Migration registry: version -> migration function
# Migrations are applied sequentially in version order
MIGRATIONS = {
    "1.1.0": migrate_to_1_1_0,
    # "1.2.0": migrate_to_1_2_0,  # Commented out for now - requires classifier parameter
}


def migrate_entry(
    entry: Entry,
    target_version: str,
    storage: "PostgresStorage",
    exporter: Optional["MarkdownExporter"] = None,
) -> None:
    """Apply all migrations from entry.bot_version to target_version.

    Migrations are applied sequentially in version order. After each migration,
    the entry is updated in the database and re-exported to markdown.

    Args:
        entry: Entry to migrate
        target_version: Version to migrate to (usually BOT_VERSION)
        storage: PostgresStorage instance for updating the entry
        exporter: MarkdownExporter instance for re-exporting (optional)

    Raises:
        Exception: If migration fails
    """
    current = entry.bot_version or "0.0.0"

    if current >= target_version:
        logger.debug(f"Entry {entry.id} already at version {current}, no migration needed")
        return

    # Find migrations to apply
    versions_needed = sorted([v for v in MIGRATIONS.keys() if current < v <= target_version])

    if not versions_needed:
        # No migrations defined, just update version
        logger.info(f"No migrations defined between {current} and {target_version}, updating version")
        entry.bot_version = target_version
        storage.update_entry(entry)
        return

    logger.info(f"Migrating entry {entry.id}: {current} → {target_version} ({len(versions_needed)} migrations)")

    for version in versions_needed:
        migration_func = MIGRATIONS[version]

        try:
            # Apply migration
            migration_func(entry)

            # Update version after successful migration
            entry.bot_version = version
            storage.update_entry(entry)

            # Re-export markdown with updated fields
            if exporter:
                exporter.export_entry(entry)
                logger.info(f"Re-exported entry {entry.id} after migration to {version}")

        except Exception as e:
            logger.error(f"Migration to {version} failed for entry {entry.id}: {e}", exc_info=True)
            raise

    logger.info(f"Successfully migrated entry {entry.id} to {target_version}")
