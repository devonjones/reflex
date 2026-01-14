"""Entry migration system for Reflex.

When the bot version changes, entries may need to be updated to match the new schema
or have fields reprocessed with improved logic. This module handles automatic migrations.
"""

from typing import TYPE_CHECKING, Optional

from cortex_utils.logging import get_logger
from packaging.version import parse as parse_version

from reflex.models.entry import Entry

if TYPE_CHECKING:
    from reflex.storage.exporter import MarkdownExporter
    from reflex.storage.postgres import PostgresStorage

logger = get_logger(__name__)

# Current bot version
BOT_VERSION = "1.3.0"


def migrate_to_1_1_0(entry: Entry) -> None:
    """Migration to version 1.1.0: Initial version tracking.

    First migration that sets bot_version field. No data changes needed.

    Args:
        entry: Entry to migrate
    """
    logger.info(f"Migrated entry {entry.id} to 1.1.0 (version tracking)")


def migrate_to_1_2_0(entry: Entry) -> None:
    """Migration to version 1.2.0: Add next_action_date field.

    The SQL migration (002_add_next_action_date.sql) already added the column with DEFAULT NULL.
    All existing entries automatically have next_action_date = NULL (show immediately in digest).
    No data migration needed - just version bump.

    Args:
        entry: Entry to migrate
    """
    # next_action_date field added via SQL migration 002
    # Field already has DEFAULT NULL, so no data changes needed
    logger.info(f"Migrated entry {entry.id} to 1.2.0 (next_action_date field added)")


def migrate_to_1_3_0(entry: Entry) -> None:
    """Migration to version 1.3.0: Add actionable field.

    The SQL migration (003_add_actionable.sql) already added the column with DEFAULT FALSE.
    Set actionable=true for admin/project categories (typically tasks/work).
    Other categories (person, idea, inbox) default to false (reference material).

    Args:
        entry: Entry to migrate
    """
    # Classify existing entries based on category
    if entry.category in ("admin", "project"):
        entry.actionable = True
        logger.info(f"Migrated entry {entry.id} to 1.3.0 (actionable=true for {entry.category})")
    else:
        entry.actionable = False
        logger.info(f"Migrated entry {entry.id} to 1.3.0 (actionable=false for {entry.category})")


# Migration registry: version -> migration function
# Migrations are applied sequentially in version order
MIGRATIONS = {
    "1.1.0": migrate_to_1_1_0,
    "1.2.0": migrate_to_1_2_0,
    "1.3.0": migrate_to_1_3_0,
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
    current_v = parse_version(entry.bot_version or "0.0.0")
    target_v = parse_version(target_version)

    if current_v >= target_v:
        logger.debug(
            f"Entry {entry.id} already at version {current_v}, no migration needed"
        )
        return

    # Find migrations to apply
    versions_needed = sorted(
        [v for v in MIGRATIONS.keys() if current_v < parse_version(v) <= target_v],
        key=parse_version,
    )

    if not versions_needed:
        # No migrations defined, just update version
        logger.info(
            f"No migrations defined between {current_v} and {target_version}, updating version"
        )
        entry.bot_version = target_version
        storage.update_entry(entry)
        return

    logger.info(
        f"Migrating entry {entry.id}: {current_v} → {target_version} "
        f"({len(versions_needed)} migrations)"
    )

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
