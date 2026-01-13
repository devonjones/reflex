#!/usr/bin/env python3
"""Run database migrations for Reflex.

This script applies SQL migrations to Postgres only.
DuckDB is accessed via the postmark DuckDB API, no direct schema management needed.
"""

import os
import sys
from pathlib import Path

import psycopg2
from cortex_utils.logging import configure_logging, get_logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

configure_logging("reflex-migrations", level="INFO")
logger = get_logger(__name__)


def run_postgres_migrations(conn: psycopg2.extensions.connection) -> None:
    """Run Postgres migrations."""
    migrations_dir = Path(__file__).parent.parent / "migrations"
    postgres_migrations = sorted(migrations_dir.glob("*.sql"))

    for migration_file in postgres_migrations:
        logger.info(f"Running Postgres migration: {migration_file.name}")
        with open(migration_file) as f:
            sql = f.read()

        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            logger.info(f"✓ Applied {migration_file.name}")
        except Exception as e:
            conn.rollback()
            logger.error(f"✗ Failed to apply {migration_file.name}: {e}")
            raise


def main() -> None:
    """Main entry point."""
    # Get database credentials from environment
    postgres_host = os.getenv("POSTGRES_HOST", "10.5.2.21")
    postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db = os.getenv("POSTGRES_DB", "cortex")
    postgres_user = os.getenv("POSTGRES_USER", "cortex")
    postgres_password = os.getenv("POSTGRES_PASSWORD")

    if not postgres_password:
        logger.error("POSTGRES_PASSWORD environment variable not set")
        sys.exit(1)

    # Run Postgres migrations
    logger.info("Connecting to Postgres...")
    try:
        pg_conn = psycopg2.connect(
            host=postgres_host,
            port=postgres_port,
            dbname=postgres_db,
            user=postgres_user,
            password=postgres_password,
        )
        run_postgres_migrations(pg_conn)
        pg_conn.close()
        logger.info("✓ Postgres migrations complete")
    except Exception as e:
        logger.error(f"✗ Postgres migration failed: {e}")
        sys.exit(1)

    logger.info("✓ All migrations completed successfully")
    logger.info("Note: DuckDB is accessed via postmark DuckDB API - no schema migration needed")


if __name__ == "__main__":
    main()
