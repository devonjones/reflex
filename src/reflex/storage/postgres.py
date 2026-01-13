"""Postgres storage for Reflex entries."""

import base64
from datetime import datetime
from typing import Optional

import httpx
import psycopg2
import psycopg2.extras
from cortex_utils.logging import get_logger

from reflex.models.entry import Entry

logger = get_logger(__name__)


class PostgresStorage:
    """Storage layer for Reflex entries."""

    def __init__(
        self,
        postgres_conn: psycopg2.extensions.connection,
        duckdb_api_url: str,
    ):
        """Initialize storage.

        Args:
            postgres_conn: Postgres connection
            duckdb_api_url: DuckDB API base URL (e.g., http://cortex-duckdb-api:8081)
        """
        self.conn = postgres_conn
        self.duckdb_api_url = duckdb_api_url.rstrip("/")
        self.http_client = httpx.Client(timeout=30.0)

    def store_entry(self, entry: Entry) -> int:
        """Store entry in Postgres + DuckDB.

        Args:
            entry: Entry to store

        Returns:
            The entry ID

        Raises:
            Exception: If storage fails
        """
        # Insert into Postgres (don't commit yet - wait for DuckDB)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reflex_entries (
                    discord_message_id,
                    discord_channel_id,
                    discord_user_id,
                    category,
                    title,
                    tags,
                    llm_confidence,
                    llm_model,
                    llm_reasoning,
                    status,
                    captured_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    entry.discord_message_id,
                    entry.discord_channel_id,
                    entry.discord_user_id,
                    entry.category,
                    entry.title,
                    entry.tags,
                    entry.llm_confidence,
                    entry.llm_model,
                    entry.llm_reasoning,
                    entry.status,
                    entry.captured_at,
                ),
            )
            entry_id = cur.fetchone()[0]

        logger.info(f"Inserted entry {entry_id} in Postgres (not committed yet)")

        # Store body in DuckDB via API
        try:
            self._store_body_in_duckdb(entry_id, entry.original_message)
        except Exception as e:
            logger.error(f"Failed to store body in DuckDB: {e}")
            # Rollback Postgres insert
            self.conn.rollback()
            raise

        # Both operations succeeded - commit the transaction
        self.conn.commit()
        logger.info(f"Committed entry {entry_id} to Postgres")

        return entry_id

    def _store_body_in_duckdb(self, entry_id: int, message: str) -> None:
        """Store message body in DuckDB via API.

        Args:
            entry_id: Entry ID (used as gmail_id key)
            message: Original message text

        Raises:
            httpx.HTTPStatusError: If API call fails
        """
        # Encode message as base64 (DuckDB API expects this format from email storage)
        body_data = base64.b64encode(message.encode("utf-8")).decode("utf-8")

        response = self.http_client.post(
            f"{self.duckdb_api_url}/body",
            json={
                "gmail_id": str(entry_id),  # Use entry_id as key
                "parts": [],  # Empty parts for now
                "body_data": body_data,
            },
        )
        response.raise_for_status()
        logger.info(f"Stored body for entry {entry_id} in DuckDB")

    def get_entry(self, entry_id: int) -> Optional[Entry]:
        """Retrieve entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            Entry or None if not found
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM reflex_entries WHERE id = %s
                """,
                (entry_id,),
            )
            row = cur.fetchone()

        if not row:
            return None

        # Fetch body from DuckDB
        try:
            body_response = self.http_client.get(
                f"{self.duckdb_api_url}/body",
                params={"gmail_id": str(entry_id)},
            )
            body_response.raise_for_status()
            body_data = body_response.json()
            # Decode base64 body_data
            original_message = base64.b64decode(body_data["body_data"]).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to fetch body for entry {entry_id}: {e}")
            original_message = ""

        return Entry(
            id=row["id"],
            discord_message_id=row["discord_message_id"],
            discord_channel_id=row["discord_channel_id"],
            discord_user_id=row["discord_user_id"],
            category=row["category"],
            title=row["title"],
            tags=row["tags"],
            llm_confidence=row["llm_confidence"],
            llm_model=row["llm_model"],
            llm_reasoning=row["llm_reasoning"],
            status=row["status"],
            captured_at=row["captured_at"],
            updated_at=row["updated_at"],
            exported_to_git=row["exported_to_git"],
            git_commit_sha=row["git_commit_sha"],
            markdown_path=row["markdown_path"],
            original_message=original_message,
        )

    def get_recent_entries(
        self, channel_id: str, limit: int = 10, category: Optional[str] = None
    ) -> list[Entry]:
        """Get recent entries from a channel.

        Args:
            channel_id: Discord channel ID
            limit: Maximum number of entries to return
            category: Optional category filter

        Returns:
            List of entries (most recent first)
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if category:
                cur.execute(
                    """
                    SELECT * FROM reflex_entries
                    WHERE discord_channel_id = %s AND category = %s
                    ORDER BY captured_at DESC
                    LIMIT %s
                    """,
                    (channel_id, category, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM reflex_entries
                    WHERE discord_channel_id = %s
                    ORDER BY captured_at DESC
                    LIMIT %s
                    """,
                    (channel_id, limit),
                )

            rows = cur.fetchall()

        entries = []
        for row in rows:
            # Note: We're not fetching bodies for list queries to avoid N+1 problem
            # Bodies can be fetched individually if needed
            entries.append(
                Entry(
                    id=row["id"],
                    discord_message_id=row["discord_message_id"],
                    discord_channel_id=row["discord_channel_id"],
                    discord_user_id=row["discord_user_id"],
                    category=row["category"],
                    title=row["title"],
                    tags=row["tags"],
                    llm_confidence=row["llm_confidence"],
                    llm_model=row["llm_model"],
                    llm_reasoning=row["llm_reasoning"],
                    status=row["status"],
                    captured_at=row["captured_at"],
                    updated_at=row["updated_at"],
                    exported_to_git=row["exported_to_git"],
                    git_commit_sha=row["git_commit_sha"],
                    markdown_path=row["markdown_path"],
                    original_message="",  # Not fetched in list queries
                )
            )

        return entries
