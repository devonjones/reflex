"""Markdown export with git operations."""

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import git
from cortex_utils.logging import get_logger
from slugify import slugify

from reflex.models.entry import Entry

logger = get_logger(__name__)


class MarkdownExporter:
    """Export entries to markdown files with git commits."""

    def __init__(self, repo_path: str, git_remote: Optional[str] = None):
        """Initialize exporter.

        Args:
            repo_path: Path to git repository
            git_remote: Optional git remote URL
        """
        self.repo_path = Path(repo_path)
        self.git_remote = git_remote

        # Initialize git repo if needed
        if not (self.repo_path / ".git").exists():
            self.repo_path.mkdir(parents=True, exist_ok=True)
            self.repo = git.Repo.init(self.repo_path)
            logger.info(f"Initialized git repo at {self.repo_path}")

            # Add remote if provided
            if self.git_remote:
                try:
                    self.repo.create_remote("origin", self.git_remote)
                    logger.info(f"Added remote: {self.git_remote}")
                except git.exc.GitCommandError as e:
                    logger.warning(f"Remote already exists or failed to add: {e}")
        else:
            self.repo = git.Repo(self.repo_path)

        # Create category directories
        for category in ["person", "project", "idea", "admin", "inbox", "archive"]:
            (self.repo_path / category).mkdir(exist_ok=True)

    def export_entry(self, entry: Entry) -> str:
        """Export entry to markdown file.

        Args:
            entry: Entry to export

        Returns:
            Path to exported file (relative to repo root)
        """
        # Entry must have captured_at for export
        assert entry.captured_at is not None, "Entry must have captured_at for export"
        captured_at = entry.captured_at  # Narrow type for mypy

        # Generate filename
        date_str = captured_at.strftime("%Y-%m-%d")
        slug = slugify(entry.title, max_length=50)
        filename = f"{date_str}-{slug}.md"

        # Determine directory
        if entry.status == "archived":
            directory = self.repo_path / "archive"
        else:
            directory = self.repo_path / entry.category

        directory.mkdir(exist_ok=True)
        filepath = directory / filename

        # Generate markdown content
        content = self._generate_markdown(entry)

        # Write file
        filepath.write_text(content, encoding="utf-8")
        logger.info(f"Exported entry {entry.id} to {filepath}")

        # Return relative path
        return str(filepath.relative_to(self.repo_path))

    def _generate_markdown(self, entry: Entry) -> str:
        """Generate markdown content for entry.

        Args:
            entry: Entry to export

        Returns:
            Markdown string
        """
        # YAML frontmatter
        frontmatter = f"""---
id: {entry.id}
discord_message_id: "{entry.discord_message_id}"
category: {entry.category}
tags: [{', '.join(entry.tags) if entry.tags else ''}]
captured_at: {captured_at.isoformat()}
llm_model: {entry.llm_model}
llm_confidence: {entry.llm_confidence}
status: {entry.status}
---

"""

        # Title
        title = f"# {entry.title}\n\n"

        # Original message
        original = f"{entry.original_message}\n\n"

        # LLM classification metadata
        metadata = f"""## LLM Classification

**Category**: {entry.category}
**Confidence**: {entry.llm_confidence:.2f}
**Reasoning**: {entry.llm_reasoning or 'N/A'}
"""

        return frontmatter + title + original + metadata

    def commit_and_push(
        self, filepath: str, entry: Entry, push: bool = True
    ) -> Optional[str]:
        """Commit file and optionally push to remote.

        Args:
            filepath: Path to file (relative to repo root)
            entry: Entry that was exported
            push: Whether to push to remote (default True)

        Returns:
            Commit SHA or None if commit failed
        """
        try:
            # Git add
            self.repo.index.add([filepath])

            # Git commit
            commit_message = f"reflex: Add {entry.category} - {entry.title}"
            commit = self.repo.index.commit(commit_message)
            logger.info(f"Committed {filepath}: {commit.hexsha[:7]}")

            # Git push (if remote configured and push=True)
            if push and self.git_remote:
                try:
                    origin = self.repo.remote("origin")
                    origin.push()
                    logger.info(f"Pushed commit {commit.hexsha[:7]} to remote")
                except git.exc.GitCommandError as e:
                    logger.warning(f"Failed to push to remote: {e}")
                    # Don't fail the whole operation if push fails

            return commit.hexsha

        except Exception as e:
            logger.error(f"Failed to commit {filepath}: {e}", exc_info=True)
            return None

    def export_and_commit_async(self, entry: Entry) -> None:
        """Export entry and commit in background thread.

        Args:
            entry: Entry to export
        """
        thread = threading.Thread(target=self._export_and_commit_sync, args=(entry,))
        thread.daemon = True
        thread.start()

    def _export_and_commit_sync(self, entry: Entry) -> None:
        """Synchronous export and commit (for background thread).

        Args:
            entry: Entry to export
        """
        try:
            filepath = self.export_entry(entry)
            commit_sha = self.commit_and_push(filepath, entry)

            if commit_sha:
                logger.info(
                    f"Successfully exported and committed entry {entry.id}: {commit_sha[:7]}"
                )
            else:
                logger.error(f"Failed to commit entry {entry.id}")

        except Exception as e:
            logger.error(
                f"Failed to export/commit entry {entry.id}: {e}", exc_info=True
            )
