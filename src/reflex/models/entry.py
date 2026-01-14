"""Data models for Reflex entries."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Entry:
    """A Reflex knowledge capture entry."""

    id: Optional[int]
    discord_message_id: str
    discord_channel_id: str
    discord_user_id: str
    category: str
    title: str
    tags: list[str]
    llm_confidence: float
    llm_model: str
    llm_reasoning: Optional[str]
    status: str
    captured_at: Optional[datetime] = None  # Let DB handle via DEFAULT NOW()
    updated_at: Optional[datetime] = None  # Let DB handle via trigger
    exported_to_git: bool = False
    git_commit_sha: Optional[str] = None
    markdown_path: Optional[str] = None
    original_message: Optional[str] = None  # For DuckDB storage, None if fetch failed
    bot_version: Optional[str] = None  # Version of bot that created/last migrated this entry
    next_action_date: Optional[datetime] = None  # When to next show in digest (NULL = show now, future = snoozed)
    actionable: bool = False  # Whether this entry requires action (tasks, reminders, etc.)


@dataclass
class ClassificationResult:
    """Result of LLM classification."""

    category: str
    confidence: float
    reasoning: str
    suggested_tags: list[str]
    model: str
    actionable: bool = False  # Whether this entry requires user action
