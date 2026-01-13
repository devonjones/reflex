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
    captured_at: datetime
    updated_at: datetime
    exported_to_git: bool
    git_commit_sha: Optional[str]
    markdown_path: Optional[str]
    original_message: str  # For DuckDB storage


@dataclass
class ClassificationResult:
    """Result of LLM classification."""

    category: str
    confidence: float
    reasoning: str
    suggested_tags: list[str]
    model: str
