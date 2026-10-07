"""The classification that files a capture must come from a closed set.

`docs/agent-isolation.md` requires it: "the model's output is validated against
a closed set before anything acts on it." The set already existed in this
codebase -- bot.py's VALID_CATEGORIES -- where it guarded the move-command
parse and NOT this path, the one that decides where a capture is filed.

Before: `category` defaulted to "inbox", a REAL category, so a truncated or
malformed response filed a capture instead of failing. `confidence` defaulted
to 0.5, which classify() compares against tier1_threshold (0.7) to decide
whether to escalate -- a manufactured number reaching a live branch.
"""

from __future__ import annotations

import json
from unittest import mock

import httpx
import pytest
from cortex_utils.llm import LLMClient, LLMError

from reflex.services.classifier import VALID_CATEGORIES, ReflexClassifier


def _classifier(payload):
    c = ReflexClassifier.__new__(ReflexClassifier)
    c.tier1_model, c.tier2_model = "m1", "m2"
    c.tier1_threshold, c.tier2_threshold = 0.7, 0.5
    c.llm = LLMClient("http://llm.test")
    c.llm.client = mock.Mock()
    c.llm.client.post.return_value = httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}]},
        request=httpx.Request("POST", "http://llm.test"),
    )
    return c


class TestTheCategoryComesFromTheClosedSet:
    def test_a_complete_valid_response_classifies(self):
        """The control. Refusing everything would satisfy the tests below."""
        r = _classifier({"category": "admin", "confidence": 0.9, "reasoning": "r"})
        result = r._classify_single("msg", "m1")
        assert result.category == "admin"
        assert result.confidence == 0.9

    @pytest.mark.parametrize("category", sorted(VALID_CATEGORIES))
    def test_every_member_of_the_set_is_accepted(self, category):
        r = _classifier({"category": category, "confidence": 0.8, "reasoning": "r"})
        assert r._classify_single("msg", "m1").category == category

    @pytest.mark.parametrize("category", ["urgent", "spam", "Inbox", "", "INBOX", "admin "])
    def test_a_category_outside_the_set_is_refused(self, category):
        """Includes near-misses: case and whitespace are not members."""
        r = _classifier({"category": category, "confidence": 0.9, "reasoning": "r"})
        with pytest.raises(LLMError, match="not one of"):
            r._classify_single("msg", "m1")

    def test_a_missing_category_is_refused_rather_than_becoming_inbox(self):
        r = _classifier({"confidence": 0.9, "reasoning": "r"})
        with pytest.raises(LLMError, match="category"):
            r._classify_single("msg", "m1")


class TestConfidenceIsNeverInvented:
    def test_a_missing_confidence_is_refused_rather_than_becoming_point_five(self):
        """0.5 is compared against tier1_threshold to decide escalation."""
        r = _classifier({"category": "admin", "reasoning": "r"})
        with pytest.raises(LLMError, match="confidence"):
            r._classify_single("msg", "m1")

    def test_a_non_numeric_confidence_is_refused(self):
        r = _classifier({"category": "admin", "confidence": "high", "reasoning": "r"})
        with pytest.raises(LLMError, match="confidence"):
            r._classify_single("msg", "m1")


def test_the_set_matches_the_one_the_move_command_already_used():
    """Two copies of a closed set that disagree is worse than one.

    bot.py's VALID_CATEGORIES guards the move-command parse. If these drift, a
    category the bot accepts by hand becomes one the classifier refuses.
    """
    from reflex.services.bot import ReflexBot

    assert set(ReflexBot.VALID_CATEGORIES) == set(VALID_CATEGORIES)
