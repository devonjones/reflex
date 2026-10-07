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


def _cascade(*payloads):
    """A classifier whose successive LLM calls return `payloads` in order.

    classify() is where this PR's behaviour change actually lives: tier 1
    raising is now ROUTINE (any out-of-set category), so the catch-and-escalate
    went from a rare path to a hot one, and tier 2 raising is a brand-new
    terminal outcome. Neither was tested on this branch or on main.
    """
    c = ReflexClassifier("http://llm.test", "m1", "m2", 0.7, 0.6)
    c.llm.client = mock.Mock()
    c.llm.client.post.side_effect = [
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(p)}, "finish_reason": "stop"}]},
            request=httpx.Request("POST", "http://llm.test"),
        )
        for p in payloads
    ]
    return c


class TestTheCascade:
    def test_tier1_out_of_set_escalates_and_tier2_is_returned(self):
        c = _cascade(
            {"category": "urgent", "confidence": 0.9, "reasoning": "t1"},
            {"category": "idea", "confidence": 0.9, "reasoning": "t2"},
        )
        assert c.classify("msg").category == "idea"
        assert c.llm.client.post.call_count == 2

    def test_tier2_also_failing_propagates_rather_than_filing(self):
        """The one that matters.

        A mutant wrapping the tier-2 call so an LLMError returns
        category="inbox", confidence=0.5 passed all 17 tests -- this PR's exact
        defect reinstated one level up. Raising drops the capture and shows the
        user a visible failure, which is the right direction under
        docs/agent-isolation.md: nothing is filed on a fabricated category.
        """
        c = _cascade(
            {"category": "urgent", "confidence": 0.9, "reasoning": "t1"},
            {"category": "alsobad", "confidence": 0.9, "reasoning": "t2"},
        )
        with pytest.raises(LLMError):
            c.classify("msg")

    def test_tier1_below_threshold_escalates(self):
        """0.5 < 0.7, so tier 1's answer is not returned even though it is valid."""
        c = _cascade(
            {"category": "admin", "confidence": 0.5, "reasoning": "t1"},
            {"category": "idea", "confidence": 0.95, "reasoning": "t2"},
        )
        assert c.classify("msg").category == "idea"
        assert c.llm.client.post.call_count == 2

    def test_tier1_at_or_above_threshold_short_circuits(self):
        """The control: escalating always would also pass the tests above."""
        c = _cascade({"category": "admin", "confidence": 0.7, "reasoning": "t1"})
        assert c.classify("msg").category == "admin"
        assert c.llm.client.post.call_count == 1, "tier 2 must not be called"


class TestIntentValidationNeverInventsAConfidence:
    """A fabricated confidence can EXECUTE a command, not just file a capture.

    bot.py:477 runs `if is_command and confidence >= tier2_threshold`, and that
    threshold comes from REFLEX_LLM_TIER2_CONFIDENCE_THRESHOLD (default 0.6).
    The old default of 0.5 was below it only by coincidence of two
    independently configured values.
    """

    def test_a_missing_confidence_is_refused(self):
        c = _cascade({"is_command": True})
        with pytest.raises(LLMError, match="confidence"):
            c._validate_intent_single("move 3 to admin", "m1")

    def test_is_command_still_defaults_closed(self):
        """Absent is_command means 'not a command', which is the safe default."""
        c = _cascade({"confidence": 0.95})
        is_command, confidence = c._validate_intent_single("hello", "m1")
        assert is_command is False
        assert confidence == 0.95
