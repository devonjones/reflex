"""The classification that files a capture must come from a closed set.

`docs/agent-isolation.md` requires it: "the model's output is validated against
a closed set before anything acts on it." The set already existed in this
codebase -- bot.py's VALID_CATEGORIES -- where it guarded the move-command
parse and NOT this path, the one that decides where a capture is filed.

Before: `category` defaulted to "inbox", a REAL category, so a response that
OMITTED the field filed a capture instead of failing. An out-of-set category
was already refused by the CHECK in migrations/001 -- so the set test here is
defence in depth in front of the database, and it earns its place by naming
the failure instead of raising IntegrityError. The omitted-field case is the
part the CHECK could not see.

`confidence` defaulted to 0.5, which classify() compares against
tier1_threshold (0.7) to decide whether to escalate, and which on the INTENT
path gates executing a command -- a manufactured number reaching a live
branch.
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

    @pytest.mark.parametrize("raw", [True, False, "inf", "1e9", 42.0, -0.5, "nan"])
    def test_a_confidence_outside_zero_to_one_is_refused(self, raw):
        """float() is not a guard.

        float(True) is 1.0 and bools do not raise, so a bool reads as total
        certainty. "inf" and "1e9" parse fine. nan is refused because every
        comparison against it is False, so `0.0 <= nan <= 1.0` fails.
        """
        r = _classifier({"category": "admin", "confidence": raw, "reasoning": "r"})
        with pytest.raises(LLMError, match="confidence"):
            r._classify_single("msg", "m1")

    @pytest.mark.parametrize("category", [["admin"], {"admin": 1}, 3, None])
    def test_a_non_string_category_is_refused_without_a_type_error(self, category):
        """`["admin"] not in VALID_CATEGORIES` raises TypeError: unhashable.

        No handler in _classify_single or classify() catches TypeError, so it
        would skip the cascade and surface raw to the operator. The isinstance
        test has to come first.
        """
        r = _classifier({"category": category, "confidence": 0.9, "reasoning": "r"})
        with pytest.raises(LLMError, match="not one of"):
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

    @pytest.mark.parametrize("raw", ["false", "no", "0", "False", "off", {"a": 1}, [0]])
    def test_a_present_non_true_is_command_does_not_become_true(self, raw):
        """bool("false") is True, so the model SAYING no meant yes.

        The original comment here claimed this path failed closed. It did so
        only for an ABSENT key: any present non-empty string is truthy, so
        {"is_command": "false", "confidence": 0.95} cleared bot.py:477's
        threshold and ran a command against an existing entry -- on a message
        the model had just declined to treat as a command.
        """
        c = _cascade({"is_command": raw, "confidence": 0.95})
        is_command, _ = c._validate_intent_single("remind me to archive this", "m1")
        assert is_command is False

    def test_a_literal_true_is_still_honoured(self):
        """The control: failing closed must not mean never recognising a command."""
        c = _cascade({"is_command": True, "confidence": 0.95})
        assert c._validate_intent_single("archive 3", "m1") == (True, 0.95)

    @pytest.mark.parametrize("raw", ["high", "very high", None, {}, []])
    def test_a_non_numeric_confidence_is_refused_not_turned_into_point_five(
        self, raw
    ):
        """The case the first version of this class missed.

        It guarded `"confidence" not in data` and left `except (ValueError,
        TypeError): confidence = 0.5` four lines below -- same method, same
        consequence, and `{"confidence": "high"}` is a likelier model output
        than an absent key. Three reviewers found it independently.
        """
        c = _cascade({"is_command": True, "confidence": raw})
        with pytest.raises(LLMError, match="confidence"):
            c._validate_intent_single("archive that", "m1")

    @pytest.mark.parametrize("raw", [True, "inf", 42.0, -1.0])
    def test_an_out_of_range_intent_confidence_is_refused(self, raw):
        """This value is the SOLE gate on executing a command (bot.py:477).

        There is no database CHECK behind the intent path the way there is
        behind a filed capture, so an out-of-range value here is unopposed.
        """
        c = _cascade({"is_command": True, "confidence": raw})
        with pytest.raises(LLMError, match="confidence"):
            c._validate_intent_single("archive that", "m1")


class TestTheIntentCascade:
    """validate_intent() is the structurally identical sibling of classify().

    Mutation testing found it entirely unpinned: breaking the escalation and
    moving the threshold boundary both left the suite green. _validate_intent_single
    was tested directly, the public method never.
    """

    def test_tier1_raising_escalates_to_tier2(self):
        c = _cascade({"is_command": True}, {"is_command": True, "confidence": 0.9})
        assert c.validate_intent("archive that") == (True, 0.9)
        assert c.llm.client.post.call_count == 2

    def test_tier1_below_threshold_escalates(self):
        c = _cascade(
            {"is_command": True, "confidence": 0.5},
            {"is_command": True, "confidence": 0.95},
        )
        assert c.validate_intent("archive that") == (True, 0.95)
        assert c.llm.client.post.call_count == 2

    def test_tier1_at_the_threshold_short_circuits(self):
        """The boundary. 0.7 >= 0.7, so tier 2 must not be called."""
        c = _cascade({"is_command": True, "confidence": 0.7})
        assert c.validate_intent("archive that") == (True, 0.7)
        assert c.llm.client.post.call_count == 1, "tier 2 must not be called"

    def test_tier2_also_failing_propagates(self):
        """Both tiers refusing must raise, not return a fabricated pair.

        A swallow here would hand bot.py:477 an (is_command, confidence) that
        no model produced.
        """
        c = _cascade({"is_command": True}, {"is_command": True})
        with pytest.raises(LLMError, match="confidence"):
            c.validate_intent("archive that")


def test_the_set_matches_the_database_check_that_actually_guards_the_path():
    """The load-bearing joint, and the one nothing pinned.

    Eight copies of this set exist in reflex. The copy that guards the only
    model-derived target -- category selects the export DIRECTORY -- is the
    CHECK in migrations/001, not any Python list. A migration that relaxes it,
    or a sixth category added to the frozenset but not the CHECK, opens that
    path with a green suite.
    """
    import pathlib
    import re

    sql = pathlib.Path("migrations/001_create_reflex_entries.sql").read_text()
    m = re.search(r"category TEXT NOT NULL CHECK \(category IN \(([^)]*)\)\)", sql)
    assert m, "the CHECK constraint on category is gone -- that IS the finding"
    assert {v.strip().strip("'") for v in m.group(1).split(",")} == set(VALID_CATEGORIES)
