"""LLM classification service with two-tier cascade."""

import json

from cortex_utils.llm import LLMClient, LLMError
from cortex_utils.logging import get_logger

from reflex.models.entry import ClassificationResult

logger = get_logger(__name__)

# Classification prompt template
CLASSIFICATION_PROMPT = """Classify this note into one category:

- **person**: Information about a person (relationship, contact, notes)
- **project**: Work related to a project or goal
- **idea**: Random thought, inspiration, future possibility
- **admin**: Task, errand, administrative matter
- **inbox**: External idea needing review (bookmark, article)

Also determine if this note is **actionable** (requires user action):
- **actionable=true**: Tasks, reminders, errands, todos, things that need doing
- **actionable=false**: Reference information, ideas for later, notes about people

Note: "{message}"

Return JSON (use English only for all fields):
{{
  "category": "...",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "suggested_tags": ["tag1", "tag2"],
  "actionable": true|false
}}
"""

# Intent validation prompt template
INTENT_VALIDATION_PROMPT = """Is this a command about existing entries, or a new thought to capture?

**COMMAND**: References existing entries ("that idea", "last note") and wants to move/tag/archive.
**CAPTURE**: Everything else - new thoughts to remember.

Message: "{message}"

Return JSON:
{{
  "is_command": true|false,
  "confidence": 0.0-1.0
}}
"""


# THE CLOSED SET the model's output is validated against. docs/agent-isolation
# requires this: "the model's output is validated against a closed set before
# anything acts on it." It already existed as bot.py's VALID_CATEGORIES, where
# it guarded the move-command parse and NOT this -- the classification that
# decides where a capture is filed. Same list, now applied to both.
VALID_CATEGORIES = frozenset({"person", "project", "idea", "admin", "inbox"})


class ReflexClassifier:
    """Two-tier LLM classifier for Reflex captures."""

    def __init__(
        self,
        litellm_base_url: str,
        tier1_model: str,
        tier2_model: str,
        tier1_threshold: float = 0.7,
        tier2_threshold: float = 0.6,
    ):
        """Initialize classifier with LiteLLM client.

        Args:
            litellm_base_url: Base URL for LiteLLM proxy
            tier1_model: Model for tier 1 (e.g., ollama/qwen2.5:7b)
            tier2_model: Model for tier 2 (e.g., gemini/gemini-1.5-flash)
            tier1_threshold: Confidence threshold for tier 1 (default 0.7)
            tier2_threshold: Confidence threshold for tier 2 (default 0.6)
        """
        self.llm = LLMClient(litellm_base_url)
        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold

    def validate_intent(self, message: str) -> tuple[bool, float]:
        """Validate if message is a command (two-tier cascade).

        Args:
            message: The message to validate

        Returns:
            Tuple of (is_command, confidence)

        Raises:
            LLMError: If both tiers fail
        """
        is_command: bool = False
        confidence: float = 0.0
        # Try Tier 1
        try:
            is_command, confidence = self._validate_intent_single(
                message, self.tier1_model
            )
            if confidence >= self.tier1_threshold:
                logger.info(
                    f"Intent validation tier 1: is_command={is_command}, confidence={confidence}"
                )
                return is_command, confidence
        except LLMError as e:
            logger.warning(f"Tier 1 intent validation failed: {e}, trying tier 2")

        # Escalate to Tier 2
        logger.info(
            f"Intent validation tier 1 confidence {confidence:.2f} "
            f"< {self.tier1_threshold}, escalating"
        )
        is_command, confidence = self._validate_intent_single(message, self.tier2_model)
        logger.info(
            f"Intent validation tier 2: is_command={is_command}, confidence={confidence}"
        )
        return is_command, confidence

    @staticmethod
    def _confidence(data: dict, where: str) -> float:
        """Parse a model-supplied confidence, or raise.

        No default and no clamp. Both invent a number, and this number gates
        execution: bot.py:477 runs a command when confidence >= the tier-2
        threshold, which is env-set (bot.py:239). float() alone is not a
        guard -- float(True) is 1.0, bools do not raise, and "inf"/"1e9"
        parse fine, so an out-of-range value reads as total certainty. The
        range test also rejects nan, since every nan comparison is False.
        """
        if "confidence" not in data:
            raise LLMError(f"{where} has no 'confidence'; response is incomplete")
        raw = data["confidence"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise LLMError(f"{where} confidence is not a number: {raw!r}")
        try:
            value = float(raw)
        except (ValueError, TypeError) as e:
            raise LLMError(f"{where} confidence is not a number: {raw!r}") from e
        if not 0.0 <= value <= 1.0:
            raise LLMError(f"{where} confidence out of range [0,1]: {value!r}")
        return value

    def _validate_intent_single(self, message: str, model: str) -> tuple[bool, float]:
        """Validate intent with a single model.

        Args:
            message: The message to validate
            model: Model name to use

        Returns:
            Tuple of (is_command, confidence)

        Raises:
            LLMError: If LLM call fails
        """
        prompt = INTENT_VALIDATION_PROMPT.format(message=message)

        try:
            result = self.llm._post_completion(
                model=model,
                prompt=prompt,
                max_tokens=50,
                response_format={"type": "json_object"},
            )
            content = self.llm._get_content_from_response(result)
            data = json.loads(content)

            # `is True`, not bool(): bool("false") is True, so a model
            # SAYING not-a-command became is_command=True and, at a plausible
            # confidence, executed one. Only the literal JSON true counts;
            # every other value -- absent, "false", 0, None, a dict -- means
            # not a command, so this is the direction that fails closed.
            is_command = data.get("is_command") is True
            # This is the path that EXECUTES, so confidence gets no default
            # and no fallback -- see _confidence.
            return is_command, self._confidence(data, "intent validation")

        except json.JSONDecodeError as e:
            logger.error(f"Intent validation returned invalid JSON: {e}")
            raise LLMError(f"Invalid JSON: {e}") from e

    def classify(self, message: str) -> ClassificationResult:
        """Classify a capture message (two-tier cascade).

        Args:
            message: The message to classify

        Returns:
            ClassificationResult with category, confidence, reasoning, tags, model

        Raises:
            LLMError: If both tiers fail
        """
        result: ClassificationResult | None = None
        # Try Tier 1
        try:
            result = self._classify_single(message, self.tier1_model)
            if result.confidence >= self.tier1_threshold:
                logger.info(
                    f"Classification tier 1: {result.category} (confidence={result.confidence})"
                )
                return result
        except LLMError as e:
            logger.warning(f"Tier 1 classification failed: {e}, trying tier 2")
            # Continue to tier 2

        # Escalate to Tier 2
        if result:
            logger.info(
                f"Classification tier 1 confidence {result.confidence:.2f} "
                f"< {self.tier1_threshold}, escalating"
            )
        result = self._classify_single(message, self.tier2_model)
        logger.info(
            f"Classification tier 2: {result.category} (confidence={result.confidence})"
        )
        return result

    def _classify_single(self, message: str, model: str) -> ClassificationResult:
        """Classify with a single model.

        Args:
            message: The message to classify
            model: Model name to use

        Returns:
            ClassificationResult

        Raises:
            LLMError: If LLM call fails
        """
        prompt = CLASSIFICATION_PROMPT.format(message=message)

        try:
            # Get raw completion for full JSON parsing
            result_dict = self.llm._post_completion(
                model=model,
                prompt=prompt,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            content = self.llm._get_content_from_response(result_dict)
            data = json.loads(content)

            # PRESENCE, then the closed set. A default turns "the model said
            # nothing" into "the model said this", and nothing downstream can
            # tell the difference: `category` defaulted to "inbox", a REAL
            # category, so a response missing the field filed a capture
            # instead of failing. An out-of-set category was already refused
            # by the CHECK in migrations/001 -- so that half of this is
            # defence in depth in front of the database, and it earns its
            # place by naming the failure instead of raising IntegrityError.
            # The missing-field half is the part the CHECK could not see.
            if "category" not in data:
                raise LLMError("classification has no 'category'; response is incomplete")
            category = data["category"]
            # isinstance first: `["admin"] not in VALID_CATEGORIES` raises
            # TypeError (unhashable), which no handler here catches, so the
            # cascade is skipped and the operator sees a raw TypeError.
            if not isinstance(category, str) or category not in VALID_CATEGORIES:
                raise LLMError(
                    f"model returned category {category!r}, which is not one of "
                    f"{sorted(VALID_CATEGORIES)}"
                )
            confidence = self._confidence(data, "classification")

            reasoning = data.get("reasoning", "")
            suggested_tags = data.get("suggested_tags", [])
            if not isinstance(suggested_tags, list):
                suggested_tags = []

            actionable = bool(data.get("actionable", False))

            return ClassificationResult(
                category=category,
                confidence=confidence,
                reasoning=reasoning,
                suggested_tags=suggested_tags,
                model=model,
                actionable=actionable,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Classification returned invalid JSON: {e}")
            raise LLMError(f"Invalid JSON: {e}") from e
