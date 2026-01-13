"""LLM classification service with two-tier cascade."""

import json
import os
from typing import Optional

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

Note: "{message}"

Return JSON:
{{
  "category": "...",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "suggested_tags": ["tag1", "tag2"]
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
            f"Intent validation tier 1 confidence {confidence:.2f} < {self.tier1_threshold}, escalating"
        )
        is_command, confidence = self._validate_intent_single(message, self.tier2_model)
        logger.info(
            f"Intent validation tier 2: is_command={is_command}, confidence={confidence}"
        )
        return is_command, confidence

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
            # Use classify method which returns JSON
            category, confidence, _ = self.llm.classify(prompt, model)
            # Parse the response - category will be the JSON string
            # Actually, we need to get the raw response. Let me check the client again.
            # The classify method expects specific format. Let me use a custom call instead.

            # Actually, looking at the client, classify expects category/confidence/reasoning
            # For intent validation, we need a custom JSON parse. Let me make a helper.
            result = self.llm._post_completion(
                model=model,
                prompt=prompt,
                max_tokens=50,
                response_format={"type": "json_object"},
            )
            content = self.llm._get_content_from_response(result)
            data = json.loads(content)

            is_command = bool(data.get("is_command", False))
            raw_confidence = data.get("confidence", 0.5)
            try:
                confidence = float(raw_confidence)
            except (ValueError, TypeError):
                logger.warning(f"Invalid confidence from intent validation: {raw_confidence}")
                confidence = 0.5

            return is_command, confidence

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
        logger.info(
            f"Classification tier 1 confidence {result.confidence:.2f} < {self.tier1_threshold}, escalating"
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

            # Parse fields
            category = data.get("category", "inbox")
            raw_confidence = data.get("confidence", 0.5)
            try:
                confidence = float(raw_confidence)
            except (ValueError, TypeError):
                logger.warning(f"Invalid confidence: {raw_confidence}")
                confidence = 0.5

            reasoning = data.get("reasoning", "")
            suggested_tags = data.get("suggested_tags", [])
            if not isinstance(suggested_tags, list):
                suggested_tags = []

            return ClassificationResult(
                category=category,
                confidence=confidence,
                reasoning=reasoning,
                suggested_tags=suggested_tags,
                model=model,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Classification returned invalid JSON: {e}")
            raise LLMError(f"Invalid JSON: {e}") from e
