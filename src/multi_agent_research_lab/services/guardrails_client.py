"""NeMo Guardrails integration — validates queries before entering the pipeline.

Two-layer defence:
1. Regex pre-filter (fast, no API call) — blocks known attack patterns immediately.
2. NeMo Guardrails LLM-rail (Colang) — semantic check for edge cases.
"""

import logging
import re
from pathlib import Path
from typing import Any

from nemoguardrails import LLMRails, RailsConfig  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_GUARDRAILS_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent / "guardrails" / "config"
)

# Layer 1 — regex patterns that always block regardless of NeMo result
_BLOCK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(previous|all|prior|your)\s+instructions", re.I),
    re.compile(r"(you\s+are\s+now|act\s+as|pretend\s+you\s+are)\s+(DAN|unrestricted|evil)", re.I),
    re.compile(r"bypass\s+your\s+(safety|guidelines|restrictions)", re.I),
    re.compile(r"forget\s+(your\s+)?(system\s+prompt|instructions|guidelines)", re.I),
    re.compile(r"how\s+to\s+(make|build|synthesize|create)\s+a?\s*(bomb|explosive|malware|virus)", re.I),  # noqa: E501
    re.compile(r"how\s+to\s+(hack|exploit|attack|compromise|crack)\b", re.I),
    re.compile(r"(step.by.step|instructions?)\s+(to\s+)?(hack|attack|exploit|poison)", re.I),
    re.compile(r"<<\s*SYS\s*>>|\[INST\]", re.I),
]


class GuardrailsResult:
    def __init__(self, allowed: bool, reason: str = "") -> None:
        self.allowed = allowed
        self.reason = reason


class GuardrailsClient:
    """Wraps NeMo Guardrails to screen queries before the research pipeline."""

    def __init__(self) -> None:
        self._rails: LLMRails | None = self._load_rails()

    def _load_rails(self) -> "LLMRails | None":
        try:
            config = RailsConfig.from_path(str(_GUARDRAILS_CONFIG_PATH))
            rails = LLMRails(config)
            logger.info("NeMo Guardrails loaded from %s", _GUARDRAILS_CONFIG_PATH)
            return rails
        except Exception as exc:
            logger.warning("Guardrails failed to load (%s) — disabled.", exc)
            return None

    def check(self, query: str) -> GuardrailsResult:
        """Return GuardrailsResult(allowed=True) if query passes all rails."""
        # Layer 1: fast regex pre-filter
        for pattern in _BLOCK_PATTERNS:
            if pattern.search(query):
                reason = f"Blocked by regex pattern: {pattern.pattern}"
                logger.warning("Guardrails REGEX BLOCKED: %s", query[:80])
                return GuardrailsResult(allowed=False, reason=reason)

        if self._rails is None:
            return GuardrailsResult(allowed=True, reason="guardrails_disabled")

        try:
            import asyncio

            response: Any = asyncio.run(
                self._rails.generate_async(messages=[{"role": "user", "content": query}])
            )
            content: str = (
                response.get("content", "") if isinstance(response, dict) else str(response)
            )

            blocked_phrases = [
                "cannot process",
                "detected a prompt injection",
                "cannot provide",
                "harmful information",
            ]
            if any(phrase in content.lower() for phrase in blocked_phrases):
                logger.warning("Guardrails BLOCKED query: %s", query[:80])
                return GuardrailsResult(allowed=False, reason=content)

            return GuardrailsResult(allowed=True)
        except Exception as exc:
            logger.warning("Guardrails check error (%s) — allowing query.", exc)
            return GuardrailsResult(allowed=True, reason=f"check_error: {exc}")
