"""Unwrap documented Strands failures without losing AWS evidence."""

from __future__ import annotations

from strands.types.exceptions import (
    ContextWindowOverflowException,
    EventLoopException,
    MaxTokensReachedException,
    ModelThrottledException,
    StructuredOutputException,
)

STRANDS_FAILURES = (
    ContextWindowOverflowException,
    EventLoopException,
    MaxTokensReachedException,
    ModelThrottledException,
    StructuredOutputException,
)


def unwrap_strands_error(exc: Exception) -> Exception:
    """Unwrap known SDK types while preserving unknown boundaries.

    ``MaxTokensReachedException`` stays intact. Peeling it would hide an
    answered model call behind a nested provider error.
    """
    seen: set[int] = set()
    current = exc
    while isinstance(current, STRANDS_FAILURES) and id(current) not in seen:
        if isinstance(current, MaxTokensReachedException):
            return current
        seen.add(id(current))
        nested = getattr(current, "original_exception", current.__cause__)
        if not isinstance(nested, Exception):
            break
        current = nested
    return current
