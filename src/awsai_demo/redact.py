"""Redaction helpers for public workshop output."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_ACCOUNT_ID_RE = re.compile(r"\b\d{12}\b")
_ACCESS_KEY_ID_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b(Authorization\s*[:=]\s*)[^\s,;]+(?:\s+[^\s,;]+)?",
)
_SESSION_TOKEN_RE = re.compile(
    r"(?i)\b((?:x-amz-security-token|session[_-]?token)\s*[:=]\s*)"
    r"[^\s,;]+",
)
_ARN_RE = re.compile(r"\barn:[^\s'\",)>\]]+")
_BASE64ISH_RE = re.compile(
    r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])",
)
_SECRET_KEY_PARTS = (
    "authorization",
    "x-amz-security-token",
    "session_token",
    "session-token",
    "secret",
    "password",
    "bearer",
    "api_key",
    "api-key",
    "access_key",
    "access-key",
)


@contextmanager
def quiet_sdk_logging() -> Iterator[None]:
    """Keep raw SDK diagnostics out of public console output.

    Structured operation outcomes carry the public diagnostics. Cost
    notices use stderr directly and remain visible before dispatch.
    """
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def redact(value: object) -> object:
    """Return *value* with public-output secrets recursively masked."""
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, Mapping):
        return {
            key: "<redacted>" if _is_sensitive_key(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, set):
        return {redact(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(redact(item) for item in value)
    return value


def sanitize_exception(exc: BaseException) -> str:
    """Return a one-line redacted exception diagnostic."""
    message = " ".join(str(exc).split())
    redacted = redact(message)
    return f"{exc.__class__.__name__}: {redacted}"


def _is_sensitive_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalised = key.strip().lower()
    return any(part in normalised for part in _SECRET_KEY_PARTS)


def _redact_string(value: str) -> str:
    redacted = _ARN_RE.sub(_redact_arn_match, value)
    redacted = _AUTH_HEADER_RE.sub(r"\1<redacted>", redacted)
    redacted = _BEARER_RE.sub("Bearer <redacted>", redacted)
    redacted = _SESSION_TOKEN_RE.sub(r"\1<redacted>", redacted)
    redacted = _ACCESS_KEY_ID_RE.sub("<access-key-id>", redacted)
    redacted = _ACCOUNT_ID_RE.sub("<account>", redacted)
    return _BASE64ISH_RE.sub(_redact_base64ish_match, redacted)


def _redact_arn_match(match: re.Match[str]) -> str:
    token = match.group(0)
    parts = token.split(":", 5)
    if len(parts) != 6:
        return token
    parts[4] = "<account>" if parts[4] else ""
    parts[5] = "<redacted>"
    return ":".join(parts)


def _redact_base64ish_match(match: re.Match[str]) -> str:
    token = match.group(0)
    if _looks_like_non_secret_hash(token):
        return token
    return "<secret>"


def _looks_like_non_secret_hash(token: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{40}", token))
