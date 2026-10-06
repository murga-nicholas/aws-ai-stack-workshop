from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from awsai_demo.redact import quiet_sdk_logging, redact, sanitize_exception

if TYPE_CHECKING:
    from pytest import LogCaptureFixture


def test_sdk_logging_is_suppressed_and_restored_on_failure(
    caplog: LogCaptureFixture,
) -> None:
    previous = logging.root.manager.disable
    logger = logging.getLogger("strands.multiagent.graph")
    with pytest.raises(ValueError, match="fixture"), quiet_sdk_logging():
        with quiet_sdk_logging():
            logger.error("private account %s", "123456789012")
        assert logging.root.manager.disable == logging.CRITICAL
        message = "fixture"
        raise ValueError(message)
    assert not caplog.records
    assert logging.root.manager.disable == previous


def test_redact_masks_string_secrets_and_preserves_hashes() -> None:
    base64ish = "A" + ("b" * 38) + "/"
    sha1 = "0123456789abcdef0123456789abcdef01234567"
    text = (
        "arn:aws:bedrock-agentcore:us-east-1:123456789012:harness/demo "
        "arn:bad 123456789012 AKIAABCDEFGHIJKLMNOP "
        f"Bearer token-value Authorization: Basic abc {base64ish} {sha1} "
        "session_token=hidden x-amz-security-token:alsohidden"
    )

    redacted = redact(text)

    assert redacted == (
        "arn:aws:bedrock-agentcore:us-east-1:<account>:<redacted> "
        "arn:bad <account> <access-key-id> "
        f"Bearer <redacted> Authorization: <redacted> <secret> {sha1} "
        "session_token=<redacted> x-amz-security-token:<redacted>"
    )


def test_redact_recurses_through_containers_and_sensitive_keys() -> None:
    value = {
        "Authorization": "Bearer hidden",
        "nested": ["123456789012", ("ASIAABCDEFGHIJKLMNOP",)],
        "set": {"keep", "123456789012"},
        "frozen": frozenset({"123456789012"}),
        1: "AKIAABCDEFGHIJKLMNOP",
    }

    redacted = redact(value)

    assert redacted == {
        "Authorization": "<redacted>",
        "nested": ["<account>", ("<access-key-id>",)],
        "set": {"keep", "<account>"},
        "frozen": frozenset({"<account>"}),
        1: "<access-key-id>",
    }
    assert redact(3) == 3


def test_sanitize_exception_is_one_line_and_redacted() -> None:
    exc = RuntimeError(
        "first line\naccount 123456789012 Authorization: Bearer token",
    )

    assert sanitize_exception(exc) == (
        "RuntimeError: first line account <account> Authorization: <redacted>"
    )
