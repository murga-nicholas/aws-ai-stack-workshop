from __future__ import annotations

import struct
import wave
import zlib
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import pytest
from botocore.exceptions import ClientError

from awsai_demo import localstack_ai_demo as demo
from awsai_demo.demo_support import DefaultBotoPort
from awsai_demo.emulator_media import DOCUMENT_PNG, prepare_audio, silent_wav
from awsai_demo.localstack_demo import LocalStackUnavailableError
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping


class Port:
    def __init__(self, failure: str | None = None) -> None:
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []
        self.failure = failure

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append((service, operation_name, params))
        if self.failure == "audio" and service == "s3":
            message = "emulator down"
            raise OSError(message)
        if self.failure == "down" and service != "s3":
            message = "service down"
            raise OSError(message)
        if (
            self.failure
            in ("LicenseError", "EntitlementMissing", "AccessDeniedException")
            and service == "bedrock"
        ):
            raise ClientError(
                {
                    "Error": {"Code": self.failure, "Message": "denied"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                operation_name,
            )
        if (
            self.failure
            in ("requires the ultimate plan", "not available in your plan")
            and service == "textract"
        ):
            raise ClientError(
                {
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": self.failure,
                    },
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                operation_name,
            )
        return {}


def info(url: str, timeout: float) -> Mapping[str, Any]:
    assert url.startswith("http://localhost:4566/_localstack/")
    assert timeout > 0
    return {
        "version": "2026.09.0",
        "image": "localstack:2026.09.0",
        "digest": "sha256:fixture",
        "plan": "Ultimate",
        "secret": "never publish",
        "other": "ignored",
    }


def test_advertised_is_not_an_emulator_measurement() -> None:
    result = demo.run_localstack_ai_demo()
    assert result["mode"] == "local_execution"
    rows = cast("dict[str, Any]", result["data"])["advertised"]
    assert len(rows) == 5
    assert not any(row["attempted"] or row["passed"] for row in rows)
    assert (
        next(row for row in rows if row["service"] == "Transcribe")["plan"]
        == "Hobby"
    )
    assert demo.run_localstack_ai_demo(execution="live")["status"] == "blocked"
    missing = demo.run_localstack_ai_demo(execution="emulator")
    assert missing["operations"][0]["error_code"] == "missing_configuration"


def test_emulator_attempts_valid_requests_and_records_metadata() -> None:
    port = Port()
    result = demo.run_localstack_ai_demo(
        execution="emulator",
        settings=Settings(localstack_auth_token="-".join(("test", "fixture"))),
        port=port,
        http_get=info,
    )
    assert result["mode"] == "local_emulator"
    assert result["status"] == "ok"
    data = cast("dict[str, Any]", result["data"])
    assert len(data["observed"]) == 6
    assert all(row["attempted"] and row["passed"] for row in data["observed"])
    assert data["emulator"]["version"] == "2026.09.0"
    assert "secret" not in data["emulator"]
    requests = {name: params for _, name, params in port.calls}
    assert requests["Converse"]["modelId"].startswith("ollama.")
    assert requests["StartTranscriptionJob"]["Media"]["MediaFileUri"].endswith(
        "silence.wav"
    )
    assert requests["PutObject"]["Body"] == silent_wav()


@pytest.mark.parametrize(
    "message,expected",
    [
        ("LicenseError", "entitlement_missing"),
        ("EntitlementMissing", "entitlement_missing"),
        ("AccessDeniedException", "authorization_denied"),
        ("requires the ultimate plan", "entitlement_missing"),
        ("not available in your plan", "entitlement_missing"),
        ("down", "emulator_unavailable"),
    ],
)
def test_observed_failures_are_not_guessed_plan_failures(
    message: str,
    expected: str,
) -> None:
    result = demo.run_localstack_ai_demo(
        execution="emulator",
        settings=Settings(localstack_auth_token="-".join(("test", "fixture"))),
        port=Port(message),
        http_get=info,
    )
    data = cast("dict[str, Any]", result["data"])
    assert expected in {row["error_code"] for row in data["observed"]}


def test_audio_failure_skips_transcription_only() -> None:
    port = Port("audio")
    result = demo.run_localstack_ai_demo(
        execution="emulator",
        settings=Settings(localstack_auth_token="-".join(("test", "fixture"))),
        port=port,
        http_get=info,
    )
    assert result["status"] == "blocked"
    assert "StartTranscriptionJob" not in [name for _, name, _ in port.calls]


@pytest.mark.parametrize(
    "exc", [OSError("down"), LocalStackUnavailableError("down")]
)
def test_probe_failure_prevents_service_dispatch(exc: Exception) -> None:
    def down(url: str, timeout: float) -> Mapping[str, Any]:
        del url, timeout
        raise exc

    port = Port()
    result = demo.run_localstack_ai_demo(
        execution="emulator",
        settings=Settings(localstack_auth_token="-".join(("test", "fixture"))),
        port=port,
        http_get=down,
    )
    assert result["mode"] == "attempt_failed"
    assert result["operations"][0]["error_code"] == "emulator_unavailable"
    assert not port.calls


def test_default_port_and_missing_optional_metadata() -> None:
    port = Port()
    result = demo.run_localstack_ai_demo(
        execution="emulator",
        settings=Settings(localstack_auth_token="-".join(("test", "fixture"))),
        policy=ExecutionPolicy(),
        http_get=lambda *_: {},
        boto_port_factory=lambda **_: DefaultBotoPort(
            cast("Any", port), "none"
        ),
    )
    assert result["status"] == "ok"
    assert cast("dict[str, Any]", result["data"])["emulator"] == {}
    assert port.calls


def test_audio_is_valid_two_second_pcm_and_region_correct() -> None:
    with wave.open(BytesIO(silent_wav())) as audio:
        assert audio.getnframes() / audio.getframerate() == 2
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
    port = Port()
    operations, uri = prepare_audio(
        port, endpoint_url="http://localhost:4566", region="eu-west-1"
    )
    assert len(operations) == 2
    assert uri is not None
    assert port.calls[0][2]["CreateBucketConfiguration"] == {
        "LocationConstraint": "eu-west-1"
    }


def test_document_png_has_valid_chunk_checksums() -> None:
    offset = 8
    assert DOCUMENT_PNG[:8] == b"\x89PNG\r\n\x1a\n"
    while offset < len(DOCUMENT_PNG):
        size = struct.unpack(">I", DOCUMENT_PNG[offset : offset + 4])[0]
        chunk = DOCUMENT_PNG[offset + 4 : offset + size + 8]
        checksum = struct.unpack(
            ">I", DOCUMENT_PNG[offset + size + 8 : offset + size + 12]
        )[0]
        assert zlib.crc32(chunk) == checksum
        offset += size + 12
