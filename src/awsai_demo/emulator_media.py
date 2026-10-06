"""Small deterministic media fixtures for emulator-only requests."""

from __future__ import annotations

import base64
import wave
from io import BytesIO
from typing import TYPE_CHECKING

from awsai_demo.demo_support import port_operation

if TYPE_CHECKING:
    from awsai_demo.contracts import OperationOutcome
    from awsai_demo.demo_support import AwsPort

DOCUMENT_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
)
AUDIO_BUCKET = "awsai-workshop-transcribe-fixture"


def silent_wav() -> bytes:
    """Return two seconds of 16 kHz mono PCM silence."""
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\0\0" * 32_000)
    return output.getvalue()


def prepare_audio(
    port: AwsPort,
    *,
    endpoint_url: str,
    region: str,
) -> tuple[list[OperationOutcome], str | None]:
    """Place the WAV in emulated S3 and record setup evidence."""
    create: dict[str, object] = {"Bucket": AUDIO_BUCKET}
    if region != "us-east-1":
        create["CreateBucketConfiguration"] = {"LocationConstraint": region}
    operations: list[OperationOutcome] = []
    for name, params in (
        ("CreateBucket", create),
        (
            "PutObject",
            {
                "Bucket": AUDIO_BUCKET,
                "Key": "silence.wav",
                "Body": silent_wav(),
                "ContentType": "audio/wav",
            },
        ),
    ):
        call = port_operation(
            port=port,
            service="s3",
            operation_name=name,
            params=params,
            execution="emulator",
            effect="write",
            phase="setup",
            fixture_id="emulator-audio-v1",
            endpoint_url=endpoint_url,
        )
        operations.append(call.outcome)
        if call.outcome["status"] != "ok":
            return operations, None
    return operations, f"s3://{AUDIO_BUCKET}/silence.wav"
