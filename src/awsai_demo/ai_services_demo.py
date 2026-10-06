"""Small, bounded examples across seven AWS AI services.

Technology: Comprehend, Translate, Polly, Transcribe, Textract,
    Rekognition and Bedrock Data Automation.
Lane: data.
Lifecycle refs: comprehend, translate, polly, transcribe, textract,
    rekognition-maintenance-features, bedrock-data-automation.
Run: uv run awsai-demo ai-services
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from awsai_demo.billing import active_budget_run, use_budget_run
from awsai_demo.contracts import result
from awsai_demo.demo_support import build_default_boto_port, build_result
from awsai_demo.emulator_media import DOCUMENT_PNG, prepare_audio
from awsai_demo.policy import Charge, PolicyError
from awsai_demo.service_rows import ServiceRow

if TYPE_CHECKING:
    from awsai_demo.contracts import (
        CredentialSource,
        DemoResult,
        OperationOutcome,
    )
    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.demo_support import AwsPort
    from awsai_demo.policy import ExecutionPolicy
    from awsai_demo.runtime import Execution, Settings

_TEXT = (
    "The support pilot costs 19680 USD for six weeks. A human must "
    "approve it before anyone accepts. Send the draft to "
    "review@example.com. Customer information stays private. "
    "No approval has been granted."
).ljust(200)
_REFS = {
    "comprehend": "comprehend",
    "translate": "translate",
    "polly": "polly",
    "transcribe": "transcribe",
    "textract": "textract",
    "rekognition": "rekognition-maintenance-features",
    "data-automation": "bedrock-data-automation",
}


def service_rows() -> dict[str, tuple[ServiceRow, ...]]:
    """Build fresh fixtures, including an unread audio stream."""
    text = {"Text": _TEXT, "LanguageCode": "en"}
    job = "awsai-transcription-contract"
    return {
        "comprehend": (
            ServiceRow(
                "comprehend",
                "DetectPiiEntities",
                text,
                {
                    "Entities": [
                        {
                            "Score": 0.99,
                            "Type": "EMAIL",
                            "BeginOffset": 111,
                            "EndOffset": 129,
                        }
                    ]
                },
                effect="infer",
                charges=(
                    Charge(
                        "comprehend.detect_pii_entities.units",
                        Decimal(3),
                        "unit",
                    ),
                ),
            ),
            ServiceRow(
                "comprehend",
                "DetectSentiment",
                text,
                {
                    "Sentiment": "NEUTRAL",
                    "SentimentScore": {
                        "Positive": 0.01,
                        "Negative": 0.01,
                        "Neutral": 0.98,
                        "Mixed": 0.0,
                    },
                },
                effect="infer",
                charges=(
                    Charge(
                        "comprehend.detect_sentiment.units", Decimal(3), "unit"
                    ),
                ),
            ),
        ),
        "translate": (
            ServiceRow(
                "translate",
                "TranslateText",
                {
                    "Text": _TEXT,
                    "SourceLanguageCode": "en",
                    "TargetLanguageCode": "es",
                },
                {
                    "TranslatedText": "El piloto requiere aprobacion humana.",
                    "SourceLanguageCode": "en",
                    "TargetLanguageCode": "es",
                },
                effect="infer",
                charges=(
                    Charge(
                        "translate.characters",
                        Decimal(len(_TEXT)),
                        "character",
                    ),
                ),
            ),
        ),
        "polly": (
            ServiceRow(
                "polly",
                "SynthesizeSpeech",
                {
                    "Text": _TEXT,
                    "OutputFormat": "pcm",
                    "VoiceId": "Joanna",
                    "Engine": "standard",
                },
                {
                    "AudioStream": BytesIO(b"\x00\x00" * 8),
                    "ContentType": "audio/pcm",
                    "RequestCharacters": len(_TEXT),
                },
                effect="infer",
                charges=(
                    Charge(
                        "polly.characters", Decimal(len(_TEXT)), "character"
                    ),
                ),
            ),
        ),
        "transcribe": (
            ServiceRow(
                "transcribe",
                "StartTranscriptionJob",
                {
                    "TranscriptionJobName": job,
                    "LanguageCode": "en-US",
                    "MediaFormat": "wav",
                    "Media": {
                        "MediaFileUri": "s3://awsai-demo-audio/pilot.wav"
                    },
                },
                {
                    "TranscriptionJob": {
                        "TranscriptionJobName": job,
                        "TranscriptionJobStatus": "IN_PROGRESS",
                    }
                },
                effect="none",
                emulator=True,
                live=False,
            ),
            ServiceRow(
                "transcribe",
                "GetTranscriptionJob",
                {"TranscriptionJobName": job},
                {
                    "TranscriptionJob": {
                        "TranscriptionJobName": job,
                        "TranscriptionJobStatus": "COMPLETED",
                    }
                },
                effect="none",
                emulator=True,
                live=False,
            ),
        ),
        "textract": (
            ServiceRow(
                "textract",
                "DetectDocumentText",
                {"Document": {"Bytes": DOCUMENT_PNG}},
                {
                    "Blocks": [
                        {
                            "BlockType": "LINE",
                            "Text": "Pilot budget: 25000 USD",
                            "Confidence": 99.0,
                        }
                    ]
                },
                effect="none",
                emulator=True,
                live=False,
            ),
        ),
        "rekognition": (
            ServiceRow(
                "rekognition",
                "DetectLabels",
                {"Image": {"Bytes": DOCUMENT_PNG}, "MaxLabels": 2},
                {"Labels": [{"Name": "Document", "Confidence": 98.0}]},
                effect="none",
                live=False,
            ),
        ),
        "data-automation": (
            ServiceRow(
                "bedrock-data-automation-runtime",
                "InvokeDataAutomationAsync",
                {
                    "inputConfiguration": {
                        "s3Uri": "s3://awsai-contract/input.pdf"
                    },
                    "outputConfiguration": {
                        "s3Uri": "s3://awsai-contract/output/"
                    },
                    "dataAutomationProfileArn": (
                        "arn:aws:bedrock:us-east-1:123456789012:"
                        "data-automation-profile/us.data-automation-v1"
                    ),
                },
                {
                    "invocationArn": (
                        "arn:aws:bedrock:us-east-1:123456789012:"
                        "data-automation-invocation/"
                        "01234567-0123-4123-8123-012345678901"
                    )
                },
                effect="none",
                live=False,
            ),
        ),
    }


def _children(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None,
    source: CredentialSource | None,
) -> list[DemoResult]:
    children = []
    for name, original_rows in service_rows().items():
        rows = original_rows
        setup: list[OperationOutcome] = []
        row_port = port
        if (
            name == "transcribe"
            and execution == "emulator"
            and port is not None
        ):
            setup, uri = prepare_audio(
                port,
                endpoint_url=settings.localstack_endpoint,
                region=settings.region,
            )
            if uri is None:
                row_port = None
            else:
                job = "awsai-ai-services-" + uuid4().hex[:12]
                rows = (
                    replace(
                        rows[0],
                        params={
                            **rows[0].params,
                            "TranscriptionJobName": job,
                            "Media": {"MediaFileUri": uri},
                        },
                    ),
                    replace(rows[1], params={"TranscriptionJobName": job}),
                )
        responses = [
            row.run(
                execution=execution,
                settings=settings,
                policy=policy,
                port=row_port,
                command="ai-services",
            )
            for row in rows
        ]
        payload = responses[0].payload
        data: dict[str, Any] = {"text_characters": len(_TEXT)}
        if name == "polly":
            stream = payload.get("AudioStream")
            if stream is not None:
                try:
                    data["audio_bytes"] = len(stream.read())
                finally:
                    stream.close()
        elif name == "comprehend":
            data["pii_types"] = [
                entity["Type"] for entity in payload.get("Entities", [])
            ]
            data["sentiment"] = responses[1].payload.get("Sentiment")
        elif name == "translate":
            data["translated_text"] = payload.get("TranslatedText")
        else:
            data["observed"] = [item.outcome["status"] for item in responses]
        children.append(
            build_result(
                demo=f"ai-services/{name}",
                technology=name,
                lane="data",
                lifecycle_refs=[_REFS[name]],
                execution=execution,
                headline=f"{name}: each operation has its own provenance.",
                operations=[*setup, *(item.outcome for item in responses)],
                settings=settings,
                data=data,
                credential_source=source,
            )
        )
    return children


def run_ai_services_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Run all service children under one shared command budget."""
    default = (
        None
        if port is not None
        else build_default_boto_port(
            execution=execution,
            settings=settings,
            policy=policy,
            session_factory=session_factory,
            csv_loader=csv_loader,
        )
    )
    port = port if default is None else default.port
    make_children = partial(
        _children,
        execution=execution,
        settings=settings,
        policy=policy,
        port=port,
        source=None if default is None else default.credential_source,
    )
    if execution == "live":
        try:
            run = active_budget_run(region=settings.region, policy=policy)
        except PolicyError:
            children = make_children()
        else:
            with use_budget_run(run):
                children = make_children()
    else:
        children = make_children()
    return result(
        demo="ai-services",
        technology="AWS AI services",
        lane="data",
        lifecycle_refs=list(_REFS.values()),
        requested_execution=execution,
        headline="Seven services, with operation-level lane evidence.",
        children=children,
    )
