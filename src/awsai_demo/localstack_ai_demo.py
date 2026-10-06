"""Advertised versus observed LocalStack AI operation coverage.

Lane: local; lifecycle: localstack-bedrock, localstack-ai-services.
Run: uv run awsai-demo localstack-ai.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from awsai_demo.contracts import operation
from awsai_demo.demo_support import (
    build_result,
    contract_only,
    local_operation,
    port_operation,
    require_port_not_run,
    resolve_boto_port,
)
from awsai_demo.emulator_media import DOCUMENT_PNG, prepare_audio
from awsai_demo.lineage import load_lineage
from awsai_demo.localstack_demo import (
    LocalStackHttpBotoPort,
    LocalStackUnavailableError,
)
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from awsai_demo.contracts import (
        DemoResult,
        Effect,
        Execution,
        OperationOutcome,
    )
    from awsai_demo.demo_support import (
        AwsPort,
        BotoPortFactory,
        DefaultBotoPort,
    )
    from awsai_demo.localstack_demo import HttpJsonGet


def probe_metadata(
    settings: Settings,
    http_get: HttpJsonGet | None,
) -> tuple[list[OperationOutcome], dict[str, str]]:
    """Read emulator identity without exposing raw health payloads."""
    operations: list[OperationOutcome] = []
    info: dict[str, str] = {}
    probe = LocalStackHttpBotoPort(settings, http_get=http_get)
    for name, call in (("health", probe.health), ("info", probe.info)):
        try:
            response = call()
        except (OSError, LocalStackUnavailableError):
            operations.append(
                operation(
                    service="localstack",
                    operation=name,
                    phase="setup",
                    execution_target="emulator",
                    mode="attempt_failed",
                    status="blocked",
                    effect="read",
                    transport="loopback",
                    endpoint_url=settings.localstack_endpoint,
                    response_received=False,
                    error_code="emulator_unavailable",
                )
            )
            break
        operations.append(
            operation(
                service="localstack",
                operation=name,
                phase="setup",
                execution_target="emulator",
                mode="local_emulator",
                effect="read",
                transport="loopback",
                endpoint_url=settings.localstack_endpoint,
                response_received=True,
            )
        )
        for key in ("version", "image", "digest", "plan"):
            if isinstance(response.get(key), str):
                info[key] = response[key]
    return operations, info


def advertised_coverage() -> list[dict[str, Any]]:
    """Join operation examples to sourced lineage rows."""
    records = {record.id: record for record in load_lineage()}
    bedrock = records["localstack-bedrock"]
    services = records["localstack-ai-services"]
    return [
        {
            "service": service,
            "plan": plan,
            "operations": operations,
            "advertised": True,
            "attempted": False,
            "passed": False,
            "error_code": None,
            "sources": record.sources,
            "advertised_scope": record.scope,
        }
        for service, plan, operations, record in (
            (
                "Bedrock",
                "Ultimate",
                ["ListFoundationModels", "Converse", "InvokeModel"],
                bedrock,
            ),
            ("Transcribe", "Hobby", ["StartTranscriptionJob"], services),
            ("Textract", "Ultimate", ["DetectDocumentText"], services),
            ("SageMaker", "Ultimate", ["ListEndpoints"], services),
            ("OpenSearch", "Hobby", [], services),
        )
    ]


def _requests(
    settings: Settings,
) -> list[tuple[str, str, dict[str, Any], Effect]]:
    model = "ollama." + settings.default_bedrock_model.removeprefix("ollama.")
    prompt = "Reply with a short greeting."
    return [
        ("bedrock", "ListFoundationModels", {}, "read"),
        (
            "bedrock-runtime",
            "Converse",
            {
                "modelId": model,
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": 64},
            },
            "infer",
        ),
        (
            "bedrock-runtime",
            "InvokeModel",
            {
                "modelId": model,
                "contentType": "application/json",
                "body": json.dumps({"prompt": prompt, "max_tokens": 64}),
            },
            "infer",
        ),
        (
            "textract",
            "DetectDocumentText",
            {
                "Document": {"Bytes": DOCUMENT_PNG},
            },
            "infer",
        ),
        ("sagemaker", "ListEndpoints", {"MaxResults": 10}, "read"),
    ]


def run_localstack_ai_demo(
    *,
    execution: Execution = "offline",
    settings: Settings | None = None,
    policy: ExecutionPolicy | None = None,
    port: AwsPort | None = None,
    http_get: HttpJsonGet | None = None,
    boto_port_factory: BotoPortFactory | None = None,
) -> DemoResult:
    """Measure attempted emulator requests; offline shows claims."""
    settings = settings or Settings()
    advertised = advertised_coverage()
    observed: list[dict[str, Any]] = []
    metadata: dict[str, str] = {}
    operations = []
    if execution == "offline":
        operations.append(
            local_operation(
                service="localstack",
                operation_name="advertised coverage table",
            )
        )
    elif execution == "live":
        operations.append(
            contract_only(
                service="localstack",
                operation_name="AI service coverage",
            )
        )
    elif settings.localstack_auth_token is None:
        operations.append(
            require_port_not_run(
                service="localstack",
                operation_name="AI service coverage",
            )
        )
    else:
        operations, metadata = probe_metadata(settings, http_get)
        if operations[-1]["status"] != "ok":
            return build_result(
                demo="localstack-ai",
                technology="LocalStack AI services",
                lane="local",
                lifecycle_refs=(
                    "localstack-bedrock",
                    "localstack-ai-services",
                ),
                execution=execution,
                settings=settings,
                operations=operations,
                headline="The LocalStack endpoint did not answer.",
                data={"advertised": advertised, "observed": observed},
            )
        if port is None:
            default = resolve_boto_port(
                execution=execution,
                settings=settings,
                policy=policy or ExecutionPolicy(),
                factory=boto_port_factory,
            )
            # The token check above ensures a port is returned.
            default = cast("DefaultBotoPort", default)
            port = default.port
        requests = _requests(settings)
        setup, uri = prepare_audio(
            port,
            endpoint_url=settings.localstack_endpoint,
            region=settings.region,
        )
        operations.extend(setup)
        if uri is not None:
            requests.append(
                (
                    "transcribe",
                    "StartTranscriptionJob",
                    {
                        "TranscriptionJobName": "awsai-silence-"
                        + uuid4().hex[:12],
                        "LanguageCode": "en-US",
                        "MediaFormat": "wav",
                        "Media": {"MediaFileUri": uri},
                    },
                    "write",
                )
            )
        for service, name, params, effect in requests:
            call = port_operation(
                port=port,
                service=service,
                operation_name=name,
                params=params,
                execution=execution,
                effect=effect,
                fixture_id="localstack-ai-request-v1",
                endpoint_url=settings.localstack_endpoint,
            )
            operations.append(call.outcome)
            observed.append(
                {
                    "service": service,
                    "operation": name,
                    "advertised": True,
                    "attempted": True,
                    "passed": call.outcome["status"] == "ok",
                    "error_code": call.outcome["error_code"],
                    "response_class": "dict"
                    if call.outcome["response_received"]
                    else None,
                }
            )
    return build_result(
        demo="localstack-ai",
        technology="LocalStack AI services",
        lane="local",
        lifecycle_refs=("localstack-bedrock", "localstack-ai-services"),
        execution=execution,
        settings=settings,
        operations=operations,
        headline="Advertised coverage and measured requests are separate.",
        data={
            "advertised": advertised,
            "observed": observed,
            "emulator": metadata,
            "audio_fixture": "two seconds of PCM silence",
        },
    )
