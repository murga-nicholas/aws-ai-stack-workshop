"""SageMaker AI endpoint contract and read-only resource inventory.

Technology: Amazon SageMaker AI.
Lane: platform.
Lifecycle refs: sagemaker-ai, sagemaker-ai-maintenance-features,
    sagemaker-profiler.
Run: uv run awsai-demo sagemaker-ai
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from awsai_demo.demo_support import build_default_boto_port, build_result
from awsai_demo.service_rows import ServiceRow

if TYPE_CHECKING:
    from awsai_demo.contracts import DemoResult
    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.demo_support import AwsPort
    from awsai_demo.policy import ExecutionPolicy
    from awsai_demo.runtime import Execution, Settings


def run_sagemaker_ai_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Validate inference shape without invoking a deployed endpoint."""
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
    rows = (
        ServiceRow(
            "sagemaker-runtime",
            "InvokeEndpoint",
            {
                "EndpointName": "awsai-contract-endpoint",
                "Body": b'{"prompt":"support pilot"}',
                "ContentType": "application/json",
            },
            {"Body": b'{"accepted":false}', "ContentType": "application/json"},
            effect="none",
            live=False,
        ),
        ServiceRow(
            "sagemaker",
            "ListEndpoints",
            {"MaxResults": 10},
            {"Endpoints": []},
            emulator=True,
        ),
        ServiceRow(
            "sagemaker",
            "ListModels",
            {"MaxResults": 10},
            {"Models": []},
            emulator=True,
        ),
    )
    responses = [
        row.run(
            execution=execution,
            settings=settings,
            policy=policy,
            port=port,
            command="sagemaker-ai",
        )
        for row in rows
    ]
    return build_result(
        demo="sagemaker-ai",
        technology="Amazon SageMaker AI",
        lane="platform",
        lifecycle_refs=[
            "sagemaker-ai",
            "sagemaker-ai-maintenance-features",
            "sagemaker-profiler",
        ],
        execution=execution,
        headline="List endpoints and models; invocation stays a contract.",
        operations=[item.outcome for item in responses],
        settings=settings,
        data={
            "endpoints_listed": len(responses[1].payload.get("Endpoints", [])),
            "models_listed": len(responses[2].payload.get("Models", [])),
            "invoke_endpoint_contract_only": True,
        },
        credential_source=None
        if default is None
        else default.credential_source,
    )
