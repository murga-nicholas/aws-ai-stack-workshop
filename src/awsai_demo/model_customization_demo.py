"""Amazon Bedrock model customization demo.

Technology: Amazon Bedrock model customization.
Lane: models.
Lifecycle refs: model-customization.
Run:
    uv run awsai-demo model-customization
    uv run awsai-demo model-customization --execution live
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from awsai_demo.demo_support import (
    AwsPort,
    build_default_boto_port,
    build_result,
    contract_only,
    fixture_operation,
    port_operation,
    require_port_not_run,
    unsupported_emulator,
    validate_request,
)
from awsai_demo.stubs import stubbed_client

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from awsai_demo.contracts import (
        CredentialSource,
        DemoResult,
        OperationOutcome,
    )
    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.policy import ExecutionPolicy
    from awsai_demo.runtime import Execution, Settings

_DEMO = "model-customization"
_TECHNOLOGY = "Bedrock model customization"
_REFS = ["model-customization"]
_SERVICE = "bedrock"
_ROLE_ARN = "arn:aws:iam::123456789012:role/awsai-demo-model-customization"
_BASE_MODEL = (
    "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-lite-v1:0"
)
_CUSTOM_MODEL = (
    "arn:aws:bedrock:us-east-1:123456789012:custom-model/demo/abcdefghijkl"
)


def run_model_customization_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Run customization contract rows and live read-only listing."""
    credential_source: CredentialSource | None = None
    if port is None:
        default_port = build_default_boto_port(
            execution=execution,
            settings=settings,
            policy=policy,
            session_factory=session_factory,
            csv_loader=csv_loader,
        )
        if default_port is not None:
            port = default_port.port
            credential_source = default_port.credential_source
    if execution == "offline":
        operations, data = _run_offline()
    elif execution == "emulator":
        operations, data = _run_emulator()
    else:
        operations, data = _run_live(settings, port)
    return build_result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="models",
        lifecycle_refs=_REFS,
        execution=execution,
        headline="Customization request contracts are validated safely.",
        operations=operations,
        settings=settings,
        data=data,
        credential_source=credential_source,
    )


def _run_offline() -> tuple[list[OperationOutcome], dict[str, object]]:
    operations: list[OperationOutcome] = []
    with stubbed_client(_SERVICE) as stubber:
        client = stubber.client
        for (
            operation_name,
            method,
            params,
            response,
            fixture_id,
        ) in _fixture_rows():
            stubber.add_response(method, response, expected_params=params)
            getattr(client, method)(**params)
            operations.append(
                fixture_operation(
                    service=_SERVICE,
                    operation_name=operation_name,
                    fixture_id=fixture_id,
                    effect="read"
                    if operation_name == "ListCustomModels"
                    else "none",
                ),
            )
    return operations, _customization_data({"modelSummaries": []})


def _run_emulator() -> tuple[list[OperationOutcome], dict[str, object]]:
    return (
        [
            unsupported_emulator(service=_SERVICE, operation_name=name)
            for name in _OPERATION_NAMES
        ],
        {"emulator": "not_supported"},
    )


def _run_live(
    settings: Settings,
    port: AwsPort | None,
) -> tuple[list[OperationOutcome], dict[str, object]]:
    operations = [
        _validated_contract_only(
            name,
            params=params,
            fixture_id=fixture_id,
        )
        for name, params, fixture_id in _contract_rows()
    ]
    payload: Mapping[str, Any] = {"modelSummaries": []}
    if port is None:
        operations.append(
            require_port_not_run(
                service=_SERVICE,
                operation_name="ListCustomModels",
            ),
        )
    else:
        called = port_operation(
            port=port,
            service=_SERVICE,
            operation_name="ListCustomModels",
            params=list_custom_models_params(),
            execution="live",
            effect="read",
            fixture_id=_fixture_id("ListCustomModels"),
        )
        operations.append(called.outcome)
        payload = called.payload
    data = _customization_data(payload)
    data["region"] = settings.region
    return operations, data


def customization_job_params(customization_type: str) -> dict[str, Any]:
    """Return a CreateModelCustomizationJob request."""
    suffix = customization_type.lower().replace("_", "-")
    return {
        "jobName": f"awsai-demo-{suffix}",
        "customModelName": f"awsai-demo-{suffix}-custom",
        "roleArn": _ROLE_ARN,
        "baseModelIdentifier": _BASE_MODEL,
        "customizationType": customization_type,
        "trainingDataConfig": {
            "s3Uri": f"s3://example-input/{suffix}.jsonl",
        },
        "outputDataConfig": {"s3Uri": f"s3://example-output/{suffix}/"},
    }


def model_import_params() -> dict[str, Any]:
    """Return a CreateModelImportJob request."""
    return {
        "jobName": "awsai-demo-import",
        "importedModelName": "awsai-demo-imported",
        "roleArn": _ROLE_ARN,
        "modelDataSource": {
            "s3DataSource": {"s3Uri": "s3://example-model/source/"},
        },
    }


def custom_model_deployment_params() -> dict[str, str]:
    """Return a CreateCustomModelDeployment request."""
    return {
        "modelDeploymentName": "awsai-demo-deployment",
        "modelArn": _CUSTOM_MODEL,
    }


def list_custom_models_params() -> dict[str, int]:
    """Return bounded ListCustomModels params."""
    return {"maxResults": 10}


def _validated_contract_only(
    name: str,
    *,
    params: Mapping[str, Any],
    fixture_id: str,
) -> OperationOutcome:
    validate_request(
        service=_SERVICE,
        operation_name=name,
        params=params,
        fixture_id=fixture_id,
    )
    return contract_only(service=_SERVICE, operation_name=name)


def _fixture_rows() -> tuple[
    tuple[str, str, dict[str, Any], dict[str, Any], str], ...
]:
    return (
        *(
            (
                "CreateModelCustomizationJob",
                "create_model_customization_job",
                customization_job_params(customization_type),
                {
                    "jobArn": (
                        "arn:aws:bedrock:us-east-1:123456789012:"
                        f"model-customization-job/{customization_type.lower()}"
                    ),
                },
                _fixture_id(
                    f"CreateModelCustomizationJob-{customization_type}"
                ),
            )
            for customization_type in _CUSTOMIZATION_TYPES
        ),
        (
            "CreateModelImportJob",
            "create_model_import_job",
            model_import_params(),
            {
                "jobArn": (
                    "arn:aws:bedrock:us-east-1:123456789012:"
                    "model-import-job/import"
                ),
            },
            _fixture_id("CreateModelImportJob"),
        ),
        (
            "CreateCustomModelDeployment",
            "create_custom_model_deployment",
            custom_model_deployment_params(),
            {
                "customModelDeploymentArn": (
                    "arn:aws:bedrock:us-east-1:123456789012:"
                    "custom-model-deployment/demo"
                ),
            },
            _fixture_id("CreateCustomModelDeployment"),
        ),
        (
            "ListCustomModels",
            "list_custom_models",
            list_custom_models_params(),
            {"modelSummaries": []},
            _fixture_id("ListCustomModels"),
        ),
    )


def _contract_rows() -> tuple[tuple[str, dict[str, Any], str], ...]:
    return (
        *(
            (
                "CreateModelCustomizationJob",
                customization_job_params(customization_type),
                _fixture_id(
                    f"CreateModelCustomizationJob-{customization_type}"
                ),
            )
            for customization_type in _CUSTOMIZATION_TYPES
        ),
        (
            "CreateModelImportJob",
            model_import_params(),
            _fixture_id("CreateModelImportJob"),
        ),
        (
            "CreateCustomModelDeployment",
            custom_model_deployment_params(),
            _fixture_id("CreateCustomModelDeployment"),
        ),
    )


def _customization_data(payload: Mapping[str, Any]) -> dict[str, object]:
    models = cast("Sequence[object]", payload.get("modelSummaries", ()))
    return {
        "customization_types": list(_CUSTOMIZATION_TYPES),
        "contract_job_count": len(_CUSTOMIZATION_TYPES),
        "custom_model_count": len(models),
    }


def _fixture_id(operation_name: str) -> str:
    return f"model-customization-{operation_name}-v1"


_CUSTOMIZATION_TYPES = (
    "FINE_TUNING",
    "REINFORCEMENT_FINE_TUNING",
    "DISTILLATION",
)
_OPERATION_NAMES = (
    "CreateModelCustomizationJob",
    "CreateModelCustomizationJob",
    "CreateModelCustomizationJob",
    "CreateModelImportJob",
    "CreateCustomModelDeployment",
    "ListCustomModels",
)
