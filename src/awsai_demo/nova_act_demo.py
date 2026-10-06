"""Amazon Nova Act service contract demo.

Technology: Amazon Nova Act.
Lane: models.
Lifecycle refs: nova-act.
Run:
    uv run awsai-demo nova-act
    uv run awsai-demo nova-act --execution live
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

_DEMO = "nova-act"
_TECHNOLOGY = "Amazon Nova Act"
_REFS = ["nova-act"]
_SERVICE = "nova-act"
_MODEL_ID = "amazon.nova-act-v1:0"
_WORKFLOW_NAME = "awsaiDemoWorkflow"


def run_nova_act_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Run Nova Act model/workflow contract rows."""
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
        headline="Nova Act list and workflow request shapes are validated.",
        operations=operations,
        settings=settings,
        data=data,
        credential_source=credential_source,
    )


def _run_offline() -> tuple[list[OperationOutcome], dict[str, object]]:
    with stubbed_client(_SERVICE) as stubber:
        client = stubber.client
        stubber.add_response(
            "list_models",
            _list_models_response(),
            expected_params=list_models_params(),
        )
        list_response = client.list_models(**list_models_params())
        stubber.add_response(
            "create_workflow_definition",
            {"status": "ACTIVE"},
            expected_params=create_workflow_definition_params(),
        )
        definition = client.create_workflow_definition(
            **create_workflow_definition_params(),
        )
        stubber.add_response(
            "create_workflow_run",
            {
                "workflowRunId": "12345678-1234-1234-1234-123456789012",
                "status": "RUNNING",
            },
            expected_params=create_workflow_run_params(),
        )
        run = client.create_workflow_run(**create_workflow_run_params())
    operations = [
        fixture_operation(
            service=_SERVICE,
            operation_name="ListModels",
            fixture_id=_fixture_id("ListModels"),
            effect="read",
        ),
        fixture_operation(
            service=_SERVICE,
            operation_name="CreateWorkflowDefinition",
            fixture_id=_fixture_id("CreateWorkflowDefinition"),
            effect="none",
        ),
        fixture_operation(
            service=_SERVICE,
            operation_name="CreateWorkflowRun",
            fixture_id=_fixture_id("CreateWorkflowRun"),
            effect="none",
        ),
    ]
    return operations, _nova_data(list_response, definition, run)


def _run_emulator() -> tuple[list[OperationOutcome], dict[str, object]]:
    return (
        [
            unsupported_emulator(
                service=_SERVICE,
                operation_name="ListModels",
            ),
            unsupported_emulator(
                service=_SERVICE,
                operation_name="CreateWorkflowDefinition",
            ),
            unsupported_emulator(
                service=_SERVICE,
                operation_name="CreateWorkflowRun",
            ),
        ],
        {"emulator": "not_supported"},
    )


def _run_live(
    settings: Settings,
    port: AwsPort | None,
) -> tuple[list[OperationOutcome], dict[str, object]]:
    operations: list[OperationOutcome] = []
    payload: Mapping[str, Any] = {"modelSummaries": [], "modelAliases": []}
    if port is None:
        operations.append(
            require_port_not_run(
                service=_SERVICE, operation_name="ListModels"
            ),
        )
    else:
        called = port_operation(
            port=port,
            service=_SERVICE,
            operation_name="ListModels",
            params=list_models_params(),
            execution="live",
            effect="read",
            fixture_id=_fixture_id("ListModels"),
        )
        operations.append(called.outcome)
        payload = called.payload
    operations.extend(
        _validated_contract_only(name)
        for name in ("CreateWorkflowDefinition", "CreateWorkflowRun")
    )
    data = _nova_data(payload, {}, {})
    data["region"] = settings.region
    return operations, data


def list_models_params() -> dict[str, int]:
    """Return required ListModels params."""
    return {"clientCompatibilityVersion": 1}


def create_workflow_definition_params() -> dict[str, str]:
    """Return a CreateWorkflowDefinition request."""
    return {"name": _WORKFLOW_NAME}


def create_workflow_run_params() -> dict[str, object]:
    """Return a CreateWorkflowRun request."""
    return {
        "workflowDefinitionName": _WORKFLOW_NAME,
        "modelId": _MODEL_ID,
        "clientInfo": {
            "compatibilityVersion": 1,
            "sdkVersion": "aws-ai-stack-workshop",
        },
    }


def _validated_contract_only(name: str) -> OperationOutcome:
    validate_request(
        service=_SERVICE,
        operation_name=name,
        params=_params_for(name),
        fixture_id=_fixture_id(name),
    )
    return contract_only(service=_SERVICE, operation_name=name)


def _params_for(name: str) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        {
            "CreateWorkflowDefinition": create_workflow_definition_params,
            "CreateWorkflowRun": create_workflow_run_params,
        }[name](),
    )


def _list_models_response() -> dict[str, Any]:
    return {
        "modelSummaries": [
            {
                "modelId": _MODEL_ID,
                "modelLifecycle": {"status": "ACTIVE"},
                "minimumCompatibilityVersion": 1,
            },
        ],
        "modelAliases": [
            {
                "aliasName": "amazon.nova-act-latest",
                "latestModelId": _MODEL_ID,
                "resolvedModelId": _MODEL_ID,
            },
        ],
        "compatibilityInformation": {
            "clientCompatibilityVersion": 1,
            "supportedModelIds": [_MODEL_ID],
        },
    }


def _nova_data(
    list_payload: Mapping[str, Any],
    definition_payload: Mapping[str, Any],
    run_payload: Mapping[str, Any],
) -> dict[str, object]:
    models = cast(
        "Sequence[Mapping[str, Any]]",
        list_payload.get("modelSummaries", ()),
    )
    aliases = cast(
        "Sequence[Mapping[str, Any]]",
        list_payload.get("modelAliases", ()),
    )
    compatibility = cast(
        "Mapping[str, Any]",
        list_payload.get("compatibilityInformation", {}),
    )
    return {
        "model_count": len(models),
        "alias_count": len(aliases),
        "supported_model_ids": list(
            compatibility.get("supportedModelIds", [])
        ),
        "workflow_definition_status": definition_payload.get("status"),
        "workflow_run_status": run_payload.get("status"),
    }


def _fixture_id(operation_name: str) -> str:
    return f"nova-act-{operation_name}-v1"
