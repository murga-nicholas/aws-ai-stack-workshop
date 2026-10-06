"""Amazon Bedrock prompt management and Flows demo.

Technology: Amazon Bedrock Prompt management and Flows.
Lane: models.
Lifecycle refs: bedrock-flows, prompt-management.
Run:
    uv run awsai-demo bedrock-flows
    uv run awsai-demo bedrock-flows --execution live
"""

from __future__ import annotations

from datetime import UTC, datetime
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
from awsai_demo.stubs import named_fixture_stream, stubbed_client

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

_DEMO = "bedrock-flows"
_TECHNOLOGY = "Bedrock Prompt management and Flows"
_REFS = ["bedrock-flows", "prompt-management"]
_SERVICE = "bedrock-agent"
_RUNTIME_SERVICE = "bedrock-agent-runtime"
_PROMPT_ID = "PROMPT12345"
_FLOW_ID = "FLOW12345"
_ALIAS_ID = "ALIAS12345"
_ROLE_ARN = "arn:aws:iam::123456789012:role/awsai-demo-flow"
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def run_bedrock_flows_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Run Prompt management and Flows without unmanaged writes."""
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
        headline="Prompt and Flow contracts are shape-checked safely.",
        operations=operations,
        settings=settings,
        data=data,
        credential_source=credential_source,
    )


def _run_offline() -> tuple[list[OperationOutcome], dict[str, object]]:
    operations: list[OperationOutcome] = []
    with stubbed_client(_SERVICE) as stubber:
        client = stubber.client
        for method, operation_name, params, response in _agent_fixture_rows():
            stubber.add_response(method, response, expected_params=params)
            getattr(client, method)(**params)
            operations.append(_fixture_agent_operation(operation_name))
    validate_request(
        service=_RUNTIME_SERVICE,
        operation_name="InvokeFlow",
        params=invoke_flow_params(),
        fixture_id=_fixture_id("InvokeFlow"),
    )
    stream = named_fixture_stream(
        fixture_id=_fixture_id("InvokeFlow"),
        events=_flow_events(),
    )
    operations.insert(
        6,
        fixture_operation(
            service=_RUNTIME_SERVICE,
            operation_name="InvokeFlow",
            fixture_id=stream.fixture_id,
            effect="none",
        ),
    )
    return operations, _flow_data(
        prompt_count=1,
        flow_count=1,
        stream_events=len(tuple(stream)),
    )


def _run_emulator() -> tuple[list[OperationOutcome], dict[str, object]]:
    operations = [
        unsupported_emulator(
            service=_SERVICE,
            operation_name=name,
            phase="setup" if name in _SETUP_NAMES else "main",
        )
        for name in _AGENT_OPERATION_NAMES
    ]
    operations.insert(
        6,
        unsupported_emulator(
            service=_RUNTIME_SERVICE,
            operation_name="InvokeFlow",
        ),
    )
    return operations, {"emulator": "not_supported"}


def _run_live(
    settings: Settings,
    port: AwsPort | None,
) -> tuple[list[OperationOutcome], dict[str, object]]:
    operations = [_validated_contract_only(name) for name in _CONTRACT_ONLY]
    prompts: Mapping[str, Any] = {"promptSummaries": []}
    flows: Mapping[str, Any] = {"flowSummaries": []}
    if port is None:
        operations.extend(
            require_port_not_run(service=_SERVICE, operation_name=name)
            for name in ("ListPrompts", "ListFlows")
        )
    else:
        for name, params in (
            ("ListPrompts", list_prompts_params()),
            ("ListFlows", list_flows_params()),
        ):
            called = port_operation(
                port=port,
                service=_SERVICE,
                operation_name=name,
                params=params,
                execution="live",
                effect="read",
                fixture_id=_fixture_id(name),
            )
            operations.append(called.outcome)
            if name == "ListPrompts":
                prompts = called.payload
            else:
                flows = called.payload
    data = _flow_data(
        prompt_count=len(
            cast("Sequence[Any]", prompts.get("promptSummaries", ()))
        ),
        flow_count=len(cast("Sequence[Any]", flows.get("flowSummaries", ()))),
        stream_events=0,
    )
    data["region"] = settings.region
    return operations, data


def create_prompt_params() -> dict[str, Any]:
    """Return a minimal CreatePrompt request."""
    return {"name": "awsaiDemoPrompt"}


def create_prompt_version_params() -> dict[str, str]:
    """Return a CreatePromptVersion request."""
    return {"promptIdentifier": _PROMPT_ID}


def create_flow_params() -> dict[str, str]:
    """Return a minimal CreateFlow request."""
    return {"name": "awsaiDemoFlow", "executionRoleArn": _ROLE_ARN}


def prepare_flow_params() -> dict[str, str]:
    """Return a PrepareFlow request."""
    return {"flowIdentifier": _FLOW_ID}


def create_flow_version_params() -> dict[str, str]:
    """Return a CreateFlowVersion request."""
    return {"flowIdentifier": _FLOW_ID}


def create_flow_alias_params() -> dict[str, Any]:
    """Return a CreateFlowAlias request."""
    return {
        "name": "awsaiDemoAlias",
        "flowIdentifier": _FLOW_ID,
        "routingConfiguration": [{"flowVersion": "1"}],
    }


def invoke_flow_params() -> dict[str, Any]:
    """Return an InvokeFlow request."""
    return {
        "flowIdentifier": _FLOW_ID,
        "flowAliasIdentifier": _ALIAS_ID,
        "inputs": [
            {
                "content": {"document": {"proposal": "support pilot"}},
                "nodeName": "FlowInputNode",
            },
        ],
    }


def list_prompts_params() -> dict[str, int]:
    """Return bounded ListPrompts params."""
    return {"maxResults": 10}


def list_flows_params() -> dict[str, int]:
    """Return bounded ListFlows params."""
    return {"maxResults": 10}


def _validated_contract_only(name: str) -> OperationOutcome:
    service = _RUNTIME_SERVICE if name == "InvokeFlow" else _SERVICE
    validate_request(
        service=service,
        operation_name=name,
        params=_params_for(name),
        fixture_id=_fixture_id(name),
    )
    return contract_only(
        service=service,
        operation_name=name,
        phase="setup" if name in _SETUP_NAMES else "main",
    )


def _fixture_agent_operation(operation_name: str) -> OperationOutcome:
    return fixture_operation(
        service=_SERVICE,
        operation_name=operation_name,
        fixture_id=_fixture_id(operation_name),
        effect="read"
        if operation_name in {"ListPrompts", "ListFlows"}
        else "none",
        phase="setup" if operation_name in _SETUP_NAMES else "main",
    )


def _agent_fixture_rows() -> tuple[
    tuple[str, str, dict[str, Any], dict[str, Any]], ...
]:
    return (
        (
            "create_prompt",
            "CreatePrompt",
            create_prompt_params(),
            _prompt_response("DRAFT"),
        ),
        (
            "create_prompt_version",
            "CreatePromptVersion",
            create_prompt_version_params(),
            _prompt_response("1"),
        ),
        (
            "create_flow",
            "CreateFlow",
            create_flow_params(),
            _flow_response("NotPrepared", "DRAFT"),
        ),
        (
            "prepare_flow",
            "PrepareFlow",
            prepare_flow_params(),
            {"id": _FLOW_ID, "status": "Prepared"},
        ),
        (
            "create_flow_version",
            "CreateFlowVersion",
            create_flow_version_params(),
            _flow_version_response(),
        ),
        (
            "create_flow_alias",
            "CreateFlowAlias",
            create_flow_alias_params(),
            _alias_response(),
        ),
        (
            "list_prompts",
            "ListPrompts",
            list_prompts_params(),
            _list_prompts_response(),
        ),
        (
            "list_flows",
            "ListFlows",
            list_flows_params(),
            _list_flows_response(),
        ),
    )


def _prompt_response(version: str) -> dict[str, Any]:
    return {
        "name": "awsaiDemoPrompt",
        "id": _PROMPT_ID,
        "arn": f"arn:aws:bedrock:us-east-1:123456789012:prompt/{_PROMPT_ID}",
        "version": version,
        "createdAt": _NOW,
        "updatedAt": _NOW,
    }


def _flow_response(status: str, version: str) -> dict[str, Any]:
    return {
        "name": "awsaiDemoFlow",
        "executionRoleArn": _ROLE_ARN,
        "id": _FLOW_ID,
        "arn": f"arn:aws:bedrock:us-east-1:123456789012:flow/{_FLOW_ID}",
        "status": status,
        "createdAt": _NOW,
        "updatedAt": _NOW,
        "version": version,
    }


def _flow_version_response() -> dict[str, Any]:
    response = _flow_response("Prepared", "1")
    response.pop("updatedAt")
    return response


def _alias_response() -> dict[str, Any]:
    return {
        "name": "awsaiDemoAlias",
        "routingConfiguration": [{"flowVersion": "1"}],
        "flowId": _FLOW_ID,
        "id": _ALIAS_ID,
        "arn": (
            f"arn:aws:bedrock:us-east-1:123456789012:flow/"
            f"{_FLOW_ID}/alias/{_ALIAS_ID}"
        ),
        "createdAt": _NOW,
        "updatedAt": _NOW,
    }


def _list_prompts_response() -> dict[str, Any]:
    return {"promptSummaries": [_prompt_response("1")]}


def _list_flows_response() -> dict[str, Any]:
    return {
        "flowSummaries": [
            {
                "name": "awsaiDemoFlow",
                "id": _FLOW_ID,
                "arn": (
                    f"arn:aws:bedrock:us-east-1:123456789012:flow/{_FLOW_ID}"
                ),
                "status": "Prepared",
                "createdAt": _NOW,
                "updatedAt": _NOW,
                "version": "DRAFT",
            },
        ],
    }


def _flow_events() -> tuple[dict[str, object], ...]:
    return (
        {
            "flowOutputEvent": {
                "content": {"document": {"summary": "approved"}},
                "nodeName": "FlowOutputNode",
                "nodeType": "FlowOutputNode",
            },
        },
        {"flowCompletionEvent": {"completionReason": "SUCCESS"}},
    )


def _params_for(name: str) -> dict[str, Any]:
    return {
        "CreatePrompt": create_prompt_params,
        "CreatePromptVersion": create_prompt_version_params,
        "CreateFlow": create_flow_params,
        "PrepareFlow": prepare_flow_params,
        "CreateFlowVersion": create_flow_version_params,
        "CreateFlowAlias": create_flow_alias_params,
        "InvokeFlow": invoke_flow_params,
    }[name]()


def _fixture_id(operation_name: str) -> str:
    return f"bedrock-flows-{operation_name}-v1"


def _flow_data(
    *,
    prompt_count: int,
    flow_count: int,
    stream_events: int,
) -> dict[str, object]:
    return {
        "prompt_id": _PROMPT_ID,
        "flow_id": _FLOW_ID,
        "flow_alias_id": _ALIAS_ID,
        "prompt_count": prompt_count,
        "flow_count": flow_count,
        "stream_event_count": stream_events,
    }


_AGENT_OPERATION_NAMES = (
    "CreatePrompt",
    "CreatePromptVersion",
    "CreateFlow",
    "PrepareFlow",
    "CreateFlowVersion",
    "CreateFlowAlias",
    "ListPrompts",
    "ListFlows",
)
_CONTRACT_ONLY = (
    "CreatePrompt",
    "CreatePromptVersion",
    "CreateFlow",
    "PrepareFlow",
    "CreateFlowVersion",
    "CreateFlowAlias",
    "InvokeFlow",
)
_SETUP_NAMES = frozenset(
    {
        "CreatePrompt",
        "CreatePromptVersion",
        "CreateFlow",
        "PrepareFlow",
        "CreateFlowVersion",
        "CreateFlowAlias",
    },
)
