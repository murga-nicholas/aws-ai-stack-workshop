"""Amazon Bedrock AgentCore Harness demo.

Technology: Amazon Bedrock AgentCore Harness.
Lane: agents.
Lifecycle refs: agentcore-harness, agents-classic.
Run:
    uv run awsai-demo agentcore-harness
    uv run awsai-demo agentcore-harness --execution emulator
    uv run awsai-demo agentcore-harness --execution live
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from awsai_demo import billing
from awsai_demo.demo_support import (
    AwsPort,
    AwsStreamPort,
    billable_not_priced,
    build_default_boto_port,
    build_result,
    fixture_operation,
    not_run_operation,
    port_operation,
    require_port_not_run,
    unsupported_emulator,
    validate_request,
)
from awsai_demo.policy import Charge, ExecutionPolicy, PolicyError
from awsai_demo.runtime import TESTED_LIVE_BEDROCK_MODEL, Execution, Settings
from awsai_demo.scenario import price_pilot
from awsai_demo.strands_multiagent_demo import model_routing
from awsai_demo.stubs import named_fixture_stream, stubbed_client

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from awsai_demo.billing import BudgetRun
    from awsai_demo.contracts import (
        CredentialSource,
        DemoResult,
        OperationOutcome,
    )
    from awsai_demo.credentials import CsvLoader, SessionFactory

_DEMO = "agentcore-harness"
_TECHNOLOGY = "Amazon Bedrock AgentCore Harness"
_REFS = ["agentcore-harness", "agents-classic"]
_CREATE_FIXTURE = "agentcore-harness-create-v1"
_GET_FIXTURE = "agentcore-harness-get-ready-v1"
_INVOKE_FIXTURE = "agentcore-harness-invoke-tool-use-v1"
_DELETE_FIXTURE = "agentcore-harness-delete-v1"
_LIST_FIXTURE = "agentcore-harness-list-v1"
_HARNESS_NAME = "awsaiDemoHarness"
_HARNESS_ID = "awsaiDemoHarness-1A2B3C4D5E"
_HARNESS_ARN = (
    f"arn:aws:bedrock-agentcore:us-east-1:123456789012:harness/{_HARNESS_ID}"
)
_ROLE_ARN = "arn:aws:iam::123456789012:role/awsai-demo-harness"
_CLIENT_TOKEN = "agentcore-harness-demo-token-012345"  # noqa: S105
_RUNTIME_SESSION_ID = "session-0123456789abcdef0123456789"
_TOOL_NAME = "accept_proposal"
_TOOL_USE_ID = "tooluse-accept-proposal-1"
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class HarnessPauseContract:
    """Stable data the decision demo can persist before a resume."""

    harness_arn: str
    runtime_session_id: str
    tool_use_id: str
    tool_name: str = _TOOL_NAME


@dataclass(frozen=True)
class HarnessInvokePause:
    """Parsed Harness stream pause evidence."""

    operations: tuple[OperationOutcome, ...]
    contract: HarnessPauseContract
    stop_reason: str


def run_agentcore_harness_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | AwsStreamPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Run the Harness demo in a lane-safe way."""
    model_id = _harness_model_id(settings, execution)
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
        operations, data = _run_offline(model_id)
    elif execution == "emulator":
        operations, data = _run_emulator(
            model_id,
            configured=port is not None,
        )
    else:
        operations, data = _run_live(settings, model_id, port, policy)
    return build_result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        execution=execution,
        headline="Harness pauses at tool use until a resume contract answers.",
        operations=operations,
        settings=settings,
        data=data,
        requested_model=model_id,
        observed_model=model_id
        if any(item["response_received"] for item in operations)
        else None,
        credential_source=credential_source,
    )


def build_harness_pause_contract(
    *,
    harness_arn: str = _HARNESS_ARN,
    runtime_session_id: str = _RUNTIME_SESSION_ID,
    tool_use_id: str = _TOOL_USE_ID,
    tool_name: str = _TOOL_NAME,
) -> HarnessPauseContract:
    """Return Harness pause data consumed by resume workers."""
    return HarnessPauseContract(
        harness_arn=harness_arn,
        runtime_session_id=runtime_session_id,
        tool_use_id=tool_use_id,
        tool_name=tool_name,
    )


def build_harness_resume_request(
    contract: HarnessPauseContract,
    *,
    approved: bool = False,
    reason: str = "Denied by default fixture.",
) -> dict[str, Any]:
    """Build a Harness toolResult request without side effects."""
    status = "success" if approved else "error"
    return {
        "harnessArn": contract.harness_arn,
        "runtimeSessionId": contract.runtime_session_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": contract.tool_use_id,
                            "status": status,
                            "content": [
                                {
                                    "json": {
                                        "approved": approved,
                                        "reason": reason,
                                    }
                                }
                            ],
                        }
                    }
                ],
            }
        ],
    }


def create_harness_params(
    *,
    model_id: str,
    role_arn: str,
    client_token: str = _CLIENT_TOKEN,
) -> dict[str, Any]:
    """Return the CreateHarness request used by the demo."""
    return {
        "harnessName": _HARNESS_NAME,
        "executionRoleArn": role_arn,
        "model": harness_model(model_id),
        "systemPrompt": harness_system_prompt(),
        "tools": [inline_function_tool(_TOOL_NAME)],
        "maxIterations": 2,
        "maxTokens": 256,
        "timeoutSeconds": 30,
        "tags": {"awsai-demo": "agentcore-harness"},
        "clientToken": client_token,
    }


def get_harness_params() -> dict[str, str]:
    """Return the GetHarness request."""
    return {"harnessId": _HARNESS_ID}


def delete_harness_params() -> dict[str, Any]:
    """Return the DeleteHarness request."""
    return {
        "harnessId": _HARNESS_ID,
        "clientToken": "agentcore-harness-delete-token-0123",
        "deleteManagedMemory": False,
    }


def list_harnesses_params() -> dict[str, int]:
    """Return a bounded ListHarnesses request."""
    return {"maxResults": 10}


def invoke_harness_params(model_id: str) -> dict[str, Any]:
    """Return the InvokeHarness request that pauses for a tool use."""
    return {
        "harnessArn": _HARNESS_ARN,
        "runtimeSessionId": _RUNTIME_SESSION_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Review the proposal and call accept_proposal "
                            "if it should proceed."
                        )
                    }
                ],
            }
        ],
        "model": harness_model(model_id),
        "systemPrompt": harness_system_prompt(),
        "tools": [inline_function_tool(_TOOL_NAME)],
        "maxIterations": 2,
        "maxTokens": 256,
        "timeoutSeconds": 30,
    }


def harness_model(model_id: str) -> dict[str, Any]:
    """Return Harness model configuration."""
    return {
        "bedrockModelConfig": {
            "modelId": model_id,
            "maxTokens": 256,
            "temperature": 0.0,
            "apiFormat": "converse_stream",
        }
    }


def harness_system_prompt() -> list[dict[str, str]]:
    """Return the Harness system prompt."""
    return [{"text": "You are a proposal approval harness."}]


def inline_function_tool(name: str) -> dict[str, Any]:
    """Return the inline function tool shape required by Harness."""
    return {
        "type": "inline_function",
        "name": name,
        "config": {
            "inlineFunction": {
                "description": "Ask a human to accept or reject a proposal.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "proposal_id": {"type": "string"},
                        "total_usd": {"type": "integer"},
                    },
                    "required": ["proposal_id", "total_usd"],
                },
            }
        },
    }


def _run_offline(
    model_id: str,
) -> tuple[list[OperationOutcome], dict[str, Any]]:
    operations: list[OperationOutcome] = []
    create_params = create_harness_params(
        model_id=model_id,
        role_arn=_ROLE_ARN,
    )
    with stubbed_client("bedrock-agentcore-control") as stubber:
        client = stubber.client
        stubber.add_response(
            "create_harness",
            {"harness": _harness_payload(model_id, status="CREATING")},
            expected_params=create_params,
        )
        create_response = _create_harness_request(
            client,
            role_arn=_ROLE_ARN,
            model_id=model_id,
            client_token=_CLIENT_TOKEN,
        )
        operations.append(
            fixture_operation(
                service="bedrock-agentcore-control",
                operation_name="CreateHarness",
                fixture_id=_CREATE_FIXTURE,
                effect="write",
                phase="setup",
            )
        )

        stubber.add_response(
            "get_harness",
            {"harness": _harness_payload(model_id, status="READY")},
            expected_params=get_harness_params(),
        )
        get_response = client.get_harness(**get_harness_params())
        operations.append(
            fixture_operation(
                service="bedrock-agentcore-control",
                operation_name="GetHarness",
                fixture_id=_GET_FIXTURE,
                effect="read",
                phase="setup",
            )
        )

    pause = _offline_invoke_pause(model_id)
    operations.extend(pause.operations)
    with stubbed_client("bedrock-agentcore-control") as stubber:
        client = stubber.client
        stubber.add_response(
            "list_harnesses",
            {"harnesses": [_harness_summary(status="READY")]},
            expected_params=list_harnesses_params(),
        )
        list_response = client.list_harnesses(**list_harnesses_params())
        operations.append(
            fixture_operation(
                service="bedrock-agentcore-control",
                operation_name="ListHarnesses",
                fixture_id=_LIST_FIXTURE,
                effect="read",
            )
        )

        stubber.add_response(
            "delete_harness",
            {},
            expected_params=delete_harness_params(),
        )
        client.delete_harness(**delete_harness_params())
        operations.append(
            fixture_operation(
                service="bedrock-agentcore-control",
                operation_name="DeleteHarness",
                fixture_id=_DELETE_FIXTURE,
                effect="delete",
                phase="teardown",
            )
        )

    return operations, _harness_data(
        create_response=create_response,
        get_response=get_response,
        list_response=list_response,
        pause_contract=pause.contract,
        stop_reason=pause.stop_reason,
    )


def _run_emulator(
    model_id: str,
    *,
    configured: bool,
) -> tuple[list[OperationOutcome], dict[str, Any]]:
    if not configured:
        return _run_emulator_missing_config(model_id)
    operations = [
        unsupported_emulator(
            service="bedrock-agentcore-control",
            operation_name="CreateHarness",
            phase="setup",
        ),
        unsupported_emulator(
            service="bedrock-agentcore-control",
            operation_name="GetHarness",
            phase="setup",
        ),
        unsupported_emulator(
            service="bedrock-agentcore",
            operation_name="InvokeHarness",
        ),
        unsupported_emulator(
            service="bedrock-agentcore-control",
            operation_name="ListHarnesses",
        ),
        unsupported_emulator(
            service="bedrock-agentcore-control",
            operation_name="DeleteHarness",
            phase="teardown",
        ),
    ]
    return operations, {"model": model_id, "emulator": "not_supported"}


def _run_emulator_missing_config(
    model_id: str,
) -> tuple[list[OperationOutcome], dict[str, Any]]:
    operations = [
        require_port_not_run(
            service="bedrock-agentcore-control",
            operation_name="CreateHarness",
            phase="setup",
        ),
        require_port_not_run(
            service="bedrock-agentcore-control",
            operation_name="GetHarness",
            phase="setup",
        ),
        require_port_not_run(
            service="bedrock-agentcore",
            operation_name="InvokeHarness",
        ),
        require_port_not_run(
            service="bedrock-agentcore-control",
            operation_name="ListHarnesses",
        ),
        require_port_not_run(
            service="bedrock-agentcore-control",
            operation_name="DeleteHarness",
            phase="teardown",
        ),
    ]
    return operations, {"model": model_id, "emulator": "missing_token"}


def _run_live(
    settings: Settings,
    model_id: str,
    port: AwsPort | AwsStreamPort | None,
    policy: ExecutionPolicy | None = None,
) -> tuple[list[OperationOutcome], dict[str, Any]]:
    operations: list[OperationOutcome] = []
    data: dict[str, Any] = {"model": model_id}
    operations.extend(
        _live_create_refused_by_pricing()
        if settings.allow_create
        else _live_create_refused_by_default()
    )
    if port is None:
        operations.append(
            require_port_not_run(
                service="bedrock-agentcore-control",
                operation_name="ListHarnesses",
            )
        )
    else:
        listed = port_operation(
            port=cast("AwsPort", port),
            service="bedrock-agentcore-control",
            operation_name="ListHarnesses",
            params=list_harnesses_params(),
            execution="live",
            effect="read",
            fixture_id=_LIST_FIXTURE,
        )
        operations.append(listed.outcome)
        data["listed"] = len(
            cast("Sequence[Any]", listed.payload.get("harnesses", ()))
        )
    operations.append(
        billable_not_priced(
            service="bedrock-agentcore-control",
            operation_name="DeleteHarness",
            phase="teardown",
        )
        if settings.allow_create
        else not_run_operation(
            service="bedrock-agentcore-control",
            operation_name="DeleteHarness",
            error_code="create_not_allowed",
            request_validated=False,
            phase="teardown",
        )
    )
    data["pricing"] = "missing"
    if settings.allow_create:
        data.update(
            harness_cost_preflight(settings, policy or ExecutionPolicy())
        )
    return operations, data


def harness_cost_preflight(
    settings: Settings,
    policy: ExecutionPolicy,
    *,
    budget_opener: Callable[..., BudgetRun] | None = None,
) -> dict[str, Any]:
    """Quote known units without treating them as a complete bound."""
    data: dict[str, Any] = {
        "pricing": "incomplete_bound",
        "invocation_executed": False,
        "unbounded_costs": [
            "Runtime memory remains billable until session termination; "
            "timeoutSeconds bounds only the invocation.",
            "Harness automatically emits chargeable CloudWatch telemetry "
            "without an application-enforced byte limit.",
            "Harness constructs model prompts internally; exact complete "
            "input counting cannot be enforced by this client.",
        ],
        "cost_source": (
            "https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/"
            "harness-operations.html"
        ),
        "runtime_max_vcpu": 2,
        "runtime_max_memory_gb": 8,
    }
    if not settings.harness_role_arn:
        data["missing_configuration"] = "AWSAI_HARNESS_ROLE_ARN"
    model_id = settings.model or TESTED_LIVE_BEDROCK_MODEL
    duration = min(30, policy.max_wall_seconds)
    iterations = min(2, policy.max_agent_iterations, policy.max_model_calls)
    data["timeout_seconds"] = duration
    try:
        open_budget = (
            billing.active_budget_run
            if budget_opener is None
            else budget_opener
        )
        budget = open_budget(region=settings.region, policy=policy).command(
            _DEMO
        )
        charges = (
            Charge(
                "agentcore.runtime.cpu_seconds",
                Decimal(2 * duration),
                "second",
            ),
            Charge(
                "agentcore.runtime.memory_gb_seconds",
                Decimal(8 * duration),
                "gb_second",
            ),
            Charge(
                f"model:{model_id}:input",
                Decimal(iterations * policy.max_input_tokens),
                "token",
                routing=model_routing(model_id),
            ),
            Charge(
                f"model:{model_id}:output",
                Decimal(iterations * policy.max_output_tokens),
                "token",
                routing=model_routing(model_id),
            ),
        )
        # Missing charges prevent turning this quote into a reservation.
        data["known_components_estimated_usd"] = float(budget.quote(charges))
    except (PolicyError, ValueError) as exc:
        data["pricing"] = "missing"
        data["price_failure"] = str(exc)
    return data


def _create_harness_request(
    client: Any,
    *,
    role_arn: str,
    model_id: str,
    client_token: str,
) -> Mapping[str, Any]:
    # slide: create-harness
    response = client.create_harness(
        harnessName=_HARNESS_NAME,
        executionRoleArn=role_arn,
        model=harness_model(model_id),
        systemPrompt=harness_system_prompt(),
        tools=[inline_function_tool(_TOOL_NAME)],
        maxIterations=2,
        maxTokens=256,
        timeoutSeconds=30,
        tags={"awsai-demo": "agentcore-harness"},
        clientToken=client_token,
    )
    # end-slide: create-harness
    return cast("Mapping[str, Any]", response)


def _offline_invoke_pause(model_id: str) -> HarnessInvokePause:
    invoke_params = invoke_harness_params(model_id)
    validate_request(
        service="bedrock-agentcore",
        operation_name="InvokeHarness",
        params=invoke_params,
        fixture_id=_INVOKE_FIXTURE,
    )
    stream = named_fixture_stream(
        fixture_id=_INVOKE_FIXTURE,
        events=_tool_use_stream_events(),
    )
    events = tuple(stream)
    contract, stop_reason = _pause_contract_from_events(events)
    resume_params = build_harness_resume_request(contract, approved=False)
    validate_request(
        service="bedrock-agentcore",
        operation_name="InvokeHarness",
        params=resume_params,
        fixture_id="agentcore-harness-resume-denied-v1",
    )
    return HarnessInvokePause(
        operations=(
            fixture_operation(
                service="bedrock-agentcore",
                operation_name="InvokeHarness",
                fixture_id=stream.fixture_id,
                effect="infer",
                usage={
                    "inputTokens": 33,
                    "outputTokens": 5,
                    "totalTokens": 38,
                },
            ),
        ),
        contract=contract,
        stop_reason=stop_reason,
    )


def _harness_data(
    *,
    create_response: Mapping[str, Any],
    get_response: Mapping[str, Any],
    list_response: Mapping[str, Any],
    pause_contract: HarnessPauseContract,
    stop_reason: str,
) -> dict[str, Any]:
    created = cast("Mapping[str, Any]", create_response["harness"])
    ready = cast("Mapping[str, Any]", get_response["harness"])
    listed = cast("Sequence[Any]", list_response["harnesses"])
    denied_resume = build_harness_resume_request(
        pause_contract,
        approved=False,
    )
    return {
        "harness_id": created["harnessId"],
        "harness_arn": created["arn"],
        "status_after_get": ready["status"],
        "listed": len(listed),
        "runtime_session_id": pause_contract.runtime_session_id,
        "tool_use_id": pause_contract.tool_use_id,
        "pause": {
            **asdict(pause_contract),
            "state": "pending",
            "stop_reason": stop_reason,
            "resume_shape": "InvokeHarness toolResult",
            "default_answer": "DENIED",
        },
        "resume_request": denied_resume,
        "approval_source": "not_answered_in_plain_run",
        "isolated_approval_fixture": "agentcore-harness-resume-approved-v1",
    }


def _live_create_refused_by_default() -> list[OperationOutcome]:
    return [
        not_run_operation(
            service="bedrock-agentcore-control",
            operation_name="CreateHarness",
            error_code="create_not_allowed",
            request_validated=False,
            phase="setup",
        ),
        not_run_operation(
            service="bedrock-agentcore-control",
            operation_name="GetHarness",
            error_code="create_not_allowed",
            request_validated=False,
            phase="setup",
        ),
        not_run_operation(
            service="bedrock-agentcore",
            operation_name="InvokeHarness",
            error_code="invoke_not_allowed",
            request_validated=False,
        ),
    ]


def _live_create_refused_by_pricing() -> list[OperationOutcome]:
    return [
        billable_not_priced(
            service="bedrock-agentcore-control",
            operation_name="CreateHarness",
            phase="setup",
        ),
        billable_not_priced(
            service="bedrock-agentcore-control",
            operation_name="GetHarness",
            phase="setup",
        ),
        billable_not_priced(
            service="bedrock-agentcore",
            operation_name="InvokeHarness",
        ),
    ]


def _harness_payload(model_id: str, *, status: str) -> dict[str, Any]:
    return {
        "harnessId": _HARNESS_ID,
        "harnessName": _HARNESS_NAME,
        "arn": _HARNESS_ARN,
        "status": status,
        "harnessVersion": "1",
        "executionRoleArn": _ROLE_ARN,
        "createdAt": _NOW,
        "updatedAt": _NOW,
        "model": harness_model(model_id),
        "systemPrompt": harness_system_prompt(),
        "tools": [inline_function_tool(_TOOL_NAME)],
        "skills": [],
        "allowedTools": [_TOOL_NAME],
        "truncation": {"strategy": "none"},
        "environment": _runtime_environment(),
        "maxIterations": 2,
        "maxTokens": 256,
        "timeoutSeconds": 30,
    }


def _harness_summary(*, status: str) -> dict[str, Any]:
    return {
        "harnessId": _HARNESS_ID,
        "harnessName": _HARNESS_NAME,
        "arn": _HARNESS_ARN,
        "status": status,
        "createdAt": _NOW,
        "updatedAt": _NOW,
        "harnessVersion": "1",
    }


def _runtime_environment() -> dict[str, Any]:
    return {
        "agentCoreRuntimeEnvironment": {
            "agentRuntimeArn": (
                "arn:aws:bedrock-agentcore:us-east-1:123456789012:"
                "runtime/awsai-demo-runtime"
            ),
            "agentRuntimeName": "awsaiDemoRuntime",
            "agentRuntimeId": "awsaiDemoRuntimeId",
            "lifecycleConfiguration": {
                "idleRuntimeSessionTimeout": 600,
                "maxLifetime": 3600,
            },
            "networkConfiguration": {"networkMode": "PUBLIC"},
        }
    }


def _tool_use_stream_events() -> tuple[Mapping[str, Any], ...]:
    return (
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {
                    "toolUse": {
                        "toolUseId": _TOOL_USE_ID,
                        "name": _TOOL_NAME,
                    }
                },
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {
                    "toolUse": {
                        "input": (
                            '{"proposal_id":"aws-ai-stack",'
                            f'"total_usd":{int(price_pilot().total_usd)}'
                            "}"
                        )
                    }
                },
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": 33,
                    "outputTokens": 5,
                    "totalTokens": 38,
                },
                "metrics": {"latencyMs": 18},
            }
        },
    )


def _pause_contract_from_events(
    events: Sequence[Mapping[str, Any]],
) -> tuple[HarnessPauseContract, str]:
    tool_use_id = ""
    tool_name = ""
    stop_reason = ""
    for event in events:
        start = cast("Mapping[str, Any]", event.get("contentBlockStart", {}))
        start_block = cast("Mapping[str, Any]", start.get("start", {}))
        tool_use = cast("Mapping[str, Any]", start_block.get("toolUse", {}))
        if tool_use:
            tool_use_id = str(tool_use["toolUseId"])
            tool_name = str(tool_use["name"])
        stop = cast("Mapping[str, Any]", event.get("messageStop", {}))
        if stop:
            stop_reason = str(stop["stopReason"])
    if not tool_use_id or not tool_name or stop_reason != "tool_use":
        msg = "Harness fixture stream did not pause on a tool use"
        raise ValueError(msg)
    return (
        build_harness_pause_contract(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
        ),
        stop_reason,
    )


def _harness_model_id(settings: Settings, execution: Execution) -> str:
    if settings.model is not None:
        return settings.model
    if execution == "emulator":
        return _localstack_model_id(settings)
    return TESTED_LIVE_BEDROCK_MODEL


def _localstack_model_id(settings: Settings) -> str:
    configured = getattr(settings, "localstack_bedrock_model", None)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    default = settings.default_bedrock_model
    if default.startswith("ollama."):
        return default
    return f"ollama.{default}"
