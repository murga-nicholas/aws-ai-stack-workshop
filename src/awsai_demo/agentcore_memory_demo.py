"""Amazon Bedrock AgentCore Memory demo.

Technology: Amazon Bedrock AgentCore Memory.
Lane: agents.
Lifecycle refs: agentcore-memory.
Run:
    uv run awsai-demo agentcore-memory
    uv run awsai-demo agentcore-memory --execution emulator
    uv run awsai-demo agentcore-memory --execution live --allow-create
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from awsai_demo.billing import active_budget_run
from awsai_demo.demo_support import (
    AwsPort,
    billable_not_priced,
    build_result,
    contract_only,
    fixture_operation,
    local_operation,
    not_run_operation,
    require_port_not_run,
    resolve_boto_port,
    unsupported_emulator,
)
from awsai_demo.owned_resources import MEMORY, OwnedResources
from awsai_demo.policy import Charge, ExecutionPolicy, PolicyError
from awsai_demo.stubs import stubbed_client

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from awsai_demo.contracts import (
        CredentialSource,
        DemoResult,
        Effect,
        ErrorCode,
        OperationOutcome,
        Phase,
    )
    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.demo_support import BotoPortFactory
    from awsai_demo.policy import PricedBudget
    from awsai_demo.runtime import Execution, Settings

_DEMO = "agentcore-memory"
_TECHNOLOGY = "Amazon Bedrock AgentCore Memory"
_REFS = ["agentcore-memory"]
_CREATE_FIXTURE = "agentcore-memory-create-v1"
_GET_FIXTURE = "agentcore-memory-get-active-v1"
_USER_EVENT_FIXTURE = "agentcore-memory-create-user-event-v1"
_ASSISTANT_EVENT_FIXTURE = "agentcore-memory-create-assistant-event-v1"
_LIST_EVENTS_FIXTURE = "agentcore-memory-list-events-v1"
_RETRIEVE_FIXTURE = "agentcore-memory-retrieve-contract-v1"
_DELETE_FIXTURE = "agentcore-memory-delete-v1"
_MEMORY_NAME = "awsai_demo_mem"
_MEMORY_ID = "awsai_demo_mem_1234567890"
_MEMORY_ARN = (
    "arn:aws:bedrock-agentcore:us-east-1:123456789012:memory/" + _MEMORY_ID
)
_CLIENT_TOKEN = "agentcore-memory-demo-token-012345"  # noqa: S105
_DELETE_TOKEN = "agentcore-memory-delete-token-0123"  # noqa: S105
_ACTOR_ID = "user-aws-ai-stack"
_SESSION_ID = "session-0123456789abcdef0123456789"
_USER_EVENT_ID = "event-user-0123456789abcdef"
_ASSISTANT_EVENT_ID = "event-assistant-0123456789"
_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_STRANDS_INTEGRATION = (
    "bedrock_agentcore.memory.integrations.strands.session_manager"
)


def run_agentcore_memory_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
    boto_port_factory: BotoPortFactory | None = None,
    event_clock: Callable[[], datetime] | None = None,
) -> DemoResult:
    """Run the AgentCore Memory demo in the requested lane."""
    credential_source: CredentialSource | None = None
    if execution == "offline":
        operations, data = _run_offline()
    elif execution == "emulator":
        default = _default_port(
            execution=execution,
            settings=settings,
            policy=policy,
            port=port,
            session_factory=session_factory,
            csv_loader=csv_loader,
            boto_port_factory=boto_port_factory,
        )
        port = default.port if default is not None else port
        credential_source = (
            None if default is None else default.credential_source
        )
        operations, data = _run_emulator(configured=port is not None)
    else:
        operations, data, credential_source = _run_live(
            settings=settings,
            policy=policy,
            port=port,
            session_factory=session_factory,
            csv_loader=csv_loader,
            boto_port_factory=boto_port_factory,
            event_clock=event_clock,
        )
    return build_result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        execution=execution,
        headline="AgentCore Memory recorded a short conversation contract.",
        operations=operations,
        settings=settings,
        data=data,
        credential_source=credential_source,
        result_error=_cleanup_result_error(data),
    )


def create_memory_params(
    *,
    name: str = _MEMORY_NAME,
    client_token: str = _CLIENT_TOKEN,
    tags: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the CreateMemory request used by the demo."""
    return {
        "name": name,
        "eventExpiryDuration": 7,
        "memoryStrategies": [],
        "tags": dict(tags) if tags is not None else _fixture_tags(),
        "clientToken": client_token,
    }


def get_memory_params(memory_id: str = _MEMORY_ID) -> dict[str, str]:
    """Return GetMemory params."""
    return {"memoryId": memory_id}


def delete_memory_params(memory_id: str = _MEMORY_ID) -> dict[str, str]:
    """Return DeleteMemory params."""
    return {"memoryId": memory_id, "clientToken": _DELETE_TOKEN}


def create_event_params(
    *,
    memory_id: str = _MEMORY_ID,
    role: str,
    text: str,
    client_token: str,
    event_timestamp: datetime = _NOW,
) -> dict[str, Any]:
    """Return a CreateEvent request for one conversational turn."""
    return {
        "memoryId": memory_id,
        "actorId": _ACTOR_ID,
        "sessionId": _SESSION_ID,
        "eventTimestamp": event_timestamp,
        "payload": [
            {"conversational": {"role": role, "content": {"text": text}}}
        ],
        "clientToken": client_token,
        "metadata": {"demo": {"stringValue": _DEMO}},
        "extractionMode": "SKIP",
    }


def list_events_params(memory_id: str = _MEMORY_ID) -> dict[str, Any]:
    """Return a bounded ListEvents request."""
    return {
        "memoryId": memory_id,
        "actorId": _ACTOR_ID,
        "sessionId": _SESSION_ID,
        "includePayloads": True,
        "maxResults": 10,
    }


def retrieve_memory_records_params(
    memory_id: str = _MEMORY_ID,
) -> dict[str, Any]:
    """Return RetrieveMemoryRecords request shape."""
    return {
        "memoryId": memory_id,
        "searchCriteria": {"searchQuery": "proposal approval", "topK": 3},
        "maxResults": 3,
    }


def memory_event_charges(count: int = 1) -> tuple[Charge, ...]:
    """Return AgentCore Memory event charges."""
    return (Charge("memory.events", Decimal(count), "event"),)


def _run_offline() -> tuple[list[OperationOutcome], dict[str, Any]]:
    operations: list[OperationOutcome] = []
    create_params = create_memory_params()
    user_params = create_event_params(
        role="USER",
        text="Please remember that the proposal needs finance review.",
        client_token=_CLIENT_TOKEN + "-user",
    )
    assistant_params = create_event_params(
        role="ASSISTANT",
        text="I will keep finance review attached to the proposal.",
        client_token=_CLIENT_TOKEN + "-assistant",
    )
    with stubbed_client("bedrock-agentcore-control") as stubber:
        client = stubber.client
        stubber.add_response(
            "create_memory",
            {"memory": _memory_payload(status="CREATING")},
            expected_params=create_params,
        )
        create_response = client.create_memory(**create_params)
        operations.append(
            _fixture(
                "bedrock-agentcore-control",
                "CreateMemory",
                _CREATE_FIXTURE,
                "write",
                "setup",
            )
        )
        stubber.add_response(
            "get_memory",
            {"memory": _memory_payload(status="ACTIVE")},
            expected_params=get_memory_params(),
        )
        get_response = client.get_memory(**get_memory_params())
        operations.append(
            _fixture(
                "bedrock-agentcore-control",
                "GetMemory",
                _GET_FIXTURE,
                "read",
                "setup",
            )
        )
    with stubbed_client("bedrock-agentcore") as stubber:
        client = stubber.client
        stubber.add_response(
            "create_event",
            {"event": _event_payload(role="USER", event_id=_USER_EVENT_ID)},
            expected_params=user_params,
        )
        user_response = client.create_event(**user_params)
        operations.append(
            _fixture(
                "bedrock-agentcore",
                "CreateEvent",
                _USER_EVENT_FIXTURE,
                "write",
                "main",
            )
        )
        stubber.add_response(
            "create_event",
            {
                "event": _event_payload(
                    role="ASSISTANT", event_id=_ASSISTANT_EVENT_ID
                )
            },
            expected_params=assistant_params,
        )
        assistant_response = client.create_event(**assistant_params)
        operations.append(
            _fixture(
                "bedrock-agentcore",
                "CreateEvent",
                _ASSISTANT_EVENT_FIXTURE,
                "write",
                "main",
            )
        )
        stubber.add_response(
            "list_events",
            {"events": [user_response["event"], assistant_response["event"]]},
            expected_params=list_events_params(),
        )
        list_response = client.list_events(**list_events_params())
        operations.append(
            _fixture(
                "bedrock-agentcore",
                "ListEvents",
                _LIST_EVENTS_FIXTURE,
                "read",
                "main",
            )
        )
        stubber.add_response(
            "retrieve_memory_records",
            {"memoryRecordSummaries": []},
            expected_params=retrieve_memory_records_params(),
        )
        retrieve_response = client.retrieve_memory_records(
            **retrieve_memory_records_params()
        )
        operations.append(
            _fixture(
                "bedrock-agentcore",
                "RetrieveMemoryRecords",
                _RETRIEVE_FIXTURE,
                "none",
                "main",
            )
        )
    operations.append(
        local_operation(
            service=_STRANDS_INTEGRATION,
            operation_name="NameStrandsIntegration",
        )
    )
    with stubbed_client("bedrock-agentcore-control") as stubber:
        client = stubber.client
        stubber.add_response(
            "delete_memory",
            {"memoryId": _MEMORY_ID, "status": "DELETING"},
            expected_params=delete_memory_params(),
        )
        delete_response = client.delete_memory(**delete_memory_params())
        operations.append(
            _fixture(
                "bedrock-agentcore-control",
                "DeleteMemory",
                _DELETE_FIXTURE,
                "delete",
                "teardown",
            )
        )
    return operations, {
        "memory_id": create_response["memory"]["id"],
        "status_after_get": get_response["memory"]["status"],
        "event_count": len(list_response["events"]),
        "retrieved_records": len(retrieve_response["memoryRecordSummaries"]),
        "delete_status": delete_response["status"],
        "strands_integration": _STRANDS_INTEGRATION,
    }


def _run_emulator(
    *,
    configured: bool,
) -> tuple[list[OperationOutcome], dict[str, Any]]:
    builder = unsupported_emulator if configured else require_port_not_run
    operations = [
        builder(
            service="bedrock-agentcore-control",
            operation_name="CreateMemory",
            phase="setup",
        ),
        builder(
            service="bedrock-agentcore-control",
            operation_name="GetMemory",
            phase="setup",
        ),
        builder(service="bedrock-agentcore", operation_name="CreateEvent"),
        builder(service="bedrock-agentcore", operation_name="CreateEvent"),
        builder(service="bedrock-agentcore", operation_name="ListEvents"),
        contract_only(
            service="bedrock-agentcore", operation_name="RetrieveMemoryRecords"
        ),
        local_operation(
            service=_STRANDS_INTEGRATION,
            operation_name="NameStrandsIntegration",
        ),
        builder(
            service="bedrock-agentcore-control",
            operation_name="DeleteMemory",
            phase="teardown",
        ),
    ]
    return operations, {
        "emulator": "not_supported" if configured else "missing_token"
    }


def _run_live(
    *,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None,
    session_factory: SessionFactory | None,
    csv_loader: CsvLoader | None,
    boto_port_factory: BotoPortFactory | None = None,
    event_clock: Callable[[], datetime] | None = None,
) -> tuple[list[OperationOutcome], dict[str, Any], CredentialSource | None]:
    if not settings.allow_create:
        return _live_create_refused(), {"allow_create": False}, None
    operations: list[OperationOutcome] = []
    data: dict[str, Any] = {"allow_create": True}
    try:
        run = active_budget_run(region=settings.region, policy=policy)
        _preflight_memory_budget(run.command(_DEMO))
    except PolicyError:
        return _pricing_refusal(), {"pricing": "missing"}, None
    default = _default_port(
        execution="live",
        settings=settings,
        policy=policy,
        port=port,
        session_factory=session_factory,
        csv_loader=csv_loader,
        boto_port_factory=boto_port_factory,
    )
    credential_source = None if default is None else default.credential_source
    port = default.port if default is not None else port
    if port is None:
        return _missing_live_port(), data, credential_source
    ctx = OwnedResources(run=run, demo=_DEMO, port=port, operations=operations)
    try:
        if not ctx.authenticate():
            data["identity"] = "unverified"
            return operations, data, credential_source
        name = f"awsai_{run.run_id}_mem"
        entry = ctx.create(
            MEMORY,
            create_memory_params(
                name=name,
                client_token=_token(run.run_id, "mem"),
                tags=_run_tags(run.run_id),
            ),
        )
        if entry is None:
            data["created"] = False
            return operations, data, credential_source
        memory_id = str(entry.exact_id)
        active = _poll_memory_active(ctx, memory_id)
        data["status_after_get"] = active
        if active != "ACTIVE":
            data["events_created"] = False
            _mark_memory_not_active(operations, active)
            return operations, data, credential_source
        if not _verify_for_child(ctx, entry, MEMORY):
            data["ownership_verified"] = False
            return operations, data, credential_source
        _create_live_events(ctx, entry, memory_id, event_clock=event_clock)
        listed = ctx.call(
            "bedrock-agentcore",
            "ListEvents",
            list_events_params(memory_id),
            effect="read",
        )
        data["event_count"] = len(
            cast("Sequence[Any]", listed.payload.get("events", ()))
        )
        operations.append(
            contract_only(
                service="bedrock-agentcore",
                operation_name="RetrieveMemoryRecords",
            )
        )
        operations.append(
            local_operation(
                service=_STRANDS_INTEGRATION,
                operation_name="NameStrandsIntegration",
            )
        )
        data["memory_id"] = memory_id
        data["strands_integration"] = _STRANDS_INTEGRATION
        return operations, data, credential_source  # noqa: TRY300
    except PolicyError as exc:
        data["policy_error"] = exc.__class__.__name__
        operations.extend(_event_pricing_refusal())
        return operations, data, credential_source
    finally:
        data["cleanup_incomplete"] = ctx.cleanup()


def _cleanup_result_error(
    data: Mapping[str, Any],
) -> tuple[ErrorCode, str] | None:
    if data.get("cleanup_incomplete") is True:
        return ("cleanup_incomplete", "Owned resources remain unresolved.")
    return None


def _poll_memory_active(ctx: OwnedResources, memory_id: str) -> str:
    status = ""
    max_polls = max(1, min(60, ctx.run.policy.max_wall_seconds))
    for attempt in range(max_polls):
        got = ctx.call(
            "bedrock-agentcore-control",
            "GetMemory",
            get_memory_params(memory_id),
            effect="read",
            phase="setup",
        )
        if got.outcome["status"] != "ok":
            return "GET_FAILED"
        memory = cast("Mapping[str, Any]", got.payload.get("memory", {}))
        status = str(memory.get("status", ""))
        if status in {"ACTIVE", "FAILED"}:
            return status
        if attempt < max_polls - 1:
            ctx.budget.ledger.check_wall_clock()
            time.sleep(1.0)
    return status or "TIMEOUT"


def _mark_memory_not_active(
    operations: list[OperationOutcome],
    status: str,
) -> None:
    for outcome in reversed(operations):
        if (
            outcome["service"] == "bedrock-agentcore-control"
            and outcome["operation"] == "GetMemory"
            and outcome["phase"] == "setup"
        ):
            if outcome["status"] != "ok":
                return
            outcome["status"] = "error" if status == "FAILED" else "blocked"
            outcome["error_code"] = (
                "validation_failed" if status == "FAILED" else "timeout"
            )
            return


def _create_live_events(
    ctx: OwnedResources,
    parent: Any,
    memory_id: str,
    *,
    event_clock: Callable[[], datetime] | None = None,
) -> None:
    for role, text, suffix in (
        ("USER", "Please remember that finance review is required.", "user"),
        (
            "ASSISTANT",
            "Finance review remains attached to the proposal.",
            "assistant",
        ),
    ):
        token = _token(ctx.run.run_id, suffix)
        child = ctx.intent(
            MEMORY,
            {},
            parent=parent,
            operation="CreateEvent",
            token=token,
        )
        ctx.store.mark_unknown(child.entry_id)
        response = ctx.call(
            "bedrock-agentcore",
            "CreateEvent",
            create_event_params(
                memory_id=memory_id,
                role=role,
                text=text,
                client_token=token,
                event_timestamp=(
                    _live_event_timestamp()
                    if event_clock is None
                    else event_clock()
                ),
            ),
            effect="write",
            charges=memory_event_charges(),
        )
        event = cast("Mapping[str, Any]", response.payload.get("event", {}))
        event_id = str(event.get("eventId", ""))
        if response.outcome["status"] == "ok" and event_id:
            ctx.store.mark_created(
                child.entry_id,
                exact_id=event_id,
                child_id=event_id,
            )


def _verify_for_child(ctx: OwnedResources, entry: Any, kind: Any) -> bool:
    return bool(ctx.verify(entry, kind, phase="setup"))


def _live_event_timestamp() -> datetime:
    return datetime.now(UTC)


def _preflight_memory_budget(budget: PricedBudget) -> Decimal:
    return budget.quote(memory_event_charges(2))


def _live_create_refused() -> list[OperationOutcome]:
    return [
        not_run_operation(
            service="bedrock-agentcore-control",
            operation_name="CreateMemory",
            error_code="create_not_allowed",
            request_validated=False,
            phase="setup",
        ),
        not_run_operation(
            service="bedrock-agentcore-control",
            operation_name="GetMemory",
            error_code="create_not_allowed",
            request_validated=False,
            phase="setup",
        ),
        not_run_operation(
            service="bedrock-agentcore",
            operation_name="CreateEvent",
            error_code="create_not_allowed",
            request_validated=False,
        ),
        not_run_operation(
            service="bedrock-agentcore",
            operation_name="CreateEvent",
            error_code="create_not_allowed",
            request_validated=False,
        ),
        not_run_operation(
            service="bedrock-agentcore",
            operation_name="ListEvents",
            error_code="create_not_allowed",
            request_validated=False,
        ),
        contract_only(
            service="bedrock-agentcore", operation_name="RetrieveMemoryRecords"
        ),
        local_operation(
            service=_STRANDS_INTEGRATION,
            operation_name="NameStrandsIntegration",
        ),
        not_run_operation(
            service="bedrock-agentcore-control",
            operation_name="DeleteMemory",
            error_code="create_not_allowed",
            request_validated=False,
            phase="teardown",
        ),
    ]


def _pricing_refusal() -> list[OperationOutcome]:
    return [
        billable_not_priced(
            service="bedrock-agentcore-control",
            operation_name="CreateMemory",
            phase="setup",
        ),
        billable_not_priced(
            service="bedrock-agentcore-control",
            operation_name="GetMemory",
            phase="setup",
        ),
        *_event_pricing_refusal(),
        billable_not_priced(
            service="bedrock-agentcore", operation_name="ListEvents"
        ),
        contract_only(
            service="bedrock-agentcore", operation_name="RetrieveMemoryRecords"
        ),
        billable_not_priced(
            service="bedrock-agentcore-control",
            operation_name="DeleteMemory",
            phase="teardown",
        ),
    ]


def _event_pricing_refusal() -> list[OperationOutcome]:
    return [
        billable_not_priced(
            service="bedrock-agentcore", operation_name="CreateEvent"
        ),
        billable_not_priced(
            service="bedrock-agentcore", operation_name="CreateEvent"
        ),
    ]


def _missing_live_port() -> list[OperationOutcome]:
    return [
        require_port_not_run(
            service="bedrock-agentcore-control",
            operation_name="CreateMemory",
            phase="setup",
        ),
        require_port_not_run(
            service="bedrock-agentcore-control",
            operation_name="GetMemory",
            phase="setup",
        ),
        require_port_not_run(
            service="bedrock-agentcore", operation_name="CreateEvent"
        ),
        require_port_not_run(
            service="bedrock-agentcore", operation_name="CreateEvent"
        ),
        require_port_not_run(
            service="bedrock-agentcore", operation_name="ListEvents"
        ),
        contract_only(
            service="bedrock-agentcore", operation_name="RetrieveMemoryRecords"
        ),
        require_port_not_run(
            service="bedrock-agentcore-control",
            operation_name="DeleteMemory",
            phase="teardown",
        ),
    ]


def _fixture(
    service: str,
    operation_name: str,
    fixture_id: str,
    effect: str,
    phase: str,
) -> OperationOutcome:
    return fixture_operation(
        service=service,
        operation_name=operation_name,
        fixture_id=fixture_id,
        effect=cast("Effect", effect),
        phase=cast("Phase", phase),
    )


def _default_port(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None,
    session_factory: SessionFactory | None,
    csv_loader: CsvLoader | None,
    boto_port_factory: BotoPortFactory | None = None,
) -> Any | None:
    if port is not None:
        return None
    return resolve_boto_port(
        execution=execution,
        settings=settings,
        policy=policy,
        factory=boto_port_factory,
        session_factory=session_factory,
        csv_loader=csv_loader,
    )


def _memory_payload(*, status: str) -> dict[str, Any]:
    return {
        "arn": _MEMORY_ARN,
        "id": _MEMORY_ID,
        "name": _MEMORY_NAME,
        "eventExpiryDuration": 7,
        "status": status,
        "createdAt": _NOW,
        "updatedAt": _NOW,
        "strategies": [],
    }


def _event_payload(*, role: str, event_id: str) -> dict[str, Any]:
    text = "Finance review required." if role == "USER" else "Noted."
    return {
        "memoryId": _MEMORY_ID,
        "actorId": _ACTOR_ID,
        "sessionId": _SESSION_ID,
        "eventId": event_id,
        "eventTimestamp": _NOW,
        "payload": [
            {"conversational": {"role": role, "content": {"text": text}}}
        ],
        "metadata": {"demo": {"stringValue": _DEMO}},
    }


def _token(run_id: str, suffix: str) -> str:
    return f"{run_id}-{suffix}-{run_id}-{suffix}-{run_id}"[:64]


def _run_tags(run_id: str) -> dict[str, str]:
    return {"run-id": run_id, "project": "aws-ai-stack-workshop"}


def _fixture_tags() -> dict[str, str]:
    return {"run-id": "offline-fixture", "project": "aws-ai-stack-workshop"}
