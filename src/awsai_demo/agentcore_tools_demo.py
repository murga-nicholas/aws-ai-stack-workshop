"""AgentCore tools: bounded sessions and explicit contracts."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from awsai_demo.billing import active_budget_run
from awsai_demo.demo_support import (
    AwsPort,
    AwsStreamPort,
    build_default_boto_port,
    build_result,
    client_method_name,
    contract_only,
    fixture_operation,
    not_run_operation,
    stream_port_operation,
    unsupported_emulator,
    validate_request,
)
from awsai_demo.owned_resources import OwnedResources
from awsai_demo.policy import Charge, ExecutionPolicy, PolicyError
from awsai_demo.runtime import Settings
from awsai_demo.scenario import price_pilot
from awsai_demo.stubs import named_fixture_stream, stubbed_client

if TYPE_CHECKING:
    from awsai_demo.billing import BudgetRun
    from awsai_demo.contracts import (
        DemoResult,
        Effect,
        Execution,
        OperationOutcome,
        Phase,
    )
    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.demo_support import DefaultBotoPort
    from awsai_demo.manifest import ManifestEntry
    from awsai_demo.policy import Reservation

SERVICE = "bedrock-agentcore"
INTERPRETER = "aws.codeinterpreter.v1"
SESSION_ID = "session-0123456789abcdef0123456789"
FIXTURE_RUN = "abcdef123456"
NOW = datetime(2026, 1, 1, tzinfo=UTC)
CAPACITY_SOURCE = (
    "https://docs.aws.amazon.com/bedrock-agentcore/latest/"
    "devguide/bedrock-agentcore-limits.html"
)
# Verified 2026-10-06: fixed session maximum, 2 vCPU / 8 GB.
SESSION_SECONDS = 300
SESSION_CHARGES = (
    Charge(
        "code_interpreter.cpu_seconds", Decimal(2 * SESSION_SECONDS), "second"
    ),
    Charge(
        "code_interpreter.memory_gb_seconds",
        Decimal(8 * SESSION_SECONDS),
        "gb_second",
    ),
)


def interpreter_code() -> str:
    """Build arithmetic from the shared scenario's exact inputs."""
    cost = price_pilot()
    return f"print({cost.staffing_usd!r} + {cost.platform_usd!r})"


def start_request(run_id: str) -> dict[str, Any]:
    """Give an idempotent session a deterministic, private run name."""
    return {
        "codeInterpreterIdentifier": INTERPRETER,
        "name": f"awsai-{run_id}-code",
        "clientToken": f"awsai-{run_id}-interpreter-start",
        "sessionTimeoutSeconds": SESSION_SECONDS,
    }


def invoke_request(session_id: str) -> dict[str, Any]:
    """Execute only the fixed arithmetic, with no files or network."""
    return {
        "codeInterpreterIdentifier": INTERPRETER,
        "sessionId": session_id,
        "name": "executeCode",
        "arguments": {"language": "python", "code": interpreter_code()},
    }


def auxiliary_contracts() -> list[tuple[str, str, dict[str, Any]]]:
    """Browser, Web Search and Payments are request contracts only."""
    manager = (
        "arn:aws:bedrock-agentcore:us-east-1:123456789012:"
        "payment-manager/workshop"
    )
    return [
        (
            SERVICE,
            "StartBrowserSession",
            {
                "browserIdentifier": "aws.browser.v1",
                "sessionTimeoutSeconds": 300,
            },
        ),
        (
            SERVICE,
            "StopBrowserSession",
            {"browserIdentifier": "aws.browser.v1", "sessionId": SESSION_ID},
        ),
        (
            "bedrock-agentcore-control",
            "CreateGatewayTarget",
            {
                "gatewayIdentifier": "gateway-1234567890",
                "name": "web_search_contract",
                "targetConfiguration": {
                    "mcp": {
                        "connector": {
                            "source": {"connectorId": "aws.web-search"}
                        }
                    }
                },
            },
        ),
        (
            SERVICE,
            "CreatePaymentSession",
            {"paymentManagerArn": manager, "expiryTimeInMinutes": 15},
        ),
        (
            SERVICE,
            "ProcessPayment",
            {
                "paymentManagerArn": manager,
                "paymentSessionId": "fixture-session-0123456789abcdef012345",
                "paymentInstrumentId": (
                    "fixture-instrument-0123456789abcdef012345"
                ),
                "paymentType": "MPP",
                "paymentInput": {
                    "mpp": {
                        "version": "1",
                        "wwwAuthenticateHeaders": ["Payment fixture"],
                    }
                },
            },
        ),
    ]


class InterpreterSession:
    """Own a system interpreter session through ambiguous responses."""

    def __init__(self, context: OwnedResources) -> None:
        """Reuse the identity, manifest and reservation ledger."""
        self.context = context

    def start(
        self, prepaid: Reservation | None = None
    ) -> ManifestEntry | None:
        """Reserve the full timeout and write intent before starting."""
        ctx = self.context
        request = start_request(ctx.run.run_id)
        entry = ctx.store.record_intent(
            run_id=ctx.run.run_id,
            account_fingerprint=cast("str", ctx.account_fingerprint),
            region=ctx.run.region,
            service=SERVICE,
            operation="StartCodeInterpreterSession",
            intended_name=request["name"],
            client_token=request["clientToken"],
            tags={},
            naming_scheme="code_interpreter",
            ownership="session",
        )
        ctx.store.mark_unknown(entry.entry_id)
        response = ctx.call(
            SERVICE,
            "StartCodeInterpreterSession",
            request,
            effect="session_start",
            phase="setup",
            charges=SESSION_CHARGES,
            prepaid=prepaid,
        )
        identifier = response.payload.get("sessionId")
        if response.outcome["status"] != "ok" or not identifier:
            return None
        return ctx.store.mark_created(entry.entry_id, exact_id=str(identifier))

    def cleanup_entry(self, entry: ManifestEntry) -> bool:
        """Reconcile by exact name, then stop only the owned session."""
        ctx = self.context
        if entry.status == "deleted":
            return False
        if (
            entry.account_fingerprint != ctx.account_fingerprint
            or entry.region != ctx.run.region
        ):
            ctx.store.mark_unresolved(
                entry.entry_id, message="identity mismatch"
            )
            return True
        current = entry
        if current.exact_id is None:
            params: dict[str, Any] = {
                "codeInterpreterIdentifier": INTERPRETER,
                "maxResults": 100,
            }
            for _ in range(10):
                response = ctx.call(
                    SERVICE,
                    "ListCodeInterpreterSessions",
                    params,
                    effect="read",
                    phase="teardown",
                )
                matches = [
                    item
                    for item in response.payload.get("items", [])
                    if item.get("name") == entry.intended_name
                ]
                if len(matches) == 1:
                    current = ctx.store.mark_created(
                        entry.entry_id, exact_id=str(matches[0]["sessionId"])
                    )
                    break
                token = response.payload.get("nextToken")
                if not token or matches:
                    break
                params["nextToken"] = token
        if current.exact_id is None:
            ctx.store.mark_unresolved(
                entry.entry_id, message="session not reconciled"
            )
            return True
        params = {
            "codeInterpreterIdentifier": INTERPRETER,
            "sessionId": current.exact_id,
        }
        found = ctx.call(
            SERVICE,
            "GetCodeInterpreterSession",
            params,
            effect="read",
            phase="teardown",
        )
        payload = found.payload
        if (
            found.outcome["status"] != "ok"
            or payload.get("name") != current.intended_name
            or payload.get("sessionId") != current.exact_id
            or payload.get("codeInterpreterIdentifier") != INTERPRETER
            or current.client_token
            != start_request(current.run_id)["clientToken"]
        ):
            ctx.store.mark_unresolved(
                entry.entry_id, message="session ownership not established"
            )
            return True
        if payload.get("status") == "TERMINATED":
            ctx.store.mark_deleted(entry.entry_id)
            return False
        # Stop's clientToken is its own idempotency key. The manifest
        # keeps the start token for the ownership check above.
        response = ctx.call(
            SERVICE,
            "StopCodeInterpreterSession",
            {
                **params,
                "clientToken": f"awsai-{current.run_id}-interpreter-stop",
            },
            effect="session_stop",
            phase="teardown",
        )
        if response.outcome["status"] != "ok":
            ctx.store.mark_delete_failed(
                entry.entry_id, message="session stop failed"
            )
            return True
        ctx.store.mark_deleted(entry.entry_id)
        return False


def run_agentcore_tools_demo(
    *,
    execution: Execution = "offline",
    settings: Settings | None = None,
    policy: ExecutionPolicy | None = None,
    port: AwsPort | AwsStreamPort | None = None,
    budget_run: BudgetRun | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Run fixtures or a reserved, explicitly permitted live session."""
    config = settings or Settings()
    limits = policy or ExecutionPolicy()
    operations: list[OperationOutcome] = []
    data: dict[str, Any] = {
        "scenario": asdict(price_pilot()),
        "capacity_source": CAPACITY_SOURCE,
        "capacity_verified": "2026-10-06",
        "session_timeout_seconds": SESSION_SECONDS,
        "stdout": None,
        "web_search_connector": "request shape; connector id is a fixture",
    }
    if execution == "offline":
        _offline(operations, data)
    elif execution == "emulator":
        operations.extend(
            unsupported_emulator(
                service=SERVICE, operation_name=name, phase=phase
            )
            for name, phase in _session_rows()
        )
    elif not config.allow_create:
        operations.extend(
            not_run_operation(
                service=SERVICE,
                operation_name=name,
                phase=phase,
                error_code="create_not_allowed",
                request_validated=False,
            )
            for name, phase in _session_rows()
        )
    else:
        _live(
            config,
            limits,
            port,
            budget_run,
            operations,
            data,
            session_factory,
            csv_loader,
        )
    for service, name, params in auxiliary_contracts():
        if execution == "offline":
            validate_request(
                service=service,
                operation_name=name,
                params=params,
                fixture_id=f"tools-{name}-v1",
            )
            operations.append(
                fixture_operation(
                    service=service,
                    operation_name=name,
                    fixture_id=f"tools-{name}-v1",
                    effect="none",
                )
            )
        else:
            operations.append(
                contract_only(service=service, operation_name=name)
            )
    return build_result(
        demo="agentcore-tools",
        technology="AgentCore managed tools",
        lane="agents",
        lifecycle_refs=[
            "agentcore-code-interpreter",
            "agentcore-browser",
            "agentcore-web-search",
            "agentcore-payments",
        ],
        execution=execution,
        headline="A bounded interpreter session; other tools are contracts.",
        operations=operations,
        settings=config,
        data=data,
        credential_source=data.pop("credential_source", "none"),
        result_error=(
            "cleanup_incomplete",
            "Owned session cleanup is incomplete.",
        )
        if data.get("cleanup_incomplete")
        else None,
    )


def _session_rows() -> tuple[tuple[str, Phase], ...]:
    return (
        ("StartCodeInterpreterSession", "setup"),
        ("InvokeCodeInterpreter", "main"),
        ("StopCodeInterpreterSession", "teardown"),
    )


def _offline(operations: list[OperationOutcome], data: dict[str, Any]) -> None:
    requests = [
        start_request(FIXTURE_RUN),
        invoke_request(SESSION_ID),
        {"codeInterpreterIdentifier": INTERPRETER, "sessionId": SESSION_ID},
    ]
    for (name, phase), params in zip(_session_rows(), requests, strict=True):
        fixture_id = f"tools-{name}-v1"
        if name == "InvokeCodeInterpreter":
            validate_request(
                service=SERVICE,
                operation_name=name,
                params=params,
                fixture_id=fixture_id,
            )
            stream = named_fixture_stream(
                fixture_id=fixture_id,
                events=[
                    {
                        "result": {
                            "content": [{"type": "text", "text": "19680.0"}],
                            "structuredContent": {
                                "stdout": "19680.0\n",
                                "exitCode": 0,
                            },
                        }
                    }
                ],
            )
            data["stdout"] = _read_stream(stream)
        else:
            response = {
                "codeInterpreterIdentifier": INTERPRETER,
                "sessionId": SESSION_ID,
                "createdAt" if phase == "setup" else "lastUpdatedAt": NOW,
            }
            with stubbed_client(SERVICE) as stub:
                method = client_method_name(name)
                stub.add_response(method, response, expected_params=params)
                getattr(stub.client, method)(**params)
        effects: dict[str, Effect] = {
            "StartCodeInterpreterSession": "session_start",
            "InvokeCodeInterpreter": "infer",
            "StopCodeInterpreterSession": "session_stop",
        }
        operations.append(
            fixture_operation(
                service=SERVICE,
                operation_name=name,
                fixture_id=fixture_id,
                effect=effects[name],
                phase=phase,
            )
        )


def _live(
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | AwsStreamPort | None,
    budget_run: BudgetRun | None,
    operations: list[OperationOutcome],
    data: dict[str, Any],
    session_factory: SessionFactory | None,
    csv_loader: CsvLoader | None,
) -> None:
    try:
        run = budget_run or active_budget_run(
            region=settings.region, policy=policy
        )
        prepaid = run.command("agentcore-tools").reserve(
            "interpreter-session",
            SESSION_CHARGES,
        )
    except PolicyError:
        operations.extend(
            not_run_operation(
                service=SERVICE,
                operation_name=name,
                phase=phase,
                error_code="budget_exceeded",
                request_validated=False,
            )
            for name, phase in _session_rows()
        )
        return
    if port is None:
        default = build_default_boto_port(
            execution="live",
            settings=settings,
            policy=policy,
            session_factory=session_factory,
            csv_loader=csv_loader,
        )
        default = cast("DefaultBotoPort", default)
        port = default.port
        data["credential_source"] = default.credential_source
    context = OwnedResources(
        run=run,
        demo="agentcore-tools",
        port=cast("AwsPort", port),
        operations=operations,
    )
    if not context.authenticate():
        return
    session = InterpreterSession(context)
    data["run_id"] = run.run_id
    try:
        entry = session.start(prepaid)
        if entry is not None:
            receipt = context.budget.reserve_amount(
                "invoke-covered-by-session", Decimal(0), kind="service"
            )
            response = stream_port_operation(
                port=cast("AwsStreamPort", port),
                service=SERVICE,
                operation_name="InvokeCodeInterpreter",
                params=invoke_request(cast("str", entry.exact_id)),
                execution="live",
                effect="infer",
                fixture_id="tools-invoke",
                reserved=True,
                endpoint_url=f"https://{SERVICE}.{settings.region}.amazonaws.com",
            )
            response.outcome["reserved_usd"] = float(receipt.amount_usd)
            operations.append(response.outcome)
            data["stdout"] = _read_stream(response.payload.get("stream", ()))
    finally:
        cleanup_results = [
            session.cleanup_entry(entry)
            for entry in reversed(context.store.list_entries())
            if entry.operation == "StartCodeInterpreterSession"
        ]
        data["cleanup_incomplete"] = any(cleanup_results)


def _read_stream(stream: Any) -> str | None:
    output = None
    for event in stream:
        if "result" in event:
            if event["result"].get("isError"):
                message = "Code Interpreter returned a failed result"
                raise ValueError(message)
            output = str(
                event["result"].get("structuredContent", {}).get("stdout", "")
            )
        elif any(key.endswith("Exception") for key in event):
            message = "Code Interpreter returned a service error event"
            raise ValueError(message)
    return output
