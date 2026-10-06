"""Typed result contract for AWS AI Stack workshop demos."""

from __future__ import annotations

from ipaddress import ip_address
from typing import TYPE_CHECKING, Literal, TypedDict, cast
from urllib.parse import SplitResult, urlsplit, urlunsplit

from awsai_demo.redact import redact

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Mode = Literal[
    "local_execution",
    "local_contract",
    "local_emulator",
    "live_model",
    "live_identity",
    "live_service",
    "attempt_failed",
    "not_run",
    "batch",
]
Status = Literal["ok", "paused", "blocked", "error"]
Execution = Literal["offline", "emulator", "live"]
Lane = Literal["models", "agents", "data", "platform", "operations", "local"]
Effect = Literal[
    "none",
    "read",
    "infer",
    "write",
    "delete",
    "session_start",
    "session_stop",
    "export",
    "invoke_named",
]
Phase = Literal["setup", "main", "teardown"]
ExecutionTarget = Literal["fixture", "local", "emulator", "aws", "openai"]
Transport = Literal["none", "loopback", "aws", "external"]
CredentialSource = Literal[
    "none",
    "profile",
    "env",
    "csv_file",
    "default_chain",
]
CostBasis = Literal["estimated", "measured"]
ErrorCode = Literal[
    "missing_configuration",
    "sdk_missing",
    "authorization_denied",
    "maintenance_mode",
    "entitlement_missing",
    "not_supported_by_emulator",
    "emulator_unavailable",
    "contract_only",
    "budget_exceeded",
    "create_not_allowed",
    "invoke_not_allowed",
    "throttled",
    "timeout",
    "validation_failed",
    "model_unavailable",
    "model_access_required",
    "not_supported_by_model",
    "max_tokens_reached",
    "cleanup_incomplete",
    "subscription_required",
]

MODES: frozenset[str] = frozenset(
    {
        "local_execution",
        "local_contract",
        "local_emulator",
        "live_model",
        "live_identity",
        "live_service",
        "attempt_failed",
        "not_run",
        "batch",
    },
)
STATUSES: frozenset[str] = frozenset({"ok", "paused", "blocked", "error"})
EXECUTIONS: frozenset[str] = frozenset({"offline", "emulator", "live"})
LANES: frozenset[str] = frozenset(
    {"models", "agents", "data", "platform", "operations", "local"},
)
EFFECTS: frozenset[str] = frozenset(
    {
        "none",
        "read",
        "infer",
        "write",
        "delete",
        "session_start",
        "session_stop",
        "export",
        "invoke_named",
    },
)
PHASES: frozenset[str] = frozenset({"setup", "main", "teardown"})
EXECUTION_TARGETS: frozenset[str] = frozenset(
    {"fixture", "local", "emulator", "aws", "openai"},
)
TRANSPORTS: frozenset[str] = frozenset(
    {"none", "loopback", "aws", "external"},
)
CREDENTIAL_SOURCES: frozenset[str] = frozenset(
    {"none", "profile", "env", "csv_file", "default_chain"},
)
COST_BASES: frozenset[str] = frozenset({"estimated", "measured"})
ERROR_CODES: frozenset[str] = frozenset(
    {
        "missing_configuration",
        "sdk_missing",
        "authorization_denied",
        "maintenance_mode",
        "entitlement_missing",
        "not_supported_by_emulator",
        "emulator_unavailable",
        "contract_only",
        "budget_exceeded",
        "create_not_allowed",
        "invoke_not_allowed",
        "throttled",
        "timeout",
        "validation_failed",
        "model_unavailable",
        "model_access_required",
        "not_supported_by_model",
        "max_tokens_reached",
        "cleanup_incomplete",
        "subscription_required",
    },
)
_LIVE_MODES: frozenset[Mode] = frozenset(
    ("live_model", "live_service", "live_identity"),
)
_LOCAL_MODE_RANK: dict[Mode, int] = {
    "local_contract": 0,
    "local_emulator": 1,
    "local_execution": 2,
}
_STATUS_RANK: dict[Status, int] = {
    "ok": 0,
    "paused": 1,
    "blocked": 2,
    "error": 3,
}
_EVIDENCE_KEYS: frozenset[str] = frozenset(
    {
        "sdk_invoked",
        "network_attempted",
        "aws_executed",
        "provider",
        "region",
        "fixture_id",
        "requested_model",
        "observed_model",
        "credential_source",
        "packages",
        "emulator",
        "estimated_cost_usd",
        "cost_basis",
    },
)
_EMULATOR_KEYS: frozenset[str] = frozenset(
    {"endpoint", "image", "digest", "edition", "plan"},
)
_OPERATION_KEYS: frozenset[str] = frozenset(
    {
        "service",
        "operation",
        "phase",
        "execution_target",
        "mode",
        "status",
        "effect",
        "transport",
        "endpoint_url",
        "response_received",
        "request_validated",
        "fixture_id",
        "http_status",
        "error_code",
        "duration_ms",
        "usage",
        "reserved_usd",
        "optional",
    },
)


class ContractError(ValueError):
    """Raised when a demo result violates the public contract."""


class Usage(TypedDict, total=False):
    """Optional token counts for a model operation."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class EmulatorEvidence(TypedDict, total=False):
    """LocalStack endpoint and image provenance."""

    endpoint: str
    image: str | None
    digest: str | None
    edition: str | None
    plan: str | None


class Evidence(TypedDict):
    """Allowlisted provenance fields for a demo result."""

    sdk_invoked: bool
    network_attempted: bool
    aws_executed: bool
    provider: str | None
    region: str | None
    fixture_id: str | None
    requested_model: str | None
    observed_model: str | None
    credential_source: CredentialSource
    packages: dict[str, str]
    emulator: EmulatorEvidence | None
    estimated_cost_usd: float | None
    cost_basis: CostBasis | None


class ResultError(TypedDict):
    """Stable error vocabulary plus a public diagnostic."""

    code: ErrorCode
    message: str


class OperationOutcome(TypedDict):
    """One external or loopback protocol operation."""

    service: str
    operation: str
    phase: Phase
    execution_target: ExecutionTarget
    mode: Mode
    status: Status
    effect: Effect
    transport: Transport
    endpoint_url: str | None
    response_received: bool
    request_validated: bool
    fixture_id: str | None
    http_status: int | None
    error_code: str | None
    duration_ms: int | None
    usage: Usage | None
    reserved_usd: float | None
    optional: bool


class DemoResult(TypedDict):
    """Top-level result envelope returned by every demo."""

    schema_version: Literal[1]
    demo: str
    technology: str
    lane: Lane
    lifecycle_refs: list[str]
    requested_execution: Execution
    mode: Mode
    status: Status
    headline: str
    operations: list[OperationOutcome]
    evidence: Evidence
    data: object | None
    error: ResultError | None
    next_steps: list[str]
    children: list[DemoResult]


def operation(
    *,
    service: str,
    operation: str,
    phase: Phase = "main",
    execution_target: ExecutionTarget = "fixture",
    mode: Mode = "local_contract",
    status: Status = "ok",
    effect: Effect = "none",
    transport: Transport = "none",
    endpoint_url: str | None = None,
    response_received: bool = True,
    request_validated: bool = True,
    fixture_id: str | None = None,
    http_status: int | None = None,
    error_code: str | None = None,
    duration_ms: int | None = None,
    usage: Usage | None = None,
    reserved_usd: float | None = None,
    optional: bool = False,
) -> OperationOutcome:
    """Build one operation outcome and validate enum-like fields."""
    _ensure_choice("phase", phase, PHASES)
    _ensure_choice("execution_target", execution_target, EXECUTION_TARGETS)
    _ensure_choice("mode", mode, MODES)
    _ensure_choice("status", status, STATUSES)
    _ensure_choice("effect", effect, EFFECTS)
    _ensure_choice("transport", transport, TRANSPORTS)
    endpoint = _sanitize_endpoint(endpoint_url)
    outcome: OperationOutcome = {
        "service": service,
        "operation": operation,
        "phase": phase,
        "execution_target": execution_target,
        "mode": mode,
        "status": status,
        "effect": effect,
        "transport": transport,
        "endpoint_url": endpoint,
        "response_received": response_received,
        "request_validated": request_validated,
        "fixture_id": fixture_id,
        "http_status": http_status,
        "error_code": error_code,
        "duration_ms": duration_ms,
        "usage": usage,
        "reserved_usd": reserved_usd,
        "optional": optional,
    }
    _validate_operation_provenance(outcome)
    return cast("OperationOutcome", redact(outcome))


def evidence(
    *,
    sdk_invoked: bool = False,
    network_attempted: bool = False,
    aws_executed: bool = False,
    provider: str | None = None,
    region: str | None = None,
    fixture_id: str | None = None,
    requested_model: str | None = None,
    observed_model: str | None = None,
    credential_source: CredentialSource = "none",
    packages: Mapping[str, str] | None = None,
    emulator: EmulatorEvidence | None = None,
    estimated_cost_usd: float | None = None,
    cost_basis: CostBasis | None = None,
) -> Evidence:
    """Build allowlisted evidence with safe defaults."""
    _ensure_choice("credential_source", credential_source, CREDENTIAL_SOURCES)
    if cost_basis is not None:
        _ensure_choice("cost_basis", cost_basis, COST_BASES)
    emulator_copy = (
        cast("EmulatorEvidence", dict(emulator))
        if emulator is not None
        else None
    )
    payload: Evidence = {
        "sdk_invoked": sdk_invoked,
        "network_attempted": network_attempted,
        "aws_executed": aws_executed,
        "provider": provider,
        "region": region,
        "fixture_id": fixture_id,
        "requested_model": requested_model,
        "observed_model": observed_model,
        "credential_source": credential_source,
        "packages": dict(packages or {}),
        "emulator": emulator_copy,
        "estimated_cost_usd": estimated_cost_usd,
        "cost_basis": cost_basis,
    }
    _validate_evidence_contract(payload)
    return cast("Evidence", redact(payload))


def error(code: ErrorCode, message: str) -> ResultError:
    """Build a stable result error."""
    _ensure_choice("code", code, ERROR_CODES)
    return {"code": code, "message": cast("str", redact(message))}


def result(
    *,
    demo: str,
    technology: str,
    lane: Lane,
    lifecycle_refs: Sequence[str],
    requested_execution: Execution,
    headline: str,
    operations: Sequence[OperationOutcome] | None = None,
    evidence: Evidence | None = None,
    data: object | None = None,
    error: ResultError | None = None,
    next_steps: Sequence[str] | None = None,
    children: Sequence[DemoResult] | None = None,
    mode: Mode | None = None,
    status: Status | None = None,
) -> DemoResult:
    """Build and validate a redacted `DemoResult`."""
    _ensure_choice("lane", lane, LANES)
    _ensure_choice("requested_execution", requested_execution, EXECUTIONS)
    refs = list(lifecycle_refs)
    if not refs:
        _raise_contract("lifecycle_refs must contain at least one id")
    operations_list = list(operations or [])
    children_list = list(children or [])
    steps = list(next_steps or [])
    if any(
        item["error_code"] == "model_access_required"
        for item in operations_list
    ):
        access_step = (
            "Submit the Anthropic model use case details in the Amazon "
            "Bedrock console for this AWS account, then retry after "
            "access becomes available."
        )
        if error is None or error["code"] == "model_unavailable":
            error = {"code": "model_access_required", "message": access_step}
        if access_step not in steps:
            steps.append(access_step)
    computed_mode = mode or aggregate_mode(operations_list, children_list)
    _ensure_choice("mode", computed_mode, MODES)
    computed_status = status or _default_status(
        operations_list,
        children_list,
        error,
        requested_execution,
    )
    _ensure_choice("status", computed_status, STATUSES)
    evidence_payload = evidence or _evidence_from_operations(operations_list)
    _validate_result_shape(
        requested_execution,
        computed_mode,
        operations_list,
        children_list,
        evidence_payload,
    )
    payload: DemoResult = {
        "schema_version": 1,
        "demo": demo,
        "technology": technology,
        "lane": lane,
        "lifecycle_refs": refs,
        "requested_execution": requested_execution,
        "mode": computed_mode,
        "status": computed_status,
        "headline": headline,
        "operations": operations_list,
        "evidence": evidence_payload,
        "data": data,
        "error": error,
        "next_steps": steps,
        "children": children_list,
    }
    return cast("DemoResult", redact(payload))


def not_run(
    *,
    demo: str,
    technology: str,
    lane: Lane,
    lifecycle_refs: Sequence[str],
    requested_execution: Execution,
    headline: str,
    code: ErrorCode = "contract_only",
    message: str = "Demo did not run.",
    next_steps: Sequence[str] | None = None,
    data: object | None = None,
) -> DemoResult:
    """Build a blocked `not_run` result for an unmet prerequisite."""
    return result(
        demo=demo,
        technology=technology,
        lane=lane,
        lifecycle_refs=lifecycle_refs,
        requested_execution=requested_execution,
        mode="not_run",
        status="blocked",
        headline=headline,
        operations=[],
        evidence=evidence(),
        data=data,
        error=error_result(code, message),
        next_steps=next_steps,
    )


def missing_configuration(
    *,
    demo: str,
    technology: str,
    lane: Lane,
    lifecycle_refs: Sequence[str],
    requested_execution: Execution,
    headline: str,
    message: str,
    next_steps: Sequence[str] | None = None,
) -> DemoResult:
    """Build a blocked `not_run` result for missing configuration."""
    return not_run(
        demo=demo,
        technology=technology,
        lane=lane,
        lifecycle_refs=lifecycle_refs,
        requested_execution=requested_execution,
        headline=headline,
        code="missing_configuration",
        message=message,
        next_steps=next_steps,
    )


def error_result(code: ErrorCode, message: str) -> ResultError:
    """Alias with a name that avoids shadowing result call sites."""
    return error(code, message)


def aggregate_status(items: Sequence[Mapping[str, object]]) -> Status:
    """Return the worst status using `ok < paused < blocked < error`."""
    worst: Status = "ok"
    for item in items:
        raw_status = item.get("status", "ok")
        _ensure_choice("status", raw_status, STATUSES)
        item_status = cast("Status", raw_status)
        if _STATUS_RANK[item_status] > _STATUS_RANK[worst]:
            worst = item_status
    return worst


def aggregate_mode(
    operations: Sequence[OperationOutcome],
    children: Sequence[DemoResult] | None = None,
) -> Mode:
    """Derive the top-level mode from operations or children."""
    if children:
        return "batch"
    if not operations:
        return "local_execution"
    modes = [item["mode"] for item in operations]
    if "live_model" in modes:
        return "live_model"
    if "live_service" in modes:
        return "live_service"
    if "live_identity" in modes:
        return "live_identity"
    local_modes = [item for item in modes if item in _LOCAL_MODE_RANK]
    if local_modes:
        return min(local_modes, key=lambda item: _LOCAL_MODE_RANK[item])
    if "attempt_failed" in modes:
        return "attempt_failed"
    return "not_run"


def _default_status(
    operations: Sequence[OperationOutcome],
    children: Sequence[DemoResult],
    result_error: ResultError | None,
    requested_execution: Execution,
) -> Status:
    if children:
        return aggregate_status(children)
    if operations:
        relevant = [item for item in operations if not _neutral_skip(item)]
        if result_error is not None or not any(
            _ran_in_lane(item, requested_execution) for item in operations
        ):
            return aggregate_status([*relevant, {"status": "blocked"}])
        return aggregate_status(relevant)
    if result_error is not None:
        return "error"
    return "ok" if requested_execution == "offline" else "blocked"


def _neutral_skip(item: OperationOutcome) -> bool:
    """Ignore deliberate omissions, preserving real failures."""
    return (
        item["mode"] == "not_run"
        and item["status"] == "blocked"
        and (
            item["error_code"]
            in {
                "contract_only",
                "create_not_allowed",
                "invoke_not_allowed",
                "not_supported_by_emulator",
                "not_supported_by_model",
            }
            or (
                item["error_code"] == "missing_configuration"
                and item["optional"]
            )
        )
    )


def _ran_in_lane(item: OperationOutcome, execution: Execution) -> bool:
    """Require an operation in the requested execution lane."""
    if item["mode"] == "not_run":
        return False
    if execution == "offline":
        return True
    targets = {"aws", "openai"} if execution == "live" else {"emulator"}
    return item["execution_target"] in targets


def _evidence_from_operations(
    operations: Sequence[OperationOutcome],
) -> Evidence:
    return evidence(
        sdk_invoked=any(
            item["mode"] != "not_run" or item["request_validated"]
            for item in operations
        ),
        network_attempted=any(
            item["transport"] in {"aws", "external", "loopback"}
            for item in operations
        ),
        aws_executed=any(
            item["execution_target"] == "aws" and item["response_received"]
            for item in operations
        ),
        fixture_id=next(
            (item["fixture_id"] for item in operations if item["fixture_id"]),
            None,
        ),
    )


def _validate_result_shape(
    requested_execution: Execution,
    mode: Mode,
    operations: Sequence[OperationOutcome],
    children: Sequence[DemoResult],
    evidence_payload: Evidence,
) -> None:
    _validate_evidence_contract(evidence_payload)
    for item in operations:
        _validate_operation_contract(item)
    if mode == "batch":
        if not children:
            _raise_contract("batch results must contain children")
        return
    if children:
        _raise_contract("children are only allowed when mode is batch")
    if mode == "not_run" and any(
        _operation_was_attempted(item) for item in operations
    ):
        _raise_contract(
            "not_run results must not contain attempted operations"
        )
    if operations and mode != aggregate_mode(operations):
        message = f"result mode {mode} contradicts operation provenance"
        _raise_contract(message)
    if requested_execution == "live":
        _validate_requested_live_has_no_fixture_replay(mode, evidence_payload)
    if mode in _LIVE_MODES:
        _validate_live_mode(mode, operations, evidence_payload)
    elif evidence_payload["aws_executed"]:
        _raise_contract("non-live modes must not claim aws_executed=True")
    if mode == "local_contract" and not _has_fixture(
        operations,
        evidence_payload,
    ):
        _raise_contract("local_contract results must name a fixture_id")
    if mode == "local_emulator" and not _has_loopback_emulator(
        evidence_payload
    ):
        message = "local_emulator requires a loopback emulator endpoint"
        _raise_contract(message)
    if mode == "attempt_failed" and not any(
        not item["response_received"] for item in operations
    ):
        message = "attempt_failed requires an operation without a response"
        _raise_contract(message)


def _validate_live_mode(
    mode: Mode,
    operations: Sequence[OperationOutcome],
    evidence_payload: Evidence,
) -> None:
    if not evidence_payload["network_attempted"]:
        message = f"{mode} requires network_attempted=True"
        _raise_contract(message)
    if not any(_establishes_live_mode(item, mode) for item in operations):
        message = f"{mode} requires a matching answered live operation"
        _raise_contract(message)


def _validate_operation_contract(item: OperationOutcome) -> None:
    keys = frozenset(item)
    if keys != _OPERATION_KEYS:
        _raise_key_contract("operation", keys, _OPERATION_KEYS)
    _validate_operation_provenance(item)


def _validate_operation_provenance(item: OperationOutcome) -> None:
    mode = item["mode"]
    if mode == "batch":
        _raise_contract("batch is not an operation mode")
    if mode == "not_run":
        _validate_not_run_operation(item)
        return
    if mode == "local_contract":
        _validate_target_transport(item, "fixture", "none")
    elif mode == "local_execution":
        _validate_local_execution_operation(item)
    elif mode == "local_emulator":
        _validate_local_emulator_operation(item)
    elif mode in _LIVE_MODES:
        _validate_live_operation(item)
    else:
        _validate_attempt_failed_operation(item)


def _validate_target_transport(
    item: OperationOutcome,
    target: ExecutionTarget,
    transport: Transport,
) -> None:
    if item["execution_target"] != target or item["transport"] != transport:
        message = f"{item['mode']} requires {target}/{transport} provenance"
        _raise_contract(message)


def _validate_local_execution_operation(item: OperationOutcome) -> None:
    if item["execution_target"] != "local":
        _raise_contract("local_execution requires local execution_target")
    if item["transport"] not in {"none", "loopback"}:
        _raise_contract("local_execution requires none or loopback transport")
    if item["fixture_id"] is not None:
        _raise_contract("local_execution operations must not name fixtures")
    if item["transport"] == "loopback" and not _has_loopback_endpoint(item):
        _raise_contract(
            "local_execution loopback operations need loopback endpoint"
        )


def _validate_local_emulator_operation(item: OperationOutcome) -> None:
    _validate_target_transport(item, "emulator", "loopback")
    if not _has_loopback_endpoint(item):
        _raise_contract("local_emulator operations need loopback endpoint")


def _validate_live_operation(item: OperationOutcome) -> None:
    if not item["response_received"]:
        _raise_contract("live operation modes require a response")
    if not _has_non_loopback_endpoint(item):
        _raise_contract("live operation modes require non-loopback endpoint")
    if not _has_live_transport_and_target(item):
        _raise_contract(
            "live operation modes require live transport and target"
        )
    if item["mode"] == "live_identity" and item["service"] != "sts":
        _raise_contract("live_identity requires sts service")
    if (
        item["mode"] == "live_identity"
        and item["operation"] != "GetCallerIdentity"
    ):
        _raise_contract("live_identity requires GetCallerIdentity operation")
    if item["mode"] in {"live_identity", "live_model"}:
        capped = (
            item["mode"] == "live_model"
            and item["error_code"] == "max_tokens_reached"
        )
        if capped and item["status"] != "blocked":
            _raise_contract("max_tokens_reached requires status blocked")
        elif not capped and item["status"] != "ok":
            _raise_contract("live identity/model operations must be ok")
        elif not capped and item["error_code"] is not None:
            _raise_contract("live identity/model operations must not error")
        if item["http_status"] is not None and item["http_status"] >= 400:
            _raise_contract(
                "live identity/model operations must have successful HTTP",
            )


def _validate_not_run_operation(item: OperationOutcome) -> None:
    if item["status"] == "ok":
        _raise_contract("not_run operations must not be ok")
    if item["effect"] != "none":
        _raise_contract("not_run operations must not claim an effect")
    if item["transport"] != "none":
        _raise_contract("not_run operations must not attempt transport")
    if item["endpoint_url"] is not None:
        _raise_contract("not_run operations must not name an endpoint")
    if item["response_received"]:
        _raise_contract("not_run operations must not receive a response")
    if item["fixture_id"] is not None:
        _raise_contract("not_run operations must not claim a fixture")
    if item["http_status"] is not None:
        _raise_contract("not_run operations must not claim an HTTP status")
    if item["usage"] is not None:
        _raise_contract("not_run operations must not claim usage")
    if item["reserved_usd"] is not None:
        _raise_contract("not_run operations must not claim a reservation")


def _validate_attempt_failed_operation(item: OperationOutcome) -> None:
    if item["response_received"]:
        _raise_contract("attempt_failed operations must not have a response")
    if item["transport"] == "none" or item["execution_target"] == "fixture":
        _raise_contract("attempt_failed requires a dispatched real call")


def _operation_was_attempted(item: OperationOutcome) -> bool:
    return item["mode"] != "not_run"


def _validate_evidence_contract(evidence_payload: Evidence) -> None:
    keys = frozenset(evidence_payload)
    if keys != _EVIDENCE_KEYS:
        _raise_key_contract("evidence", keys, _EVIDENCE_KEYS)
    emulator = evidence_payload["emulator"]
    if emulator is not None:
        emulator_keys = frozenset(emulator)
        if not emulator_keys.issubset(_EMULATOR_KEYS):
            _raise_key_contract(
                "emulator evidence", emulator_keys, _EMULATOR_KEYS
            )


def _validate_requested_live_has_no_fixture_replay(
    mode: Mode,
    evidence_payload: Evidence,
) -> None:
    if mode == "local_contract" or evidence_payload["fixture_id"] is not None:
        _raise_contract("live requests must not replay fixtures")


def _raise_key_contract(
    label: str,
    actual: frozenset[str],
    allowed: frozenset[str],
) -> None:
    unknown = sorted(actual - allowed)
    if unknown:
        message = f"{label} contains unknown keys: {', '.join(unknown)}"
        _raise_contract(message)
    missing = sorted(allowed - actual)
    message = f"{label} is missing keys: {', '.join(missing)}"
    _raise_contract(message)


def _has_fixture(
    operations: Sequence[OperationOutcome],
    evidence_payload: Evidence,
) -> bool:
    return evidence_payload["fixture_id"] is not None or any(
        item["fixture_id"] for item in operations
    )


def _has_loopback_emulator(evidence_payload: Evidence) -> bool:
    emulator = evidence_payload["emulator"]
    if emulator is None:
        return False
    endpoint = emulator.get("endpoint")
    return isinstance(endpoint, str) and _is_loopback_endpoint(endpoint)


def _establishes_live_mode(item: OperationOutcome, mode: Mode) -> bool:
    return (
        item["mode"] == mode
        and item["response_received"]
        and _has_non_loopback_endpoint(item)
        and _has_live_transport_and_target(item)
    )


def _has_non_loopback_endpoint(item: OperationOutcome) -> bool:
    endpoint = item["endpoint_url"]
    return isinstance(endpoint, str) and _is_non_loopback_endpoint(endpoint)


def _has_loopback_endpoint(item: OperationOutcome) -> bool:
    endpoint = item["endpoint_url"]
    return isinstance(endpoint, str) and _is_loopback_endpoint(endpoint)


def _has_live_transport_and_target(item: OperationOutcome) -> bool:
    target = item["execution_target"]
    transport = item["transport"]
    if target == "aws":
        return transport == "aws"
    if target == "openai":
        return transport == "external"
    return False


def _is_non_loopback_endpoint(endpoint_url: str) -> bool:
    parsed = urlsplit(endpoint_url)
    host = parsed.hostname
    if host is None or _sanitized_netloc(parsed) is None:
        return False
    return not _is_loopback_host(host.lower())


def _is_loopback_endpoint(endpoint_url: str) -> bool:
    parsed = urlsplit(endpoint_url)
    host = parsed.hostname
    if host is None or _sanitized_netloc(parsed) is None:
        return False
    return _is_loopback_host(host.lower())


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _sanitize_endpoint(endpoint_url: str | None) -> str | None:
    if endpoint_url is None:
        return None
    parsed = urlsplit(endpoint_url)
    if not parsed.scheme or parsed.hostname is None:
        return cast("str", redact(endpoint_url))
    netloc = _sanitized_netloc(parsed)
    if netloc is None:
        netloc = _host_without_port(parsed)
    clean = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    return cast("str", redact(clean))


def _host_without_port(parsed: SplitResult) -> str:
    host = cast("str", parsed.hostname)
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _sanitized_netloc(parsed: SplitResult) -> str | None:
    host = cast("str", parsed.hostname)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is None:
        return host
    return f"{host}:{port}"


def _ensure_choice(name: str, value: object, choices: frozenset[str]) -> None:
    if value not in choices:
        expected = ", ".join(sorted(choices))
        message = f"{name} must be one of: {expected}"
        _raise_contract(message)


def _raise_contract(message: str) -> None:
    raise ContractError(message)
