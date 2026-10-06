"""Amazon Bedrock Guardrails demo.

Technology: Amazon Bedrock Guardrails.
Lane: models.
Lifecycle refs: guardrails, comprehend-prompt-safety.
Run:
    uv run awsai-demo guardrails
    uv run awsai-demo guardrails --execution emulator
    uv run awsai-demo guardrails --execution live --allow-create
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
    fixture_operation,
    not_run_operation,
    port_operation,
    require_port_not_run,
    resolve_boto_port,
    unsupported_emulator,
)
from awsai_demo.owned_resources import GUARDRAIL, OwnedResources
from awsai_demo.policy import (
    Charge,
    ExecutionPolicy,
    PolicyError,
    ReservationUnavailable,
)
from awsai_demo.stubs import stubbed_client

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping, Sequence

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

_DEMO = "guardrails"
_TECHNOLOGY = "Amazon Bedrock Guardrails"
_REFS = ["guardrails", "comprehend-prompt-safety"]
_CREATE_FIXTURE = "guardrails-create-v1"
_VERSION_FIXTURE = "guardrails-version-v1"
_INPUT_FIXTURE = "guardrails-apply-input-email-v1"
_OUTPUT_FIXTURE = "guardrails-apply-output-safe-v1"
_DELETE_FIXTURE = "guardrails-delete-v1"
_GET_FIXTURE = "guardrails-get-policy-v1"
_GUARDRAIL_NAME = "awsai-demo-guard"
_GUARDRAIL_ID = "gr1234567890"
_GUARDRAIL_VERSION = "1"
_GUARDRAIL_ARN = (
    "arn:aws:bedrock:us-east-1:123456789012:guardrail/" + _GUARDRAIL_ID
)
_CLIENT_TOKEN = "guardrails-demo-token-012345678901"  # noqa: S105
_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_INPUT_TEXT = "Please email the proposal to alice@example.com."
_OUTPUT_TEXT = "The proposal is approved for internal review."
_KNOWN_POLICY_FEATURES = (
    "guardrails.content_filter",
    "guardrails.denied_topic",
    "guardrails.sensitive_information",
)
_GUARDRAIL_POLL_SECONDS = 0.5


def run_guardrails_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
    boto_port_factory: BotoPortFactory | None = None,
    clock: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> DemoResult:
    """Run the Guardrails demo in a lane-safe way."""
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
            clock=clock,
            sleeper=sleeper,
        )
    return build_result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="models",
        lifecycle_refs=_REFS,
        execution=execution,
        headline="Guardrail input and output actions are recorded per call.",
        operations=operations,
        settings=settings,
        data=data,
        credential_source=credential_source,
        result_error=_cleanup_result_error(data),
    )


def create_guardrail_params(
    *,
    name: str = _GUARDRAIL_NAME,
    client_token: str = _CLIENT_TOKEN,
    tags: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Return the CreateGuardrail request used by the demo."""
    return {
        "name": name,
        "blockedInputMessaging": "Blocked by the workshop guardrail.",
        "blockedOutputsMessaging": "Output blocked by the workshop guardrail.",
        "topicPolicyConfig": {
            "topicsConfig": [
                {
                    "name": "NoProcurementFraud",
                    "definition": "Requests to bypass procurement controls.",
                    "examples": ["Hide this vendor payment from review."],
                    "type": "DENY",
                    "inputAction": "BLOCK",
                    "outputAction": "BLOCK",
                    "inputEnabled": True,
                    "outputEnabled": True,
                }
            ]
        },
        "contentPolicyConfig": {
            "filtersConfig": [
                {
                    "type": "PROMPT_ATTACK",
                    "inputStrength": "HIGH",
                    "outputStrength": "NONE",
                    "inputAction": "BLOCK",
                    "outputAction": "NONE",
                    "inputEnabled": True,
                    "outputEnabled": True,
                }
            ]
        },
        "sensitiveInformationPolicyConfig": {
            "piiEntitiesConfig": [
                {
                    "type": "EMAIL",
                    "action": "ANONYMIZE",
                    "inputAction": "ANONYMIZE",
                    "outputAction": "ANONYMIZE",
                    "inputEnabled": True,
                    "outputEnabled": True,
                }
            ]
        },
        "tags": list(tags) if tags is not None else _fixture_tags(),
        "clientRequestToken": client_token,
    }


def create_guardrail_version_params(
    guardrail_id: str = _GUARDRAIL_ID,
    *,
    client_token: str = _CLIENT_TOKEN,
) -> dict[str, str]:
    """Return the CreateGuardrailVersion request."""
    return {
        "guardrailIdentifier": guardrail_id,
        "description": "Workshop version for deterministic apply fixtures.",
        "clientRequestToken": client_token,
    }


def delete_guardrail_params(
    guardrail_id: str = _GUARDRAIL_ID,
) -> dict[str, str]:
    """Return the DeleteGuardrail request."""
    return {"guardrailIdentifier": guardrail_id}


def get_guardrail_params(
    guardrail_id: str,
    guardrail_version: str,
) -> dict[str, str]:
    """Return the GetGuardrail request for named live resources."""
    return {
        "guardrailIdentifier": guardrail_id,
        "guardrailVersion": guardrail_version,
    }


def apply_guardrail_params(
    *,
    guardrail_id: str = _GUARDRAIL_ID,
    guardrail_version: str = _GUARDRAIL_VERSION,
    source: str,
    text: str,
) -> dict[str, Any]:
    """Return an ApplyGuardrail request for a single text block."""
    return {
        "guardrailIdentifier": guardrail_id,
        "guardrailVersion": guardrail_version,
        "source": source,
        "content": [{"text": {"text": text}}],
        "outputScope": "FULL",
    }


def guardrail_apply_charges(
    params: Mapping[str, Any],
    *,
    features: Sequence[str] = _KNOWN_POLICY_FEATURES,
) -> tuple[Charge, ...]:
    """Return conservative charges for one ApplyGuardrail request."""
    units = _text_units(params)
    if not features:
        message = "guardrail policy features are unknown"
        raise ReservationUnavailable(message)
    return tuple(Charge(feature, units, "text_unit") for feature in features)


def _run_offline() -> tuple[list[OperationOutcome], dict[str, Any]]:
    operations: list[OperationOutcome] = []
    create_params = create_guardrail_params()
    version_params = create_guardrail_version_params()
    input_params = apply_guardrail_params(source="INPUT", text=_INPUT_TEXT)
    output_params = apply_guardrail_params(source="OUTPUT", text=_OUTPUT_TEXT)
    with stubbed_client("bedrock") as stubber:
        client = stubber.client
        stubber.add_response(
            "create_guardrail",
            _create_response(),
            expected_params=create_params,
        )
        create_response = client.create_guardrail(**create_params)
        operations.append(
            _fixture(
                "bedrock", "CreateGuardrail", _CREATE_FIXTURE, "write", "setup"
            )
        )
        stubber.add_response(
            "create_guardrail_version",
            {"guardrailId": _GUARDRAIL_ID, "version": _GUARDRAIL_VERSION},
            expected_params=version_params,
        )
        version_response = client.create_guardrail_version(**version_params)
        operations.append(
            _fixture(
                "bedrock",
                "CreateGuardrailVersion",
                _VERSION_FIXTURE,
                "write",
                "setup",
            )
        )
    with stubbed_client("bedrock-runtime") as stubber:
        client = stubber.client
        stubber.add_response(
            "apply_guardrail",
            _apply_response("GUARDRAIL_INTERVENED"),
            expected_params=input_params,
        )
        input_response, input_action = _apply_guardrail_request(
            client, input_params
        )
        operations.append(
            _fixture(
                "bedrock-runtime",
                "ApplyGuardrail",
                _INPUT_FIXTURE,
                "infer",
                "main",
            )
        )
        stubber.add_response(
            "apply_guardrail",
            _apply_response("NONE"),
            expected_params=output_params,
        )
        output_response, output_action = _apply_guardrail_request(
            client, output_params
        )
        operations.append(
            _fixture(
                "bedrock-runtime",
                "ApplyGuardrail",
                _OUTPUT_FIXTURE,
                "infer",
                "main",
            )
        )
    with stubbed_client("bedrock") as stubber:
        client = stubber.client
        stubber.add_response(
            "delete_guardrail",
            {},
            expected_params=delete_guardrail_params(),
        )
        client.delete_guardrail(**delete_guardrail_params())
        operations.append(
            _fixture(
                "bedrock",
                "DeleteGuardrail",
                _DELETE_FIXTURE,
                "delete",
                "teardown",
            )
        )
    data = {
        "guardrail_id": create_response["guardrailId"],
        "guardrail_version": version_response["version"],
        "input_action": input_action,
        "output_action": output_action,
        "guardrail": {
            "input_action": input_action,
            "output_action": output_action,
        },
        "input_usage": input_response.get("usage", {}),
        "output_usage": output_response.get("usage", {}),
        "policy_features": list(_KNOWN_POLICY_FEATURES),
    }
    return operations, data


def _run_emulator(
    *,
    configured: bool,
) -> tuple[list[OperationOutcome], dict[str, Any]]:
    builder = unsupported_emulator if configured else require_port_not_run
    return [
        builder(
            service="bedrock", operation_name="CreateGuardrail", phase="setup"
        ),
        builder(
            service="bedrock",
            operation_name="CreateGuardrailVersion",
            phase="setup",
        ),
        builder(service="bedrock-runtime", operation_name="ApplyGuardrail"),
        builder(service="bedrock-runtime", operation_name="ApplyGuardrail"),
        builder(
            service="bedrock",
            operation_name="DeleteGuardrail",
            phase="teardown",
        ),
    ], {"emulator": "not_supported" if configured else "missing_token"}


def _run_live(
    *,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None,
    session_factory: SessionFactory | None,
    csv_loader: CsvLoader | None,
    boto_port_factory: BotoPortFactory | None = None,
    clock: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> tuple[list[OperationOutcome], dict[str, Any], CredentialSource | None]:
    if settings.allow_create:
        return _run_live_create(
            settings=settings,
            policy=policy,
            port=port,
            session_factory=session_factory,
            csv_loader=csv_loader,
            boto_port_factory=boto_port_factory,
            clock=clock,
            sleeper=sleeper,
        )
    return _run_live_named(
        settings=settings,
        policy=policy,
        port=port,
        session_factory=session_factory,
        csv_loader=csv_loader,
        boto_port_factory=boto_port_factory,
    )


def _run_live_create(
    *,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None,
    session_factory: SessionFactory | None,
    csv_loader: CsvLoader | None,
    boto_port_factory: BotoPortFactory | None = None,
    clock: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> tuple[list[OperationOutcome], dict[str, Any], CredentialSource | None]:
    operations: list[OperationOutcome] = []
    data: dict[str, Any] = {"mode": "create"}
    resolved_clock, resolved_sleep = _resolve_poll_ports(clock, sleeper)
    try:
        run = active_budget_run(region=settings.region, policy=policy)
        _preflight_guardrail_budget(run.command(_DEMO))
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
    cleanup_incomplete = True
    try:
        if not ctx.authenticate():
            data["identity"] = "unverified"
            return operations, data, credential_source
        name = f"awsai-{run.run_id}-guard"
        token = _token(run.run_id, "guard")
        create_params = create_guardrail_params(
            name=name,
            client_token=token,
            tags=_run_tags(run.run_id),
        )
        entry = ctx.create(GUARDRAIL, create_params)
        if entry is None:
            data["created"] = False
            return operations, data, credential_source
        guardrail_id = str(entry.exact_id)
        if not _verify_for_child(ctx, entry, GUARDRAIL):
            data["ownership_verified"] = False
            return operations, data, credential_source
        draft_status, draft_error = _await_guardrail_ready(
            ctx,
            guardrail_id,
            version=None,
            clock=resolved_clock,
            sleeper=resolved_sleep,
        )
        if draft_status != "READY":
            _record_readiness(data, draft_status, draft_error)
            return operations, data, credential_source
        # CreateGuardrailVersion is a different operation. Reusing the
        # create clientRequestToken is a ConflictException, not a retry.
        version = _create_live_version(
            ctx,
            entry,
            guardrail_id,
            _token(run.run_id, "guard-v1"),
        )
        if version is None:
            data["versioned"] = False
            return operations, data, credential_source
        # CreateGuardrailVersion is HTTP 202. Poll that version: its
        # status stays VERSIONING until GetGuardrail reports READY.
        version_status, version_error = _await_guardrail_ready(
            ctx,
            guardrail_id,
            version=version,
            clock=resolved_clock,
            sleeper=resolved_sleep,
        )
        if version_status != "READY":
            _record_readiness(data, version_status, version_error)
            return operations, data, credential_source
        data.update(
            _apply_live_pair(
                ctx,
                guardrail_id=guardrail_id,
                guardrail_version=version,
                features=_policy_features_from_create(create_params),
            )
        )
        data["guardrail_id"] = guardrail_id
        data["guardrail_version"] = version
        data["guardrail"] = {
            "input_action": data.get("input_action"),
            "output_action": data.get("output_action"),
        }
        return operations, data, credential_source  # noqa: TRY300
    except PolicyError as exc:
        data["policy_error"] = exc.__class__.__name__
        operations.extend(_apply_pricing_refusal())
        return operations, data, credential_source
    finally:
        cleanup_incomplete = ctx.cleanup()
        data["cleanup_incomplete"] = cleanup_incomplete


def _run_live_named(
    *,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None,
    session_factory: SessionFactory | None,
    csv_loader: CsvLoader | None,
    boto_port_factory: BotoPortFactory | None = None,
) -> tuple[list[OperationOutcome], dict[str, Any], CredentialSource | None]:
    operations: list[OperationOutcome] = []
    guardrail_id = settings.guardrail_id
    version = settings.guardrail_version
    if guardrail_id is None or version is None:
        return (
            [
                not_run_operation(
                    service="bedrock-runtime",
                    operation_name="ApplyGuardrail",
                    error_code="missing_configuration",
                    request_validated=False,
                    optional=True,
                ),
                not_run_operation(
                    service="bedrock-runtime",
                    operation_name="ApplyGuardrail",
                    error_code="missing_configuration",
                    request_validated=False,
                    optional=True,
                ),
            ],
            {"mode": "named", "guardrail_configured": False},
            None,
        )
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
        return _missing_named_port(), {"mode": "named"}, credential_source
    get_result = port_operation(
        port=port,
        service="bedrock",
        operation_name="GetGuardrail",
        params=get_guardrail_params(guardrail_id, version),
        execution="live",
        effect="read",
        phase="setup",
        fixture_id=_GET_FIXTURE,
    )
    operations.append(get_result.outcome)
    if get_result.outcome["status"] != "ok":
        return operations, {"mode": "named"}, credential_source
    try:
        features = _policy_features_from_get(get_result.payload)
        budget = active_budget_run(
            region=settings.region, policy=policy
        ).command(_DEMO)
        data = _apply_live_pair_with_budget(
            port=port,
            budget=budget,
            operations=operations,
            guardrail_id=guardrail_id,
            guardrail_version=version,
            features=features,
        )
    except PolicyError:
        operations.extend(_apply_pricing_refusal())
        data = {"pricing": "missing_or_unsupported"}
    data["mode"] = "named"
    data["guardrail"] = {
        "input_action": data.get("input_action"),
        "output_action": data.get("output_action"),
    }
    return operations, data, credential_source


def _cleanup_result_error(
    data: MutableMapping[str, Any],
) -> tuple[ErrorCode, str] | None:
    if data.get("cleanup_incomplete") is True:
        return ("cleanup_incomplete", "Owned resources remain unresolved.")
    readiness = data.pop("readiness_error", None)
    if isinstance(readiness, tuple):
        code, message = readiness
        return (cast("ErrorCode", code), str(message))
    return None


def _resolve_poll_ports(
    clock: Callable[[], float] | None,
    sleeper: Callable[[float], None] | None,
) -> tuple[Callable[[], float], Callable[[float], None]]:
    """Return the clock and sleeper, using real time by default."""
    resolved_clock = time.monotonic if clock is None else clock
    resolved_sleep = time.sleep if sleeper is None else sleeper
    return resolved_clock, resolved_sleep


def _await_guardrail_ready(
    ctx: OwnedResources,
    guardrail_id: str,
    *,
    version: str | None,
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> tuple[str, tuple[ErrorCode, str] | None]:
    """Poll GetGuardrail until READY, FAILED, or the wall clock."""
    started = len(ctx.operations)
    deadline = clock() + ctx.run.policy.max_wall_seconds
    label = "draft" if version is None else f"version {version}"
    params: dict[str, str] = {"guardrailIdentifier": guardrail_id}
    if version is not None:
        params["guardrailVersion"] = version
    while True:
        if clock() >= deadline:
            _block_guardrail_readiness(ctx.operations, started, code="timeout")
            message = (
                f"Guardrail {label} was not READY before the wall-clock limit."
            )
            return "TIMEOUT", ("timeout", message)
        got = ctx.call(
            "bedrock",
            "GetGuardrail",
            params,
            effect="read",
            phase="setup",
        )
        if got.outcome["status"] != "ok":
            return "GET_FAILED", None
        status = str(got.payload.get("status", ""))
        if status == "READY":
            return "READY", None
        if status == "FAILED":
            _block_guardrail_readiness(
                ctx.operations,
                started,
                code="validation_failed",
            )
            return "FAILED", (
                "validation_failed",
                _failed_guardrail_message(label, got.payload),
            )
        sleeper(_GUARDRAIL_POLL_SECONDS)


def _failed_guardrail_message(label: str, payload: Mapping[str, Any]) -> str:
    """Name a FAILED guardrail and any statusReasons."""
    raw = payload.get("statusReasons")
    reasons = raw if isinstance(raw, list) else []
    detail = "; ".join(str(item) for item in reasons)
    if detail:
        return f"Guardrail {label} status is FAILED: {detail}."
    return f"Guardrail {label} status is FAILED."


def _record_readiness(
    data: dict[str, Any],
    status: str,
    readiness_error: tuple[ErrorCode, str] | None,
) -> None:
    """Remember a blocked readiness failure for the result."""
    data["guardrail_status"] = status
    if readiness_error is not None:
        data["readiness_error"] = readiness_error


def _block_guardrail_readiness(
    operations: list[OperationOutcome],
    started: int,
    *,
    code: ErrorCode,
) -> None:
    """Mark the readiness poll blocked and ignore earlier reads."""
    for outcome in reversed(operations[started:]):
        if outcome["operation"] != "GetGuardrail":
            continue
        if outcome["phase"] != "setup" or outcome["status"] != "ok":
            return
        outcome["status"] = "blocked"
        outcome["error_code"] = code
        return


def _create_live_version(
    ctx: OwnedResources,
    entry: Any,
    guardrail_id: str,
    token: str,
) -> str | None:
    child = ctx.intent(
        GUARDRAIL,
        {},
        parent=entry,
        operation="CreateGuardrailVersion",
        token=token,
    )
    ctx.store.mark_unknown(child.entry_id)
    response = ctx.call(
        "bedrock",
        "CreateGuardrailVersion",
        create_guardrail_version_params(guardrail_id, client_token=token),
        effect="write",
        phase="setup",
    )
    version = str(response.payload.get("version", ""))
    if response.outcome["status"] != "ok" or not version:
        return None
    ctx.store.mark_created(child.entry_id, exact_id=version, child_id=version)
    return version


def _verify_for_child(ctx: OwnedResources, entry: Any, kind: Any) -> bool:
    return bool(ctx.verify(entry, kind, phase="setup"))


def _apply_live_pair(
    ctx: OwnedResources,
    *,
    guardrail_id: str,
    guardrail_version: str,
    features: Sequence[str],
) -> dict[str, Any]:
    data: dict[str, Any] = {"policy_features": list(features)}
    for label, source, text in _APPLY_CASES:
        params = apply_guardrail_params(
            guardrail_id=guardrail_id,
            guardrail_version=guardrail_version,
            source=source,
            text=text,
        )
        result = ctx.call(
            "bedrock-runtime",
            "ApplyGuardrail",
            params,
            effect="infer",
            charges=guardrail_apply_charges(params, features=features),
        )
        data[f"{label}_action"] = result.payload.get("action")
    return data


def _apply_live_pair_with_budget(
    *,
    port: AwsPort,
    budget: PricedBudget,
    operations: list[OperationOutcome],
    guardrail_id: str,
    guardrail_version: str,
    features: Sequence[str],
) -> dict[str, Any]:
    data: dict[str, Any] = {"policy_features": list(features)}
    for label, source, text in _APPLY_CASES:
        params = apply_guardrail_params(
            guardrail_id=guardrail_id,
            guardrail_version=guardrail_version,
            source=source,
            text=text,
        )
        charges = guardrail_apply_charges(params, features=features)
        reservation = budget.reserve(
            f"{len(operations)}:ApplyGuardrail", charges
        )
        result = port_operation(
            port=port,
            service="bedrock-runtime",
            operation_name="ApplyGuardrail",
            params=params,
            execution="live",
            effect="infer",
            fixture_id=_INPUT_FIXTURE if label == "input" else _OUTPUT_FIXTURE,
            reserved=True,
        )
        result.outcome["reserved_usd"] = float(reservation.amount_usd)
        operations.append(result.outcome)
        data[f"{label}_action"] = result.payload.get("action")
    return data


def _preflight_guardrail_budget(budget: PricedBudget) -> Decimal:
    charges: list[Charge] = []
    for _, source, text in _APPLY_CASES:
        params = apply_guardrail_params(source=source, text=text)
        charges.extend(guardrail_apply_charges(params))
    return budget.quote(tuple(charges))


def _apply_guardrail_request(
    client: Any,
    params: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    # slide: apply-guardrail
    response = client.apply_guardrail(
        guardrailIdentifier=params["guardrailIdentifier"],
        guardrailVersion=params["guardrailVersion"],
        source=params["source"],
        content=params["content"],
        outputScope=params["outputScope"],
    )
    action = response["action"]
    # end-slide: apply-guardrail
    return cast("Mapping[str, Any]", response), str(action)


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


def _pricing_refusal() -> list[OperationOutcome]:
    return [
        billable_not_priced(
            service="bedrock", operation_name="CreateGuardrail", phase="setup"
        ),
        billable_not_priced(
            service="bedrock",
            operation_name="CreateGuardrailVersion",
            phase="setup",
        ),
        *_apply_pricing_refusal(),
        billable_not_priced(
            service="bedrock",
            operation_name="DeleteGuardrail",
            phase="teardown",
        ),
    ]


def _apply_pricing_refusal() -> list[OperationOutcome]:
    return [
        billable_not_priced(
            service="bedrock-runtime", operation_name="ApplyGuardrail"
        ),
        billable_not_priced(
            service="bedrock-runtime", operation_name="ApplyGuardrail"
        ),
    ]


def _missing_live_port() -> list[OperationOutcome]:
    return [
        require_port_not_run(
            service="bedrock", operation_name="CreateGuardrail", phase="setup"
        ),
        require_port_not_run(
            service="bedrock",
            operation_name="CreateGuardrailVersion",
            phase="setup",
        ),
        require_port_not_run(
            service="bedrock-runtime", operation_name="ApplyGuardrail"
        ),
        require_port_not_run(
            service="bedrock-runtime", operation_name="ApplyGuardrail"
        ),
        require_port_not_run(
            service="bedrock",
            operation_name="DeleteGuardrail",
            phase="teardown",
        ),
    ]


def _missing_named_port() -> list[OperationOutcome]:
    return [
        require_port_not_run(
            service="bedrock", operation_name="GetGuardrail", phase="setup"
        ),
        require_port_not_run(
            service="bedrock-runtime", operation_name="ApplyGuardrail"
        ),
        require_port_not_run(
            service="bedrock-runtime", operation_name="ApplyGuardrail"
        ),
    ]


def _policy_features_from_create(params: Mapping[str, Any]) -> tuple[str, ...]:
    features: list[str] = []
    if params.get("contentPolicyConfig"):
        features.append("guardrails.content_filter")
    if params.get("topicPolicyConfig"):
        features.append("guardrails.denied_topic")
    if params.get("sensitiveInformationPolicyConfig"):
        features.append("guardrails.sensitive_information")
    unknown = [
        key
        for key in (
            "wordPolicyConfig",
            "contextualGroundingPolicyConfig",
            "automatedReasoningPolicyConfig",
        )
        if params.get(key)
    ]
    if unknown:
        message = f"unsupported guardrail price features: {','.join(unknown)}"
        raise ReservationUnavailable(message)
    return tuple(features)


def _policy_features_from_get(payload: Mapping[str, Any]) -> tuple[str, ...]:
    features: list[str] = []
    if payload.get("contentPolicy"):
        features.append("guardrails.content_filter")
    if payload.get("topicPolicy"):
        features.append("guardrails.denied_topic")
    if payload.get("sensitiveInformationPolicy"):
        features.append("guardrails.sensitive_information")
    unknown = [
        key
        for key in (
            "wordPolicy",
            "contextualGroundingPolicy",
            "automatedReasoningPolicy",
        )
        if payload.get(key)
    ]
    if unknown or not features:
        message = "named guardrail contains unpriced policy features"
        raise ReservationUnavailable(message)
    return tuple(features)


def _text_units(params: Mapping[str, Any]) -> Decimal:
    blocks = cast("Sequence[Mapping[str, Any]]", params["content"])
    char_count = 0
    for block in blocks:
        text_block = cast("Mapping[str, str]", block.get("text", {}))
        char_count += len(text_block.get("text", ""))
    return Decimal(max(1, (char_count + 999) // 1000))


def _token(run_id: str, suffix: str) -> str:
    return f"{run_id}-{suffix}-{run_id}-{suffix}-{run_id}"[:64]


def _run_tags(run_id: str) -> list[dict[str, str]]:
    return [
        {"key": "run-id", "value": run_id},
        {"key": "project", "value": "aws-ai-stack-workshop"},
    ]


def _fixture_tags() -> list[dict[str, str]]:
    return [
        {"key": "run-id", "value": "offline-fixture"},
        {"key": "project", "value": "aws-ai-stack-workshop"},
    ]


def _create_response() -> dict[str, Any]:
    return {
        "guardrailId": _GUARDRAIL_ID,
        "guardrailArn": _GUARDRAIL_ARN,
        "version": "DRAFT",
        "createdAt": _NOW,
    }


def _apply_response(action: str) -> dict[str, Any]:
    return {
        "action": action,
        "outputs": [],
        "assessments": [],
        "usage": {
            "topicPolicyUnits": 1,
            "contentPolicyUnits": 1,
            "wordPolicyUnits": 0,
            "sensitiveInformationPolicyUnits": 1,
            "sensitiveInformationPolicyFreeUnits": 0,
            "contextualGroundingPolicyUnits": 0,
        },
    }


_APPLY_CASES = (
    ("input", "INPUT", _INPUT_TEXT),
    ("output", "OUTPUT", _OUTPUT_TEXT),
)
