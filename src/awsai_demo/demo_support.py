"""Shared helpers for demo modules that emit contract evidence."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from ipaddress import ip_address
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from urllib.parse import urlsplit

from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from awsai_demo.contracts import (
    CredentialSource,
    DemoResult,
    Effect,
    EmulatorEvidence,
    ErrorCode,
    Execution,
    ExecutionTarget,
    Lane,
    Mode,
    OperationOutcome,
    Phase,
    Transport,
    Usage,
    error,
    evidence,
    operation,
    result,
)
from awsai_demo.credentials import select_session
from awsai_demo.serialization import normalize_timestamps
from awsai_demo.stubs import validate_operation_request

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.policy import ExecutionPolicy
    from awsai_demo.runtime import Settings


@dataclass(frozen=True)
class AwsResponse:
    """A port response plus optional endpoint evidence."""

    payload: Mapping[str, Any]
    endpoint_url: str | None = None


class AwsPort(Protocol):
    """Injected AWS-like JSON operation port."""

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse | Mapping[str, Any]:
        """Call one AWS-like operation and return JSON-like data."""


class AwsStreamPort(Protocol):
    """Injected AWS-like event-stream operation port."""

    def stream(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse | Sequence[Mapping[str, Any]]:
        """Call one AWS-like stream and return replay events."""


class BotoSessionPort(Protocol):
    """Small boto3 Session surface needed by the demos."""

    def client(self, service_name: str, **kwargs: Any) -> Any:
        """Return a boto client for one service."""


@dataclass(frozen=True)
class BotoAwsPort:
    """Generic boto-backed adapter for read-only or reserved calls."""

    session: BotoSessionPort
    region_name: str
    endpoint_url: str | None = None
    config: object | None = None
    loopback_only: bool = False

    def __post_init__(self) -> None:
        """Reject accidental non-loopback emulator endpoints."""
        if self.loopback_only and not _is_loopback_endpoint(self.endpoint_url):
            msg = "emulator ports require a loopback endpoint"
            raise ValueError(msg)

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse:
        """Dispatch one boto client operation."""
        client = self._client(service)
        method = getattr(client, client_method_name(operation_name))
        try:
            payload = method(**dict(params))
        except (
            ClientError,
            ConnectTimeoutError,
            EndpointConnectionError,
            OSError,
            ReadTimeoutError,
        ) as exc:
            _attach_endpoint(exc, _client_endpoint(client, self.endpoint_url))
            raise
        if not isinstance(payload, dict):
            msg = "boto operation returned a non-mapping payload"
            raise TypeError(msg)
        return AwsResponse(
            normalize_timestamps(payload),
            _client_endpoint(client, self.endpoint_url),
        )

    def stream(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse:
        """Dispatch one boto event-stream operation."""
        return self.call(service, operation_name, params)

    def _client(self, service: str) -> Any:
        kwargs: dict[str, object] = {"region_name": self.region_name}
        if self.endpoint_url is not None:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.config is not None:
            kwargs["config"] = self.config
        return self.session.client(service, **kwargs)


@dataclass(frozen=True)
class DemoOperation:
    """Operation outcome plus response data used by a demo."""

    outcome: OperationOutcome
    payload: Mapping[str, Any]
    aws_error_code: str | None = None


@dataclass(frozen=True)
class DefaultBotoPort:
    """Default SDK port plus credential-source evidence."""

    port: BotoAwsPort
    credential_source: CredentialSource


BotoPortFactory = Callable[..., DefaultBotoPort | None]


def client_method_name(operation_name: str) -> str:
    """Return the boto3 client method for a PascalCase operation."""
    first_pass = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", operation_name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()


def validate_request(
    *,
    service: str,
    operation_name: str,
    params: Mapping[str, Any],
    fixture_id: str,
) -> None:
    """Validate params against the installed botocore model."""
    validate_operation_request(
        service_name=service,
        operation_name=operation_name,
        params=params,
        fixture_id=fixture_id,
    )


def bounded_boto_config(policy: ExecutionPolicy) -> object:
    """Return a botocore Config bounded by the execution policy."""
    from botocore.config import Config

    timeout = max(1, min(policy.max_wall_seconds, 30))
    return Config(
        retries={
            "mode": "standard",
            "total_max_attempts": policy.total_max_attempts,
        },
        connect_timeout=timeout,
        read_timeout=timeout,
    )


def build_default_boto_port(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DefaultBotoPort | None:
    """Build a bounded boto adapter for emulator or live execution."""
    if execution == "offline":
        return None
    if execution == "emulator" and settings.localstack_auth_token is None:
        return None
    selected = select_session(
        execution=execution,
        settings=settings,
        session_factory=session_factory,
        csv_loader=csv_loader,
    )
    endpoint_url = (
        settings.localstack_endpoint if execution == "emulator" else None
    )
    return DefaultBotoPort(
        port=BotoAwsPort(
            session=cast("BotoSessionPort", selected.session),
            region_name=selected.region,
            endpoint_url=endpoint_url,
            config=bounded_boto_config(policy),
            loopback_only=execution == "emulator",
        ),
        credential_source=selected.source,
    )


def resolve_boto_port(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    factory: BotoPortFactory | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DefaultBotoPort | None:
    """Build the default boto port, or call an injected factory.

    Tests pass a factory when they need a stand-in for credential
    selection. Production calls leave the factory unset.
    """
    build = build_default_boto_port if factory is None else factory
    return build(
        execution=execution,
        settings=settings,
        policy=policy,
        session_factory=session_factory,
        csv_loader=csv_loader,
    )


def fixture_operation(
    *,
    service: str,
    operation_name: str,
    fixture_id: str,
    effect: Effect,
    phase: Phase = "main",
    request_validated: bool = True,
    usage: Mapping[str, int] | None = None,
) -> OperationOutcome:
    """Build an offline fixture operation outcome."""
    return operation(
        service=service,
        operation=operation_name,
        phase=phase,
        execution_target="fixture",
        mode="local_contract",
        status="ok",
        effect=effect,
        transport="none",
        response_received=True,
        request_validated=request_validated,
        fixture_id=fixture_id,
        usage=_usage(usage),
    )


def local_operation(
    *,
    service: str,
    operation_name: str,
    effect: Effect = "none",
    phase: Phase = "main",
) -> OperationOutcome:
    """Build a local in-process operation outcome."""
    return operation(
        service=service,
        operation=operation_name,
        phase=phase,
        execution_target="local",
        mode="local_execution",
        status="ok",
        effect=effect,
        transport="none",
        response_received=True,
        request_validated=True,
    )


def not_run_operation(
    *,
    service: str,
    operation_name: str,
    error_code: ErrorCode,
    request_validated: bool,
    phase: Phase = "main",
    status: Literal["blocked", "error", "paused"] = "blocked",
    optional: bool = False,
) -> OperationOutcome:
    """Build a skipped operation outcome for the public contract."""
    return operation(
        service=service,
        operation=operation_name,
        phase=phase,
        execution_target="aws",
        mode="not_run",
        status=status,
        effect="none",
        transport="none",
        response_received=False,
        request_validated=request_validated,
        error_code=error_code,
        optional=optional,
    )


def port_operation(
    *,
    port: AwsPort,
    service: str,
    operation_name: str,
    params: Mapping[str, Any],
    execution: Execution,
    effect: Effect,
    fixture_id: str,
    phase: Phase = "main",
    endpoint_url: str | None = None,
    reserved: bool = False,
) -> DemoOperation:
    """Validate and invoke an injected AWS-like port."""
    _enforce_live_reservation(
        execution=execution, effect=effect, reserved=reserved
    )
    validate_request(
        service=service,
        operation_name=operation_name,
        params=params,
        fixture_id=fixture_id,
    )
    try:
        raw_response = port.call(service, operation_name, params)
    except ClientError as exc:
        return DemoOperation(
            outcome=_client_error_outcome(
                service=service,
                operation_name=operation_name,
                execution=execution,
                effect=effect,
                phase=phase,
                endpoint_url=_error_endpoint(
                    exc,
                    port=port,
                    endpoint_url=endpoint_url,
                ),
                exc=exc,
            ),
            payload={},
            aws_error_code=str(exc.response.get("Error", {}).get("Code", "")),
        )
    except (
        ConnectTimeoutError,
        EndpointConnectionError,
        OSError,
        ReadTimeoutError,
    ) as exc:
        return DemoOperation(
            outcome=_attempt_failed_outcome(
                service=service,
                operation_name=operation_name,
                execution=execution,
                effect=effect,
                phase=phase,
                endpoint_url=_error_endpoint(
                    exc,
                    port=port,
                    endpoint_url=endpoint_url,
                ),
                exc=exc,
            ),
            payload={},
        )
    payload, response_endpoint = _normalize_response(raw_response)
    return DemoOperation(
        outcome=_answered_operation(
            service=service,
            operation_name=operation_name,
            execution=execution,
            effect=effect,
            phase=phase,
            endpoint_url=response_endpoint or endpoint_url,
        ),
        payload=payload,
    )


def stream_port_operation(
    *,
    port: AwsStreamPort,
    service: str,
    operation_name: str,
    params: Mapping[str, Any],
    execution: Execution,
    effect: Effect,
    fixture_id: str,
    phase: Phase = "main",
    endpoint_url: str | None = None,
    reserved: bool = False,
) -> DemoOperation:
    """Validate and invoke an injected AWS-like event-stream port."""
    _enforce_live_reservation(
        execution=execution, effect=effect, reserved=reserved
    )
    validate_request(
        service=service,
        operation_name=operation_name,
        params=params,
        fixture_id=fixture_id,
    )
    try:
        raw_response = port.stream(service, operation_name, params)
    except ClientError as exc:
        return DemoOperation(
            outcome=_client_error_outcome(
                service=service,
                operation_name=operation_name,
                execution=execution,
                effect=effect,
                phase=phase,
                endpoint_url=_error_endpoint(
                    exc,
                    port=port,
                    endpoint_url=endpoint_url,
                ),
                exc=exc,
            ),
            payload={},
            aws_error_code=str(exc.response.get("Error", {}).get("Code", "")),
        )
    except (
        ConnectTimeoutError,
        EndpointConnectionError,
        OSError,
        ReadTimeoutError,
    ) as exc:
        return DemoOperation(
            outcome=_attempt_failed_outcome(
                service=service,
                operation_name=operation_name,
                execution=execution,
                effect=effect,
                phase=phase,
                endpoint_url=_error_endpoint(
                    exc,
                    port=port,
                    endpoint_url=endpoint_url,
                ),
                exc=exc,
            ),
            payload={},
        )
    payload, response_endpoint = _normalize_stream(raw_response)
    return DemoOperation(
        outcome=_answered_operation(
            service=service,
            operation_name=operation_name,
            execution=execution,
            effect=effect,
            phase=phase,
            endpoint_url=response_endpoint or endpoint_url,
        ),
        payload=payload,
    )


def build_result(
    *,
    demo: str,
    technology: str,
    lane: Lane,
    lifecycle_refs: Sequence[str],
    execution: Execution,
    headline: str,
    operations: Sequence[OperationOutcome],
    settings: Settings,
    data: Mapping[str, Any] | None = None,
    requested_model: str | None = None,
    observed_model: str | None = None,
    credential_source: CredentialSource | None = None,
    result_error: tuple[ErrorCode, str] | None = None,
) -> DemoResult:
    """Build a demo result with common safe evidence fields."""
    reservations = [
        item["reserved_usd"]
        for item in operations
        if item["reserved_usd"] is not None
    ]
    return result(
        demo=demo,
        technology=technology,
        lane=lane,
        lifecycle_refs=lifecycle_refs,
        requested_execution=execution,
        headline=headline,
        operations=operations,
        evidence=evidence(
            sdk_invoked=any(item["request_validated"] for item in operations),
            network_attempted=any(
                item["transport"] in {"aws", "external", "loopback"}
                for item in operations
            ),
            aws_executed=any(
                item["execution_target"] == "aws" and item["response_received"]
                for item in operations
            ),
            provider=None,
            region=settings.region,
            fixture_id=next(
                (
                    item["fixture_id"]
                    for item in operations
                    if item["fixture_id"]
                ),
                None,
            ),
            requested_model=requested_model,
            observed_model=observed_model,
            credential_source=credential_source or "none",
            packages=_sdk_packages(operations),
            emulator=_emulator_evidence(settings, operations),
            estimated_cost_usd=sum(reservations) if reservations else None,
            cost_basis="estimated" if reservations else None,
        ),
        data=dict(data or {}),
        error=None
        if result_error is None
        else error(result_error[0], result_error[1]),
        status="error"
        if result_error is not None and result_error[0] == "cleanup_incomplete"
        else None,
    )


def billable_not_priced(
    *,
    service: str,
    operation_name: str,
    phase: Phase = "main",
) -> OperationOutcome:
    """Return a fail-closed outcome for a live billable operation."""
    return not_run_operation(
        service=service,
        operation_name=operation_name,
        phase=phase,
        error_code="budget_exceeded",
        request_validated=False,
    )


def require_port_not_run(
    *,
    service: str,
    operation_name: str,
    phase: Phase = "main",
) -> OperationOutcome:
    """Return a skipped outcome for a missing port."""
    return not_run_operation(
        service=service,
        operation_name=operation_name,
        phase=phase,
        error_code="missing_configuration",
        request_validated=False,
    )


def unsupported_emulator(
    *,
    service: str,
    operation_name: str,
    phase: Phase = "main",
) -> OperationOutcome:
    """Return a skipped outcome for rows absent from LocalStack."""
    return not_run_operation(
        service=service,
        operation_name=operation_name,
        phase=phase,
        error_code="not_supported_by_emulator",
        request_validated=False,
    )


def contract_only(
    *,
    service: str,
    operation_name: str,
    phase: Phase = "main",
) -> OperationOutcome:
    """Return a skipped outcome for documented contract-only rows."""
    return not_run_operation(
        service=service,
        operation_name=operation_name,
        phase=phase,
        error_code="contract_only",
        request_validated=False,
    )


def _answered_operation(
    *,
    service: str,
    operation_name: str,
    execution: Execution,
    effect: Effect,
    phase: Phase,
    endpoint_url: str | None,
) -> OperationOutcome:
    target, mode, transport = _answered_provenance(
        execution,
        service,
        operation_name,
    )
    return operation(
        service=service,
        operation=operation_name,
        phase=phase,
        execution_target=target,
        mode=mode,
        status="ok",
        effect=effect,
        transport=transport,
        endpoint_url=endpoint_url,
        response_received=True,
        request_validated=True,
    )


def _enforce_live_reservation(
    *,
    execution: Execution,
    effect: Effect,
    reserved: bool,
) -> None:
    if (
        execution == "live"
        and effect in _BILLABLE_LIVE_EFFECTS
        and not reserved
    ):
        msg = "live billable port calls require a prior reservation"
        raise ValueError(msg)


def _answered_provenance(
    execution: Execution,
    service: str,
    operation_name: str,
) -> tuple[ExecutionTarget, Mode, Transport]:
    if execution == "emulator":
        return "emulator", "local_emulator", "loopback"
    if execution == "live":
        if service == "sts" and operation_name == "GetCallerIdentity":
            return "aws", "live_identity", "aws"
        if service == "bedrock-runtime" and operation_name in {
            "Converse",
            "ConverseStream",
            "InvokeModel",
        }:
            return "aws", "live_model", "aws"
        return "aws", "live_service", "aws"
    return "fixture", "local_contract", "none"


def public_aws_error_code(
    payload: Mapping[str, Any], status_code: int | None
) -> str:
    """Classify observed AWS diagnostics without exposing their text."""
    code = str(payload.get("Code", "ClientError"))
    message = str(payload.get("Message", "")).lower()
    if (
        code == "ResourceNotFoundException"
        and "model use case details have not been submitted" in message
        and "anthropic use case details" in message
    ):
        return "model_access_required"
    return {"SubscriptionRequiredException": "subscription_required"}.get(
        code, "authorization_denied" if status_code == 403 else code
    )


def _client_error_outcome(
    *,
    service: str,
    operation_name: str,
    execution: Execution,
    effect: Effect,
    phase: Phase,
    endpoint_url: str | None,
    exc: ClientError,
) -> OperationOutcome:
    metadata = exc.response.get("ResponseMetadata", {})
    error_payload = exc.response.get("Error", {})
    status_code = int(metadata.get("HTTPStatusCode", 0)) or None
    public_code = public_aws_error_code(error_payload, status_code)
    if execution == "emulator" and _entitlement_diagnostic(error_payload):
        public_code = "entitlement_missing"
    return operation(
        service=service,
        operation=operation_name,
        phase=phase,
        execution_target="aws" if execution == "live" else "emulator",
        mode="live_service" if execution == "live" else "local_emulator",
        status="blocked",
        effect=effect,
        transport="aws" if execution == "live" else "loopback",
        endpoint_url=endpoint_url,
        response_received=True,
        request_validated=True,
        http_status=status_code,
        error_code=public_code,
    )


def _entitlement_diagnostic(payload: Mapping[str, Any]) -> bool:
    code = str(payload.get("Code", "")).lower()
    message = str(payload.get("Message", "")).lower()
    return (
        "license" in code
        or "entitlement" in code
        or "requires the ultimate plan" in message
        or "not available in your plan" in message
    )


def _attempt_failed_outcome(
    *,
    service: str,
    operation_name: str,
    execution: Execution,
    effect: Effect,
    phase: Phase,
    endpoint_url: str | None,
    exc: BaseException,
) -> OperationOutcome:
    del exc
    target: ExecutionTarget = "aws" if execution == "live" else "emulator"
    transport: Transport = "aws" if execution == "live" else "loopback"
    code: ErrorCode = (
        "emulator_unavailable" if execution == "emulator" else "timeout"
    )
    return operation(
        service=service,
        operation=operation_name,
        phase=phase,
        execution_target=target,
        mode="attempt_failed",
        status="blocked",
        effect=effect,
        transport=transport,
        endpoint_url=endpoint_url,
        response_received=False,
        request_validated=True,
        error_code=code,
    )


def _attach_endpoint(exc: BaseException, endpoint_url: str | None) -> None:
    if endpoint_url is not None:
        cast("Any", exc)._awsai_endpoint_url = endpoint_url


def _error_endpoint(
    exc: BaseException,
    *,
    port: object,
    endpoint_url: str | None,
) -> str | None:
    attached = getattr(exc, "_awsai_endpoint_url", None)
    if isinstance(attached, str):
        return attached
    if endpoint_url is not None:
        return endpoint_url
    port_endpoint = getattr(port, "endpoint_url", None)
    return port_endpoint if isinstance(port_endpoint, str) else None


def _normalize_response(
    response: AwsResponse | Mapping[str, Any],
) -> tuple[Mapping[str, Any], str | None]:
    if isinstance(response, AwsResponse):
        return normalize_timestamps(response.payload), response.endpoint_url
    return normalize_timestamps(response), None


def _normalize_stream(
    response: AwsResponse | Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], str | None]:
    if isinstance(response, AwsResponse):
        return normalize_timestamps(response.payload), response.endpoint_url
    return {"stream": normalize_timestamps(tuple(response))}, None


def _client_endpoint(client: object, fallback: str | None) -> str | None:
    meta = getattr(client, "meta", None)
    endpoint = getattr(meta, "endpoint_url", None)
    if isinstance(endpoint, str):
        return endpoint
    return fallback


def _is_loopback_endpoint(endpoint_url: str | None) -> bool:
    if endpoint_url is None:
        return False
    parsed = urlsplit(endpoint_url)
    if parsed.hostname is None:
        return False
    host = parsed.hostname.lower()
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _usage(usage: Mapping[str, int] | None) -> Usage | None:
    if usage is None:
        return None
    result_usage: Usage = {}
    if "inputTokens" in usage:
        result_usage["input_tokens"] = usage["inputTokens"]
    if "outputTokens" in usage:
        result_usage["output_tokens"] = usage["outputTokens"]
    if "totalTokens" in usage:
        result_usage["total_tokens"] = usage["totalTokens"]
    return result_usage


def _sdk_packages(
    operations: Sequence[OperationOutcome],
) -> dict[str, str]:
    if not any(item["request_validated"] for item in operations):
        return {}
    return {
        "boto3": metadata.version("boto3"),
        "botocore": metadata.version("botocore"),
    }


def _emulator_evidence(
    settings: Settings,
    operations: Sequence[OperationOutcome],
) -> EmulatorEvidence | None:
    if not any(item["mode"] == "local_emulator" for item in operations):
        return None
    return {"endpoint": settings.localstack_endpoint}


_BILLABLE_LIVE_EFFECTS: frozenset[Effect] = frozenset(
    {
        "infer",
        "write",
        "delete",
        "session_start",
        "session_stop",
        "export",
        "invoke_named",
    }
)
