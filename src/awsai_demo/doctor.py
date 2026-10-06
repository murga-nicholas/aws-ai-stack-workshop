"""Configuration-only doctor checks with injected live probes."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Literal, Protocol, cast
from urllib.parse import urlsplit, urlunsplit

from awsai_demo.contracts import OperationOutcome, operation
from awsai_demo.policy import ExecutionPolicy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from awsai_demo.credentials import SelectedSession
    from awsai_demo.runtime import Settings

CheckStatus = Literal["ok", "warning", "blocked"]
ProbeMode = Literal["live_identity", "live_service"]


def installed_package_versions() -> dict[str, str | None]:
    """Read local distribution metadata without network calls."""
    packages = {
        "boto3": "boto3",
        "strands_agents": "strands-agents",
        "bedrock_agentcore": "bedrock-agentcore",
        "a2a_sdk": "a2a-sdk",
        "mcp": "mcp",
    }
    versions: dict[str, str | None] = {}
    for key, distribution in packages.items():
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[key] = None
    return versions


class ProbePort(Protocol):
    """Injected live-read probe port for explicit `doctor --probe`."""

    def probe(self, settings: Settings) -> DoctorProbeResult:
        """Run bounded read-only checks and return safe diagnostics."""


@dataclass(frozen=True)
class DoctorCheck:
    """One safe doctor diagnostic."""

    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    """Doctor output suitable for CLI pretty or JSON rendering."""

    checks: tuple[DoctorCheck, ...]
    probed: bool
    operations: tuple[OperationOutcome, ...] = ()

    @property
    def status(self) -> CheckStatus:
        """Return the worst check status."""
        if any(check.status == "blocked" for check in self.checks):
            return "blocked"
        if any(check.status == "warning" for check in self.checks):
            return "warning"
        return "ok"


@dataclass(frozen=True)
class DoctorProbeResult:
    """Probe checks plus operation evidence."""

    checks: tuple[DoctorCheck, ...]
    operations: tuple[OperationOutcome, ...]


@dataclass(frozen=True)
class ProbeCall:
    """One bounded read-only SDK probe."""

    service_name: str
    operation_name: str
    mode: ProbeMode


class BotoSessionLike(Protocol):
    """Minimal boto3 Session surface used by doctor probes."""

    def client(self, service_name: str, **kwargs: object) -> object:
        """Return a service client."""


class AwsProbePort:
    """Read-only AWS probe port from a selected session."""

    def __init__(
        self,
        selected_session: SelectedSession,
        *,
        calls: Sequence[ProbeCall] | None = None,
        policy: ExecutionPolicy | None = None,
    ) -> None:
        """Store probe calls without constructing any clients yet."""
        self._session = cast("BotoSessionLike", selected_session.session)
        self._region = selected_session.region
        self._policy = policy or ExecutionPolicy()
        self._calls = tuple(calls or _DEFAULT_PROBE_CALLS)
        _validate_probe_shapes(self._calls)

    def probe(self, settings: Settings) -> DoctorProbeResult:
        """Run STS plus bounded read-only List calls."""
        del settings
        deadline = time.monotonic() + float(self._policy.max_wall_seconds)
        checks: list[DoctorCheck] = []
        operations: list[OperationOutcome] = []
        for call in self._calls:
            check, outcome = self._run_call(call, deadline)
            checks.append(check)
            operations.append(outcome)
        return DoctorProbeResult(
            checks=tuple(checks),
            operations=tuple(operations),
        )

    def _run_call(
        self, call: ProbeCall, deadline: float
    ) -> tuple[DoctorCheck, OperationOutcome]:
        if time.monotonic() >= deadline:
            return _not_run_probe(call, "wall-clock limit exceeded")
        remaining_seconds = max(1.0, deadline - time.monotonic())
        per_attempt_seconds = max(
            1.0,
            remaining_seconds / max(1, self._policy.total_max_attempts),
        )
        method_name = _client_method_name(call.operation_name)
        from botocore.config import Config
        from botocore.exceptions import (
            ClientError,
            ConnectTimeoutError,
            EndpointConnectionError,
            ReadTimeoutError,
        )

        client = self._session.client(
            call.service_name,
            region_name=self._region,
            config=Config(
                connect_timeout=min(per_attempt_seconds, 5.0),
                read_timeout=per_attempt_seconds,
                retries={
                    "total_max_attempts": self._policy.total_max_attempts,
                    "mode": "standard",
                },
            ),
        )
        try:
            response = getattr(client, method_name)()
        except ClientError as exc:
            return _client_error_probe(call, exc, client)
        except (
            ConnectTimeoutError,
            EndpointConnectionError,
            ReadTimeoutError,
        ) as exc:
            return _timeout_probe(call, exc.__class__.__name__)
        count = _safe_response_count(response)
        return (
            DoctorCheck(
                call.service_name, "ok", f"{call.operation_name}: {count}"
            ),
            operation(
                service=call.service_name,
                operation=call.operation_name,
                execution_target="aws",
                mode=call.mode,
                status="ok",
                effect="read",
                transport="aws",
                endpoint_url=_client_endpoint(client),
                response_received=True,
                request_validated=True,
            ),
        )


def inspect_configuration(settings: Settings) -> DoctorReport:
    """Inspect configuration without file reads or networking."""
    checks = [
        DoctorCheck("region", "ok", settings.region),
        DoctorCheck("credential_source", "ok", _credential_source(settings)),
        DoctorCheck(
            "aws_creds_file",
            "ok" if settings.aws_creds_file_path is not None else "warning",
            "configured"
            if settings.aws_creds_file_path is not None
            else "not set",
        ),
        DoctorCheck(
            "localstack_endpoint",
            "ok",
            _safe_endpoint_detail(settings.localstack_endpoint),
        ),
        DoctorCheck(
            "openai_api_key",
            "ok" if settings.openai_api_key is not None else "warning",
            "configured" if settings.openai_api_key is not None else "not set",
        ),
    ]
    return DoctorReport(checks=tuple(checks), probed=False)


def run_doctor(
    settings: Settings,
    *,
    probe: bool = False,
    probe_port: ProbePort | None = None,
    selected_session: SelectedSession | None = None,
    policy: ExecutionPolicy | None = None,
) -> DoctorReport:
    """Run config-only doctor unless a probe is supplied."""
    report = inspect_configuration(settings)
    if not probe:
        return report
    port = probe_port
    if port is None and selected_session is not None:
        port = AwsProbePort(selected_session, policy=policy)
    if port is None:
        check = DoctorCheck(
            "probe_port",
            "blocked",
            "doctor --probe requires an injected probe port",
        )
        return DoctorReport(checks=(*report.checks, check), probed=False)
    probe_result = port.probe(settings)
    return DoctorReport(
        checks=(*report.checks, *probe_result.checks),
        probed=True,
        operations=probe_result.operations,
    )


def _credential_source(settings: Settings) -> str:
    if settings.aws_profile is not None:
        return "profile"
    if settings.aws_creds_file_path is not None:
        return "csv_file"
    return "default_chain"


def _client_error_probe(
    call: ProbeCall,
    exc: object,
    client: object,
) -> tuple[DoctorCheck, OperationOutcome]:
    response = getattr(exc, "response", {})
    error_payload = response.get("Error", {})
    metadata = response.get("ResponseMetadata", {})
    code = str(error_payload.get("Code", exc.__class__.__name__))
    http_status = int(metadata.get("HTTPStatusCode", 0)) or None
    mapped_code = "authorization_denied" if http_status == 403 else code
    detail = f"{call.operation_name}: {mapped_code}"
    return (
        DoctorCheck(call.service_name, "blocked", detail),
        operation(
            service=call.service_name,
            operation=call.operation_name,
            execution_target="aws",
            mode="live_service",
            status="blocked",
            effect="read",
            transport="aws",
            endpoint_url=_client_endpoint(client),
            response_received=True,
            request_validated=True,
            http_status=http_status,
            error_code=mapped_code,
        ),
    )


def _timeout_probe(
    call: ProbeCall,
    code: str,
) -> tuple[DoctorCheck, OperationOutcome]:
    detail = f"{call.operation_name}: {code}"
    return (
        DoctorCheck(call.service_name, "blocked", detail),
        operation(
            service=call.service_name,
            operation=call.operation_name,
            execution_target="aws",
            mode="attempt_failed",
            status="error",
            effect="read",
            transport="aws",
            endpoint_url=None,
            response_received=False,
            request_validated=True,
            error_code=code,
        ),
    )


def _not_run_probe(
    call: ProbeCall,
    detail: str,
) -> tuple[DoctorCheck, OperationOutcome]:
    return (
        DoctorCheck(call.service_name, "blocked", detail),
        operation(
            service=call.service_name,
            operation=call.operation_name,
            execution_target="aws",
            mode="not_run",
            status="blocked",
            effect="none",
            transport="none",
            endpoint_url=None,
            response_received=False,
            # The probe operation model was validated at AwsProbePort
            # construction; no network request was dispatched.
            request_validated=True,
            http_status=None,
            error_code="timeout",
            usage=None,
            reserved_usd=None,
        ),
    )


def _client_method_name(operation_name: str) -> str:
    first_pass = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", operation_name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()


def _safe_response_count(response: object) -> str:
    if not isinstance(response, dict):
        return "response"
    for value in response.values():
        if isinstance(value, list):
            return f"{len(value)} item(s)"
    return f"{len(response)} field(s)"


def _client_endpoint(client: object) -> str | None:
    meta = getattr(client, "meta", None)
    endpoint_url = getattr(meta, "endpoint_url", None)
    if isinstance(endpoint_url, str):
        return endpoint_url
    return None


def _safe_endpoint_detail(endpoint_url: str) -> str:
    parsed = urlsplit(endpoint_url)
    if not parsed.scheme or parsed.hostname is None:
        return "configured"
    netloc = parsed.hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _validate_probe_shapes(calls: Sequence[ProbeCall]) -> None:
    from botocore.loaders import Loader
    from botocore.model import ServiceModel

    loader = Loader()
    for call in calls:
        service_model = ServiceModel(
            loader.load_service_model(call.service_name, "service-2"),
            service_name=call.service_name,
        )
        operation_model = service_model.operation_model(call.operation_name)
        required = (
            []
            if operation_model.input_shape is None
            else operation_model.input_shape.required_members
        )
        if required:
            msg = f"{call.service_name}.{call.operation_name} requires input"
            raise ValueError(msg)


_DEFAULT_PROBE_CALLS = (
    ProbeCall("sts", "GetCallerIdentity", "live_identity"),
    ProbeCall("bedrock", "ListFoundationModels", "live_service"),
    ProbeCall(
        "bedrock-agentcore-control", "ListAgentRuntimes", "live_service"
    ),
    ProbeCall("bedrock-agentcore", "ListRecommendations", "live_service"),
    ProbeCall("s3vectors", "ListVectorBuckets", "live_service"),
    ProbeCall("nova-act", "ListWorkflowDefinitions", "live_service"),
    ProbeCall("bedrock-agent-runtime", "ListSessions", "live_service"),
)
