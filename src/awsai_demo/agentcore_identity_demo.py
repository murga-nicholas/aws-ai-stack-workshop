"""AgentCore Identity request contracts and read-only list demo.

Lane: agents; lifecycle: agentcore-identity.
Run: uv run awsai-demo agentcore-identity --execution offline.
"""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING, Any, cast

from awsai_demo.contracts import (
    CredentialSource,
    DemoResult,
    Execution,
    OperationOutcome,
    evidence,
    operation,
    result,
)
from awsai_demo.demo_support import (
    AwsPort,
    BotoAwsPort,
    BotoSessionPort,
    bounded_boto_config,
    not_run_operation,
    port_operation,
    require_port_not_run,
    resolve_boto_port,
    unsupported_emulator,
    validate_request,
)
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings
from awsai_demo.stubs import validate_operation_request

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from awsai_demo.credentials import SelectedSession
    from awsai_demo.demo_support import BotoPortFactory


FIXTURE_PREFIX = "agentcore-identity-request"
_DEMO = "agentcore-identity"
_TECHNOLOGY = "AgentCore Identity"
_REFS = ("agentcore-identity",)
_CONTROL = "bedrock-agentcore-control"
_RUNTIME = "bedrock-agentcore"


def run_agentcore_identity_demo(
    *,
    execution: Execution = "offline",
    settings: Settings | None = None,
    policy: ExecutionPolicy | None = None,
    port: AwsPort | None = None,
    selected_session: SelectedSession | None = None,
    boto_port_factory: BotoPortFactory | None = None,
) -> DemoResult:
    """Run AgentCore Identity request contracts and live reads."""
    active_settings = settings or Settings()
    chosen_policy = policy or ExecutionPolicy()
    if execution == "offline":
        return _run_offline(active_settings)
    if execution == "emulator":
        return _run_emulator(active_settings)
    return _run_live_read_only(
        active_settings,
        chosen_policy,
        port=port,
        selected_session=selected_session,
        boto_port_factory=boto_port_factory,
    )


def validate_agentcore_identity_contracts() -> list[OperationOutcome]:
    """Validate AgentCore Identity request shapes."""
    operations: list[OperationOutcome] = []
    for service, name, params in _request_rows():
        validation = validate_operation_request(
            service_name=service,
            operation_name=name,
            params=params,
            fixture_id=_fixture_id(name),
        )
        operations.append(
            operation(
                service=service,
                operation=name,
                mode="local_contract",
                effect="read" if name.startswith("Get") else "write",
                fixture_id=validation.fixture_id,
                request_validated=validation.request_validated,
            ),
        )
    return operations


def _run_offline(settings: Settings) -> DemoResult:
    operations = validate_agentcore_identity_contracts()
    return _identity_result(
        execution="offline",
        settings=settings,
        operations=operations,
        headline="AgentCore Identity request shapes validated locally.",
        provider="botocore-contracts",
        credential_source="none",
        data={
            "inbound": "workload identity validates agent callers",
            "outbound": "credential providers mint resource tokens",
            "decorators": (
                "bedrock_agentcore.identity.requires_access_token",
            ),
            "request_contracts": [row[1] for row in _request_rows()],
        },
    )


def _run_emulator(settings: Settings) -> DemoResult:
    operations = [
        unsupported_emulator(
            service=_CONTROL,
            operation_name="CreateWorkloadIdentity",
        ),
        unsupported_emulator(
            service=_RUNTIME,
            operation_name="GetWorkloadAccessToken",
        ),
    ]
    return _identity_result(
        execution="emulator",
        settings=settings,
        operations=operations,
        headline="AgentCore Identity has no LocalStack emulator path.",
        provider="localstack-contract-only",
        credential_source="none",
        data={"emulator": "not_supported"},
    )


def _run_live_read_only(
    settings: Settings,
    policy: ExecutionPolicy,
    *,
    port: AwsPort | None,
    selected_session: SelectedSession | None,
    boto_port_factory: BotoPortFactory | None = None,
) -> DemoResult:
    active_port = port
    credential_source = _credential_source(selected_session)
    if active_port is None:
        active_port, credential_source = _live_port_from_session(
            settings,
            policy,
            selected_session,
            boto_port_factory=boto_port_factory,
        )
    operations = [_validated_not_run(row) for row in _request_rows()]
    data: dict[str, object] = {
        "request_contracts": [row[1] for row in _request_rows()],
        "decorators": ("bedrock_agentcore.identity.requires_access_token",),
    }
    if active_port is None:
        operations.extend(
            [
                require_port_not_run(
                    service=_CONTROL,
                    operation_name="ListWorkloadIdentities",
                ),
                require_port_not_run(
                    service=_CONTROL,
                    operation_name="ListOauth2CredentialProviders",
                ),
            ],
        )
        data["live_adapter"] = "missing"
    else:
        workload, providers = _list_live_identity(active_port, settings)
        operations.extend((workload[0], providers[0]))
        data["live_workload_identities"] = len(
            cast(
                "Sequence[object]",
                workload[1].get("workloadIdentities", ()),
            ),
        )
        data["live_credential_providers"] = len(
            cast(
                "Sequence[object]",
                providers[1].get("credentialProviders", ()),
            ),
        )
    return _identity_result(
        execution="live",
        settings=settings,
        operations=operations,
        headline="Live AgentCore Identity path lists read-only resources.",
        provider="bedrock-agentcore-control",
        credential_source=credential_source,
        data=data,
    )


def create_workload_identity_request() -> dict[str, Any]:
    """Return a valid CreateWorkloadIdentity request fixture."""
    return {
        "allowedResourceOauth2ReturnUrls": [
            "https://example.com/awsai/callback",
        ],
        "name": "awsai-demo-workload",
        "tags": {"project": "aws-ai-stack-workshop"},
    }


def create_oauth2_credential_provider_request() -> dict[str, Any]:
    """Return a valid CreateOauth2CredentialProvider request fixture."""
    return {
        "credentialProviderVendor": "CustomOauth2",
        "name": "awsai-demo-oauth",
        "oauth2ProviderConfigInput": {
            "customOauth2ProviderConfig": {
                "oauthDiscovery": {
                    "discoveryUrl": (
                        "https://example.com/.well-known/openid-configuration"
                    ),
                },
            },
        },
        "tags": {"project": "aws-ai-stack-workshop"},
    }


def get_workload_access_token_request() -> dict[str, Any]:
    """Return a valid GetWorkloadAccessToken request fixture."""
    return {"workloadName": "awsai-demo-workload"}


def get_resource_oauth2_token_request() -> dict[str, Any]:
    """Return a valid GetResourceOauth2Token request fixture."""
    return {
        "oauth2Flow": "M2M",
        "resourceCredentialProviderName": "awsai-demo-oauth",
        "scopes": ["support:read"],
        "workloadIdentityToken": "workload-token-redacted",
    }


def list_workload_identities_request() -> dict[str, Any]:
    """Return a read-only ListWorkloadIdentities request."""
    return {"maxResults": 10}


def list_oauth2_credential_providers_request() -> dict[str, Any]:
    """Return a read-only ListOauth2CredentialProviders request."""
    return {"maxResults": 10}


def _list_live_identity(
    port: AwsPort,
    settings: Settings,
) -> tuple[
    tuple[OperationOutcome, Mapping[str, Any]],
    tuple[OperationOutcome, Mapping[str, Any]],
]:
    workload = _live_read_operation(
        port=port,
        settings=settings,
        operation_name="ListWorkloadIdentities",
        params=list_workload_identities_request(),
    )
    providers = _live_read_operation(
        port=port,
        settings=settings,
        operation_name="ListOauth2CredentialProviders",
        params=list_oauth2_credential_providers_request(),
    )
    return workload, providers


def _live_read_operation(
    *,
    port: AwsPort,
    settings: Settings,
    operation_name: str,
    params: Mapping[str, Any],
) -> tuple[OperationOutcome, Mapping[str, Any]]:
    demo_operation = port_operation(
        port=port,
        service=_CONTROL,
        operation_name=operation_name,
        params=params,
        execution="live",
        effect="read",
        fixture_id=_fixture_id(operation_name),
        endpoint_url=_control_endpoint(settings),
    )
    return demo_operation.outcome, demo_operation.payload


def _identity_result(
    *,
    execution: Execution,
    settings: Settings,
    operations: Sequence[OperationOutcome],
    headline: str,
    provider: str,
    credential_source: CredentialSource,
    data: Mapping[str, object],
) -> DemoResult:
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        requested_execution=execution,
        headline=headline,
        operations=operations,
        evidence=evidence(
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
            provider=provider,
            region=settings.region,
            credential_source=credential_source,
            fixture_id=next(
                (
                    item["fixture_id"]
                    for item in operations
                    if item["fixture_id"]
                ),
                None,
            )
            if execution != "live"
            else None,
            packages=_package_versions(
                ("bedrock-agentcore", "botocore"),
            ),
        ),
        data=dict(data),
    )


def _validated_not_run(
    row: tuple[str, str, Mapping[str, Any]],
) -> OperationOutcome:
    service, name, params = row
    validate_request(
        service=service,
        operation_name=name,
        params=params,
        fixture_id=_fixture_id(name),
    )
    return not_run_operation(
        service=service,
        operation_name=name,
        error_code="contract_only",
        request_validated=True,
    )


def _request_rows() -> tuple[tuple[str, str, dict[str, Any]], ...]:
    return (
        (
            _CONTROL,
            "CreateWorkloadIdentity",
            create_workload_identity_request(),
        ),
        (
            _CONTROL,
            "CreateOauth2CredentialProvider",
            create_oauth2_credential_provider_request(),
        ),
        (
            _RUNTIME,
            "GetWorkloadAccessToken",
            get_workload_access_token_request(),
        ),
        (
            _RUNTIME,
            "GetResourceOauth2Token",
            get_resource_oauth2_token_request(),
        ),
    )


def _fixture_id(operation_name: str) -> str:
    return f"{FIXTURE_PREFIX}-{operation_name}-v1"


def _live_port_from_session(
    settings: Settings,
    policy: ExecutionPolicy,
    selected_session: SelectedSession | None,
    *,
    boto_port_factory: BotoPortFactory | None = None,
) -> tuple[AwsPort | None, CredentialSource]:
    if selected_session is None:
        default_port = resolve_boto_port(
            execution="live",
            settings=settings,
            policy=policy,
            factory=boto_port_factory,
        )
        if default_port is None:
            return None, "none"
        return default_port.port, default_port.credential_source
    return (
        BotoAwsPort(
            session=cast("BotoSessionPort", selected_session.session),
            region_name=selected_session.region,
            config=bounded_boto_config(policy),
        ),
        selected_session.source,
    )


def _control_endpoint(settings: Settings) -> str:
    return f"https://bedrock-agentcore-control.{settings.region}.amazonaws.com"


def _credential_source(
    selected_session: SelectedSession | None,
) -> CredentialSource:
    if selected_session is None:
        return "none"
    return selected_session.source


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions
