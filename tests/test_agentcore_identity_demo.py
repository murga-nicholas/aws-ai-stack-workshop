from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from botocore.exceptions import ClientError, EndpointConnectionError

import awsai_demo.agentcore_identity_demo as identity
from awsai_demo.agentcore_identity_demo import (
    create_oauth2_credential_provider_request,
    create_workload_identity_request,
    get_resource_oauth2_token_request,
    get_workload_access_token_request,
    list_oauth2_credential_providers_request,
    list_workload_identities_request,
    run_agentcore_identity_demo,
    validate_agentcore_identity_contracts,
)
from awsai_demo.credentials import SelectedSession
from awsai_demo.demo_support import AwsResponse
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pytest


class FakePort:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse | Mapping[str, Any]:
        self.calls.append((service, operation_name, dict(params)))
        if operation_name == "ListWorkloadIdentities":
            return AwsResponse(
                {"workloadIdentities": [{"name": "workload"}]},
                "https://bedrock-agentcore-control.us-east-1.amazonaws.com",
            )
        return {"credentialProviders": [{"name": "provider"}]}


class RaisingPort:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse:
        del service, operation_name, params
        raise self.exc


class DefaultPort:
    def __init__(self, port: object) -> None:
        self.port = port
        self.credential_source = "env"


class FakeClient:
    class Meta:
        """Fake botocore metadata."""

        endpoint_url = (
            "https://bedrock-agentcore-control.us-west-2.amazonaws.com"
        )

    meta = Meta()

    def list_workload_identities(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        assert kwargs == {"maxResults": 10}
        return {"workloadIdentities": []}

    def list_oauth2_credential_providers(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        assert kwargs == {"maxResults": 10}
        return {"credentialProviders": [{"name": "provider"}]}


class FakeSession:
    def __init__(self) -> None:
        self.services: list[str] = []
        self.kwargs: dict[str, object] | None = None

    def client(self, service_name: str, **kwargs: object) -> FakeClient:
        self.services.append(service_name)
        self.kwargs = dict(kwargs)
        return FakeClient()


def test_offline_identity_contracts_explain_token_flows() -> None:
    built = run_agentcore_identity_demo(execution="offline")

    assert built["mode"] == "local_contract"
    assert built["status"] == "ok"
    data = built["data"]
    assert isinstance(data, dict)
    assert "requires_access_token" in data["decorators"][0]
    assert data["request_contracts"] == [
        "CreateWorkloadIdentity",
        "CreateOauth2CredentialProvider",
        "GetWorkloadAccessToken",
        "GetResourceOauth2Token",
    ]


def test_identity_request_builders_validate_against_botocore() -> None:
    operations = validate_agentcore_identity_contracts()

    assert [op["operation"] for op in operations] == [
        "CreateWorkloadIdentity",
        "CreateOauth2CredentialProvider",
        "GetWorkloadAccessToken",
        "GetResourceOauth2Token",
    ]
    assert create_workload_identity_request()["name"]
    assert (
        create_oauth2_credential_provider_request()["credentialProviderVendor"]
        == "CustomOauth2"
    )
    assert get_workload_access_token_request() == {
        "workloadName": "awsai-demo-workload",
    }
    assert get_resource_oauth2_token_request()["oauth2Flow"] == "M2M"
    assert list_workload_identities_request() == {"maxResults": 10}
    assert list_oauth2_credential_providers_request() == {"maxResults": 10}


def test_emulator_and_live_missing_port_are_explicit() -> None:
    emulator = run_agentcore_identity_demo(execution="emulator")
    live = run_agentcore_identity_demo(
        execution="live",
        boto_port_factory=lambda **_kwargs: None,
    )

    assert emulator["mode"] == "not_run"
    assert {op["error_code"] for op in emulator["operations"]} == {
        "not_supported_by_emulator",
    }
    assert live["mode"] == "not_run"
    live_data = cast("dict[str, Any]", live["data"])
    assert live_data["live_adapter"] == "missing"
    assert {op["fixture_id"] for op in live["operations"]} == {None}


def test_live_injected_port_and_selected_session() -> None:
    port = FakePort()
    injected = run_agentcore_identity_demo(execution="live", port=port)

    assert injected["mode"] == "live_service"
    injected_data = cast("dict[str, Any]", injected["data"])
    assert injected_data["live_workload_identities"] == 1
    assert injected_data["live_credential_providers"] == 1
    assert [call[1] for call in port.calls] == [
        "ListWorkloadIdentities",
        "ListOauth2CredentialProviders",
    ]

    session = FakeSession()
    selected = SelectedSession(
        session=session,
        source="profile",
        region="us-west-2",
    )
    boto = run_agentcore_identity_demo(
        execution="live",
        settings=Settings(region="us-west-2"),
        selected_session=selected,
    )
    assert boto["evidence"]["credential_source"] == "profile"
    boto_data = cast("dict[str, Any]", boto["data"])
    assert boto_data["live_credential_providers"] == 1
    assert session.services == [
        "bedrock-agentcore-control",
        "bedrock-agentcore-control",
    ]
    assert session.kwargs is not None
    assert session.kwargs["region_name"] == "us-west-2"


def test_live_default_port_and_failed_identity_reads() -> None:
    defaulted = run_agentcore_identity_demo(
        execution="live",
        boto_port_factory=lambda **_kwargs: DefaultPort(FakePort()),
    )
    assert defaulted["evidence"]["credential_source"] == "env"

    denied = run_agentcore_identity_demo(
        execution="live",
        port=RaisingPort(
            ClientError(
                {
                    "Error": {"Code": "AccessDeniedException"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                "ListWorkloadIdentities",
            ),
        ),
    )
    assert denied["mode"] == "live_service"
    assert denied["operations"][-2]["error_code"] == "authorization_denied"

    failed = run_agentcore_identity_demo(
        execution="live",
        port=RaisingPort(
            EndpointConnectionError(endpoint_url="https://example.com"),
        ),
    )
    assert failed["mode"] == "attempt_failed"
    assert failed["operations"][-1]["error_code"] == "timeout"


def test_package_version_missing_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast("Any", identity)
    monkeypatch.setattr(
        module.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(
            module.metadata.PackageNotFoundError,
        ),
    )

    assert identity._package_versions(("missing",)) == {
        "missing": "missing",
    }
