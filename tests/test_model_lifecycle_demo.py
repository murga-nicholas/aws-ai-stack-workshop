from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from botocore.exceptions import ClientError

from awsai_demo.demo_support import AwsResponse, validate_request
from awsai_demo.model_lifecycle_demo import (
    _catalog_data,
    _lineage_model_count,
    get_foundation_model_params,
    run_model_lifecycle_demo,
)
from awsai_demo.model_lifecycle_demo import (
    _run_live as _run_model_lifecycle_live,
)
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pytest


def _data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", result["data"])


class LifecyclePort:
    def __init__(self, *, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse:
        self.calls.append((service, operation_name, params))
        if operation_name == "ListFoundationModels":
            payload: Mapping[str, Any] = {
                "modelSummaries": [
                    {
                        "modelId": "amazon.nova-2-lite-v1:0",
                        "providerName": "Amazon",
                        "modelLifecycle": {"status": "ACTIVE"},
                    },
                    {
                        "modelId": "legacy.model",
                        "providerName": "Other",
                        "modelLifecycle": {"status": "LEGACY"},
                    },
                ]
            }
        elif operation_name == "GetFoundationModel":
            payload = {
                "modelDetails": {
                    "modelId": params["modelIdentifier"],
                    "modelLifecycle": {
                        "status": "ACTIVE",
                        "startOfLifeTime": datetime(2025, 10, 15, tzinfo=UTC),
                    },
                }
            }
        else:
            payload = {
                "inferenceProfileSummaries": [
                    {"inferenceProfileId": "global.amazon.nova-2-lite-v1:0"},
                    {"inferenceProfileId": "eu.amazon.nova-2-lite-v1:0"},
                ]
            }
        return AwsResponse(payload, self.endpoint_url)


class DeniedLifecyclePort(LifecyclePort):
    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse:
        self.calls.append((service, operation_name, params))
        raise ClientError(
            {
                "Error": {"Code": "AccessDeniedException"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            operation_name,
        )


class ProfileDeniedLifecyclePort(LifecyclePort):
    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse:
        if operation_name == "ListInferenceProfiles":
            self.calls.append((service, operation_name, params))
            raise ClientError(
                {
                    "Error": {"Code": "AccessDeniedException"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                operation_name,
            )
        return super().call(service, operation_name, params)


class LifecycleSession:
    def __init__(self, *, endpoint_url: str) -> None:
        self.clients: list[tuple[str, dict[str, object]]] = []
        self.client_instance = LifecycleClient(endpoint_url=endpoint_url)

    def client(self, service_name: str, **kwargs: object) -> LifecycleClient:
        self.clients.append((service_name, kwargs))
        endpoint = kwargs.get(
            "endpoint_url",
            self.client_instance.meta.endpoint_url,
        )
        self.client_instance.meta = LifecycleMeta(str(endpoint))
        return self.client_instance


class LifecycleClient:
    def __init__(self, *, endpoint_url: str) -> None:
        self.meta = LifecycleMeta(endpoint_url)
        self.calls: list[tuple[str, Mapping[str, Any]]] = []
        self.port = LifecyclePort(endpoint_url=endpoint_url)

    def list_foundation_models(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(("list_foundation_models", kwargs))
        return dict(
            self.port.call("bedrock", "ListFoundationModels", kwargs).payload
        )

    def get_foundation_model(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(("get_foundation_model", kwargs))
        return dict(
            self.port.call("bedrock", "GetFoundationModel", kwargs).payload
        )

    def list_inference_profiles(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(("list_inference_profiles", kwargs))
        return dict(
            self.port.call("bedrock", "ListInferenceProfiles", kwargs).payload
        )


class LifecycleMeta:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url


def test_model_lifecycle_offline_counts_fixture_catalog() -> None:
    result = run_model_lifecycle_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["status"] == "ok"
    assert result["mode"] == "local_contract"
    assert _data(result)["model_count"] == 2
    assert _data(result)["active_count"] == 1
    assert _data(result)["legacy_count"] == 1
    assert _data(result)["global_profiles"] == 1
    assert _data(result)["foundation_lifecycle"]["startOfLifeTime"] == (
        "2025-12-02T00:00:00+00:00"
    )
    json.dumps(result)


def test_model_lifecycle_emulator_missing_and_injected_ports() -> None:
    missing = run_model_lifecycle_demo(
        execution="emulator",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )
    port = LifecyclePort(endpoint_url="http://localhost:4566")
    answered = run_model_lifecycle_demo(
        execution="emulator",
        settings=Settings(localstack_endpoint="http://localhost:4566"),
        policy=ExecutionPolicy(),
        port=port,
    )

    assert missing["mode"] == "local_execution"
    assert missing["status"] == "blocked"
    assert missing["operations"][0]["error_code"] == "missing_configuration"
    assert answered["mode"] == "local_emulator"
    assert [call[1] for call in port.calls] == [
        "ListFoundationModels",
        "GetFoundationModel",
    ]
    assert _data(answered)["provider_count"] == 2
    assert _data(answered)["global_profiles"] == 0


def test_model_lifecycle_emulator_builds_default_port() -> None:
    session = LifecycleSession(endpoint_url="http://localhost:4566")

    def factory(**kwargs: str) -> LifecycleSession:
        assert kwargs["region_name"] == "us-east-1"
        return session

    result = run_model_lifecycle_demo(
        execution="emulator",
        settings=Settings(
            localstack_auth_token="token",  # noqa: S106
            default_bedrock_model="qwen-test",
        ),
        policy=ExecutionPolicy(),
        session_factory=factory,
    )

    assert result["mode"] == "local_emulator"
    assert session.clients[0][1]["endpoint_url"] == "http://localhost:4566"
    assert session.client_instance.calls[1][1]["modelIdentifier"] == (
        "ollama.qwen-test"
    )


def test_model_lifecycle_live_read_only_rows_use_port() -> None:
    port = LifecyclePort(
        endpoint_url="https://bedrock.us-east-1.amazonaws.com"
    )
    result = run_model_lifecycle_demo(
        execution="live",
        settings=Settings(region="us-east-1"),
        policy=ExecutionPolicy(),
        port=port,
    )

    assert result["mode"] == "live_service"
    assert result["status"] == "ok"
    assert _data(result)["region"] == "us-east-1"
    assert _data(result)["global_profiles"] == 1
    assert _data(result)["eu_profiles"] == 1
    assert [call[1] for call in port.calls] == [
        "ListFoundationModels",
        "GetFoundationModel",
        "ListInferenceProfiles",
    ]
    assert "read_at" in _data(result)
    assert _data(result)["catalog_read_succeeded"] is True
    assert _data(result)["foundation_lifecycle"]["startOfLifeTime"] == (
        "2025-10-15T00:00:00+00:00"
    )
    json.dumps(result)


def test_model_lifecycle_live_without_port_reports_no_catalog() -> None:
    operations, data = _run_model_lifecycle_live(Settings(), None)

    assert operations[0]["operation"] == "ListFoundationModels"
    assert operations[0]["error_code"] == "missing_configuration"
    assert data["catalog_read_succeeded"] is False
    assert "model_count" not in data


def test_model_lifecycle_live_profile_failure_keeps_catalog_counts() -> None:
    result = run_model_lifecycle_demo(
        execution="live",
        settings=Settings(region="us-east-1"),
        policy=ExecutionPolicy(),
        port=ProfileDeniedLifecyclePort(
            endpoint_url="https://bedrock.us-east-1.amazonaws.com"
        ),
    )
    data = _data(result)

    assert data["catalog_read_succeeded"] is True
    assert data["model_count"] == 2
    assert "global_profiles" not in data


def test_model_lifecycle_emulator_model_id_fallbacks() -> None:
    prefixed_port = LifecyclePort(endpoint_url="http://localhost:4566")
    run_model_lifecycle_demo(
        execution="emulator",
        settings=Settings(default_bedrock_model="ollama.existing"),
        policy=ExecutionPolicy(),
        port=prefixed_port,
    )
    configured_settings = Settings()
    object.__setattr__(
        configured_settings,
        "localstack_bedrock_model",
        "ollama.configured",
    )
    configured_port = LifecyclePort(endpoint_url="http://localhost:4566")
    run_model_lifecycle_demo(
        execution="emulator",
        settings=configured_settings,
        policy=ExecutionPolicy(),
        port=configured_port,
    )

    assert prefixed_port.calls[1][2]["modelIdentifier"] == "ollama.existing"
    assert configured_port.calls[1][2]["modelIdentifier"] == (
        "ollama.configured"
    )


def test_model_lifecycle_live_builds_default_read_port() -> None:
    session = LifecycleSession(
        endpoint_url="https://bedrock.us-east-1.amazonaws.com"
    )

    def factory(**kwargs: str) -> LifecycleSession:
        assert kwargs == {"region_name": "us-east-1"}
        return session

    result = run_model_lifecycle_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        session_factory=factory,
    )

    assert result["mode"] == "live_service"
    assert result["status"] == "ok"
    assert _data(result)["model_count"] == 2
    assert session.client_instance.calls[0][0] == "list_foundation_models"
    assert _data(result)["foundation_lifecycle"]["startOfLifeTime"] == (
        "2025-10-15T00:00:00+00:00"
    )
    json.dumps(result)


def test_model_lifecycle_live_failed_read_does_not_invent_counts() -> None:
    result = run_model_lifecycle_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=DeniedLifecyclePort(
            endpoint_url="https://bedrock.us-east-1.amazonaws.com"
        ),
    )

    assert result["status"] == "blocked"
    assert _data(result)["catalog_read_succeeded"] is False
    assert "model_count" not in _data(result)


def test_model_lifecycle_helpers_validate_shapes_and_edge_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_request(
        service="bedrock",
        operation_name="GetFoundationModel",
        params=get_foundation_model_params(),
        fixture_id="get-model",
    )
    data = _catalog_data(
        {"modelSummaries": [{"modelLifecycle": "unknown"}]},
        {"modelDetails": {}},
        {"inferenceProfileSummaries": [{"inferenceProfileId": "apac.model"}]},
    )
    monkeypatch.chdir(tmp_path)
    missing = _lineage_model_count()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "lineage.yaml").write_text("[]\n", encoding="utf-8")

    assert data["active_count"] == 0
    assert data["apac_profiles"] == 1
    assert missing == 0
    assert _lineage_model_count() == 0
