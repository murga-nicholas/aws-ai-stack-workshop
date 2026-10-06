from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from awsai_demo.demo_support import AwsResponse
from awsai_demo.model_customization_demo import _run_live as run_live_rows
from awsai_demo.model_customization_demo import (
    custom_model_deployment_params,
    customization_job_params,
    list_custom_models_params,
    model_import_params,
    run_model_customization_demo,
)
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping


def _data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", result["data"])


class CustomizationPort:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse:
        self.calls.append((service, operation_name, params))
        return AwsResponse(
            {"modelSummaries": [{"modelName": "custom-one"}]},
            self.endpoint_url,
        )


class CustomizationSession:
    def __init__(self) -> None:
        self.client_instance = CustomizationClient()
        self.clients: list[tuple[str, dict[str, object]]] = []

    def client(
        self,
        service_name: str,
        **kwargs: object,
    ) -> CustomizationClient:
        self.clients.append((service_name, kwargs))
        self.client_instance.meta = CustomizationMeta(
            str(kwargs.get("endpoint_url", "https://bedrock.aws")),
        )
        return self.client_instance


class CustomizationClient:
    def __init__(self) -> None:
        self.meta = CustomizationMeta(
            "https://bedrock.us-east-1.amazonaws.com"
        )
        self.port = CustomizationPort(self.meta.endpoint_url)

    def list_custom_models(self, **kwargs: Any) -> dict[str, object]:
        return dict(
            self.port.call("bedrock", "ListCustomModels", kwargs).payload,
        )


class CustomizationMeta:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url


def test_model_customization_offline_validates_all_rows() -> None:
    result = run_model_customization_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["mode"] == "local_contract"
    assert result["status"] == "ok"
    assert [item["operation"] for item in result["operations"]] == [
        "CreateModelCustomizationJob",
        "CreateModelCustomizationJob",
        "CreateModelCustomizationJob",
        "CreateModelImportJob",
        "CreateCustomModelDeployment",
        "ListCustomModels",
    ]
    assert len({item["fixture_id"] for item in result["operations"]}) == 6
    assert _data(result)["contract_job_count"] == 3
    assert _data(result)["custom_model_count"] == 0


def test_model_customization_emulator_is_unsupported() -> None:
    result = run_model_customization_demo(
        execution="emulator",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["mode"] == "not_run"
    assert len(result["operations"]) == 6
    assert {item["error_code"] for item in result["operations"]} == {
        "not_supported_by_emulator",
    }


def test_model_customization_live_lists_read_only_models() -> None:
    port = CustomizationPort("https://bedrock.us-east-1.amazonaws.com")
    result = run_model_customization_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=port,
    )

    assert result["mode"] == "live_service"
    assert result["status"] == "ok"
    assert [call[1] for call in port.calls] == ["ListCustomModels"]
    assert _data(result)["custom_model_count"] == 1
    assert result["operations"][0]["error_code"] == "contract_only"
    assert result["operations"][-1]["mode"] == "live_service"


def test_model_customization_live_missing_and_default_port() -> None:
    operations, data = run_live_rows(Settings(), None)
    session = CustomizationSession()

    result = run_model_customization_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        session_factory=lambda **_kwargs: session,
    )

    assert operations[-1]["error_code"] == "missing_configuration"
    assert data["custom_model_count"] == 0
    assert result["evidence"]["credential_source"] == "default_chain"
    assert session.clients[0][0] == "bedrock"


def test_model_customization_request_helpers() -> None:
    fine = customization_job_params("FINE_TUNING")
    rft = customization_job_params("REINFORCEMENT_FINE_TUNING")

    assert fine["customizationType"] == "FINE_TUNING"
    assert "reinforcement" in cast("str", rft["jobName"])
    assert model_import_params()["modelDataSource"] == {
        "s3DataSource": {"s3Uri": "s3://example-model/source/"},
    }
    assert custom_model_deployment_params()["modelArn"].startswith(
        "arn:aws:bedrock",
    )
    assert list_custom_models_params() == {"maxResults": 10}
