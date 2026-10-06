from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from awsai_demo.demo_support import AwsResponse
from awsai_demo.nova_act_demo import _run_live as run_live_rows
from awsai_demo.nova_act_demo import (
    create_workflow_definition_params,
    create_workflow_run_params,
    list_models_params,
    run_nova_act_demo,
)
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping


def _data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", result["data"])


class NovaPort:
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
            {
                "modelSummaries": [
                    {
                        "modelId": "amazon.nova-act-v1:0",
                        "modelLifecycle": {"status": "ACTIVE"},
                        "minimumCompatibilityVersion": 1,
                    },
                    {
                        "modelId": "amazon.nova-act-preview:0",
                        "modelLifecycle": {"status": "PREVIEW"},
                        "minimumCompatibilityVersion": 1,
                    },
                ],
                "modelAliases": [],
                "compatibilityInformation": {
                    "clientCompatibilityVersion": 1,
                    "supportedModelIds": ["amazon.nova-act-v1:0"],
                },
            },
            self.endpoint_url,
        )


class NovaSession:
    def __init__(self) -> None:
        self.client_instance = NovaClient()
        self.clients: list[tuple[str, dict[str, object]]] = []

    def client(self, service_name: str, **kwargs: object) -> NovaClient:
        self.clients.append((service_name, kwargs))
        self.client_instance.meta = NovaMeta(
            str(kwargs.get("endpoint_url", "https://nova-act.aws")),
        )
        return self.client_instance


class NovaClient:
    def __init__(self) -> None:
        self.meta = NovaMeta("https://nova-act.us-east-1.amazonaws.com")
        self.port = NovaPort(self.meta.endpoint_url)

    def list_models(self, **kwargs: Any) -> dict[str, object]:
        return dict(self.port.call("nova-act", "ListModels", kwargs).payload)


class NovaMeta:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url


def test_nova_act_offline_uses_strict_fixtures() -> None:
    result = run_nova_act_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["mode"] == "local_contract"
    assert result["status"] == "ok"
    assert [item["operation"] for item in result["operations"]] == [
        "ListModels",
        "CreateWorkflowDefinition",
        "CreateWorkflowRun",
    ]
    assert _data(result)["model_count"] == 1
    assert _data(result)["workflow_run_status"] == "RUNNING"


def test_nova_act_emulator_is_unsupported() -> None:
    result = run_nova_act_demo(
        execution="emulator",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["mode"] == "not_run"
    assert len(result["operations"]) == 3
    assert {item["error_code"] for item in result["operations"]} == {
        "not_supported_by_emulator",
    }


def test_nova_act_live_lists_models_with_port() -> None:
    port = NovaPort("https://nova-act.us-east-1.amazonaws.com")
    result = run_nova_act_demo(
        execution="live",
        settings=Settings(region="us-east-1"),
        policy=ExecutionPolicy(),
        port=port,
    )

    assert result["mode"] == "live_service"
    assert result["status"] == "ok"
    assert [call[1] for call in port.calls] == ["ListModels"]
    assert _data(result)["model_count"] == 2
    assert result["operations"][1]["operation"] == "CreateWorkflowDefinition"
    assert result["operations"][1]["error_code"] == "contract_only"


def test_nova_act_live_missing_and_default_port() -> None:
    operations, data = run_live_rows(Settings(), None)
    session = NovaSession()

    result = run_nova_act_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        session_factory=lambda **_kwargs: session,
    )

    assert operations[0]["error_code"] == "missing_configuration"
    assert data["model_count"] == 0
    assert result["evidence"]["credential_source"] == "default_chain"
    assert session.clients[0][0] == "nova-act"


def test_nova_act_request_helpers() -> None:
    assert list_models_params() == {"clientCompatibilityVersion": 1}
    assert create_workflow_definition_params() == {
        "name": "awsaiDemoWorkflow",
    }
    run_params = create_workflow_run_params()
    assert run_params["workflowDefinitionName"] == "awsaiDemoWorkflow"
    assert (
        cast("Mapping[str, int]", run_params["clientInfo"])[
            "compatibilityVersion"
        ]
        == 1
    )
