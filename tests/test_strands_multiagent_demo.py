from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

import awsai_demo.strands_multiagent_demo as multi
from awsai_demo.credentials import SelectedSession
from awsai_demo.offline import ScriptedModel, ScriptedUsage, text_script
from awsai_demo.policy import (
    ExecutionPolicy,
    PolicyLimitExceeded,
    Reservation,
    TokenCount,
)
from awsai_demo.providers import ModelSelection, SdkMissingError
from awsai_demo.runtime import TESTED_LIVE_BEDROCK_MODEL, Settings


class ExactCounter:
    def count_tokens(
        self,
        model_id: str,
        serialized_request: bytes,
    ) -> int:
        assert model_id == TESTED_LIVE_BEDROCK_MODEL
        expected = f'"modelId":"{TESTED_LIVE_BEDROCK_MODEL}"'.encode()
        assert expected in serialized_request
        return 41


class FakeBudget:
    def __init__(self) -> None:
        self.labels: list[str] = []
        self.routings: list[str] = []

    def reserve_model(
        self,
        reservation_id: str,
        *,
        model_id: str,
        serialized_request: bytes,
        exact_counter: ExactCounter,
        routing: str = "in-region",
    ) -> tuple[Reservation, TokenCount]:
        tokens = exact_counter.count_tokens(model_id, serialized_request)
        self.labels.append(reservation_id)
        self.routings.append(routing)
        return (
            Reservation(
                reservation_id=reservation_id,
                amount_usd=Decimal("0.000001"),
                kind="model",
                total_reserved_usd=Decimal("0.000001"),
                cleanup_reserved_usd=Decimal("0"),
            ),
            TokenCount(model_id, tokens, "exact", "unit-test"),
        )


class FakeRun:
    def __init__(self, budget: FakeBudget) -> None:
        self.budget = budget

    def command(self, name: str) -> FakeBudget:
        assert name == "strands-multiagent"
        return self.budget


class FakeBedrockModel(ScriptedModel):
    def __init__(self) -> None:
        super().__init__(
            text_script(
                "live response",
                usage=ScriptedUsage(11, 7, 18),
            ),
        )

    def format_request(
        self,
        *args: object,
        **kwargs: object,
    ) -> dict[str, Any]:
        assert args or kwargs
        return {"modelId": TESTED_LIVE_BEDROCK_MODEL, "messages": []}


class FakeCountClient:
    class Meta:
        """Fake botocore metadata."""

        endpoint_url = "https://bedrock-runtime.us-east-1.amazonaws.com"

    meta = Meta()

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def count_tokens(self, **kwargs: object) -> dict[str, object]:
        self.requests.append(dict(kwargs))
        return {"inputTokens": 41}


class FakeCountBedrockModel(FakeBedrockModel):
    def __init__(self, client: FakeCountClient) -> None:
        super().__init__()
        self.client = client


def test_offline_runs_real_graph_swarm_and_tool_agent() -> None:
    built = multi.run_strands_multiagent_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(max_model_calls=6, max_agent_iterations=6),
    )

    assert built["mode"] == "local_contract"
    assert built["status"] == "ok"
    assert built["evidence"]["fixture_id"] == multi.FIXTURE_ID
    data = cast("dict[str, Any]", built["data"])
    assert data["graph_nodes"] == ("researcher", "costing", "reviewer")
    assert data["swarm_nodes"] == ("researcher", "reviewer")
    assert data["model_calls"] == (
        "graph.researcher",
        "graph.costing",
        "graph.reviewer",
        "swarm.researcher",
        "swarm.reviewer",
        "tool.reviewer",
    )
    assert data["observed_framework_calls"] == {
        "graph": 3,
        "swarm": 2,
        "agent_as_tool": 1,
    }
    assert [op["mode"] for op in built["operations"]].count(
        "local_contract",
    ) == 6
    assert [op["mode"] for op in built["operations"]].count(
        "local_execution",
    ) == 3


def test_emulator_and_provider_refusals_are_explicit() -> None:
    emulator = multi.run_strands_multiagent_demo(execution="emulator")
    provider = multi.run_strands_multiagent_demo(
        execution="live",
        settings=Settings(provider="offline"),
    )

    assert emulator["mode"] == "not_run"
    assert {op["error_code"] for op in emulator["operations"]} == {
        "not_supported_by_emulator",
    }
    assert provider["mode"] == "not_run"
    provider_error = provider["error"]
    assert provider_error is not None
    assert provider_error["code"] == "validation_failed"


def test_live_refuses_unbounded_model_calls() -> None:
    def fake_factory(**_kwargs: object) -> ModelSelection:
        return ModelSelection(
            model=object(),
            provider="bedrock",
            model_id=None,
        )

    built = multi.run_strands_multiagent_demo(
        execution="live",
        settings=Settings(),
        model_factory=fake_factory,
    )

    assert built["mode"] == "not_run"
    assert built["status"] == "blocked"
    built_error = built["error"]
    assert built_error is not None
    assert built_error["code"] == "budget_exceeded"
    assert built["evidence"]["requested_model"] == TESTED_LIVE_BEDROCK_MODEL


def test_live_without_counter_uses_sdk_count_tokens() -> None:
    budget = FakeBudget()
    client = FakeCountClient()

    def fake_factory(**_kwargs: object) -> ModelSelection:
        return ModelSelection(
            model=FakeCountBedrockModel(client),
            provider="bedrock",
            model_id=TESTED_LIVE_BEDROCK_MODEL,
        )

    built = multi.run_strands_multiagent_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(max_model_calls=6, max_agent_iterations=6),
        model_factory=fake_factory,
        budget_opener=lambda **_kwargs: FakeRun(budget),
    )

    assert built["mode"] == "live_model"
    assert len(client.requests) == 6
    assert {op["operation"] for op in built["operations"]} >= {
        "CountTokens",
        "BedrockModel.stream:tool.reviewer",
    }


def test_live_with_counter_reserves_before_fake_bedrock_dispatch() -> None:
    budget = FakeBudget()

    def fake_factory(**_kwargs: object) -> ModelSelection:
        return ModelSelection(
            model=FakeBedrockModel(),
            provider="bedrock",
            model_id=TESTED_LIVE_BEDROCK_MODEL,
        )

    built = multi.run_strands_multiagent_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(max_model_calls=6, max_agent_iterations=6),
        exact_counter=ExactCounter(),
        model_factory=fake_factory,
        budget_opener=lambda **_kwargs: FakeRun(budget),
    )

    assert built["mode"] == "live_model"
    assert built["evidence"]["estimated_cost_usd"] == 0.000006
    assert budget.labels == [
        "strands-multiagent:1:graph.researcher",
        "strands-multiagent:2:graph.costing",
        "strands-multiagent:3:graph.reviewer",
        "strands-multiagent:4:swarm.researcher",
        "strands-multiagent:5:swarm.reviewer",
        "strands-multiagent:6:tool.reviewer",
    ]
    assert budget.routings == ["cross-region-global"] * 6
    data = cast("dict[str, Any]", built["data"])
    assert data["model_calls"] == (
        "graph.researcher",
        "graph.costing",
        "graph.reviewer",
        "swarm.researcher",
        "swarm.reviewer",
        "tool.reviewer",
    )
    live_ops = [op for op in built["operations"] if op["mode"] == "live_model"]
    assert len(live_ops) == 6
    usages = [cast("dict[str, int]", op["usage"]) for op in live_ops]
    assert {usage["total_tokens"] for usage in usages} == {18}


def test_dispatch_limits_and_hook_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = iter((0.0, 999.0))
    limits = multi.DispatchLimits(
        ExecutionPolicy(max_wall_seconds=1),
        clock=lambda: next(calls),
    )
    with pytest.raises(PolicyLimitExceeded):
        limits.before_dispatch()
    limits = multi.DispatchLimits(ExecutionPolicy(max_agent_iterations=1))
    limits.before_dispatch("agent")
    with pytest.raises(PolicyLimitExceeded):
        limits.before_dispatch("agent")

    hook = multi.SwarmHandoffHook()
    hook.handoff_triggered = True
    hook.before_node(cast("Any", SimpleNamespace(node_id="researcher")))
    assert hook.handoff_triggered is True

    module = cast("Any", multi)
    monkeypatch.setattr(
        module.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(
            module.metadata.PackageNotFoundError,
        ),
    )
    assert multi._package_versions(("missing-package",)) == {
        "missing-package": "missing",
    }


def test_additional_live_and_helper_branches() -> None:
    limits = multi.DispatchLimits(ExecutionPolicy(max_model_calls=1))
    limits.before_dispatch()
    with pytest.raises(PolicyLimitExceeded):
        limits.before_dispatch()

    class ConfigModel:
        def __init__(self) -> None:
            self.updated: dict[str, object] = {}

        def get_config(self) -> dict[str, object]:
            return {"temperature": 0}

        def update_config(self, **model_config: object) -> None:
            self.updated.update(model_config)

    config_model = ConfigModel()
    reserving = multi.ReservingModel(
        label="x",
        inner=config_model,
        model_id=TESTED_LIVE_BEDROCK_MODEL,
        calls=[],
        completed=[],
        charged=[],
        count_operations=[],
        limits=multi.DispatchLimits(ExecutionPolicy()),
        budget=object(),
        exact_counter=ExactCounter(),
        max_output_tokens=1,
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )
    assert reserving.get_config() == {"temperature": 0}
    reserving.update_config(top_p=1)
    assert config_model.updated == {"top_p": 1}

    def missing_factory(**_kwargs: object) -> ModelSelection:
        package = "strands-agents"
        raise SdkMissingError(package)

    missing = multi.run_strands_multiagent_demo(
        execution="live",
        exact_counter=ExactCounter(),
        model_factory=missing_factory,
    )
    assert missing["error"] is not None
    assert missing["error"]["code"] == "sdk_missing"

    class FailingModel:
        def format_request(
            self,
            *args: object,
            **kwargs: object,
        ) -> dict[str, Any]:
            assert args or kwargs
            return {"modelId": TESTED_LIVE_BEDROCK_MODEL, "messages": []}

        def stream(self, *_args: object, **_kwargs: object) -> object:
            message = "network down"
            raise OSError(message)

    def failing_factory(**_kwargs: object) -> ModelSelection:
        return ModelSelection(
            model=FailingModel(),
            provider="bedrock",
            model_id=TESTED_LIVE_BEDROCK_MODEL,
        )

    failed = multi.run_strands_multiagent_demo(
        execution="live",
        exact_counter=ExactCounter(),
        model_factory=failing_factory,
        budget_opener=lambda **_kwargs: FakeRun(FakeBudget()),
    )
    assert failed["error"] is not None
    assert failed["error"]["code"] == "timeout"
    assert failed["evidence"]["estimated_cost_usd"] == 0.000001
    failed_ops = failed["operations"]
    assert [op["operation"] for op in failed_ops] == [
        "BedrockModel.stream:graph.researcher",
    ]
    assert failed_ops[0]["reserved_usd"] == 0.000001

    client_denied = multi.run_strands_multiagent_demo(
        execution="live",
        exact_counter=ExactCounter(),
        model_factory=lambda **_kwargs: ModelSelection(
            model=ClientErrorModel(),
            provider="bedrock",
            model_id=TESTED_LIVE_BEDROCK_MODEL,
        ),
        budget_opener=lambda **_kwargs: FakeRun(FakeBudget()),
    )
    assert client_denied["error"] is not None
    assert client_denied["error"]["code"] == "model_unavailable"

    class FakeSession:
        pass

    selected = SelectedSession(
        session=FakeSession(),
        source="profile",
        region="us-east-1",
    )
    selection, source = multi._live_model_selection(
        settings=Settings(),
        policy=ExecutionPolicy(),
        model_id=TESTED_LIVE_BEDROCK_MODEL,
        model_factory=lambda **_kwargs: ModelSelection(
            model=FakeBedrockModel(),
            provider="bedrock",
            model_id=TESTED_LIVE_BEDROCK_MODEL,
        ),
        selected_session=selected,
    )
    assert selection.provider == "bedrock"
    assert source == "profile"

    selected_by_helper, helper_source = multi._live_model_selection(
        settings=Settings(),
        policy=ExecutionPolicy(),
        model_id=TESTED_LIVE_BEDROCK_MODEL,
        model_factory=None,
        selected_session=None,
        session_selector=lambda **_kwargs: selected,
        model_builder=lambda **_kwargs: ModelSelection(
            model=FakeBedrockModel(),
            provider="bedrock",
            model_id=TESTED_LIVE_BEDROCK_MODEL,
        ),
    )
    assert selected_by_helper.model_id == TESTED_LIVE_BEDROCK_MODEL
    assert helper_source == "profile"

    serialized = multi._serialized_model_request(
        inner=object(),
        model_id=TESTED_LIVE_BEDROCK_MODEL,
        messages=[],
        tool_specs=None,
        system_prompt=None,
        tool_choice=None,
        system_prompt_content=None,
        invocation_state=None,
        max_tokens=1,
    )
    assert b'"modelId"' in serialized.body

    assert multi._usage_from_event({}) is None
    assert multi._usage_from_event({"metadata": {}}) is None
    assert multi._usage_from_event(
        cast("Any", {"metadata": {"usage": {"inputTokens": 1}}}),
    ) == {"input_tokens": 1}
    assert multi._usage_from_event(
        cast("Any", {"metadata": {"usage": {"outputTokens": 2}}}),
    ) == {"output_tokens": 2}
    assert multi._usage_from_event(
        cast(
            "Any",
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 1,
                        "outputTokens": 2,
                        "totalTokens": 3,
                    },
                },
            },
        ),
    ) == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    assert multi._model_routing("us.amazon.nova") == "cross-region"
    assert multi._model_routing("amazon.nova") == "in-region"
    empty_config = multi.ReservingModel(
        label="x",
        inner=object(),
        model_id=TESTED_LIVE_BEDROCK_MODEL,
        calls=[],
        completed=[],
        charged=[],
        count_operations=[],
        limits=multi.DispatchLimits(ExecutionPolicy()),
        budget=object(),
        exact_counter=ExactCounter(),
        max_output_tokens=1,
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )
    assert empty_config.get_config() == {}
    empty_config.update_config(top_p=1)

    class PolicyBudget:
        def reserve_model(self, *_args: object, **_kwargs: object) -> object:
            message = "policy denied"
            raise PolicyLimitExceeded(message)

    denied_reservation = multi.ReservingModel(
        label="x",
        inner=FakeBedrockModel(),
        model_id=TESTED_LIVE_BEDROCK_MODEL,
        calls=[],
        completed=[],
        charged=[],
        count_operations=[],
        limits=multi.DispatchLimits(ExecutionPolicy()),
        budget=PolicyBudget(),
        exact_counter=ExactCounter(),
        max_output_tokens=1,
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )
    with pytest.raises(PolicyLimitExceeded):
        denied_reservation.stream([], None, None)

    assert (
        multi._live_failed_result(
            settings=Settings(),
            model_id=TESTED_LIVE_BEDROCK_MODEL,
            provider="bedrock",
            observed_model=None,
            credential_source="none",
            live_calls=(),
            charged_calls=(),
            operations=(),
            code="timeout",
            message="no attempt",
        )["mode"]
        == "not_run"
    )


def test_live_count_and_async_failure_paths() -> None:
    class CountFailCounter(ExactCounter):
        endpoint_url = "https://bedrock-runtime.us-east-1.amazonaws.com"

        def count_tokens(
            self,
            model_id: str,
            serialized_request: bytes,
        ) -> int:
            super().count_tokens(model_id, serialized_request)
            raise EndpointConnectionError(endpoint_url=self.endpoint_url)

    count_failed = multi.run_strands_multiagent_demo(
        execution="live",
        exact_counter=CountFailCounter(),
        model_factory=lambda **_kwargs: ModelSelection(
            model=FakeBedrockModel(),
            provider="bedrock",
            model_id=TESTED_LIVE_BEDROCK_MODEL,
        ),
        budget_opener=lambda **_kwargs: FakeRun(FakeBudget()),
    )
    assert count_failed["mode"] == "attempt_failed"
    assert count_failed["operations"][0]["operation"] == "CountTokens"

    class AsyncFailModel(FakeBedrockModel):
        async def _fail(self) -> Any:
            yield {"messageStart": {"role": "assistant"}}
            message = "stream broke"
            raise OSError(message)

        def stream(self, *_args: object, **_kwargs: object) -> Any:
            return self._fail()

    async_failed = multi.run_strands_multiagent_demo(
        execution="live",
        exact_counter=ExactCounter(),
        model_factory=lambda **_kwargs: ModelSelection(
            model=AsyncFailModel(),
            provider="bedrock",
            model_id=TESTED_LIVE_BEDROCK_MODEL,
        ),
    )
    assert async_failed["status"] == "blocked"
    assert async_failed["operations"][0]["operation"].startswith(
        "BedrockModel.stream",
    )

    denied = multi._live_failure_operation(
        operation_name="CountTokens",
        effect="read",
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        exc=ClientError(
            {
                "Error": {"Code": "AccessDeniedException"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "CountTokens",
        ),
    )
    assert denied["mode"] == "live_service"
    assert denied["error_code"] == "authorization_denied"


class ClientErrorModel:
    def format_request(
        self,
        *args: object,
        **kwargs: object,
    ) -> dict[str, Any]:
        assert args or kwargs
        return {"modelId": TESTED_LIVE_BEDROCK_MODEL, "messages": []}

    def stream(self, *_args: object, **_kwargs: object) -> object:
        raise ClientError(
            {
                "Error": {"Code": "AccessDeniedException"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "Converse",
        )
