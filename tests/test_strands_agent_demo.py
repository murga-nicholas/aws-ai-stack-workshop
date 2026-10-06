from __future__ import annotations

from typing import Any, cast

import pytest
from botocore.exceptions import ClientError

from awsai_demo.policy import ExecutionPolicy, PolicyLimitExceeded
from awsai_demo.providers import ModelSelection, SdkMissingError
from awsai_demo.runtime import TESTED_LIVE_BEDROCK_MODEL, Settings
from awsai_demo.strands_agent_demo import (
    FIXTURE_ID,
    DispatchLimits,
    PricePilotScriptedModel,
    _forced_tool_name,
    _latest_message_has_tool_result,
    _package_versions,
    build_agent,
    proposal_recommendation,
    run_strands_agent_demo,
)


def emulator_settings() -> Settings:
    return Settings(localstack_auth_token="".join(("unit", "test")))


def test_strands_agent_runs_real_tool_loop() -> None:
    built = run_strands_agent_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(max_agent_iterations=4),
    )

    assert built["mode"] == "local_contract"
    assert built["evidence"]["fixture_id"] == FIXTURE_ID
    assert built["data"]["summary"]["total_usd"] == 19680
    assert built["data"]["summary"]["headroom_usd"] == 5320
    assert built["data"]["model_calls"] == (
        "price_tool",
        "final_answer",
        "structured_output",
    )
    assert [op["mode"] for op in built["operations"]].count(
        "local_execution",
    ) == 2
    assert [op["mode"] for op in built["operations"]].count(
        "local_contract",
    ) == 3


def test_strands_agent_live_refuses_unpriced_model_call() -> None:
    from awsai_demo.policy import ReservationUnavailable

    def missing_prices(**_kwargs: object) -> object:
        message = "The injected snapshot is unavailable"
        raise ReservationUnavailable(message)

    built = run_strands_agent_demo(
        execution="live",
        settings=Settings(),
        budget_opener=missing_prices,
    )

    assert built["mode"] == "not_run"
    assert built["status"] == "blocked"
    assert built["error"]["code"] == "budget_exceeded"
    assert built["evidence"]["provider"] == "bedrock"
    assert built["evidence"]["requested_model"] == TESTED_LIVE_BEDROCK_MODEL
    assert built["operations"][0]["error_code"] == "budget_exceeded"


def test_strands_agent_live_openai_budget_refusal_keeps_provider() -> None:
    built = run_strands_agent_demo(
        execution="live",
        settings=Settings(provider="openai", model="gpt-test"),
    )

    assert built["mode"] == "not_run"
    assert built["evidence"]["provider"] == "openai"
    assert built["evidence"]["requested_model"] == "gpt-test"
    assert built["operations"][0]["service"] == "openai"


def test_strands_agent_refuses_provider_for_wrong_lane() -> None:
    built = run_strands_agent_demo(
        execution="offline",
        settings=Settings(provider="openai"),
    )
    emulator = run_strands_agent_demo(
        execution="emulator",
        settings=Settings(provider="openai"),
    )
    live = run_strands_agent_demo(
        execution="live",
        settings=Settings(provider="offline"),
    )

    assert built["mode"] == "not_run"
    assert built["error"]["code"] == "validation_failed"
    assert emulator["error"]["code"] == "validation_failed"
    assert live["error"]["code"] == "validation_failed"


def test_strands_agent_emulator_requires_localstack_token() -> None:
    built = run_strands_agent_demo(execution="emulator", settings=Settings())

    assert built["mode"] == "not_run"
    assert built["error"]["code"] == "missing_configuration"


def test_strands_agent_emulator_uses_injected_provider_model() -> None:
    def fake_factory(**_kwargs: object) -> ModelSelection:
        return ModelSelection(
            model=PricePilotScriptedModel(),
            provider="bedrock-emulator",
            model_id="ollama.fake",
        )

    built = run_strands_agent_demo(
        execution="emulator",
        settings=emulator_settings(),
        policy=ExecutionPolicy(max_agent_iterations=4),
        model_factory=fake_factory,
    )

    assert built["mode"] == "local_emulator"
    assert built["evidence"]["emulator"] == {
        "endpoint": "http://localhost:4566"
    }
    assert built["evidence"]["requested_model"] == "ollama.fake"
    assert built["data"]["summary_type"] == "PilotSummary"
    assert built["data"]["model_calls"] == (
        "BedrockModel.stream",
        "BedrockModel.stream",
        "BedrockModel.stream",
    )


def test_strands_agent_emulator_reports_sdk_missing() -> None:
    def fake_factory(**_kwargs: object) -> ModelSelection:
        package = "strands-agents"
        raise SdkMissingError(package)

    built = run_strands_agent_demo(
        execution="emulator",
        settings=emulator_settings(),
        model_factory=fake_factory,
    )

    assert built["mode"] == "not_run"
    assert built["error"]["code"] == "sdk_missing"


def test_strands_agent_emulator_reports_provider_configuration_error() -> None:
    def fake_factory(**_kwargs: object) -> ModelSelection:
        message = "bad provider setup"
        raise ValueError(message)

    built = run_strands_agent_demo(
        execution="emulator",
        settings=emulator_settings(),
        model_factory=fake_factory,
    )

    assert built["mode"] == "not_run"
    assert built["error"]["code"] == "missing_configuration"


def test_strands_agent_emulator_reports_preflight_limit_refusal() -> None:
    def fake_factory(**_kwargs: object) -> ModelSelection:
        return ModelSelection(
            model=PricePilotScriptedModel(),
            provider="bedrock-emulator",
            model_id="ollama.fake",
        )

    built = run_strands_agent_demo(
        execution="emulator",
        settings=emulator_settings(),
        policy=ExecutionPolicy(max_model_calls=1, max_agent_iterations=4),
        model_factory=fake_factory,
    )

    assert built["mode"] == "local_emulator"
    assert built["error"]["code"] == "budget_exceeded"
    assert built["data"]["model_calls"] == ("BedrockModel.stream",)
    assert built["data"]["tool_calls"] == ("price_pilot",)
    assert [op["mode"] for op in built["operations"]] == [
        "local_emulator",
        "local_execution",
        "local_execution",
        "not_run",
    ]


def test_strands_agent_emulator_reports_client_error_response() -> None:
    class DeniedModel:
        stateful = False

        def stream(self, *_args: object, **_kwargs: object) -> object:
            raise ClientError(
                {
                    "Error": {"Code": "AccessDeniedException"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                "Converse",
            )

    def fake_factory(**_kwargs: object) -> ModelSelection:
        return ModelSelection(
            model=DeniedModel(),
            provider="bedrock-emulator",
            model_id="ollama.fake",
        )

    built = run_strands_agent_demo(
        execution="emulator",
        settings=emulator_settings(),
        model_factory=fake_factory,
    )

    assert built["mode"] == "local_emulator"
    assert built["error"]["code"] == "authorization_denied"
    assert built["operations"][0]["response_received"] is True


def test_strands_agent_emulator_reports_failed_dispatch() -> None:
    class FailingModel:
        stateful = False

        def stream(self, *_args: object, **_kwargs: object) -> object:
            message = "localstack unavailable"
            raise OSError(message)

    def fake_factory(**_kwargs: object) -> ModelSelection:
        return ModelSelection(
            model=FailingModel(),
            provider="bedrock-emulator",
            model_id="ollama.fake",
        )

    built = run_strands_agent_demo(
        execution="emulator",
        settings=emulator_settings(),
        model_factory=fake_factory,
    )

    assert built["mode"] == "attempt_failed"
    assert built["error"]["code"] == "emulator_unavailable"
    assert built["operations"][0]["response_received"] is False


def test_strands_agent_emulator_reraises_unknown_local_errors() -> None:
    class BuggyModel:
        stateful = False

        def stream(self, *_args: object, **_kwargs: object) -> object:
            message = "local coding bug"
            raise RuntimeError(message)

    def fake_factory(**_kwargs: object) -> ModelSelection:
        return ModelSelection(
            model=BuggyModel(),
            provider="bedrock-emulator",
            model_id="ollama.fake",
        )

    with pytest.raises(RuntimeError, match="local coding bug"):
        run_strands_agent_demo(
            execution="emulator",
            settings=emulator_settings(),
            model_factory=fake_factory,
        )


def test_build_agent_uses_framework_tool_and_structured_output() -> None:
    model = PricePilotScriptedModel()
    agent = build_agent(model)

    result = agent("price the pilot", limits={"turns": 4})

    assert result.structured_output is not None
    assert result.structured_output.total_usd == 19680
    assert model.calls == ["price_tool", "final_answer", "structured_output"]


def test_strands_agent_raises_when_framework_omits_summary() -> None:
    class NoSummaryModel(PricePilotScriptedModel):
        def stream(self, *args: object, **kwargs: object) -> object:
            self.calls.append("final_answer")
            return super().stream(*args, **kwargs)

    def fake_agent(model: object) -> object:
        del model

        class EmptyResultAgent:
            def __call__(self, *_args: object, **_kwargs: object) -> object:
                class EmptyResult:
                    structured_output = None

                return EmptyResult()

        return EmptyResultAgent()

    with pytest.raises(TypeError, match="typed PilotSummary"):
        run_strands_agent_demo(
            execution="offline",
            settings=Settings(),
            scripted_model=NoSummaryModel,
            agent_builder=fake_agent,
        )


def test_proposal_recommendation_rejects_unaffordable_cost() -> None:
    from awsai_demo import scenario

    text = proposal_recommendation(scenario.price_pilot(budget=10))

    assert text.startswith("Do not proceed")


def test_scripted_model_defensive_helpers_cover_invalid_inputs() -> None:
    assert _latest_message_has_tool_result([]) is False
    assert _forced_tool_name(cast("Any", {"tool": "price_pilot"})) is None
    assert _forced_tool_name(cast("Any", {"tool": {}})) is None
    assert _package_versions(("not-a-real-awsai-demo-package",)) == {
        "not-a-real-awsai-demo-package": "missing",
    }


def test_dispatch_limits_refuse_before_excess_dispatch() -> None:
    limits = DispatchLimits(ExecutionPolicy(max_model_calls=1))

    limits.before_dispatch()

    with pytest.raises(PolicyLimitExceeded, match="model-call"):
        limits.before_dispatch()


def test_dispatch_limits_refuse_agent_iteration_and_wall_time() -> None:
    iteration_limits = DispatchLimits(
        ExecutionPolicy(max_model_calls=4, max_agent_iterations=1),
    )
    iteration_limits.before_dispatch()
    with pytest.raises(PolicyLimitExceeded, match="agent-iteration"):
        iteration_limits.before_dispatch()

    ticks = iter((0.0, 2.0))
    wall_limits = DispatchLimits(
        ExecutionPolicy(max_wall_seconds=1),
        clock=lambda: next(ticks),
    )
    with pytest.raises(PolicyLimitExceeded, match="wall-clock"):
        wall_limits.before_dispatch()
