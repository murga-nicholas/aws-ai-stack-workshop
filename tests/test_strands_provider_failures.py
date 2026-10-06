from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from strands.types.exceptions import (
    EventLoopException,
    ModelThrottledException,
)

from awsai_demo import billing
from awsai_demo.bounded_bedrock import BoundedBedrockModel
from awsai_demo.decision_demo import (
    ResumeRequest,
    resume_from_disk,
    run_decision_demo,
)
from awsai_demo.decision_engine import DecisionModel
from awsai_demo.decision_state import policy_fingerprint
from awsai_demo.live_agent import LiveAgentModel
from awsai_demo.offline import ScriptedModel, text_script
from awsai_demo.policy import ExecutionPolicy, ReservationUnavailable
from awsai_demo.providers import ModelSelection
from awsai_demo.runtime import TESTED_LIVE_BEDROCK_MODEL, Settings
from awsai_demo.strands_agent_demo import run_strands_agent_demo
from awsai_demo.strands_errors import unwrap_strands_error
from awsai_demo.strands_multiagent_demo import (
    count_tokens_model_id,
    run_strands_multiagent_demo,
)

if TYPE_CHECKING:
    from pathlib import Path


class Prices:
    def rate(self, _feature: str, **_kwargs: Any) -> Decimal:
        return Decimal("0.000001")


class Counter:
    def count_tokens(self, _model_id: str, _request: bytes) -> int:
        return 100


def budget(root: Path) -> billing.BudgetRun:
    return billing.open_budget_run(
        root=root,
        region="us-east-1",
        policy=ExecutionPolicy(),
        prices=Prices(),
    )


def test_unwrap_known_exceptions_and_cycles() -> None:
    original = OSError("fixture")
    wrapped = ModelThrottledException("fixture")
    wrapped.__cause__ = original
    assert unwrap_strands_error(EventLoopException(wrapped)) is original
    assert unwrap_strands_error(original) is original
    isolated = ModelThrottledException("no cause")
    assert unwrap_strands_error(isolated) is isolated
    cyclic = EventLoopException(original)
    cyclic.original_exception = cyclic
    assert unwrap_strands_error(cyclic) is cyclic


@pytest.mark.parametrize("prefix", ["", "global.", "us.", "eu.", "apac."])
def test_count_tokens_uses_foundation_model_identifier(prefix: str) -> None:
    model = "amazon.nova-2-lite-v1:0"
    assert count_tokens_model_id(prefix + model) == model


@pytest.mark.parametrize(
    ("code", "message", "status"),
    [
        ("ThrottlingException", "throttled fixture", 429),
        ("ValidationException", "Input is too long for requested model", 400),
        (
            "ValidationException",
            "Conversation blocks and tool result blocks "
            "cannot be provided in the same turn.",
            400,
        ),
    ],
)
def test_real_bedrock_sdk_failure_preserves_reservation_without_extra_retry(
    tmp_path: Path,
    code: str,
    message: str,
    status: int,
) -> None:
    model = BoundedBedrockModel(
        model_id=TESTED_LIVE_BEDROCK_MODEL,
        boto_session=boto3.Session(
            aws_access_key_id="test",
            aws_secret_access_key="test",  # noqa: S106
            region_name="us-east-1",
        ),
        max_tokens=256,
        use_native_token_count=False,
    )
    messages: Any = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "one",
                        "name": "price",
                        "input": {},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "one",
                        "content": [{"text": "19680"}],
                    }
                }
            ],
        },
        {"role": "user", "content": [{"text": "Continue."}]},
    ]
    tools: Any = [
        {
            "name": "price",
            "description": "Price a pilot",
            "inputSchema": {"json": {"type": "object"}},
        }
    ]
    formatted = model.format_request(messages, tools)
    assert [item["role"] for item in formatted["messages"]] == [
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    run = budget(tmp_path)
    observed = LiveAgentModel(
        demo="sdk-regression",
        settings=Settings(),
        policy=run.policy,
        run=run,
        model_factory=lambda **_kwargs: ModelSelection(
            model, "bedrock", TESTED_LIVE_BEDROCK_MODEL
        ),
    )
    with Stubber(model.client) as stubber:
        stubber.add_response(
            "count_tokens",
            {"inputTokens": 200},
            {
                "modelId": count_tokens_model_id(TESTED_LIVE_BEDROCK_MODEL),
                "input": {
                    "converse": {
                        key: formatted[key]
                        for key in ("messages", "system", "toolConfig")
                    }
                },
            },
        )
        stubber.add_client_error(
            "converse_stream",
            service_error_code=code,
            service_message=message,
            http_status_code=status,
            expected_params=formatted,
        )

        async def consume() -> None:
            async for _event in observed.stream(messages, tools):
                pass

        with pytest.raises(ClientError) as failure:
            asyncio.run(consume())
        stubber.assert_no_pending_responses()
    assert failure.value.response["Error"]["Code"] == code
    outcome = observed.operations[-1]
    assert outcome["mode"] == "live_service"
    assert outcome["http_status"] == status
    assert outcome["error_code"] == code
    assert outcome["reserved_usd"] > 0
    assert run.command("sdk-regression").ledger.snapshot().model_calls == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_unknown_provider_exceptions_are_not_relabelled(
    tmp_path: Path,
    asynchronous: bool,
) -> None:
    class Model(ScriptedModel):
        async def fail(self) -> Any:
            yield {}
            message = "local implementation defect"
            raise TypeError(message)

        def stream(self, *_args: Any, **_kwargs: Any) -> Any:
            if asynchronous:
                return self.fail()
            message = "local implementation defect"
            raise TypeError(message)

    model = LiveAgentModel(
        demo="unknown",
        settings=Settings(),
        policy=ExecutionPolicy(),
        run=budget(tmp_path),
        exact_counter=Counter(),
        model_factory=lambda **_kwargs: ModelSelection(
            Model(()), "bedrock", TESTED_LIVE_BEDROCK_MODEL
        ),
    )

    async def consume() -> None:
        async for _event in model.stream([]):
            pass

    with pytest.raises(TypeError, match="implementation defect"):
        asyncio.run(consume())


@pytest.mark.parametrize(
    "command", [run_strands_agent_demo, run_strands_multiagent_demo]
)
def test_demo_handles_documented_framework_boundary(
    tmp_path: Path,
    command: Any,
) -> None:
    def unavailable(**_kwargs: Any) -> Any:
        raise EventLoopException(ModelThrottledException("fixture"))

    with billing.use_budget_run(budget(tmp_path)):
        outcome = command(
            execution="live",
            model_factory=unavailable,
            exact_counter=Counter(),
        )
    assert outcome["status"] == "blocked"
    assert outcome["error"]["code"] == "model_unavailable"


def test_decision_preserves_billed_output_that_did_not_interrupt(
    tmp_path: Path,
) -> None:
    with billing.use_budget_run(budget(tmp_path / "budget")):
        outcome = run_decision_demo(
            execution="live",
            checkpoint_root=tmp_path / "state",
            exact_counter=Counter(),
            model_factory=lambda **_kwargs: ModelSelection(
                ScriptedModel(text_script("I did not request approval.")),
                "bedrock",
                TESTED_LIVE_BEDROCK_MODEL,
            ),
        )
    assert outcome["mode"] == "live_model"
    assert outcome["status"] == "blocked"
    assert outcome["error"]["code"] == "validation_failed"
    assert outcome["operations"][0]["reserved_usd"] > 0


@pytest.mark.parametrize("price_missing", [True, False])
def test_resume_failure_retains_effect_and_structured_evidence(
    tmp_path: Path,
    price_missing: bool,
) -> None:
    run = budget(tmp_path / "budget")
    root = tmp_path / "state"
    with billing.use_budget_run(run):
        pause = run_decision_demo(
            execution="live",
            checkpoint_root=root,
            exact_counter=Counter(),
            model_factory=lambda **_kwargs: ModelSelection(
                DecisionModel(run.policy),
                "bedrock",
                TESTED_LIVE_BEDROCK_MODEL,
            ),
        )
    request = ResumeRequest(
        root=str(root),
        run_id=pause["data"]["run_id"],
        action="approve",
        approver="Pat",
        approval_source="cli",
        execution="live",
        model_id=TESTED_LIVE_BEDROCK_MODEL,
        policy_fingerprint=policy_fingerprint(run.policy),
    )

    class FailingModel(ScriptedModel):
        def stream(self, *_args: Any, **_kwargs: Any) -> Any:
            raise EventLoopException(
                ClientError(
                    {
                        "Error": {"Code": "ThrottlingException"},
                        "ResponseMetadata": {"HTTPStatusCode": 429},
                    },
                    "Converse",
                )
            )

    def worker_model(*_args: Any) -> LiveAgentModel:
        if price_missing:
            message = "missing prices"
            raise ReservationUnavailable(message)
        return LiveAgentModel(
            demo="decision",
            settings=Settings(),
            policy=run.policy,
            run=run,
            exact_counter=Counter(),
            model_factory=lambda **_kwargs: ModelSelection(
                FailingModel(()),
                "bedrock",
                TESTED_LIVE_BEDROCK_MODEL,
            ),
        )

    failed = resume_from_disk(request, resume_model=worker_model)
    assert failed.status == "blocked"
    assert failed.effects == (0 if price_missing else 1)
    assert any(op["status"] == "blocked" for op in failed.operations)
    if not price_missing:
        paid = [op for op in failed.operations if op["reserved_usd"]]
        assert paid[0]["mode"] == "live_service"
        replay = resume_from_disk(
            request.model_copy(update={"action": "replay"})
        )
        assert replay.effects == 1


def test_offline_resume_preserves_local_model_failure(tmp_path: Path) -> None:
    pause = run_decision_demo(checkpoint_root=tmp_path)
    request = ResumeRequest(
        root=str(tmp_path),
        run_id=pause["data"]["run_id"],
        action="approve",
        approver="Pat",
        approval_source="cli",
        policy_fingerprint=policy_fingerprint(ExecutionPolicy()),
    )

    class BrokenModel(ScriptedModel):
        def stream(self, *_args: Any, **_kwargs: Any) -> Any:
            message = "local fixture failed"
            raise ValueError(message)

    with pytest.raises((ValueError, EventLoopException), match="fixture"):
        resume_from_disk(request, model=BrokenModel(()))
