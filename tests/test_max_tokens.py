"""A capped model answer stays live, blocked, and partially used."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError
from strands.types.exceptions import (
    EventLoopException,
    MaxTokensReachedException,
)

from awsai_demo import billing
from awsai_demo.decision_demo import run_decision_demo
from awsai_demo.offline import ScriptedModel
from awsai_demo.policy import ExecutionPolicy, Reservation
from awsai_demo.providers import ModelSelection
from awsai_demo.runtime import TESTED_LIVE_BEDROCK_MODEL
from awsai_demo.strands_agent_demo import run_strands_agent_demo
from awsai_demo.strands_errors import unwrap_strands_error
from awsai_demo.strands_multiagent_demo import (
    ReservedLiveCall,
    flag_max_tokens,
    run_strands_multiagent_demo,
)

if TYPE_CHECKING:
    from pathlib import Path

_OUTPUT_CAP = "output cap"


class Prices:
    def rate(self, _feature: str, **_kwargs: Any) -> Decimal:
        return Decimal("0.000001")


class Counter:
    def count_tokens(self, _model_id: str, _request: bytes) -> int:
        return 12


def budget(root: Path) -> billing.BudgetRun:
    return billing.open_budget_run(
        root=root,
        region="us-east-1",
        policy=ExecutionPolicy(),
        prices=Prices(),
    )


class CappedModel(ScriptedModel):
    def __init__(self, *, during_stream: bool) -> None:
        super().__init__(())
        self.during_stream = during_stream

    async def _capped_stream(self) -> Any:
        yield {
            "metadata": {
                "usage": {
                    "inputTokens": 12,
                    "outputTokens": 512,
                    "totalTokens": 524,
                }
            }
        }
        raise MaxTokensReachedException(_OUTPUT_CAP)

    def stream(self, *_args: Any, **_kwargs: Any) -> Any:
        if self.during_stream:
            return self._capped_stream()
        raise MaxTokensReachedException(_OUTPUT_CAP)


def _factory(during_stream: bool) -> Any:
    def build(**_kwargs: Any) -> ModelSelection:
        return ModelSelection(
            CappedModel(during_stream=during_stream),
            "bedrock",
            TESTED_LIVE_BEDROCK_MODEL,
        )

    return build


def _reservation() -> Reservation:
    return Reservation(
        "reserved",
        Decimal("0.01"),
        "model",
        Decimal("0.01"),
        Decimal("0"),
    )


def test_unwrap_keeps_max_tokens_even_when_a_client_error_is_nested() -> None:
    capped = MaxTokensReachedException(_OUTPUT_CAP)
    capped.original_exception = ClientError(
        {
            "Error": {"Code": "ThrottlingException"},
            "ResponseMetadata": {"HTTPStatusCode": 429},
        },
        "Converse",
    )
    assert unwrap_strands_error(capped) is capped
    assert unwrap_strands_error(EventLoopException(capped)) is capped


def test_flag_retags_only_the_latest_answered_call() -> None:
    usage = {"input_tokens": 12, "output_tokens": 512}
    answered = ReservedLiveCall("one", _reservation(), 12, usage, None)
    earlier = ReservedLiveCall("zero", _reservation(), 12, usage, "timeout")
    calls = [earlier, answered]
    flag_max_tokens(calls)
    assert calls[1].error_code == "max_tokens_reached"
    assert calls[1].usage == usage
    flag_max_tokens(calls)
    assert calls[0].error_code == "timeout"
    skipped = [
        ReservedLiveCall("one", _reservation(), 12, usage, None),
        ReservedLiveCall("two", _reservation(), 12, usage, "timeout"),
    ]
    flag_max_tokens(skipped)
    assert skipped[0].error_code == "max_tokens_reached"
    assert skipped[1].error_code == "timeout"
    flag_max_tokens([])


@pytest.mark.parametrize("during_stream", [False, True])
@pytest.mark.parametrize(
    "command",
    [run_strands_agent_demo, run_strands_multiagent_demo, run_decision_demo],
)
def test_demos_keep_partial_usage_when_output_is_capped(
    tmp_path: Path,
    during_stream: bool,
    command: Any,
) -> None:
    kwargs: dict[str, Any] = {
        "execution": "live",
        "exact_counter": Counter(),
        "model_factory": _factory(during_stream),
    }
    if command is run_decision_demo:
        kwargs["checkpoint_root"] = tmp_path / "state"
    with billing.use_budget_run(budget(tmp_path / "budget")):
        outcome = command(**kwargs)
    assert outcome["status"] == "blocked"
    assert outcome["mode"] == "live_model"
    assert outcome["error"]["code"] == "max_tokens_reached"
    capped = [
        item
        for item in outcome["operations"]
        if item["error_code"] == "max_tokens_reached"
    ]
    assert len(capped) == 1
    assert capped[0]["mode"] == "live_model"
    assert capped[0]["status"] == "blocked"
    if during_stream:
        assert capped[0]["usage"] == {
            "input_tokens": 12,
            "output_tokens": 512,
            "total_tokens": 524,
        }
    else:
        assert capped[0]["usage"] is None
    assert all(
        item["error_code"] != "model_unavailable"
        for item in outcome["operations"]
    )
