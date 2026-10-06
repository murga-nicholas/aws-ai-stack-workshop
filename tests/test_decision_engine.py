from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

from awsai_demo.decision_engine import DecisionModel, build_agent
from awsai_demo.offline import ScriptedModel, text_script
from awsai_demo.policy import ExecutionPolicy, PolicyLimitExceededError

if TYPE_CHECKING:
    from pathlib import Path


def collect(model: DecisionModel, messages: list) -> list:
    async def run() -> list:
        return [event async for event in model.stream(messages)]

    return asyncio.run(run())


def test_script_drives_tool_then_response_and_limits() -> None:
    model = DecisionModel(ExecutionPolicy(max_model_calls=2))
    request = collect(model, [])
    assert request[-1]["messageStop"]["stopReason"] == "tool_use"
    answer = collect(
        model,
        [{"role": "user", "content": [{"toolResult": {"toolUseId": "x"}}]}],
    )
    assert answer[-1]["messageStop"]["stopReason"] == "end_turn"
    assert all(op["mode"] == "local_contract" for op in model.operations)
    with pytest.raises(PolicyLimitExceededError, match="iteration limit"):
        collect(model, [])
    expired = DecisionModel(ExecutionPolicy(max_wall_seconds=1))
    expired.started = time.monotonic() - 2
    with pytest.raises(PolicyLimitExceededError, match="wall-clock"):
        collect(expired, [])


def test_delegate_is_observed_as_emulator_without_network() -> None:
    model = DecisionModel(
        ExecutionPolicy(),
        delegate=ScriptedModel(text_script("fake emulator port")),
        endpoint="http://localhost:4566",
    )
    assert collect(model, [])
    assert model.operations[0]["mode"] == "local_emulator"


def test_non_offline_cannot_fall_back_to_script(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="real provider"):
        build_agent(
            directory=tmp_path,
            run_id="abcdef123456",
            acceptance_tool=object(),
            policy=ExecutionPolicy(),
            execution="emulator",
        )
