import asyncio
import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import BaseModel

import awsai_demo.offline as offline_module
from awsai_demo.offline import (
    ScriptedModel,
    ScriptedUsage,
    strands_available,
    text_script,
)

if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def test_scripted_model_stream_replays_deep_copies() -> None:
    async def collect() -> tuple[
        list[dict[str, object]], list[dict[str, object]]
    ]:
        model = ScriptedModel([{"event": {"items": []}}], temperature=0)

        first = [event async for event in model.stream([])]
        first[0]["event"]["items"].append("mutated")
        second = [event async for event in model.stream([])]
        assert model.get_config() == {"temperature": 0}
        return first, second

    _, second = asyncio.run(collect())

    assert second == [{"event": {"items": []}}]


def test_scripted_model_update_config_and_fixture_id() -> None:
    model = ScriptedModel([{"event": {"items": []}}], temperature=0)

    model.update_config(max_tokens=10)

    assert model.fixture_id == "strands-script-default-v1"
    assert model.get_config() == {"temperature": 0, "max_tokens": 10}


def test_scripted_model_structured_output_with_plain_class() -> None:
    class Box:
        def __init__(self, value: str) -> None:
            self.value = value

    async def collect() -> list[dict[str, Box]]:
        model = ScriptedModel([{"structuredOutput": {"value": "ok"}}])
        return [
            event async for event in model.structured_output(Box, prompt=[])
        ]

    outputs = asyncio.run(collect())

    assert outputs[0]["output"].value == "ok"


def test_scripted_model_structured_output_with_pydantic_model() -> None:
    class Box(BaseModel):
        value: str

    async def collect() -> list[dict[str, Box]]:
        model = ScriptedModel([{"structuredOutput": {"value": "ok"}}])
        return [
            event async for event in model.structured_output(Box, prompt=[])
        ]

    outputs = asyncio.run(collect())

    assert outputs[0]["output"].value == "ok"


def test_scripted_model_structured_output_returns_existing_instance() -> None:
    class Box:
        def __init__(self, value: str) -> None:
            self.value = value

    box = Box("ok")

    async def collect() -> list[dict[str, Box]]:
        model = ScriptedModel([{"structuredOutput": box}])
        return [
            event async for event in model.structured_output(Box, prompt=[])
        ]

    outputs = asyncio.run(collect())

    assert outputs[0]["output"].value == "ok"
    assert outputs[0]["output"] is not box


def test_scripted_model_structured_output_defaults_to_empty_payload() -> None:
    class EmptyBox:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    async def collect() -> list[dict[str, EmptyBox]]:
        model = ScriptedModel([{"messageStop": {"stopReason": "end_turn"}}])
        return [
            event
            async for event in model.structured_output(EmptyBox, prompt=[])
        ]

    outputs = asyncio.run(collect())

    assert outputs[0]["output"].kwargs == {}


def test_text_script_includes_usage_only_when_present() -> None:
    without_usage = text_script("hello")
    with_usage = text_script(
        "hello",
        usage=ScriptedUsage(input_tokens=1, output_tokens=2, total_tokens=3),
    )

    assert without_usage[-1] == {"messageStop": {"stopReason": "end_turn"}}
    assert with_usage[-1]["metadata"]["usage"]["totalTokens"] == 3


def test_text_script_closes_block_so_real_strands_retains_answer() -> None:
    from strands import Agent

    response = Agent(
        model=ScriptedModel(text_script("Approval required.")),
        callback_handler=None,
        retry_strategy=None,
    )("Price it")
    assert str(response).strip() == "Approval required."
    assert response.message["content"] == [{"text": "Approval required."}]


def test_scripted_model_is_strands_subclass_when_available() -> None:
    if not strands_available():
        assert ScriptedModel.__mro__[-1] is object
        return

    from strands.models.model import Model

    assert issubclass(ScriptedModel, Model)


def test_module_falls_back_when_strands_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_hook(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "strands.models.model":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_hook)
    module = _load_offline_module("awsai_demo_offline_no_strands")

    assert module.strands_available() is False
    assert module.ScriptedModel.__mro__[-1] is object


def _load_offline_module(module_name: str) -> ModuleType:
    module_path = Path(str(offline_module.__file__))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
