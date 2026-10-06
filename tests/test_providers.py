import builtins
from collections.abc import Callable
from typing import Any, cast

import pytest

from awsai_demo.credentials import SelectedSession
from awsai_demo.offline import ScriptedModel
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.providers import SdkMissing, create_model
from awsai_demo.runtime import (
    TESTED_LIVE_BEDROCK_MODEL,
    Execution,
    ProviderName,
    Settings,
)


class FakeBotoSession:
    region_name = "us-east-1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def client(self, service_name: str, **kwargs: object) -> "FakeClient":
        self.calls.append((service_name, kwargs))
        return FakeClient(kwargs.get("region_name", self.region_name))


class FakeEvents:
    def register(self, event_name: str, handler: object) -> None:
        del event_name, handler


class FakeMeta:
    def __init__(self, region_name: object) -> None:
        self.region_name = str(region_name)
        self.events = FakeEvents()


class FakeClient:
    def __init__(self, region_name: object) -> None:
        self.meta = FakeMeta(region_name)


def test_create_model_offline_returns_scripted_model() -> None:
    selection = create_model(
        provider="offline",
        execution="offline",
        settings=Settings(),
    )

    assert isinstance(selection.model, ScriptedModel)
    assert selection.provider == "offline"
    assert selection.model_id is None


def test_create_bedrock_requires_selected_session() -> None:
    with pytest.raises(ValueError, match="selected session"):
        create_model(
            provider="bedrock",
            execution="live",
            settings=Settings(),
        )


def test_create_emulator_requires_selected_session() -> None:
    with pytest.raises(ValueError, match="selected session"):
        create_model(
            provider="bedrock",
            execution="emulator",
            settings=Settings(),
        )


def test_create_openai_requires_api_key_before_import() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_model(
            provider="openai",
            execution="live",
            settings=Settings(model="gpt-test", openai_api_key=None),
        )


def test_create_openai_requires_requested_model() -> None:
    with pytest.raises(ValueError, match="--model"):
        create_model(
            provider="openai",
            execution="live",
            settings=Settings(openai_api_key="key"),
        )


def test_create_openai_passes_requested_model_or_reports_missing_sdk() -> None:
    try:
        selection = create_model(
            provider="openai",
            execution="live",
            settings=Settings(model="gpt-test", openai_api_key="key"),
            policy=ExecutionPolicy(max_output_tokens=5),
        )
    except SdkMissing as exc:
        assert exc.package == "strands-agents"
    else:
        assert selection.provider == "openai"
        assert selection.model_id == "gpt-test"
        assert selection.model.get_config()["model_id"] == "gpt-test"


def test_create_openai_uses_explicit_model_or_reports_missing_sdk() -> None:
    try:
        selection = create_model(
            provider="openai",
            execution="live",
            settings=Settings(openai_api_key="key"),
            model_id="gpt-explicit",
        )
    except SdkMissing as exc:
        assert exc.package == "strands-agents"
    else:
        assert selection.model_id == "gpt-explicit"


def test_create_bedrock_or_reports_missing_sdk() -> None:
    try:
        selection = create_model(
            provider="bedrock",
            execution="live",
            settings=Settings(default_bedrock_model="amazon.test"),
            selected_session=SelectedSession(
                session=FakeBotoSession(),
                source="none",
                region="us-east-1",
            ),
            policy=ExecutionPolicy(max_output_tokens=5),
        )
    except SdkMissing as exc:
        assert exc.package == "strands-agents"
    else:
        assert selection.provider == "bedrock"
        assert selection.model_id == TESTED_LIVE_BEDROCK_MODEL
        formatted = selection.model.format_request(
            [{"role": "user", "content": [{"text": "Summarize the pilot."}]}]
        )
        assert formatted["inferenceConfig"] == {"maxTokens": 5}


def test_create_bedrock_uses_explicit_model_or_reports_missing_sdk() -> None:
    try:
        selection = create_model(
            provider="bedrock",
            execution="live",
            settings=Settings(model="settings-model"),
            selected_session=SelectedSession(
                session=FakeBotoSession(),
                source="none",
                region="us-east-1",
            ),
            model_id="explicit-model",
        )
    except SdkMissing as exc:
        assert exc.package == "strands-agents"
    else:
        assert selection.model_id == "explicit-model"


def test_create_bedrock_uses_settings_model_or_reports_missing_sdk() -> None:
    try:
        selection = create_model(
            provider="bedrock",
            execution="live",
            settings=Settings(model="settings-model"),
            selected_session=SelectedSession(
                session=FakeBotoSession(),
                source="none",
                region="us-east-1",
            ),
        )
    except SdkMissing as exc:
        assert exc.package == "strands-agents"
    else:
        assert selection.model_id == "settings-model"


def test_create_emulator_prefixes_ollama_model_or_reports_missing_sdk() -> (
    None
):
    session = FakeBotoSession()
    try:
        selection = create_model(
            provider="bedrock",
            execution="emulator",
            settings=Settings(default_bedrock_model="qwen"),
            selected_session=SelectedSession(
                session=session,
                source="none",
                region="us-east-1",
            ),
        )
    except SdkMissing as exc:
        assert exc.package == "strands-agents"
    else:
        assert selection.provider == "bedrock-emulator"
        assert selection.model_id == "ollama.qwen"
        config = session.calls[0][1]["config"]
        assert config.retries["total_max_attempts"] == 2
        assert config.connect_timeout == config.read_timeout == 30


def test_create_model_rejects_unknown_combination() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_model(
            provider="openai",
            execution="offline",
            settings=Settings(openai_api_key="key"),
        )


@pytest.mark.parametrize(
    ("provider", "execution"),
    [
        ("offline", "live"),
        ("offline", "emulator"),
        ("openai", "emulator"),
    ],
)
def test_create_model_rejects_cross_lane_providers(
    provider: str,
    execution: str,
) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_model(
            provider=cast("ProviderName", provider),
            execution=cast("Execution", execution),
            settings=Settings(openai_api_key="key"),
        )


def test_create_model_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        create_model(
            provider=cast("ProviderName", "unknown"),
            execution="live",
            settings=Settings(openai_api_key="key"),
        )


@pytest.mark.parametrize("module_name", ["awsai_demo.bounded_bedrock"])
def test_bedrock_sdk_missing_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    monkeypatch.setattr(
        builtins,
        "__import__",
        _missing_import(module_name),
    )

    with pytest.raises(SdkMissing) as exc_info:
        create_model(
            provider="bedrock",
            execution="live",
            settings=Settings(),
            selected_session=SelectedSession(
                session=FakeBotoSession(),
                source="none",
                region="us-east-1",
            ),
        )

    assert exc_info.value.package == "strands-agents"


def test_emulator_sdk_missing_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        builtins,
        "__import__",
        _missing_import("strands.models.bedrock"),
    )

    with pytest.raises(SdkMissing) as exc_info:
        create_model(
            provider="bedrock",
            execution="emulator",
            settings=Settings(),
            selected_session=SelectedSession(
                session=FakeBotoSession(),
                source="none",
                region="us-east-1",
            ),
        )

    assert exc_info.value.package == "strands-agents"


def test_openai_sdk_missing_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        builtins,
        "__import__",
        _missing_import("strands.models.openai"),
    )

    with pytest.raises(SdkMissing) as exc_info:
        create_model(
            provider="openai",
            execution="live",
            settings=Settings(model="gpt-test", openai_api_key="key"),
        )

    assert exc_info.value.package == "strands-agents"


def _missing_import(module_name: str) -> Callable[..., Any]:
    original_import = builtins.__import__

    def import_hook(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == module_name:
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    return import_hook
