from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from a2a.types import Message, Part, Role, TextPart

import awsai_demo.a2a_demo as demo
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping


def test_a2a_actual_loopback_card_and_task_exchange() -> None:
    result = demo.run_a2a_demo()
    assert result["mode"] == "local_contract"
    assert result["status"] == "ok"
    assert result["data"]["card_name"] == "Pilot adviser"
    assert result["data"]["card_url"].startswith("http://127.0.0.1:")
    text = json.dumps(result["data"]["task_updates"])
    assert "19680" in text
    assert "Human approval is required" in text
    assert result["operations"][0]["transport"] == "loopback"
    assert not result["evidence"]["aws_executed"]


def test_a2a_readiness_timeout_and_failure_close_server() -> None:
    server = SimpleNamespace(started=False)
    with pytest.raises(TimeoutError, match="did not start"):
        demo.wait_started(server, timeout=0)
    demo.wait_started(SimpleNamespace(started=True), timeout=0)

    async def failed(_: str) -> dict[str, Any]:
        message = "exchange failed"
        raise ValueError(message)

    with pytest.raises(ValueError, match="exchange failed"):
        demo.local_exchange(exchanger=failed)


def test_a2a_message_update_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resolver:
        def __init__(self, _: object, __: str) -> None:
            pass

        async def get_agent_card(self) -> SimpleNamespace:
            return SimpleNamespace(name="Example", url="http://127.0.0.1:1")

    class Client:
        async def send_message(self, _: Message) -> AsyncIterator[Message]:
            yield Message(
                message_id="m",
                role=Role.agent,
                parts=[Part(root=TextPart(text="hello"))],
            )

    class Factory:
        def __init__(self, _: object) -> None:
            pass

        def create(self, _: object) -> Client:
            return Client()

    monkeypatch.setattr(demo, "A2ACardResolver", Resolver)
    monkeypatch.setattr(demo, "ClientFactory", Factory)
    result = asyncio.run(demo.exchange("http://127.0.0.1:1"))
    assert result["task_updates"][0]["messageId"] == "m"


class RegistryPort:
    def call(
        self, service: str, operation_name: str, params: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        assert service == "agent-registry"
        assert operation_name == "SearchDiscoverableRegistryRecords"
        assert params["registryIds"] == ["registry-example"]
        return {"registryRecords": [{}]}


def test_a2a_registry_requires_configuration_and_never_uses_emulator() -> None:
    missing = demo.run_a2a_demo(execution="live")
    assert missing["operations"][-1]["error_code"] == "missing_configuration"
    emulator = demo.run_a2a_demo(execution="emulator", settings=Settings())
    assert (
        emulator["operations"][-1]["error_code"] == "not_supported_by_emulator"
    )
    result = demo.run_a2a_demo(
        execution="live",
        settings=Settings(registry_id="registry-example"),
        port=RegistryPort(),
    )
    assert result["mode"] == "live_service"
    assert result["data"]["registry_records"] == 1


def test_a2a_registry_default_selection_preserves_source() -> None:
    result = demo.run_a2a_demo(
        execution="live",
        settings=Settings(registry_id="registry-example"),
        boto_port_factory=lambda **_: SimpleNamespace(
            port=RegistryPort(), credential_source="profile"
        ),
    )
    assert result["evidence"]["credential_source"] == "profile"
