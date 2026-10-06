"""Scripted Strands model used for deterministic offline agent runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable, Iterable
    from threading import Event

    from strands.agent.agent_metadata import AgentMetadata
    from strands.types.content import Message, SystemContentBlock
    from strands.types.streaming import StreamEvent
    from strands.types.tools import ToolChoice, ToolSpec

T = TypeVar("T")


class ModelConfigurable(Protocol):
    """Small public surface shared by Strands and the fallback model."""

    def get_config(self) -> dict[str, Any]:
        """Return model configuration."""

    def update_config(self, **model_config: Any) -> None:
        """Update model configuration."""


@dataclass(frozen=True)
class ScriptedUsage:
    """Token usage carried by a scripted model response."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


try:
    from strands.models.model import Model as _StrandsModel
except ModuleNotFoundError:
    _STRANDS_AVAILABLE = False

    class _StrandsModel:  # type: ignore[no-redef]
        """Fallback base when Strands is not installed."""

else:
    _STRANDS_AVAILABLE = True


class ScriptedModel(_StrandsModel):
    """A deterministic model that replays Strands stream events."""

    def __init__(
        self,
        events: Iterable[dict[str, Any]],
        *,
        fixture_id: str = "strands-script-default-v1",
        **model_config: Any,
    ) -> None:
        """Store a replayable script and model configuration."""
        self.fixture_id = fixture_id
        self._events = tuple(deepcopy(tuple(events)))
        self._config: dict[str, Any] = dict(model_config)

    def get_config(self) -> dict[str, Any]:
        """Return a copy of the current model config."""
        return dict(self._config)

    def update_config(self, **model_config: Any) -> None:
        """Merge model configuration updates."""
        self._config.update(model_config)

    def stream(
        self,
        messages: list[Message],
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        cancel_signal: Event | None = None,
        agent_metadata: AgentMetadata | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Return an async iterable over deep-copied scripted events."""
        del (
            messages,
            tool_specs,
            system_prompt,
            tool_choice,
            system_prompt_content,
            invocation_state,
            cancel_signal,
            agent_metadata,
            kwargs,
        )
        return self._stream_events()

    def structured_output(
        self,
        output_model: type[T],
        prompt: list[Message],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Yield one structured output from script data."""
        del prompt, system_prompt, kwargs
        return self._structured_events(output_model)

    async def _stream_events(self) -> AsyncGenerator[StreamEvent, None]:
        for event in self._events:
            yield cast("StreamEvent", deepcopy(event))

    async def _structured_events(
        self,
        output_model: type[T],
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        payload = _first_structured_payload(self._events)
        yield {"output": _coerce_structured(output_model, payload)}


def strands_available() -> bool:
    """Return whether the real Strands base class is importable."""
    return _STRANDS_AVAILABLE


def text_script(
    text: str,
    *,
    stop_reason: str = "end_turn",
    usage: ScriptedUsage | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build a small Converse-like text script."""
    usage_payload = usage or ScriptedUsage()
    events: list[dict[str, Any]] = [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": text},
            },
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": stop_reason}},
    ]
    if usage_payload != ScriptedUsage():
        events.append(
            {
                "metadata": {
                    "usage": {
                        "inputTokens": usage_payload.input_tokens,
                        "outputTokens": usage_payload.output_tokens,
                        "totalTokens": usage_payload.total_tokens,
                    },
                },
            },
        )
    return tuple(events)


def _first_structured_payload(events: Iterable[dict[str, Any]]) -> object:
    for event in events:
        if "structuredOutput" in event:
            return event["structuredOutput"]
    return {}


def _coerce_structured[T](output_model: type[T], payload: object) -> T:
    validator = getattr(output_model, "model_validate", None)
    if callable(validator):
        return cast("T", validator(payload))
    if isinstance(payload, output_model):
        return payload
    return output_model(**cast("dict[str, Any]", payload))
