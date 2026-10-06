"""Local BedrockAgentCoreApp worker for the runtime demo.

Lane: agents; lifecycle: agentcore-runtime.
Run: uv run awsai-demo agentcore-runtime --execution offline.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from strands import Agent

from awsai_demo import scenario
from awsai_demo.offline import ScriptedModel, ScriptedUsage, text_script
from awsai_demo.redact import sanitize_exception

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from bedrock_agentcore.runtime import BedrockAgentCoreApp


class RuntimeScriptedModel(ScriptedModel):
    """Scripted model used inside the local AgentCore runtime app."""

    def __init__(self) -> None:
        """Create the deterministic worker model."""
        super().__init__(
            text_script(
                "The support pilot fits inside the local budget.",
                usage=ScriptedUsage(70, 18, 88),
            ),
            fixture_id="strands-script-agentcore-runtime-worker-v1",
        )
        self.calls: list[str] = []

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Record the local model call before replaying the script."""
        self.calls.append("runtime.scripted_model")
        return super().stream(*args, **kwargs)


def build_runtime_agent(model: RuntimeScriptedModel) -> Agent:
    """Build the Strands agent hosted by the local runtime app."""
    return Agent(
        model=model,
        callback_handler=None,
        retry_strategy=None,
    )


def invoke_runtime_agent(
    agent: Agent,
    model: RuntimeScriptedModel,
    payload: dict[str, Any],
) -> dict[str, object]:
    """Invoke the hosted local agent and return public JSON."""
    prompt = _payload_prompt(payload)
    agent(prompt)
    cost = scenario.price_pilot()
    return {
        "accepted": False,
        "budget_usd": cost.budget_usd,
        "headroom_usd": cost.headroom_usd,
        "model_calls": tuple(model.calls),
        "prompt": prompt,
        "staffing_usd": cost.staffing_usd,
        "platform_usd": cost.platform_usd,
        "total_usd": cost.total_usd,
        "within_budget": cost.within_budget,
        "weeks": cost.weeks,
    }


def build_app() -> BedrockAgentCoreApp:
    """Create the local AgentCore Runtime contract app."""
    from awsai_demo.agentcore_runtime_demo import build_app as demo_build_app

    return demo_build_app()


def main(
    argv: Sequence[str] | None = None,
    *,
    app_factory: Callable[[], BedrockAgentCoreApp] | None = None,
) -> int:
    """Run the worker on a loopback HTTP port."""
    try:
        host, port = _parse_args(argv)
        app = build_app() if app_factory is None else app_factory()
        app.run(host=host, port=port)
    except Exception as exc:
        safe = sanitize_exception(exc)
        print(f"{type(exc).__name__}: {safe}", file=sys.stderr)
        return 1
    else:
        return 0


def _parse_args(argv: Sequence[str] | None) -> tuple[str, int]:
    args = list(sys.argv[1:] if argv is None else argv)
    host = "127.0.0.1"
    port = 8080
    while args:
        option = args.pop(0)
        if option == "--host":
            host = _pop_value(args, "--host")
        elif option == "--port":
            port = int(_pop_value(args, "--port"))
        else:
            message = "unknown worker argument"
            raise ValueError(message)
    if host != "127.0.0.1":
        message = "runtime worker only binds 127.0.0.1"
        raise ValueError(message)
    if port < 1 or port > 65_535:
        message = "runtime worker port is out of range"
        raise ValueError(message)
    return host, port


def _pop_value(args: list[str], option: str) -> str:
    if not args:
        message = f"{option} requires a value"
        raise ValueError(message)
    return args.pop(0)


def _payload_prompt(payload: dict[str, Any]) -> str:
    value = payload.get("prompt")
    if isinstance(value, str) and value.strip():
        return value
    return scenario.BRIEF


if __name__ == "__main__":
    raise SystemExit(main())
