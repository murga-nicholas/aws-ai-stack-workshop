"""Real Strands agent and file sessions behind the decision demo."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from strands import Agent
from strands.session.file_session_manager import FileSessionManager

from awsai_demo.contracts import OperationOutcome, operation
from awsai_demo.live_agent import LiveAgentModel
from awsai_demo.offline import ScriptedModel, text_script
from awsai_demo.policy import PolicyLimitExceededError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from strands.models.model import Model
    from strands.types.content import Message
    from strands.types.streaming import StreamEvent

    from awsai_demo.contracts import Execution
    from awsai_demo.policy import ExecutionPolicy

FIXTURE_ID = "strands-script-decision-v1"


def approval_script() -> tuple[dict[str, Any], ...]:
    """Ask the actual Strands loop to call the approval tool."""
    return (
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {
                    "toolUse": {
                        "toolUseId": "accept-pilot",
                        "name": "accept_proposal",
                    }
                },
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": json.dumps({})}},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
    )


class DecisionModel(ScriptedModel):
    """Script offline turns or observe a real emulator model call."""

    def __init__(
        self,
        policy: ExecutionPolicy,
        *,
        delegate: Model | None = None,
        endpoint: str | None = None,
    ) -> None:
        """Start per-command model and iteration limits."""
        super().__init__((), fixture_id=FIXTURE_ID)
        self.policy = policy
        self.delegate = delegate
        self.endpoint = endpoint
        self._operations: list[OperationOutcome] = []
        self.calls = 0
        self.started = time.monotonic()

    @property
    def operations(self) -> list[OperationOutcome]:
        """Use actual reserved dispatch evidence for a live model."""
        if isinstance(self.delegate, LiveAgentModel):
            return self.delegate.operations
        return self._operations

    async def stream(
        self,
        messages: list[Message],
        tool_specs: Any = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run one bounded call with honest provenance."""
        limit = min(
            self.policy.max_model_calls, self.policy.max_agent_iterations
        )
        if self.calls >= limit:
            message = "Decision model-call or iteration limit reached"
            raise PolicyLimitExceededError(message)
        if time.monotonic() - self.started >= self.policy.max_wall_seconds:
            message = "Decision wall-clock limit reached"
            raise PolicyLimitExceededError(message)
        self.calls += 1
        model = self.delegate
        if model is None:
            completed = any(
                "toolResult" in block
                for item in messages
                for block in item["content"]
            )
            script = (
                text_script("The recorded approval decision was applied.")
                if completed
                else approval_script()
            )
            model = ScriptedModel(script, fixture_id=FIXTURE_ID)
        async for event in model.stream(
            messages, tool_specs, system_prompt, **kwargs
        ):
            yield event
        if isinstance(self.delegate, LiveAgentModel):
            return
        self._operations.append(
            operation(
                service="strands",
                operation="Model.stream",
                execution_target="fixture"
                if self.delegate is None
                else "emulator",
                mode="local_contract"
                if self.delegate is None
                else "local_emulator",
                transport="none" if self.delegate is None else "loopback",
                endpoint_url=self.endpoint,
                fixture_id=FIXTURE_ID if self.delegate is None else None,
                request_validated=False,
            )
        )


class TrackedFileSessionManager(FileSessionManager):
    """Record session-file reads without exposing their contents."""

    def __init__(self, directory: Path, run_id: str) -> None:
        """Set the read log before loading the session."""
        self.files_read: list[str] = []
        self.directory = directory
        super().__init__(session_id=run_id, storage_dir=str(directory))

    def _read_file(self, path: str) -> dict[str, Any]:
        from pathlib import Path

        payload = super()._read_file(path)
        self.files_read.append(
            Path(path).relative_to(self.directory).as_posix()
        )
        return payload


def build_agent(
    *,
    directory: Path,
    run_id: str,
    acceptance_tool: object,
    policy: ExecutionPolicy,
    execution: Execution,
    model: Model | None = None,
    endpoint: str | None = None,
) -> tuple[Agent, DecisionModel, TrackedFileSessionManager]:
    """Reconstruct the agent around its persisted session."""
    if execution != "offline" and model is None:
        message = "A non-offline decision requires a real provider model"
        raise ValueError(message)
    observed = DecisionModel(policy, delegate=model, endpoint=endpoint)
    session = TrackedFileSessionManager(directory / "strands", run_id)
    agent = Agent(
        model=observed,
        tools=[acceptance_tool],
        system_prompt=(
            (
                "Call accept_proposal first, before any other text. "
                "Then report only the tool's decision in one short "
                "sentence. Never invent human approval."
            )
            if execution == "live"
            else (
                "Call accept_proposal to request approval of the "
                "priced pilot. "
                "Never invent human approval. Report the returned decision."
            )
        ),
        session_manager=session,
        agent_id="decision",
        callback_handler=None,
        # The provider owns its one retry; never multiply retry layers.
        retry_strategy=None,
    )
    return agent, observed, session
