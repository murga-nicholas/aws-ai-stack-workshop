"""Strands multi-agent graph, swarm and tool-agent demo.

Lane: agents; lifecycle: strands-multiagent, agent-squad.
Run: uv run awsai-demo strands-multiagent --execution offline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from decimal import Decimal
from importlib import metadata
from typing import TYPE_CHECKING, Any, Protocol, cast

from botocore.exceptions import BotoCoreError, ClientError
from strands import Agent, tool
from strands.hooks.events import BeforeNodeCallEvent
from strands.multiagent import GraphBuilder, Swarm
from strands.types.exceptions import MaxTokensReachedException

from awsai_demo import billing, scenario
from awsai_demo.contracts import (
    CredentialSource,
    DemoResult,
    Effect,
    ErrorCode,
    Execution,
    OperationOutcome,
    ResultError,
    Usage,
    error,
    evidence,
    not_run,
    operation,
    result,
)
from awsai_demo.credentials import select_session
from awsai_demo.demo_support import (
    billable_not_priced,
    not_run_operation,
    public_aws_error_code,
    unsupported_emulator,
)
from awsai_demo.offline import ScriptedModel, ScriptedUsage, text_script
from awsai_demo.policy import (
    ExecutionPolicy,
    PolicyError,
    PolicyLimitExceeded,
    Reservation,
    TokenCounter,
)
from awsai_demo.providers import ModelSelection, SdkMissingError, create_model
from awsai_demo.redact import quiet_sdk_logging, sanitize_exception
from awsai_demo.runtime import TESTED_LIVE_BEDROCK_MODEL, Settings
from awsai_demo.strands_errors import STRANDS_FAILURES, unwrap_strands_error

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Callable, Sequence

    from strands.hooks import HookRegistry
    from strands.types.content import Message, SystemContentBlock
    from strands.types.streaming import StreamEvent
    from strands.types.tools import ToolChoice, ToolSpec

    from awsai_demo.billing import BudgetRun
    from awsai_demo.credentials import SelectedSession


FIXTURE_ID = "strands-script-multiagent-v1"
_DEMO = "strands-multiagent"
_TECHNOLOGY = "Strands multi-agent"
_REFS = (
    "strands-multiagent",
    "agents-classic-multi-agent",
    "agent-squad",
    "multi-agent-orchestrator",
)
_USAGE: Usage = {
    "input_tokens": 80,
    "output_tokens": 20,
    "total_tokens": 100,
}
_LIVE_SYSTEM_PROMPT = (
    "If a tool is available, call it before any other text. "
    "Then answer in one short sentence."
)


@dataclass(frozen=True)
class WorkflowTranscript:
    """Observed framework and model-call order."""

    graph_nodes: tuple[str, ...]
    swarm_nodes: tuple[str, ...]
    tool_result: str
    model_calls: tuple[str, ...]


@dataclass(frozen=True)
class ReservedLiveCall:
    """A delegated live model call with reservation evidence."""

    label: str
    reservation: Reservation
    input_tokens: int
    usage: Usage | None
    error_code: str | None = None


def flag_max_tokens(calls: list[ReservedLiveCall]) -> None:
    """Retag the latest answered call when output hit its cap.

    A call already stored as ``max_tokens_reached`` stays as it is,
    so the handler does not add a second label.
    """
    if any(item.error_code == "max_tokens_reached" for item in calls):
        return
    for index in range(len(calls) - 1, -1, -1):
        item = calls[index]
        if item.error_code is not None:
            continue
        calls[index] = ReservedLiveCall(
            item.label,
            item.reservation,
            item.input_tokens,
            item.usage,
            "max_tokens_reached",
        )
        return


@dataclass(frozen=True)
class SerializedModelRequest:
    """Serialized request plus its formatted Converse payload."""

    body: bytes
    formatted: dict[str, Any]


class ModelFactory(Protocol):
    """Factory surface for provider-backed Strands models."""

    def __call__(
        self,
        *,
        provider: str,
        execution: Execution,
        settings: Settings,
        policy: ExecutionPolicy,
        selected_session: SelectedSession | None,
        model_id: str | None = None,
    ) -> ModelSelection:
        """Return a provider model selection."""


class AgentModelBuilder(Protocol):
    """Build a model instance for one named agent role."""

    def __call__(
        self,
        label: str,
        calls: list[str],
        limits: DispatchLimits,
    ) -> object:
        """Return a Strands-compatible model."""


class DispatchLimits:
    """Enforce global calls and per-agent iterations."""

    def __init__(
        self,
        policy: ExecutionPolicy,
        *,
        clock: Any = time.monotonic,
    ) -> None:
        """Start counters for the Strands workflow."""
        self._policy = policy
        self._clock = cast("Any", clock)
        self._started_at = float(self._clock())
        self._agent_iterations: dict[str, int] = {}
        self.model_calls = 0

    def before_dispatch(self, label: str | None = None) -> None:
        """Raise before exceeding the configured limits."""
        if float(self._clock()) - self._started_at > (
            self._policy.max_wall_seconds
        ):
            message = "wall-clock limit exceeded before multi-agent dispatch"
            raise PolicyLimitExceeded(message)
        if label is not None:
            iterations = self._agent_iterations.get(label, 0)
            if iterations >= self._policy.max_agent_iterations:
                message = "agent iteration limit exceeded before dispatch"
                raise PolicyLimitExceeded(message)
        if self.model_calls >= self._policy.max_model_calls:
            message = "model-call limit exceeded before multi-agent dispatch"
            raise PolicyLimitExceeded(message)
        self.model_calls += 1
        if label is not None:
            self._agent_iterations[label] = iterations + 1


class RecordedTextModel(ScriptedModel):
    """Scripted model that records each Strands invocation."""

    def __init__(
        self,
        label: str,
        calls: list[str],
        limits: DispatchLimits,
    ) -> None:
        """Store the call label and replayable text response."""
        super().__init__(
            text_script(
                f"{label} completed the support pilot step.",
                usage=ScriptedUsage(80, 20, 100),
            ),
            fixture_id=FIXTURE_ID,
        )
        self.label = label
        self._calls = calls
        self._limits = limits

    def stream(
        self,
        messages: list[Message],
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        cancel_signal: object | None = None,
        agent_metadata: object | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Record the call, enforce policy and replay the script."""
        self._limits.before_dispatch(self.label)
        self._calls.append(self.label)
        return super().stream(
            messages,
            tool_specs,
            system_prompt,
            tool_choice=tool_choice,
            system_prompt_content=system_prompt_content,
            invocation_state=invocation_state,
            cancel_signal=cast("Any", cancel_signal),
            agent_metadata=cast("Any", agent_metadata),
            **kwargs,
        )


class ReservingModel(ScriptedModel):
    """Reserve a live request before delegating to a provider model."""

    def __init__(
        self,
        *,
        label: str,
        inner: object,
        model_id: str,
        calls: list[str],
        completed: list[ReservedLiveCall],
        charged: list[ReservedLiveCall],
        count_operations: list[OperationOutcome],
        limits: DispatchLimits,
        budget: Any,
        exact_counter: TokenCounter,
        max_output_tokens: int,
        endpoint_url: str,
        reservation_namespace: str = _DEMO,
    ) -> None:
        """Bind the inner model and local budget."""
        super().__init__((), fixture_id=FIXTURE_ID)
        self.label = label
        self._inner = inner
        self._model_id = model_id
        self._calls = calls
        self._completed = completed
        self._charged = charged
        self._count_operations = count_operations
        self._limits = limits
        self._budget = budget
        self._exact_counter = exact_counter
        self._max_output_tokens = max_output_tokens
        self._endpoint_url = endpoint_url
        self._reservation_namespace = reservation_namespace

    def get_config(self) -> dict[str, Any]:
        """Return the inner model configuration."""
        config = getattr(self._inner, "get_config", None)
        if callable(config):
            return cast("dict[str, Any]", config())
        return {}

    def update_config(self, **model_config: Any) -> None:
        """Forward model configuration updates to the inner model."""
        updater = getattr(self._inner, "update_config", None)
        if callable(updater):
            updater(**model_config)

    def stream(
        self,
        messages: list[Message],
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        cancel_signal: object | None = None,
        agent_metadata: object | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Reserve, delegate and record observed usage."""
        self._limits.before_dispatch(self.label)
        ordinal = self._limits.model_calls
        serialized = _serialized_model_request(
            inner=self._inner,
            model_id=self._model_id,
            messages=messages,
            tool_specs=tool_specs,
            system_prompt=system_prompt,
            tool_choice=tool_choice,
            system_prompt_content=system_prompt_content,
            invocation_state=invocation_state,
            max_tokens=self._max_output_tokens,
        )
        if isinstance(self._exact_counter, BedrockCountTokensCounter):
            self._exact_counter.formatted_request = serialized.formatted
        try:
            reservation, count = self._budget.reserve_model(
                f"{self._reservation_namespace}:{ordinal}:{self.label}",
                model_id=self._model_id,
                serialized_request=serialized.body,
                exact_counter=self._exact_counter,
                routing=_model_routing(self._model_id),
            )
        except (ClientError, BotoCoreError, OSError, PolicyError) as exc:
            self._append_count_operation()
            if isinstance(exc, (ClientError, BotoCoreError, OSError)):
                self._count_operations.append(
                    _live_failure_operation(
                        operation_name="CountTokens",
                        effect="read",
                        endpoint_url=self._counter_endpoint_url(),
                        exc=exc,
                    ),
                )
            raise
        self._append_count_operation()
        charged_call = ReservedLiveCall(
            self.label,
            reservation,
            count.tokens,
            None,
        )
        self._charged.append(charged_call)
        try:
            stream = cast("Any", self._inner).stream(
                messages,
                tool_specs,
                system_prompt,
                tool_choice=tool_choice,
                system_prompt_content=system_prompt_content,
                invocation_state=invocation_state,
                cancel_signal=cancel_signal,
                agent_metadata=agent_metadata,
                **kwargs,
            )
        except Exception as wrapped:
            cause = unwrap_strands_error(wrapped)
            if self._keep_capped_call(reservation, count.tokens, None, cause):
                raise cause from None
            if not isinstance(cause, (ClientError, BotoCoreError, OSError)):
                raise
            self._count_operations.append(
                _live_failure_operation(
                    operation_name=f"BedrockModel.stream:{self.label}",
                    effect="infer",
                    endpoint_url=self._endpoint_url,
                    exc=cause,
                    reserved_usd=float(reservation.amount_usd),
                ),
            )
            raise cause from None
        self._calls.append(self.label)
        return self._record_stream(stream, reservation, count.tokens)

    def _keep_capped_call(
        self,
        reservation: Reservation,
        input_tokens: int,
        usage: Usage | None,
        cause: Exception,
    ) -> bool:
        if not isinstance(cause, MaxTokensReachedException):
            return False
        self._completed.append(
            ReservedLiveCall(
                self.label,
                reservation,
                input_tokens,
                usage,
                "max_tokens_reached",
            )
        )
        return True

    def _append_count_operation(self) -> None:
        count_operation = getattr(self._exact_counter, "last_operation", None)
        if isinstance(count_operation, dict):
            candidate = cast("OperationOutcome", count_operation)
            self._count_operations.append(candidate)

    def _counter_endpoint_url(self) -> str | None:
        endpoint = getattr(self._exact_counter, "endpoint_url", None)
        return endpoint if isinstance(endpoint, str) else None

    async def _record_stream(
        self,
        stream: AsyncIterable[StreamEvent],
        reservation: Reservation,
        input_tokens: int,
    ) -> AsyncIterable[StreamEvent]:
        usage: Usage | None = None
        try:
            async for event in stream:
                usage = _usage_from_event(event) or usage
                yield event
        except Exception as wrapped:
            cause = unwrap_strands_error(wrapped)
            if self._keep_capped_call(reservation, input_tokens, usage, cause):
                raise cause from None
            if not isinstance(cause, (ClientError, BotoCoreError, OSError)):
                raise
            self._count_operations.append(
                _live_failure_operation(
                    operation_name=f"BedrockModel.stream:{self.label}",
                    effect="infer",
                    endpoint_url=self._endpoint_url,
                    exc=cause,
                    reserved_usd=float(reservation.amount_usd),
                ),
            )
            raise cause from None
        self._completed.append(
            ReservedLiveCall(self.label, reservation, input_tokens, usage),
        )


class BedrockCountTokensCounter:
    """Count tokens through Bedrock Runtime CountTokens directly."""

    def __init__(self, client: Any, endpoint_url: str) -> None:
        """Retain the SDK client and endpoint provenance."""
        self._client = client
        self._endpoint_url = endpoint_url
        self.endpoint_url = endpoint_url
        self.formatted_request: dict[str, Any] = {}
        self.last_operation: OperationOutcome | None = None

    def count_tokens(
        self,
        model_id: str,
        serialized_request: bytes,
    ) -> int:
        """Call CountTokens for the formatted Converse request."""
        del serialized_request
        self.last_operation = None
        converse = {
            key: self.formatted_request[key]
            for key in (
                "messages",
                "system",
                "toolConfig",
                "additionalModelRequestFields",
            )
            if key in self.formatted_request
        }
        response = self._client.count_tokens(
            modelId=count_tokens_model_id(model_id),
            input={"converse": converse},
        )
        tokens = int(response["inputTokens"])
        self.last_operation = operation(
            service="bedrock-runtime",
            operation="CountTokens",
            execution_target="aws",
            mode="live_service",
            phase="setup",
            effect="read",
            transport="aws",
            endpoint_url=self._endpoint_url,
            response_received=True,
            request_validated=True,
            usage={"input_tokens": tokens},
        )
        return tokens


def count_tokens_model_id(model_id: str) -> str:
    """Count against the foundation model behind a routing profile."""
    if model_id.startswith(("global.", "us.", "eu.", "apac.")):
        return model_id.split(".", 1)[1]
    return model_id


class SwarmHandoffHook:
    """Trigger one real Swarm handoff without a second model turn."""

    def __init__(self) -> None:
        """Create a one-shot hook provider."""
        self.handoff_triggered = False

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register the callback with Strands."""
        del kwargs
        registry.add_callback(BeforeNodeCallEvent, self.before_node)

    def before_node(self, event: BeforeNodeCallEvent) -> None:
        """Route researcher to reviewer once."""
        if event.node_id != "researcher" or self.handoff_triggered:
            return
        self.handoff_triggered = True
        swarm = cast("Any", event.source)
        swarm._handle_handoff(
            swarm.nodes["reviewer"],
            "Review the local cost plan.",
            {"handoff": "scripted-by-hook"},
        )


@quiet_sdk_logging()
def run_strands_multiagent_demo(
    *,
    execution: Execution = "offline",
    settings: Settings | None = None,
    policy: ExecutionPolicy | None = None,
    exact_counter: TokenCounter | None = None,
    model_factory: ModelFactory | None = None,
    selected_session: SelectedSession | None = None,
    budget_opener: Callable[..., BudgetRun] | None = None,
) -> DemoResult:
    """Run the Strands multi-agent proof in the requested lane."""
    active_settings = settings or Settings()
    chosen_policy = policy or ExecutionPolicy()
    if execution == "offline":
        return _run_offline(active_settings, chosen_policy)
    if execution == "emulator":
        return _run_emulator(active_settings)
    if active_settings.provider == "offline":
        return not_run(
            demo=_DEMO,
            technology=_TECHNOLOGY,
            lane="agents",
            lifecycle_refs=_REFS,
            requested_execution="live",
            headline="Offline provider cannot satisfy live model calls.",
            code="validation_failed",
            message="Use the Bedrock provider for live multi-agent calls.",
        )
    return _run_live(
        active_settings,
        chosen_policy,
        exact_counter,
        model_factory=model_factory,
        selected_session=selected_session,
        budget_opener=budget_opener,
    )


def run_local_workflows(policy: ExecutionPolicy) -> WorkflowTranscript:
    """Execute Graph, Swarm and an agent-as-tool call locally."""
    return _run_workflows(policy, _recorded_model_builder)


def _recorded_model_builder(
    label: str,
    calls: list[str],
    limits: DispatchLimits,
) -> RecordedTextModel:
    return RecordedTextModel(label, calls, limits)


def _run_workflows(
    policy: ExecutionPolicy,
    model_builder: AgentModelBuilder,
    system_prompt: str | None = None,
) -> WorkflowTranscript:
    calls: list[str] = []
    limits = DispatchLimits(policy)
    graph_nodes = _run_graph(calls, limits, model_builder, system_prompt)
    swarm_nodes = _run_swarm(calls, limits, model_builder, system_prompt)
    tool_result = _run_agent_tool(calls, limits, model_builder, system_prompt)
    return WorkflowTranscript(
        graph_nodes=graph_nodes,
        swarm_nodes=swarm_nodes,
        tool_result=tool_result,
        model_calls=tuple(calls),
    )


def _run_graph(
    calls: list[str],
    limits: DispatchLimits,
    model_builder: AgentModelBuilder,
    system_prompt: str | None,
) -> tuple[str, ...]:
    agents = [
        _agent(
            "researcher",
            "graph.researcher",
            calls,
            limits,
            model_builder,
            system_prompt,
        ),
        _agent(
            "costing",
            "graph.costing",
            calls,
            limits,
            model_builder,
            system_prompt,
        ),
        _agent(
            "reviewer",
            "graph.reviewer",
            calls,
            limits,
            model_builder,
            system_prompt,
        ),
    ]
    builder = GraphBuilder()
    for agent in agents:
        builder.add_node(agent, agent.name)
    builder.add_edge("researcher", "costing")
    builder.add_edge("costing", "reviewer")
    graph = (
        builder.set_entry_point("researcher")
        .set_max_node_executions(3)
        .build()
    )
    graph(scenario.BRIEF)
    return ("researcher", "costing", "reviewer")


def _run_swarm(
    calls: list[str],
    limits: DispatchLimits,
    model_builder: AgentModelBuilder,
    system_prompt: str | None,
) -> tuple[str, ...]:
    researcher = _agent(
        "researcher",
        "swarm.researcher",
        calls,
        limits,
        model_builder,
        system_prompt,
    )
    reviewer = _agent(
        "reviewer",
        "swarm.reviewer",
        calls,
        limits,
        model_builder,
        system_prompt,
    )
    swarm = Swarm(
        [researcher, reviewer],
        entry_point=researcher,
        max_handoffs=3,
        max_iterations=3,
        hooks=[SwarmHandoffHook()],
    )
    outcome = swarm("Coordinate the support pilot review.")
    return tuple(cast("Any", item).node_id for item in outcome.node_history)


def _run_agent_tool(
    calls: list[str],
    limits: DispatchLimits,
    model_builder: AgentModelBuilder,
    system_prompt: str | None,
) -> str:
    reviewer_model = model_builder("tool.reviewer", calls, limits)
    if system_prompt is None:
        reviewer = Agent(
            model=cast("Any", reviewer_model),
            name="reviewer",
            callback_handler=None,
            retry_strategy=None,
        )
    else:
        reviewer = Agent(
            model=cast("Any", reviewer_model),
            name="reviewer",
            system_prompt=system_prompt,
            callback_handler=None,
            retry_strategy=None,
        )

    @tool
    def ask_reviewer(question: str) -> str:
        """Ask reviewer through a real Strands tool wrapper."""
        answer = reviewer(question)
        return str(getattr(answer, "stop_reason", "end_turn"))

    if system_prompt is None:
        host = Agent(
            model=cast("Any", reviewer_model),
            tools=[ask_reviewer],
            callback_handler=None,
            retry_strategy=None,
        )
    else:
        host = Agent(
            model=cast("Any", reviewer_model),
            tools=[ask_reviewer],
            system_prompt=system_prompt,
            callback_handler=None,
            retry_strategy=None,
        )
    response = cast("Any", host.tool).ask_reviewer(
        question="Should the pilot proceed?",
    )
    content = response["content"]
    return str(content[0]["text"])


def _agent(
    name: str,
    label: str,
    calls: list[str],
    limits: DispatchLimits,
    model_builder: AgentModelBuilder,
    system_prompt: str | None,
) -> Agent:
    model = cast("Any", model_builder(label, calls, limits))
    if system_prompt is None:
        return Agent(
            model=model,
            name=name,
            callback_handler=None,
            retry_strategy=None,
        )
    return Agent(
        model=model,
        name=name,
        system_prompt=system_prompt,
        callback_handler=None,
        retry_strategy=None,
    )


def _run_offline(settings: Settings, policy: ExecutionPolicy) -> DemoResult:
    transcript = run_local_workflows(policy)
    operations = [
        *_model_fixture_operations(transcript.model_calls),
        *_framework_operations(),
    ]
    return _demo_result(
        execution="offline",
        settings=settings,
        headline="Real Strands Graph, Swarm and tool-agent ran locally.",
        operations=operations,
        provider="offline",
        fixture_id=FIXTURE_ID,
        requested_model="ScriptedModel",
        observed_model="RecordedTextModel",
        data=_workflow_data(transcript, mode="offline"),
    )


def _run_emulator(settings: Settings) -> DemoResult:
    operations = [
        unsupported_emulator(
            service="strands",
            operation_name="Graph/Swarm/agent-as-tool",
        ),
        unsupported_emulator(
            service="bedrock-runtime",
            operation_name="BedrockModel.stream multi-agent calls",
        ),
    ]
    return _demo_result(
        execution="emulator",
        settings=settings,
        headline="Strands multi-agent has no LocalStack emulator path.",
        operations=operations,
        provider="localstack-contract-only",
        fixture_id=None,
        requested_model=None,
        observed_model=None,
        data={"emulator": "not_supported"},
    )


def _run_live(
    settings: Settings,
    policy: ExecutionPolicy,
    exact_counter: TokenCounter | None,
    *,
    model_factory: ModelFactory | None,
    selected_session: SelectedSession | None,
    budget_opener: Callable[..., BudgetRun] | None = None,
) -> DemoResult:
    model_id = settings.model or TESTED_LIVE_BEDROCK_MODEL
    credential_source: CredentialSource = "none"
    provider = "bedrock"
    observed_model: str | None = None
    live_calls: list[ReservedLiveCall] = []
    charged_calls: list[ReservedLiveCall] = []
    count_operations: list[OperationOutcome] = []
    try:
        selection, credential_source = _live_model_selection(
            settings=settings,
            policy=policy,
            model_id=model_id,
            model_factory=model_factory,
            selected_session=selected_session,
        )
        provider = selection.provider
        observed_model = selection.model_id or model_id
        counter = exact_counter or _default_counter(selection.model)
        open_budget = (
            billing.active_budget_run
            if budget_opener is None
            else budget_opener
        )
        budget = open_budget(
            region=settings.region,
            policy=policy,
        ).command(_DEMO)
        transcript = _run_workflows(
            policy,
            lambda label, calls, limits: ReservingModel(
                label=label,
                inner=selection.model,
                model_id=model_id,
                calls=calls,
                completed=live_calls,
                charged=charged_calls,
                count_operations=count_operations,
                limits=limits,
                budget=budget,
                exact_counter=counter,
                max_output_tokens=policy.max_output_tokens,
                endpoint_url=_bedrock_runtime_endpoint(settings),
            ),
            _LIVE_SYSTEM_PROMPT,
        )
    except SdkMissingError as exc:
        return _live_not_run(
            settings,
            model_id,
            code="sdk_missing",
            message=sanitize_exception(exc),
        )
    except (PolicyError, TypeError, ValueError) as exc:
        return _live_budget_refused(
            settings,
            model_id,
            message=str(exc),
            prior_operations=[
                *count_operations,
                *_live_model_operations(
                    settings=settings,
                    live_calls=live_calls,
                ),
            ],
        )
    except ClientError as exc:
        return _live_failed_result(
            settings=settings,
            model_id=model_id,
            provider=provider,
            observed_model=observed_model,
            credential_source=credential_source,
            live_calls=live_calls,
            charged_calls=charged_calls,
            operations=count_operations,
            code="model_unavailable",
            message=sanitize_exception(exc),
        )
    except STRANDS_FAILURES as exc:
        cause = unwrap_strands_error(exc)
        if isinstance(cause, MaxTokensReachedException):
            flag_max_tokens(live_calls)
        return _live_failed_result(
            settings=settings,
            model_id=model_id,
            provider=provider,
            observed_model=observed_model,
            credential_source=credential_source,
            live_calls=live_calls,
            charged_calls=charged_calls,
            operations=count_operations,
            code=(
                "max_tokens_reached"
                if isinstance(cause, MaxTokensReachedException)
                else "model_unavailable"
            ),
            message=sanitize_exception(cause),
        )
    except (BotoCoreError, OSError) as exc:
        return _live_failed_result(
            settings=settings,
            model_id=model_id,
            provider=provider,
            observed_model=observed_model,
            credential_source=credential_source,
            live_calls=live_calls,
            charged_calls=charged_calls,
            operations=count_operations,
            code="timeout",
            message=sanitize_exception(exc),
        )
    operations = [
        *_framework_operations(),
        *count_operations,
        *_live_model_operations(settings=settings, live_calls=live_calls),
    ]
    return _demo_result(
        execution="live",
        settings=settings,
        headline="Real Strands multi-agent model calls completed live.",
        operations=operations,
        provider=selection.provider,
        fixture_id=None,
        requested_model=model_id,
        observed_model=selection.model_id or model_id,
        data={
            **_workflow_data(transcript, mode="live"),
            "credential_source": credential_source,
        },
        credential_source=credential_source,
        estimated_cost=sum(
            (item.reservation.amount_usd for item in charged_calls),
            Decimal("0"),
        ),
    )


def _default_counter(model: object) -> TokenCounter:
    client = getattr(model, "client", None)
    count_tokens = getattr(client, "count_tokens", None)
    if not callable(count_tokens):
        message = "Live Bedrock CountTokens client is unavailable"
        raise TypeError(message)
    meta = getattr(client, "meta", None)
    endpoint_url = getattr(meta, "endpoint_url", None)
    endpoint = (
        endpoint_url
        if isinstance(endpoint_url, str)
        else "https://bedrock-runtime.us-east-1.amazonaws.com"
    )
    return BedrockCountTokensCounter(client, endpoint)


def _live_model_selection(
    *,
    settings: Settings,
    policy: ExecutionPolicy,
    model_id: str,
    model_factory: ModelFactory | None,
    selected_session: SelectedSession | None,
    session_selector: Callable[..., SelectedSession] | None = None,
    model_builder: Callable[..., ModelSelection] | None = None,
) -> tuple[ModelSelection, CredentialSource]:
    active_session = selected_session
    credential_source: CredentialSource = "none"
    choose_session = (
        select_session if session_selector is None else session_selector
    )
    if active_session is None and model_factory is None:
        active_session = choose_session(execution="live", settings=settings)
    if active_session is not None:
        credential_source = active_session.source
    factory = model_factory or model_builder or create_model
    return (
        factory(
            provider="bedrock",
            execution="live",
            settings=settings,
            policy=policy,
            selected_session=active_session,
            model_id=model_id,
        ),
        credential_source,
    )


def _live_budget_refused(
    settings: Settings,
    model_id: str,
    *,
    message: str = "Live model calls require an exact request token counter.",
    prior_operations: Sequence[OperationOutcome] = (),
) -> DemoResult:
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        requested_execution="live",
        headline="Live multi-agent model calls need priced reservations.",
        operations=[
            *prior_operations,
            billable_not_priced(
                service="bedrock-runtime",
                operation_name="BedrockModel.stream multi-agent calls",
            ),
        ],
        evidence=evidence(
            provider="bedrock",
            region=settings.region,
            requested_model=model_id,
            packages=_package_versions(("strands-agents", "botocore")),
        ),
        data={"model": model_id, "pricing": "missing"},
        error=error("budget_exceeded", message),
    )


def _live_failed_result(
    *,
    settings: Settings,
    model_id: str,
    provider: str,
    observed_model: str | None,
    credential_source: CredentialSource,
    live_calls: Sequence[ReservedLiveCall],
    charged_calls: Sequence[ReservedLiveCall],
    operations: Sequence[OperationOutcome],
    code: ErrorCode,
    message: str,
) -> DemoResult:
    recorded_operations = [
        *operations,
        *_live_model_operations(settings=settings, live_calls=live_calls),
    ]
    if not recorded_operations:
        return _live_not_run(
            settings,
            model_id,
            code=code,
            message=message,
        )
    return _demo_result(
        execution="live",
        settings=settings,
        headline="Live multi-agent dispatch stopped after a provider error.",
        operations=recorded_operations,
        provider=provider,
        fixture_id=None,
        requested_model=model_id,
        observed_model=observed_model or model_id,
        data={
            "model": model_id,
            "live_dispatch": "partial",
            "completed_model_calls": tuple(item.label for item in live_calls),
        },
        credential_source=credential_source,
        estimated_cost=sum(
            (item.reservation.amount_usd for item in charged_calls),
            Decimal("0"),
        ),
        result_error=error(code, message),
    )


def _live_not_run(
    settings: Settings,
    model_id: str,
    *,
    code: ErrorCode,
    message: str,
) -> DemoResult:
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        requested_execution="live",
        headline="Live multi-agent dispatch did not complete.",
        operations=[
            not_run_operation(
                service="bedrock-runtime",
                operation_name="BedrockModel.stream multi-agent calls",
                error_code=code,
                request_validated=False,
            ),
        ],
        evidence=evidence(
            provider="bedrock",
            region=settings.region,
            requested_model=model_id,
            packages=_package_versions(("strands-agents", "botocore")),
        ),
        data={"model": model_id, "live_dispatch": "not_run"},
        error=error(code, message),
    )


def _model_fixture_operations(calls: Sequence[str]) -> list[OperationOutcome]:
    return [
        operation(
            service="bedrock-runtime",
            operation=f"ScriptedModel.stream:{label}",
            mode="local_contract",
            effect="infer",
            fixture_id=FIXTURE_ID,
            usage=_USAGE,
        )
        for label in calls
    ]


def _live_failure_operation(
    *,
    operation_name: str,
    effect: Effect,
    endpoint_url: str | None,
    exc: ClientError | BotoCoreError | OSError,
    reserved_usd: float | None = None,
) -> OperationOutcome:
    if isinstance(exc, ClientError):
        metadata_payload = exc.response.get("ResponseMetadata", {})
        status_code = int(metadata_payload.get("HTTPStatusCode", 0)) or None
        error_payload = exc.response.get("Error", {})
        error_code = public_aws_error_code(error_payload, status_code)
        return operation(
            service="bedrock-runtime",
            operation=operation_name,
            execution_target="aws",
            mode="live_service",
            status="blocked",
            effect=effect,
            transport="aws",
            endpoint_url=endpoint_url,
            response_received=True,
            request_validated=True,
            http_status=status_code,
            error_code=error_code,
            reserved_usd=reserved_usd,
        )
    return operation(
        service="bedrock-runtime",
        operation=operation_name,
        execution_target="aws",
        mode="attempt_failed",
        status="blocked",
        effect=effect,
        transport="aws",
        endpoint_url=endpoint_url,
        response_received=False,
        request_validated=True,
        error_code="timeout",
        reserved_usd=reserved_usd,
    )


def _model_routing(model_id: str) -> str:
    if model_id.startswith("global."):
        return "cross-region-global"
    if model_id.startswith(("us.", "eu.", "apac.")):
        return "cross-region"
    return "in-region"


def _live_model_operations(
    *,
    settings: Settings,
    live_calls: Sequence[ReservedLiveCall],
) -> list[OperationOutcome]:
    outcomes: list[OperationOutcome] = []
    for item in live_calls:
        capped = item.error_code is not None
        outcomes.append(
            operation(
                service="bedrock-runtime",
                operation=f"BedrockModel.stream:{item.label}",
                execution_target="aws",
                mode="live_model",
                status="blocked" if capped else "ok",
                effect="infer",
                transport="aws",
                endpoint_url=_bedrock_runtime_endpoint(settings),
                response_received=True,
                request_validated=True,
                error_code=item.error_code,
                usage=item.usage,
                reserved_usd=float(item.reservation.amount_usd),
            )
        )
    return outcomes


def _framework_operations() -> list[OperationOutcome]:
    return [
        operation(
            service="strands",
            operation="Graph.__call__ researcher->costing->reviewer",
            execution_target="local",
            mode="local_execution",
        ),
        operation(
            service="strands",
            operation="Swarm.__call__ researcher->reviewer",
            execution_target="local",
            mode="local_execution",
        ),
        operation(
            service="strands",
            operation="Agent.tool.ask_reviewer -> reviewer Agent.__call__",
            execution_target="local",
            mode="local_execution",
        ),
    ]


def _workflow_data(
    transcript: WorkflowTranscript,
    *,
    mode: str,
) -> dict[str, object]:
    return {
        "workflow_mode": mode,
        "agent_tool_invocation": "programmatic Agent.tool.ask_reviewer",
        "graph_nodes": transcript.graph_nodes,
        "swarm_nodes": transcript.swarm_nodes,
        "agent_tool_result": transcript.tool_result,
        "model_calls": transcript.model_calls,
        "max_model_calls": 6,
        "agent_iteration_limit_scope": "per model label",
        "observed_framework_calls": {
            "graph": 3,
            "swarm": 2,
            "agent_as_tool": 1,
        },
    }


def _demo_result(
    *,
    execution: Execution,
    settings: Settings,
    headline: str,
    operations: Sequence[OperationOutcome],
    provider: str,
    fixture_id: str | None,
    requested_model: str | None,
    observed_model: str | None,
    data: dict[str, object],
    credential_source: CredentialSource = "none",
    estimated_cost: Decimal | int = 0,
    result_error: ResultError | None = None,
) -> DemoResult:
    cost = float(estimated_cost) if estimated_cost else None
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        requested_execution=execution,
        headline=headline,
        operations=operations,
        evidence=evidence(
            sdk_invoked=True,
            network_attempted=any(
                item["transport"] in {"aws", "loopback", "external"}
                for item in operations
            ),
            aws_executed=any(
                item["execution_target"] == "aws" and item["response_received"]
                for item in operations
            ),
            provider=provider,
            region=settings.region,
            fixture_id=fixture_id,
            requested_model=requested_model,
            observed_model=observed_model,
            credential_source=credential_source,
            packages=_package_versions(("strands-agents", "botocore")),
            estimated_cost_usd=cost,
            cost_basis="estimated" if cost is not None else None,
        ),
        data=data,
        error=result_error,
    )


def _serialized_model_request(
    *,
    inner: object,
    model_id: str,
    messages: Sequence[Message],
    tool_specs: Sequence[ToolSpec] | None,
    system_prompt: str | None,
    tool_choice: ToolChoice | None,
    system_prompt_content: Sequence[SystemContentBlock] | None,
    invocation_state: dict[str, Any] | None,
    max_tokens: int,
) -> SerializedModelRequest:
    formatter = getattr(inner, "format_request", None)
    if callable(formatter):
        formatted = formatter(
            list(messages),
            None if tool_specs is None else list(tool_specs),
            None
            if system_prompt_content is None
            else list(system_prompt_content),
            tool_choice,
        )
        body = json.dumps(
            formatted,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return SerializedModelRequest(body, cast("dict[str, Any]", formatted))
    payload = {
        "inferenceConfig": {"maxTokens": max_tokens},
        "invocationState": invocation_state,
        "messages": messages,
        "modelId": model_id,
        "system": system_prompt,
        "systemPromptContent": system_prompt_content,
        "toolChoice": tool_choice,
        "toolSpecs": tool_specs,
    }
    body = json.dumps(
        payload,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return SerializedModelRequest(body, payload)


def _usage_from_event(event: StreamEvent) -> Usage | None:
    raw = cast("dict[str, Any]", event)
    metadata_payload = raw.get("metadata")
    if not isinstance(metadata_payload, dict):
        return None
    usage_payload = metadata_payload.get("usage")
    if not isinstance(usage_payload, dict):
        return None
    usage: Usage = {}
    if "inputTokens" in usage_payload:
        usage["input_tokens"] = int(usage_payload["inputTokens"])
    if "outputTokens" in usage_payload:
        usage["output_tokens"] = int(usage_payload["outputTokens"])
    if "totalTokens" in usage_payload:
        usage["total_tokens"] = int(usage_payload["totalTokens"])
    return usage or None


def _bedrock_runtime_endpoint(settings: Settings) -> str:
    return f"https://bedrock-runtime.{settings.region}.amazonaws.com"


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


# Shared, tested dispatch adapters for the other agent demos.
live_model_selection = _live_model_selection
live_model_operations = _live_model_operations
model_routing = _model_routing
default_counter = _default_counter
