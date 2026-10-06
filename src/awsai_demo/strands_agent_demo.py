"""Strands single-agent demo with a real local tool loop.

Lane: agents; lifecycle: strands-agents.
Run: uv run awsai-demo strands-agent --execution offline.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from importlib import metadata
from typing import TYPE_CHECKING, Any, Protocol, cast

from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel
from strands import Agent, tool
from strands.types.exceptions import MaxTokensReachedException

from awsai_demo import scenario
from awsai_demo.contracts import (
    DemoResult,
    Execution,
    OperationOutcome,
    Usage,
    error,
    evidence,
    missing_configuration,
    not_run,
    operation,
    result,
)
from awsai_demo.demo_support import billable_not_priced, build_result
from awsai_demo.live_agent import LiveAgentModel
from awsai_demo.offline import ScriptedModel
from awsai_demo.policy import ExecutionPolicy, PolicyError, PolicyLimitExceeded
from awsai_demo.providers import ModelSelection, SdkMissingError, create_model
from awsai_demo.redact import quiet_sdk_logging, sanitize_exception
from awsai_demo.runtime import TESTED_LIVE_BEDROCK_MODEL, Settings
from awsai_demo.strands_errors import STRANDS_FAILURES, unwrap_strands_error
from awsai_demo.strands_multiagent_demo import flag_max_tokens

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Callable, Iterable, Sequence

    from strands.types.content import Message, SystemContentBlock
    from strands.types.streaming import StreamEvent
    from strands.types.tools import ToolChoice, ToolSpec

    from awsai_demo.credentials import SelectedSession
    from awsai_demo.policy import TokenCounter

FIXTURE_ID = "strands-script-price-pilot-v1"
_REFS = ("strands-agents",)
_DEMO = "strands-agent"
_TECHNOLOGY = "Strands Agents"
_TOOL_CALLS: ContextVar[list[str] | None] = ContextVar(
    "awsai_strands_tool_calls",
    default=None,
)


class ModelFactory(Protocol):
    """Factory surface used to build provider-backed Strands models."""

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
        """Return a model selection for the requested lane."""


class PilotSummary(BaseModel):
    """Typed summary returned by the Strands tool and final output."""

    weeks: int
    staffing_usd: float
    platform_usd: float
    total_usd: float
    budget_usd: float
    headroom_usd: float
    within_budget: bool
    recommendation: str

    @classmethod
    def from_cost(cls, cost: scenario.PilotCost) -> PilotSummary:
        """Convert the shared scenario cost into the agent schema."""
        return cls(
            weeks=cost.weeks,
            staffing_usd=cost.staffing_usd,
            platform_usd=cost.platform_usd,
            total_usd=cost.total_usd,
            budget_usd=cost.budget_usd,
            headroom_usd=cost.headroom_usd,
            within_budget=cost.within_budget,
            recommendation=proposal_recommendation(cost),
        )


# fmt: off
# slide: agent
@tool
def price_pilot() -> PilotSummary:
    """Price the support pilot without approving it."""
    return _priced_summary()


def build_agent(model: Any) -> Agent:
    """Build the Strands agent with typed output."""
    return Agent(
        model=model,
        tools=[price_pilot],
        structured_output_model=PilotSummary,
        callback_handler=None,
        retry_strategy=None)
# end-slide: agent
# fmt: on


class DispatchLimits:
    """Enforce Strands dispatch limits before model calls."""

    def __init__(
        self,
        policy: ExecutionPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Start a local dispatch counter."""
        self._policy = policy
        self._clock = clock
        self._started_at = clock()
        self.model_calls = 0
        self.agent_iterations = 0

    def before_dispatch(self) -> None:
        """Raise before a model request would exceed local limits."""
        if self._clock() - self._started_at > self._policy.max_wall_seconds:
            message = "wall-clock limit exceeded before Strands dispatch"
            raise PolicyLimitExceeded(message)
        if self.model_calls >= self._policy.max_model_calls:
            message = "model-call limit exceeded before Strands dispatch"
            raise PolicyLimitExceeded(message)
        if self.agent_iterations >= self._policy.max_agent_iterations:
            message = "agent-iteration limit exceeded before Strands dispatch"
            raise PolicyLimitExceeded(message)
        self.model_calls += 1
        self.agent_iterations += 1


class PricePilotScriptedModel(ScriptedModel):
    """Scripted model that drives one real Strands tool loop."""

    def __init__(self, limits: DispatchLimits | None = None) -> None:
        """Create the named fixture with call counters."""
        super().__init__((), fixture_id=FIXTURE_ID)
        self.calls: list[str] = []
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
        """Return the next stream based on the framework state."""
        if self._limits is not None:
            self._limits.before_dispatch()
        del (
            tool_specs,
            system_prompt,
            system_prompt_content,
            invocation_state,
            cancel_signal,
            agent_metadata,
            kwargs,
        )
        if _forced_tool_name(tool_choice) == "PilotSummary":
            self.calls.append("structured_output")
            return _stream_events(
                _tool_use_events("summary-1", "PilotSummary")
            )
        if _latest_message_has_tool_result(messages):
            self.calls.append("final_answer")
            return _stream_events(
                _text_events("The pilot costs 19680 USD and needs approval."),
            )
        self.calls.append("price_tool")
        return _stream_events(_tool_use_events("price-1", "price_pilot"))


class ObservedModel:
    """Count provider model stream calls while delegating to Strands."""

    def __init__(
        self,
        model: object,
        call_label: str,
        limits: DispatchLimits,
    ) -> None:
        """Wrap a provider model without hiding its attributes."""
        self._model = model
        self._call_label = call_label
        self._limits = limits
        self.calls: list[str] = []

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Record a provider model stream after it completes."""
        self._limits.before_dispatch()
        model = cast("Any", self._model)
        stream = cast("Callable[..., Any]", model.stream)
        events = cast("AsyncIterable[StreamEvent]", stream(*args, **kwargs))
        return self._record_completed_stream(events)

    async def _record_completed_stream(
        self,
        events: AsyncIterable[StreamEvent],
    ) -> AsyncIterable[StreamEvent]:
        async for event in events:
            yield event
        self.calls.append(self._call_label)

    def __getattr__(self, name: str) -> Any:
        """Delegate provider-specific attributes."""
        return getattr(self._model, name)


def _priced_summary() -> PilotSummary:
    calls = _TOOL_CALLS.get()
    if calls is not None:
        calls.append("price_pilot")
    return PilotSummary.from_cost(scenario.price_pilot())


def proposal_recommendation(cost: scenario.PilotCost) -> str:
    """Return the recommendation while preserving the human gate."""
    acceptable, reasons = scenario.proposal_is_acceptable(cost)
    if acceptable:
        return reasons[0]
    return "Do not proceed: " + "; ".join(reasons)


@quiet_sdk_logging()
def run_strands_agent_demo(
    *,
    execution: Execution = "offline",
    settings: Settings | None = None,
    policy: ExecutionPolicy | None = None,
    selected_session: SelectedSession | None = None,
    model_factory: ModelFactory | None = None,
    exact_counter: TokenCounter | None = None,
    scripted_model: Callable[[DispatchLimits], Any] | None = None,
    agent_builder: Callable[[Any], Any] | None = None,
    budget_opener: Callable[..., Any] | None = None,
) -> DemoResult:
    """Run the Strands agent proof in the requested lane."""
    active_settings = settings or Settings()
    chosen_policy = policy or ExecutionPolicy()
    provider = active_settings.provider
    if execution == "offline":
        if provider not in (None, "offline"):
            return _provider_refused(
                execution=execution,
                provider=provider,
            )
        return _run_offline(
            execution,
            active_settings,
            chosen_policy,
            scripted_model=scripted_model,
            agent_builder=agent_builder,
        )
    if execution == "emulator":
        if provider not in (None, "bedrock"):
            return _provider_refused(
                execution=execution,
                provider=provider,
            )
        return _run_emulator(
            active_settings,
            chosen_policy,
            selected_session=selected_session,
            model_factory=model_factory,
        )
    if provider == "offline":
        return _provider_refused(execution=execution, provider=provider)
    if provider == "openai":
        return _run_live_budget_blocked(active_settings, provider="openai")
    return _run_live_agent(
        active_settings,
        chosen_policy,
        selected_session=selected_session,
        model_factory=model_factory,
        exact_counter=exact_counter,
        budget_opener=budget_opener,
    )


def _run_live_agent(
    settings: Settings,
    policy: ExecutionPolicy,
    *,
    selected_session: SelectedSession | None,
    model_factory: ModelFactory | None,
    exact_counter: TokenCounter | None,
    budget_opener: Callable[..., Any] | None = None,
) -> DemoResult:
    observed: LiveAgentModel | None = None
    calls: list[str] = []
    data: dict[str, Any] = {}
    failure = None
    try:
        observed = LiveAgentModel(
            demo=_DEMO,
            settings=settings,
            policy=policy,
            selected_session=selected_session,
            model_factory=model_factory,
            exact_counter=exact_counter,
            budget_opener=budget_opener,
        )
        summary, stop_reason, _ = _invoke_agent(
            observed, policy, tool_calls=calls
        )
        data = {
            "summary": summary.model_dump(),
            "summary_type": "PilotSummary",
            "stop_reason": stop_reason,
            "model_calls": observed.calls,
        }
    except (PolicyError, SdkMissingError, TypeError, ValueError) as exc:
        failure = ("budget_exceeded", sanitize_exception(exc))
    except (ClientError, BotoCoreError, OSError) as exc:
        failure = ("model_unavailable", sanitize_exception(exc))
    except STRANDS_FAILURES as exc:
        cause = unwrap_strands_error(exc)
        capped = isinstance(cause, MaxTokensReachedException)
        if observed is not None and capped:
            flag_max_tokens(observed.completed)
        failure = (
            "max_tokens_reached" if capped else "model_unavailable",
            sanitize_exception(cause),
        )
    operations = [] if observed is None else observed.operations
    if failure is not None and not any(
        item["status"] != "ok" for item in operations
    ):
        operations.append(
            billable_not_priced(
                service="bedrock-runtime", operation_name="BedrockModel.stream"
            )
        )
    if observed is not None:
        operations.extend(_local_framework_operations(calls))
    built = build_result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=list(_REFS),
        execution="live",
        settings=settings,
        headline="Strands calls reserve their complete model requests.",
        operations=operations,
        data=data,
        requested_model=settings.model or TESTED_LIVE_BEDROCK_MODEL,
        credential_source=None
        if observed is None
        else observed.credential_source,
        result_error=cast("Any", failure),
    )
    built["evidence"]["provider"] = "bedrock"
    return built


def _run_offline(
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    *,
    scripted_model: Callable[[DispatchLimits], Any] | None = None,
    agent_builder: Callable[[Any], Any] | None = None,
) -> DemoResult:
    model_type = (
        PricePilotScriptedModel if scripted_model is None else scripted_model
    )
    model = model_type(DispatchLimits(policy))
    summary, stop_reason, tool_calls = _invoke_agent(
        model,
        policy,
        agent_builder=agent_builder,
    )
    operations = _operations_for_model_calls(model.calls)
    operations.extend(_local_framework_operations(tool_calls))
    return _agent_result(
        execution=execution,
        settings=settings,
        headline="Real Strands Agent loop called price_pilot locally.",
        operations=operations,
        provider="offline",
        fixture_id=FIXTURE_ID,
        requested_model="ScriptedModel",
        observed_model="PricePilotScriptedModel",
        summary=summary,
        model_calls=tuple(model.calls),
        stop_reason=stop_reason,
    )


def _provider_refused(
    *, execution: Execution, provider: str | None
) -> DemoResult:
    provider_name = provider or "default"
    message = f"Provider {provider_name} is not valid for {execution}."
    return not_run(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        requested_execution=execution,
        headline="Requested provider is not supported for this lane.",
        code="validation_failed",
        message=message,
    )


def _run_emulator(
    settings: Settings,
    policy: ExecutionPolicy,
    *,
    selected_session: SelectedSession | None,
    model_factory: ModelFactory | None,
) -> DemoResult:
    if settings.localstack_auth_token is None:
        return missing_configuration(
            demo=_DEMO,
            technology=_TECHNOLOGY,
            lane="agents",
            lifecycle_refs=_REFS,
            requested_execution="emulator",
            headline="LocalStack auth token is not configured.",
            message="Set LOCALSTACK_AUTH_TOKEN before emulator runs.",
            next_steps=["Start LocalStack, then rerun with emulator mode."],
        )
    try:
        selection = _create_emulator_model(
            settings=settings,
            policy=policy,
            selected_session=selected_session,
            model_factory=model_factory,
        )
    except SdkMissingError as exc:
        return not_run(
            demo=_DEMO,
            technology=_TECHNOLOGY,
            lane="agents",
            lifecycle_refs=_REFS,
            requested_execution="emulator",
            headline="Strands Bedrock provider SDK is not installed.",
            code="sdk_missing",
            message=sanitize_exception(exc),
        )
    except ValueError as exc:
        return missing_configuration(
            demo=_DEMO,
            technology=_TECHNOLOGY,
            lane="agents",
            lifecycle_refs=_REFS,
            requested_execution="emulator",
            headline="Strands emulator provider is not configured.",
            message=sanitize_exception(exc),
        )
    observed = ObservedModel(
        selection.model,
        "BedrockModel.stream",
        DispatchLimits(policy),
    )
    tool_calls_list: list[str] = []
    try:
        summary, stop_reason, tool_calls = _invoke_agent(
            observed,
            policy,
            tool_calls=tool_calls_list,
        )
    except ClientError as exc:
        return _emulator_client_error(
            settings,
            selection,
            exc,
            completed_calls=tuple(observed.calls),
            tool_calls=tuple(tool_calls_list),
        )
    except Exception as exc:
        if _is_wrapped_policy_limit(exc):
            return _emulator_preflight_blocked(
                settings,
                selection,
                exc,
                completed_calls=tuple(observed.calls),
                tool_calls=tuple(tool_calls_list),
            )
        if _is_endpoint_failure(exc):
            return _emulator_attempt_failed(settings, selection, exc)
        raise RuntimeError(sanitize_exception(exc)) from exc
    operations = _emulator_model_operations(
        observed.calls,
        endpoint_url=settings.localstack_endpoint,
    )
    operations.extend(_local_framework_operations(tool_calls))
    return _agent_result(
        execution="emulator",
        settings=settings,
        headline="Real Strands Agent loop called LocalStack Bedrock model.",
        operations=operations,
        provider=selection.provider,
        fixture_id=None,
        requested_model=selection.model_id,
        observed_model=selection.model_id,
        summary=summary,
        model_calls=tuple(observed.calls),
        stop_reason=stop_reason,
    )


def _run_live_budget_blocked(
    settings: Settings,
    *,
    provider: str,
) -> DemoResult:
    model_id = settings.model or (
        TESTED_LIVE_BEDROCK_MODEL if provider == "bedrock" else None
    )
    service = "openai" if provider == "openai" else "bedrock-runtime"
    operation_name = (
        "OpenAIModel.stream" if provider == "openai" else "Converse"
    )
    operations = [
        billable_not_priced(
            service=service,
            operation_name=operation_name,
        )
    ]
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        requested_execution="live",
        headline="Live Strands model calls need a verified price row.",
        operations=operations,
        evidence=evidence(
            provider=provider,
            region=settings.region,
            requested_model=model_id,
            observed_model=None,
            packages=_package_versions(("strands-agents", "botocore")),
        ),
        data={"model": model_id, "pricing": "missing", "provider": provider},
        error=error(
            "budget_exceeded",
            "Model call cost cannot yet be reserved from pricing snapshot.",
        ),
    )


def _create_emulator_model(
    *,
    settings: Settings,
    policy: ExecutionPolicy,
    selected_session: SelectedSession | None,
    model_factory: ModelFactory | None,
) -> ModelSelection:
    from awsai_demo.credentials import select_session

    active_session = selected_session or select_session(
        execution="emulator",
        settings=settings,
    )
    factory = model_factory or create_model
    return factory(
        provider="bedrock",
        execution="emulator",
        settings=settings,
        policy=policy,
        selected_session=active_session,
    )


def _invoke_agent(
    model: object,
    policy: ExecutionPolicy,
    *,
    tool_calls: list[str] | None = None,
    agent_builder: Callable[[Any], Any] | None = None,
) -> tuple[PilotSummary, str | None, tuple[str, ...]]:
    active_tool_calls = [] if tool_calls is None else tool_calls
    token = _TOOL_CALLS.set(active_tool_calls)
    build = build_agent if agent_builder is None else agent_builder
    try:
        agent_result = build(model)(
            scenario.BRIEF,
            limits={"turns": policy.max_agent_iterations},
        )
    finally:
        _TOOL_CALLS.reset(token)
    summary = getattr(agent_result, "structured_output", None)
    if not isinstance(summary, PilotSummary):
        message = "Strands agent did not produce typed PilotSummary output"
        raise TypeError(message)
    stop_reason = getattr(agent_result, "stop_reason", None)
    return (
        summary,
        None if stop_reason is None else str(stop_reason),
        tuple(active_tool_calls),
    )


def _agent_result(
    *,
    execution: Execution,
    settings: Settings,
    headline: str,
    operations: Sequence[OperationOutcome],
    provider: str,
    fixture_id: str | None,
    requested_model: str | None,
    observed_model: str | None,
    summary: PilotSummary,
    model_calls: tuple[str, ...],
    stop_reason: str | None,
) -> DemoResult:
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
            network_attempted=execution == "emulator",
            aws_executed=False,
            provider=provider,
            region=settings.region,
            fixture_id=fixture_id,
            requested_model=requested_model,
            observed_model=observed_model,
            packages=_package_versions(("strands-agents", "botocore")),
            emulator={"endpoint": settings.localstack_endpoint}
            if execution == "emulator"
            else None,
        ),
        data={
            "summary": summary.model_dump(),
            "summary_type": summary.__class__.__name__,
            "model_calls": model_calls,
            "stop_reason": stop_reason,
        },
    )


def _is_wrapped_policy_limit(exc: Exception) -> bool:
    message = str(exc)
    return "limit exceeded before Strands dispatch" in message


def _is_endpoint_failure(exc: Exception) -> bool:
    return isinstance(exc, OSError)


def _emulator_preflight_blocked(
    settings: Settings,
    selection: ModelSelection,
    exc: Exception,
    *,
    completed_calls: tuple[str, ...],
    tool_calls: tuple[str, ...],
) -> DemoResult:
    operations = _emulator_model_operations(
        completed_calls,
        endpoint_url=settings.localstack_endpoint,
    )
    operations.extend(_local_framework_operations(tool_calls))
    operations.append(
        operation(
            service="bedrock-runtime",
            operation="BedrockModel.stream:refused",
            execution_target="emulator",
            mode="not_run",
            status="blocked",
            response_received=False,
            request_validated=False,
            error_code="budget_exceeded",
        )
    )
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        requested_execution="emulator",
        headline="Strands emulator dispatch was refused by local limits.",
        operations=operations,
        evidence=evidence(
            sdk_invoked=bool(completed_calls),
            network_attempted=bool(completed_calls),
            provider=selection.provider,
            region=settings.region,
            requested_model=selection.model_id,
            packages=_package_versions(("strands-agents", "botocore")),
            emulator={"endpoint": settings.localstack_endpoint}
            if completed_calls
            else None,
        ),
        data={
            "model": selection.model_id,
            "model_calls": completed_calls,
            "tool_calls": tool_calls,
        },
        error=error("budget_exceeded", sanitize_exception(exc)),
    )


def _emulator_client_error(
    settings: Settings,
    selection: ModelSelection,
    exc: ClientError,
    *,
    completed_calls: tuple[str, ...],
    tool_calls: tuple[str, ...],
) -> DemoResult:
    metadata = exc.response.get("ResponseMetadata", {})
    payload = exc.response.get("Error", {})
    status_code = int(metadata.get("HTTPStatusCode", 0)) or None
    code = str(payload.get("Code", exc.__class__.__name__))
    public_code = "authorization_denied" if status_code == 403 else code
    operations = _emulator_model_operations(
        completed_calls,
        endpoint_url=settings.localstack_endpoint,
    )
    operations.extend(_local_framework_operations(tool_calls))
    operations.append(
        operation(
            service="bedrock-runtime",
            operation="BedrockModel.stream",
            execution_target="emulator",
            mode="local_emulator",
            status="blocked",
            effect="infer",
            transport="loopback",
            endpoint_url=settings.localstack_endpoint,
            response_received=True,
            request_validated=True,
            http_status=status_code,
            error_code=public_code,
        )
    )
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        requested_execution="emulator",
        headline="LocalStack Bedrock model returned a service error.",
        operations=operations,
        evidence=evidence(
            sdk_invoked=True,
            network_attempted=True,
            provider=selection.provider,
            region=settings.region,
            requested_model=selection.model_id,
            packages=_package_versions(("strands-agents", "botocore")),
            emulator={"endpoint": settings.localstack_endpoint},
        ),
        data={
            "model": selection.model_id,
            "model_calls": completed_calls,
            "tool_calls": tool_calls,
        },
        error=error("authorization_denied", sanitize_exception(exc))
        if status_code == 403
        else error("emulator_unavailable", sanitize_exception(exc)),
    )


def _emulator_attempt_failed(
    settings: Settings,
    selection: ModelSelection,
    exc: Exception,
) -> DemoResult:
    operations = [
        operation(
            service="bedrock-runtime",
            operation="BedrockModel.stream",
            execution_target="emulator",
            mode="attempt_failed",
            status="blocked",
            effect="infer",
            transport="loopback",
            endpoint_url=settings.localstack_endpoint,
            response_received=False,
            request_validated=True,
            error_code="emulator_unavailable",
        )
    ]
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="agents",
        lifecycle_refs=_REFS,
        requested_execution="emulator",
        headline="LocalStack Bedrock model call did not complete.",
        operations=operations,
        evidence=evidence(
            sdk_invoked=True,
            network_attempted=True,
            provider=selection.provider,
            region=settings.region,
            requested_model=selection.model_id,
            observed_model=None,
            packages=_package_versions(("strands-agents", "botocore")),
            emulator={"endpoint": settings.localstack_endpoint},
        ),
        data={"model": selection.model_id},
        error=error("emulator_unavailable", sanitize_exception(exc)),
    )


def _operations_for_model_calls(
    calls: Sequence[str],
) -> list[OperationOutcome]:
    usage: Usage = {
        "input_tokens": 80,
        "output_tokens": 20,
        "total_tokens": 100,
    }
    return [
        operation(
            service="bedrock-runtime",
            operation=f"ScriptedModel.stream:{name}",
            mode="local_contract",
            effect="infer",
            fixture_id=FIXTURE_ID,
            usage=usage,
        )
        for name in calls
    ]


def _emulator_model_operations(
    calls: Sequence[str],
    *,
    endpoint_url: str,
) -> list[OperationOutcome]:
    return [
        operation(
            service="bedrock-runtime",
            operation=f"{name}:{index}",
            execution_target="emulator",
            mode="local_emulator",
            effect="infer",
            transport="loopback",
            endpoint_url=endpoint_url,
            usage=None,
        )
        for index, name in enumerate(calls, start=1)
    ]


def _local_framework_operations(
    tool_calls: Sequence[str],
) -> list[OperationOutcome]:
    operations = [
        operation(
            service="strands",
            operation="Agent.__call__",
            execution_target="local",
            mode="local_execution",
        )
    ]
    operations.extend(
        operation(
            service="strands",
            operation=name,
            execution_target="local",
            mode="local_execution",
        )
        for name in tool_calls
    )
    return operations


def _tool_use_events(
    tool_use_id: str,
    name: str,
) -> tuple[dict[str, object], ...]:
    payload = _summary_payload() if name == "PilotSummary" else {}
    return (
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {
                    "toolUse": {"toolUseId": tool_use_id, "name": name},
                },
            },
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": _json_payload(payload)}},
            },
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        _metadata_event(),
    )


def _text_events(text: str) -> tuple[dict[str, object], ...]:
    return (
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": text},
            },
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        _metadata_event(),
    )


async def _stream_events(
    events: Iterable[dict[str, object]],
) -> AsyncIterable[StreamEvent]:
    for event in events:
        yield cast("StreamEvent", event)


def _latest_message_has_tool_result(messages: Sequence[Message]) -> bool:
    if not messages:
        return False
    return any("toolResult" in item for item in messages[-1]["content"])


def _forced_tool_name(tool_choice: ToolChoice | None) -> str | None:
    raw_choice = cast("dict[str, Any] | None", tool_choice)
    if raw_choice is None or "tool" not in raw_choice:
        return None
    tool_payload = raw_choice["tool"]
    if not isinstance(tool_payload, dict):
        return None
    name = tool_payload.get("name")
    return None if name is None else str(name)


def _summary_payload() -> dict[str, object]:
    return PilotSummary.from_cost(scenario.price_pilot()).model_dump()


def _json_payload(value: object) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _metadata_event() -> dict[str, object]:
    return {
        "metadata": {
            "usage": {
                "inputTokens": 80,
                "outputTokens": 20,
                "totalTokens": 100,
            },
            "metrics": {"latencyMs": 1},
        },
    }


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions
