"""Compare approval lanes using a real resumable Strands agent.

Lane: agents. Lifecycle: strands-agents, agentcore-harness,
agents-classic.
Run: uv run awsai-demo decision
Run: uv run awsai-demo decision --simulate-approval
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from botocore.exceptions import BotoCoreError, ClientError
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field
from strands.types.exceptions import MaxTokensReachedException

from awsai_demo import billing
from awsai_demo.contracts import (
    DemoResult,
    ErrorCode,
    Execution,
    OperationOutcome,
    Status,
    evidence,
    not_run,
    operation,
    result,
)
from awsai_demo.decision_state import (
    Approval,
    ApprovalSink,
    ApprovalSource,
    SQLiteApprovalSink,
    StoredDecision,
    policy_fingerprint,
    read_decision,
    session_directory,
    validate_approval,
    validate_resume,
    write_decision,
)
from awsai_demo.demo_support import build_result, not_run_operation
from awsai_demo.live_agent import LiveAgentModel
from awsai_demo.network import (
    NetworkPolicy,
    active_policy,
    environment_for_subprocess,
)
from awsai_demo.policy import BudgetLedger, ExecutionPolicy, PolicyError
from awsai_demo.redact import quiet_sdk_logging, sanitize_exception
from awsai_demo.runtime import TESTED_LIVE_BEDROCK_MODEL, Settings
from awsai_demo.scenario import (
    BRIEF,
    PilotCost,
    price_pilot,
    proposal_is_acceptable,
    proposal_version,
)
from awsai_demo.strands_errors import STRANDS_FAILURES, unwrap_strands_error
from awsai_demo.strands_multiagent_demo import flag_max_tokens
from awsai_demo.stubs import named_fixture_stream, validate_operation_request

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from strands.models.model import Model
    from strands.types.interrupt import InterruptResponseContent

    from awsai_demo.credentials import SelectedSession
    from awsai_demo.policy import TokenCounter
    from awsai_demo.strands_multiagent_demo import ModelFactory

_REFS = ["strands-agents", "agentcore-harness", "agents-classic"]


class ResumeRequest(BaseModel):
    """Non-secret instructions passed to a separate worker process."""

    model_config = ConfigDict(extra="forbid")
    root: str
    run_id: str
    action: Literal["approve", "deny", "replay"]
    approver: str = ""
    approval_source: ApprovalSource = "none"
    execution: Execution = "offline"
    model_id: str = "scripted-decision"
    region: str = "us-east-1"
    policy_fingerprint: str
    policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    aws_profile: str | None = None
    aws_creds_file_path: str | None = None
    budget_directory: str | None = None
    budget_aggregate: bool = False


class ResumeReport(BaseModel):
    """Public recovery evidence, excluding private session contents."""

    model_config = ConfigDict(extra="forbid")
    run_id: str
    pause_pid: int
    resume_pid: int
    files_read: list[str]
    effects: int
    approval_source: ApprovalSource
    status: Status
    operations: list[OperationOutcome]


class ResumeWorker(Protocol):
    """Process boundary for a resumed command."""

    def run(self, request: ResumeRequest) -> ResumeReport:
        """Resume from durable state, returning public evidence."""


class SubprocessResumeWorker:
    """Run the worker with inherited egress protection and a timeout."""

    def __init__(self, policy: ExecutionPolicy) -> None:
        """Retain the command's application limits."""
        self.policy = policy

    def run(self, request: ResumeRequest) -> ResumeReport:
        """Launch a separate interpreter with sanitized output."""
        directory = session_directory(Path(request.root), request.run_id)
        env = environment_for_subprocess(
            active_policy() or NetworkPolicy(),
            bootstrap_dir=directory / "bootstrap",
        )
        # Fixed interpreter and module; argument list, never a shell.
        completed = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "awsai_demo.resume_worker",
                "--request",
                request.model_dump_json(),
            ],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=self.policy.max_wall_seconds,
            check=False,
        )
        if completed.returncode:
            message = sanitize_exception(RuntimeError(completed.stderr))
            raise RuntimeError(message)
        if completed.stderr:
            print(
                sanitize_exception(RuntimeError(completed.stderr)),
                file=sys.stderr,
            )
        # Accept only structured worker output as public evidence.
        return ResumeReport.model_validate_json(completed.stdout)


def apply_approval(
    cost: PilotCost,
    response: object,
    sink: ApprovalSink,
    after_effect: Callable[[], None] | None,
) -> str:
    """Require explicit approval before the business effect."""
    approval = Approval.model_validate(response)
    validate_approval(cost, approval)
    if not approval.approved:
        return "Human denied the proposal; no business effect."
    if not cost.within_budget:
        return "Proposal is over budget; no business effect."
    sink.commit(cost, approval)
    if after_effect is not None:
        after_effect()
    return "Approval recorded; the business action exists exactly once."


def acceptance_tool(
    cost: PilotCost,
    sink: ApprovalSink,
    after_effect: Callable[[], None] | None = None,
) -> object:
    """Create the real interrupting tool, with its sink injected."""
    from strands import tool

    # slide: interrupt
    @tool(context=True)
    def accept_proposal(tool_context: Any) -> str:
        """Request explicit approval before accepting the proposal."""
        approval = tool_context.interrupt(
            "approval", reason={"proposal_version": proposal_version(cost)}
        )
        return apply_approval(cost, approval, sink, after_effect)

    # end-slide: interrupt

    return accept_proposal


def _local_operation(name: str, status: Status = "ok") -> OperationOutcome:
    return operation(
        service="strands",
        operation=name,
        execution_target="local",
        mode="local_execution",
        status=status,
        request_validated=False,
    )


def _model_id(execution: Execution, settings: Settings) -> str:
    if execution == "offline":
        return "scripted-decision"
    if execution == "live":
        return settings.model or TESTED_LIVE_BEDROCK_MODEL
    return settings.model or settings.default_bedrock_model


def _dependencies(
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    root: Path,
    *,
    sink: ApprovalSink | None = None,
    model: Model | None = None,
) -> tuple[ApprovalSink, Model | None]:
    if execution == "emulator":
        from awsai_demo.credentials import select_session
        from awsai_demo.dynamo_sink import DynamoApprovalSink
        from awsai_demo.providers import create_model

        selected = select_session(execution="emulator", settings=settings)
        return (
            sink or DynamoApprovalSink(settings=settings),
            model
            or cast(
                "Model",
                create_model(
                    provider="bedrock",
                    execution="emulator",
                    settings=settings,
                    policy=policy,
                    selected_session=selected,
                ).model,
            ),
        )
    return sink or SQLiteApprovalSink(root / "ledger.sqlite3"), model


def pause_decision(
    *,
    root: Path,
    run_id: str,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    cost: PilotCost,
    sink: ApprovalSink | None = None,
    model: Model | None = None,
) -> ResumeReport:
    """Run to a real interrupt and persist the paused session."""
    from awsai_demo.decision_engine import build_agent

    directory = session_directory(root, run_id)
    directory.mkdir(parents=True, exist_ok=True)
    sink, model = _dependencies(
        execution, settings, policy, root, sink=sink, model=model
    )
    with FileLock(str(directory / "resume.lock"), timeout=30):
        agent, observed, _ = build_agent(
            directory=directory,
            run_id=run_id,
            acceptance_tool=acceptance_tool(cost, sink),
            policy=policy,
            execution=execution,
            model=model,
            endpoint=settings.localstack_endpoint
            if execution == "emulator"
            else None,
        )
        turn = agent(BRIEF)
        if turn.stop_reason != "interrupt" or not turn.interrupts:
            message = "Agent did not reach the required approval interrupt"
            raise ValueError(message)
        record = StoredDecision(
            run_id=run_id,
            execution=execution,
            model_id=_model_id(execution, settings),
            region=settings.region,
            policy_fingerprint=policy_fingerprint(policy),
            proposal_version=proposal_version(cost),
            pause_pid=os.getpid(),
            interrupt_id=turn.interrupts[0].id,
        )
        write_decision(directory, record)
    return ResumeReport(
        run_id=run_id,
        pause_pid=os.getpid(),
        resume_pid=0,
        files_read=[],
        effects=0,
        approval_source="none",
        status="paused",
        operations=[
            *observed.operations,
            _local_operation("accept_proposal.interrupt", "paused"),
            _local_operation("Agent.loop", "paused"),
        ],
    )


def resume_from_disk(
    request: ResumeRequest,
    *,
    policy: ExecutionPolicy | None = None,
    cost: PilotCost | None = None,
    sink: ApprovalSink | None = None,
    model: Model | None = None,
    after_effect: Callable[[], None] | None = None,
    resume_model: Callable[..., LiveAgentModel] | None = None,
) -> ResumeReport:
    """Load under a lock and answer the persisted interrupt."""
    from awsai_demo.decision_engine import build_agent

    limits = policy or request.policy
    proposal = cost or price_pilot()
    root = Path(request.root)
    directory = session_directory(root, request.run_id)
    if request.policy_fingerprint != policy_fingerprint(limits):
        message = "Worker policy differs from the parent policy"
        raise ValueError(message)
    with FileLock(str(directory / "resume.lock"), timeout=30):
        record = read_decision(directory)
        validate_resume(
            record,
            execution=request.execution,
            model_id=request.model_id,
            region=request.region,
            policy=limits,
            cost=proposal,
        )
        if record.run_id != request.run_id:
            message = "Checkpoint run id does not match the requested run"
            raise ValueError(message)
        settings = Settings.from_env()
        settings = replace(
            settings,
            region=request.region,
            model=request.model_id,
            aws_profile=request.aws_profile,
            aws_creds_file_path=None
            if request.aws_creds_file_path is None
            else Path(request.aws_creds_file_path),
        )
        sink, model = _dependencies(
            request.execution, settings, limits, root, sink=sink, model=model
        )
        files = ["decision.json"]
        operations = [_local_operation("approval_sink.lookup")]
        if request.action != "replay" and not record.completed:
            approval = Approval(
                approved=request.action == "approve",
                proposal_version=record.proposal_version,
                approver=request.approver,
                source=request.approval_source,
            )
            validate_approval(proposal, approval)
            if record.approval is not None and record.approval != approval:
                message = "A persisted approval intent cannot be replaced"
                raise ValueError(message)
            # Intent survives a crash after the business transaction but
            # before Strands saves its next checkpoint.
            record.approval = approval
            write_decision(directory, record)
        if request.action != "replay" and not record.completed:
            if request.execution == "live" and model is None:
                try:
                    build_model = (
                        live_resume_model
                        if resume_model is None
                        else resume_model
                    )
                    model = build_model(request, settings, limits)
                except (PolicyError, ValueError, TypeError) as exc:
                    return _resume_failure_report(
                        request,
                        record,
                        files,
                        operations,
                        sink,
                        proposal,
                        settings,
                        None,
                        exc,
                    )
            agent, observed, session = build_agent(
                directory=directory,
                run_id=request.run_id,
                acceptance_tool=acceptance_tool(proposal, sink, after_effect),
                policy=limits,
                execution=request.execution,
                model=model,
                endpoint=settings.localstack_endpoint
                if request.execution == "emulator"
                else None,
            )
            responses: list[InterruptResponseContent] = [
                {
                    "interruptResponse": {
                        "interruptId": record.interrupt_id,
                        "response": cast(
                            "Approval", record.approval
                        ).model_dump(),
                    }
                }
            ]
            try:
                agent(responses)
            except (
                PolicyError,
                ClientError,
                BotoCoreError,
                OSError,
                ValueError,
                *STRANDS_FAILURES,
            ) as exc:
                if request.execution != "live":
                    raise
                files.extend(session.files_read)
                return _resume_failure_report(
                    request,
                    record,
                    files,
                    operations,
                    sink,
                    proposal,
                    settings,
                    cast("LiveAgentModel", model),
                    unwrap_strands_error(exc),
                )
            files.extend(session.files_read)
            operations.extend(observed.operations)
            operations.append(_local_operation("Agent.resume"))
            operations.append(_local_operation("accept_proposal"))
            record.completed = True
            write_decision(directory, record)
        effect = sink.find(proposal)
        operations.extend(
            cast(
                "Sequence[OperationOutcome]",
                getattr(sink, "operations", ()),
            )
        )
        status: Status = "paused"
        if record.approval is not None:
            status = (
                "ok"
                if effect is not None and record.approval.approved
                else "blocked"
            )
        source = "none" if record.approval is None else record.approval.source
        return ResumeReport(
            run_id=request.run_id,
            pause_pid=record.pause_pid,
            resume_pid=os.getpid(),
            files_read=sorted(set(files)),
            effects=int(effect is not None),
            approval_source=source,
            status=status,
            operations=operations,
        )


def _classic_lane(report: ResumeReport, cost: PilotCost) -> DemoResult:
    params = {
        "agentId": "ABCDEFGHIJ",
        "agentAliasId": "TSTALIASID",
        "sessionId": report.run_id,
        "inputText": BRIEF,
    }
    validated = validate_operation_request(
        service_name="bedrock-agent-runtime",
        operation_name="InvokeAgent",
        params=params,
        fixture_id="classic-return-control-v1",
    )
    operations = [
        operation(
            service=validated.service,
            operation=validated.operation,
            fixture_id=validated.fixture_id,
        )
    ]
    stream = named_fixture_stream(
        fixture_id=validated.fixture_id,
        events=[
            {
                "returnControl": {
                    "invocationId": "approval-contract",
                    "invocationInputs": [
                        {
                            "functionInvocationInput": {
                                "actionGroup": "pilot",
                                "function": "accept_proposal",
                            }
                        }
                    ],
                }
            }
        ],
    )
    return_control = next(iter(stream))["returnControl"]
    approved = report.status == "ok"
    if report.status != "paused":
        resumed = {
            "agentId": "ABCDEFGHIJ",
            "agentAliasId": "TSTALIASID",
            "sessionId": report.run_id,
            "sessionState": {
                "invocationId": "approval-contract",
                "returnControlInvocationResults": [
                    {
                        "functionResult": {
                            "actionGroup": "pilot",
                            "function": "accept_proposal",
                            "responseBody": {
                                "TEXT": {
                                    "body": (
                                        f"approved={approved}; "
                                        f"version={proposal_version(cost)}"
                                    )
                                }
                            },
                        }
                    }
                ],
            },
        }
        validate_operation_request(
            service_name=validated.service,
            operation_name=validated.operation,
            params=resumed,
            fixture_id="classic-return-result-v1",
        )
        operations.append(
            operation(
                service=validated.service,
                operation=validated.operation,
                fixture_id="classic-return-result-v1",
            )
        )
    return result(
        demo="decision-classic",
        technology="Agents Classic return of control",
        lane="agents",
        lifecycle_refs=["agents-classic"],
        requested_execution="offline",
        headline="Named contract fixture; no managed agent executed.",
        operations=operations,
        evidence=evidence(
            sdk_invoked=True,
            fixture_id=validated.fixture_id,
            packages={
                "boto3": validated.boto3_version,
                "botocore": validated.botocore_version,
            },
        ),
        data={
            "returnControl": return_control,
            "approved": approved,
            "proposal_version": proposal_version(cost),
        },
    )


def _matrix() -> tuple[list[list[str]], dict[str, str]]:
    rows = [
        ["model choice", "application", "application", "application"],
        ["orchestration loop", "aws_service", "aws_service", "framework"],
        ["tool authorisation", "application", "application", "application"],
        ["approval pause", "aws_service", "aws_service", "framework"],
        ["state persistence", "aws_service", "aws_service", "framework"],
        ["resume after crash", "application", "application", "application"],
        ["deduplication", "application", "application", "application"],
        ["tracing", "aws_service", "aws_service", "framework"],
        ["cost limits", "application", "application", "application"],
    ]
    notes = {
        "model choice": "The application selects the model in each lane.",
        "orchestration loop": "AWS owns managed loops; Strands owns its loop.",
        "tool authorisation": "Application checks guard business effects.",
        "approval pause": "Return control, toolUse, or a Strands interrupt.",
        "state persistence": "Managed sessions or the FileSessionManager.",
        "resume after crash": "The caller reconstructs its session.",
        "deduplication": "The business-action key belongs to the application.",
        "tracing": "Service traces or the framework's local instrumentation.",
        "cost limits": "Application reservations; never a billing ceiling.",
    }
    return rows, notes


def _decision_result(
    report: ResumeReport,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    cost: PilotCost,
    replay_effects: int | None,
) -> DemoResult:
    from awsai_demo.agentcore_harness_demo import (
        build_harness_pause_contract,
        build_harness_resume_request,
        run_agentcore_harness_demo,
    )

    if execution == "offline":
        classic = _classic_lane(report, cost)
        harness = run_agentcore_harness_demo(
            execution="offline", settings=settings, policy=policy
        )
        if report.status != "paused":
            resumed = build_harness_resume_request(
                build_harness_pause_contract(),
                approved=report.status == "ok",
                reason=f"Bound to {proposal_version(cost)}",
            )
            validate_operation_request(
                service_name="bedrock-agentcore",
                operation_name="InvokeHarness",
                params=resumed,
                fixture_id="harness-tool-result-v1",
            )
            harness = result(
                demo=harness["demo"],
                technology=harness["technology"],
                lane="agents",
                lifecycle_refs=harness["lifecycle_refs"],
                requested_execution="offline",
                headline="The approved or denied resume contract validated.",
                status="ok",
                operations=[
                    *harness["operations"],
                    operation(
                        service="bedrock-agentcore",
                        operation="InvokeHarness",
                        fixture_id="harness-tool-result-v1",
                    ),
                ],
                evidence=harness["evidence"],
                data={"resume_contract": resumed},
            )
    else:
        classic = not_run(
            demo="decision-classic",
            technology="Agents Classic",
            lane="agents",
            lifecycle_refs=["agents-classic"],
            requested_execution=execution,
            headline="The Classic comparison is an offline contract only.",
        )
        harness = not_run(
            demo="decision-harness",
            technology="AgentCore harness",
            lane="agents",
            lifecycle_refs=["agentcore-harness"],
            requested_execution=execution,
            headline="Use the separately bounded agentcore-harness demo.",
        )
    emulated = execution == "emulator"
    strands = result(
        demo="decision-strands",
        technology="Strands with a durable approval sink",
        lane="agents",
        lifecycle_refs=["strands-agents"],
        requested_execution=execution,
        headline="At most once applies only to the sink effect.",
        operations=report.operations,
        status=report.status,
        evidence=evidence(
            sdk_invoked=True,
            network_attempted=any(
                op["transport"] in {"aws", "loopback", "external"}
                for op in report.operations
            ),
            aws_executed=any(
                op["execution_target"] == "aws" and op["response_received"]
                for op in report.operations
            ),
            fixture_id=next(
                (
                    op["fixture_id"]
                    for op in report.operations
                    if op["fixture_id"]
                ),
                None,
            ),
            emulator={
                "endpoint": settings.localstack_endpoint,
                "image": None,
                "digest": None,
            }
            if emulated
            else None,
        ),
    )
    children = [classic, harness, strands]
    matrix, notes = _matrix()
    return result(
        demo="decision",
        technology="One decision, three implementations",
        lane="agents",
        lifecycle_refs=_REFS,
        requested_execution=execution,
        headline=f"Run {report.run_id}: {report.status}; explicit approval.",
        children=children,
        data={
            "run_id": report.run_id,
            "pause_pid": report.pause_pid,
            "resume_pid": report.resume_pid or None,
            "files_read": report.files_read,
            "effects_after_resume": report.effects,
            "effects_after_replay": replay_effects,
            "approval_source": report.approval_source,
            "approver_is_authenticated_identity": False,
            "approval_note": "The approver is a caller-supplied audit label.",
            "scenario": {
                "budget": int(cost.budget_usd),
                "staffing_total": int(cost.staffing_usd),
                "platform_total": int(cost.platform_usd),
                "total": int(cost.total_usd),
                "headroom": int(cost.headroom_usd),
            },
            "checks": proposal_is_acceptable(cost),
            "lanes": [
                [label, child["technology"], child["mode"], child["status"]]
                for label, child in zip(("A", "B", "C"), children, strict=True)
            ],
            "responsibility_matrix": matrix,
            "responsibility_notes": notes,
        },
    )


@quiet_sdk_logging()
def run_decision_demo(
    *,
    execution: Execution = "offline",
    settings: Settings | None = None,
    policy: ExecutionPolicy | None = None,
    simulate_approval: bool = False,
    resume: str | None = None,
    replay: str | None = None,
    approve: bool = False,
    deny: bool = False,
    approver: str | None = None,
    checkpoint_root: Path = Path(".awsai_checkpoints"),
    worker: ResumeWorker | None = None,
    selected_session: SelectedSession | None = None,
    model_factory: ModelFactory | None = None,
    exact_counter: TokenCounter | None = None,
    budget_opener: Callable[..., billing.BudgetRun] | None = None,
    pause: Callable[..., ResumeReport] | None = None,
) -> DemoResult:
    """Pause by default; never synthesize a human approval."""
    config = settings or Settings()
    limits = policy or ExecutionPolicy()
    expected_provider = "offline" if execution == "offline" else "bedrock"
    if config.provider is not None and config.provider != expected_provider:
        message = "Decision provider must match its execution lane"
        raise ValueError(message)
    if simulate_approval and execution != "offline":
        message = "--simulate-approval is offline only"
        raise ValueError(message)
    if execution == "emulator" and not config.localstack_auth_token:
        return not_run(
            demo="decision",
            technology="Human-approved decision",
            lane="agents",
            lifecycle_refs=_REFS,
            requested_execution=execution,
            headline="LocalStack auth token is required.",
            code="missing_configuration",
        )
    if resume and (approve == deny or not approver or not approver.strip()):
        message = "Resume needs exactly one decision and a named approver"
        raise ValueError(message)
    # A leading hex letter avoids confusing this id with an account.
    identifier = resume or replay or "a" + uuid.uuid4().hex[:11]
    root = checkpoint_root.resolve()
    proposal = price_pilot()
    process = worker or SubprocessResumeWorker(limits)
    replay_effects = None
    live_model = None
    budget_run = None
    if execution == "live" and not replay:
        try:
            open_budget = (
                billing.active_budget_run
                if budget_opener is None
                else budget_opener
            )
            budget_run = open_budget(region=config.region, policy=limits)
            if not resume:
                live_model = LiveAgentModel(
                    demo="decision",
                    settings=config,
                    policy=limits,
                    run=budget_run,
                    selected_session=selected_session,
                    model_factory=model_factory,
                    exact_counter=exact_counter,
                )
        except (PolicyError, ValueError, TypeError) as exc:
            return _live_refused(config, None, exc)
    request = ResumeRequest(
        root=str(root),
        run_id=identifier,
        action="replay" if replay else "approve" if approve else "deny",
        approver=approver or "",
        approval_source="cli" if resume else "none",
        execution=execution,
        model_id=_model_id(execution, config),
        region=config.region,
        policy_fingerprint=policy_fingerprint(limits),
        policy=limits,
        aws_profile=config.aws_profile,
        aws_creds_file_path=None
        if config.aws_creds_file_path is None
        else str(config.aws_creds_file_path),
        budget_directory=None
        if budget_run is None
        else str(budget_run.directory),
        budget_aggregate=budget_run is not None
        and budget_run.aggregate is not None,
    )
    if resume or replay:
        report = process.run(request)
        if replay:
            replay_effects = report.effects
    else:
        try:
            record_pause = pause_decision if pause is None else pause
            report = record_pause(
                root=root,
                run_id=identifier,
                execution=execution,
                settings=config,
                policy=limits,
                cost=proposal,
                model=live_model,
            )
        except (
            PolicyError,
            ClientError,
            BotoCoreError,
            OSError,
            ValueError,
            *STRANDS_FAILURES,
        ) as exc:
            if execution != "live":
                raise
            return _live_refused(config, live_model, unwrap_strands_error(exc))
        if simulate_approval:
            request.action = "approve"
            request.approver = "workshop-simulation"
            request.approval_source = "simulated"
            report = process.run(request)
            request.action = "replay"
            replay_effects = process.run(request).effects
    return _decision_result(
        report, execution, config, limits, proposal, replay_effects
    )


def live_resume_model(
    request: ResumeRequest,
    settings: Settings,
    policy: ExecutionPolicy,
    *,
    model_type: Callable[..., LiveAgentModel] | None = None,
) -> LiveAgentModel:
    """Reopen the parent's durable command and aggregate budgets."""
    if request.budget_directory is None:
        message = "Live worker requires its parent's budget directory"
        raise ValueError(message)
    directory = Path(request.budget_directory)
    run = billing.BudgetRun(
        directory.name,
        directory,
        settings.region,
        policy,
        billing.SnapshotPrices(Path("data/pricing_snapshot.json")),
        BudgetLedger(directory / "budget.json", policy, all_live=True)
        if request.budget_aggregate
        else None,
    )
    build = LiveAgentModel if model_type is None else model_type
    return build(
        demo="decision",
        settings=settings,
        policy=policy,
        run=run,
    )


def _live_refused(
    settings: Settings, model: LiveAgentModel | None, exc: Exception
) -> DemoResult:
    capped = isinstance(exc, MaxTokensReachedException)
    if model is not None and capped:
        flag_max_tokens(model.completed)
    code: ErrorCode = (
        "budget_exceeded"
        if isinstance(exc, PolicyError)
        else "validation_failed"
        if isinstance(exc, ValueError)
        else "max_tokens_reached"
        if capped
        else "model_unavailable"
    )
    operations = [] if model is None else model.operations
    if not any(item["status"] != "ok" for item in operations):
        operations.append(
            not_run_operation(
                service="bedrock-runtime",
                operation_name="DecisionModel.stream",
                error_code=code,
                request_validated=False,
            )
        )
    return build_result(
        demo="decision",
        technology="Human-approved decision",
        lane="agents",
        lifecycle_refs=_REFS,
        execution="live",
        settings=settings,
        operations=operations,
        headline="The live decision stopped before an unreserved dispatch.",
        data={"approval_source": "none", "effects_after_resume": 0},
        result_error=(code, sanitize_exception(exc)),
    )


def _resume_failure_report(
    request: ResumeRequest,
    record: StoredDecision,
    files: list[str],
    operations: list[OperationOutcome],
    sink: ApprovalSink,
    proposal: PilotCost,
    settings: Settings,
    model: LiveAgentModel | None,
    exc: Exception,
) -> ResumeReport:
    """Retain structured evidence if failure follows the sink effect."""
    failed = _live_refused(settings, model, exc)
    return ResumeReport(
        run_id=request.run_id,
        pause_pid=record.pause_pid,
        resume_pid=os.getpid(),
        files_read=sorted(set(files)),
        effects=int(sink.find(proposal) is not None),
        approval_source=cast("Approval", record.approval).source,
        status="blocked",
        operations=[
            *operations,
            *failed["operations"],
            _local_operation("Agent.resume", "blocked"),
        ],
    )
