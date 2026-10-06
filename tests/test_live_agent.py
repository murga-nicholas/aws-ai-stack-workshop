from __future__ import annotations

import json
import subprocess
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError

from awsai_demo import billing
from awsai_demo.agentcore_harness_demo import harness_cost_preflight
from awsai_demo.decision_demo import (
    ResumeReport,
    ResumeRequest,
    SubprocessResumeWorker,
    live_resume_model,
    resume_from_disk,
    run_decision_demo,
)
from awsai_demo.decision_engine import DecisionModel
from awsai_demo.decision_state import policy_fingerprint
from awsai_demo.live_agent import LiveAgentModel
from awsai_demo.policy import ExecutionPolicy, ReservationUnavailable
from awsai_demo.providers import ModelSelection
from awsai_demo.runtime import TESTED_LIVE_BEDROCK_MODEL, Settings
from awsai_demo.strands_agent_demo import (
    PricePilotScriptedModel,
    run_strands_agent_demo,
)

if TYPE_CHECKING:
    from pathlib import Path


class Prices:
    def rate(self, _feature: str, **_kwargs: Any) -> Decimal:
        return Decimal("0.000001")


class Counter:
    def count_tokens(self, _model_id: str, request: bytes) -> int:
        payload = json.loads(request)
        assert payload["messages"]
        return 300


def price_factory(**_kwargs: Any) -> ModelSelection:
    return ModelSelection(
        PricePilotScriptedModel(), "bedrock", TESTED_LIVE_BEDROCK_MODEL
    )


def decision_factory(**_kwargs: Any) -> ModelSelection:
    return ModelSelection(
        DecisionModel(ExecutionPolicy()),
        "bedrock",
        TESTED_LIVE_BEDROCK_MODEL,
    )


def budget(tmp_path: Path, **kwargs: Any) -> billing.BudgetRun:
    return billing.open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        **kwargs,
    )


def test_single_agent_live_reserves_every_actual_turn(tmp_path: Path) -> None:
    run = budget(tmp_path)
    with billing.use_budget_run(run):
        built = run_strands_agent_demo(
            execution="live",
            model_factory=price_factory,
            exact_counter=Counter(),
        )
    assert built["mode"] == "live_model"
    assert built["status"] == "ok"
    assert built["data"]["summary_type"] == "PilotSummary"
    calls = [op for op in built["operations"] if op["mode"] == "live_model"]
    assert len(calls) == 3
    assert all(op["reserved_usd"] > 0 for op in calls)
    assert run.command("strands-agent").ledger.snapshot().model_calls == 3


def test_live_reservations_survive_pause_resume_and_replay(
    tmp_path: Path,
) -> None:
    limits = ExecutionPolicy()
    run = budget(tmp_path / "budget", aggregate=True)
    settings = Settings()
    with billing.use_budget_run(run):
        built = run_decision_demo(
            execution="live",
            checkpoint_root=tmp_path / "state",
            model_factory=decision_factory,
            exact_counter=Counter(),
        )
        assert built["children"][2]["status"] == "paused"
        assert built["data"]["approval_source"] == "none"
        assert built["data"]["effects_after_resume"] == 0
        request = ResumeRequest(
            root=str(tmp_path / "state"),
            run_id=built["data"]["run_id"],
            action="approve",
            approver="human",
            approval_source="cli",
            execution="live",
            model_id=TESTED_LIVE_BEDROCK_MODEL,
            policy_fingerprint=policy_fingerprint(limits),
        )
        model = LiveAgentModel(
            demo="decision",
            settings=settings,
            policy=limits,
            model_factory=decision_factory,
            exact_counter=Counter(),
        )
        resumed = resume_from_disk(request, model=model)
        replayed = resume_from_disk(
            request.model_copy(update={"action": "replay"})
        )
    assert resumed.effects == replayed.effects == 1
    snapshot = run.command("decision").ledger.snapshot()
    assert snapshot.model_calls == 2
    assert len(snapshot.reservations) == 2
    assert run.aggregate is not None
    assert run.aggregate.snapshot().reserved_usd == snapshot.reserved_usd


def test_single_agent_and_decision_keep_failed_dispatch_evidence(
    tmp_path: Path,
) -> None:
    class Failing(PricePilotScriptedModel):
        def stream(self, *_args: Any, **_kwargs: Any) -> Any:
            raise ClientError(
                {
                    "Error": {"Code": "AccessDeniedException"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                "Converse",
            )

    def factory(**_kwargs: Any) -> ModelSelection:
        return ModelSelection(Failing(), "bedrock", TESTED_LIVE_BEDROCK_MODEL)

    for command in (run_strands_agent_demo, run_decision_demo):
        with billing.use_budget_run(budget(tmp_path / command.__name__)):
            extra = (
                {"checkpoint_root": tmp_path / "state"}
                if (command is run_decision_demo)
                else {}
            )
            result = command(
                execution="live",
                model_factory=factory,
                exact_counter=Counter(),
                **extra,
            )
        assert result["mode"] == "live_service"
        assert result["status"] == "blocked"
        assert result["operations"][0]["reserved_usd"] > 0


def test_live_worker_reopens_parent_budget_and_profile(
    tmp_path: Path,
) -> None:
    request = ResumeRequest(
        root=str(tmp_path),
        run_id="abcdef123456",
        action="approve",
        policy_fingerprint=policy_fingerprint(ExecutionPolicy()),
    )
    with pytest.raises(ValueError, match="budget directory"):
        live_resume_model(request, Settings(), ExecutionPolicy())
    captured = []

    def factory(**kwargs: Any) -> Any:
        captured.append(kwargs)
        message = "unpriced"
        raise ReservationUnavailable(message)

    for aggregate in (True, False):
        with pytest.raises(ReservationUnavailable):
            live_resume_model(
                request.model_copy(
                    update={
                        "budget_directory": str(tmp_path / "abcdef123456"),
                        "budget_aggregate": aggregate,
                    }
                ),
                Settings(aws_profile="selected"),
                ExecutionPolicy(),
                model_type=factory,
            )
    assert captured[0]["settings"].aws_profile == "selected"
    assert captured[0]["run"].aggregate is not None
    assert captured[1]["run"].aggregate is None
    # The production default builds the real model. Session construction
    # does not call AWS; dispatch would.
    built = live_resume_model(
        request.model_copy(
            update={
                "budget_directory": str(tmp_path / "default-model"),
                "budget_aggregate": False,
            }
        ),
        Settings(),
        ExecutionPolicy(),
    )
    assert built.credential_source == "default_chain"
    assert built.model_id == TESTED_LIVE_BEDROCK_MODEL


def test_harness_quotes_known_components_without_dispatch(
    tmp_path: Path,
) -> None:
    run = budget(tmp_path)
    with billing.use_budget_run(run):
        data = harness_cost_preflight(
            Settings(harness_role_arn="arn:aws:iam::123456789012:role/demo"),
            run.policy,
        )
    assert data["pricing"] == "incomplete_bound"
    assert data["known_components_estimated_usd"] > 0
    assert not data["invocation_executed"]
    assert len(data["unbounded_costs"]) == 3
    assert run.command("agentcore-harness").ledger.snapshot().reserved_usd == 0


def test_harness_reports_missing_prices_and_role() -> None:
    def missing(**_kwargs: Any) -> Any:
        message = "No pricing snapshot"
        raise ReservationUnavailable(message)

    data = harness_cost_preflight(
        Settings(),
        ExecutionPolicy(),
        budget_opener=missing,
    )
    assert data["pricing"] == "missing"
    assert data["missing_configuration"] == "AWSAI_HARNESS_ROLE_ARN"


def test_live_cli_resume_carries_parent_budget_and_credential_selector(
    tmp_path: Path,
) -> None:
    run = budget(tmp_path / "budget")
    state_root = tmp_path / "state"
    settings = Settings(
        aws_profile="chosen", aws_creds_file_path=tmp_path / "unread.csv"
    )
    requests = []

    def resume_model(
        request: ResumeRequest, config: Settings, limits: ExecutionPolicy
    ) -> LiveAgentModel:
        assert config.aws_profile == "chosen"
        assert request.budget_directory == str(run.directory)
        assert config.aws_creds_file_path == settings.aws_creds_file_path
        return LiveAgentModel(
            demo="decision",
            settings=config,
            policy=limits,
            model_factory=decision_factory,
            exact_counter=Counter(),
            run=run,
        )

    class Worker:
        def run(self, request: ResumeRequest) -> ResumeReport:
            requests.append(request)
            return resume_from_disk(request, resume_model=resume_model)

    with billing.use_budget_run(run):
        pause = run_decision_demo(
            execution="live",
            settings=settings,
            checkpoint_root=state_root,
            model_factory=decision_factory,
            exact_counter=Counter(),
        )
        run_id = pause["data"]["run_id"]
        resumed = run_decision_demo(
            execution="live",
            settings=settings,
            checkpoint_root=state_root,
            resume=run_id,
            approve=True,
            approver="Pat",
            worker=Worker(),
        )
        replayed = run_decision_demo(
            execution="live",
            settings=settings,
            checkpoint_root=state_root,
            replay=run_id,
            worker=Worker(),
        )
    assert resumed["data"]["effects_after_resume"] == 1
    assert replayed["data"]["effects_after_replay"] == 1
    assert requests[0].budget_directory == str(run.directory)
    assert requests[0].aws_profile == "chosen"
    assert requests[1].budget_directory is None


def test_worker_forwards_sanitized_cost_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = ResumeRequest(
        root=str(tmp_path),
        run_id="abcdef123456",
        action="replay",
        policy_fingerprint=policy_fingerprint(ExecutionPolicy()),
    )
    report = ResumeReport(
        run_id=request.run_id,
        pause_pid=1,
        resume_pid=2,
        files_read=[],
        effects=0,
        approval_source="none",
        status="paused",
        operations=[],
    )
    monkeypatch.setattr(
        "awsai_demo.decision_demo.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            report.model_dump_json(),
            "Reserved USD 0.001; application-enforced limits",
        ),
    )
    assert SubprocessResumeWorker(ExecutionPolicy()).run(request) == report
    assert "Reserved USD 0.001" in capsys.readouterr().err


def test_offline_decision_preserves_local_filesystem_errors(
    tmp_path: Path,
) -> None:
    def disk_full(**_kwargs: Any) -> Any:
        message = "checkpoint filesystem is full"
        raise OSError(message)

    with pytest.raises(OSError, match="filesystem"):
        run_decision_demo(checkpoint_root=tmp_path, pause=disk_full)
