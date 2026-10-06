from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from awsai_demo.contracts import operation
from awsai_demo.decision_demo import (
    ResumeReport,
    ResumeRequest,
    SubprocessResumeWorker,
    _dependencies,
    _model_id,
    apply_approval,
    pause_decision,
    resume_from_disk,
    run_decision_demo,
)
from awsai_demo.decision_engine import DecisionModel
from awsai_demo.decision_state import (
    Approval,
    SQLiteApprovalSink,
    policy_fingerprint,
    read_decision,
    write_decision,
)
from awsai_demo.network import NetworkPolicy, environment_for_subprocess
from awsai_demo.offline import ScriptedModel, text_script
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.resume_worker import main as worker_main
from awsai_demo.runtime import Settings
from awsai_demo.scenario import price_pilot, proposal_version

if TYPE_CHECKING:
    from pathlib import Path


def paused(root: Path, *, run_id: str = "abcdef123456") -> ResumeRequest:
    limits = ExecutionPolicy()
    report = pause_decision(
        root=root,
        run_id=run_id,
        execution="offline",
        settings=Settings(),
        policy=limits,
        cost=price_pilot(),
    )
    assert report.status == "paused"
    assert report.effects == 0
    return ResumeRequest(
        root=str(root),
        run_id=run_id,
        action="approve",
        approver="Pat",
        approval_source="cli",
        policy_fingerprint=policy_fingerprint(limits),
    )


def test_approval_denial_and_passive_replay(tmp_path: Path) -> None:
    request = paused(tmp_path)
    before = resume_from_disk(request.model_copy(update={"action": "replay"}))
    assert before.status == "paused"
    assert before.approval_source == "none"
    denied = resume_from_disk(request.model_copy(update={"action": "deny"}))
    assert denied.status == "blocked"
    assert denied.effects == 0
    assert len(denied.files_read) >= 4
    replayed = resume_from_disk(
        request.model_copy(update={"action": "replay"})
    )
    assert replayed.effects == 0
    assert replayed.status == "blocked"
    assert not any(
        op["operation"] == "Model.stream" for op in replayed.operations
    )

    new = paused(tmp_path, run_id="abcdef123457")
    approved = resume_from_disk(new)
    assert approved.effects == 1
    repeated = resume_from_disk(new.model_copy(update={"approver": "Sam"}))
    assert repeated.effects == 1
    assert (
        SQLiteApprovalSink(tmp_path / "ledger.sqlite3")
        .find(price_pilot())
        .approver
        == "Pat"
    )


def test_worker_reads_session_in_a_separate_process(tmp_path: Path) -> None:
    request = paused(tmp_path)
    worker = SubprocessResumeWorker(ExecutionPolicy())
    report = worker.run(request)
    assert report.pause_pid == os.getpid()
    assert report.resume_pid != report.pause_pid
    assert report.effects == 1
    assert len(report.files_read) >= 4
    assert any("message_" in name for name in report.files_read)
    assert (
        worker.run(request.model_copy(update={"action": "replay"})).effects
        == 1
    )


def test_concurrent_process_resumes_commit_one_effect(tmp_path: Path) -> None:
    request = paused(tmp_path)
    worker = SubprocessResumeWorker(ExecutionPolicy())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                worker.run, request.model_copy(update={"approver": name})
            )
            for name in ("Pat", "Sam")
        ]
        reports = [future.result() for future in futures]
    assert [report.effects for report in reports] == [1, 1]
    assert reports[0].resume_pid != reports[1].resume_pid
    sink = SQLiteApprovalSink(tmp_path / "ledger.sqlite3")
    assert sink.find(price_pilot()).approver in {"Pat", "Sam"}


def test_crash_after_effect_before_checkpoint_then_replay(
    tmp_path: Path,
) -> None:
    request = paused(tmp_path)
    code = "\n".join(
        [
            "import os, sys, coverage",
            "from awsai_demo.decision_demo import ResumeRequest",
            "from awsai_demo.decision_demo import resume_from_disk",
            "def crash():",
            "    current = coverage.Coverage.current()",
            "    if current is not None: current.save()",
            "    os._exit(86)",
            "request = ResumeRequest.model_validate_json(sys.argv[1])",
            "resume_from_disk(request, after_effect=crash)",
        ]
    )
    env = environment_for_subprocess(
        NetworkPolicy(), bootstrap_dir=tmp_path / "crash-bootstrap"
    )
    completed = subprocess.run(  # noqa: S603 - fixed test-only Python code
        [sys.executable, "-c", code, request.model_dump_json()],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 86, completed.stderr
    record = read_decision(tmp_path / request.run_id)
    assert record.completed is False
    assert record.approval.approved is True
    recovered = SubprocessResumeWorker(ExecutionPolicy()).run(
        request.model_copy(update={"action": "replay"})
    )
    assert recovered.effects == 1
    assert recovered.status == "ok"
    assert not any(
        op["operation"] == "Model.stream" for op in recovered.operations
    )
    with pytest.raises(ValueError, match="intent cannot be replaced"):
        resume_from_disk(request.model_copy(update={"action": "deny"}))
    assert read_decision(tmp_path / request.run_id).approval.approved is True
    completed_recovery = resume_from_disk(request)
    assert completed_recovery.effects == 1
    assert completed_recovery.status == "ok"
    assert read_decision(tmp_path / request.run_id).completed is True


def test_stale_settings_policy_or_proposal_are_refused(tmp_path: Path) -> None:
    request = paused(tmp_path)
    with pytest.raises(ValueError, match="proposal version changed"):
        resume_from_disk(request, cost=price_pilot(weeks=7))
    with pytest.raises(ValueError, match="Worker policy differs"):
        resume_from_disk(
            request, policy=replace(ExecutionPolicy(), max_model_calls=2)
        )
    with pytest.raises(ValueError, match="settings"):
        resume_from_disk(request.model_copy(update={"region": "eu-west-1"}))
    record = read_decision(tmp_path / request.run_id)
    record.run_id = "aaaaaaaaaaaa"
    write_decision(tmp_path / request.run_id, record)
    with pytest.raises(ValueError, match="run id"):
        resume_from_disk(request)
    with pytest.raises(RuntimeError, match="run id"):
        SubprocessResumeWorker(ExecutionPolicy()).run(request)
    assert (
        SQLiteApprovalSink(tmp_path / "ledger.sqlite3").find(price_pilot())
        is None
    )


def test_tool_rejects_denial_and_over_budget_without_effect(
    tmp_path: Path,
) -> None:
    sink = SQLiteApprovalSink(tmp_path / "sink.sqlite3")
    cost = price_pilot(budget=100)
    approval = Approval(
        approved=True,
        proposal_version=proposal_version(cost),
        approver="Pat",
        source="cli",
    )
    assert "over budget" in apply_approval(cost, approval, sink, None)
    assert sink.find(cost) is None
    with pytest.raises(ValueError):
        apply_approval(cost, {}, sink, None)


def test_worker_sanitizes_unknown_input_and_returns_json(
    tmp_path: Path,
) -> None:
    out, err = io.StringIO(), io.StringIO()
    assert worker_main(["--secret", "AKIAABCDEFGHIJKLMNOP"], stderr=err) == 1
    assert "AKIAABCDEFGHIJKLMNOP" not in err.getvalue()
    request = paused(tmp_path)
    assert (
        worker_main(
            ["--request", request.model_dump_json()], stdout=out, stderr=err
        )
        == 0
    )
    assert json.loads(out.getvalue())["effects"] == 1


def test_emulator_dependencies_use_dummy_clients_without_dispatch(
    tmp_path: Path,
) -> None:
    credential = "local-fixture"
    settings = Settings(localstack_auth_token=credential)
    sink, model = _dependencies(
        "emulator", settings, ExecutionPolicy(), tmp_path
    )
    assert model is not None
    assert sink is not None
    replacement = SQLiteApprovalSink(tmp_path / "fake.sqlite3")
    scripted = DecisionModel(ExecutionPolicy())
    assert _dependencies(
        "emulator",
        settings,
        ExecutionPolicy(),
        tmp_path,
        sink=replacement,
        model=scripted,
    ) == (replacement, scripted)
    assert _model_id("emulator", Settings(model="chosen")) == "chosen"
    assert _model_id("emulator", Settings()) == "qwen2.5:0.5b"


def test_model_that_skips_interrupt_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="required approval interrupt"):
        pause_decision(
            root=tmp_path,
            run_id="abcdef123456",
            execution="emulator",
            settings=Settings(),
            policy=ExecutionPolicy(),
            cost=price_pilot(),
            sink=SQLiteApprovalSink(tmp_path / "fake.sqlite3"),
            model=ScriptedModel(text_script("fake emulator bypass attempt")),
        )


def test_demo_defaults_pause_and_simulation_is_explicit(
    tmp_path: Path,
) -> None:
    normal = run_decision_demo(checkpoint_root=tmp_path)
    assert normal["status"] == "paused"
    assert normal["data"]["approval_source"] == "none"
    assert normal["data"]["effects_after_resume"] == 0
    approved = run_decision_demo(
        checkpoint_root=tmp_path,
        simulate_approval=True,
        settings=Settings(),
        policy=ExecutionPolicy(),
    )
    data = approved["data"]
    assert approved["status"] == "ok"
    assert data["approval_source"] == "simulated"
    assert data["pause_pid"] != data["resume_pid"]
    assert data["effects_after_resume"] == data["effects_after_replay"] == 1
    assert len(data["responsibility_matrix"]) == 9
    assert len(data["lanes"]) == 3
    replayed = run_decision_demo(
        checkpoint_root=tmp_path, replay=data["run_id"]
    )
    assert replayed["data"]["effects_after_replay"] == 1
    manual = run_decision_demo(
        checkpoint_root=tmp_path,
        resume=normal["data"]["run_id"],
        approve=True,
        approver="Pat",
        worker=SubprocessResumeWorker(ExecutionPolicy()),
    )
    assert manual["status"] == "ok"


def test_demo_rejects_implicit_or_wrong_lane_approval() -> None:
    from awsai_demo.policy import ReservationUnavailable

    def missing_prices(**_kwargs: object) -> object:
        message = "The injected snapshot is unavailable"
        raise ReservationUnavailable(message)

    with pytest.raises(ValueError, match="offline only"):
        run_decision_demo(execution="emulator", simulate_approval=True)
    with pytest.raises(ValueError, match="provider"):
        run_decision_demo(settings=Settings(provider="openai"))
    for kwargs in (
        {"resume": "abcdef123456"},
        {"resume": "abcdef123456", "approve": True, "approver": " "},
        {
            "resume": "abcdef123456",
            "approve": True,
            "deny": True,
            "approver": "Pat",
        },
    ):
        with pytest.raises(ValueError, match="named approver"):
            run_decision_demo(**kwargs)
    refused = run_decision_demo(
        execution="live",
        budget_opener=missing_prices,
    )
    assert refused["mode"] == "not_run"
    assert refused["error"]["code"] == "budget_exceeded"
    missing = run_decision_demo(execution="emulator")
    assert missing["error"]["code"] == "missing_configuration"


def test_emulator_resume_uses_injected_worker_without_fixture_fallback(
    tmp_path: Path,
) -> None:
    class Worker:
        def run(self, request: ResumeRequest) -> ResumeReport:
            assert request.execution == "emulator"
            return ResumeReport(
                run_id=request.run_id,
                pause_pid=1,
                resume_pid=2,
                files_read=["decision.json"],
                effects=1,
                approval_source="cli",
                status="ok",
                operations=[
                    operation(
                        service="dynamodb",
                        operation="PutItem",
                        mode="local_emulator",
                        execution_target="emulator",
                        transport="loopback",
                        endpoint_url="http://localhost:4566",
                        effect="write",
                    )
                ],
            )

    credential = "local-fixture"
    output = run_decision_demo(
        execution="emulator",
        settings=Settings(localstack_auth_token=credential),
        checkpoint_root=tmp_path,
        resume="abcdef123456",
        approve=True,
        approver="Pat",
        worker=Worker(),
    )
    assert output["children"][0]["mode"] == "not_run"
    assert output["children"][1]["mode"] == "not_run"
    assert output["children"][2]["mode"] == "local_emulator"
    assert output["data"]["approval_source"] == "cli"
