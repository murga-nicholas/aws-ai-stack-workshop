from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from awsai_demo.decision_state import (
    Approval,
    SQLiteApprovalSink,
    StoredDecision,
    effect_for,
    policy_fingerprint,
    read_decision,
    session_directory,
    validate_approval,
    validate_resume,
    write_decision,
)
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.scenario import price_pilot, proposal_version

if TYPE_CHECKING:
    from pathlib import Path


def approval() -> Approval:
    return Approval(
        approved=True,
        proposal_version=proposal_version(price_pilot()),
        approver="Pat",
        source="cli",
    )


def test_sink_deduplicates_business_action_not_approval(
    tmp_path: Path,
) -> None:
    sink = SQLiteApprovalSink(tmp_path / "private" / "sink.sqlite3")
    cost = price_pilot()
    assert sink.find(cost) is None
    first = sink.commit(cost, approval())
    second = sink.commit(
        cost, approval().model_copy(update={"approver": "Sam"})
    )
    assert first == second == sink.find(cost)
    assert second.approver == "Pat"
    assert first.proposal_version == proposal_version(cost)


def test_invalid_approval_never_reaches_the_sink(tmp_path: Path) -> None:
    sink = SQLiteApprovalSink(tmp_path / "sink.sqlite3")
    cost = price_pilot()
    for update in (
        {"proposal_version": "stale"},
        {"approver": " "},
        {"source": "none"},
        {"approved": False},
    ):
        with pytest.raises(ValueError):
            sink.commit(cost, approval().model_copy(update=update))
        assert sink.find(cost) is None
    with pytest.raises(ValueError, match="stale"):
        validate_approval(price_pilot(weeks=7), approval())
    assert effect_for(cost, approval()).approved_at.endswith("+00:00")


def test_checkpoint_roundtrip_and_resume_binding(tmp_path: Path) -> None:
    limits = ExecutionPolicy()
    cost = price_pilot()
    record = StoredDecision(
        run_id="abcdef123456",
        execution="offline",
        model_id="scripted-decision",
        region="us-east-1",
        policy_fingerprint=policy_fingerprint(limits),
        proposal_version=proposal_version(cost),
        pause_pid=10,
        interrupt_id="tool-approval",
    )
    directory = session_directory(tmp_path, record.run_id)
    write_decision(directory, record)
    assert read_decision(directory) == record
    arguments = {
        "execution": "offline",
        "model_id": "scripted-decision",
        "region": "us-east-1",
        "policy": limits,
        "cost": cost,
    }
    validate_resume(record, **arguments)
    for update in (
        {"execution": "emulator"},
        {"model_id": "changed"},
        {"region": "eu-west-1"},
        {"policy": replace(limits, max_model_calls=2)},
        {"cost": price_pilot(weeks=7)},
    ):
        with pytest.raises(ValueError, match="changed"):
            validate_resume(record, **(arguments | update))
    with pytest.raises(ValueError, match="run-id"):
        session_directory(tmp_path, "../escape")
