"""Private checkpoints and an idempotent business approval sink."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from awsai_demo.scenario import action_key, proposal_version

if TYPE_CHECKING:
    from pathlib import Path

    from awsai_demo.contracts import Execution
    from awsai_demo.policy import ExecutionPolicy
    from awsai_demo.scenario import PilotCost

ApprovalSource = Literal["none", "cli", "simulated"]


class Approval(BaseModel):
    """An explicit decision bound to an exact proposal version."""

    model_config = ConfigDict(extra="forbid")
    approved: bool
    proposal_version: str
    approver: str
    source: ApprovalSource


class StoredDecision(BaseModel):
    """Non-secret settings needed to reconstruct a paused agent."""

    model_config = ConfigDict(extra="forbid")
    run_id: str
    execution: Literal["offline", "emulator", "live"]
    model_id: str
    region: str
    policy_fingerprint: str
    proposal_version: str
    pause_pid: int
    interrupt_id: str
    approval: Approval | None = None
    completed: bool = False


class Effect(BaseModel):
    """Private audit data for the one committed business action."""

    model_config = ConfigDict(extra="forbid")
    action_key: str
    proposal_version: str
    approval_id: str
    approver: str
    approval_source: ApprovalSource
    approved_at: str


class ApprovalSink(Protocol):
    """Deduplicate the business action at its effect boundary."""

    def find(self, cost: PilotCost) -> Effect | None:
        """Find the original committed effect, if any."""

    def commit(self, cost: PilotCost, approval: Approval) -> Effect:
        """Atomically insert or return the already committed effect."""


def validate_approval(cost: PilotCost, approval: Approval) -> None:
    """Reject unnamed, stale or implicit approval before any write."""
    if approval.proposal_version != proposal_version(cost):
        message = "Approval is stale: the proposal version changed"
        raise ValueError(message)
    if not approval.approver.strip() or approval.source == "none":
        message = "A named, explicit approval decision is required"
        raise ValueError(message)


def effect_for(cost: PilotCost, approval: Approval) -> Effect:
    """Prepare an audit record after validating affirmative approval."""
    validate_approval(cost, approval)
    if not approval.approved:
        message = "Denied proposals cannot enter the business sink"
        raise ValueError(message)
    key = action_key(cost)
    audit = f"{key}:{approval.source}:{approval.approver}"
    return Effect(
        action_key=key,
        proposal_version=approval.proposal_version,
        approval_id=hashlib.sha256(audit.encode()).hexdigest(),
        approver=approval.approver,
        approval_source=approval.source,
        approved_at=datetime.now(UTC).isoformat(),
    )


class SQLiteApprovalSink:
    """Use a transaction and primary key for at-most-once effects."""

    def __init__(self, path: Path) -> None:
        """Create the ledger with a unique business-action key."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(path, timeout=30) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS effects "
                "(action_key TEXT PRIMARY KEY, audit TEXT NOT NULL)"
            )

    def find(self, cost: PilotCost) -> Effect | None:
        """Read the committed audit row without performing an effect."""
        with sqlite3.connect(self.path, timeout=30) as connection:
            row = connection.execute(
                "SELECT audit FROM effects WHERE action_key = ?",
                (action_key(cost),),
            ).fetchone()
        return None if row is None else Effect.model_validate_json(row[0])

    def commit(self, cost: PilotCost, approval: Approval) -> Effect:
        """Commit once and return the original approval audit row."""
        effect = effect_for(cost, approval)
        with sqlite3.connect(self.path, timeout=30) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO effects(action_key, audit) "
                "VALUES (?, ?)",
                (effect.action_key, effect.model_dump_json()),
            )
            row = connection.execute(
                "SELECT audit FROM effects WHERE action_key = ?",
                (effect.action_key,),
            ).fetchone()
        return Effect.model_validate_json(row[0])


def session_directory(root: Path, run_id: str) -> Path:
    """Confine all session paths to a validated run identifier."""
    if re.fullmatch(r"[0-9a-f]{12}", run_id) is None:
        message = "run-id must be exactly 12 lowercase hexadecimal characters"
        raise ValueError(message)
    return root / run_id


def write_decision(directory: Path, record: StoredDecision) -> None:
    """Flush and atomically replace the private checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "decision.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(record.model_dump_json(indent=2))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(directory / "decision.json")


def read_decision(directory: Path) -> StoredDecision:
    """Validate persisted settings before reconstructing an agent."""
    return StoredDecision.model_validate_json(
        (directory / "decision.json").read_text(encoding="utf-8")
    )


def policy_fingerprint(policy: ExecutionPolicy) -> str:
    """Bind resumption to the same immutable application limits."""
    serialized = json.dumps(asdict(policy), default=str, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def validate_resume(
    record: StoredDecision,
    *,
    execution: Execution,
    model_id: str,
    region: str,
    policy: ExecutionPolicy,
    cost: PilotCost,
) -> None:
    """Refuse changed settings or proposals before resumption."""
    expected = (
        execution,
        model_id,
        region,
        policy_fingerprint(policy),
        proposal_version(cost),
    )
    stored = (
        record.execution,
        record.model_id,
        record.region,
        record.policy_fingerprint,
        record.proposal_version,
    )
    if expected != stored:
        message = "Resume settings, policy or proposal version changed"
        raise ValueError(message)
