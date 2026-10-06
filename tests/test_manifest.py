from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from awsai_demo.manifest import (
    CleanupResult,
    ManifestEntry,
    ManifestError,
    ManifestStore,
    ManifestValidationError,
    OwnershipEvidence,
    _FileLock,
    cleanup_manifest,
    mark_created,
    mark_unknown,
    record_intent,
    verify_ownership,
)

if TYPE_CHECKING:
    from pathlib import Path

RUN_ID = "abcdef123456"


class Clock:
    def __init__(self) -> None:
        self.now = 1.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


class Port:
    def __init__(
        self,
        *,
        recovered: ManifestEntry | None = None,
        evidence: OwnershipEvidence | None = None,
        results: list[CleanupResult] | None = None,
    ) -> None:
        self.recovered = recovered
        self.evidence = evidence
        self.results = [] if results is None else results
        self.deleted: list[str] = []

    def reconcile(self, entry: ManifestEntry) -> ManifestEntry | None:
        assert entry.entry_id
        return self.recovered

    def ownership_evidence(
        self,
        entry: ManifestEntry,
    ) -> OwnershipEvidence | None:
        assert entry.entry_id
        return self.evidence

    def delete(self, entry: ManifestEntry) -> CleanupResult:
        self.deleted.append(entry.entry_id)
        if self.results:
            return self.results.pop(0)
        return CleanupResult("deleted", "deleted")


def make_store(tmp_path: Path) -> ManifestStore:
    return ManifestStore(tmp_path / "run" / "manifest.json", clock=Clock())


def add_guardrail(
    store: ManifestStore,
    *,
    client_value: str | None = None,
) -> ManifestEntry:
    client = "client-default" if client_value is None else client_value
    return record_intent(
        store,
        run_id=RUN_ID,
        account_fingerprint="acct",
        region="us-east-1",
        service="bedrock",
        operation="CreateGuardrail",
        intended_name=f"awsai-{RUN_ID}-guard",
        client_token=client,
        tags={"run-id": RUN_ID, "project": "awsai-workshop"},
        naming_scheme="guardrail",
    )


def test_intent_is_durable_idempotent_and_updatable(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    entry = add_guardrail(store)
    assert entry.status == "intent"
    assert store.list_entries() == [entry]

    same = add_guardrail(store)
    assert same == entry
    assert len(store.list_entries()) == 1

    created = store.mark_created(
        entry.entry_id,
        exact_id="guardrail-id",
        arn="arn:aws:bedrock:::guardrail/guardrail-id",
    )
    assert created.status == "created"
    assert created.exact_id == "guardrail-id"

    unknown = mark_unknown(store, entry.entry_id, message="timeout")
    assert unknown.status == "unknown"
    assert unknown.last_error == "timeout"

    deleted = store.mark_deleted(entry.entry_id, message="gone")
    assert deleted.status == "deleted"
    assert deleted.last_error == "gone"

    unresolved = store.mark_unresolved(entry.entry_id, message="unsafe")
    assert unresolved.status == "unresolved"

    failed = store.mark_delete_failed(entry.entry_id, message="denied")
    assert failed.status == "delete_failed"
    assert failed.delete_attempts == 1

    filtered = store.list_entries(statuses={"delete_failed"})
    assert filtered == [failed]


def test_manifest_validation_and_conflicts(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    entry = add_guardrail(store)
    client = "client-one"
    with pytest.raises(ManifestValidationError):
        ManifestEntry(
            entry_id="bad",
            run_id="",
            account_fingerprint="acct",
            region="us-east-1",
            service="svc",
            operation="op",
            intended_name="awsai-name",
            client_token=client,
            tags={},
            naming_scheme="generic",
        )
    with pytest.raises(ManifestValidationError):
        ManifestEntry(
            entry_id="child",
            run_id=RUN_ID,
            account_fingerprint="acct",
            region="us-east-1",
            service="svc",
            operation="op",
            intended_name="awsai-child",
            client_token=client,
            tags={},
            naming_scheme="generic",
            ownership="manifest_child",
        )
    with pytest.raises(ManifestValidationError):
        replace(entry, delete_attempts=-1)
    with pytest.raises(ManifestValidationError):
        mark_created(store, entry.entry_id, exact_id="")
    with pytest.raises(ManifestValidationError):
        store.mark_unknown("missing")
    with pytest.raises(ManifestValidationError):
        conflict_client = "client-default"
        store.record_intent(
            run_id=RUN_ID,
            account_fingerprint="acct",
            region="us-east-1",
            service="bedrock",
            operation="CreateGuardrail",
            intended_name=f"awsai-{RUN_ID}-guard",
            client_token=conflict_client,
            tags={"run-id": "different"},
            naming_scheme="guardrail",
        )

    bad_store = ManifestStore(tmp_path / "bad.json")
    bad_store._path.write_text(
        '{"version": 999, "entries": []}',
        encoding="utf-8",
    )
    with pytest.raises(ManifestValidationError):
        bad_store.list_entries()


def test_upsert_reconciled_requires_existing_entry(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    entry = add_guardrail(store)
    with pytest.raises(ManifestValidationError):
        store.upsert_reconciled(replace(entry, entry_id="unknown"))


def test_ownership_verification_modes() -> None:
    client = "client-one"
    other_client = "client-two"
    tagged = ManifestEntry(
        entry_id="tagged",
        run_id=RUN_ID,
        account_fingerprint="acct",
        region="us-east-1",
        service="bedrock",
        operation="CreateGuardrail",
        intended_name=f"awsai-{RUN_ID}-guard",
        client_token=client,
        tags={"run-id": RUN_ID},
        naming_scheme="guardrail",
        status="created",
        exact_id="gid",
    )
    evidence = OwnershipEvidence(
        exact_id="gid",
        account_fingerprint="acct",
        region="us-east-1",
        tags={"run-id": RUN_ID},
    )
    assert verify_ownership(tagged, evidence)
    assert not verify_ownership(
        tagged,
        replace(evidence, account_fingerprint="other"),
    )
    assert not verify_ownership(tagged, replace(evidence, region="eu-west-1"))
    assert not verify_ownership(tagged, replace(evidence, exact_id="other"))
    assert not verify_ownership(
        tagged,
        replace(evidence, tags={"run-id": "other"}),
    )
    assert not verify_ownership(
        replace(tagged, intended_name="not-owned"),
        evidence,
    )

    generic = replace(
        tagged,
        intended_name=f"awsai-{RUN_ID}-idx",
        naming_scheme="vector_index",
    )
    assert verify_ownership(generic, evidence)

    child = ManifestEntry(
        entry_id="child",
        run_id=RUN_ID,
        account_fingerprint="acct",
        region="us-east-1",
        service="s3vectors",
        operation="CreateIndex",
        intended_name=f"awsai-{RUN_ID}-idx",
        client_token=client,
        tags={},
        naming_scheme="vector_index",
        ownership="manifest_child",
        parent_entry_id="parent",
        child_id="idx-id",
    )
    assert verify_ownership(
        child,
        OwnershipEvidence(
            exact_id="idx-id",
            account_fingerprint="acct",
            region="us-east-1",
            parent_verified=True,
        ),
    )
    assert not verify_ownership(
        child,
        OwnershipEvidence(
            exact_id="idx-id",
            account_fingerprint="acct",
            region="us-east-1",
            parent_verified=False,
        ),
    )

    session = ManifestEntry(
        entry_id="session",
        run_id=RUN_ID,
        account_fingerprint="acct",
        region="us-east-1",
        service="codeinterpreter",
        operation="StartCodeInterpreterSession",
        intended_name=f"awsai-{RUN_ID}-session",
        client_token=client,
        tags={},
        naming_scheme="generic",
        ownership="session",
        status="created",
        exact_id="session-id",
    )
    assert verify_ownership(
        session,
        OwnershipEvidence(
            exact_id="session-id",
            account_fingerprint="acct",
            region="us-east-1",
            name=f"awsai-{RUN_ID}-session",
            client_token=client,
        ),
    )
    assert not verify_ownership(
        session,
        OwnershipEvidence(
            exact_id="session-id",
            account_fingerprint="acct",
            region="us-east-1",
            name=f"awsai-{RUN_ID}-session",
            client_token=other_client,
        ),
    )


def test_cleanup_dry_run_reverse_order_and_deleted_skip(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    parent_client = "client-parent"
    child_client = "client-child"
    parent = add_guardrail(store, client_value=parent_client)
    child = store.record_intent(
        run_id=RUN_ID,
        account_fingerprint="acct",
        region="us-east-1",
        service="bedrock",
        operation="CreateGuardrailVersion",
        intended_name=f"awsai-{RUN_ID}-guard-v1",
        client_token=child_client,
        tags={},
        naming_scheme="generic",
        ownership="manifest_child",
        parent_entry_id=parent.entry_id,
        child_id="version",
    )
    report = cleanup_manifest(store, {}, execute=False)
    assert [action.entry_id for action in report.actions] == [
        child.entry_id,
        parent.entry_id,
    ]
    assert [action.outcome for action in report.actions] == [
        "would_delete",
        "would_delete",
    ]

    store.mark_deleted(child.entry_id)
    second = cleanup_manifest(store, {}, execute=False)
    assert second.actions[0].outcome == "skipped"


def test_cleanup_missing_adapter_and_intent_not_found(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    missing = add_guardrail(store)
    report = cleanup_manifest(store, {}, execute=True)
    assert report.cleanup_incomplete
    assert report.actions[0].outcome == "unresolved"
    assert store.list_entries()[0].status == "unresolved"

    other_store = make_store(tmp_path / "other")
    intent = add_guardrail(other_store, client_value="client-other")
    port = Port(recovered=None)
    clean = cleanup_manifest(other_store, {"bedrock": port}, execute=True)
    assert not clean.cleanup_incomplete
    assert clean.actions[0].outcome == "not_found"
    assert other_store.list_entries()[0].status == "deleted"
    assert missing.entry_id != intent.entry_id


def test_cleanup_unresolved_reconcile_paths(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    unknown = add_guardrail(store, client_value="client-unknown")
    store.mark_unknown(unknown.entry_id)
    unknown_report = cleanup_manifest(
        store,
        {"bedrock": Port(recovered=None)},
        execute=True,
    )
    assert unknown_report.cleanup_incomplete
    assert unknown_report.actions[0].outcome == "unresolved"

    created_store = make_store(tmp_path / "created")
    created = add_guardrail(created_store, client_value="client-created")
    created_store.upsert_reconciled(replace(created, status="created"))
    created_report = cleanup_manifest(
        created_store,
        {"bedrock": Port(recovered=None)},
        execute=True,
    )
    assert created_report.cleanup_incomplete
    assert created_report.actions[0].message == "not reconciled"

    recovered_store = make_store(tmp_path / "recovered")
    base = add_guardrail(recovered_store, client_value="client-recovered")
    recovered_store.upsert_reconciled(replace(base, status="created"))
    recovered = replace(base, status="created", exact_id="gid")
    evidence = OwnershipEvidence(
        exact_id="gid",
        account_fingerprint="acct",
        region="us-east-1",
        tags={"run-id": RUN_ID},
    )
    recovered_report = cleanup_manifest(
        recovered_store,
        {
            "bedrock": Port(
                recovered=recovered,
                evidence=evidence,
            ),
        },
        execute=True,
    )
    assert not recovered_report.cleanup_incomplete
    assert recovered_report.actions[0].outcome == "deleted"


def test_cleanup_reconciles_verifies_and_retries(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    entry = add_guardrail(store)
    store.mark_unknown(entry.entry_id, message="timeout")
    recovered = replace(
        entry,
        status="created",
        exact_id="gid",
        updated_at=99.0,
    )
    evidence = OwnershipEvidence(
        exact_id="gid",
        account_fingerprint="acct",
        region="us-east-1",
        tags={"run-id": RUN_ID},
    )
    port = Port(
        recovered=recovered,
        evidence=evidence,
        results=[
            CleanupResult("failed", "retry"),
            CleanupResult("deleted", "done"),
        ],
    )
    report = cleanup_manifest(
        store,
        {"bedrock": port},
        execute=True,
        max_attempts=2,
    )
    assert not report.cleanup_incomplete
    assert report.actions[0].outcome == "deleted"
    final = store.list_entries()[0]
    assert final.status == "deleted"
    assert final.delete_attempts == 1


def test_cleanup_refuses_unowned_and_reports_final_failure(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    entry = add_guardrail(store)
    store.mark_created(entry.entry_id, exact_id="gid")
    report = cleanup_manifest(
        store,
        {"bedrock": Port(evidence=None)},
        execute=True,
    )
    assert report.cleanup_incomplete
    assert report.actions[0].message == "ownership not established"
    assert store.list_entries()[0].status == "unresolved"

    fail_store = make_store(tmp_path / "fail")
    failed = add_guardrail(fail_store)
    fail_store.mark_created(failed.entry_id, exact_id="gid")
    evidence = OwnershipEvidence(
        exact_id="gid",
        account_fingerprint="acct",
        region="us-east-1",
        tags={"run-id": RUN_ID},
    )
    fail_report = cleanup_manifest(
        fail_store,
        {
            "bedrock": Port(
                evidence=evidence,
                results=[CleanupResult("failed", "nope")],
            ),
        },
        execute=True,
        max_attempts=1,
    )
    assert fail_report.cleanup_incomplete
    assert fail_report.actions[0].outcome == "failed"
    assert fail_store.list_entries()[0].status == "delete_failed"


def test_cleanup_handles_not_found_and_invalid_attempts(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    entry = add_guardrail(store)
    store.mark_created(entry.entry_id, exact_id="gid")
    evidence = OwnershipEvidence(
        exact_id="gid",
        account_fingerprint="acct",
        region="us-east-1",
        tags={"run-id": RUN_ID},
    )
    report = cleanup_manifest(
        store,
        {
            "bedrock": Port(
                evidence=evidence,
                results=[CleanupResult("not_found", "gone")],
            ),
        },
        execute=True,
    )
    assert report.actions[0].outcome == "not_found"
    with pytest.raises(ManifestValidationError):
        cleanup_manifest(store, {}, max_attempts=0)


def test_manifest_lock_timeout(tmp_path: Path) -> None:
    lock_path = tmp_path / "manifest.lock"
    _FileLock(lock_path).__exit__(None, None, None)
    with (
        _FileLock(lock_path),
        pytest.raises(ManifestError),
        _FileLock(lock_path, timeout_seconds=0, poll_seconds=0),
    ):
        pass
