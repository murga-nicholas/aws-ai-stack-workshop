"""Durable resource intent manifest and cleanup orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, Self, cast

from filelock import FileLock, Timeout

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping
    from pathlib import Path
    from types import TracebackType

ManifestStatus = Literal[
    "intent",
    "created",
    "unknown",
    "deleted",
    "delete_failed",
    "unresolved",
]
OwnershipMode = Literal["tagged", "manifest_child", "session"]
CleanupOutcome = Literal[
    "would_delete",
    "deleted",
    "not_found",
    "unresolved",
    "failed",
    "skipped",
]
DeleteOutcome = Literal["deleted", "not_found", "failed"]

_MANIFEST_VERSION: Final = 1
_NAME_SCHEMES: Final[dict[str, re.Pattern[str]]] = {
    "harness": re.compile(r"^awsai_[0-9a-f]{12}_h$"),
    "memory": re.compile(r"^awsai_[0-9a-f]{12}_mem$"),
    "guardrail": re.compile(r"^awsai-[0-9a-f]{12}-guard$"),
    "vector_bucket": re.compile(r"^awsai-[0-9a-f]{12}-vec$"),
}


class ManifestError(Exception):
    """Base class for manifest failures."""


class ManifestValidationError(ManifestError):
    """Raised when a manifest entry is invalid."""


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One durable resource intent or cleanup record."""

    entry_id: str
    run_id: str
    account_fingerprint: str
    region: str
    service: str
    operation: str
    intended_name: str
    client_token: str
    tags: Mapping[str, str]
    naming_scheme: str
    ownership: OwnershipMode = "tagged"
    status: ManifestStatus = "intent"
    exact_id: str | None = None
    arn: str | None = None
    parent_entry_id: str | None = None
    child_id: str | None = None
    last_error: str | None = None
    delete_attempts: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        """Validate immutable entry fields and freeze tag contents."""
        required = {
            "entry_id": self.entry_id,
            "run_id": self.run_id,
            "account_fingerprint": self.account_fingerprint,
            "region": self.region,
            "service": self.service,
            "operation": self.operation,
            "intended_name": self.intended_name,
            "client_token": self.client_token,
            "naming_scheme": self.naming_scheme,
        }
        for name, value in required.items():
            if not value:
                message = f"{name} must not be empty"
                raise ManifestValidationError(message)
        if self.ownership == "manifest_child" and self.parent_entry_id is None:
            message = "manifest_child entries need parent_entry_id"
            raise ManifestValidationError(message)
        if self.delete_attempts < 0:
            message = "delete_attempts must not be negative"
            raise ManifestValidationError(message)
        object.__setattr__(self, "tags", dict(self.tags))

    def to_json(self) -> dict[str, Any]:
        """Serialize the entry for manifest storage."""
        return {
            "entry_id": self.entry_id,
            "run_id": self.run_id,
            "account_fingerprint": self.account_fingerprint,
            "region": self.region,
            "service": self.service,
            "operation": self.operation,
            "intended_name": self.intended_name,
            "client_token": self.client_token,
            "tags": dict(self.tags),
            "naming_scheme": self.naming_scheme,
            "ownership": self.ownership,
            "status": self.status,
            "exact_id": self.exact_id,
            "arn": self.arn,
            "parent_entry_id": self.parent_entry_id,
            "child_id": self.child_id,
            "last_error": self.last_error,
            "delete_attempts": self.delete_attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Self:
        """Deserialize one manifest entry."""
        return cls(
            entry_id=str(raw["entry_id"]),
            run_id=str(raw["run_id"]),
            account_fingerprint=str(raw["account_fingerprint"]),
            region=str(raw["region"]),
            service=str(raw["service"]),
            operation=str(raw["operation"]),
            intended_name=str(raw["intended_name"]),
            client_token=str(raw["client_token"]),
            tags=cast("Mapping[str, str]", raw["tags"]),
            naming_scheme=str(raw["naming_scheme"]),
            ownership=cast("OwnershipMode", raw["ownership"]),
            status=cast("ManifestStatus", raw["status"]),
            exact_id=_optional_str(raw.get("exact_id")),
            arn=_optional_str(raw.get("arn")),
            parent_entry_id=_optional_str(raw.get("parent_entry_id")),
            child_id=_optional_str(raw.get("child_id")),
            last_error=_optional_str(raw.get("last_error")),
            delete_attempts=int(raw["delete_attempts"]),
            created_at=float(raw["created_at"]),
            updated_at=float(raw["updated_at"]),
        )


@dataclass(frozen=True, slots=True)
class OwnershipEvidence:
    """Evidence supplied by an injected adapter before cleanup."""

    exact_id: str
    account_fingerprint: str
    region: str
    tags: Mapping[str, str] = field(default_factory=dict)
    name: str | None = None
    client_token: str | None = None
    parent_verified: bool = False


@dataclass(frozen=True, slots=True)
class CleanupResult:
    """Result returned by an injected cleanup adapter."""

    outcome: DeleteOutcome
    message: str = ""


@dataclass(frozen=True, slots=True)
class CleanupAction:
    """One cleanup decision reported to the CLI."""

    entry_id: str
    outcome: CleanupOutcome
    message: str


@dataclass(frozen=True, slots=True)
class CleanupReport:
    """Cleanup summary with incomplete status made explicit."""

    dry_run: bool
    actions: tuple[CleanupAction, ...]
    cleanup_incomplete: bool


class CleanupPort(Protocol):
    """Injected service adapter for reconciliation and deletion."""

    def reconcile(self, entry: ManifestEntry) -> ManifestEntry | None:
        """Recover exact ids for an intent or unknown entry."""

    def ownership_evidence(
        self,
        entry: ManifestEntry,
    ) -> OwnershipEvidence | None:
        """Return ownership evidence for a manifest entry."""

    def delete(self, entry: ManifestEntry) -> CleanupResult:
        """Delete exactly the resource represented by the entry."""


class ManifestStore:
    """Locked JSON manifest for resource intents and outcomes."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Create a manifest wrapper without writing to disk."""
        self._path = path
        self._clock = time.time if clock is None else clock
        self._lock = _FileLock(path.with_name(f"{path.name}.lock"))

    def record_intent(
        self,
        *,
        run_id: str,
        account_fingerprint: str,
        region: str,
        service: str,
        operation: str,
        intended_name: str,
        client_token: str,
        tags: Mapping[str, str],
        naming_scheme: str,
        ownership: OwnershipMode = "tagged",
        parent_entry_id: str | None = None,
        child_id: str | None = None,
    ) -> ManifestEntry:
        """Append an intent before the write or session call is sent."""
        now = float(self._clock())
        entry = ManifestEntry(
            entry_id=_entry_id(
                run_id=run_id,
                service=service,
                operation=operation,
                intended_name=intended_name,
                client_token=client_token,
                parent_entry_id=parent_entry_id,
            ),
            run_id=run_id,
            account_fingerprint=account_fingerprint,
            region=region,
            service=service,
            operation=operation,
            intended_name=intended_name,
            client_token=client_token,
            tags=tags,
            naming_scheme=naming_scheme,
            ownership=ownership,
            parent_entry_id=parent_entry_id,
            child_id=child_id,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            data = self._load_data()
            entries = _entries_from_data(data)
            existing = _find_entry(entries, entry.entry_id)
            if existing is not None:
                _assert_same_intent(existing, entry)
                return existing
            entries.append(entry)
            self._save_entries(entries)
        return entry

    def mark_created(
        self,
        entry_id: str,
        *,
        exact_id: str,
        arn: str | None = None,
        child_id: str | None = None,
    ) -> ManifestEntry:
        """Update an intent after a successful create response."""
        if not exact_id:
            message = "exact_id must not be empty"
            raise ManifestValidationError(message)

        def updater(entry: ManifestEntry) -> ManifestEntry:
            return replace(
                entry,
                status="created",
                exact_id=exact_id,
                arn=arn,
                child_id=child_id if child_id is not None else entry.child_id,
                last_error=None,
                updated_at=float(self._clock()),
            )

        return self._update(entry_id, updater)

    def mark_unknown(
        self,
        entry_id: str,
        *,
        message: str = "ambiguous create failure",
    ) -> ManifestEntry:
        """Mark an entry for later reconciliation."""

        def updater(entry: ManifestEntry) -> ManifestEntry:
            return replace(
                entry,
                status="unknown",
                last_error=message,
                updated_at=float(self._clock()),
            )

        return self._update(entry_id, updater)

    def mark_deleted(
        self,
        entry_id: str,
        *,
        message: str = "",
    ) -> ManifestEntry:
        """Mark an entry as cleaned up or confirmed absent."""

        def updater(entry: ManifestEntry) -> ManifestEntry:
            return replace(
                entry,
                status="deleted",
                last_error=message or None,
                updated_at=float(self._clock()),
            )

        return self._update(entry_id, updater)

    def mark_unresolved(self, entry_id: str, *, message: str) -> ManifestEntry:
        """Mark an entry unresolved without claiming deletion."""

        def updater(entry: ManifestEntry) -> ManifestEntry:
            return replace(
                entry,
                status="unresolved",
                last_error=message,
                updated_at=float(self._clock()),
            )

        return self._update(entry_id, updater)

    def mark_delete_failed(
        self,
        entry_id: str,
        *,
        message: str,
    ) -> ManifestEntry:
        """Record a failed deletion attempt."""

        def updater(entry: ManifestEntry) -> ManifestEntry:
            return replace(
                entry,
                status="delete_failed",
                last_error=message,
                delete_attempts=entry.delete_attempts + 1,
                updated_at=float(self._clock()),
            )

        return self._update(entry_id, updater)

    def upsert_reconciled(self, entry: ManifestEntry) -> ManifestEntry:
        """Persist a reconciled entry returned by an adapter."""
        with self._lock:
            entries = _entries_from_data(self._load_data())
            replaced = False
            updated = replace(entry, updated_at=float(self._clock()))
            next_entries: list[ManifestEntry] = []
            for current in entries:
                if current.entry_id == entry.entry_id:
                    next_entries.append(updated)
                    replaced = True
                else:
                    next_entries.append(current)
            if not replaced:
                message = "cannot reconcile unknown manifest entry"
                raise ManifestValidationError(message)
            self._save_entries(next_entries)
        return updated

    def list_entries(
        self,
        *,
        statuses: Collection[ManifestStatus] | None = None,
    ) -> list[ManifestEntry]:
        """Return manifest entries, optionally filtered by status."""
        with self._lock:
            entries = _entries_from_data(self._load_data())
        if statuses is None:
            return entries
        wanted = set(statuses)
        return [entry for entry in entries if entry.status in wanted]

    def _update(
        self,
        entry_id: str,
        updater: Callable[[ManifestEntry], ManifestEntry],
    ) -> ManifestEntry:
        with self._lock:
            entries = _entries_from_data(self._load_data())
            next_entries: list[ManifestEntry] = []
            updated: ManifestEntry | None = None
            for entry in entries:
                if entry.entry_id == entry_id:
                    updated = updater(entry)
                    next_entries.append(updated)
                else:
                    next_entries.append(entry)
            if updated is None:
                message = "manifest entry not found"
                raise ManifestValidationError(message)
            self._save_entries(next_entries)
            return updated

    def _load_data(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"version": _MANIFEST_VERSION, "entries": []}
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        data = cast("dict[str, Any]", raw)
        if int(data.get("version", 0)) != _MANIFEST_VERSION:
            message = "unsupported manifest version"
            raise ManifestValidationError(message)
        return data

    def _save_entries(self, entries: list[ManifestEntry]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": _MANIFEST_VERSION,
            "entries": [entry.to_json() for entry in entries],
        }
        temporary = self._path.with_name(f"{self._path.name}.tmp")
        text = json.dumps(data, indent=2, sort_keys=True)
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(f"{text}\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self._path)


def record_intent(store: ManifestStore, **kwargs: Any) -> ManifestEntry:
    """Convenience wrapper for integrations that pass dynamic fields."""
    return store.record_intent(**kwargs)


def mark_created(
    store: ManifestStore,
    entry_id: str,
    *,
    exact_id: str,
    arn: str | None = None,
    child_id: str | None = None,
) -> ManifestEntry:
    """Convenience wrapper around ManifestStore.mark_created."""
    return store.mark_created(
        entry_id,
        exact_id=exact_id,
        arn=arn,
        child_id=child_id,
    )


def mark_unknown(
    store: ManifestStore,
    entry_id: str,
    *,
    message: str = "ambiguous create failure",
) -> ManifestEntry:
    """Convenience wrapper around ManifestStore.mark_unknown."""
    return store.mark_unknown(entry_id, message=message)


def verify_ownership(
    entry: ManifestEntry,
    evidence: OwnershipEvidence,
) -> bool:
    """Return True only when cleanup ownership is established."""
    if evidence.account_fingerprint != entry.account_fingerprint:
        return False
    if evidence.region != entry.region:
        return False
    expected_id = entry.child_id or entry.exact_id
    if expected_id is not None and evidence.exact_id != expected_id:
        return False
    if entry.ownership == "tagged":
        return (
            entry.exact_id is not None
            and evidence.exact_id == entry.exact_id
            and evidence.tags.get("run-id") == entry.run_id
            and _matches_name_scheme(entry)
        )
    if entry.ownership == "manifest_child":
        return (
            entry.parent_entry_id is not None
            and evidence.parent_verified
            and expected_id is not None
            and evidence.exact_id == expected_id
        )
    return (
        entry.ownership == "session"
        and entry.exact_id is not None
        and evidence.exact_id == entry.exact_id
        and evidence.name == entry.intended_name
        and evidence.client_token == entry.client_token
    )


def cleanup_manifest(
    store: ManifestStore,
    ports: Mapping[str, CleanupPort],
    *,
    execute: bool = False,
    max_attempts: int = 2,
) -> CleanupReport:
    """Inspect or execute reverse-order manifest cleanup."""
    if max_attempts <= 0:
        message = "max_attempts must be positive"
        raise ManifestValidationError(message)
    actions: list[CleanupAction] = []
    incomplete = False
    entries = list(reversed(store.list_entries()))
    for entry in entries:
        if entry.status == "deleted":
            actions.append(
                CleanupAction(entry.entry_id, "skipped", "already deleted"),
            )
            continue
        if not execute:
            actions.append(
                CleanupAction(entry.entry_id, "would_delete", entry.service),
            )
            continue
        port = ports.get(entry.service)
        if port is None:
            incomplete = True
            actions.append(
                CleanupAction(entry.entry_id, "unresolved", "missing adapter"),
            )
            store.mark_unresolved(entry.entry_id, message="missing adapter")
            continue
        current = _reconcile_for_cleanup(store, port, entry)
        if current is None:
            refreshed = _find_entry(store.list_entries(), entry.entry_id)
            if refreshed is not None and refreshed.status == "deleted":
                actions.append(
                    CleanupAction(entry.entry_id, "not_found", "not found"),
                )
                continue
            incomplete = True
            actions.append(
                CleanupAction(entry.entry_id, "unresolved", "not reconciled"),
            )
            store.mark_unresolved(entry.entry_id, message="not reconciled")
            continue
        evidence = port.ownership_evidence(current)
        if evidence is None or not verify_ownership(current, evidence):
            incomplete = True
            actions.append(
                CleanupAction(
                    current.entry_id,
                    "unresolved",
                    "ownership not established",
                ),
            )
            store.mark_unresolved(
                current.entry_id,
                message="ownership not established",
            )
            continue
        action = _delete_with_retries(store, port, current, max_attempts)
        actions.append(action)
        if action.outcome == "failed":
            incomplete = True
    return CleanupReport(
        dry_run=not execute,
        actions=tuple(actions),
        cleanup_incomplete=incomplete,
    )


def _entries_from_data(data: Mapping[str, Any]) -> list[ManifestEntry]:
    raw_entries = cast("list[Mapping[str, Any]]", data["entries"])
    return [ManifestEntry.from_json(raw) for raw in raw_entries]


def _find_entry(
    entries: list[ManifestEntry],
    entry_id: str,
) -> ManifestEntry | None:
    return next(
        (entry for entry in entries if entry.entry_id == entry_id),
        None,
    )


def _assert_same_intent(
    existing: ManifestEntry,
    requested: ManifestEntry,
) -> None:
    comparable_existing = replace(
        existing,
        status="intent",
        exact_id=None,
        arn=None,
        last_error=None,
        delete_attempts=0,
        created_at=requested.created_at,
        updated_at=requested.updated_at,
    )
    if comparable_existing != requested:
        message = "intent id reused with different contents"
        raise ManifestValidationError(message)


def _reconcile_for_cleanup(
    store: ManifestStore,
    port: CleanupPort,
    entry: ManifestEntry,
) -> ManifestEntry | None:
    if entry.status == "intent" or entry.status == "unknown":
        recovered = port.reconcile(entry)
        if recovered is None:
            if entry.status == "intent":
                store.mark_deleted(entry.entry_id, message="not found")
            return None
        return store.upsert_reconciled(recovered)
    if entry.exact_id is None and entry.child_id is None:
        recovered = port.reconcile(entry)
        if recovered is None:
            return None
        return store.upsert_reconciled(recovered)
    return entry


def _delete_with_retries(
    store: ManifestStore,
    port: CleanupPort,
    entry: ManifestEntry,
    max_attempts: int,
) -> CleanupAction:
    last_message = ""
    for _attempt in range(max_attempts):
        result = port.delete(entry)
        if result.outcome == "deleted":
            store.mark_deleted(entry.entry_id, message=result.message)
            return CleanupAction(entry.entry_id, "deleted", result.message)
        if result.outcome == "not_found":
            store.mark_deleted(entry.entry_id, message=result.message)
            return CleanupAction(entry.entry_id, "not_found", result.message)
        last_message = result.message or "delete failed"
        store.mark_delete_failed(entry.entry_id, message=last_message)
    return CleanupAction(entry.entry_id, "failed", last_message)


def _matches_name_scheme(entry: ManifestEntry) -> bool:
    pattern = _NAME_SCHEMES.get(entry.naming_scheme)
    if pattern is not None:
        return pattern.fullmatch(entry.intended_name) is not None
    return entry.intended_name.startswith(("awsai-", "awsai_"))


def _entry_id(
    *,
    run_id: str,
    service: str,
    operation: str,
    intended_name: str,
    client_token: str,
    parent_entry_id: str | None,
) -> str:
    material = "\0".join(
        (
            run_id,
            service,
            operation,
            intended_name,
            client_token,
            parent_entry_id or "",
        ),
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"{service}:{operation}:{digest}"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


@dataclass(slots=True)
class _FileLock:
    path: Path
    timeout_seconds: float = 10.0
    poll_seconds: float = 0.01
    _lock: FileLock | None = field(default=None, init=False)

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(self.path)
        try:
            lock.acquire(
                timeout=self.timeout_seconds,
                poll_interval=self.poll_seconds,
            )
        except Timeout as error:
            message = "timed out waiting for manifest lock"
            raise ManifestError(message) from error
        self._lock = lock
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._lock is not None:
            self._lock.release()
            self._lock = None
