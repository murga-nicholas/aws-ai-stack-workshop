"""Explicit cleanup from a private run manifest."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from awsai_demo.agentcore_tools_demo import InterpreterSession
from awsai_demo.billing import SnapshotPrices, open_budget_run
from awsai_demo.demo_support import build_default_boto_port, build_result
from awsai_demo.manifest import ManifestStore
from awsai_demo.owned_resources import OwnedResources
from awsai_demo.policy import ExecutionPolicy, PolicyError
from awsai_demo.vector_resources import VectorResources

if TYPE_CHECKING:
    from awsai_demo.contracts import DemoResult, OperationOutcome
    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.demo_support import AwsPort, DefaultBotoPort
    from awsai_demo.runtime import Settings


def execute_cleanup(
    *,
    run_id: str,
    root: Path,
    settings: Settings,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Verify ownership before teardown with the selected identity."""
    policy = ExecutionPolicy()
    operations: list[OperationOutcome] = []
    store = ManifestStore(root / run_id / "manifest.json")
    entries = store.list_entries()
    if entries:
        settings = replace(settings, region=entries[0].region)
    incomplete = False
    source = None
    if any(entry.status != "deleted" for entry in entries):
        if port is None:
            default = cast(
                "DefaultBotoPort",
                build_default_boto_port(
                    execution="live",
                    settings=settings,
                    policy=policy,
                    session_factory=session_factory,
                    csv_loader=csv_loader,
                ),
            )
            port, source = default.port, default.credential_source
        run = open_budget_run(
            region=settings.region,
            policy=policy,
            root=root,
            run_id=run_id,
            prices=SnapshotPrices(Path("data/pricing_snapshot.json")),
        )
        context = OwnedResources(
            run=run, demo="cleanup", port=port, operations=operations
        )
        if context.authenticate():
            session = InterpreterSession(context)
            for entry in reversed(entries):
                if entry.operation == "StartCodeInterpreterSession":
                    incomplete |= session.cleanup_entry(entry)
            if any(entry.service == "s3vectors" for entry in entries):
                try:
                    incomplete |= VectorResources(context).cleanup()
                except PolicyError:
                    incomplete = True
            incomplete |= context.cleanup()
        else:
            incomplete = True
    return build_result(
        demo="cleanup",
        technology="Owned resource cleanup",
        lane="operations",
        lifecycle_refs=["iam-sts"],
        execution="live",
        settings=settings,
        headline="Ownership-verified cleanup of the named run.",
        operations=operations,
        credential_source=source,
        data={
            "run_id": run_id,
            "dry_run": False,
            "cleanup_incomplete": incomplete,
            "entries": [
                {
                    "service": entry.service,
                    "operation": entry.operation,
                    "status": entry.status,
                }
                for entry in store.list_entries()
            ],
        },
        result_error=(
            "cleanup_incomplete",
            "Some owned resources remain unresolved.",
        )
        if incomplete
        else None,
    )
