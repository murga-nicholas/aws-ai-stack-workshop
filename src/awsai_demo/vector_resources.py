"""Durable ownership and reverse teardown for S3 vector resources."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from awsai_demo.demo_support import BotoAwsPort, bounded_boto_config
from awsai_demo.manifest import OwnershipEvidence, verify_ownership
from awsai_demo.policy import Charge

if TYPE_CHECKING:
    from collections.abc import Mapping

    from awsai_demo.contracts import Effect, Phase
    from awsai_demo.demo_support import DemoOperation
    from awsai_demo.manifest import ManifestEntry
    from awsai_demo.owned_resources import OwnedResources


def request_charges() -> tuple[Charge, ...]:
    """Sum distinct request tiers for a conservative request bound."""
    return tuple(
        Charge(f"s3vectors.requests.tier{tier}", Decimal(1), "request")
        for tier in (1, 2, 3)
    )


class VectorResources:
    """Own exact bucket and index names."""

    def __init__(self, context: OwnedResources) -> None:
        """Bind the shared manifest and already-selected identity."""
        self.context = context
        if isinstance(context.port, BotoAwsPort):
            context.port = replace(
                context.port,
                config=bounded_boto_config(
                    replace(
                        context.run.policy,
                        total_max_attempts=1,
                        openai_max_retries=0,
                    )
                ),
            )

    def call(
        self,
        operation: str,
        params: Mapping[str, Any],
        *,
        effect: Effect,
        phase: Phase = "main",
        charges: tuple[Charge, ...] = (),
    ) -> DemoOperation:
        """Reserve requests, including cleanup, before dispatch."""
        ctx = self.context
        amount = ctx.budget.quote((*request_charges(), *charges))
        reservation = ctx.budget.reserve_amount(
            f"vector:{ctx.reservation_namespace}:{len(ctx.operations)}:"
            f"{operation}",
            amount,
            kind="cleanup" if phase == "teardown" else "service",
        )
        return ctx.call(
            "s3vectors",
            operation,
            params,
            effect=effect,
            phase=phase,
            prepaid=reservation,
        )

    def create(
        self,
        operation: str,
        params: Mapping[str, Any],
        parent: ManifestEntry | None = None,
    ) -> ManifestEntry | None:
        """Record intent, then persist returned identity."""
        ctx = self.context
        if ctx.account_fingerprint is None:
            message = "S3 vector ownership needs STS identity first"
            raise ValueError(message)
        name = str(
            params["vectorBucketName"]
            if parent is None
            else params["indexName"]
        )
        entry = ctx.store.record_intent(
            run_id=ctx.run.run_id,
            account_fingerprint=ctx.account_fingerprint,
            region=ctx.run.region,
            service="s3vectors",
            operation=operation,
            intended_name=name,
            client_token=f"{ctx.run.run_id}-{operation}",
            tags={
                "run-id": ctx.run.run_id,
                "project": "aws-ai-stack-workshop",
            },
            naming_scheme="vector_bucket"
            if parent is None
            else "vector_index",
            ownership="tagged" if parent is None else "manifest_child",
            parent_entry_id=None if parent is None else parent.entry_id,
        )
        ctx.store.mark_unknown(entry.entry_id)
        response = self.call(operation, params, effect="write", phase="setup")
        arn = response.payload.get(
            "vectorBucketArn" if parent is None else "indexArn"
        )
        if response.outcome["status"] != "ok" or not arn:
            return None
        return ctx.store.mark_created(
            entry.entry_id,
            exact_id=str(arn),
            arn=str(arn),
            child_id=None if parent is None else str(arn),
        )

    def verify_bucket(
        self, entry: ManifestEntry, *, phase: Phase = "teardown"
    ) -> ManifestEntry | None:
        """Verify exact name, tags, account and region."""
        ctx = self.context
        if (
            ctx.account_fingerprint != entry.account_fingerprint
            or ctx.run.region != entry.region
            or ctx.run.run_id != entry.run_id
            or entry.intended_name != f"awsai-{entry.run_id}-vec"
        ):
            return None
        found = self.call(
            "GetVectorBucket",
            {"vectorBucketName": entry.intended_name},
            effect="read",
            phase=phase,
        )
        if found.outcome["http_status"] == 404:
            self._mark_deleted_tree(entry.entry_id)
            return None
        bucket = found.payload.get("vectorBucket", {})
        arn = str(bucket.get("vectorBucketArn", ""))
        parts = arn.split(":", maxsplit=5)
        if (
            len(parts) != 6
            or bucket.get("vectorBucketName") != entry.intended_name
        ):
            return None
        if entry.exact_id is None:
            entry = ctx.store.mark_created(
                entry.entry_id, exact_id=arn, arn=arn
            )
        tags = self.call(
            "ListTagsForResource",
            {"resourceArn": arn},
            effect="read",
            phase=phase,
        )
        identity = OwnershipEvidence(
            exact_id=arn,
            account_fingerprint=hashlib.sha256(parts[4].encode()).hexdigest()[
                :16
            ],
            region=parts[3],
            tags=tags.payload.get("tags", {}),
            name=entry.intended_name,
        )
        if not verify_ownership(entry, identity):
            return None
        return entry

    def _mark_deleted_tree(self, entry_id: str) -> None:
        """Mark the exact resource and descendants absent."""
        store = self.context.store
        store.mark_deleted(entry_id)
        for child in store.list_entries():
            if child.parent_entry_id == entry_id:
                self._mark_deleted_tree(child.entry_id)

    def cleanup(self) -> bool:
        """Delete verified indexes before their bucket."""
        ctx = self.context
        for recorded in reversed(ctx.store.list_entries()):
            entry = next(
                item
                for item in ctx.store.list_entries()
                if item.entry_id == recorded.entry_id
            )
            if (
                entry.service != "s3vectors"
                or entry.status == "deleted"
                or entry.operation == "PutVectors"
            ):
                continue
            if entry.operation == "CreateIndex":
                parents = {
                    item.entry_id: item for item in ctx.store.list_entries()
                }
                parent = parents[entry.parent_entry_id or ""]
                owned = self.verify_bucket(parent)
                if owned is None:
                    current = next(
                        item
                        for item in ctx.store.list_entries()
                        if item.entry_id == entry.entry_id
                    )
                    if current.status == "deleted":
                        continue
                    ctx.store.mark_unresolved(
                        entry.entry_id, message="parent ownership unavailable"
                    )
                    continue
                found = self.call(
                    "GetIndex",
                    {
                        "vectorBucketName": parent.intended_name,
                        "indexName": entry.intended_name,
                    },
                    effect="read",
                    phase="teardown",
                )
                if found.outcome["http_status"] == 404:
                    self._mark_deleted_tree(entry.entry_id)
                    continue
                index = found.payload.get("index", {})
                arn = str(index.get("indexArn", ""))
                expected = f"{owned.exact_id}/index/{entry.intended_name}"
                if (
                    arn != expected
                    or index.get("indexName") != entry.intended_name
                ):
                    ctx.store.mark_unresolved(
                        entry.entry_id, message="index identity mismatch"
                    )
                    continue
                if entry.exact_id is None:
                    entry = ctx.store.mark_created(
                        entry.entry_id, exact_id=arn, arn=arn, child_id=arn
                    )
                if entry.exact_id != arn:
                    ctx.store.mark_unresolved(
                        entry.entry_id, message="index identity changed"
                    )
                    continue
                operation, params = "DeleteIndex", {"indexArn": arn}
            else:
                owned = self.verify_bucket(entry)
                if owned is None:
                    current = next(
                        item
                        for item in ctx.store.list_entries()
                        if item.entry_id == entry.entry_id
                    )
                    if current.status != "deleted":
                        ctx.store.mark_unresolved(
                            entry.entry_id,
                            message="bucket ownership unavailable",
                        )
                    continue
                operation, params = (
                    "DeleteVectorBucket",
                    {"vectorBucketArn": str(owned.exact_id)},
                )
            for _ in range(ctx.run.policy.total_max_attempts):
                response = self.call(
                    operation, params, effect="delete", phase="teardown"
                )
                if (
                    response.outcome["status"] == "ok"
                    or response.outcome["http_status"] == 404
                ):
                    self._mark_deleted_tree(entry.entry_id)
                    break
                ctx.store.mark_delete_failed(
                    entry.entry_id, message="vector deletion failed"
                )
        return any(
            entry.service == "s3vectors" and entry.status != "deleted"
            for entry in ctx.store.list_entries()
        )
