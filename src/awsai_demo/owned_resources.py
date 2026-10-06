"""Durable, ownership-verified guardrail and memory lifecycles."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from awsai_demo.demo_support import (
    BotoAwsPort,
    DemoOperation,
    bounded_boto_config,
    port_operation,
)
from awsai_demo.manifest import (
    ManifestEntry,
    ManifestStore,
    OwnershipEvidence,
    verify_ownership,
)
from awsai_demo.policy import ReservationUnavailable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from awsai_demo.billing import BudgetRun
    from awsai_demo.contracts import Effect, OperationOutcome, Phase
    from awsai_demo.demo_support import AwsPort
    from awsai_demo.policy import Charge, Reservation


@dataclass(frozen=True)
class ResourceKind:
    """Service-specific identity and cleanup fields."""

    service: str
    noun: str
    id_key: str
    id_argument: str
    arn_key: str
    list_key: str
    response_key: str | None = None


GUARDRAIL = ResourceKind(
    "bedrock",
    "Guardrail",
    "guardrailId",
    "guardrailIdentifier",
    "guardrailArn",
    "guardrails",
)
MEMORY = ResourceKind(
    "bedrock-agentcore-control",
    "Memory",
    "id",
    "memoryId",
    "arn",
    "memories",
    "memory",
)
RESOURCE_KINDS = {"CreateGuardrail": GUARDRAIL, "CreateMemory": MEMORY}


class OwnedResources:
    """Persist intent before dispatch and retain cleanup on failures."""

    def __init__(
        self,
        *,
        run: BudgetRun,
        demo: str,
        port: AwsPort,
        operations: list[OperationOutcome],
    ) -> None:
        """Bind an explicit SDK port and the command's shared budget."""
        self.reservation_namespace = uuid.uuid4().hex
        self.run = run
        self.port = port
        self.operations = operations
        self.budget = run.command(demo)
        self.store = ManifestStore(run.directory / "manifest.json")
        self.account_fingerprint: str | None = None

    def call(
        self,
        service: str,
        operation: str,
        params: Mapping[str, Any],
        *,
        effect: Effect,
        phase: Phase = "main",
        charges: tuple[Charge, ...] = (),
        prepaid: Reservation | None = None,
    ) -> DemoOperation:
        """Reserve billable units before dispatch."""
        if phase != "teardown":
            self.budget.ledger.check_wall_clock()
        reservation = prepaid
        identifier = (
            f"{self.reservation_namespace}:{len(self.operations)}:{operation}"
        )
        if prepaid is not None:
            pass
        elif charges:
            reservation = self.budget.reserve(identifier, charges)
        elif effect in {"write", "delete", "session_stop"}:
            reservation = self.budget.reserve_amount(
                identifier,
                Decimal(0),
                kind="cleanup" if phase == "teardown" else "service",
            )
        elif effect == "infer":
            message = "Inference requires a priced reservation"
            raise ReservationUnavailable(message)
        dispatch_port = self.port
        if effect == "delete" and isinstance(dispatch_port, BotoAwsPort):
            # Cleanup retries here; disable nested SDK retries.
            dispatch_port = replace(
                dispatch_port,
                config=bounded_boto_config(
                    replace(
                        self.run.policy,
                        total_max_attempts=1,
                        openai_max_retries=0,
                    )
                ),
            )
        response = port_operation(
            port=dispatch_port,
            service=service,
            operation_name=operation,
            params=params,
            execution="live",
            effect=effect,
            phase=phase,
            fixture_id=f"owned-{operation}",
            reserved=reservation is not None,
            endpoint_url=f"https://{service}.{self.run.region}.amazonaws.com",
        )
        if reservation is not None:
            response.outcome["reserved_usd"] = float(reservation.amount_usd)
        self.operations.append(response.outcome)
        return response

    def authenticate(self) -> bool:
        """Use the selected identity for private ownership checks."""
        response = self.call("sts", "GetCallerIdentity", {}, effect="read")
        account = str(response.payload.get("Account", ""))
        if response.outcome["status"] != "ok" or not account.isdigit():
            if response.outcome["mode"] == "live_identity":
                response.outcome["mode"] = "live_service"
            return False
        self.account_fingerprint = hashlib.sha256(
            account.encode()
        ).hexdigest()[:16]
        return True

    def intent(
        self,
        kind: ResourceKind,
        params: Mapping[str, Any],
        *,
        parent: ManifestEntry | None = None,
        operation: str | None = None,
        token: str | None = None,
    ) -> ManifestEntry:
        """Write parent or child intent before any request is sent."""
        if self.account_fingerprint is None:
            message = "Ownership identity must be verified before create"
            raise ValueError(message)
        return self.store.record_intent(
            run_id=self.run.run_id,
            account_fingerprint=self.account_fingerprint,
            region=self.run.region,
            service=kind.service,
            operation=operation or f"Create{kind.noun}",
            intended_name=str(
                params.get("name", parent.intended_name if parent else "")
            ),
            client_token=token
            or str(
                params.get("clientToken", params.get("clientRequestToken", ""))
            ),
            tags={
                "run-id": self.run.run_id,
                "project": "aws-ai-stack-workshop",
            },
            naming_scheme=kind.noun.lower(),
            ownership="manifest_child" if parent else "tagged",
            parent_entry_id=parent.entry_id if parent else None,
        )

    def create(
        self,
        kind: ResourceKind,
        params: Mapping[str, Any],
    ) -> ManifestEntry | None:
        """Create a parent, retaining ambiguous results for cleanup."""
        entry = self.intent(kind, params)
        self.store.mark_unknown(entry.entry_id)
        response = self.call(
            kind.service,
            f"Create{kind.noun}",
            params,
            effect="write",
            phase="setup",
        )
        payload = self._payload(kind, response)
        identifier = payload.get(kind.id_key)
        if response.outcome["status"] != "ok" or not identifier:
            return None
        return self.store.mark_created(
            entry.entry_id,
            exact_id=str(identifier),
            arn=str(payload.get(kind.arn_key, "")) or None,
        )

    def verify(
        self,
        entry: ManifestEntry,
        kind: ResourceKind,
        *,
        phase: Phase = "teardown",
    ) -> bool:
        """Read exact resource identity and tags before any deletion."""
        found = self.call(
            kind.service,
            f"Get{kind.noun}",
            {kind.id_argument: entry.exact_id},
            effect="read",
            phase=phase,
        )
        payload = self._payload(kind, found)
        arn = str(payload.get(kind.arn_key, ""))
        parts = arn.split(":", maxsplit=5)
        if len(parts) != 6:
            return False
        tags_response = self.call(
            kind.service,
            "ListTagsForResource",
            {"resourceARN" if kind == GUARDRAIL else "resourceArn": arn},
            effect="read",
            phase=phase,
        )
        raw_tags = tags_response.payload.get("tags", {})
        tags = (
            {item["key"]: item["value"] for item in raw_tags}
            if isinstance(raw_tags, list)
            else raw_tags
        )
        identity = OwnershipEvidence(
            exact_id=str(payload.get(kind.id_key, "")),
            account_fingerprint=hashlib.sha256(parts[4].encode()).hexdigest()[
                :16
            ],
            region=parts[3],
            tags=tags,
            name=payload.get("name"),
        )
        return (
            self.account_fingerprint == entry.account_fingerprint
            and self.run.region == entry.region
            and found.outcome["status"] == "ok"
            and tags_response.outcome["status"] == "ok"
            and identity.name == entry.intended_name
            and verify_ownership(entry, identity)
        )

    def reconcile(
        self,
        entry: ManifestEntry,
        kind: ResourceKind,
    ) -> ManifestEntry | None:
        """Recover an exact named match with bounded pagination."""
        params: dict[str, Any] = {"maxResults": 100}
        for _ in range(10):
            response = self.call(
                kind.service,
                "ListMemories" if kind == MEMORY else "ListGuardrails",
                params,
                effect="read",
                phase="teardown",
            )
            if response.outcome["status"] != "ok":
                return None
            for item in response.payload.get(kind.list_key, []):
                candidate = replace(
                    entry,
                    exact_id=str(item[kind.id_key]),
                    arn=item.get(kind.arn_key),
                    status="created",
                )
                if self.verify(candidate, kind):
                    return self.store.upsert_reconciled(candidate)
            token = response.payload.get("nextToken")
            if not token:
                return None
            params["nextToken"] = token
        return None

    def cleanup(self) -> bool:
        """Delete owned parents before marking children gone."""
        incomplete = False
        for entry in reversed(self.store.list_entries()):
            if (
                entry.status == "deleted"
                or entry.service == "s3vectors"
                or entry.ownership == "manifest_child"
            ):
                continue
            kind = RESOURCE_KINDS.get(entry.operation)
            if kind is None:
                incomplete = True
                continue
            current = entry if entry.exact_id else self.reconcile(entry, kind)
            if current is None or not self.verify(current, kind):
                self.store.mark_unresolved(
                    entry.entry_id, message="ownership not established"
                )
                incomplete = True
                continue
            removed = False
            for _ in range(self.run.policy.total_max_attempts):
                response = self.call(
                    kind.service,
                    f"Delete{kind.noun}",
                    {kind.id_argument: current.exact_id},
                    effect="delete",
                    phase="teardown",
                )
                if (
                    response.outcome["status"] == "ok"
                    or response.outcome["http_status"] == 404
                ):
                    self.store.mark_deleted(current.entry_id)
                    for child in self.store.list_entries():
                        if child.parent_entry_id == current.entry_id:
                            self.store.mark_deleted(child.entry_id)
                    removed = True
                    break
                self.store.mark_delete_failed(
                    current.entry_id, message="delete failed"
                )
            incomplete |= not removed
        return incomplete

    @staticmethod
    def _payload(
        kind: ResourceKind, response: DemoOperation
    ) -> Mapping[str, Any]:
        return (
            response.payload
            if kind.response_key is None
            else response.payload.get(kind.response_key, {})
        )
