from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError

from awsai_demo.billing import open_budget_run, use_budget_run
from awsai_demo.demo_support import AwsResponse, BotoAwsPort

if TYPE_CHECKING:
    from awsai_demo.manifest import ManifestStore
from awsai_demo.owned_resources import OwnedResources
from awsai_demo.policy import Charge, ExecutionPolicy, ReservationUnavailable
from awsai_demo.resource_cleanup import execute_cleanup
from awsai_demo.runtime import Settings
from awsai_demo.s3_vectors_demo import run_s3_vectors_demo
from awsai_demo.vector_resources import VectorResources

_ACCOUNT = "123456789012"
_RUN = "abcdef123456"
_BUCKET = f"awsai-{_RUN}-vec"
_ARN = f"arn:aws:s3vectors:us-east-1:{_ACCOUNT}:bucket/{_BUCKET}"


class Prices:
    def __init__(self, *, missing: bool = False, after: int = 1000) -> None:
        self.missing, self.after = missing, after
        self.calls = 0

    def rate(self, *_: Any, **__: Any) -> Decimal:
        self.calls += 1
        if self.missing or self.calls > self.after:
            message = "Unpriced unit"
            raise ReservationUnavailable(message)
        return Decimal("0.000001")


class VectorPort:
    def __init__(
        self,
        *,
        fail: str = "",
        tags: bool = True,
        bad: str = "",
        ambiguous: bool = False,
        delete_failures: int = 0,
    ) -> None:
        self.fail, self.tags, self.bad = fail, tags, bad
        self.ambiguous, self.delete_failures = ambiguous, delete_failures
        self.calls: list[str] = []
        self.bucket = False
        self.index = False
        self.manifest: ManifestStore | None = None

    def call(
        self, service: str, operation_name: str, params: Any
    ) -> AwsResponse:
        del params
        self.calls.append(operation_name)
        if operation_name.startswith("Create") and self.manifest is not None:
            entries = self.manifest.list_entries()
            assert entries[-1].status == "unknown"
        if operation_name == "CreateVectorBucket":
            self.bucket = True
        if operation_name == "CreateIndex":
            self.index = True
        if operation_name == self.fail:
            if self.ambiguous:
                message = "connection lost after dispatch"
                raise OSError(message)
            raise self.error(operation_name, 403)
        if operation_name.startswith("Delete") and self.delete_failures:
            self.delete_failures -= 1
            raise self.error(operation_name, 500)
        if operation_name == "GetVectorBucket" and not self.bucket:
            raise self.error(operation_name, 404)
        if operation_name == "GetIndex" and not self.index:
            raise self.error(operation_name, 404)
        if operation_name == "DeleteIndex":
            self.index = False
        if operation_name == "DeleteVectorBucket":
            if self.index:
                raise self.error(operation_name, 409)
            self.bucket = False
        bucket = {"vectorBucketName": _BUCKET, "vectorBucketArn": _ARN}
        index = {"indexName": "pilot", "indexArn": f"{_ARN}/index/pilot"}
        if self.bad == "bucket_arn":
            bucket["vectorBucketArn"] = "invalid"
        elif self.bad == "bucket_name":
            bucket["vectorBucketName"] = "foreign"
        elif self.bad == "index":
            index["indexArn"] = f"{_ARN}/index/foreign"
        payloads: dict[str, Any] = {
            "GetCallerIdentity": {"Account": _ACCOUNT},
            "CreateVectorBucket": {"vectorBucketArn": _ARN},
            "CreateIndex": {"indexArn": f"{_ARN}/index/pilot"},
            "GetVectorBucket": {"vectorBucket": bucket},
            "GetIndex": {"index": index},
            "ListTagsForResource": {
                "tags": {"run-id": _RUN} if self.tags else {}
            },
            "QueryVectors": {
                "vectors": [{"key": "approval", "distance": 0.01}]
            },
        }
        return AwsResponse(
            payloads.get(operation_name, {}),
            f"https://{service}.us-east-1.amazonaws.com",
        )

    @staticmethod
    def error(name: str, status: int) -> ClientError:
        return ClientError(
            {
                "Error": {
                    "Code": "ResourceNotFoundException"
                    if status == 404
                    else "AccessDeniedException",
                    "Message": "synthetic diagnostic",
                },
                "ResponseMetadata": {"HTTPStatusCode": status},
            },
            name,
        )


def run_context(
    tmp_path: Any,
    port: VectorPort,
    *,
    prices: Prices | None = None,
    policy: ExecutionPolicy | None = None,
) -> OwnedResources:
    run = open_budget_run(
        region="us-east-1",
        policy=policy or ExecutionPolicy(),
        root=tmp_path,
        prices=prices or Prices(),
        run_id=_RUN,
    )
    ctx = OwnedResources(run=run, demo="s3-vectors", port=port, operations=[])
    port.manifest = ctx.store
    return ctx


def live_demo(
    tmp_path: Any,
    port: VectorPort,
    *,
    prices: Prices | None = None,
    policy: ExecutionPolicy | None = None,
) -> tuple[Any, OwnedResources]:
    ctx = run_context(tmp_path, port, prices=prices, policy=policy)
    with use_budget_run(ctx.run):
        answer = run_s3_vectors_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=ctx.run.policy,
            port=port,
        )
    return answer, ctx


def test_vector_offline_and_emulator() -> None:
    for execution in ("offline", "emulator"):
        answer = run_s3_vectors_demo(
            execution=execution, settings=Settings(), policy=ExecutionPolicy()
        )
        assert answer["status"] == (
            "ok" if execution == "offline" else "blocked"
        )
        assert answer["operations"][-1]["phase"] == "teardown"
    answer = run_s3_vectors_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=VectorPort(),
    )
    assert answer["operations"][-1]["error_code"] == "create_not_allowed"


def test_vector_success_and_owned_cleanup(tmp_path: Any) -> None:
    port = VectorPort(delete_failures=1)
    answer, ctx = live_demo(tmp_path, port)
    assert answer["data"]["matches_reference"] is True
    assert answer["data"]["cleanup_incomplete"] is False
    assert not port.bucket and not port.index
    assert all(entry.status == "deleted" for entry in ctx.store.list_entries())
    assert port.calls.index("DeleteIndex") < port.calls.index(
        "DeleteVectorBucket"
    )
    assert [
        item
        for item in answer["operations"]
        if item["service"] == "s3vectors" and item["status"] == "ok"
    ]
    assert VectorResources(ctx).cleanup() is False


@pytest.mark.parametrize(
    "fail,ambiguous",
    [
        ("GetCallerIdentity", False),
        ("CreateVectorBucket", True),
        ("CreateIndex", True),
        ("PutVectors", False),
        ("QueryVectors", False),
    ],
)
def test_vector_failure_retains_cleanup(
    tmp_path: Any, fail: str, ambiguous: bool
) -> None:
    port = VectorPort(fail=fail, ambiguous=ambiguous)
    answer, _ = live_demo(tmp_path, port)
    assert answer["status"] == "blocked"
    assert answer["data"]["cleanup_incomplete"] is False
    assert not port.bucket and not port.index


@pytest.mark.parametrize(
    "bad,tags",
    [
        ("bucket_arn", True),
        ("bucket_name", True),
        ("", False),
        ("index", True),
    ],
)
def test_vector_foreign_resource_is_not_deleted(
    tmp_path: Any, bad: str, tags: bool
) -> None:
    port = VectorPort(bad=bad, tags=tags)
    answer, ctx = live_demo(tmp_path, port)
    assert answer["status"] == "error"
    assert answer["error"]["code"] == "cleanup_incomplete"
    assert any(
        item.status == "unresolved" for item in ctx.store.list_entries()
    )
    if bad != "index":
        assert "DeleteVectorBucket" not in port.calls


def test_vector_price_missing_never_creates(tmp_path: Any) -> None:
    port = VectorPort()
    answer, _ = live_demo(tmp_path, port, prices=Prices(missing=True))
    assert answer["operations"][-1]["error_code"] == "budget_exceeded"
    assert not port.calls


def test_vector_cleanup_price_failure_and_delete_failure(
    tmp_path: Any,
) -> None:
    answer, _ = live_demo(
        tmp_path / "prices", VectorPort(), prices=Prices(after=12)
    )
    assert answer["data"]["cleanup_incomplete"] is True
    answer, _ = live_demo(
        tmp_path / "deletion", VectorPort(delete_failures=20)
    )
    assert answer["data"]["cleanup_incomplete"] is True


def create_pair(ctx: OwnedResources) -> tuple[VectorResources, Any, Any]:
    assert ctx.authenticate()
    resources = VectorResources(ctx)
    parent = resources.create(
        "CreateVectorBucket", {"vectorBucketName": _BUCKET}
    )
    assert parent is not None
    child = resources.create(
        "CreateIndex",
        {
            "vectorBucketName": _BUCKET,
            "indexName": "pilot",
            "dimension": 8,
            "dataType": "float32",
            "distanceMetric": "cosine",
        },
        parent,
    )
    assert child is not None
    return resources, parent, child


def test_vector_reconciliation_missing_and_changed(tmp_path: Any) -> None:
    port = VectorPort()
    ctx = run_context(tmp_path, port)
    with pytest.raises(ValueError, match="STS"):
        VectorResources(ctx).create(
            "CreateVectorBucket", {"vectorBucketName": _BUCKET}
        )
    resources, _parent, child = create_pair(ctx)
    port.index = False
    assert resources.cleanup() is False
    assert all(entry.status == "deleted" for entry in ctx.store.list_entries())
    # Confirmed absence also resolves the exact manifest descendants.
    ctx = run_context(tmp_path / "absent", port)
    resources, _parent, child = create_pair(ctx)
    port.bucket = False
    assert resources.cleanup() is False
    assert ctx.store.list_entries()[0].status == "deleted"
    # Exact recorded ids are immutable ownership evidence.
    port = VectorPort()
    ctx = run_context(tmp_path / "changed", port)
    resources, _parent, child = create_pair(ctx)
    ctx.store.mark_created(
        child.entry_id,
        exact_id=f"{_ARN}/index/old",
        child_id=f"{_ARN}/index/old",
    )
    assert resources.cleanup() is True
    assert "DeleteIndex" not in port.calls


def test_vector_parent_account_mismatch(tmp_path: Any) -> None:
    port = VectorPort()
    ctx = run_context(tmp_path, port)
    resources, parent, _ = create_pair(ctx)
    ctx.account_fingerprint = "other"
    assert resources.verify_bucket(parent) is None


def test_vector_unknown_create_without_identifier(tmp_path: Any) -> None:
    class EmptyCreate(VectorPort):
        def call(
            self, service: str, operation_name: str, params: Any
        ) -> AwsResponse:
            result = super().call(service, operation_name, params)
            return (
                AwsResponse({}, result.endpoint_url)
                if operation_name == "CreateVectorBucket"
                else result
            )

    answer, _ = live_demo(tmp_path, EmptyCreate())
    assert answer["data"]["cleanup_incomplete"] is False


def test_vector_boto_adapter_disables_create_retries(tmp_path: Any) -> None:
    ctx = run_context(tmp_path, VectorPort())
    ctx.port = BotoAwsPort(session=object(), region_name="us-east-1")
    VectorResources(ctx)
    assert ctx.port.config.retries["total_max_attempts"] == 1


def test_cleanup_command_vector_adapter(tmp_path: Any) -> None:

    port = VectorPort()
    ctx = run_context(tmp_path, port)
    resources, _, _ = create_pair(ctx)
    # Adapter behavior itself is covered with the same shared context.
    assert resources.cleanup() is False
    result = execute_cleanup(
        run_id=_RUN, root=tmp_path, settings=Settings(), port=port
    )
    assert result["data"]["cleanup_incomplete"] is False


def test_vector_missing_snapshot_skips_creates(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    port = VectorPort()
    result = run_s3_vectors_demo(
        execution="live",
        settings=Settings(allow_create=True),
        policy=ExecutionPolicy(),
        port=port,
    )
    assert result["operations"][-1]["error_code"] == "budget_exceeded"
    assert not port.calls


@pytest.mark.parametrize("priced", [True, False])
def test_cleanup_command_pending_vectors(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, priced: bool
) -> None:
    from awsai_demo.pricing import PriceRow, PriceSnapshot, write_snapshot

    monkeypatch.chdir(tmp_path)
    port = VectorPort()
    ctx = run_context(tmp_path, port)
    create_pair(ctx)
    if priced:
        rows = tuple(
            PriceRow(
                "AmazonS3",
                f"s3vectors.requests.tier{i}",
                "us-east-1",
                f"Vectors-Request-Tier{i}",
                "standard",
                "in-region",
                "request",
                Decimal("0.000001"),
                "Requests",
                Decimal("0.000001"),
                "2026-10-06",
                f"SKU{i}",
                f"RATE{i}",
                "0",
                "Inf",
                "request fixture",
            )
            for i in (1, 2, 3)
        )
        snapshot = PriceSnapshot(
            1,
            "us-east-1",
            ("us-east-1",),
            "2026-10-06",
            ("AmazonS3",),
            rows,
            (),
            (),
        )
        write_snapshot(snapshot, tmp_path / "data" / "pricing_snapshot.json")
    result = execute_cleanup(
        run_id=_RUN, root=tmp_path, settings=Settings(), port=port
    )
    assert result["data"]["cleanup_incomplete"] is not priced
    assert (not port.bucket) is priced


def test_cleanup_empty_manifest_selects_no_identity(tmp_path: Any) -> None:
    port = VectorPort()
    result = execute_cleanup(
        run_id=_RUN, root=tmp_path, settings=Settings(), port=port
    )
    assert not port.calls and result["data"]["cleanup_incomplete"] is False


def pending_write(ctx: OwnedResources, child: Any) -> None:
    ctx.store.record_intent(
        run_id=_RUN,
        account_fingerprint=child.account_fingerprint,
        region="us-east-1",
        service="s3vectors",
        operation="PutVectors",
        intended_name="pilot",
        client_token=f"{_RUN}-put",
        tags={},
        naming_scheme="vector_index",
        ownership="manifest_child",
        parent_entry_id=child.entry_id,
    )


@pytest.mark.parametrize("missing", ["index", "bucket"])
def test_absent_resource_cascades_pending_descendants(
    tmp_path: Any, missing: str
) -> None:
    port = VectorPort()
    ctx = run_context(tmp_path, port)
    resources, _, child = create_pair(ctx)
    pending_write(ctx, child)
    port.index = False
    if missing == "bucket":
        port.bucket = False
    assert resources.cleanup() is False
    assert all(entry.status == "deleted" for entry in ctx.store.list_entries())
    assert "DeleteIndex" not in port.calls


@pytest.mark.parametrize("mismatch", ["account", "region", "run", "name"])
def test_absence_in_foreign_identity_is_not_cleanup(
    tmp_path: Any, mismatch: str
) -> None:
    from dataclasses import replace

    port = VectorPort()
    ctx = run_context(tmp_path, port)
    resources, parent, _ = create_pair(ctx)
    port.bucket = False
    if mismatch == "account":
        ctx.account_fingerprint = "foreign"
    elif mismatch == "region":
        parent = replace(parent, region="eu-west-1")
    elif mismatch == "run":
        parent = replace(parent, run_id="000000000000")
    else:
        parent = replace(parent, intended_name="foreign")
    before = len(port.calls)
    assert resources.verify_bucket(parent) is None
    assert len(port.calls) == before
    assert ctx.store.list_entries()[0].status == "created"


def test_fresh_dispatch_context_reserves_again(tmp_path: Any) -> None:
    port = VectorPort()
    first = run_context(tmp_path, port)
    second = OwnedResources(
        run=first.run, demo="s3-vectors", port=port, operations=[]
    )
    for ctx in (first, second):
        VectorResources(ctx).call(
            "GetVectorBucket",
            {"vectorBucketName": _BUCKET},
            effect="read",
            phase="teardown",
        )
    snapshot = first.budget.ledger.snapshot()
    assert len(snapshot.reservations) == 2
    assert snapshot.cleanup_reserved_usd == Decimal("0.000012")
    assert first.reservation_namespace != second.reservation_namespace
    # Generic billable resource calls need a fresh namespace too.
    for ctx in (first, second):
        ctx.call(
            "comprehend",
            "DetectSentiment",
            {"Text": "approval pending", "LanguageCode": "en"},
            effect="infer",
            charges=(
                Charge(
                    "comprehend.detect_sentiment.units", Decimal(3), "unit"
                ),
            ),
        )
    snapshot = first.budget.ledger.snapshot()
    assert len(snapshot.reservations) == 4
    assert snapshot.reserved_usd == Decimal("0.000012")


def test_vector_cleanup_rejects_lost_parent_tags(tmp_path: Any) -> None:
    port = VectorPort()
    ctx = run_context(tmp_path, port)
    resources, _, child = create_pair(ctx)
    pending_write(ctx, child)
    port.tags = False
    assert resources.cleanup() is True
    assert "DeleteIndex" not in port.calls
    assert "DeleteVectorBucket" not in port.calls
    assert all(entry.status != "deleted" for entry in ctx.store.list_entries())


def test_vector_missing_bucket_without_children(tmp_path: Any) -> None:
    port = VectorPort()
    ctx = run_context(tmp_path, port)
    assert ctx.authenticate()
    resources = VectorResources(ctx)
    assert resources.create(
        "CreateVectorBucket", {"vectorBucketName": _BUCKET}
    )
    port.bucket = False
    assert resources.cleanup() is False
    assert ctx.store.list_entries()[0].status == "deleted"
