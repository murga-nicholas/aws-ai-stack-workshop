"""Deterministic vector retrieval and ownership-verified S3 Vectors.

Technology: Amazon S3 Vectors.
Lane: data.
Lifecycle refs: s3-vectors.
Run: uv run awsai-demo s3-vectors
"""

from __future__ import annotations

from decimal import Decimal
from math import sqrt
from typing import TYPE_CHECKING, Any

from awsai_demo.billing import active_budget_run
from awsai_demo.demo_support import (
    billable_not_priced,
    build_default_boto_port,
    build_result,
    local_operation,
    not_run_operation,
)
from awsai_demo.owned_resources import OwnedResources
from awsai_demo.policy import Charge, PolicyError
from awsai_demo.service_rows import ServiceRow
from awsai_demo.vector_resources import VectorResources, request_charges

if TYPE_CHECKING:
    from awsai_demo.contracts import DemoResult, OperationOutcome
    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.demo_support import AwsPort
    from awsai_demo.policy import ExecutionPolicy
    from awsai_demo.runtime import Execution, Settings

VECTORS: tuple[dict[str, Any], ...] = (
    {
        "key": "approval",
        "data": {"float32": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    },
    {
        "key": "budget",
        "data": {"float32": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    },
    {
        "key": "support",
        "data": {"float32": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    },
)
_QUERY = [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def cosine_search() -> list[dict[str, Any]]:
    """Rank the eight-dimensional fixture without a model call."""
    ranked = []
    for item in VECTORS:
        vector = item["data"]["float32"]
        score = sum(a * b for a, b in zip(vector, _QUERY, strict=True)) / (
            sqrt(sum(x * x for x in vector)) * sqrt(sum(x * x for x in _QUERY))
        )
        ranked.append({"key": item["key"], "distance": 1 - score})
    return sorted(ranked, key=lambda item: float(item["distance"]))


def _rows(bucket: str) -> tuple[ServiceRow, ...]:
    name = {"vectorBucketName": bucket, "indexName": "pilot"}
    arn = f"arn:aws:s3vectors:us-east-1:123456789012:bucket/{bucket}"
    return (
        ServiceRow(
            "s3vectors",
            "CreateVectorBucket",
            {"vectorBucketName": bucket},
            {"vectorBucketArn": arn},
            effect="write",
            live=False,
        ),
        ServiceRow(
            "s3vectors",
            "CreateIndex",
            {
                **name,
                "dimension": 8,
                "dataType": "float32",
                "distanceMetric": "cosine",
            },
            {"indexArn": f"{arn}/index/pilot"},
            effect="write",
            live=False,
        ),
        ServiceRow(
            "s3vectors",
            "PutVectors",
            {**name, "vectors": list(VECTORS)},
            {},
            effect="write",
            live=False,
        ),
        ServiceRow(
            "s3vectors",
            "QueryVectors",
            {
                **name,
                "queryVector": {"float32": _QUERY},
                "topK": 1,
                "returnDistance": True,
            },
            {
                "vectors": [
                    {
                        "key": "approval",
                        "distance": cosine_search()[0]["distance"],
                    }
                ],
                "distanceMetric": "cosine",
            },
            live=False,
        ),
        ServiceRow(
            "s3vectors", "DeleteIndex", name, {}, effect="delete", live=False
        ),
        ServiceRow(
            "s3vectors",
            "DeleteVectorBucket",
            {"vectorBucketName": bucket},
            {},
            effect="delete",
            live=False,
        ),
    )


def _data_charges() -> tuple[Charge, ...]:
    # https://aws.amazon.com/s3/pricing/ specifies a 128 KiB PUT minimum
    # and 256 returned bytes/result. One MiB exceeds both here.
    # One month also covers normal reclamation (up to one day).
    megabyte_gb = Decimal(1024 * 1024) / Decimal(1024**3)
    return (
        Charge("s3vectors.put_gb", megabyte_gb, "gb"),
        Charge("s3vectors.query_returned_gb", megabyte_gb, "gb"),
        Charge("s3vectors.storage_gb_month", megabyte_gb, "gb_month"),
        *(
            Charge(f"s3vectors.query_processed_gb.tier{i}", megabyte_gb, "gb")
            for i in (1, 2, 3)
        ),
    )


def _live(
    *,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort,
    operations: list[OperationOutcome],
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "cleanup_incomplete": False,
        "reference": cosine_search(),
        "price_bound": (
            "sum of all request and processed-byte tiers; "
            "1 MiB data for one month"
        ),
    }
    if not settings.allow_create:
        operations.append(
            not_run_operation(
                service="s3vectors",
                operation_name="CreateVectorBucket",
                phase="setup",
                error_code="create_not_allowed",
                request_validated=False,
            )
        )
        return data
    ctx = None
    try:
        run = active_budget_run(region=settings.region, policy=policy)
        data["run_id"] = run.run_id
        ctx = OwnedResources(
            run=run, demo="s3-vectors", port=port, operations=operations
        )
        # Verify all required prices before sending a create request.
        ctx.budget.quote((*request_charges(), *_data_charges()))
        if not ctx.authenticate():
            return data
        resources = VectorResources(ctx)
        bucket_name = f"awsai-{run.run_id}-vec"
        rows = _rows(bucket_name)
        parent = resources.create(
            "CreateVectorBucket",
            {
                "vectorBucketName": bucket_name,
                "tags": {
                    "run-id": run.run_id,
                    "project": "aws-ai-stack-workshop",
                },
            },
        )
        if (
            parent is None
            or resources.verify_bucket(parent, phase="setup") is None
        ):
            return data
        index = resources.create("CreateIndex", rows[1].params, parent)
        if index is None:
            return data
        # The write intent belongs to the exact index just created.
        pending = ctx.store.record_intent(
            run_id=run.run_id,
            account_fingerprint=parent.account_fingerprint,
            region=run.region,
            service="s3vectors",
            operation="PutVectors",
            intended_name="pilot",
            client_token=f"{run.run_id}-put",
            tags={},
            naming_scheme="vector_index",
            ownership="manifest_child",
            parent_entry_id=index.entry_id,
        )
        ctx.store.mark_unknown(pending.entry_id)
        response = resources.call(
            "PutVectors",
            rows[2].params,
            effect="write",
            charges=_data_charges(),
        )
        if response.outcome["status"] != "ok":
            return data
        ctx.store.mark_created(
            pending.entry_id,
            exact_id=index.exact_id or "",
            child_id=index.exact_id,
        )
        response = resources.call(
            "QueryVectors", rows[3].params, effect="read"
        )
        data["matches"] = response.payload.get("vectors", [])
        data["matches_reference"] = [
            item["key"] for item in response.payload.get("vectors", [])
        ] == ["approval"]
    except PolicyError:
        operations.append(
            billable_not_priced(
                service="s3vectors", operation_name="ReservedVectorLifecycle"
            )
        )
    finally:
        if ctx is not None:
            try:
                data["cleanup_incomplete"] = VectorResources(ctx).cleanup()
            except PolicyError:
                data["cleanup_incomplete"] = True
    return data


def run_s3_vectors_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Compare a local reference with the selected API lane."""
    default = (
        None
        if port is not None
        else build_default_boto_port(
            execution=execution,
            settings=settings,
            policy=policy,
            session_factory=session_factory,
            csv_loader=csv_loader,
        )
    )
    port = port if default is None else default.port
    operations: list[OperationOutcome] = []
    if execution != "emulator":
        operations.append(
            local_operation(service="local", operation_name="CosineSearch")
        )
    if execution == "live" and port is not None:
        data = _live(
            settings=settings, policy=policy, port=port, operations=operations
        )
    else:
        responses = [
            row.run(
                execution=execution,
                settings=settings,
                policy=policy,
                port=port,
                command="s3-vectors",
            )
            for row in _rows("awsai-000000000001-vec")
        ]
        for index, response in enumerate(responses):
            response.outcome["phase"] = (
                "setup" if index < 2 else "teardown" if index > 3 else "main"
            )
            operations.append(response.outcome)
        data = {
            "reference": cosine_search(),
            "matches": responses[3].payload.get("vectors", []),
            "cleanup_incomplete": False,
        }
    return build_result(
        demo="s3-vectors",
        technology="Amazon S3 Vectors",
        lane="data",
        lifecycle_refs=["s3-vectors"],
        execution=execution,
        headline="Eight-dimensional vectors share a cosine-search reference.",
        operations=operations,
        settings=settings,
        data=data,
        credential_source=None
        if default is None
        else default.credential_source,
        result_error=(
            "cleanup_incomplete",
            "Vector resources remain unresolved; use the run manifest.",
        )
        if data["cleanup_incomplete"]
        else None,
    )
