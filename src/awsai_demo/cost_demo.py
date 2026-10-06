"""Estimate a bounded decision workload using dated Price List evidence.

Technology: Amazon Bedrock pricing.
Lane: operations.
Lifecycle refs: bedrock-service-tiers, bedrock-inference-profiles.
Run: uv run awsai-demo cost
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from awsai_demo.demo_support import (
    AwsPort,
    DefaultBotoPort,
    build_default_boto_port,
    build_result,
    fixture_operation,
    port_operation,
)
from awsai_demo.pricing import (
    PRICING_REGION,
    PriceBook,
    PriceSnapshot,
    PricingError,
    _rows_from_product,
    load_snapshot,
)
from awsai_demo.runtime import with_cli_overrides
from awsai_demo.stubs import stubbed_client

if TYPE_CHECKING:
    from awsai_demo.contracts import (
        CredentialSource,
        DemoResult,
        Execution,
        OperationOutcome,
    )
    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.policy import ExecutionPolicy
    from awsai_demo.runtime import Settings

_FIXTURE_ID = "cost-price-list-v1"
_COMPARISON_MODEL = "global.amazon.nova-2-lite-v1:0"
_TIERS = ("standard", "flex", "priority", "batch")
_ROUTES = ("in-region", "cross-region-global")


def run_cost_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
    snapshot_path: Path = Path("data/pricing_snapshot.json"),
) -> DemoResult:
    """Compare estimated costs without fabricating missing rates."""
    operations: list[OperationOutcome] = []
    credential_source: CredentialSource | None = None
    snapshot = _read_snapshot(snapshot_path)
    if execution == "offline":
        params = _params(settings.region, "AmazonBedrock")
        with stubbed_client("pricing") as stubber:
            stubber.add_response(
                "get_products", {"PriceList": []}, expected_params=params
            )
            stubber.client.get_products(**params)
        operations.append(
            fixture_operation(
                service="pricing",
                operation_name="GetProducts",
                fixture_id=_FIXTURE_ID,
                effect="read",
            )
        )
    elif execution == "live":
        if port is None:
            selected = build_default_boto_port(
                execution="live",
                settings=with_cli_overrides(settings, region=PRICING_REGION),
                policy=policy,
                session_factory=session_factory,
                csv_loader=csv_loader,
            )
            live_port = cast("DefaultBotoPort", selected)
            port = live_port.port
            credential_source = live_port.credential_source
        snapshot = _live_snapshot(settings.region, port, operations)
    data = _cost_data(
        snapshot,
        settings.model or _COMPARISON_MODEL,
        settings.region,
        policy,
    )
    return build_result(
        demo="cost",
        technology="Cost per successful task",
        lane="operations",
        lifecycle_refs=["bedrock-service-tiers", "bedrock-inference-profiles"],
        execution=execution,
        settings=settings,
        headline="Estimated token cost includes every assumed retry.",
        operations=operations,
        data=data,
        credential_source=credential_source,
    )


def _read_snapshot(path: Path) -> PriceSnapshot | None:
    try:
        return load_snapshot(path)
    except (OSError, ValueError, PricingError):
        return None


def _params(region: str, service: str) -> dict[str, Any]:
    return {
        "ServiceCode": service,
        "FormatVersion": "aws_v1",
        "MaxResults": 100,
        "Filters": [
            {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region}
        ],
    }


def _live_snapshot(
    region: str,
    port: AwsPort,
    operations: list[OperationOutcome],
) -> PriceSnapshot:
    import json
    from datetime import UTC, datetime

    rows = []
    read_at = datetime.now(UTC).isoformat()
    for service in ("AmazonBedrock", "AmazonBedrockFoundationModels"):
        params = _params(region, service)
        while True:
            called = port_operation(
                port=port,
                service="pricing",
                operation_name="GetProducts",
                params=params,
                execution="live",
                effect="read",
                fixture_id=_FIXTURE_ID,
            )
            operations.append(called.outcome)
            for document in called.payload.get("PriceList", []):
                rows.extend(
                    _rows_from_product(
                        service_code=service,
                        document=json.loads(document),
                        requested_region=region,
                        read_at=read_at,
                    )
                )
            token = called.payload.get("NextToken")
            if not token:
                break
            params["NextToken"] = token
    return PriceSnapshot(
        1,
        PRICING_REGION,
        (region,),
        read_at,
        ("AmazonBedrock", "AmazonBedrockFoundationModels"),
        tuple(rows),
        (),
        (),
    )


def _cost_data(
    snapshot: PriceSnapshot | None,
    model: str,
    region: str,
    policy: ExecutionPolicy,
) -> dict[str, object]:
    model_id = model.removeprefix("global.").removeprefix("us.")
    book = None if snapshot is None else PriceBook(snapshot)
    rows: list[dict[str, object]] = []
    # Illustrative usage: three calls, including every allowed attempt.
    calls = min(3, policy.max_model_calls)
    tokens_in, tokens_out = 500, min(100, policy.max_output_tokens)
    attempts = policy.total_max_attempts
    for tier in _TIERS:
        for routing in _ROUTES:
            value: str | None = None
            if book is not None:
                try:
                    rates = [
                        book.rate(
                            f"model:{model_id}:{direction}",
                            region=region,
                            unit="token",
                            tier=tier,
                            routing=routing,
                        )
                        for direction in ("input", "output")
                    ]
                    usd = (
                        (rates[0] * tokens_in + rates[1] * tokens_out)
                        * calls
                        * attempts
                    )
                    value = str(usd.quantize(Decimal("0.000000001")))
                except PricingError:
                    pass
            rows.append(
                {
                    "tier": tier,
                    "routing": routing,
                    "usd_per_successful_run": value,
                }
            )
    return {
        "tiers": rows,
        "read_at": None if snapshot is None else snapshot.read_at,
        "basis": "estimated",
        "model": model_id,
        "assumptions": {
            "successful_runs": 1,
            "model_calls": calls,
            "attempts_per_call": attempts,
            "input_tokens_per_attempt": tokens_in,
            "output_tokens_per_attempt": tokens_out,
        },
        "unavailable": [
            row for row in rows if row["usd_per_successful_run"] is None
        ],
    }
