from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from awsai_demo.cost_demo import run_cost_demo
from awsai_demo.demo_support import AwsResponse
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.pricing import PriceRow, PriceSnapshot, write_snapshot
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _snapshot(path: Path) -> None:
    rows = tuple(
        PriceRow(
            "AmazonBedrock",
            f"model:amazon.nova-2-lite-v1:0:{direction}",
            "us-east-1",
            "USE1-Nova2.0Lite-input-tokens",
            tier,
            route,
            "token",
            Decimal("0.000001"),
            "token",
            Decimal("0.000001"),
            "2026-10-06T00:00:00+00:00",
            f"{tier}-{route}-{direction}",
            "rate",
            "0",
            "Inf",
            "public price",
        )
        for tier in ("standard", "flex", "priority", "batch")
        for route in ("in-region", "cross-region-global")
        for direction in ("input", "output")
    )
    write_snapshot(
        PriceSnapshot(
            1,
            "us-east-1",
            ("us-east-1",),
            "2026-10-06T00:00:00+00:00",
            ("AmazonBedrock",),
            rows,
            (),
            (),
        ),
        path,
    )


def test_cost_offline_reads_snapshot_counts_retries_and_marks_estimated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snapshot.json"
    _snapshot(path)
    result = run_cost_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(),
        snapshot_path=path,
    )
    assert result["status"] == "ok"
    assert result["mode"] == "local_contract"
    assert result["data"]["basis"] == "estimated"
    assert result["data"]["read_at"] == "2026-10-06T00:00:00+00:00"
    assert len(result["data"]["tiers"]) == 8
    assert result["data"]["unavailable"] == []
    assert {
        row["usd_per_successful_run"] for row in result["data"]["tiers"]
    } == {"0.003600000"}
    assert result["operations"][0]["operation"] == "GetProducts"


def test_cost_missing_or_incomplete_snapshot_never_fabricates_zero(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.json"
    missing = run_cost_demo(
        execution="emulator",
        settings=Settings(),
        policy=ExecutionPolicy(),
        snapshot_path=path,
    )
    assert missing["data"]["read_at"] is None
    assert len(missing["data"]["unavailable"]) == 8
    assert missing["operations"] == []
    _snapshot(path)
    unavailable = run_cost_demo(
        execution="emulator",
        settings=Settings(model="other.model"),
        policy=ExecutionPolicy(),
        snapshot_path=path,
    )
    assert all(
        row["usd_per_successful_run"] is None
        for row in unavailable["data"]["tiers"]
    )


class PricePort:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    def call(
        self, service: str, operation_name: str, params: Mapping[str, Any]
    ) -> AwsResponse:
        assert service == "pricing"
        assert operation_name == "GetProducts"
        self.calls.append(dict(params))
        if len(self.calls) == 1:
            return AwsResponse(
                {
                    "PriceList": [
                        json.dumps(
                            {
                                "product": {
                                    "sku": "model",
                                    "attributes": {
                                        "modelId": "amazon.nova-2-lite-v1:0",
                                        "usagetype": "USE1-Nova2.0Lite-input",
                                    },
                                },
                                "terms": {
                                    "OnDemand": {
                                        "term": {
                                            "priceDimensions": {
                                                "rate": {
                                                    "description": "input",
                                                    "unit": "Tokens",
                                                    "pricePerUnit": {
                                                        "USD": "0.000001"
                                                    },
                                                },
                                            }
                                        }
                                    }
                                },
                            }
                        )
                    ],
                    "NextToken": "next-page",
                },
                "https://api.pricing.us-east-1.amazonaws.com",
            )
        return AwsResponse(
            {"PriceList": []}, "https://api.pricing.us-east-1.amazonaws.com"
        )


def test_cost_live_is_read_only_paginated_price_evidence(
    tmp_path: Path,
) -> None:
    port = PricePort()
    result = run_cost_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=port,
        snapshot_path=tmp_path / "missing.json",
    )
    assert result["mode"] == "live_service"
    assert len(port.calls) == 3
    assert port.calls[1]["NextToken"] == "next-page"
    assert result["data"]["read_at"]
    assert all(op["effect"] == "read" for op in result["operations"])


def test_cost_live_default_session_selects_pricing_region(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, str]] = []

    class Client:
        meta = SimpleNamespace(
            endpoint_url="https://api.pricing.us-east-1.amazonaws.com"
        )

        def get_products(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["Filters"] == [
                {
                    "Type": "TERM_MATCH",
                    "Field": "regionCode",
                    "Value": "eu-west-1",
                }
            ]
            return {"PriceList": []}

    class Session:
        def client(self, service_name: str, **kwargs: object) -> Client:
            assert service_name == "pricing"
            assert kwargs["region_name"] == "us-east-1"
            return Client()

    def factory(**kwargs: str) -> Session:
        calls.append(kwargs)
        return Session()

    result = run_cost_demo(
        execution="live",
        settings=Settings(region="eu-west-1", aws_profile="example"),
        policy=ExecutionPolicy(),
        session_factory=factory,
        snapshot_path=tmp_path / "missing.json",
    )
    assert result["evidence"]["credential_source"] == "profile"
    assert calls == [{"profile_name": "example", "region_name": "us-east-1"}]
