from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EndpointConnectionError,
)

import awsai_demo.pricing as pricing
from awsai_demo.policy import ReservationUnavailable
from awsai_demo.pricing import (
    AmbiguousPriceError,
    PriceBook,
    PriceIssue,
    PriceNotFoundError,
    PriceRow,
    PriceSnapshot,
    PricingError,
    SnapshotError,
    build_price_snapshot,
    load_snapshot,
    write_snapshot,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class FakePricingClient:
    def __init__(
        self,
        *,
        describe_pages: list[Mapping[str, Any]] | None = None,
        describe_error: ClientError | BotoCoreError | OSError | None = None,
        products: dict[
            tuple[str, str],
            list[Mapping[str, Any] | ClientError | BotoCoreError | OSError],
        ],
    ) -> None:
        self.describe_pages = list(describe_pages or [])
        self.describe_error = describe_error
        self.products = {key: list(value) for key, value in products.items()}
        self.describe_calls: list[dict[str, Any]] = []
        self.product_calls: list[dict[str, Any]] = []

    def describe_services(self, **kwargs: Any) -> Mapping[str, Any]:
        self.describe_calls.append(kwargs)
        if self.describe_error is not None:
            raise self.describe_error
        return self.describe_pages.pop(0)

    def get_products(self, **kwargs: Any) -> Mapping[str, Any]:
        self.product_calls.append(kwargs)
        region = kwargs["Filters"][0]["Value"]
        pages = self.products[(kwargs["ServiceCode"], region)]
        page = pages.pop(0)
        if isinstance(page, (ClientError, BotoCoreError, OSError)):
            raise page
        return page


def test_price_snapshot_parses_features_and_exact_rates(
    tmp_path: Path,
) -> None:
    client = _full_client()

    snapshot = build_price_snapshot(
        client,
        regions=["us-east-1", "us-east-1"],
        clock=lambda: datetime(2026, 10, 6, tzinfo=UTC),
    )
    path = tmp_path / "pricing_snapshot.json"
    write_snapshot(snapshot, path)
    serialized = path.read_text(encoding="utf-8")
    loaded = load_snapshot(path)
    book = PriceBook.from_file(path)

    assert "123456789012" not in serialized
    assert "AKIAABCDEFGHIJKLMNOP" not in serialized
    assert "0.000002" in serialized
    assert snapshot.requested_regions == ("us-east-1",)
    assert client.describe_calls == [
        {"MaxResults": 100},
        {"MaxResults": 100, "NextToken": "more"},
    ]
    assert book.lookup(
        feature="model:amazon.nova-2-lite-v1:0:input",
        region="us-east-1",
        unit="token",
        tier="standard",
        routing="in-region",
    ) == Decimal("0.000002")
    assert book.rate(
        "model:global.amazon.nova-2-lite-v1:0:output",
        region="us-east-1",
        unit="token",
        tier="priority",
        routing="cross-region-global",
    ) == Decimal("0.000004")
    assert book.rate(
        "guardrails.denied_topic",
        region="us-east-1",
        unit="text_unit",
        tier="standard",
        routing="in-region",
    ) == Decimal("0.0000003")
    assert book.rate(
        "agentcore.runtime.cpu_seconds",
        region="us-east-1",
        unit="second",
        tier="standard",
        routing="in-region",
    ) == Decimal("0.00001")
    assert book.rate(
        "agentcore.runtime.memory_gb_seconds",
        region="us-east-1",
        unit="gb_second",
        tier="standard",
        routing="in-region",
    ) == Decimal("0.000016")
    assert book.rate(
        "memory.events",
        region="us-east-1",
        unit="event",
        tier="standard",
        routing="in-region",
    ) == Decimal("0.000001")
    assert book.rate(
        "s3vectors.storage_gb_month",
        region="us-east-1",
        unit="gb_month",
        tier="standard",
        routing="in-region",
    ) == Decimal("0.25")
    assert book.rate(
        "translate.characters",
        region="us-east-1",
        unit="character",
        tier="standard",
        routing="in-region",
    ) == Decimal("0.000015")
    unknown = [row for row in loaded.rows if row.feature == "unknown"]
    assert not unknown
    assert loaded.issue_counts()["unknown_feature"] > 0
    assert snapshot.unmatched_rows[0].raw_unit == "Requests"
    assert SnapshotError("svc", "op", "msg").to_json() == {
        "service_code": "svc",
        "operation": "op",
        "message": "msg",
    }
    assert (
        SnapshotError.from_json(
            {"service_code": "svc", "operation": "op", "message": "msg"}
        ).operation
        == "op"
    )
    assert loaded.to_json()["source"] == {
        "api": "AWS Price List DescribeServices/GetProducts",
        "endpoint_region": "us-east-1",
    }


def test_price_book_refuses_missing_unknown_and_ambiguous_rows() -> None:
    row = _row(
        feature="s3vectors.requests",
        unit="request",
        usd=Decimal("0.01"),
        sku="A",
        rate_code="A.rate",
    )
    duplicate = _row(
        feature="s3vectors.requests",
        unit="request",
        usd=Decimal("0.02"),
        sku="B",
        rate_code="B.rate",
    )
    unknown = _row(feature="unknown", unit="Mystery")
    book = PriceBook(
        _snapshot(rows=(row, duplicate, unknown), missing=(), errors=())
    )

    with pytest.raises(AmbiguousPriceError):
        book.rate(
            "s3vectors.requests",
            region="us-east-1",
            unit="request",
            tier="standard",
            routing="in-region",
        )
    with pytest.raises(PriceNotFoundError):
        book.rate(
            "unknown",
            region="us-east-1",
            unit="Mystery",
            tier="standard",
            routing="in-region",
        )
    with pytest.raises(ReservationUnavailable):
        book.rate(
            "polly.characters",
            region="us-east-1",
            unit="character",
            tier="standard",
            routing="in-region",
        )


def test_generic_s3_request_is_not_classified_as_s3_vectors() -> None:
    assert (
        pricing._feature("AmazonS3", {}, "Storage request", "Request")
        == "unknown"
    )


def test_actual_service_codes_units_and_vectors_tiers() -> None:
    selected = pricing._select_service_codes(
        ["comprehend", "comprehendmedical", "translate", "AmazonS3"]
    )
    assert selected["comprehend"] == ("comprehend",)
    assert selected["translate"] == ("translate",)
    cases = [
        (
            "comprehend",
            "USE1-DetectPiiEntities",
            "Unit",
            "comprehend.detect_pii_entities.units",
            "unit",
        ),
        (
            "comprehend",
            "USE1-DetectSentiment",
            "Unit",
            "comprehend.detect_sentiment.units",
            "unit",
        ),
        (
            "translate",
            "USE1-TextTranslation",
            "Character",
            "translate.characters",
            "character",
        ),
        (
            "AmazonS3",
            "Vectors-TimedStorage-ByteHrs",
            "GB-Mo",
            "s3vectors.storage_gb_month",
            "gb_month",
        ),
        ("AmazonS3", "Vectors-Put-Bytes", "GB", "s3vectors.put_gb", "gb"),
        (
            "AmazonS3",
            "Vectors-Query-Returned-Bytes",
            "GB",
            "s3vectors.query_returned_gb",
            "gb",
        ),
    ]
    cases += [
        (
            "AmazonS3",
            f"Vectors-Request-{tier}",
            "Requests",
            f"s3vectors.requests.{tier.lower()}",
            "request",
        )
        for tier in ("Tier1", "Tier2", "Tier3")
    ]
    cases += [
        (
            "AmazonS3",
            f"Vectors-Query-ProcessedBytes-{tier}",
            "GB",
            f"s3vectors.query_processed_gb.{tier.lower()}",
            "gb",
        )
        for tier in ("Tier1", "Tier2", "Tier3")
    ]
    for service, usage, raw_unit, feature, unit in cases:
        description = "Vectors storage" if "Storage" in usage else usage
        attrs = {"usagetype": usage}
        observed = pricing._feature(service, attrs, description, raw_unit)
        assert observed == feature
        assert pricing._normalize_unit_price(
            observed, raw_unit, Decimal("0.01")
        ) == (unit, Decimal("0.01"))
        assert pricing._tier(attrs, description) == "standard"
    for service, usage in (
        ("translate", "USE1-TextTranslationJob"),
        ("translate", "USE1-DocumentTranslation"),
        ("AmazonPolly", "NeuralCharacters"),
    ):
        assert (
            pricing._feature(service, {"usagetype": usage}, "", "Character")
            == "unknown"
        )


def test_snapshot_compact_summary_review_and_legacy_loading(
    tmp_path: Path,
) -> None:
    snapshot = build_price_snapshot(_full_client(), regions=["us-east-1"])
    path = tmp_path / "review.pricing-review.json"
    pricing.write_review(snapshot, path)
    serialized = path.read_text(encoding="utf-8")
    assert "123456789012" not in serialized
    assert "AKIAABCDEFGHIJKLMNOP" not in serialized
    assert "arn:aws:iam::<account>:<redacted>" in serialized
    assert (
        len(json.loads(serialized)["rows"])
        == snapshot.issue_counts()["unknown_feature"]
    )
    assert not any(row.feature == "unknown" for row in snapshot.rows)
    document = snapshot.to_json()
    assert document["missing_summary"]["counts_by_kind"]["unknown_feature"] > 0
    assert not any(
        issue["kind"] == "unknown_feature" for issue in document["missing"]
    )
    legacy = dict(document)
    legacy.pop("missing_summary")
    nova = _row(feature="model:Nova 2.0 Lite:input").to_json()
    nova.update(
        service_code="AmazonBedrock",
        usage_type="USE1-Nova2.0Lite-input-tokens",
    )
    legacy["rows"] = [nova]
    loaded = PriceSnapshot.from_json(legacy)
    assert loaded.rows[0].feature == "model:amazon.nova-2-lite-v1:0:input"
    nova["usage_type"] = "USE1-Nova2.0Lite-cache-read-input-tokens"
    assert PriceSnapshot.from_json(legacy).rows[0].feature == "unknown"
    assert "agentcore.harness.invocations" not in str(
        pricing._REQUIRED_FEATURES
    )


def test_unknown_price_variants_fail_closed() -> None:
    assert (
        pricing._feature("AmazonBedrock", {}, "Guardrail word policy", "Text")
        == "unknown"
    )
    assert (
        pricing._feature(
            "AmazonBedrockAgentCore",
            {},
            "AgentCore Harness capacity",
            "Request",
        )
        == "unknown"
    )
    assert (
        pricing._feature(
            "AmazonBedrockAgentCore",
            {},
            "AgentCore Harness invocation request",
            "Request",
        )
        == "unknown"
    )
    assert pricing._routing({}, "custom routing variant") == "unknown"
    assert pricing._routing({}, "cross-region inference") == "cross-region"
    assert pricing._tier({"serviceTier": "premium"}, "") == "unknown"
    assert pricing._tier({}, "premium tier") == "unknown"


def test_snapshot_records_missing_services_and_product_errors() -> None:
    denied = ClientError(
        {
            "Error": {"Code": "AccessDeniedException"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        "GetProducts",
    )
    client = FakePricingClient(
        describe_pages=[
            {
                "Services": [
                    {"ServiceCode": "AmazonBedrock"},
                    {"ServiceCode": "AmazonComprehend"},
                ]
            }
        ],
        products={
            ("AmazonBedrock", "us-east-1"): [
                {
                    "PriceList": [
                        _product(
                            "AmazonBedrock",
                            "GOOD",
                            {"modelId": "amazon.nova-2-lite-v1:0"},
                            "1K tokens",
                            "0.002",
                            "standard input tokens",
                        ),
                        json.dumps({"product": {"sku": "BROKEN"}}),
                        "not-json",
                    ]
                }
            ],
            ("AmazonComprehend", "us-east-1"): [denied],
        },
    )

    snapshot = build_price_snapshot(
        client,
        regions=["us-east-1"],
        clock=lambda: "fixed",
    )

    assert snapshot.rows[0].sku == "GOOD"
    assert snapshot.errors[0].operation == "ParseProduct"
    assert snapshot.errors[1].operation == "ParseProduct"
    assert snapshot.errors[2].operation == "GetProducts"
    assert any(issue.kind == "missing_service" for issue in snapshot.missing)
    assert any(issue.kind == "missing_feature" for issue in snapshot.missing)


def test_snapshot_records_describe_services_error() -> None:
    denied = ClientError(
        {
            "Error": {"Code": "AccessDeniedException"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        "DescribeServices",
    )
    client = FakePricingClient(describe_error=denied, products={})

    snapshot = build_price_snapshot(client, regions=["us-east-1"])

    assert snapshot.service_codes == ()
    assert snapshot.errors[0].operation == "DescribeServices"
    assert {issue.kind for issue in snapshot.missing} == {"missing_service"}


def test_snapshot_records_transport_errors() -> None:
    transport = EndpointConnectionError(endpoint_url="https://pricing.test")
    describe = FakePricingClient(describe_error=transport, products={})

    describe_snapshot = build_price_snapshot(
        describe,
        regions=["us-east-1"],
    )

    assert describe_snapshot.errors[0].operation == "DescribeServices"

    products = FakePricingClient(
        describe_pages=[{"Services": [{"ServiceCode": "AmazonBedrock"}]}],
        products={("AmazonBedrock", "us-east-1"): [transport]},
    )

    products_snapshot = build_price_snapshot(
        products,
        regions=["us-east-1"],
    )

    assert products_snapshot.errors[0].operation == "GetProducts"


def test_snapshot_json_validation_errors() -> None:
    with pytest.raises(PricingError, match="kind"):
        PriceIssue.from_json(
            {
                "kind": "bogus",
                "feature": "x",
                "region": "us-east-1",
                "message": "bad",
            }
        )
    with pytest.raises(PricingError, match="USD"):
        PriceRow.from_json({**_row().to_json(), "usd": "not-decimal"})
    with pytest.raises(PricingError, match="finite"):
        _row(usd=Decimal("NaN"))
    with pytest.raises(PricingError, match="raw_usd"):
        _row(raw_usd=Decimal("-1"))
    with pytest.raises(PricingError, match="schema"):
        _snapshot(rows=(), missing=(), errors=(), schema_version=99)
    with pytest.raises(PricingError, match="requested_regions"):
        _snapshot(rows=(), missing=(), errors=(), regions=())
    with pytest.raises(PricingError, match="object"):
        PriceSnapshot.from_json(
            {
                "schema_version": 1,
                "pricing_region": "us-east-1",
                "requested_regions": ["us-east-1"],
                "read_at": "now",
                "service_codes": [],
                "rows": [1],
                "missing": [],
                "errors": [],
            }
        )
    with pytest.raises(PricingError, match="integer"):
        PriceSnapshot.from_json(
            {
                "schema_version": "1",
                "pricing_region": "us-east-1",
                "requested_regions": ["us-east-1"],
                "read_at": "now",
                "service_codes": [],
                "rows": [],
                "missing": [],
                "errors": [],
            }
        )
    with pytest.raises(PricingError, match="service_code"):
        SnapshotError.from_json(
            {"service_code": 1, "operation": "op", "message": "m"}
        )
    with pytest.raises(PricingError, match="rows"):
        PriceSnapshot.from_json(
            {
                "schema_version": 1,
                "pricing_region": "us-east-1",
                "requested_regions": ["us-east-1"],
                "read_at": "now",
                "service_codes": [],
                "rows": {},
                "missing": [],
                "errors": [],
            }
        )
    with pytest.raises(PricingError, match="requested_regions"):
        PriceSnapshot.from_json(
            {
                "schema_version": 1,
                "pricing_region": "us-east-1",
                "requested_regions": [1],
                "read_at": "now",
                "service_codes": [],
                "rows": [],
                "missing": [],
                "errors": [],
            }
        )


def _full_client() -> FakePricingClient:
    services = [
        {"ServiceCode": "AmazonBedrock"},
        {"ServiceCode": "AmazonBedrockAgentCore"},
        {"ServiceCode": "AmazonS3Vectors"},
        {"ServiceCode": "AmazonComprehend"},
        {"ServiceCode": "AmazonTranslate"},
        {"ServiceCode": "AmazonPolly"},
    ]
    products: dict[
        tuple[str, str],
        list[Mapping[str, Any] | ClientError | BotoCoreError | OSError],
    ] = {
        ("AmazonBedrock", "us-east-1"): [
            {
                "PriceList": [
                    _product(
                        "AmazonBedrock",
                        "BR-IN",
                        {"modelId": "amazon.nova-2-lite-v1:0"},
                        "1K tokens",
                        "0.002",
                        "standard input tokens",
                    ),
                    _product(
                        "AmazonBedrock",
                        "BR-OUT",
                        {"modelId": "amazon.nova-2-lite-v1:0"},
                        "Tokens",
                        "0.000004",
                        "global priority output tokens",
                    ),
                ],
                "NextToken": "bedrock-more",
            },
            {
                "PriceList": [
                    _product(
                        "AmazonBedrock",
                        "GR-CONTENT",
                        {},
                        "1K text units",
                        "0.0002",
                        "Guardrail content filter text units",
                    ),
                    _product(
                        "AmazonBedrock",
                        "GR-TOPIC",
                        {},
                        "Text unit",
                        "0.0000003",
                        "Guardrail denied topic text unit",
                    ),
                    _product(
                        "AmazonBedrock",
                        "GR-PII",
                        {},
                        "Text unit",
                        "0.0000004",
                        "Guardrail sensitive information text unit",
                    ),
                    _product(
                        "AmazonBedrock",
                        "BR-UNKNOWN",
                        {"modelId": "amazon.nova-2-lite-v1:0"},
                        "Requests",
                        "1",
                        "input token unit not supported",
                    ),
                    _product(
                        "AmazonBedrock",
                        "BR-MODELNAME",
                        {"modelName": "anthropic.claude-v1"},
                        "Tokens",
                        "0.0001",
                        "input tokens via modelName",
                    ),
                    _product(
                        "AmazonBedrock",
                        "BR-NO-DIR",
                        {"modelId": "amazon.nova-2-lite-v1:0"},
                        "Tokens",
                        "0.0001",
                        "token unit without direction",
                    ),
                    _product(
                        "AmazonBedrock",
                        "BR-NO-MODEL",
                        {},
                        "Tokens",
                        "0.0001",
                        "input tokens without model id",
                    ),
                    _empty_product("AmazonBedrock", "BR-EMPTY"),
                ]
            },
        ],
        ("AmazonBedrockAgentCore", "us-east-1"): [
            {
                "PriceList": [
                    _product(
                        "AmazonBedrockAgentCore",
                        "AC-CPU",
                        {},
                        "vCPU-Hours",
                        "0.036",
                        "AgentCore Runtime CPU hours",
                    ),
                    _product(
                        "AmazonBedrockAgentCore",
                        "AC-MEM",
                        {},
                        "GB-Hours",
                        "0.0576",
                        "AgentCore Runtime memory GB-Hours",
                    ),
                    _product(
                        "AmazonBedrockAgentCore",
                        "AC-HAR",
                        {},
                        "1K Requests",
                        "1",
                        "AgentCore Harness request",
                    ),
                    _product(
                        "AmazonBedrockAgentCore",
                        "AC-EVT",
                        {},
                        "1000 Events",
                        "0.001",
                        "AgentCore Memory event",
                    ),
                    _product(
                        "AmazonBedrockAgentCore",
                        "AC-CODE-CPU",
                        {},
                        "Second",
                        "0.00002",
                        "AgentCore Code Interpreter CPU seconds",
                    ),
                    _product(
                        "AmazonBedrockAgentCore",
                        "AC-CODE-MEM",
                        {},
                        "GB-second",
                        "0.00003",
                        "AgentCore Code Interpreter memory GB-second",
                    ),
                    _product(
                        "AmazonBedrockAgentCore",
                        "AC-UNK",
                        {},
                        "Mystery",
                        "1",
                        "AgentCore unknown unit",
                    ),
                ]
            }
        ],
        ("AmazonS3Vectors", "us-east-1"): [
            {
                "PriceList": [
                    _product(
                        "AmazonS3Vectors",
                        "S3V-REQ",
                        {},
                        "Request",
                        "0.0001",
                        "Vector query request",
                    ),
                    _product(
                        "AmazonS3Vectors",
                        "S3V-STO",
                        {},
                        "GB-Month",
                        "0.25",
                        "Vector storage GB-Month",
                    ),
                    _product(
                        "AmazonS3Vectors",
                        "S3V-UNK",
                        {},
                        "Thing",
                        "1",
                        "Vector odd unit arn:aws:iam::123456789012:role/demo "
                        "AKIAABCDEFGHIJKLMNOP",
                    ),
                    _product(
                        "AmazonS3Vectors",
                        "S3V-NOVEC",
                        {},
                        "Request",
                        "1",
                        "Storage request without keyword",
                    ),
                ]
            }
        ],
        ("AmazonComprehend", "us-east-1"): [
            {
                "PriceList": [
                    _product(
                        "AmazonComprehend",
                        "COMP",
                        {},
                        "Characters",
                        "0.0000001",
                        "Comprehend characters",
                    )
                ]
            }
        ],
        ("AmazonTranslate", "us-east-1"): [
            {
                "PriceList": [
                    _product(
                        "AmazonTranslate",
                        "TRANS",
                        {},
                        "1K characters",
                        "0.015",
                        "TranslateText characters",
                    )
                ]
            }
        ],
        ("AmazonPolly", "us-east-1"): [
            {
                "PriceList": [
                    _product(
                        "AmazonPolly",
                        "POLLY",
                        {},
                        "Characters",
                        "0.000004",
                        "Polly characters",
                    )
                ]
            }
        ],
    }
    return FakePricingClient(
        describe_pages=[
            {"Services": services[:2], "NextToken": "more"},
            {"Services": services[2:]},
        ],
        products=products,
    )


def _product(
    service_code: str,
    sku: str,
    attrs: Mapping[str, str],
    unit: str,
    usd: str,
    description: str,
) -> str:
    document = {
        "product": {
            "sku": sku,
            "productFamily": "AI",
            "attributes": {
                "servicecode": service_code,
                "regionCode": "us-east-1",
                "usagetype": f"USE1-{sku}",
                **attrs,
            },
        },
        "terms": {
            "OnDemand": {
                f"{sku}.term": {
                    "priceDimensions": {
                        f"{sku}.rate": {
                            "unit": unit,
                            "pricePerUnit": {"USD": usd},
                            "beginRange": "0",
                            "endRange": "Inf",
                            "description": description,
                        }
                    }
                }
            }
        },
    }
    return json.dumps(document)


def _snapshot(
    *,
    rows: tuple[PriceRow, ...],
    missing: tuple[PriceIssue, ...],
    errors: tuple[SnapshotError, ...],
    schema_version: int = 1,
    regions: tuple[str, ...] = ("us-east-1",),
) -> PriceSnapshot:
    return PriceSnapshot(
        schema_version=schema_version,
        pricing_region="us-east-1",
        requested_regions=regions,
        read_at="now",
        service_codes=("AmazonS3Vectors",),
        rows=rows,
        missing=missing,
        errors=errors,
    )


def _row(
    *,
    feature: str = "model:m:input",
    unit: str = "token",
    usd: Decimal = Decimal("0.01"),
    raw_usd: Decimal | None = None,
    sku: str = "SKU",
    rate_code: str = "SKU.rate",
) -> PriceRow:
    return PriceRow(
        service_code="svc",
        feature=feature,
        region="us-east-1",
        usage_type="USE1-Test",
        tier="standard",
        routing="in-region",
        unit=unit,
        usd=usd,
        raw_unit=unit,
        raw_usd=usd if raw_usd is None else raw_usd,
        read_at="now",
        sku=sku,
        rate_code=rate_code,
        begin_range="0",
        end_range="Inf",
        description="test",
    )


def _empty_product(service_code: str, sku: str) -> str:
    return json.dumps(
        {
            "product": {
                "sku": sku,
                "productFamily": "AI",
                "attributes": {
                    "servicecode": service_code,
                    "regionCode": "us-east-1",
                    "usagetype": f"USE1-{sku}",
                },
            },
            "terms": {"OnDemand": {f"{sku}.term": {"priceDimensions": {}}}},
        }
    )
