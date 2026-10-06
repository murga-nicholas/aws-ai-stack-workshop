"""Regressions for the shapes in the measured October price snapshot."""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from awsai_demo import pricing
from awsai_demo.pricing import AmbiguousPriceError, PriceBook, PriceSnapshot

ROOT = Path(__file__).resolve().parents[1]


def product_rows(
    service: str,
    usage: str,
    unit: str,
    rate: str,
    *,
    attributes: dict[str, str] | None = None,
    description: str = "Price per input unit",
) -> list[pricing.PriceRow]:
    return pricing._rows_from_product(
        service_code=service,
        requested_region="us-east-1",
        read_at="2026-10-06T00:00:00Z",
        document={
            "product": {
                "sku": "measured-shape",
                "attributes": {"usagetype": usage, **(attributes or {})},
            },
            "terms": {
                "OnDemand": {
                    "term": {
                        "priceDimensions": {
                            "rate": {
                                "unit": unit,
                                "pricePerUnit": {"USD": rate},
                                "beginRange": "0",
                                "endRange": "Inf",
                                "description": description,
                            }
                        }
                    }
                }
            },
        },
    )


@pytest.mark.parametrize(
    ("suffix", "raw_rate", "tier", "routing", "direction", "expected"),
    [
        (
            "InputTokenCount_Global",
            "1",
            "standard",
            "cross-region-global",
            "input",
            "0.000001",
        ),
        (
            "OutputTokenCount_Global",
            "5",
            "standard",
            "cross-region-global",
            "output",
            "0.000005",
        ),
        (
            "InputTokenCount",
            "1.1",
            "standard",
            "in-region",
            "input",
            "0.0000011",
        ),
        (
            "OutputTokenCount",
            "5.5",
            "standard",
            "in-region",
            "output",
            "0.0000055",
        ),
        (
            "InputTokenCount_Batch",
            "0.55",
            "batch",
            "in-region",
            "input",
            "0.00000055",
        ),
    ],
)
def test_haiku_marketplace_service_name_and_million_token_units(
    suffix: str,
    raw_rate: str,
    tier: str,
    routing: str,
    direction: str,
    expected: str,
) -> None:
    row = product_rows(
        "AmazonBedrockFoundationModels",
        f"USE1-MP:USE1_{suffix}-Units",
        "1M tokens",
        raw_rate,
        attributes={
            "servicename": "Claude Haiku 4.5 (Amazon Bedrock Edition)",
        },
        description="AWS Marketplace software usage|us-east-1|Token usage",
    )[0]

    assert row.feature == (
        f"model:anthropic.claude-haiku-4-5-20251001-v1:0:{direction}"
    )
    assert row.usd == Decimal(expected)
    assert row.unit == "token"
    assert (row.tier, row.routing) == (tier, routing)


@pytest.mark.parametrize("prefix", ["", "USE1-", "USW2-", "EU-"])
@pytest.mark.parametrize(
    ("usage", "unit", "feature", "canonical"),
    [
        ("Request-Tier2", "Requests", "requests.tier2", "request"),
        ("TimedStorage-ByteHrs", "GB-Mo", "storage_gb_month", "gb_month"),
        ("Query-ProcessedBytes-Tier3", "GB", "query_processed_gb.tier3", "gb"),
        ("Query-Returned-Bytes", "GB", "query_returned_gb", "gb"),
        ("Put-Bytes", "GB", "put_gb", "gb"),
    ],
)
def test_vectors_usage_wins_over_generic_storage_service_name(
    prefix: str, usage: str, unit: str, feature: str, canonical: str
) -> None:
    row = product_rows(
        "AmazonS3",
        f"{prefix}Vectors-{usage}",
        unit,
        "0.000055",
        attributes={"servicename": "Amazon Simple Storage Service"},
        description="S3 Vectors billed operation",
    )[0]

    assert row.feature == f"s3vectors.{feature}"
    assert row.unit == canonical
    assert row.usd == Decimal("0.000055")
    assert (row.tier, row.routing) == ("standard", "in-region")


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ("USE1-gpt-oss-20b-input-tokens", "openai.gpt-oss-20b-1:0"),
        ("EU-gpt-oss-120b-output-tokens-batch", "openai.gpt-oss-120b-1:0"),
        (
            "USW2-openai.gpt-oss-20b-mantle-input-tokens-standard",
            "openai.gpt-oss-20b",
        ),
        ("gpt-oss-20b-input-tokens-flex", "openai.gpt-oss-20b-1:0"),
        ("USE1-openai.gpt-oss-20b-input-tokens", ""),
        ("USE1-gpt-oss-20b-mantle-input-tokens", ""),
        ("USE1-GPT-OSS-Safeguard-20B-input-tokens", ""),
        ("USE1-gpt-oss-20b-input-tokens-custom", ""),
    ],
)
def test_gpt_oss_usage_mapping_separates_native_and_mantle(
    usage: str, expected: str
) -> None:
    assert pricing._model_id({"usagetype": usage}) == expected


@pytest.mark.parametrize(
    "unit", ["1M tokens", "1,000,000 tokens", "1 million tokens"]
)
def test_per_million_units_do_not_become_per_thousand(unit: str) -> None:
    assert pricing._normalize_unit_price(
        "model:m:input", unit, Decimal("1.1")
    ) == ("token", Decimal("0.0000011"))


def comprehend_bands() -> list[pricing.PriceRow]:
    base = product_rows(
        "comprehend",
        "USE1-DetectPiiEntities",
        "Unit",
        "0.0001",
        attributes={"group": "Global-Comprehend"},
    )[0]
    return [
        replace(
            base,
            begin_range=begin,
            end_range=end,
            usd=Decimal(rate),
            rate_code=f"rate-{begin}",
        )
        for begin, end, rate in [
            ("0", "10000000", "0.0001"),
            ("10000000", "50000000", "0.00005"),
            ("50000000", "100000000", "0.000025"),
            ("100000000", "Inf", "0.000005"),
        ]
    ]


def comprehend_book(rows: list[pricing.PriceRow]) -> PriceBook:
    return PriceBook(
        PriceSnapshot(
            schema_version=1,
            pricing_region="us-east-1",
            requested_regions=("us-east-1",),
            read_at="2026-10-06",
            service_codes=("comprehend",),
            rows=tuple(rows),
            missing=(),
            errors=(),
        )
    )


def lookup_comprehend(book: PriceBook) -> Decimal:
    return book.rate(
        "comprehend.detect_pii_entities.units",
        region="us-east-1",
        unit="unit",
        tier="standard",
        routing="in-region",
    )


def test_comprehend_volume_bands_use_highest_rate_without_global_routing() -> (
    None
):
    rows = comprehend_bands()
    assert {row.routing for row in rows} == {"in-region"}
    assert lookup_comprehend(comprehend_book(list(reversed(rows)))) == Decimal(
        "0.0001"
    )

    legacy = {**rows[0].to_json(), "routing": "cross-region-global"}
    loaded = pricing._loaded_row(legacy)
    assert loaded.routing == "in-region"


@pytest.mark.parametrize(
    "mutation",
    [
        {"sku": "different-product"},
        {"begin_range": "9999999"},
        {"begin_range": "not-a-number"},
        {"begin_range": "NaN"},
        {"end_range": "10000000"},
        {"end_range": "99999999"},
    ],
)
def test_comprehend_refuses_conflicting_or_incomplete_bands(
    mutation: dict[str, str],
) -> None:
    rows = comprehend_bands()
    rows[-1] = replace(rows[-1], **mutation)
    with pytest.raises(AmbiguousPriceError):
        lookup_comprehend(comprehend_book(rows))


def test_comprehend_finite_volume_schedule_does_not_prove_upper_bound() -> (
    None
):
    rows = comprehend_bands()
    rows[-1] = replace(rows[-1], end_range="200000000")
    with pytest.raises(AmbiguousPriceError):
        lookup_comprehend(comprehend_book(rows))


def test_committed_guardrail_rows_use_the_highest_text_unit() -> None:
    book = PriceBook.from_file(ROOT / "data" / "pricing_snapshot.json")
    dimensions = {
        "region": "us-east-1",
        "tier": "standard",
        "routing": "in-region",
        "unit": "text_unit",
    }
    for feature in (
        "guardrails.content_filter",
        "guardrails.denied_topic",
        "guardrails.sensitive_information",
    ):
        matches = [
            row
            for row in book.snapshot.rows
            if row.feature == feature
            and row.region == dimensions["region"]
            and row.tier == dimensions["tier"]
            and row.routing == dimensions["routing"]
            and row.unit == dimensions["unit"]
        ]
        assert matches
        chosen = book.lookup_row(feature=feature, **dimensions)
        assert chosen.usd == max(row.usd for row in matches)
        assert chosen.usd > 0
    for feature in (
        "guardrails.content_filter",
        "guardrails.sensitive_information",
    ):
        rows = [
            row
            for row in book.snapshot.rows
            if row.feature == feature and row.region == "us-east-1"
        ]
        assert len({row.sku for row in rows}) > 1
        assert min(row.usd for row in rows) < max(row.usd for row in rows)


def test_haiku_unknown_or_cache_marketplace_product_remains_unpriced() -> None:
    for service_name, usage in (
        ("Different Model (Amazon Bedrock Edition)", "InputTokenCount"),
        (
            "Claude Haiku 4.5 (Amazon Bedrock Edition)",
            "CacheReadInputTokenCount",
        ),
    ):
        row = product_rows(
            "AmazonBedrockFoundationModels",
            f"USE1-MP:USE1_{usage}-Units",
            "1M tokens",
            "1",
            attributes={"servicename": service_name},
        )[0]
        assert row.feature == "unknown"
