"""Conservative Price List snapshot loading and matching."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from botocore.exceptions import BotoCoreError, ClientError

from awsai_demo.policy import ReservationUnavailable
from awsai_demo.redact import redact, sanitize_exception

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

PricingIssueKind = Literal[
    "missing_service",
    "missing_feature",
    "unknown_feature",
]
PRICING_REGION = "us-east-1"
SCHEMA_VERSION = 1
UNKNOWN_FEATURE = "unknown"
_SNAPSHOT_READ_ERRORS = (ClientError, BotoCoreError, OSError)

_SERVICE_GROUP_CANDIDATES: dict[str, tuple[str, ...]] = {
    "bedrock": ("AmazonBedrock", "AmazonBedrockFoundationModels"),
    "agentcore": (
        "AmazonBedrockAgentCore",
        "AmazonBedrockAgentCoreRuntime",
        "AWSAgentCore",
    ),
    "s3vectors": ("AmazonS3Vectors", "AmazonS3"),
    "comprehend": ("comprehend", "AmazonComprehend", "Comprehend"),
    "translate": ("translate", "AmazonTranslate"),
    "polly": ("AmazonPolly",),
}
_REQUIRED_FEATURES: dict[str, tuple[str, ...]] = {
    "bedrock": (
        "model:*:input",
        "model:*:output",
        "guardrails.content_filter",
        "guardrails.denied_topic",
        "guardrails.sensitive_information",
    ),
    "agentcore": (
        "agentcore.runtime.cpu_seconds",
        "agentcore.runtime.memory_gb_seconds",
        "memory.events",
        "code_interpreter.cpu_seconds",
        "code_interpreter.memory_gb_seconds",
    ),
    "s3vectors": (
        "s3vectors.requests.tier1",
        "s3vectors.requests.tier2",
        "s3vectors.requests.tier3",
        "s3vectors.storage_gb_month",
        "s3vectors.put_gb",
        "s3vectors.query_processed_gb.tier1",
        "s3vectors.query_returned_gb",
    ),
    "comprehend": (
        "comprehend.detect_pii_entities.units",
        "comprehend.detect_sentiment.units",
    ),
    "translate": ("translate.characters",),
    "polly": ("polly.characters",),
}


class PricingClient(Protocol):
    """Small AWS Price List client surface used by the snapshotter."""

    def describe_services(self, **kwargs: Any) -> Mapping[str, Any]:
        """Return Price List service metadata."""

    def get_products(self, **kwargs: Any) -> Mapping[str, Any]:
        """Return Price List product pages."""


class PricingError(ReservationUnavailable):
    """Base class for local pricing failures."""


class PriceNotFoundError(PricingError):
    """Raised when a conservative price lookup has no row."""


class AmbiguousPriceError(PricingError):
    """Raised when a price lookup matches multiple distinct rates."""


PriceNotFound = PriceNotFoundError
AmbiguousPrice = AmbiguousPriceError


@dataclass(frozen=True, slots=True)
class PriceRow:
    """One OnDemand price dimension for reservation lookup."""

    service_code: str
    feature: str
    region: str
    usage_type: str
    tier: str
    routing: str
    unit: str
    usd: Decimal
    raw_unit: str
    raw_usd: Decimal
    read_at: str
    sku: str
    rate_code: str
    begin_range: str
    end_range: str
    description: str

    def __post_init__(self) -> None:
        """Validate finite non-negative USD values."""
        if not self.usd.is_finite() or self.usd < 0:
            message = "usd must be finite and non-negative"
            raise PricingError(message)
        if not self.raw_usd.is_finite() or self.raw_usd < 0:
            message = "raw_usd must be finite and non-negative"
            raise PricingError(message)

    def to_json(self) -> dict[str, str]:
        """Return a stable JSON representation."""
        return {
            "service_code": self.service_code,
            "feature": self.feature,
            "region": self.region,
            "usage_type": self.usage_type,
            "tier": self.tier,
            "routing": self.routing,
            "unit": self.unit,
            "usd": str(self.usd),
            "raw_unit": self.raw_unit,
            "raw_usd": str(self.raw_usd),
            "read_at": self.read_at,
            "sku": self.sku,
            "rate_code": self.rate_code,
            "begin_range": self.begin_range,
            "end_range": self.end_range,
            "description": self.description,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> PriceRow:
        """Load a price row, preserving Decimal precision."""
        return cls(
            service_code=_required_str(data, "service_code"),
            feature=_required_str(data, "feature"),
            region=_required_str(data, "region"),
            usage_type=_required_str(data, "usage_type"),
            tier=_required_str(data, "tier"),
            routing=_required_str(data, "routing"),
            unit=_required_str(data, "unit"),
            usd=_decimal(_required_str(data, "usd")),
            raw_unit=_required_str(data, "raw_unit"),
            raw_usd=_decimal(_required_str(data, "raw_usd")),
            read_at=_required_str(data, "read_at"),
            sku=_required_str(data, "sku"),
            rate_code=_required_str(data, "rate_code"),
            begin_range=_required_str(data, "begin_range"),
            end_range=_required_str(data, "end_range"),
            description=_required_str(data, "description"),
        )


@dataclass(frozen=True, slots=True)
class PriceIssue:
    """A missing or unclassified unit found while snapshotting."""

    kind: PricingIssueKind
    service_code: str | None
    feature: str
    region: str
    message: str

    def to_json(self) -> dict[str, str | None]:
        """Return a stable JSON representation."""
        return {
            "kind": self.kind,
            "service_code": self.service_code,
            "feature": self.feature,
            "region": self.region,
            "message": self.message,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> PriceIssue:
        """Load an issue from JSON data."""
        kind = _required_str(data, "kind")
        if kind not in {
            "missing_service",
            "missing_feature",
            "unknown_feature",
        }:
            message = f"unknown price issue kind {kind!r}"
            raise PricingError(message)
        service_code = data.get("service_code")
        return cls(
            kind=cast("PricingIssueKind", kind),
            service_code=service_code
            if isinstance(service_code, str)
            else None,
            feature=_required_str(data, "feature"),
            region=_required_str(data, "region"),
            message=_required_str(data, "message"),
        )


@dataclass(frozen=True, slots=True)
class SnapshotError:
    """A redacted non-fatal AWS Price List read error."""

    service_code: str
    operation: str
    message: str

    def to_json(self) -> dict[str, str]:
        """Return a stable JSON representation."""
        return {
            "service_code": self.service_code,
            "operation": self.operation,
            "message": self.message,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> SnapshotError:
        """Load an error from JSON data."""
        return cls(
            service_code=_required_str(data, "service_code"),
            operation=_required_str(data, "operation"),
            message=_required_str(data, "message"),
        )


@dataclass(frozen=True, slots=True)
class PriceSnapshot:
    """A generated Price List snapshot for live reservation checks."""

    schema_version: int
    pricing_region: str
    requested_regions: tuple[str, ...]
    read_at: str
    service_codes: tuple[str, ...]
    rows: tuple[PriceRow, ...]
    missing: tuple[PriceIssue, ...]
    errors: tuple[SnapshotError, ...]
    unmatched_rows: tuple[PriceRow, ...] = ()
    unknown_count: int = 0

    def __post_init__(self) -> None:
        """Validate the snapshot envelope."""
        if self.schema_version != SCHEMA_VERSION:
            message = "unsupported pricing snapshot schema version"
            raise PricingError(message)
        if not self.requested_regions:
            message = "requested_regions must not be empty"
            raise PricingError(message)

    def to_json(self) -> dict[str, object]:
        """Return a stable JSON document."""
        return {
            "schema_version": self.schema_version,
            "pricing_region": self.pricing_region,
            "requested_regions": list(self.requested_regions),
            "read_at": self.read_at,
            "service_codes": list(self.service_codes),
            "rows": [
                row.to_json()
                for row in self.rows
                if row.feature != UNKNOWN_FEATURE
            ],
            "missing": [
                issue.to_json()
                for issue in self.missing
                if issue.kind != "unknown_feature"
            ],
            "missing_summary": {
                "counts_by_kind": self.issue_counts(),
                "required_but_missing": [
                    issue.to_json()
                    for issue in self.missing
                    if issue.kind != "unknown_feature"
                ],
            },
            "errors": [error.to_json() for error in self.errors],
            "source": {
                "api": "AWS Price List DescribeServices/GetProducts",
                "endpoint_region": self.pricing_region,
            },
        }

    def issue_counts(self) -> dict[str, int]:
        """Count unmatched dimensions without copying their payloads."""
        counts: Counter[str] = Counter(issue.kind for issue in self.missing)
        counts["unknown_feature"] = max(
            counts["unknown_feature"],
            self.unknown_count,
            len(self.unmatched_rows),
        )
        return dict(counts)

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> PriceSnapshot:
        """Load a snapshot from a JSON mapping."""
        rows = _required_sequence(data, "rows")
        missing = _required_sequence(data, "missing")
        errors = _required_sequence(data, "errors")
        return cls(
            schema_version=_required_int(data, "schema_version"),
            pricing_region=_required_str(data, "pricing_region"),
            requested_regions=tuple(_str_sequence(data, "requested_regions")),
            read_at=_required_str(data, "read_at"),
            service_codes=tuple(_str_sequence(data, "service_codes")),
            rows=tuple(_loaded_row(_mapping(item)) for item in rows),
            missing=tuple(
                PriceIssue.from_json(_mapping(item)) for item in missing
            ),
            errors=tuple(
                SnapshotError.from_json(_mapping(item)) for item in errors
            ),
            unknown_count=_unknown_count(data),
        )


class PriceBook:
    """Conservative exact-match lookup over a price snapshot."""

    def __init__(self, snapshot: PriceSnapshot) -> None:
        """Store the immutable snapshot."""
        self.snapshot = snapshot

    @classmethod
    def from_file(cls, path: Path) -> PriceBook:
        """Load a price book from a JSON snapshot file."""
        return cls(load_snapshot(path))

    def rate(
        self,
        feature: str,
        *,
        region: str,
        unit: str,
        tier: str,
        routing: str,
    ) -> Decimal:
        """Return one exact conservative USD rate or raise."""
        return self.lookup_row(
            feature=feature,
            region=region,
            tier=tier,
            routing=routing,
            unit=unit,
        ).usd

    def lookup(
        self,
        *,
        feature: str,
        region: str,
        tier: str,
        routing: str,
        unit: str,
    ) -> Decimal:
        """Return one exact conservative USD rate or raise."""
        return self.rate(
            feature,
            region=region,
            tier=tier,
            routing=routing,
            unit=unit,
        )

    def lookup_row(
        self,
        *,
        feature: str,
        region: str,
        tier: str,
        routing: str,
        unit: str,
    ) -> PriceRow:
        """Return one exact row or raise on unsafe matches."""
        if feature == UNKNOWN_FEATURE:
            message = "unknown feature rows are not reservable"
            raise PriceNotFoundError(message)
        feature_candidates = _feature_candidates(feature, routing)
        matches = [
            row
            for row in self.snapshot.rows
            if row.feature in feature_candidates
            and row.region == region
            and row.tier == tier
            and row.routing == routing
            and row.unit == unit
        ]
        if not matches:
            message = f"missing price row for {feature} in {region}"
            raise PriceNotFoundError(message)
        unique = {
            (row.usd, row.sku, row.rate_code, row.begin_range, row.end_range)
            for row in matches
        }
        if len(unique) != 1:
            conservative = _graduated_comprehend_rate(matches)
            if conservative is None:
                conservative = _conservative_guardrail_rate(matches)
            if conservative is not None:
                return conservative
            message = f"ambiguous price rows for {feature} in {region}"
            raise AmbiguousPriceError(message)
        return matches[0]


def _conservative_guardrail_rate(
    rows: Sequence[PriceRow],
) -> PriceRow | None:
    """Bound parallel guardrail SKUs by their highest OnDemand rate.

    One feature can have several TextUnit SKUs, including a free row.
    Reservation uses the highest rate.
    """
    if not rows[0].feature.startswith("guardrails."):
        return None
    return max(rows, key=lambda row: row.usd)


def _graduated_comprehend_rate(rows: Sequence[PriceRow]) -> PriceRow | None:
    """Bound one Comprehend SKU's volume bands by their highest rate."""
    if (
        not rows[0].feature.startswith("comprehend.")
        or len({(row.service_code, row.sku, row.usage_type) for row in rows})
        != 1
    ):
        return None
    try:
        bands = sorted(
            (
                (Decimal(row.begin_range), Decimal(row.end_range), row)
                for row in rows
            ),
            key=lambda band: band[0],
        )
        boundary = Decimal(0)
        for begin, end, _ in bands:
            if begin != boundary or not begin.is_finite() or end <= begin:
                return None
            boundary = end
        if boundary != Decimal("Infinity"):
            return None
    except InvalidOperation:
        return None
    return max(rows, key=lambda row: row.usd)


def build_price_snapshot(
    client: PricingClient,
    *,
    regions: Sequence[str],
    clock: Any | None = None,
) -> PriceSnapshot:
    """Read Price List pages into a conservative snapshot."""
    read_at = _read_at(clock)
    requested_regions = tuple(dict.fromkeys(regions))
    rows: list[PriceRow] = []
    errors: list[SnapshotError] = []
    try:
        discovered = _describe_service_codes(client)
    except _SNAPSHOT_READ_ERRORS as exc:
        discovered = frozenset[str]()
        errors.append(
            SnapshotError(
                "pricing", "DescribeServices", sanitize_exception(exc)
            )
        )
    selected = _select_service_codes(discovered)
    for region in requested_regions:
        for service_codes in selected.values():
            if not service_codes:
                continue
            for service_code in service_codes:
                try:
                    rows.extend(
                        _read_products(
                            client,
                            service_code=service_code,
                            region=region,
                            read_at=read_at,
                            errors=errors,
                        )
                    )
                except _SNAPSHOT_READ_ERRORS as exc:
                    errors.append(
                        SnapshotError(
                            service_code,
                            "GetProducts",
                            sanitize_exception(exc),
                        )
                    )
    service_codes = tuple(
        sorted({code for codes in selected.values() for code in codes})
    )
    missing = _missing_issues(
        selected=selected,
        rows=rows,
        regions=requested_regions,
    )
    return PriceSnapshot(
        schema_version=SCHEMA_VERSION,
        pricing_region=PRICING_REGION,
        requested_regions=requested_regions,
        read_at=read_at,
        service_codes=service_codes,
        rows=tuple(row for row in rows if row.feature != UNKNOWN_FEATURE),
        missing=tuple(
            issue for issue in missing if issue.kind != "unknown_feature"
        ),
        errors=tuple(errors),
        unmatched_rows=tuple(
            row for row in rows if row.feature == UNKNOWN_FEATURE
        ),
    )


def load_snapshot(path: Path) -> PriceSnapshot:
    """Load a price snapshot JSON file."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return PriceSnapshot.from_json(_mapping(loaded))


def write_snapshot(snapshot: PriceSnapshot, path: Path) -> None:
    """Write a deterministic snapshot JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(redact(snapshot.to_json()), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_review(snapshot: PriceSnapshot, path: Path) -> None:
    """Write requested unmatched dimensions for local review."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            redact(
                {
                    "read_at": snapshot.read_at,
                    "rows": [row.to_json() for row in snapshot.unmatched_rows],
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _unknown_count(data: Mapping[str, object]) -> int:
    summary = data.get("missing_summary")
    if summary is None:
        return 0
    counts = _mapping(_mapping(summary).get("counts_by_kind"))
    return _required_int(counts, "unknown_feature")


def _loaded_row(data: Mapping[str, object]) -> PriceRow:
    row = PriceRow.from_json(data)
    if row.feature.startswith("comprehend."):
        return replace(row, tier="standard", routing="in-region")
    if row.feature.startswith("model:Nova 2.0 Lite:"):
        attributes = {
            "modelName": "Nova 2.0 Lite",
            "usagetype": row.usage_type,
        }
        feature = _feature(
            row.service_code, attributes, row.description, row.raw_unit
        )
        return replace(row, feature=feature)
    return row


def _describe_service_codes(client: PricingClient) -> frozenset[str]:
    service_codes: set[str] = set()
    params: dict[str, object] = {"MaxResults": 100}
    while True:
        response = client.describe_services(**params)
        for item in cast(
            "Sequence[Mapping[str, object]]", response["Services"]
        ):
            service_codes.add(_required_str(item, "ServiceCode"))
        token = response.get("NextToken")
        if not isinstance(token, str) or not token:
            return frozenset(service_codes)
        params["NextToken"] = token


def _select_service_codes(
    discovered: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    discovered_set = frozenset(discovered)
    return {
        group: tuple(code for code in candidates if code in discovered_set)
        for group, candidates in _SERVICE_GROUP_CANDIDATES.items()
    }


def _read_products(
    client: PricingClient,
    *,
    service_code: str,
    region: str,
    read_at: str,
    errors: list[SnapshotError],
) -> list[PriceRow]:
    rows: list[PriceRow] = []
    params: dict[str, object] = {
        "ServiceCode": service_code,
        "FormatVersion": "aws_v1",
        "Filters": [
            {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region}
        ],
        "MaxResults": 100,
    }
    while True:
        response = client.get_products(**params)
        for raw in cast("Sequence[str]", response.get("PriceList", ())):
            try:
                rows.extend(
                    _rows_from_product(
                        service_code=service_code,
                        document=_mapping(json.loads(raw)),
                        requested_region=region,
                        read_at=read_at,
                    )
                )
            except (json.JSONDecodeError, PricingError) as exc:
                errors.append(
                    SnapshotError(
                        service_code,
                        "ParseProduct",
                        sanitize_exception(exc),
                    )
                )
        token = response.get("NextToken")
        if not isinstance(token, str) or not token:
            return rows
        params["NextToken"] = token


def _rows_from_product(
    *,
    service_code: str,
    document: Mapping[str, object],
    requested_region: str,
    read_at: str,
) -> list[PriceRow]:
    product = _mapping(document.get("product"))
    terms = _mapping(document.get("terms"))
    attributes = _mapping(product.get("attributes", {}))
    sku = _required_str(product, "sku")
    region = _str_value(attributes.get("regionCode")) or requested_region
    usage_type = _str_value(attributes.get("usagetype"))
    ondemand_terms = _mapping(terms.get("OnDemand", {}))
    rows: list[PriceRow] = []
    for term in ondemand_terms.values():
        dimensions = _mapping(_mapping(term).get("priceDimensions", {}))
        for rate_code, dimension in dimensions.items():
            dimension_map = _mapping(dimension)
            description = _str_value(dimension_map.get("description"))
            raw_unit = _str_value(dimension_map.get("unit"))
            raw_usd = _usd(dimension_map)
            feature = _feature(service_code, attributes, description, raw_unit)
            normalized = _normalize_unit_price(feature, raw_unit, raw_usd)
            if normalized is None:
                feature = UNKNOWN_FEATURE
                unit = raw_unit or UNKNOWN_FEATURE
                usd = raw_usd
            else:
                unit, usd = normalized
            rows.append(
                PriceRow(
                    service_code=service_code,
                    feature=feature,
                    region=region,
                    usage_type=usage_type,
                    tier=_tier(attributes, description)
                    if feature.startswith("model:")
                    else "standard",
                    routing=_routing(attributes, description)
                    if feature.startswith("model:")
                    else "in-region",
                    unit=unit,
                    usd=usd,
                    raw_unit=raw_unit,
                    raw_usd=raw_usd,
                    read_at=read_at,
                    sku=sku,
                    rate_code=str(rate_code),
                    begin_range=_str_value(dimension_map.get("beginRange")),
                    end_range=_str_value(dimension_map.get("endRange")),
                    description=description,
                )
            )
    if not rows:
        message = f"product {sku} has no OnDemand price dimensions"
        raise PricingError(message)
    return rows


def _feature(
    service_code: str,
    attributes: Mapping[str, object],
    description: str,
    unit: str,
) -> str:
    text = _text_blob(service_code, attributes, description, unit)
    semantic_text = _text_blob(service_code, attributes, description, "")
    if _is_bedrock_service(service_code):
        guardrail = _guardrail_feature(semantic_text)
        if guardrail is not None:
            return guardrail
        model = _model_id(attributes)
        direction = _token_direction(text)
        if (
            model
            and direction is not None
            and not any(
                marker in semantic_text
                for marker in (
                    "cache",
                    "custom-model",
                    "image",
                    "audio",
                    "video",
                )
            )
        ):
            return f"model:{model}:{direction}"
    if "agentcore" in text:
        agentcore = _agentcore_feature(semantic_text)
        if agentcore is not None:
            return agentcore
    if service_code in {"AmazonS3Vectors", "AmazonS3"}:
        s3vectors = _s3vectors_feature(
            _text_blob(
                service_code,
                {"usagetype": attributes.get("usagetype")},
                description,
                unit,
            )
        )
        if s3vectors is not None:
            return s3vectors
    if service_code in {"comprehend", "AmazonComprehend", "Comprehend"}:
        for operation, name in (
            ("detectpiientities", "detect_pii_entities"),
            ("detectsentiment", "detect_sentiment"),
        ):
            if operation in text:
                return f"comprehend.{name}.units"
        return UNKNOWN_FEATURE
    if service_code in {"translate", "AmazonTranslate"}:
        return (
            "translate.characters"
            if _character_unit(text)
            and not any(
                marker in text
                for marker in ("job", "document", "activecustom")
            )
            and any(
                marker in text
                for marker in (
                    "texttranslation",
                    "translatetext",
                    "translatechars",
                )
            )
            else UNKNOWN_FEATURE
        )
    if service_code == "AmazonPolly":
        return (
            "polly.characters"
            if _character_unit(text)
            and not any(
                marker in text
                for marker in ("neural", "generative", "longform", "long-form")
            )
            else UNKNOWN_FEATURE
        )
    return UNKNOWN_FEATURE


def _feature_candidates(feature: str, routing: str) -> frozenset[str]:
    candidates = {feature}
    if routing == "cross-region-global" and feature.startswith(
        "model:global."
    ):
        model_id, direction = feature.removeprefix("model:").rsplit(":", 1)
        candidates.add(f"model:{model_id.removeprefix('global.')}:{direction}")
    return frozenset(candidates)


def _normalize_unit_price(
    feature: str,
    raw_unit: str,
    raw_usd: Decimal,
) -> tuple[str, Decimal] | None:
    unit_text = raw_unit.lower()
    canonical = _canonical_unit(feature)
    if canonical is None:
        return None
    if canonical in {"token", "character", "text_unit"}:
        if canonical == "token" and re.fullmatch(
            r"(?:1m|1[ ,]?000[ ,]?000|1 million) tokens?", unit_text
        ):
            return canonical, raw_usd / Decimal(1_000_000)
        if _per_thousand(unit_text):
            return canonical, raw_usd / Decimal(1000)
        if canonical == "token" and "token" in unit_text:
            return canonical, raw_usd
        if canonical == "character" and _character_unit(unit_text):
            return canonical, raw_usd
        if canonical == "text_unit" and "text" in unit_text:
            return canonical, raw_usd
        return None
    if canonical == "event" and "event" in unit_text:
        divisor = Decimal(1000) if _per_thousand(unit_text) else Decimal(1)
        return canonical, raw_usd / divisor
    if canonical == "gb_second" and _memory_seconds(unit_text):
        divisor = Decimal(3600) if _hour(unit_text) else Decimal(1)
        return canonical, raw_usd / divisor
    if canonical == "second" and ("second" in unit_text or _hour(unit_text)):
        divisor = Decimal(3600) if _hour(unit_text) else Decimal(1)
        return canonical, raw_usd / divisor
    if canonical == "request" and (
        "request" in unit_text or "invocation" in unit_text
    ):
        divisor = Decimal(1000) if _per_thousand(unit_text) else Decimal(1)
        return canonical, raw_usd / divisor
    if canonical == "gb_month" and (
        "gb-mo" in unit_text or "gb month" in unit_text
    ):
        return canonical, raw_usd
    if canonical in {"gb", "unit"} and unit_text == canonical:
        return canonical, raw_usd
    return None


def _per_thousand(unit_text: str) -> bool:
    return "1k" in unit_text or "1000" in unit_text or "1,000" in unit_text


def _hour(unit_text: str) -> bool:
    return "hour" in unit_text or "hr" in unit_text


def _canonical_unit(feature: str) -> str | None:
    if feature.startswith("model:"):
        return "token"
    if feature.startswith("guardrails."):
        return "text_unit"
    if feature.endswith(".characters"):
        return "character"
    if feature == "memory.events":
        return "event"
    if feature.endswith(".cpu_seconds"):
        return "second"
    if feature.endswith(".memory_gb_seconds"):
        return "gb_second"
    if feature == "s3vectors.requests" or feature.startswith(
        "s3vectors.requests."
    ):
        return "request"
    if feature == "s3vectors.storage_gb_month":
        return "gb_month"
    if feature.startswith("comprehend.") and feature.endswith(".units"):
        return "unit"
    if feature in {
        "s3vectors.put_gb",
        "s3vectors.query_returned_gb",
    } or feature.startswith("s3vectors.query_processed_gb."):
        return "gb"
    return None


def _guardrail_feature(text: str) -> str | None:
    if "guardrail" not in text:
        return None
    if "denied" in text or "topic" in text:
        return "guardrails.denied_topic"
    if "sensitive" in text or "pii" in text:
        return "guardrails.sensitive_information"
    if "content" in text or "filter" in text:
        return "guardrails.content_filter"
    return UNKNOWN_FEATURE


def _agentcore_feature(text: str) -> str | None:
    if "memory" in text and "event" in text:
        return "memory.events"
    if "code" in text and "interpreter" in text and "cpu" in text:
        return "code_interpreter.cpu_seconds"
    if "code" in text and "interpreter" in text and _memory_seconds(text):
        return "code_interpreter.memory_gb_seconds"
    if "runtime" in text and "cpu" in text:
        return "agentcore.runtime.cpu_seconds"
    if "runtime" in text and _memory_seconds(text):
        return "agentcore.runtime.memory_gb_seconds"
    return None


def _s3vectors_feature(text: str) -> str | None:
    if "vector" not in text:
        return None
    for usage, feature in (
        ("vectors-put-bytes", "s3vectors.put_gb"),
        ("vectors-query-returned-bytes", "s3vectors.query_returned_gb"),
    ):
        if usage in text:
            return feature
    for tier in ("tier1", "tier2", "tier3"):
        if f"vectors-query-processedbytes-{tier}" in text:
            return f"s3vectors.query_processed_gb.{tier}"
        if f"vectors-request-{tier}" in text:
            return f"s3vectors.requests.{tier}"
    if "storage" in text or "gb-month" in text:
        return "s3vectors.storage_gb_month"
    if "request" in text or "api" in text or "query" in text:
        return "s3vectors.requests"
    return None


def _missing_issues(
    *,
    selected: Mapping[str, Sequence[str]],
    rows: Sequence[PriceRow],
    regions: Sequence[str],
) -> list[PriceIssue]:
    issues: list[PriceIssue] = []
    for region in regions:
        for group, required_features in _REQUIRED_FEATURES.items():
            service_codes = selected[group]
            if not service_codes:
                issues.append(
                    PriceIssue(
                        "missing_service",
                        None,
                        group,
                        region,
                        "no discovered Price List service code",
                    )
                )
                continue
            issues.extend(
                PriceIssue(
                    "missing_feature",
                    ",".join(service_codes),
                    feature,
                    region,
                    "no matching OnDemand price row",
                )
                for feature in required_features
                if not _has_feature(rows, feature, region)
            )
    issues.extend(
        PriceIssue(
            "unknown_feature",
            row.service_code,
            row.sku,
            row.region,
            "preserved SKU could not be normalized",
        )
        for row in rows
        if row.feature == UNKNOWN_FEATURE
    )
    return issues


def _has_feature(rows: Sequence[PriceRow], pattern: str, region: str) -> bool:
    if "*" not in pattern:
        return any(
            row.feature == pattern and row.region == region for row in rows
        )
    prefix, suffix = pattern.split("*", 1)
    return any(
        row.feature.startswith(prefix)
        and row.feature.endswith(suffix)
        and row.region == region
        for row in rows
    )


def _is_bedrock_service(service_code: str) -> bool:
    return service_code in {"AmazonBedrock", "AmazonBedrockFoundationModels"}


def _model_id(attributes: Mapping[str, object]) -> str:
    usage = _str_value(attributes.get("usagetype")).lower()
    matched = re.fullmatch(
        r"(?:[a-z0-9]+-)?(?P<mantle>openai\.)?gpt-oss-(?P<size>20|120)b"
        r"(?P<endpoint>-mantle)?-(?:input|output)-tokens"
        r"(?:-(?:standard|flex|priority|batch))?",
        usage,
    )
    if matched is not None:
        if bool(matched["mantle"]) != bool(matched["endpoint"]):
            return ""
        version = "" if matched["endpoint"] else "-1:0"
        return f"openai.gpt-oss-{matched['size']}b{version}"
    for key in ("modelId", "modelID", "model", "modelName", "servicename"):
        value = _str_value(attributes.get(key))
        aliases = {
            "Nova 2.0 Lite": "amazon.nova-2-lite-v1:0",
            "GPT OSS 20B": "openai.gpt-oss-20b-1:0",
            "GPT OSS 120B": "openai.gpt-oss-120b-1:0",
            "Claude Haiku 4.5 (Amazon Bedrock Edition)": (
                "anthropic.claude-haiku-4-5-20251001-v1:0"
            ),
        }
        if value in aliases:
            return aliases[value]
        if value and " " not in value and ("." in value or ":" in value):
            return value
    return ""


def _token_direction(text: str) -> str | None:
    if "output" in text:
        return "output"
    if "input" in text:
        return "input"
    return None


def _tier(attributes: Mapping[str, object], description: str) -> str:
    text = _text_blob("", attributes, description, "")
    # S3 tiers identify separate features, not Bedrock service tiers.
    if re.search(r"vectors-(?:request|query-processedbytes)-tier[123]", text):
        return "standard"
    for value in ("priority", "flex", "batch"):
        if value in text:
            return value
    for key, attr_value in attributes.items():
        raw = _str_value(attr_value).lower()
        if "tier" in str(key).lower() and raw and "standard" not in raw:
            return UNKNOWN_FEATURE
    if "tier" in text and "standard" not in text:
        return UNKNOWN_FEATURE
    return "standard"


def _routing(attributes: Mapping[str, object], description: str) -> str:
    text = _text_blob("", attributes, description, "")
    if "global" in text:
        return "cross-region-global"
    if "cross-region" in text or "cross region" in text:
        return "cross-region"
    if (
        ("routing" in text or "route" in text)
        and "in-region" not in text
        and "in region" not in text
    ):
        return UNKNOWN_FEATURE
    return "in-region"


def _character_unit(text: str) -> bool:
    return "character" in text or " char" in f" {text}"


def _memory_seconds(text: str) -> bool:
    return (
        "memory" in text
        or "gb-second" in text
        or "gb second" in text
        or "gb-hour" in text
        or "gb hour" in text
    )


def _text_blob(
    service_code: str,
    attributes: Mapping[str, object],
    description: str,
    unit: str,
) -> str:
    values = [service_code, description, unit]
    values.extend(_str_value(value) for value in attributes.values())
    return " ".join(values).lower()


def _usd(dimension: Mapping[str, object]) -> Decimal:
    price_per_unit = _mapping(dimension.get("pricePerUnit"))
    return _decimal(_required_str(price_per_unit, "USD"))


def _read_at(clock: Any | None) -> str:
    if clock is not None:
        value = clock()
        if isinstance(value, datetime):
            return value.astimezone(UTC).isoformat()
        return str(value)
    return datetime.now(UTC).isoformat()


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        message = "expected JSON object"
        raise PricingError(message)
    return cast("Mapping[str, object]", value)


def _required_sequence(
    data: Mapping[str, object],
    key: str,
) -> Sequence[object]:
    value = data.get(key)
    if not isinstance(value, list):
        message = f"{key} must be a list"
        raise PricingError(message)
    return value


def _str_sequence(data: Mapping[str, object], key: str) -> list[str]:
    values = _required_sequence(data, key)
    if not all(isinstance(item, str) for item in values):
        message = f"{key} must contain only strings"
        raise PricingError(message)
    return cast("list[str]", values)


def _required_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        message = f"{key} must be an integer"
        raise PricingError(message)
    return value


def _required_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        message = f"{key} must be a string"
        raise PricingError(message)
    return value


def _str_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def _decimal(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        message = f"invalid USD decimal {value!r}"
        raise PricingError(message) from exc
