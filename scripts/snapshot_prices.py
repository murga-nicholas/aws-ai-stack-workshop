"""Generate the generated AWS Price List pricing snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO, cast

from awsai_demo.cli import CommandParser
from awsai_demo.credentials import select_session
from awsai_demo.pricing import (
    PRICING_REGION,
    PriceSnapshot,
    PricingClient,
    build_price_snapshot,
    write_review,
    write_snapshot,
)
from awsai_demo.redact import quiet_sdk_logging, redact, sanitize_exception
from awsai_demo.runtime import (
    DEFAULT_REGION,
    Settings,
    load_settings,
    with_cli_overrides,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from awsai_demo.credentials import CsvLoader, SessionFactory


class PricingSession(Protocol):
    """Small boto3 Session surface needed by the snapshot CLI."""

    def client(self, service_name: str, **kwargs: Any) -> PricingClient:
        """Return a pricing client."""


@quiet_sdk_logging()
def snapshot_prices(
    *,
    regions: Sequence[str],
    output: Path,
    profile: str | None = None,
    settings: Settings | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
    clock: Any | None = None,
    review_output: Path | None = None,
) -> PriceSnapshot:
    """Read Price List and write a generated pricing snapshot."""
    base_settings = Settings() if settings is None else settings
    effective_settings = with_cli_overrides(base_settings, profile=profile)
    pricing_settings = with_cli_overrides(
        effective_settings,
        region=PRICING_REGION,
    )
    selected = select_session(
        execution="live",
        settings=pricing_settings,
        session_factory=session_factory,
        csv_loader=csv_loader,
    )
    session = cast("PricingSession", selected.session)
    client = session.client("pricing", region_name=PRICING_REGION)
    snapshot = build_price_snapshot(client, regions=regions, clock=clock)
    write_snapshot(snapshot, output)
    if review_output is not None:
        write_review(snapshot, review_output)
    return snapshot


def main(
    argv: Sequence[str] | None = None,
    *,
    session_factory: SessionFactory | None = None,
    settings_loader: Callable[[], Settings] | None = None,
    csv_loader: CsvLoader | None = None,
    clock: Any | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """CLI entry point."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    try:
        args = _parser().parse_args(argv)
        settings = (
            load_settings if settings_loader is None else settings_loader
        )()
        regions = args.region or [DEFAULT_REGION]
        snapshot = snapshot_prices(
            regions=regions,
            output=args.output,
            profile=args.profile,
            settings=settings,
            session_factory=session_factory,
            csv_loader=csv_loader,
            clock=clock,
            review_output=args.review_output,
        )
        out.write(
            json.dumps(
                redact(
                    {
                        "output": str(args.output),
                        "regions": list(snapshot.requested_regions),
                        "rows": len(snapshot.rows),
                        "missing": snapshot.issue_counts(),
                        "errors": len(snapshot.errors),
                    }
                ),
                sort_keys=True,
            )
            + "\n"
        )
    except Exception as exc:
        err.write(sanitize_exception(exc) + "\n")
        return 1
    return 0


def _parser() -> CommandParser:
    parser = CommandParser(
        description="Generate data/pricing_snapshot.json from AWS Price List.",
    )
    parser.add_argument(
        "--region",
        action="append",
        default=None,
        help="Requested workload region to filter products; repeatable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pricing_snapshot.json"),
        help="Snapshot output path.",
    )
    parser.add_argument(
        "--review-output",
        type=_review_path,
        default=None,
        help="Optional unmatched rows (*.pricing-review.json, git-ignored).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="AWS profile for the read-only Price List session.",
    )
    return parser


def _review_path(value: str) -> Path:
    path = Path(value)
    if not path.name.endswith(".pricing-review.json"):
        message = "Review filename must end with .pricing-review.json"
        raise ValueError(message)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
