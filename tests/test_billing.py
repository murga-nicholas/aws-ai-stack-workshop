from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from awsai_demo import billing
from awsai_demo.cli import build_parser, dispatch
from awsai_demo.contracts import result
from awsai_demo.policy import Charge, ExecutionPolicy, ReservationUnavailable
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from awsai_demo.contracts import DemoResult
    from awsai_demo.registry import DemoSpec


def test_run_ledgers_and_active_scope(tmp_path: Path) -> None:
    prices = Mock()
    prices.rate.return_value = Decimal("0.00001")
    policy = ExecutionPolicy()
    run = billing.open_budget_run(
        region="us-east-1",
        policy=policy,
        root=tmp_path,
        prices=prices,
        aggregate=True,
        run_id="abcdef123456",
    )
    with billing.use_budget_run(run):
        assert (
            billing.active_budget_run(region=run.region, policy=policy) is run
        )
        with pytest.raises(ValueError, match="shared run"):
            billing.active_budget_run(region="eu-west-1", policy=policy)
        for name in ("one", "two"):
            run.command(name).reserve(
                "request", (Charge("feature", Decimal(1), "request"),)
            )
    assert run.aggregate.snapshot().reserved_usd == Decimal("0.00004")
    single = billing.open_budget_run(
        region=run.region,
        policy=policy,
        root=tmp_path,
        prices=prices,
    )
    single.command("demo").reserve(
        "request", (Charge("feature", Decimal(1), "request"),)
    )
    assert (single.directory / "budget.json").is_file()
    assert len(single.run_id) == 12
    for identifier in ("short", "z" * 12):
        with pytest.raises(ValueError, match="hexadecimal"):
            billing.open_budget_run(
                region=run.region,
                policy=policy,
                prices=prices,
                run_id=identifier,
            )


def test_lazy_prices_load_once_and_missing_fail_closed(
    tmp_path: Path,
) -> None:
    prices = Mock()
    prices.rate.return_value = Decimal(1)
    loader = Mock(return_value=prices)
    lazy = billing.SnapshotPrices(tmp_path / "snapshot.json", loader=loader)
    for _ in range(2):
        assert (
            lazy.rate(
                "feature", region="r", unit="u", tier="t", routing="route"
            )
            == 1
        )
    loader.assert_called_once()
    missing = billing.SnapshotPrices(tmp_path / "missing.json")
    with pytest.raises(ReservationUnavailable, match="unavailable"):
        missing.rate(
            "feature", region="r", unit="u", tier="t", routing="route"
        )
    opened = billing.open_budget_run(
        region="r",
        policy=ExecutionPolicy(),
        root=tmp_path,
        loader=loader,
    )
    assert (
        billing.active_budget_run(
            region="r",
            policy=opened.policy,
            opener=lambda **_: opened,
        )
        is opened
    )


def test_snapshot_reader_and_cost_redaction(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "missing.json"
    with pytest.raises(ReservationUnavailable, match="unavailable"):
        billing.load_prices(path)
    path.write_text("invalid", encoding="utf-8")
    with pytest.raises(ReservationUnavailable, match="unavailable"):
        billing.load_prices(path)
    book = Mock()
    assert billing.load_prices(path, reader=lambda _path: book) is book
    billing.cost_notice("Reserved USD 0.01 account 123456789012")
    captured = capsys.readouterr()
    assert not captured.out
    assert "123456789012" not in captured.err


def test_all_shares_one_lazy_budget_without_credentials(
    tmp_path: Path,
) -> None:
    runs: list[billing.BudgetRun] = []

    def execute(
        spec: DemoSpec,
        args: argparse.Namespace,
        settings: Settings,
    ) -> DemoResult:
        runs.append(
            billing.active_budget_run(
                region=settings.region,
                policy=ExecutionPolicy(),
            )
        )
        return result(
            demo=spec.name,
            technology=spec.technology,
            lane=spec.lane,
            lifecycle_refs=spec.lifecycle_refs,
            requested_execution=args.execution,
            headline="Read only fixture",
        )

    payload = dispatch(
        build_parser().parse_args(
            ["all", "--execution", "live", "--lane", "local"]
        ),
        Settings(),
        executor=execute,
        run_directory=tmp_path,
    )
    assert payload["mode"] == "batch"
    assert len(runs) == 2 and runs[0] is runs[1]
    assert runs[0].aggregate is not None
