"""Connect price evidence, durable budgets and live dispatch."""

from __future__ import annotations

import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from awsai_demo.policy import (
    BudgetLedger,
    ExecutionPolicy,
    PricedBudget,
    PriceResolver,
    ReservationUnavailable,
)
from awsai_demo.redact import redact

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from decimal import Decimal


@dataclass(frozen=True)
class BudgetRun:
    """One run id and aggregate ledger shared by live commands."""

    run_id: str
    directory: Path
    region: str
    policy: ExecutionPolicy
    prices: PriceResolver
    aggregate: BudgetLedger | None = None

    def command(self, name: str) -> PricedBudget:
        """Open a durable command ledger in this run."""
        path = (
            self.directory / "commands" / name / "budget.json"
            if self.aggregate is not None
            else self.directory / "budget.json"
        )
        return PricedBudget(
            prices=self.prices,
            policy=self.policy,
            region=self.region,
            ledger=BudgetLedger(path, self.policy),
            aggregate=self.aggregate,
            command_id=name,
            notice=cost_notice,
        )


_ACTIVE_RUN: ContextVar[BudgetRun | None] = ContextVar(
    "awsai_budget_run", default=None
)


class SnapshotPrices:
    """Defer snapshot reads for read-only commands."""

    def __init__(
        self,
        path: Path,
        *,
        loader: Callable[[Path], PriceResolver] | None = None,
    ) -> None:
        """Retain the public snapshot path and an optional loader."""
        self.path = path
        self._prices: PriceResolver | None = None
        self._loader = loader

    def rate(
        self,
        feature: str,
        *,
        region: str,
        unit: str,
        tier: str,
        routing: str,
    ) -> Decimal:
        """Load verified prices on the first billable reservation."""
        if self._prices is None:
            loader = load_prices if self._loader is None else self._loader
            self._prices = loader(self.path)
        return self._prices.rate(
            feature, region=region, unit=unit, tier=tier, routing=routing
        )


def cost_notice(message: str) -> None:
    """Write only redacted cost estimates to stderr."""
    print(str(redact(message)), file=sys.stderr)


def load_prices(
    path: Path,
    *,
    reader: Callable[[Path], PriceResolver] | None = None,
) -> PriceResolver:
    """Load the generated snapshot, never substituting zero prices."""
    from awsai_demo.pricing import PriceBook

    read = PriceBook.from_file if reader is None else reader
    try:
        return read(path)
    except (OSError, ValueError) as exc:
        message = "Verified pricing snapshot is unavailable or invalid"
        raise ReservationUnavailable(message) from exc


def open_budget_run(
    *,
    region: str,
    policy: ExecutionPolicy,
    root: Path = Path(".awsai_runs"),
    snapshot_path: Path = Path("data/pricing_snapshot.json"),
    aggregate: bool = False,
    prices: PriceResolver | None = None,
    loader: Callable[[Path], PriceResolver] = load_prices,
    run_id: str | None = None,
) -> BudgetRun:
    """Resolve prices before creating a run or selecting credentials."""
    resolved = prices if prices is not None else loader(snapshot_path)
    identifier = run_id or "a" + uuid.uuid4().hex[:11]
    if len(identifier) != 12 or any(
        char not in "0123456789abcdef" for char in identifier
    ):
        message = "Budget run id must be 12 lowercase hexadecimal characters"
        raise ValueError(message)
    directory = root.resolve() / identifier
    return BudgetRun(
        identifier,
        directory,
        region,
        policy,
        resolved,
        BudgetLedger(directory / "budget.json", policy, all_live=True)
        if aggregate
        else None,
    )


@contextmanager
def use_budget_run(run: BudgetRun) -> Iterator[None]:
    """Share a live run without making any service calls."""
    token = _ACTIVE_RUN.set(run)
    try:
        yield
    finally:
        _ACTIVE_RUN.reset(token)


def active_budget_run(
    *,
    region: str,
    policy: ExecutionPolicy,
    opener: Callable[..., BudgetRun] | None = None,
) -> BudgetRun:
    """Use the all-run context or open a single command run."""
    current = _ACTIVE_RUN.get()
    if current is not None:
        if current.region != region or current.policy != policy:
            message = "Live command differs from its shared run policy"
            raise ValueError(message)
        return current
    open_run = open_budget_run if opener is None else opener
    return open_run(region=region, policy=policy)
