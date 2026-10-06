"""Deterministic costing for the six-week customer-support pilot."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

PILOT_WEEKS = 6
PILOT_BUDGET_USD = 25_000
RATE_CARD_USD_PER_WEEK = MappingProxyType(
    {"solution-architect": 2400, "python-engineer": 1900, "qa-engineer": 1500}
)
DEFAULT_TEAM = MappingProxyType(
    {"solution-architect": 0.5, "python-engineer": 1.0}
)
PLATFORM_COST_USD_PER_WEEK = 180
BRIEF = (
    "Cost a six-week customer-support pilot within USD 25,000. "
    "AWS resources and permissions are not provisioned yet. "
    "Do not accept the proposal without named human approval."
)


@dataclass(frozen=True)
class PilotCost:
    """A computed proposal; being affordable does not grant approval."""

    weeks: int
    team: tuple[tuple[str, float], ...]
    staffing_usd: float
    platform_usd: float
    total_usd: float
    budget_usd: float
    headroom_usd: float
    within_budget: bool


def price_pilot(
    weeks: int = PILOT_WEEKS,
    team: Mapping[str, float] = DEFAULT_TEAM,
    budget: float = PILOT_BUDGET_USD,
) -> PilotCost:
    """Price staffing and platform without accepting the proposal."""
    if isinstance(weeks, bool) or not isinstance(weeks, int) or weeks <= 0:
        message = "weeks must be a positive integer"
        raise ValueError(message)
    if not math.isfinite(budget) or budget <= 0:
        message = "budget must be finite and positive"
        raise ValueError(message)
    staffing = 0.0
    for role, fte in team.items():
        if role not in RATE_CARD_USD_PER_WEEK:
            message = f"Unknown role: {role}"
            raise ValueError(message)
        if not math.isfinite(fte) or fte < 0:
            message = "FTE must be finite and nonnegative"
            raise ValueError(message)
        staffing += RATE_CARD_USD_PER_WEEK[role] * fte * weeks
    platform = float(PLATFORM_COST_USD_PER_WEEK * weeks)
    total = staffing + platform
    return PilotCost(
        weeks,
        tuple(sorted(team.items())),
        staffing,
        platform,
        total,
        budget,
        budget - total,
        total <= budget,
    )


def proposal_is_acceptable(cost: PilotCost) -> tuple[bool, tuple[str, ...]]:
    """Check affordability; human approval is still required."""
    if not cost.within_budget:
        return False, ("Proposal exceeds the budget",)
    return True, ("Within budget; named human approval is still required",)


def proposal_version(cost: PilotCost) -> str:
    """Hash the proposal to bind approval to its exact contents."""
    payload = json.dumps(asdict(cost), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def action_key(cost: PilotCost) -> str:
    """Deduplicate the business action independently of its approver."""
    return "accept-proposal:" + proposal_version(cost)
