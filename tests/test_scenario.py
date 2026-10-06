from __future__ import annotations

import math

import pytest

from awsai_demo.scenario import (
    action_key,
    price_pilot,
    proposal_is_acceptable,
    proposal_version,
)


def test_default_proposal_and_version() -> None:
    cost = price_pilot()
    assert (cost.staffing_usd, cost.platform_usd, cost.total_usd) == (
        18600,
        1080,
        19680,
    )
    assert cost.headroom_usd == 5320
    assert cost.within_budget
    assert proposal_is_acceptable(cost)[0]
    assert "approval" in proposal_is_acceptable(cost)[1][0]
    assert action_key(cost) == "accept-proposal:" + proposal_version(cost)
    assert proposal_version(cost) == proposal_version(
        price_pilot(team={"python-engineer": 1.0, "solution-architect": 0.5})
    )
    assert proposal_version(price_pilot(weeks=5)) != proposal_version(cost)
    assert not proposal_is_acceptable(price_pilot(budget=10))[0]


@pytest.mark.parametrize("weeks", [0, -1, True, 1.5])
def test_invalid_weeks(weeks: int) -> None:
    with pytest.raises(ValueError, match="weeks"):
        price_pilot(weeks=weeks)


@pytest.mark.parametrize("budget", [0, -1, math.nan, math.inf])
def test_invalid_budget(budget: float) -> None:
    with pytest.raises(ValueError, match="budget"):
        price_pilot(budget=budget)


@pytest.mark.parametrize("fte", [-1, math.nan, math.inf])
def test_invalid_fte(fte: float) -> None:
    with pytest.raises(ValueError, match="FTE"):
        price_pilot(team={"python-engineer": fte})


def test_unknown_role_and_empty_team() -> None:
    with pytest.raises(ValueError, match="Unknown role"):
        price_pilot(team={"wizard": 1})
    assert price_pilot(team={}).staffing_usd == 0
