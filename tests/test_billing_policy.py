from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from awsai_demo.policy import (
    BYTE_UPPER_BOUND_EVIDENCE,
    DEFAULT_BYTE_UPPER_BOUND_MODELS,
    BudgetExceeded,
    BudgetLedger,
    Charge,
    ExecutionPolicy,
    InputTooLarge,
    PolicyValidationError,
    PricedBudget,
    ReservationUnavailable,
    count_input_tokens,
)

if TYPE_CHECKING:
    from pathlib import Path


class Prices:
    def __init__(self, rate: str = "0.000001") -> None:
        self.value = Decimal(rate)
        self.calls: list[str] = []

    def rate(
        self,
        feature: str,
        *,
        region: str,
        unit: str,
        tier: str,
        routing: str,
    ) -> Decimal:
        assert (region, tier, routing) == (
            "us-east-1",
            "standard",
            "in-region",
        )
        assert unit
        self.calls.append(feature)
        return self.value


def budget(
    path: Path, prices: Prices, *, aggregate: bool = False
) -> PricedBudget:
    policy = ExecutionPolicy()
    return PricedBudget(
        prices=prices,
        region="us-east-1",
        policy=policy,
        ledger=BudgetLedger(path / "command.json", policy),
        aggregate=BudgetLedger(path / "all.json", policy, all_live=True)
        if aggregate
        else None,
        command_id="demo",
    )


def test_byte_bound_models_have_specific_source_evidence() -> None:
    assert (
        frozenset(
            BYTE_UPPER_BOUND_EVIDENCE,
        )
        == DEFAULT_BYTE_UPPER_BOUND_MODELS
    )
    for model in DEFAULT_BYTE_UPPER_BOUND_MODELS:
        counted = count_input_tokens(model, "Unicode: € 😀".encode())
        assert counted.tokens == len("Unicode: € 😀".encode()) + 256
        assert counted.method == "upper_bound"
        assert BYTE_UPPER_BOUND_EVIDENCE[model]["source"].startswith(
            "https://github.com/openai/gpt-oss/",
        )


def test_priced_service_reservation_is_durable_and_shared(
    tmp_path: Path,
) -> None:
    prices = Prices()
    first = budget(tmp_path, prices, aggregate=True)
    notices: list[str] = []
    first.notice = notices.append
    receipt = first.reserve(
        "apply", (Charge("guardrail", Decimal(1), "text_unit"),)
    )
    assert receipt.amount_usd == Decimal("0.000002")
    assert first.ledger.snapshot().reserved_usd == receipt.amount_usd
    assert first.aggregate.snapshot().reserved_usd == receipt.amount_usd
    second = budget(tmp_path, prices, aggregate=True)
    assert (
        second.reserve(
            "apply", (Charge("guardrail", Decimal(1), "text_unit"),)
        )
        == receipt
    )
    assert "not an AWS billing ceiling" in notices[0]
    first.reserve_amount("delete", Decimal(0), kind="cleanup")
    assert first.ledger.snapshot().cleanup_reserved_usd == 0


@pytest.mark.parametrize("quantity", ["0", "-1", "NaN", "Infinity"])
def test_invalid_quantity_never_reaches_pricing(
    tmp_path: Path, quantity: str
) -> None:
    prices = Prices()
    with pytest.raises(PolicyValidationError, match="quantities"):
        budget(tmp_path, prices).quote(
            (Charge("feature", Decimal(quantity), "unit"),)
        )
    assert prices.calls == []


def test_unknown_or_invalid_prices_refuse_before_reserving(
    tmp_path: Path,
) -> None:
    for rate in ("-1", "NaN", "Infinity"):
        with pytest.raises(ReservationUnavailable, match="price"):
            budget(tmp_path, Prices(rate)).reserve(
                "bad", (Charge("feature", Decimal(1), "unit"),)
            )
    with pytest.raises(ReservationUnavailable, match="explicit"):
        budget(tmp_path, Prices()).quote(())
    with pytest.raises(BudgetExceeded):
        budget(tmp_path, Prices("1")).reserve(
            "expensive", (Charge("feature", Decimal(1), "unit"),)
        )


def test_model_count_and_both_token_directions_are_reserved(
    tmp_path: Path,
) -> None:
    class Counter:
        def count_tokens(
            self, model_id: str, serialized_request: bytes
        ) -> int:
            assert model_id == "test-model"
            assert serialized_request == b'{"messages":[]}'
            return 100

    prices = Prices()
    session = budget(tmp_path, prices)
    receipt, count = session.reserve_model(
        "model-1",
        model_id="test-model",
        serialized_request=b'{"messages":[]}',
        exact_counter=Counter(),
    )
    assert count.tokens == 100
    reserved_tokens = (100 + session.policy.max_output_tokens) * (
        session.policy.total_max_attempts
    )
    assert receipt.amount_usd == Decimal("0.000001") * reserved_tokens
    assert prices.calls == [
        "model:test-model:input",
        "model:test-model:output",
    ]
    assert session.ledger.snapshot().model_calls == 1
    session.policy = ExecutionPolicy(max_input_tokens=50)
    with pytest.raises(InputTooLarge):
        session.reserve_model(
            "model-2",
            model_id="test-model",
            serialized_request=b'{"messages":[]}',
            exact_counter=Counter(),
        )
