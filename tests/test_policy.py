from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from awsai_demo.policy import (
    BudgetExceeded,
    BudgetLedger,
    CountingUnavailable,
    ExecutionPolicy,
    InputTooLarge,
    LedgerSnapshot,
    PolicyError,
    PolicyLimitExceeded,
    PolicyValidationError,
    ReservationRequest,
    ReservationUnavailable,
    TokenCount,
    _FileLock,
    count_input_tokens,
    enforce_input_limit,
    estimate_model_reservation,
    reserve_command_and_aggregate,
)

if TYPE_CHECKING:
    from pathlib import Path


class ExactCounter:
    def __init__(self, value: int | None) -> None:
        self.value = value

    def count_tokens(
        self,
        model_id: str,
        serialized_request: bytes,
    ) -> int | None:
        assert model_id
        assert serialized_request
        return self.value


def test_execution_policy_defaults_and_validation() -> None:
    policy = ExecutionPolicy()
    assert policy.retry_count == 1
    assert policy.dispatch_attempts == 2
    assert policy.budget_limit() == Decimal("0.05")
    assert policy.budget_limit(all_live=True) == Decimal("0.50")

    with pytest.raises(PolicyValidationError):
        ExecutionPolicy(max_input_tokens=0)
    with pytest.raises(PolicyValidationError):
        ExecutionPolicy(openai_max_retries=-1, total_max_attempts=1)
    with pytest.raises(PolicyValidationError):
        ExecutionPolicy(total_max_attempts=3, openai_max_retries=1)
    with pytest.raises(PolicyValidationError):
        ExecutionPolicy(max_estimated_usd=Decimal("-0.01"))
    with pytest.raises(PolicyValidationError):
        ExecutionPolicy(
            max_estimated_usd=Decimal("0.10"),
            max_estimated_usd_all=Decimal("0.05"),
        )


def test_token_counting_exact_upper_bound_and_refusal() -> None:
    exact = count_input_tokens(
        "model",
        '{"messages":[]}',
        exact_counter=ExactCounter(7),
    )
    assert exact == TokenCount(
        model_id="model",
        tokens=7,
        method="exact",
        evidence="bedrock-runtime.CountTokens",
    )

    bounded = count_input_tokens(
        "evidence-model",
        "é",
        exact_counter=ExactCounter(None),
        byte_upper_bound_models=frozenset({"evidence-model"}),
    )
    assert bounded.tokens == len("é".encode()) + 256
    assert bounded.method == "upper_bound"

    with pytest.raises(CountingUnavailable):
        count_input_tokens("undocumented", b"payload")
    with pytest.raises(PolicyValidationError):
        count_input_tokens("model", b"payload", exact_counter=ExactCounter(-1))


def test_input_limit_and_model_estimates_are_conservative() -> None:
    policy = ExecutionPolicy(max_input_tokens=3, max_output_tokens=5)
    enforce_input_limit(
        policy,
        TokenCount("model", 3, "exact", "test"),
    )
    with pytest.raises(InputTooLarge):
        enforce_input_limit(
            policy,
            TokenCount("model", 4, "exact", "test"),
        )

    estimate = estimate_model_reservation(
        policy,
        input_tokens=10,
        input_price_per_1k=Decimal("0.001"),
        output_price_per_1k=Decimal("0.003"),
    )
    assert estimate == Decimal("0.000050000")
    tiny = estimate_model_reservation(
        policy,
        input_tokens=1,
        input_price_per_1k=Decimal("0.0000000001"),
        output_price_per_1k=Decimal("0"),
        attempts=1,
    )
    assert tiny == Decimal("0.000000001")

    with pytest.raises(ReservationUnavailable):
        estimate_model_reservation(
            policy,
            input_tokens=1,
            input_price_per_1k=None,
            output_price_per_1k=Decimal("1"),
        )
    with pytest.raises(PolicyValidationError):
        estimate_model_reservation(
            policy,
            input_tokens=-1,
            input_price_per_1k=Decimal("1"),
            output_price_per_1k=Decimal("1"),
        )
    with pytest.raises(PolicyValidationError):
        estimate_model_reservation(
            policy,
            input_tokens=1,
            input_price_per_1k=Decimal("1"),
            output_price_per_1k=Decimal("1"),
            attempts=0,
        )


def test_reservation_request_from_units_validation() -> None:
    request = ReservationRequest.from_units(
        reservation_id="svc",
        kind="service",
        units=Decimal("2"),
        unit_price_usd=Decimal("0.001"),
        attempts=2,
        description="priced service",
    )
    assert request.amount_usd == Decimal("0.004000000")
    assert request.description == "priced service"

    with pytest.raises(PolicyValidationError):
        ReservationRequest("", Decimal("0"), "service")
    with pytest.raises(PolicyValidationError):
        ReservationRequest("x", Decimal("0"), "service", command_id="")
    with pytest.raises(PolicyValidationError):
        ReservationRequest("x", Decimal("-0.1"), "service")
    with pytest.raises(ReservationUnavailable):
        ReservationRequest.from_units(
            reservation_id="missing",
            kind="service",
            units=Decimal("1"),
            unit_price_usd=None,
        )
    with pytest.raises(PolicyValidationError):
        ReservationRequest.from_units(
            reservation_id="negative",
            kind="service",
            units=Decimal("-1"),
            unit_price_usd=Decimal("1"),
        )
    with pytest.raises(PolicyValidationError):
        ReservationRequest.from_units(
            reservation_id="attempts",
            kind="service",
            units=Decimal("1"),
            unit_price_usd=Decimal("1"),
            attempts=0,
        )


def test_budget_ledger_reservations_are_locked_and_idempotent(
    tmp_path: Path,
) -> None:
    ledger = BudgetLedger(
        tmp_path / "run" / "budget.json",
        ExecutionPolicy(max_estimated_usd=Decimal("0.01")),
    )
    request = ReservationRequest(
        "call-1",
        Decimal("0.004"),
        "model",
        "first model call",
    )
    reservation = ledger.reserve(request)
    assert reservation.total_reserved_usd == Decimal("0.004000000")

    second = BudgetLedger(
        tmp_path / "run" / "budget.json",
        ExecutionPolicy(max_estimated_usd=Decimal("0.01")),
    )
    assert second.reserve(request).total_reserved_usd == Decimal("0.004000000")
    snapshot = second.snapshot()
    assert isinstance(snapshot, LedgerSnapshot)
    assert snapshot.reservations["call-1"] == request
    assert snapshot.limit_usd == Decimal("0.01")

    with pytest.raises(PolicyValidationError):
        ledger.reserve(ReservationRequest("call-1", Decimal("0.005"), "model"))
    with pytest.raises(BudgetExceeded):
        ledger.reserve(ReservationRequest("call-2", Decimal("0.010"), "model"))


def test_budget_ledger_cleanup_all_mode_counters_and_clock(
    tmp_path: Path,
) -> None:
    now = 10.0

    def clock() -> float:
        return now

    path = tmp_path / "budget.json"
    policy = ExecutionPolicy(
        max_model_calls=1,
        max_agent_iterations=1,
        max_estimated_usd=Decimal("0.01"),
        max_estimated_usd_all=Decimal("0.02"),
        max_wall_seconds=5,
        cleanup_allowance_usd=Decimal("0.005"),
    )
    normal = BudgetLedger(path, policy, clock=clock)
    with pytest.raises(BudgetExceeded):
        normal.reserve(
            ReservationRequest("too-much", Decimal("0.015"), "model"),
        )

    aggregate = BudgetLedger(path, policy, all_live=True, clock=clock)
    aggregate.reserve(ReservationRequest("all-ok", Decimal("0.015"), "model"))
    aggregate.reserve(
        ReservationRequest("cleanup", Decimal("0.004"), "cleanup"),
    )
    with pytest.raises(BudgetExceeded):
        aggregate.reserve(
            ReservationRequest("cleanup-2", Decimal("0.002"), "cleanup"),
        )

    assert aggregate.record_model_call() == 1
    with pytest.raises(PolicyLimitExceeded):
        aggregate.record_model_call()
    assert aggregate.record_agent_iteration() == 1
    with pytest.raises(PolicyLimitExceeded):
        aggregate.record_agent_iteration()

    now = 20.0
    late_cleanup = aggregate.reserve(
        ReservationRequest("late-cleanup", Decimal("0.001"), "cleanup"),
    )
    assert late_cleanup.cleanup_reserved_usd == Decimal("0.005000000")
    with pytest.raises(PolicyLimitExceeded):
        aggregate.check_wall_clock()


def test_budget_ledger_reopen_uses_persisted_limits(
    tmp_path: Path,
) -> None:
    now = 0.0

    def clock() -> float:
        return now

    path = tmp_path / "budget.json"
    original = BudgetLedger(
        path,
        ExecutionPolicy(
            max_estimated_usd=Decimal("0.01"),
            cleanup_allowance_usd=Decimal("0.002"),
            max_model_calls=1,
            max_wall_seconds=5,
        ),
        clock=clock,
    )
    original.reserve(ReservationRequest("initial", Decimal("0.006"), "model"))
    original.reserve(
        ReservationRequest("cleanup", Decimal("0.001"), "cleanup"),
    )
    assert original.record_model_call() == 1

    reopened = BudgetLedger(
        path,
        ExecutionPolicy(
            max_estimated_usd=Decimal("1.00"),
            max_estimated_usd_all=Decimal("1.00"),
            cleanup_allowance_usd=Decimal("1.00"),
            max_model_calls=99,
            max_wall_seconds=999,
        ),
        clock=clock,
    )
    with pytest.raises(BudgetExceeded):
        reopened.reserve(
            ReservationRequest("bypass", Decimal("0.005"), "model"),
        )
    with pytest.raises(BudgetExceeded):
        reopened.reserve(
            ReservationRequest("cleanup-bypass", Decimal("0.002"), "cleanup"),
        )
    with pytest.raises(PolicyLimitExceeded):
        reopened.record_model_call()

    now = 10.0
    with pytest.raises(PolicyLimitExceeded):
        reopened.reserve(ReservationRequest("late", Decimal("0.001"), "model"))
    cleanup = reopened.reserve(
        ReservationRequest("late-cleanup", Decimal("0.001"), "cleanup"),
    )
    assert cleanup.cleanup_reserved_usd == Decimal("0.002000000")


def test_command_and_aggregate_reservation_is_atomic(
    tmp_path: Path,
) -> None:
    policy = ExecutionPolicy(
        max_estimated_usd=Decimal("0.01"),
        max_estimated_usd_all=Decimal("0.02"),
    )
    command = BudgetLedger(tmp_path / "cmd" / "budget.json", policy)
    aggregate = BudgetLedger(
        tmp_path / "all" / "budget.json",
        policy,
        all_live=True,
    )
    pair = reserve_command_and_aggregate(
        command,
        aggregate,
        ReservationRequest("invoke", Decimal("0.006"), "model"),
        command_id="cmd-a",
    )
    assert pair.command.reservation_id == "invoke"
    assert pair.aggregate.reservation_id == "cmd-a:invoke"
    assert command.snapshot().reservations["invoke"].command_id == "cmd-a"
    assert (
        aggregate.snapshot().reservations["cmd-a:invoke"].command_id == "cmd-a"
    )

    same = reserve_command_and_aggregate(
        command,
        aggregate,
        ReservationRequest("invoke", Decimal("0.006"), "model"),
        command_id="cmd-a",
    )
    assert same.command.total_reserved_usd == Decimal("0.006000000")
    cleanup = reserve_command_and_aggregate(
        command,
        aggregate,
        ReservationRequest("cleanup", Decimal("0.001"), "cleanup"),
        command_id="cmd-a",
    )
    assert cleanup.command.cleanup_reserved_usd == Decimal("0.001000000")


def test_command_and_aggregate_reservation_rejects_invalid_or_partial(
    tmp_path: Path,
) -> None:
    command = BudgetLedger(
        tmp_path / "cmd" / "budget.json",
        ExecutionPolicy(max_estimated_usd=Decimal("0.01")),
    )
    aggregate = BudgetLedger(
        tmp_path / "all" / "budget.json",
        ExecutionPolicy(
            max_estimated_usd=Decimal("0.005"),
            max_estimated_usd_all=Decimal("0.005"),
        ),
        all_live=True,
    )

    with pytest.raises(PolicyValidationError):
        reserve_command_and_aggregate(
            command,
            aggregate,
            ReservationRequest("op", Decimal("0.001"), "model"),
            command_id="",
        )
    with pytest.raises(PolicyValidationError):
        reserve_command_and_aggregate(
            command,
            aggregate,
            ReservationRequest(
                "op",
                Decimal("0.001"),
                "model",
                command_id="other",
            ),
            command_id="cmd-a",
        )
    with pytest.raises(PolicyValidationError):
        reserve_command_and_aggregate(
            command,
            BudgetLedger(command._path, ExecutionPolicy()),
            ReservationRequest("op", Decimal("0.001"), "model"),
            command_id="cmd-a",
        )
    with pytest.raises(BudgetExceeded):
        reserve_command_and_aggregate(
            command,
            aggregate,
            ReservationRequest("op", Decimal("0.006"), "model"),
            command_id="cmd-a",
        )
    assert command.snapshot().reserved_usd == Decimal("0")


def test_budget_ledger_snapshot_without_file_and_lock_timeout(
    tmp_path: Path,
) -> None:
    ledger = BudgetLedger(
        tmp_path / "missing" / "budget.json",
        ExecutionPolicy(),
    )
    assert ledger.snapshot().reserved_usd == Decimal("0")

    lock_path = tmp_path / "held.lock"
    _FileLock(lock_path).__exit__(None, None, None)
    with (
        _FileLock(lock_path),
        pytest.raises(PolicyError),
        _FileLock(lock_path, timeout_seconds=0, poll_seconds=0),
    ):
        pass
