"""Execution limits, token accounting and local budget reservations."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from decimal import ROUND_CEILING, Decimal
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, Self, cast

from filelock import FileLock, Timeout

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import TracebackType

ReservationKind = Literal["model", "service", "cleanup"]
TokenCountMethod = Literal["exact", "upper_bound"]

BYTE_UPPER_BOUND_EVIDENCE: Final[dict[str, dict[str, str]]] = {
    model: {
        "tokenizer": "o200k_harmony byte-level BPE",
        "source": (
            "https://github.com/openai/gpt-oss/blob/main/gpt_oss/tokenizer.py"
        ),
        "encoding_source": (
            "https://github.com/openai/tiktoken/blob/main/"
            "tiktoken_ext/openai_public.py"
        ),
        "verified": "2026-10-06",
        "scope": "Complete serialized text requests plus 256 framing tokens",
    }
    for model in ("openai.gpt-oss-20b-1:0", "openai.gpt-oss-120b-1:0")
}
DEFAULT_BYTE_UPPER_BOUND_MODELS: Final[frozenset[str]] = frozenset(
    BYTE_UPPER_BOUND_EVIDENCE,
)
_PROTOCOL_OVERHEAD_TOKENS: Final = 256
_USD_QUANTUM: Final = Decimal("0.000000001")


class PolicyError(Exception):
    """Base class for execution-policy failures."""


class PolicyValidationError(PolicyError):
    """Raised when a policy or reservation request is invalid."""


class CountingUnavailableError(PolicyError):
    """Raised when input cannot be counted conservatively."""


class InputTooLargeError(PolicyError):
    """Raised when a model request exceeds the input token limit."""


class ReservationUnavailableError(PolicyError):
    """Raised when a live operation cannot be priced before dispatch."""


class BudgetExceededError(PolicyError):
    """Raised when a durable reservation exceeds the local budget."""

    code = "budget_exceeded"


class PolicyLimitExceededError(PolicyError):
    """Raised when a non-budget execution limit would be exceeded."""


CountingUnavailable = CountingUnavailableError
InputTooLarge = InputTooLargeError
ReservationUnavailable = ReservationUnavailableError
BudgetExceeded = BudgetExceededError
PolicyLimitExceeded = PolicyLimitExceededError


class TokenCounter(Protocol):
    """Counts tokens for a complete serialized model request."""

    def count_tokens(
        self,
        model_id: str,
        serialized_request: bytes,
    ) -> int | None:
        """Return an exact count, or None when unsupported."""


class PriceResolver(Protocol):
    """Resolve a verified USD rate for one precisely named unit."""

    def rate(
        self,
        feature: str,
        *,
        region: str,
        unit: str,
        tier: str,
        routing: str,
    ) -> Decimal:
        """Return USD per unit, refusing missing or ambiguous prices."""


@dataclass(frozen=True, slots=True)
class Charge:
    """Billable quantity with explicit pricing dimensions."""

    feature: str
    quantity: Decimal
    unit: str
    tier: str = "standard"
    routing: str = "in-region"


class PricedBudget:
    """Resolve prices and reserve before dispatch in shared ledgers."""

    def __init__(
        self,
        *,
        prices: PriceResolver,
        policy: ExecutionPolicy,
        region: str,
        ledger: BudgetLedger,
        aggregate: BudgetLedger | None = None,
        command_id: str = "command",
        notice: Callable[[str], None] | None = None,
    ) -> None:
        """Bind prices to command and aggregate budgets."""
        self.prices = prices
        self.policy = policy
        self.region = region
        self.ledger = ledger
        self.aggregate = aggregate
        self.command_id = command_id
        self.notice = notice

    def quote(self, charges: tuple[Charge, ...]) -> Decimal:
        """Conservatively price every unit, including SDK retries."""
        if not charges:
            message = "billable operation needs explicit priced units"
            raise ReservationUnavailable(message)
        total = Decimal("0")
        for charge in charges:
            if not charge.quantity.is_finite() or charge.quantity <= 0:
                message = "billable quantities must be finite and positive"
                raise PolicyValidationError(message)
            rate = self.prices.rate(
                charge.feature,
                region=self.region,
                unit=charge.unit,
                tier=charge.tier,
                routing=charge.routing,
            )
            if not rate.is_finite() or rate < 0:
                message = "price must be finite and nonnegative"
                raise ReservationUnavailable(message)
            total += rate * charge.quantity
        return _ceil_usd(total * self.policy.dispatch_attempts)

    def reserve(
        self,
        reservation_id: str,
        charges: tuple[Charge, ...],
        *,
        kind: ReservationKind = "service",
    ) -> Reservation:
        """Persist the full priced bound before returning permission."""
        return self.reserve_amount(
            reservation_id, self.quote(charges), kind=kind
        )

    def reserve_amount(
        self,
        reservation_id: str,
        amount_usd: Decimal,
        *,
        kind: ReservationKind,
    ) -> Reservation:
        """Persist a bound, including free control-plane calls."""
        request = ReservationRequest(reservation_id, amount_usd, kind)
        if self.aggregate is None:
            reservation = self.ledger.reserve(request)
        else:
            reservation = reserve_command_and_aggregate(
                self.ledger,
                self.aggregate,
                request,
                command_id=self.command_id,
            ).command
        if self.notice is not None:
            self.notice(
                f"Reserved USD {reservation.amount_usd:.9f}; "
                "application-enforced limits, not an AWS billing ceiling"
            )
        return reservation

    def reserve_model(
        self,
        reservation_id: str,
        *,
        model_id: str,
        serialized_request: bytes,
        exact_counter: TokenCounter | None = None,
        tier: str = "standard",
        routing: str = "in-region",
    ) -> tuple[Reservation, TokenCount]:
        """Count complete input and reserve input and output tokens."""
        count = count_input_tokens(
            model_id, serialized_request, exact_counter=exact_counter
        )
        enforce_input_limit(self.policy, count)
        amount = self.quote(
            (
                Charge(
                    f"model:{model_id}:input",
                    Decimal(max(1, count.tokens)),
                    "token",
                    tier,
                    routing,
                ),
                Charge(
                    f"model:{model_id}:output",
                    Decimal(self.policy.max_output_tokens),
                    "token",
                    tier,
                    routing,
                ),
            )
        )
        self.ledger.record_model_call()
        reservation = self.reserve_amount(reservation_id, amount, kind="model")
        return reservation, count


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Application-enforced limits for one command run."""

    max_input_tokens: int = 2_000
    max_output_tokens: int = 512
    max_model_calls: int = 6
    max_agent_iterations: int = 4
    total_max_attempts: int = 2
    openai_max_retries: int = 1
    max_wall_seconds: int = 120
    max_estimated_usd: Decimal = Decimal("0.05")
    max_estimated_usd_all: Decimal = Decimal("0.50")
    cleanup_allowance_usd: Decimal = Decimal("0.01")

    def __post_init__(self) -> None:
        """Validate the frozen policy after dataclass initialization."""
        positive_ints = {
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_model_calls": self.max_model_calls,
            "max_agent_iterations": self.max_agent_iterations,
            "total_max_attempts": self.total_max_attempts,
            "max_wall_seconds": self.max_wall_seconds,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                message = f"{name} must be positive"
                raise PolicyValidationError(message)
        if self.openai_max_retries < 0:
            message = "openai_max_retries must not be negative"
            raise PolicyValidationError(message)
        if self.openai_max_retries != self.retry_count:
            message = "retry settings must describe one shared retry layer"
            raise PolicyValidationError(message)
        budgets = {
            "max_estimated_usd": self.max_estimated_usd,
            "max_estimated_usd_all": self.max_estimated_usd_all,
            "cleanup_allowance_usd": self.cleanup_allowance_usd,
        }
        for budget_name, budget_value in budgets.items():
            if budget_value < Decimal("0"):
                message = f"{budget_name} must not be negative"
                raise PolicyValidationError(message)
        if self.max_estimated_usd_all < self.max_estimated_usd:
            message = "all-mode budget must be at least the single-run budget"
            raise PolicyValidationError(message)

    @property
    def retry_count(self) -> int:
        """Return retries allowed after the first attempt."""
        return self.total_max_attempts - 1

    @property
    def dispatch_attempts(self) -> int:
        """Return the dispatch reservation multiplier."""
        return self.total_max_attempts

    def budget_limit(self, *, all_live: bool = False) -> Decimal:
        """Return the normal or aggregate live budget."""
        if all_live:
            return self.max_estimated_usd_all
        return self.max_estimated_usd


@dataclass(frozen=True, slots=True)
class TokenCount:
    """A token count and the method used to obtain it."""

    model_id: str
    tokens: int
    method: TokenCountMethod
    evidence: str


@dataclass(frozen=True, slots=True)
class ReservationRequest:
    """A durable budget reservation requested before dispatch."""

    reservation_id: str
    amount_usd: Decimal
    kind: ReservationKind
    description: str = ""
    command_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the reservation request."""
        if not self.reservation_id:
            message = "reservation_id must not be empty"
            raise PolicyValidationError(message)
        if self.command_id == "":
            message = "command_id must not be empty"
            raise PolicyValidationError(message)
        if self.amount_usd < Decimal("0"):
            message = "amount_usd must not be negative"
            raise PolicyValidationError(message)

    @classmethod
    def from_units(
        cls,
        *,
        reservation_id: str,
        kind: ReservationKind,
        units: Decimal,
        unit_price_usd: Decimal | None,
        attempts: int = 1,
        description: str = "",
    ) -> Self:
        """Build a service reservation from priced units."""
        if unit_price_usd is None:
            message = "missing price row for live reservation"
            raise ReservationUnavailable(message)
        if units < Decimal("0"):
            message = "units must not be negative"
            raise PolicyValidationError(message)
        if attempts <= 0:
            message = "attempts must be positive"
            raise PolicyValidationError(message)
        amount = units * unit_price_usd * Decimal(attempts)
        return cls(
            reservation_id=reservation_id,
            amount_usd=_ceil_usd(amount),
            kind=kind,
            description=description,
        )


@dataclass(frozen=True, slots=True)
class Reservation:
    """A reservation persisted in the shared budget ledger."""

    reservation_id: str
    amount_usd: Decimal
    kind: ReservationKind
    total_reserved_usd: Decimal
    cleanup_reserved_usd: Decimal


@dataclass(frozen=True, slots=True)
class ReservationPair:
    """Per-command and aggregate reservations recorded atomically."""

    command: Reservation
    aggregate: Reservation


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """A typed view of the durable budget ledger."""

    limit_usd: Decimal
    cleanup_limit_usd: Decimal
    reserved_usd: Decimal
    cleanup_reserved_usd: Decimal
    model_calls: int
    agent_iterations: int
    started_at: float
    reservations: dict[str, ReservationRequest]


class BudgetLedger:
    """Locked JSON ledger shared by parent and child processes."""

    def __init__(
        self,
        path: Path,
        policy: ExecutionPolicy,
        *,
        all_live: bool = False,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Create a ledger wrapper without touching the filesystem."""
        self._path = path
        self._policy = policy
        self._all_live = all_live
        self._clock = time.time if clock is None else clock
        self._lock = _FileLock(path.with_name(f"{path.name}.lock"))

    def snapshot(self) -> LedgerSnapshot:
        """Return the current ledger state under the file lock."""
        with self._lock:
            state = self._load_state()
        return _state_to_snapshot(state)

    def reserve(self, request: ReservationRequest) -> Reservation:
        """Persist an idempotent reservation before a live operation."""
        if request.kind != "cleanup":
            self.check_wall_clock()
        with self._lock:
            state = self._load_state()
            reservation = _apply_reservation_to_state(state, request)
            self._save_state(state)
            return reservation

    def record_model_call(self) -> int:
        """Increment and persist the model-call counter."""
        with self._lock:
            state = self._load_state()
            value = int(state["model_calls"]) + 1
            if value > int(state["max_model_calls"]):
                message = "model-call limit exceeded"
                raise PolicyLimitExceeded(message)
            state["model_calls"] = value
            self._save_state(state)
            return value

    def record_agent_iteration(self) -> int:
        """Increment and persist the agent-iteration counter."""
        with self._lock:
            state = self._load_state()
            value = int(state["agent_iterations"]) + 1
            if value > int(state["max_agent_iterations"]):
                message = "agent-iteration limit exceeded"
                raise PolicyLimitExceeded(message)
            state["agent_iterations"] = value
            self._save_state(state)
            return value

    def check_wall_clock(self) -> None:
        """Refuse work after the configured wall-clock budget."""
        with self._lock:
            state = self._load_state()
        self._check_wall_clock_state(state)

    def _check_wall_clock_state(self, state: dict[str, Any]) -> None:
        elapsed = float(self._clock()) - float(state["started_at"])
        if elapsed > int(state["max_wall_seconds"]):
            message = "wall-clock limit exceeded"
            raise PolicyLimitExceeded(message)

    def _load_state(self) -> dict[str, Any]:
        if not self._path.exists():
            return {
                "limit_usd": str(
                    self._policy.budget_limit(all_live=self._all_live),
                ),
                "cleanup_limit_usd": str(self._policy.cleanup_allowance_usd),
                "reserved_usd": "0",
                "cleanup_reserved_usd": "0",
                "max_model_calls": self._policy.max_model_calls,
                "max_agent_iterations": self._policy.max_agent_iterations,
                "max_wall_seconds": self._policy.max_wall_seconds,
                "model_calls": 0,
                "agent_iterations": 0,
                "started_at": float(self._clock()),
                "reservations": {},
            }
        text = self._path.read_text(encoding="utf-8")
        data = json.loads(text)
        return cast("dict[str, Any]", data)

    def _save_state(self, state: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f"{self._path.name}.tmp")
        text = json.dumps(state, indent=2, sort_keys=True)
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(f"{text}\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self._path)


def reserve_command_and_aggregate(
    command_ledger: BudgetLedger,
    aggregate_ledger: BudgetLedger,
    request: ReservationRequest,
    *,
    command_id: str,
) -> ReservationPair:
    """Atomically reserve per-command and aggregate live budgets.

    Use this helper for ``all --execution live``. The command ledger
    enforces the per-command limit; the aggregate ledger enforces the
    all-run total. Cleanup uses each ledger's cleanup allowance.
    """
    if not command_id:
        message = "command_id must not be empty"
        raise PolicyValidationError(message)
    if request.command_id not in (None, command_id):
        message = "request command_id does not match the aggregate command"
        raise PolicyValidationError(message)
    if command_ledger._path == aggregate_ledger._path:
        message = "command and aggregate ledgers must be separate files"
        raise PolicyValidationError(message)

    command_request = replace(request, command_id=command_id)
    aggregate_request = replace(
        request,
        reservation_id=f"{command_id}:{request.reservation_id}",
        command_id=command_id,
    )
    first, second = _ordered_ledgers(command_ledger, aggregate_ledger)
    with first._lock, second._lock:
        command_state = command_ledger._load_state()
        aggregate_state = aggregate_ledger._load_state()
        if request.kind != "cleanup":
            command_ledger._check_wall_clock_state(command_state)
        command_reservation = _apply_reservation_to_state(
            command_state,
            command_request,
        )
        aggregate_reservation = _apply_reservation_to_state(
            aggregate_state,
            aggregate_request,
        )
        command_ledger._save_state(command_state)
        aggregate_ledger._save_state(aggregate_state)
    return ReservationPair(
        command=command_reservation,
        aggregate=aggregate_reservation,
    )


def count_input_tokens(
    model_id: str,
    serialized_request: str | bytes,
    *,
    exact_counter: TokenCounter | None = None,
    byte_upper_bound_models: frozenset[str] = DEFAULT_BYTE_UPPER_BOUND_MODELS,
) -> TokenCount:
    """Count tokens exactly or by a documented byte upper bound."""
    payload = (
        serialized_request
        if isinstance(serialized_request, bytes)
        else serialized_request.encode("utf-8")
    )
    if exact_counter is not None:
        exact = exact_counter.count_tokens(model_id, payload)
        if exact is not None:
            if exact < 0:
                message = "exact token counter returned a negative value"
                raise PolicyValidationError(message)
            return TokenCount(
                model_id=model_id,
                tokens=exact,
                method="exact",
                evidence="bedrock-runtime.CountTokens",
            )
    if model_id not in byte_upper_bound_models:
        message = "model lacks exact counter and documented byte bound"
        raise CountingUnavailable(message)
    return TokenCount(
        model_id=model_id,
        tokens=len(payload) + _PROTOCOL_OVERHEAD_TOKENS,
        method="upper_bound",
        evidence="utf8_bytes_plus_protocol_overhead",
    )


def enforce_input_limit(policy: ExecutionPolicy, count: TokenCount) -> None:
    """Raise when a request exceeds the configured input-token limit."""
    if count.tokens > policy.max_input_tokens:
        message = "input token limit exceeded"
        raise InputTooLarge(message)


def estimate_model_reservation(
    policy: ExecutionPolicy,
    *,
    input_tokens: int,
    input_price_per_1k: Decimal | None,
    output_price_per_1k: Decimal | None,
    attempts: int | None = None,
) -> Decimal:
    """Estimate the pre-dispatch reservation for one model call."""
    if input_price_per_1k is None or output_price_per_1k is None:
        message = "missing model price row for live reservation"
        raise ReservationUnavailable(message)
    if input_tokens < 0:
        message = "input_tokens must not be negative"
        raise PolicyValidationError(message)
    attempt_count = policy.dispatch_attempts if attempts is None else attempts
    if attempt_count <= 0:
        message = "attempts must be positive"
        raise PolicyValidationError(message)
    input_cost = Decimal(input_tokens) * input_price_per_1k
    output_cost = Decimal(policy.max_output_tokens) * output_price_per_1k
    total = (input_cost + output_cost) / Decimal(1_000) * attempt_count
    return _ceil_usd(total)


def _ceil_usd(value: Decimal) -> Decimal:
    return value.quantize(_USD_QUANTUM, rounding=ROUND_CEILING)


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _ordered_ledgers(
    first: BudgetLedger,
    second: BudgetLedger,
) -> tuple[BudgetLedger, BudgetLedger]:
    ledgers = sorted(
        (first, second),
        key=lambda ledger: str(ledger._lock.path),
    )
    return ledgers[0], ledgers[1]


def _apply_reservation_to_state(
    state: dict[str, Any],
    request: ReservationRequest,
) -> Reservation:
    existing = _reservation_from_state(
        state["reservations"],
        request.reservation_id,
    )
    if existing is not None:
        _assert_same_reservation(existing, request)
        return _reservation_result(request, state)

    cleanup = request.kind == "cleanup"
    key = "cleanup_reserved_usd" if cleanup else "reserved_usd"
    limit_key = "cleanup_limit_usd" if cleanup else "limit_usd"
    new_total = _decimal(state[key]) + request.amount_usd
    if new_total > _decimal(state[limit_key]):
        message = "reservation would exceed local budget"
        raise BudgetExceeded(message)
    state[key] = str(_ceil_usd(new_total))
    reservations = cast("dict[str, Any]", state["reservations"])
    reservations[request.reservation_id] = _reservation_to_json(request)
    return _reservation_result(request, state)


def _reservation_to_json(
    request: ReservationRequest,
) -> dict[str, str | None]:
    return {
        "reservation_id": request.reservation_id,
        "amount_usd": str(request.amount_usd),
        "kind": request.kind,
        "description": request.description,
        "command_id": request.command_id,
    }


def _reservation_from_state(
    reservations: Any,
    reservation_id: str,
) -> ReservationRequest | None:
    values = cast("dict[str, Any]", reservations)
    if reservation_id not in values:
        return None
    raw = cast("dict[str, Any]", values[reservation_id])
    return ReservationRequest(
        reservation_id=str(raw["reservation_id"]),
        amount_usd=_decimal(raw["amount_usd"]),
        kind=cast("ReservationKind", raw["kind"]),
        description=str(raw["description"]),
        command_id=_optional_str(raw.get("command_id")),
    )


def _assert_same_reservation(
    existing: ReservationRequest,
    requested: ReservationRequest,
) -> None:
    if existing != requested:
        message = "reservation id reused with different contents"
        raise PolicyValidationError(message)


def _reservation_result(
    request: ReservationRequest,
    state: dict[str, Any],
) -> Reservation:
    return Reservation(
        reservation_id=request.reservation_id,
        amount_usd=request.amount_usd,
        kind=request.kind,
        total_reserved_usd=_decimal(state["reserved_usd"]),
        cleanup_reserved_usd=_decimal(state["cleanup_reserved_usd"]),
    )


def _state_to_snapshot(state: dict[str, Any]) -> LedgerSnapshot:
    raw_reservations = cast("dict[str, Any]", state["reservations"])
    reservations = {
        key: _reservation_from_state(raw_reservations, key)
        for key in raw_reservations
    }
    return LedgerSnapshot(
        limit_usd=_decimal(state["limit_usd"]),
        cleanup_limit_usd=_decimal(state["cleanup_limit_usd"]),
        reserved_usd=_decimal(state["reserved_usd"]),
        cleanup_reserved_usd=_decimal(state["cleanup_reserved_usd"]),
        model_calls=int(state["model_calls"]),
        agent_iterations=int(state["agent_iterations"]),
        started_at=float(state["started_at"]),
        reservations={
            key: value
            for key, value in reservations.items()
            if value is not None
        },
    )


@dataclass(slots=True)
class _FileLock:
    path: Path
    timeout_seconds: float = 10.0
    poll_seconds: float = 0.01
    _lock: FileLock | None = field(default=None, init=False)

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(self.path)
        try:
            lock.acquire(
                timeout=self.timeout_seconds,
                poll_interval=self.poll_seconds,
            )
        except Timeout as error:
            message = "timed out waiting for budget ledger lock"
            raise PolicyError(message) from error
        self._lock = lock
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._lock is not None:
            self._lock.release()
            self._lock = None
