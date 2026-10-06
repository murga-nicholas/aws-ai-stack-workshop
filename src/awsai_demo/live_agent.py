"""Priced provider models shared by the executable agent demos."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

from awsai_demo import billing
from awsai_demo import strands_multiagent_demo as shared
from awsai_demo.policy import Charge
from awsai_demo.runtime import TESTED_LIVE_BEDROCK_MODEL

if TYPE_CHECKING:
    from collections.abc import Callable

    from awsai_demo.billing import BudgetRun
    from awsai_demo.contracts import OperationOutcome
    from awsai_demo.credentials import SelectedSession
    from awsai_demo.policy import ExecutionPolicy, TokenCounter
    from awsai_demo.runtime import Settings


class LiveAgentModel(shared.ReservingModel):
    """Reserve every real model dispatch and retain partial outcomes."""

    def __init__(
        self,
        *,
        demo: str,
        settings: Settings,
        policy: ExecutionPolicy,
        run: BudgetRun | None = None,
        selected_session: SelectedSession | None = None,
        model_factory: shared.ModelFactory | None = None,
        exact_counter: TokenCounter | None = None,
        budget_opener: Callable[..., BudgetRun] | None = None,
    ) -> None:
        """Resolve prices before credential selection and dispatch."""
        model_id = settings.model or TESTED_LIVE_BEDROCK_MODEL
        self.settings = settings
        if run is None:
            open_budget = (
                billing.active_budget_run
                if budget_opener is None
                else budget_opener
            )
            run = open_budget(region=settings.region, policy=policy)
        self.run = run
        budget = self.run.command(demo)
        # Refuse missing prices before selecting credentials.
        budget.quote(
            tuple(
                Charge(
                    f"model:{model_id}:{direction}",
                    Decimal(1),
                    "token",
                    routing=shared.model_routing(model_id),
                )
                for direction in ("input", "output")
            )
        )
        selection, self.credential_source = shared.live_model_selection(
            settings=settings,
            policy=policy,
            model_id=model_id,
            model_factory=model_factory,
            selected_session=selected_session,
        )
        self.calls: list[str] = []
        self.completed: list[shared.ReservedLiveCall] = []
        self.charged: list[shared.ReservedLiveCall] = []
        self.count_operations: list[OperationOutcome] = []
        self.model_id = model_id
        super().__init__(
            label=demo,
            inner=selection.model,
            model_id=model_id,
            calls=self.calls,
            completed=self.completed,
            charged=self.charged,
            count_operations=self.count_operations,
            limits=shared.DispatchLimits(policy),
            budget=budget,
            exact_counter=exact_counter
            or shared.default_counter(selection.model),
            max_output_tokens=policy.max_output_tokens,
            endpoint_url=(
                f"https://bedrock-runtime.{settings.region}.amazonaws.com"
            ),
            reservation_namespace=f"{demo}:{uuid4().hex}",
        )

    @property
    def operations(self) -> list[OperationOutcome]:
        """Return counts, failures and completed provider calls."""
        return [
            *self.count_operations,
            *shared.live_model_operations(
                settings=self.settings, live_calls=self.completed
            ),
        ]
