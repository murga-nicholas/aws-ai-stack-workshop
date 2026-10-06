"""Small operation tables for bounded service demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from awsai_demo.billing import active_budget_run
from awsai_demo.demo_support import (
    DemoOperation,
    billable_not_priced,
    client_method_name,
    contract_only,
    fixture_operation,
    port_operation,
    require_port_not_run,
    unsupported_emulator,
)
from awsai_demo.policy import PolicyError
from awsai_demo.stubs import stubbed_client

if TYPE_CHECKING:
    from collections.abc import Mapping

    from awsai_demo.contracts import Effect
    from awsai_demo.demo_support import AwsPort
    from awsai_demo.policy import Charge, ExecutionPolicy
    from awsai_demo.runtime import Execution, Settings


@dataclass(frozen=True)
class ServiceRow:
    """One documented API row and its explicit lane support."""

    service: str
    operation: str
    params: Mapping[str, Any]
    response: Mapping[str, Any]
    effect: Effect = "read"
    emulator: bool = False
    live: bool = True
    charges: tuple[Charge, ...] = ()

    def run(
        self,
        *,
        execution: Execution,
        settings: Settings,
        policy: ExecutionPolicy,
        port: AwsPort | None,
        command: str,
    ) -> DemoOperation:
        """Dispatch the selected lane; reserve before paid calls."""
        fixture = f"{command}-{self.operation}-v1"
        if execution == "offline":
            method = client_method_name(self.operation)
            with stubbed_client(self.service) as stubber:
                stubber.add_response(
                    method, self.response, expected_params=self.params
                )
                payload = getattr(stubber.client, method)(**self.params)
            return DemoOperation(
                fixture_operation(
                    service=self.service,
                    operation_name=self.operation,
                    fixture_id=fixture,
                    effect=self.effect,
                ),
                payload,
            )
        if execution == "emulator" and not self.emulator:
            return DemoOperation(
                unsupported_emulator(
                    service=self.service, operation_name=self.operation
                ),
                {},
            )
        if execution == "live" and not self.live:
            return DemoOperation(
                contract_only(
                    service=self.service, operation_name=self.operation
                ),
                {},
            )
        if port is None:
            return DemoOperation(
                require_port_not_run(
                    service=self.service, operation_name=self.operation
                ),
                {},
            )
        reservation = None
        if execution == "live" and self.charges:
            try:
                budget = active_budget_run(
                    region=settings.region, policy=policy
                ).command(command)
                reservation = budget.reserve(
                    f"{self.service}:{self.operation}", self.charges
                )
            except PolicyError:
                return DemoOperation(
                    billable_not_priced(
                        service=self.service, operation_name=self.operation
                    ),
                    {},
                )
        endpoint = (
            settings.localstack_endpoint
            if execution == "emulator"
            else f"https://{self.service}.{settings.region}.amazonaws.com"
        )
        response = port_operation(
            port=port,
            service=self.service,
            operation_name=self.operation,
            params=self.params,
            execution=execution,
            effect=self.effect,
            fixture_id=fixture,
            endpoint_url=endpoint,
            reserved=reservation is not None,
        )
        if reservation is not None:
            response.outcome["reserved_usd"] = float(reservation.amount_usd)
        return response
