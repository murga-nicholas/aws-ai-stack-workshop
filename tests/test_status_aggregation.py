from __future__ import annotations

from typing import Any

import pytest

from awsai_demo.contracts import error, evidence, operation, result


def report(operations: list[Any], execution: str = "live") -> Any:
    return result(
        demo="aggregation",
        technology="Operation outcomes",
        lane="operations",
        lifecycle_refs=["bedrock"],
        requested_execution=execution,
        headline="Executed work determines success.",
        operations=operations,
        evidence=evidence(emulator={"endpoint": "http://localhost:4566"})
        if execution == "emulator"
        else None,
    )


def live_call(**changes: Any) -> Any:
    return operation(
        **{
            "service": "bedrock",
            "operation": "ListFoundationModels",
            "mode": "live_service",
            "execution_target": "aws",
            "transport": "aws",
            "endpoint_url": "https://bedrock.us-east-1.amazonaws.com",
            **changes,
        }
    )


def skipped(code: str, **changes: Any) -> Any:
    return operation(
        service="bedrock",
        operation="OptionalExample",
        mode="not_run",
        status=changes.pop("status", "blocked"),
        execution_target="aws",
        response_received=False,
        error_code=code,
        **changes,
    )


@pytest.mark.parametrize(
    "code",
    [
        "contract_only",
        "create_not_allowed",
        "invoke_not_allowed",
        "not_supported_by_emulator",
    ],
)
def test_deliberate_skip_keeps_its_reason_without_poisoning_success(
    code: str,
) -> None:
    omitted = skipped(code)
    combined = report([live_call(), omitted])
    assert combined["status"] == "ok"
    assert combined["mode"] == "live_service"
    assert combined["operations"][1]["error_code"] == code
    assert combined["operations"][1]["status"] == "blocked"
    assert report([omitted])["status"] == "blocked"
    assert report([omitted])["mode"] == "not_run"


def test_optional_configuration_is_explicit_and_required_stays_blocked() -> (
    None
):
    assert (
        report([live_call(), skipped("missing_configuration", optional=True)])[
            "status"
        ]
        == "ok"
    )
    assert (
        report([live_call(), skipped("missing_configuration")])["status"]
        == "blocked"
    )


@pytest.mark.parametrize(
    "code",
    ["budget_exceeded", "authorization_denied", "subscription_required"],
)
def test_real_refusals_are_never_suppressed(code: str) -> None:
    assert report([live_call(), skipped(code, optional=True)])["status"] == (
        "blocked"
    )
    denied = live_call(status="blocked", error_code=code, http_status=400)
    assert report([live_call(), denied])["status"] == "blocked"


def test_errors_and_attempt_failures_cannot_be_downgraded() -> None:
    assert (
        report([live_call(), skipped("contract_only", status="error")])[
            "status"
        ]
        == "error"
    )
    failed = live_call(
        mode="attempt_failed", status="blocked", response_received=False
    )
    assert report([failed, skipped("contract_only")])["status"] == "blocked"
    assert report([failed])["mode"] == "attempt_failed"


def test_local_work_cannot_turn_an_unexecuted_live_lane_green() -> None:
    local = operation(
        service="workshop",
        operation="Explain",
        mode="local_execution",
        execution_target="local",
    )
    combined = report([local, skipped("create_not_allowed")])
    assert combined["status"] == "blocked"
    assert combined["mode"] == "local_execution"
    assert report([local], "offline")["status"] == "ok"
    emulator = operation(
        service="bedrock-runtime",
        operation="Converse",
        mode="local_emulator",
        execution_target="emulator",
        transport="loopback",
        endpoint_url="http://localhost:4566",
    )
    assert (
        report([emulator, skipped("not_supported_by_emulator")], "emulator")[
            "status"
        ]
        == "ok"
    )


def test_batch_keeps_blocked_children_visible() -> None:
    good = report([live_call(), skipped("contract_only")])
    unavailable = report([skipped("contract_only")])
    batch = result(
        demo="all",
        technology="Catalogue",
        lane="operations",
        lifecycle_refs=["bedrock"],
        requested_execution="live",
        headline="Child results remain visible.",
        children=[good, unavailable],
    )
    assert batch["status"] == "blocked"


def test_partial_success_cannot_hide_a_top_level_timeout() -> None:
    base = report([live_call()])
    base.pop("schema_version")
    base.pop("status")
    base["error"] = error("timeout", "Later model call exceeded wall time.")
    assert result(**base)["status"] == "blocked"


@pytest.mark.parametrize("execution", ["live", "emulator"])
def test_empty_external_lane_cannot_claim_success(execution: str) -> None:
    assert report([], execution)["status"] == "blocked"
