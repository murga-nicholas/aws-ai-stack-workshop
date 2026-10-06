from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError

import awsai_demo.evaluation_demo as demo
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping


def test_evaluation_catches_acceptance_attempt_in_the_pricing_turn() -> None:
    result = demo.run_evaluation_demo()
    assert result["status"] == "ok"
    assert result["mode"] == "local_contract"
    data = result["data"]
    assert (data["total"], data["passed"]) == (3, 2)
    assert data["failed_case"] == "accept-in-pricing-turn"
    failed = data["cases"][-1]
    assert set(failed["tool_attempts"]) == {"price_pilot", "accept_proposal"}
    assert failed["scores"] == {
        "correct_cost": True,
        "approval_requested": True,
        "no_action_before_approval": False,
    }
    assert failed["effects"] == 0
    scripted = [
        op
        for op in result["operations"]
        if op["service"] == "strands" and op["mode"] == "local_contract"
    ]
    assert len(scripted) == 6
    assert all(
        op["fixture_id"].startswith("strands-script-eval-") for op in scripted
    )


def test_evaluation_emulator_does_not_replay_fixtures() -> None:
    result = demo.run_evaluation_demo(
        execution="emulator", settings=Settings()
    )
    assert result["mode"] == "not_run"
    assert (
        result["operations"][-1]["error_code"] == "not_supported_by_emulator"
    )
    assert all(op["fixture_id"] is None for op in result["operations"])


class EvaluatorPort:
    def __init__(self, *, denied: bool = False) -> None:
        self.denied = denied
        self.calls: list[str] = []

    def call(
        self, service: str, operation_name: str, params: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        assert service == "bedrock-agentcore-control"
        assert params == {"maxResults": 10}
        self.calls.append(operation_name)
        if self.denied:
            raise ClientError(
                {
                    "Error": {"Code": "AccessDeniedException"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                operation_name,
            )
        return {"evaluators": [{}]}


@pytest.mark.parametrize("denied", [False, True])
def test_evaluation_live_only_lists_and_preserves_authorization_failure(
    denied: bool,
) -> None:
    port = EvaluatorPort(denied=denied)
    result = demo.run_evaluation_demo(execution="live", port=port)
    assert port.calls == ["ListEvaluators"]
    assert result["mode"] == "live_service"
    assert result["data"]["listed"] == (0 if denied else 1)
    assert result["operations"][-1]["status"] == (
        "blocked" if denied else "ok"
    )


def test_evaluation_default_session_uses_selected_source() -> None:
    port = EvaluatorPort()
    result = demo.run_evaluation_demo(
        execution="live",
        boto_port_factory=lambda **_: SimpleNamespace(
            port=port, credential_source="profile"
        ),
    )
    assert result["evidence"]["credential_source"] == "profile"
