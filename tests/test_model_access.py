from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError

from awsai_demo.contracts import error, operation, result
from awsai_demo.demo_support import port_operation, public_aws_error_code
from awsai_demo.kendra_demo import run_kendra_demo
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings
from awsai_demo.strands_multiagent_demo import _live_failure_operation

ACCESS_MESSAGE = (
    "Model use case details have not been submitted for this account. "
    "Fill out the Anthropic use case details form before using the model."
)


@pytest.mark.parametrize(
    ("code", "message", "expected"),
    [
        ("ResourceNotFoundException", ACCESS_MESSAGE, "model_access_required"),
        (
            "ResourceNotFoundException",
            "Model not found",
            "ResourceNotFoundException",
        ),
        (
            "ResourceNotFoundException",
            "Model use case details have not been submitted",
            "ResourceNotFoundException",
        ),
        ("ValidationException", ACCESS_MESSAGE, "ValidationException"),
    ],
)
def test_access_diagnostic_is_narrow_and_shared(
    code: str, message: str, expected: str
) -> None:
    payload = {"Code": code, "Message": message}
    assert public_aws_error_code(payload, 404) == expected
    failure = ClientError(
        {"Error": payload, "ResponseMetadata": {"HTTPStatusCode": 404}},
        "Converse",
    )

    class Port:
        endpoint_url = "https://bedrock-runtime.us-east-1.amazonaws.com"

        def call(self, *_args: Any) -> Any:
            raise failure

    called = port_operation(
        port=Port(),
        service="bedrock-runtime",
        operation_name="Converse",
        params={
            "modelId": "fixture",
            "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
        },
        execution="live",
        effect="infer",
        reserved=True,
        fixture_id="access",
    )
    assert called.outcome["error_code"] == expected
    assert called.aws_error_code == code
    stranded = _live_failure_operation(
        operation_name="ConverseStream",
        effect="infer",
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        exc=failure,
        reserved_usd=0.01,
    )
    assert stranded["error_code"] == expected
    assert stranded["mode"] == "live_service"


@pytest.mark.parametrize(
    "prior", [None, "model_unavailable", "budget_exceeded"]
)
def test_access_next_step_is_actionable_and_idempotent(
    prior: str | None,
) -> None:
    failed = operation(
        service="bedrock-runtime",
        operation="Converse",
        mode="live_service",
        execution_target="aws",
        transport="aws",
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        status="blocked",
        error_code="model_access_required",
    )
    args: Any = {
        "demo": "test",
        "technology": "Bedrock",
        "lane": "models",
        "lifecycle_refs": ["bedrock"],
        "requested_execution": "live",
        "headline": "Access diagnostic",
        "operations": [failed],
        "error": None if prior is None else error(prior, "original"),
    }
    first = result(**args)
    assert first["error"]["code"] == (
        "budget_exceeded"
        if prior == "budget_exceeded"
        else "model_access_required"
    )
    assert "Anthropic" in first["next_steps"][0]
    assert "Bedrock console" in first["next_steps"][0]
    assert (
        result(**args, next_steps=first["next_steps"])["next_steps"]
        == first["next_steps"]
    )


def test_kendra_exposes_observed_aws_code() -> None:
    class Port:
        def call(self, *_args: Any) -> Any:
            raise ClientError(
                {
                    "Error": {"Code": "SubscriptionRequiredException"},
                    "ResponseMetadata": {"HTTPStatusCode": 400},
                },
                "ListIndices",
            )

    captured = run_kendra_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=Port(),
    )
    assert (
        captured["data"]["list_indices_code"]
        == "SubscriptionRequiredException"
    )
    assert captured["operations"][1]["error_code"] == "subscription_required"
