from __future__ import annotations

import pytest

from awsai_demo.contracts import (
    aggregate_mode,
    aggregate_status,
    error,
    evidence,
    missing_configuration,
    not_run,
    operation,
    result,
)


def _base_kwargs() -> dict[str, object]:
    return {
        "demo": "demo",
        "technology": "Tech",
        "lane": "models",
        "lifecycle_refs": ["ref"],
        "requested_execution": "offline",
        "headline": "ok",
    }


def _live_service_op() -> dict[str, object]:
    return operation(
        service="bedrock",
        operation="ListFoundationModels",
        execution_target="aws",
        mode="live_service",
        effect="read",
        transport="aws",
        endpoint_url="https://bedrock.us-east-1.amazonaws.com/models",
    )


def _skipped_op(
    *,
    status: str = "blocked",
    request_validated: bool = False,
) -> dict[str, object]:
    return operation(
        service="bedrock",
        operation="ListFoundationModels",
        execution_target="aws",
        mode="not_run",
        status=status,
        transport="none",
        response_received=False,
        request_validated=request_validated,
        error_code="budget_exceeded",
    )


def test_operation_builds_and_sanitizes_endpoint() -> None:
    outcome = operation(
        service="bedrock-runtime",
        operation="Converse",
        endpoint_url="https://bedrock.example/path?secret=hidden",
        fixture_id="fixture-1",
        usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    )
    live = operation(
        service="bedrock",
        operation="ListFoundationModels",
        execution_target="aws",
        mode="live_service",
        effect="read",
        transport="aws",
        endpoint_url="https://user:pass@bedrock.example:443/path?q=secret",
    )
    invalid_port = operation(
        service="svc",
        operation="op",
        endpoint_url="https://example.test:bad/path?secret=hidden",
    )
    invalid_ipv6_port = operation(
        service="svc",
        operation="op",
        endpoint_url="http://[::1]:bad/path?secret=hidden",
    )

    assert outcome["endpoint_url"] == "https://bedrock.example/path"
    assert outcome["fixture_id"] == "fixture-1"
    assert outcome["usage"] == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
    }
    assert live["endpoint_url"] == "https://bedrock.example:443/path"
    assert invalid_port["endpoint_url"] == "https://example.test/path"
    assert invalid_ipv6_port["endpoint_url"] == "http://[::1]/path"


def test_operation_rejects_invalid_literals_and_redacts_plain_endpoint() -> (
    None
):
    with pytest.raises(ValueError, match="phase must be one of"):
        operation(
            service="svc",
            operation="op",
            phase="bad",  # type: ignore[arg-type]
        )

    outcome = operation(
        service="svc",
        operation="op",
        endpoint_url="not-a-url AKIAABCDEFGHIJKLMNOP",
    )
    assert outcome["endpoint_url"] == "not-a-url <access-key-id>"


def test_operation_rejects_false_local_provenance() -> None:
    with pytest.raises(ValueError, match="fixture/none provenance"):
        operation(
            service="svc",
            operation="op",
            transport="loopback",
            endpoint_url="http://localhost:1",
        )
    with pytest.raises(ValueError, match="local execution_target"):
        operation(
            service="svc",
            operation="op",
            mode="local_execution",
            execution_target="fixture",
        )
    with pytest.raises(ValueError, match="none or loopback transport"):
        operation(
            service="svc",
            operation="op",
            mode="local_execution",
            execution_target="local",
            transport="aws",
        )
    with pytest.raises(ValueError, match="must not name fixtures"):
        operation(
            service="svc",
            operation="op",
            mode="local_execution",
            execution_target="local",
            fixture_id="fx",
        )
    with pytest.raises(ValueError, match="need loopback endpoint"):
        operation(
            service="svc",
            operation="op",
            mode="local_execution",
            execution_target="local",
            transport="loopback",
            endpoint_url="https://example.test",
        )


def test_operation_accepts_true_local_execution_variants() -> None:
    stdio = operation(
        service="mcp",
        operation="tools/call",
        mode="local_execution",
        execution_target="local",
        transport="none",
    )
    loopback = operation(
        service="a2a",
        operation="POST /tasks",
        mode="local_execution",
        execution_target="local",
        transport="loopback",
        endpoint_url="http://[::1]:5151/tasks",
    )

    assert stdio["transport"] == "none"
    assert loopback["endpoint_url"] == "http://[::1]:5151/tasks"


def test_operation_rejects_false_emulator_and_live_provenance() -> None:
    with pytest.raises(ValueError, match="emulator/loopback provenance"):
        operation(
            service="localstack",
            operation="GET /_localstack/health",
            mode="local_emulator",
            execution_target="local",
            transport="loopback",
            endpoint_url="http://localhost:4566",
        )
    with pytest.raises(ValueError, match="need loopback endpoint"):
        operation(
            service="localstack",
            operation="GET /_localstack/health",
            mode="local_emulator",
            execution_target="emulator",
            transport="loopback",
            endpoint_url="http://127.attacker:4566",
        )
    with pytest.raises(
        ValueError, match="live operation modes require a response"
    ):
        operation(
            service="bedrock",
            operation="ListFoundationModels",
            execution_target="aws",
            mode="live_service",
            transport="aws",
            endpoint_url="https://bedrock.us-east-1.amazonaws.com",
            response_received=False,
        )
    with pytest.raises(ValueError, match="non-loopback endpoint"):
        operation(
            service="bedrock",
            operation="ListFoundationModels",
            execution_target="aws",
            mode="live_service",
            transport="aws",
            endpoint_url="http://127.0.0.1:1",
        )
    with pytest.raises(ValueError, match="non-loopback endpoint"):
        operation(
            service="bedrock",
            operation="ListFoundationModels",
            execution_target="aws",
            mode="live_service",
            transport="aws",
            endpoint_url="not-a-url",
        )
    with pytest.raises(ValueError, match="live transport and target"):
        operation(
            service="bedrock",
            operation="ListFoundationModels",
            execution_target="local",
            mode="live_service",
            transport="loopback",
            endpoint_url="https://bedrock.us-east-1.amazonaws.com",
        )
    with pytest.raises(ValueError, match="live_identity requires sts"):
        operation(
            service="bedrock",
            operation="GetCallerIdentity",
            execution_target="aws",
            mode="live_identity",
            transport="aws",
            endpoint_url="https://sts.us-east-1.amazonaws.com",
        )


def test_live_identity_and_model_require_successful_operations() -> None:
    with pytest.raises(ValueError, match="GetCallerIdentity operation"):
        operation(
            service="sts",
            operation="AssumeRole",
            execution_target="aws",
            mode="live_identity",
            effect="read",
            transport="aws",
            endpoint_url="https://sts.us-east-1.amazonaws.com",
        )

    with pytest.raises(ValueError, match="must be ok"):
        operation(
            service="bedrock-runtime",
            operation="Converse",
            execution_target="aws",
            mode="live_model",
            status="blocked",
            effect="infer",
            transport="aws",
            endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        )

    with pytest.raises(ValueError, match="must not error"):
        operation(
            service="sts",
            operation="GetCallerIdentity",
            execution_target="aws",
            mode="live_identity",
            effect="read",
            transport="aws",
            endpoint_url="https://sts.us-east-1.amazonaws.com",
            error_code="authorization_denied",
        )

    with pytest.raises(ValueError, match="successful HTTP"):
        operation(
            service="bedrock-runtime",
            operation="Converse",
            execution_target="aws",
            mode="live_model",
            effect="infer",
            transport="aws",
            endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            http_status=403,
        )

    denied_service_probe = operation(
        service="sts",
        operation="GetCallerIdentity",
        execution_target="aws",
        mode="live_service",
        status="error",
        effect="read",
        transport="aws",
        endpoint_url="https://sts.us-east-1.amazonaws.com",
        http_status=403,
        error_code="authorization_denied",
    )
    assert denied_service_probe["mode"] == "live_service"


def test_operation_rejects_result_only_or_failed_call_mismatches() -> None:
    with pytest.raises(ValueError, match="batch is not an operation mode"):
        operation(service="svc", operation="op", mode="batch")

    skipped = _skipped_op()
    assert skipped["mode"] == "not_run"
    assert _skipped_op(status="paused")["status"] == "paused"
    assert _skipped_op(status="error")["status"] == "error"

    with pytest.raises(ValueError, match="must not have a response"):
        operation(
            service="svc",
            operation="op",
            mode="attempt_failed",
            execution_target="aws",
            transport="aws",
            endpoint_url="https://svc.amazonaws.com",
        )
    with pytest.raises(ValueError, match="dispatched real call"):
        operation(
            service="svc",
            operation="op",
            mode="attempt_failed",
            response_received=False,
        )


def test_not_run_operation_rejects_attempt_and_claims() -> None:
    with pytest.raises(ValueError, match="must not be ok"):
        operation(
            service="svc",
            operation="op",
            mode="not_run",
            response_received=False,
        )
    with pytest.raises(ValueError, match="must not claim an effect"):
        operation(
            service="svc",
            operation="op",
            mode="not_run",
            status="blocked",
            effect="read",
            response_received=False,
        )
    with pytest.raises(ValueError, match="must not attempt transport"):
        operation(
            service="svc",
            operation="op",
            mode="not_run",
            status="blocked",
            transport="loopback",
            response_received=False,
        )
    with pytest.raises(ValueError, match="must not name an endpoint"):
        operation(
            service="svc",
            operation="op",
            mode="not_run",
            status="blocked",
            endpoint_url="https://svc.amazonaws.com",
            response_received=False,
        )
    with pytest.raises(ValueError, match="must not receive a response"):
        operation(
            service="svc",
            operation="op",
            mode="not_run",
            status="blocked",
        )
    with pytest.raises(ValueError, match="must not claim a fixture"):
        operation(
            service="svc",
            operation="op",
            mode="not_run",
            status="blocked",
            response_received=False,
            fixture_id="fx",
        )
    with pytest.raises(ValueError, match="must not claim an HTTP status"):
        operation(
            service="svc",
            operation="op",
            mode="not_run",
            status="blocked",
            response_received=False,
            http_status=403,
        )
    with pytest.raises(ValueError, match="must not claim usage"):
        operation(
            service="svc",
            operation="op",
            mode="not_run",
            status="blocked",
            response_received=False,
            usage={"input_tokens": 1},
        )
    with pytest.raises(ValueError, match="must not claim a reservation"):
        operation(
            service="svc",
            operation="op",
            mode="not_run",
            status="blocked",
            response_received=False,
            reserved_usd=0.01,
        )


def test_evidence_defaults_validation_redaction_and_allowlist() -> None:
    item = evidence(
        provider="arn:aws:bedrock:us-east-1:123456789012:model/demo",
        credential_source="profile",
        packages={"botocore": "1"},
        cost_basis="estimated",
        estimated_cost_usd=0.01,
        emulator={"endpoint": "http://localhost:4566", "digest": None},
    )

    assert item["credential_source"] == "profile"
    assert item["provider"] == "arn:aws:bedrock:us-east-1:<account>:<redacted>"
    assert item["packages"] == {"botocore": "1"}
    assert item["emulator"] == {
        "endpoint": "http://localhost:4566",
        "digest": None,
    }

    with pytest.raises(ValueError, match="credential_source must be one of"):
        evidence(credential_source="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cost_basis must be one of"):
        evidence(cost_basis="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="emulator evidence contains unknown"):
        evidence(emulator={"endpoint": "http://localhost:4566", "raw": "x"})


def test_error_builder_validates_and_redacts_message() -> None:
    built = error(
        "authorization_denied",
        "account 123456789012 denied Authorization: Bearer secret-token",
    )

    assert built == {
        "code": "authorization_denied",
        "message": "account <account> denied Authorization: <redacted>",
    }
    with pytest.raises(ValueError, match="code must be one of"):
        error("bad", "message")  # type: ignore[arg-type]


def test_result_builds_local_contract_and_recursively_redacts() -> None:
    op = operation(
        service="bedrock-runtime",
        operation="Converse",
        fixture_id="fixture-1",
    )
    built = result(
        **_base_kwargs(),
        mode="local_contract",
        operations=[op],
        evidence=evidence(fixture_id="fixture-1"),
        data={"arn": "arn:aws:iam::123456789012:role/demo"},
        error=error("validation_failed", "bad ASIAABCDEFGHIJKLMNOP"),
        next_steps=["Use 123456789012"],
    )

    assert built["mode"] == "local_contract"
    assert built["status"] == "blocked"
    assert built["data"] == {"arn": "arn:aws:iam::<account>:<redacted>"}
    assert built["error"] == {
        "code": "validation_failed",
        "message": "bad <access-key-id>",
    }
    assert built["next_steps"] == ["Use <account>"]


def test_result_validates_required_shape_and_runtime_keys() -> None:
    with pytest.raises(ValueError, match="lifecycle_refs"):
        result(**(_base_kwargs() | {"lifecycle_refs": []}))

    with pytest.raises(ValueError, match="children are only allowed"):
        result(
            **_base_kwargs(),
            mode="local_execution",
            children=[not_run(**_base_kwargs())],
        )

    with pytest.raises(ValueError, match="batch results must contain"):
        result(**_base_kwargs(), mode="batch")

    raw_evidence = dict(evidence())
    raw_evidence["raw"] = "sdk response"
    with pytest.raises(ValueError, match="evidence contains unknown keys"):
        result(**_base_kwargs(), evidence=raw_evidence)

    missing_key_evidence = dict(evidence())
    del missing_key_evidence["packages"]
    with pytest.raises(ValueError, match="evidence is missing keys"):
        result(**_base_kwargs(), evidence=missing_key_evidence)

    raw_operation = dict(operation(service="svc", operation="op"))
    raw_operation["raw"] = "sdk response"
    with pytest.raises(ValueError, match="operation contains unknown keys"):
        result(**_base_kwargs(), operations=[raw_operation])

    errored = result(
        **_base_kwargs(),
        error=error("timeout", "request timed out"),
    )
    assert errored["status"] == "error"


def test_result_rejects_false_provenance() -> None:
    fixtureless = operation(service="svc", operation="op")
    with pytest.raises(ValueError, match="fixture_id"):
        result(
            **_base_kwargs(),
            mode="local_contract",
            operations=[fixtureless],
        )

    fixture_op = operation(
        service="svc",
        operation="op",
        fixture_id="fixture-1",
    )
    with pytest.raises(ValueError, match="contradicts operation provenance"):
        result(
            **_base_kwargs(),
            mode="local_execution",
            operations=[fixture_op],
            evidence=evidence(fixture_id="fixture-1"),
        )

    with pytest.raises(ValueError, match="not_run results"):
        result(
            **_base_kwargs(),
            mode="not_run",
            operations=[fixture_op],
            evidence=evidence(fixture_id="fixture-1"),
        )

    with pytest.raises(ValueError, match="aws_executed"):
        result(
            **_base_kwargs(),
            mode="local_execution",
            evidence=evidence(aws_executed=True),
        )

    with pytest.raises(ValueError, match="loopback emulator"):
        result(
            **_base_kwargs(),
            mode="local_emulator",
            evidence=evidence(emulator={"endpoint": "https://aws.example"}),
        )

    with pytest.raises(ValueError, match="loopback emulator"):
        result(
            **_base_kwargs(),
            mode="local_emulator",
            evidence=evidence(),
        )

    with pytest.raises(ValueError, match="loopback emulator"):
        result(
            **_base_kwargs(),
            mode="local_emulator",
            evidence=evidence(emulator={"endpoint": "localhost:4566"}),
        )

    with pytest.raises(ValueError, match="operation without a response"):
        result(**_base_kwargs(), mode="attempt_failed")


def test_result_accepts_skipped_not_run_operations() -> None:
    skipped = _skipped_op()
    built = result(
        **_base_kwargs(),
        mode="not_run",
        operations=[skipped],
    )

    assert built["mode"] == "not_run"
    assert built["status"] == "blocked"
    assert built["evidence"]["sdk_invoked"] is False
    assert built["evidence"]["network_attempted"] is False
    assert built["evidence"]["aws_executed"] is False

    validated = result(
        **_base_kwargs(),
        mode="not_run",
        operations=[_skipped_op(request_validated=True)],
    )
    assert validated["evidence"]["sdk_invoked"] is True


def test_requested_live_cannot_replay_fixtures() -> None:
    fixture_op = operation(
        service="bedrock-runtime",
        operation="Converse",
        fixture_id="fx",
    )
    with pytest.raises(ValueError, match="live requests must not replay"):
        result(
            **(_base_kwargs() | {"requested_execution": "live"}),
            mode="local_contract",
            operations=[fixture_op],
            evidence=evidence(fixture_id="fx"),
        )
    with pytest.raises(ValueError, match="live requests must not replay"):
        result(
            **(_base_kwargs() | {"requested_execution": "live"}),
            evidence=evidence(fixture_id="fx"),
        )

    skipped = not_run(
        **(_base_kwargs() | {"requested_execution": "live"}),
        message="missing live prerequisites",
    )
    assert skipped["mode"] == "not_run"


def test_live_result_validation_and_attempt_failed() -> None:
    live_op = operation(
        service="bedrock-runtime",
        operation="Converse",
        execution_target="aws",
        mode="live_model",
        effect="infer",
        transport="aws",
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com/model",
        reserved_usd=0.01,
    )
    openai_op = operation(
        service="openai",
        operation="responses.create",
        execution_target="openai",
        mode="live_model",
        effect="infer",
        transport="external",
        endpoint_url="https://api.openai.com/v1/responses",
    )
    built = result(
        **(_base_kwargs() | {"requested_execution": "live"}),
        operations=[live_op],
    )
    openai_built = result(
        **(_base_kwargs() | {"requested_execution": "live"}),
        operations=[openai_op],
    )

    assert built["mode"] == "live_model"
    assert built["evidence"]["network_attempted"] is True
    assert built["evidence"]["aws_executed"] is True
    assert openai_built["mode"] == "live_model"
    assert openai_built["evidence"]["aws_executed"] is False

    with pytest.raises(ValueError, match="network_attempted=True"):
        result(
            **_base_kwargs(),
            mode="live_service",
            operations=[_live_service_op()],
            evidence=evidence(network_attempted=False),
        )

    with pytest.raises(ValueError, match="matching answered live operation"):
        result(
            **_base_kwargs(),
            mode="live_model",
            evidence=evidence(network_attempted=True),
        )

    failed = operation(
        service="svc",
        operation="op",
        mode="attempt_failed",
        execution_target="aws",
        transport="aws",
        endpoint_url="https://svc.amazonaws.com",
        response_received=False,
        status="error",
    )
    assert (
        result(**_base_kwargs(), operations=[failed])["mode"]
        == "attempt_failed"
    )


def test_live_identity_and_service_successes() -> None:
    identity = operation(
        service="sts",
        operation="GetCallerIdentity",
        execution_target="aws",
        mode="live_identity",
        effect="read",
        transport="aws",
        endpoint_url="https://sts.us-east-1.amazonaws.com",
    )
    identity_result = result(
        **(_base_kwargs() | {"requested_execution": "live"}),
        operations=[identity],
    )
    service_result = result(
        **(_base_kwargs() | {"requested_execution": "live"}),
        operations=[_live_service_op()],
    )

    assert identity_result["mode"] == "live_identity"
    assert service_result["mode"] == "live_service"


def test_local_emulator_success_accepts_loopback_endpoint() -> None:
    built = result(
        **_base_kwargs(),
        mode="local_emulator",
        operations=[
            operation(
                service="sts",
                operation="GetCallerIdentity",
                execution_target="emulator",
                mode="local_emulator",
                transport="loopback",
                endpoint_url="http://localhost:4566",
                effect="read",
            ),
        ],
        evidence=evidence(emulator={"endpoint": "http://[::1]:4566"}),
    )

    assert built["mode"] == "local_emulator"


def test_batch_and_aggregate_helpers() -> None:
    ok_child = result(**_base_kwargs())
    blocked_child = not_run(
        **(_base_kwargs() | {"headline": "blocked"}),
        code="missing_configuration",
    )
    batch = result(
        **(_base_kwargs() | {"demo": "all"}),
        children=[ok_child, blocked_child],
    )

    assert batch["mode"] == "batch"
    assert batch["status"] == "blocked"
    assert aggregate_status([]) == "ok"
    assert aggregate_status([ok_child, {"status": "paused"}]) == "paused"
    assert aggregate_mode([], [ok_child]) == "batch"
    assert aggregate_mode([]) == "local_execution"
    assert aggregate_mode([_skipped_op()]) == "not_run"
    assert (
        aggregate_mode([_skipped_op(), _live_service_op()]) == "live_service"
    )
    assert (
        aggregate_mode(
            [
                operation(
                    service="sts",
                    operation="GetCallerIdentity",
                    execution_target="aws",
                    mode="live_identity",
                    transport="aws",
                    endpoint_url="https://sts.amazonaws.com",
                ),
            ],
        )
        == "live_identity"
    )
    assert aggregate_mode([_live_service_op()]) == "live_service"
    assert (
        aggregate_mode(
            [
                operation(
                    service="svc",
                    operation="op",
                    mode="local_execution",
                    execution_target="local",
                ),
                operation(
                    service="svc",
                    operation="op",
                    mode="local_emulator",
                    execution_target="emulator",
                    transport="loopback",
                    endpoint_url="http://localhost:4566",
                ),
                operation(service="svc", operation="op", fixture_id="fx"),
            ],
        )
        == "local_contract"
    )

    with pytest.raises(ValueError, match="status must be one of"):
        aggregate_status([{"status": "bad"}])


def test_not_run_and_missing_configuration_helpers() -> None:
    skipped = not_run(
        **_base_kwargs(),
        code="not_supported_by_emulator",
        message="Needs live",
        next_steps=["Use --execution live"],
        data={"reason": "contract"},
    )
    missing = missing_configuration(
        **_base_kwargs(),
        message="Set AWSAI_HARNESS_ROLE_ARN",
    )

    assert skipped["mode"] == "not_run"
    assert skipped["status"] == "blocked"
    assert skipped["error"] == {
        "code": "not_supported_by_emulator",
        "message": "Needs live",
    }
    assert skipped["next_steps"] == ["Use --execution live"]
    assert skipped["data"] == {"reason": "contract"}
    assert missing["error"] == {
        "code": "missing_configuration",
        "message": "Set AWSAI_HARNESS_ROLE_ARN",
    }


def test_invalid_result_literals_are_rejected() -> None:
    with pytest.raises(ValueError, match="lane must be one of"):
        result(**(_base_kwargs() | {"lane": "bad"}))
    with pytest.raises(ValueError, match="requested_execution must be one of"):
        result(**(_base_kwargs() | {"requested_execution": "bad"}))
    with pytest.raises(ValueError, match="mode must be one of"):
        result(**_base_kwargs(), mode="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="status must be one of"):
        result(**_base_kwargs(), status="bad")  # type: ignore[arg-type]


def test_max_tokens_reached_stays_a_blocked_live_answer() -> None:
    usage = {"input_tokens": 12, "output_tokens": 512, "total_tokens": 524}
    capped = operation(
        service="bedrock-runtime",
        operation="Converse",
        execution_target="aws",
        mode="live_model",
        status="blocked",
        effect="infer",
        transport="aws",
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        error_code="max_tokens_reached",
        usage=usage,
    )
    built = result(
        **(_base_kwargs() | {"requested_execution": "live"}),
        operations=[capped],
        error=error("max_tokens_reached", "The model hit max_output_tokens"),
    )
    assert built["status"] == "blocked"
    assert built["mode"] == "live_model"
    assert built["operations"][0]["usage"] == usage
    assert built["error"]["code"] == "max_tokens_reached"
    with pytest.raises(ValueError, match="requires status blocked"):
        operation(
            service="bedrock-runtime",
            operation="Converse",
            execution_target="aws",
            mode="live_model",
            status="ok",
            effect="infer",
            transport="aws",
            endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            error_code="max_tokens_reached",
        )
