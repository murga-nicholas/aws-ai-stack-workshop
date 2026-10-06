from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from botocore.exceptions import ClientError

from awsai_demo.billing import open_budget_run, use_budget_run
from awsai_demo.contracts import operation
from awsai_demo.demo_support import AwsResponse, BotoAwsPort
from awsai_demo.guardrails_demo import (
    _GUARDRAIL_POLL_SECONDS,
    _INPUT_TEXT,
    _OUTPUT_TEXT,
    _apply_response,
    _block_guardrail_readiness,
    _missing_live_port,
    _missing_named_port,
    _policy_features_from_create,
    _policy_features_from_get,
    _run_tags,
    _text_units,
    _token,
    _verify_for_child,
    apply_guardrail_params,
    create_guardrail_params,
    create_guardrail_version_params,
    guardrail_apply_charges,
    run_guardrails_demo,
)
from awsai_demo.manifest import ManifestStore
from awsai_demo.policy import ExecutionPolicy, ReservationUnavailable
from awsai_demo.runtime import Settings
from awsai_demo.stubs import stubbed_client, validate_operation_request

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class Prices:
    def __init__(
        self,
        *,
        fail: bool = False,
        fail_after: int | None = None,
    ) -> None:
        self.fail = fail
        self.fail_after = fail_after
        self.calls: list[tuple[str, str]] = []

    def rate(
        self,
        feature: str,
        *,
        region: str,
        unit: str,
        tier: str,
        routing: str,
    ) -> Decimal:
        del tier, routing
        self.calls.append((feature, unit))
        if self.fail or (
            self.fail_after is not None and len(self.calls) > self.fail_after
        ):
            message = "missing price"
            raise ReservationUnavailable(message)
        assert region == "us-east-1"
        return Decimal("0.000001")


class GuardrailPort:
    def __init__(
        self,
        *,
        auth_ok: bool = True,
        create_ok: bool = True,
        verify_ok: bool = True,
        version_ok: bool = True,
        unknown_policy: bool = False,
        get_ok: bool = True,
        delete_ok: bool = True,
    ) -> None:
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []
        self.endpoint_url = "https://bedrock.us-east-1.amazonaws.com"
        self.name = "existing-guard"
        self.guardrail_id = "gr1234567890"
        self.run_id = "abcdef123456"
        self.auth_ok = auth_ok
        self.create_ok = create_ok
        self.verify_ok = verify_ok
        self.version_ok = version_ok
        self.unknown_policy = unknown_policy
        self.get_ok = get_ok
        self.delete_ok = delete_ok

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse:
        self.calls.append((service, operation_name, params))
        endpoint = f"https://{service}.us-east-1.amazonaws.com"
        if operation_name == "GetCallerIdentity":
            payload = {"Account": "123456789012"} if self.auth_ok else {}
            return AwsResponse(payload, endpoint)
        if operation_name == "CreateGuardrail":
            self.name = str(params["name"])
            self.run_id = str(params["tags"][0]["value"])
            if not self.create_ok:
                return AwsResponse({}, endpoint)
            return AwsResponse(
                {
                    "guardrailId": self.guardrail_id,
                    "guardrailArn": self._arn(),
                    "version": "DRAFT",
                },
                endpoint,
            )
        if operation_name == "CreateGuardrailVersion":
            if not self.version_ok:
                return AwsResponse(
                    {"guardrailId": self.guardrail_id}, endpoint
                )
            return AwsResponse(
                {"guardrailId": self.guardrail_id, "version": "1"},
                endpoint,
            )
        if operation_name == "GetGuardrail":
            if not self.get_ok:
                raise ClientError(
                    {
                        "Error": {"Code": "AccessDeniedException"},
                        "ResponseMetadata": {"HTTPStatusCode": 403},
                    },
                    "GetGuardrail",
                )
            return AwsResponse(self._guardrail_payload(), endpoint)
        if operation_name == "ListGuardrails":
            return AwsResponse({"guardrails": []}, endpoint)
        if operation_name == "ListTagsForResource":
            run_id = self.run_id if self.verify_ok else "other"
            return AwsResponse(
                {"tags": [{"key": "run-id", "value": run_id}]},
                endpoint,
            )
        if operation_name == "ApplyGuardrail":
            action = (
                "GUARDRAIL_INTERVENED"
                if params["source"] == "INPUT"
                else "NONE"
            )
            return AwsResponse({"action": action}, endpoint)
        if operation_name == "DeleteGuardrail":
            if not self.delete_ok:
                raise ClientError(
                    {
                        "Error": {"Code": "AccessDeniedException"},
                        "ResponseMetadata": {"HTTPStatusCode": 403},
                    },
                    "DeleteGuardrail",
                )
            return AwsResponse({}, endpoint)
        raise AssertionError(operation_name)

    def _arn(self) -> str:
        return (
            "arn:aws:bedrock:us-east-1:123456789012:guardrail/"
            + self.guardrail_id
        )

    def _guardrail_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "guardrailId": self.guardrail_id,
            "guardrailArn": self._arn(),
            "version": "1",
            "status": "READY",
            "blockedInputMessaging": "Blocked",
            "blockedOutputsMessaging": "Blocked",
            "contentPolicy": {"filters": []},
            "topicPolicy": {"topics": []},
            "sensitiveInformationPolicy": {"piiEntities": []},
        }
        if self.unknown_policy:
            payload["wordPolicy"] = {"words": [{"text": "secret"}]}
        return payload


class EmptySession:
    def client(self, service_name: str, **kwargs: Any) -> object:
        raise AssertionError((service_name, kwargs))


def test_guardrails_offline_fixture_and_marker() -> None:
    result = run_guardrails_demo(
        execution="offline",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )

    assert result["status"] == "ok"
    assert [item["operation"] for item in result["operations"]] == [
        "CreateGuardrail",
        "CreateGuardrailVersion",
        "ApplyGuardrail",
        "ApplyGuardrail",
        "DeleteGuardrail",
    ]
    data = _result_data(result)
    assert data["input_action"] == "GUARDRAIL_INTERVENED"
    assert _mapping(data["guardrail"])["output_action"] == "NONE"
    marker = _marker_lines(
        "src/awsai_demo/guardrails_demo.py", "apply-guardrail"
    )
    assert len(marker) <= 14
    assert all(len(line) <= 72 for line in marker)
    assert "client.apply_guardrail(" in "\n".join(marker)
    assert 'action = response["action"]' in "\n".join(marker)


def test_guardrails_request_helpers_validate_and_refuse_unknown() -> None:
    validate_operation_request(
        service_name="bedrock",
        operation_name="CreateGuardrail",
        params=create_guardrail_params(),
        fixture_id="guardrails-create-v1",
    )
    validate_operation_request(
        service_name="bedrock-runtime",
        operation_name="ApplyGuardrail",
        params=apply_guardrail_params(source="INPUT", text="a@b.example"),
        fixture_id="guardrails-apply-input-email-v1",
    )
    assert (
        guardrail_apply_charges(
            apply_guardrail_params(source="OUTPUT", text="safe")
        )[0].feature
        == "guardrails.content_filter"
    )
    with pytest.raises(ReservationUnavailable):
        guardrail_apply_charges(
            apply_guardrail_params(source="INPUT", text="safe"),
            features=(),
        )
    with pytest.raises(ReservationUnavailable):
        _policy_features_from_create(
            {**create_guardrail_params(), "wordPolicyConfig": {"words": []}}
        )
    assert _policy_features_from_create({}) == ()
    with pytest.raises(ReservationUnavailable):
        _policy_features_from_get({})
    assert _text_units(
        {"content": [{"text": {"text": "x" * 1001}}, {"text": {}}]}
    ) == Decimal(2)
    assert _missing_live_port()[0]["error_code"] == "missing_configuration"
    assert _missing_named_port()[0]["operation"] == "GetGuardrail"


def test_guardrails_emulator_missing_and_configured() -> None:
    missing = run_guardrails_demo(
        execution="emulator",
        settings=Settings(),
        policy=ExecutionPolicy(),
    )
    calls: list[dict[str, str]] = []

    def factory(**kwargs: str) -> EmptySession:
        calls.append(kwargs)
        return EmptySession()

    configured = run_guardrails_demo(
        execution="emulator",
        settings=Settings(localstack_auth_token="token"),  # noqa: S106
        policy=ExecutionPolicy(),
        session_factory=factory,
    )

    assert {item["error_code"] for item in missing["operations"]} == {
        "missing_configuration"
    }
    assert {item["error_code"] for item in configured["operations"]} == {
        "not_supported_by_emulator"
    }
    assert calls and calls[0]["region_name"] == "us-east-1"


def test_guardrails_live_pricing_refuses_before_dispatch(
    tmp_path: Path,
) -> None:
    port = GuardrailPort()
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(fail=True),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_guardrails_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=port,
        )

    assert result["status"] == "blocked"
    assert result["operations"][0]["error_code"] == "budget_exceeded"
    assert port.calls == []


def test_guardrails_live_create_versions_applies_and_cleans(
    tmp_path: Path,
) -> None:
    port = GuardrailPort()
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_guardrails_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=port,
        )

    operations = [item["operation"] for item in result["operations"]]
    data = _result_data(result)
    assert data["input_action"] == "GUARDRAIL_INTERVENED"
    assert _mapping(data["guardrail"])["output_action"] == "NONE"
    assert operations[:2] == ["GetCallerIdentity", "CreateGuardrail"]
    assert "CreateGuardrailVersion" in operations
    assert operations.count("ApplyGuardrail") == 2
    assert operations[-1] == "DeleteGuardrail"
    assert any(item["reserved_usd"] for item in result["operations"])
    assert data["cleanup_incomplete"] is False


@pytest.mark.parametrize(
    ("port", "key"),
    [
        (GuardrailPort(auth_ok=False), "identity"),
        (GuardrailPort(create_ok=False), "created"),
        (GuardrailPort(verify_ok=False), "ownership_verified"),
        (GuardrailPort(version_ok=False), "versioned"),
    ],
)
def test_guardrails_live_create_stops_on_setup_failures(
    tmp_path: Path,
    port: GuardrailPort,
    key: str,
) -> None:
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_guardrails_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=port,
        )

    assert key in _result_data(result)
    if key in {"created", "ownership_verified"}:
        assert result["status"] == "error"
        assert _result_data(result)["cleanup_incomplete"] is True


def test_guardrails_live_create_catches_policy_error_after_create(
    tmp_path: Path,
) -> None:
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(fail_after=6),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_guardrails_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=GuardrailPort(),
        )

    data = _result_data(result)
    assert data["policy_error"] == "ReservationUnavailableError"
    assert any(
        item["operation"] == "ApplyGuardrail"
        and item["error_code"] == "budget_exceeded"
        for item in result["operations"]
    )


def test_guardrails_live_delete_denied_reports_cleanup_error(
    tmp_path: Path,
) -> None:
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_guardrails_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            port=GuardrailPort(delete_ok=False),
        )

    assert result["status"] == "error"
    error = result["error"]
    assert error is not None
    assert error["code"] == "cleanup_incomplete"
    assert _result_data(result)["cleanup_incomplete"] is True


def test_guardrails_live_named_refuses_unpriced_policy(
    tmp_path: Path,
) -> None:
    port = GuardrailPort(unknown_policy=True)
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_guardrails_demo(
            execution="live",
            settings=Settings(
                guardrail_id="gr1234567890",
                guardrail_version="1",
            ),
            policy=run.policy,
            port=port,
        )

    assert [call[1] for call in port.calls] == ["GetGuardrail"]
    assert result["operations"][1]["error_code"] == "budget_exceeded"
    assert _result_data(result)["pricing"] == "missing_or_unsupported"


def test_guardrails_live_named_success_and_missing_config(
    tmp_path: Path,
) -> None:
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        success = run_guardrails_demo(
            execution="live",
            settings=Settings(
                guardrail_id="gr1234567890",
                guardrail_version="1",
            ),
            policy=run.policy,
            port=GuardrailPort(),
        )
    missing = run_guardrails_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=GuardrailPort(),
    )

    assert _result_data(success)["input_action"] == "GUARDRAIL_INTERVENED"
    guardrail = _mapping(_result_data(success)["guardrail"])
    assert guardrail["output_action"] == "NONE"
    assert _result_data(missing)["guardrail_configured"] is False


def test_guardrails_live_named_missing_default_port(tmp_path: Path) -> None:
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )
    with use_budget_run(run):
        result = run_guardrails_demo(
            execution="live",
            settings=Settings(
                guardrail_id="gr1234567890",
                guardrail_version="1",
            ),
            policy=run.policy,
            boto_port_factory=lambda **_: None,
        )

    assert result["operations"][0]["error_code"] == "missing_configuration"


def test_guardrails_live_named_stops_when_get_fails(
    tmp_path: Path,
) -> None:
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )

    with use_budget_run(run):
        result = run_guardrails_demo(
            execution="live",
            settings=Settings(
                guardrail_id="gr1234567890",
                guardrail_version="1",
            ),
            policy=run.policy,
            port=GuardrailPort(get_ok=False),
        )

    assert result["operations"][0]["operation"] == "GetGuardrail"
    assert result["operations"][0]["error_code"] == "authorization_denied"
    assert result["data"] == {"mode": "named"}


def test_guardrails_live_create_missing_default_port(tmp_path: Path) -> None:
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        root=tmp_path,
        prices=Prices(),
        run_id="abcdef123456",
    )
    with use_budget_run(run):
        result = run_guardrails_demo(
            execution="live",
            settings=Settings(allow_create=True),
            policy=run.policy,
            boto_port_factory=lambda **_: None,
        )

    assert result["operations"][0]["error_code"] == "missing_configuration"


_STUB_RUN = "abcdef123456"
_STUB_ACCOUNT = "123456789012"
_STUB_GUARD_ID = "gr1234567890"
_STUB_ARN = (
    "arn:aws:bedrock:us-east-1:"
    + _STUB_ACCOUNT
    + ":guardrail/"
    + _STUB_GUARD_ID
)
_STUB_NAME = f"awsai-{_STUB_RUN}-guard"
_STUB_WHEN = datetime(2026, 1, 1, tzinfo=UTC)


class _PollClock:
    def __init__(self, *, jump: float | None = None) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self.jump = jump

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds if self.jump is None else self.jump


class _ExpireClock:
    def __init__(self) -> None:
        self.reads = 0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        self.reads += 1
        if self.reads == 1:
            return 0.0
        return 50.0

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class _StubSession:
    def __init__(self, clients: Mapping[str, Any]) -> None:
        self._clients = clients

    def client(self, service_name: str, **kwargs: Any) -> Any:
        assert kwargs["region_name"] == "us-east-1"
        return self._clients[service_name]


def _guard_body(
    status: str,
    version: str,
    *,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": _STUB_NAME,
        "guardrailId": _STUB_GUARD_ID,
        "guardrailArn": _STUB_ARN,
        "version": version,
        "status": status,
        "createdAt": _STUB_WHEN,
        "updatedAt": _STUB_WHEN,
        "blockedInputMessaging": "Blocked by the workshop guardrail.",
        "blockedOutputsMessaging": (
            "Output blocked by the workshop guardrail."
        ),
    }
    if reasons is not None:
        body["statusReasons"] = reasons
    return body


def _add_get(
    bedrock: Any,
    status: str,
    version: str,
    *,
    numbered: bool = False,
    reasons: list[str] | None = None,
) -> None:
    params = {"guardrailIdentifier": _STUB_GUARD_ID}
    if numbered:
        params["guardrailVersion"] = version
    bedrock.add_response(
        "get_guardrail",
        _guard_body(status, version, reasons=reasons),
        expected_params=params,
    )


def _add_tags(bedrock: Any) -> None:
    bedrock.add_response(
        "list_tags_for_resource",
        {"tags": [{"key": "run-id", "value": _STUB_RUN}]},
        expected_params={"resourceARN": _STUB_ARN},
    )


def _add_delete(bedrock: Any) -> None:
    bedrock.add_response(
        "delete_guardrail",
        {},
        expected_params={"guardrailIdentifier": _STUB_GUARD_ID},
    )


def _add_owned_start(sts: Any, bedrock: Any) -> None:
    sts.add_response(
        "get_caller_identity",
        {
            "UserId": "AIDACKCEVSQ6C2EXAMPLE",
            "Account": _STUB_ACCOUNT,
            "Arn": f"arn:aws:iam::{_STUB_ACCOUNT}:user/demo",
        },
        expected_params={},
    )
    bedrock.add_response(
        "create_guardrail",
        {
            "guardrailId": _STUB_GUARD_ID,
            "guardrailArn": _STUB_ARN,
            "version": "DRAFT",
            "createdAt": _STUB_WHEN,
        },
        expected_params=create_guardrail_params(
            name=_STUB_NAME,
            client_token=_token(_STUB_RUN, "guard"),
            tags=_run_tags(_STUB_RUN),
        ),
    )
    _add_get(bedrock, "READY", "DRAFT")
    _add_tags(bedrock)


def _add_teardown(bedrock: Any) -> None:
    _add_get(bedrock, "READY", "DRAFT")
    _add_tags(bedrock)
    _add_delete(bedrock)


def _add_version(bedrock: Any) -> None:
    bedrock.add_response(
        "create_guardrail_version",
        {"guardrailId": _STUB_GUARD_ID, "version": "1"},
        expected_params=create_guardrail_version_params(
            _STUB_GUARD_ID,
            client_token=_token(_STUB_RUN, "guard-v1"),
        ),
    )


def _run_stubbed(
    tmp_path: Path,
    prepare: Callable[[Any, Any, Any], None],
    clock: _PollClock | _ExpireClock,
    *,
    wall: int = 120,
) -> Mapping[str, Any]:
    with (
        stubbed_client("sts") as sts,
        stubbed_client("bedrock") as bedrock,
        stubbed_client("bedrock-runtime") as runtime,
    ):
        prepare(sts, bedrock, runtime)
        run = open_budget_run(
            region="us-east-1",
            policy=ExecutionPolicy(max_wall_seconds=wall),
            root=tmp_path,
            prices=Prices(),
            run_id=_STUB_RUN,
        )
        with use_budget_run(run):
            return run_guardrails_demo(
                execution="live",
                settings=Settings(allow_create=True),
                policy=run.policy,
                port=BotoAwsPort(
                    session=_StubSession(
                        {
                            "sts": sts.client,
                            "bedrock": bedrock.client,
                            "bedrock-runtime": runtime.client,
                        }
                    ),
                    region_name="us-east-1",
                ),
                clock=clock,
                sleeper=clock.sleep,
            )


def test_stubber_waits_for_ready_before_version_and_apply(
    tmp_path: Path,
) -> None:
    def prepare(sts: Any, bedrock: Any, runtime: Any) -> None:
        _add_owned_start(sts, bedrock)
        _add_get(bedrock, "CREATING", "DRAFT")
        _add_get(bedrock, "READY", "DRAFT")
        _add_version(bedrock)
        _add_get(bedrock, "VERSIONING", "1", numbered=True)
        _add_get(bedrock, "READY", "1", numbered=True)
        runtime.add_response(
            "apply_guardrail",
            _apply_response("GUARDRAIL_INTERVENED"),
            expected_params=apply_guardrail_params(
                guardrail_id=_STUB_GUARD_ID,
                guardrail_version="1",
                source="INPUT",
                text=_INPUT_TEXT,
            ),
        )
        runtime.add_response(
            "apply_guardrail",
            _apply_response("NONE"),
            expected_params=apply_guardrail_params(
                guardrail_id=_STUB_GUARD_ID,
                guardrail_version="1",
                source="OUTPUT",
                text=_OUTPUT_TEXT,
            ),
        )
        _add_teardown(bedrock)

    clock = _PollClock()
    result = _run_stubbed(tmp_path, prepare, clock)
    operations = [item["operation"] for item in result["operations"]]

    assert operations == [
        "GetCallerIdentity",
        "CreateGuardrail",
        "GetGuardrail",
        "ListTagsForResource",
        "GetGuardrail",
        "GetGuardrail",
        "CreateGuardrailVersion",
        "GetGuardrail",
        "GetGuardrail",
        "ApplyGuardrail",
        "ApplyGuardrail",
        "GetGuardrail",
        "ListTagsForResource",
        "DeleteGuardrail",
    ]
    assert clock.sleeps == [
        _GUARDRAIL_POLL_SECONDS,
        _GUARDRAIL_POLL_SECONDS,
    ]
    assert result["status"] == "ok"
    assert _result_data(result)["input_action"] == "GUARDRAIL_INTERVENED"
    assert _result_data(result)["cleanup_incomplete"] is False
    create_token = _token(_STUB_RUN, "guard")
    version_token = _token(_STUB_RUN, "guard-v1")
    assert create_token != version_token
    recorded = {
        entry.operation: entry.client_token
        for entry in ManifestStore(
            tmp_path.resolve() / _STUB_RUN / "manifest.json"
        ).list_entries()
    }
    assert recorded["CreateGuardrail"] == create_token
    assert recorded["CreateGuardrailVersion"] == version_token


def test_stubber_failed_draft_is_blocked_then_deleted(
    tmp_path: Path,
) -> None:
    def prepare(sts: Any, bedrock: Any, _runtime: Any) -> None:
        _add_owned_start(sts, bedrock)
        _add_get(
            bedrock,
            "FAILED",
            "DRAFT",
            reasons=["topic policy rejected"],
        )
        _add_teardown(bedrock)

    clock = _PollClock()
    result = _run_stubbed(tmp_path, prepare, clock)
    error = result["error"]

    assert clock.sleeps == []
    assert "CreateGuardrailVersion" not in [
        item["operation"] for item in result["operations"]
    ]
    assert result["status"] == "blocked"
    assert error is not None
    assert error["code"] == "validation_failed"
    assert (
        error["message"]
        == "Guardrail draft status is FAILED: topic policy rejected."
    )
    poll = result["operations"][4]
    assert poll["operation"] == "GetGuardrail"
    assert poll["status"] == "blocked"
    assert poll["error_code"] == "validation_failed"
    assert result["operations"][-1]["operation"] == "DeleteGuardrail"
    assert _result_data(result)["cleanup_incomplete"] is False
    assert _result_data(result)["guardrail_status"] == "FAILED"


def test_stubber_readiness_timeout_is_blocked_then_deleted(
    tmp_path: Path,
) -> None:
    def prepare(sts: Any, bedrock: Any, _runtime: Any) -> None:
        _add_owned_start(sts, bedrock)
        _add_get(bedrock, "CREATING", "DRAFT")
        _add_get(bedrock, "CREATING", "DRAFT")
        _add_teardown(bedrock)

    clock = _PollClock()
    result = _run_stubbed(tmp_path, prepare, clock, wall=1)
    error = result["error"]

    assert clock.sleeps == [
        _GUARDRAIL_POLL_SECONDS,
        _GUARDRAIL_POLL_SECONDS,
    ]
    assert result["status"] == "blocked"
    assert error is not None
    assert error["code"] == "timeout"
    assert error["message"] == (
        "Guardrail draft was not READY before the wall-clock limit."
    )
    assert result["operations"][5]["status"] == "blocked"
    assert result["operations"][5]["error_code"] == "timeout"
    assert result["operations"][-1]["operation"] == "DeleteGuardrail"
    assert "CreateGuardrailVersion" not in [
        item["operation"] for item in result["operations"]
    ]


def test_stubber_failed_version_blocks_before_apply(
    tmp_path: Path,
) -> None:
    def prepare(sts: Any, bedrock: Any, _runtime: Any) -> None:
        _add_owned_start(sts, bedrock)
        _add_get(bedrock, "READY", "DRAFT")
        _add_version(bedrock)
        _add_get(bedrock, "FAILED", "1", numbered=True)
        _add_teardown(bedrock)

    result = _run_stubbed(tmp_path, prepare, _PollClock())
    error = result["error"]

    assert result["status"] == "blocked"
    assert error is not None
    assert error["code"] == "validation_failed"
    assert error["message"] == "Guardrail version 1 status is FAILED."
    assert "ApplyGuardrail" not in [
        item["operation"] for item in result["operations"]
    ]
    assert result["operations"][-1]["operation"] == "DeleteGuardrail"


def test_stubber_readiness_get_error_skips_version(
    tmp_path: Path,
) -> None:
    def prepare(sts: Any, bedrock: Any, _runtime: Any) -> None:
        _add_owned_start(sts, bedrock)
        bedrock.add_client_error(
            "get_guardrail",
            service_error_code="AccessDeniedException",
            service_message="denied",
            expected_params={"guardrailIdentifier": _STUB_GUARD_ID},
            http_status_code=403,
        )
        _add_teardown(bedrock)

    clock = _PollClock()
    result = _run_stubbed(tmp_path, prepare, clock)

    assert clock.sleeps == []
    assert result["error"] is None
    assert _result_data(result)["guardrail_status"] == "GET_FAILED"
    assert result["operations"][4]["error_code"] == "authorization_denied"
    assert "CreateGuardrailVersion" not in [
        item["operation"] for item in result["operations"]
    ]
    assert result["operations"][-1]["operation"] == "DeleteGuardrail"


def test_stubber_expired_clock_does_not_poll(
    tmp_path: Path,
) -> None:
    def prepare(sts: Any, bedrock: Any, _runtime: Any) -> None:
        _add_owned_start(sts, bedrock)
        _add_teardown(bedrock)

    clock = _ExpireClock()
    result = _run_stubbed(tmp_path, prepare, clock, wall=1)
    gets = [
        item
        for item in result["operations"]
        if item["operation"] == "GetGuardrail"
    ]
    error = result["error"]

    assert clock.sleeps == []
    assert len(gets) == 2
    assert gets[0]["status"] == "ok"
    assert error is not None
    assert error["code"] == "timeout"
    assert "CreateGuardrailVersion" not in [
        item["operation"] for item in result["operations"]
    ]


def test_readiness_mark_ignores_unrelated_and_failed_gets() -> None:
    other = operation(service="local", operation="Other")
    teardown = operation(
        service="bedrock",
        operation="GetGuardrail",
        phase="teardown",
    )
    denied = operation(
        service="bedrock",
        operation="GetGuardrail",
        phase="setup",
        status="blocked",
        error_code="authorization_denied",
    )
    ready = operation(
        service="bedrock",
        operation="GetGuardrail",
        phase="setup",
    )

    _block_guardrail_readiness([ready, other], 0, code="timeout")
    assert other["status"] == "ok"
    assert ready["status"] == "blocked"
    assert ready["error_code"] == "timeout"

    _block_guardrail_readiness([teardown], 0, code="timeout")
    assert teardown["status"] == "ok"
    _block_guardrail_readiness([denied], 0, code="validation_failed")
    assert denied["error_code"] == "authorization_denied"
    _block_guardrail_readiness([], 0, code="timeout")


def test_guardrails_verify_child_uses_setup_phase() -> None:
    class PhaseVerifier:
        def __init__(self) -> None:
            self.phase = ""

        def verify(
            self,
            _entry: object,
            _kind: object,
            *,
            phase: str,
        ) -> bool:
            self.phase = phase
            return True

    verifier = PhaseVerifier()
    assert _verify_for_child(cast("Any", verifier), object(), object())
    assert verifier.phase == "setup"


def _result_data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(result["data"])


def _mapping(value: object) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", value)


def _marker_lines(path: str, marker: str) -> list[str]:
    start = f"# slide: {marker}"
    end = f"# end-slide: {marker}"
    active = False
    lines: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.strip() == start:
                active = True
                continue
            if line.strip() == end:
                break
            if active:
                lines.append(line.removeprefix("    "))
    return lines
