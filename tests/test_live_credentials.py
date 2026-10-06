from __future__ import annotations

import io
import json
from types import ModuleType
from typing import TYPE_CHECKING

import pytest
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

from awsai_demo.cli import (
    _catalogue_missing_credentials,
    build_parser,
    dispatch,
    doctor_result,
    execute_demo,
    main,
)
from awsai_demo.contracts import DemoResult, result
from awsai_demo.credentials import (
    LIVE_CREDENTIAL_HEADLINE,
    LIVE_CREDENTIALS_MESSAGE,
    LIVE_CREDENTIALS_NEXT_STEP,
    live_credentials_resolved,
    resolved_credential,
)
from awsai_demo.registry import get_demo, list_demos
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_SOURCES = (
    "--profile",
    "AWS_PROFILE",
    "AWS_CREDS_FILE_PATH",
    "default chain",
)


class EmptySession:
    def __init__(self) -> None:
        self.clients: list[str] = []

    def get_credentials(self) -> None:
        return None

    def client(self, name: str, **_kwargs: object) -> object:
        self.clients.append(name)
        message = "live call dispatched without credentials"
        raise AssertionError(message)


class ReadySession:
    def get_credentials(self) -> object:
        return object()


class NestedReadySession:
    def __init__(self) -> None:
        self._session = ReadySession()


class NoCredentialSession(EmptySession):
    def get_credentials(self) -> None:
        raise NoCredentialsError


class PartialCredentialSession(EmptySession):
    def get_credentials(self) -> None:
        raise PartialCredentialsError(
            provider="env",
            cred_var="aws_secret_access_key",
        )


def _empty_factory(**_kwargs: str) -> EmptySession:
    return EmptySession()


def _ready_factory(**_kwargs: str) -> ReadySession:
    return ReadySession()


def _assert_missing(payload: DemoResult) -> None:
    message = "" if payload["error"] is None else payload["error"]["message"]
    positions = [message.index(source) for source in _SOURCES]
    assert payload["mode"] == "not_run"
    assert payload["status"] == "blocked"
    assert payload["error"] is not None
    assert payload["error"]["code"] == "missing_configuration"
    assert payload["headline"] == LIVE_CREDENTIAL_HEADLINE
    assert message == LIVE_CREDENTIALS_MESSAGE
    assert positions == sorted(positions)
    assert payload["next_steps"] == [LIVE_CREDENTIALS_NEXT_STEP]
    assert payload["operations"] == []


def _forbid_import(name: str) -> ModuleType:
    message = f"demo module loaded: {name}"
    raise AssertionError(message)


def test_resolved_credential_reports_each_local_outcome() -> None:
    class Boom:
        def get_credentials(self) -> None:
            message = "unknown credential failure"
            raise RuntimeError(message)

    assert resolved_credential(object()) is None
    assert resolved_credential(EmptySession()) is None
    assert resolved_credential(NoCredentialSession()) is None
    assert resolved_credential(PartialCredentialSession()) is None
    assert resolved_credential(ReadySession()) is not None
    assert resolved_credential(NestedReadySession()) is not None
    with pytest.raises(RuntimeError, match="unknown credential"):
        resolved_credential(Boom())


def test_live_resolution_uses_the_injected_factory_only() -> None:
    seen: list[dict[str, str]] = []

    def factory(**kwargs: str) -> EmptySession:
        seen.append(kwargs)
        return EmptySession()

    assert (
        live_credentials_resolved(
            settings=Settings(region="us-east-2"),
            session_factory=factory,
        )
        is False
    )
    assert (
        live_credentials_resolved(
            settings=Settings(aws_profile="present"),
            session_factory=_ready_factory,
        )
        is True
    )
    assert seen == [{"region_name": "us-east-2"}]


def test_every_live_demo_stops_before_dispatch() -> None:
    session = EmptySession()
    parser = build_parser()

    def factory(**_kwargs: str) -> EmptySession:
        return session

    for spec in list_demos():
        args = parser.parse_args([spec.name, "--execution", "live"])
        payload = execute_demo(
            spec,
            args,
            Settings(),
            module_loader=_forbid_import,
            session_factory=factory,
        )
        _assert_missing(payload)
        assert payload["demo"] == spec.name
    assert session.clients == []


@pytest.mark.parametrize(
    "factory",
    [
        lambda **_kwargs: NoCredentialSession(),
        lambda **_kwargs: PartialCredentialSession(),
    ],
)
def test_credential_errors_are_missing_configuration_not_crashes(
    factory: Callable[..., EmptySession],
) -> None:
    payload = execute_demo(
        get_demo("bedrock-runtime"),
        build_parser().parse_args(["bedrock-runtime", "--execution", "live"]),
        Settings(),
        module_loader=_forbid_import,
        session_factory=factory,
    )
    _assert_missing(payload)


def test_resolved_credentials_still_run_the_demo() -> None:
    module = ModuleType("ready")

    def run(**_kwargs: object) -> DemoResult:
        return result(
            demo="bedrock-runtime",
            technology="Bedrock inference",
            lane="models",
            lifecycle_refs=["bedrock"],
            requested_execution="live",
            headline="Injected runner.",
        )

    module.run_bedrock_runtime_demo = run  # type: ignore[attr-defined]
    payload = execute_demo(
        get_demo("bedrock-runtime"),
        build_parser().parse_args(["bedrock-runtime", "--execution", "live"]),
        Settings(),
        module_loader=lambda _name: module,
        session_factory=_ready_factory,
    )
    assert payload["headline"] == "Injected runner."
    assert payload["error"] is None


def test_offline_does_not_resolve_live_credentials() -> None:
    module = ModuleType("offline_cost")

    def run(**kwargs: object) -> DemoResult:
        assert kwargs["execution"] == "offline"
        return result(
            demo="cost",
            technology="Cost per successful task",
            lane="operations",
            lifecycle_refs=["bedrock-service-tiers"],
            requested_execution="offline",
            headline="Offline runner.",
        )

    module.run_cost_demo = run  # type: ignore[attr-defined]

    def factory(**kwargs: str) -> object:
        message = f"live credential lookup during offline: {kwargs}"
        raise AssertionError(message)

    payload = execute_demo(
        get_demo("cost"),
        build_parser().parse_args(["cost"]),
        Settings(),
        module_loader=lambda _name: module,
        session_factory=factory,
    )
    assert payload["headline"] == "Offline runner."


def test_cli_live_without_credentials_exits_zero() -> None:
    seen: list[dict[str, str]] = []

    def factory(**kwargs: str) -> EmptySession:
        seen.append(kwargs)
        return EmptySession()

    out, err = io.StringIO(), io.StringIO()
    code = main(
        [
            "bedrock-runtime",
            "--execution",
            "live",
            "--profile",
            "cli-name",
            "--format",
            "json",
        ],
        settings_loader=Settings,
        session_factory=factory,
        stdout=out,
        stderr=err,
    )
    payload = json.loads(out.getvalue())
    assert code == 0
    assert err.getvalue() == ""
    _assert_missing(payload)
    assert seen == [
        {"profile_name": "cli-name", "region_name": "us-east-1"},
    ]


def test_all_live_without_credentials_reserves_nothing(
    tmp_path: Path,
) -> None:
    payload = dispatch(
        build_parser().parse_args(
            ["all", "--execution", "live", "--lane", "local"]
        ),
        Settings(),
        run_directory=tmp_path,
        session_factory=_empty_factory,
    )
    assert payload["mode"] == "batch"
    assert payload["status"] == "blocked"
    assert [child["demo"] for child in payload["children"]] == [
        "localstack",
        "localstack-ai",
    ]
    assert all(child["operations"] == [] for child in payload["children"])
    for child in payload["children"]:
        _assert_missing(child)
    assert list(tmp_path.iterdir()) == []


def test_catalogue_gate_allows_a_resolved_session() -> None:
    assert (
        _catalogue_missing_credentials(
            execution="live",
            settings=Settings(),
            executor=execute_demo,
            session_factory=_ready_factory,
        )
        is False
    )


def test_doctor_probe_without_credentials_does_not_call_sts() -> None:
    session = EmptySession()

    def factory(**_kwargs: str) -> EmptySession:
        return session

    payload = doctor_result(Settings(), True, session_factory=factory)
    _assert_missing(payload)
    assert payload["demo"] == "doctor"
    assert payload["data"] is None
    assert session.clients == []


def test_custom_doctor_runner_skips_the_credential_gate() -> None:
    seen: list[bool] = []

    def runner(_settings: Settings, probe: bool) -> DemoResult:
        seen.append(probe)
        return result(
            demo="doctor",
            technology="stand-in",
            lane="operations",
            lifecycle_refs=["iam-sts"],
            requested_execution="offline",
            headline="Stand-in doctor.",
        )

    def factory(**_kwargs: str) -> EmptySession:
        message = "stand-in doctor must not resolve credentials"
        raise AssertionError(message)

    payload = dispatch(
        build_parser().parse_args(["doctor", "--probe"]),
        Settings(),
        doctor_runner=runner,
        session_factory=factory,
    )
    assert seen == [True]
    assert payload["headline"] == "Stand-in doctor."
