from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from awsai_demo.cli import (
    _network_policy,
    build_parser,
    dispatch,
    doctor_result,
    execute_demo,
    main,
    render,
)
from awsai_demo.contracts import DemoResult, result
from awsai_demo.manifest import ManifestStore
from awsai_demo.network import NetworkPolicy, environment_for_subprocess
from awsai_demo.registry import get_demo
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    import argparse

    from awsai_demo.policy import ExecutionPolicy
    from awsai_demo.registry import DemoSpec


def local_result(status: str = "ok") -> DemoResult:
    return result(
        demo="example",
        technology="Example",
        lane="models",
        lifecycle_refs=["bedrock"],
        requested_execution="offline",
        headline="Example",
        status=status,
    )


def test_catalogue_main_and_render() -> None:
    out, err = io.StringIO(), io.StringIO()
    assert (
        main(
            ["list", "--format", "json"],
            settings_loader=Settings,
            stdout=out,
            stderr=err,
        )
        == 0
    )
    payload = json.loads(out.getvalue())
    assert len(payload["data"]) == 30
    assert not err.getvalue()
    assert "local_execution" in render(local_result(), "pretty")
    assert json.loads(render(local_result(), "json"))["status"] == "ok"


def test_utility_dispatch_and_missing_demos(tmp_path: Path) -> None:
    parser = build_parser()
    assert dispatch(parser.parse_args(["lineage"]), Settings())["data"]
    filtered = dispatch(
        parser.parse_args(["lineage", "--status", "sunset"]), Settings()
    )
    assert all(
        row["normalized_status"] == "sunset" for row in filtered["data"]
    )
    assert (
        dispatch(parser.parse_args(["doctor"]), Settings())["mode"]
        == "local_execution"
    )
    cost = dispatch(parser.parse_args(["cost"]), Settings())
    assert cost["mode"] == "local_contract"
    assert cost["status"] == "ok"
    assert dispatch(
        parser.parse_args(["cleanup"]), Settings(), run_directory=tmp_path
    )["data"]["dry_run"]


@pytest.mark.parametrize("status", ["ok", "paused", "blocked", "error"])
def test_all_exit_codes(status: str) -> None:
    def executor(
        _: DemoSpec, __: argparse.Namespace, ___: Settings
    ) -> DemoResult:
        return local_result(status)

    out = io.StringIO()
    code = main(
        ["all", "--lane", "local", "--format", "json"],
        settings_loader=Settings,
        executor=executor,
        stdout=out,
    )
    payload = json.loads(out.getvalue())
    assert payload["status"] == status
    assert len(payload["children"]) == 2
    assert code == int(status == "error")


def test_sanitized_errors() -> None:
    def explode() -> Settings:
        message = "bad account 123456789012\nBearer demo-token \u2014 failure"
        raise RuntimeError(message)

    out, err = io.StringIO(), io.StringIO()
    assert (
        main(["doctor"], settings_loader=explode, stdout=out, stderr=err) == 1
    )
    assert "123456789012" not in err.getvalue()
    assert "demo-token" not in err.getvalue()
    assert len(err.getvalue().splitlines()) == 1
    assert err.getvalue().isascii()
    assert not out.getvalue()
    assert main(["not-a-command"], stdout=out, stderr=err) == 1


def test_pretty_catalogue_and_batch_are_compact() -> None:
    for command in ("list", "all"):
        out = io.StringIO()
        assert (
            main(
                [command, "--lane", "local"],
                settings_loader=Settings,
                stdout=out,
            )
            == 0
        )
        assert "localstack-ai" in out.getvalue()
        assert len(out.getvalue().splitlines()) == 4


@pytest.mark.parametrize(
    "arguments",
    [
        ["--approve"],
        ["--resume", "run"],
        ["--approver", "Pat"],
        ["--resume", "run", "--deny"],
        ["--resume", "run", "--approve", "--approver", " "],
        ["--simulate-approval", "--execution", "emulator"],
    ],
)
def test_decision_invalid_approval(arguments: list[str]) -> None:
    with pytest.raises(ValueError):
        dispatch(
            build_parser().parse_args(["decision", *arguments]), Settings()
        )


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--simulate-approval"],
        ["--replay", "run"],
        ["--resume", "run", "--approve", "--approver", "Pat"],
        ["--resume", "run", "--deny", "--approver", "Pat"],
    ],
)
def test_decision_accepts_explicit_protocol(arguments: list[str]) -> None:
    accepted = dispatch(
        build_parser().parse_args(["decision", *arguments]),
        Settings(),
        executor=lambda *_: local_result(),
    )
    assert accepted["status"] == "ok"


def test_network_configuration() -> None:
    parser = build_parser()
    assert (
        _network_policy(
            parser.parse_args(["cost", "--execution", "live"]), Settings()
        )
        is None
    )
    emulator = _network_policy(
        parser.parse_args(["cost", "--execution", "emulator"]), Settings()
    )
    assert emulator is not None
    emulator.assert_connect_allowed("localhost", 4566)
    otlp = parser.parse_args(["observability", "--trace", "otlp"])
    allowed = _network_policy(
        otlp, Settings(otel_exporter_otlp_endpoint="http://127.0.0.1:4321")
    )
    assert allowed is not None
    allowed.assert_connect_allowed("127.0.0.1", 4321)
    for endpoint in (None, "https://localhost:4321", "http://localhost"):
        with pytest.raises(ValueError, match="OTLP"):
            _network_policy(
                otlp, Settings(otel_exporter_otlp_endpoint=endpoint)
            )
    with pytest.raises(ValueError, match="OTLP"):
        _network_policy(
            parser.parse_args(
                ["cost", "--execution", "emulator", "--trace", "otlp"]
            ),
            Settings(),
        )


def test_cleanup_inventory(tmp_path: Path) -> None:
    run_id = "a" * 12
    store = ManifestStore(tmp_path / run_id / "manifest.json")
    store.record_intent(
        run_id=run_id,
        account_fingerprint="fingerprint",
        region="us-east-1",
        service="bedrock",
        operation="CreateGuardrail",
        intended_name=f"awsai-{run_id}-guard",
        client_token=run_id,
        tags={"run-id": run_id},
        naming_scheme="guardrail",
    )
    parser = build_parser()
    for args in (["cleanup"], ["cleanup", "--run-id", run_id]):
        report = dispatch(
            parser.parse_args(args), Settings(), run_directory=tmp_path
        )
        assert len(report["data"]["entries"]) == 1
        assert "exact_id" not in json.dumps(report)
    with pytest.raises(ValueError, match="requires --run-id"):
        dispatch(
            parser.parse_args(["cleanup", "--execute"]),
            Settings(),
            run_directory=tmp_path,
        )
    report = dispatch(
        parser.parse_args(["cleanup", "--execute", "--run-id", "b" * 12]),
        Settings(),
        run_directory=tmp_path,
    )
    assert report["data"]["dry_run"] is False
    assert (
        _network_policy(
            parser.parse_args(["cleanup", "--execute", "--run-id", "b" * 12]),
            Settings(),
        )
        is None
    )
    with pytest.raises(ValueError, match="run-id"):
        dispatch(
            parser.parse_args(["cleanup", "--run-id", "../other"]),
            Settings(),
            run_directory=tmp_path,
        )


def test_lazy_runner_adapters() -> None:
    spec = get_demo("bedrock-runtime")
    args = build_parser().parse_args([spec.name])
    module = ModuleType("fake")

    def run(
        *, execution: str, settings: Settings, policy: ExecutionPolicy
    ) -> DemoResult:
        assert execution == "offline"
        assert settings.region == "us-east-1"
        assert policy.max_model_calls == 6
        return local_result()

    async def async_run(**kwargs: object) -> DemoResult:
        assert kwargs["execution"] == "offline"
        return local_result()

    for runner in (run, async_run):
        setattr(module, spec.entrypoint, runner)
        assert (
            execute_demo(
                spec, args, Settings(), module_loader=lambda _: module
            )["status"]
            == "ok"
        )

    def missing(_: str) -> ModuleType:
        message = "optional_sdk"
        raise ModuleNotFoundError(message, name="optional_sdk")

    def absent(_: str) -> ModuleType:
        raise ModuleNotFoundError(spec.module, name=spec.module)

    assert (
        execute_demo(spec, args, Settings(), module_loader=absent)["error"][
            "code"
        ]
        == "contract_only"
    )

    assert (
        execute_demo(spec, args, Settings(), module_loader=missing)["error"][
            "code"
        ]
        == "sdk_missing"
    )
    for runner in (missing_runner, missing_async_runner):
        setattr(module, spec.entrypoint, runner)
        report = execute_demo(
            spec, args, Settings(), module_loader=lambda _: module
        )
        assert report["error"]["code"] == "sdk_missing"


def missing_runner(**_: object) -> DemoResult:
    message = "optional_sdk"
    raise ModuleNotFoundError(message, name="optional_sdk")


async def missing_async_runner(**_: object) -> DemoResult:
    return missing_runner()


def test_module_entrypoint_offline(tmp_path: Path) -> None:
    env = environment_for_subprocess(
        NetworkPolicy(),
        os.environ,
        bootstrap_dir=tmp_path / "bootstrap",
    )
    env["AWS_CREDS_FILE_PATH"] = str(tmp_path / "must-not-be-read.csv")
    env["COVERAGE_PROCESS_START"] = str(Path("pyproject.toml").resolve())
    completed = subprocess.run(
        [sys.executable, "-m", "awsai_demo.cli", "list", "--format", "json"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(json.loads(completed.stdout)["data"]) == 30


def test_probe_wrapper_uses_only_injected_clients() -> None:
    client = Mock()
    client.meta.endpoint_url = "https://example.amazonaws.com"
    session = Mock()
    session.client.return_value = client

    def factory(**_: str) -> object:
        return session

    report = doctor_result(Settings(), True, session_factory=factory)
    assert report["mode"] == "live_service"
    assert report["data"]["probed"]
    assert session.client.called


def test_probe_wrapper_accepts_an_injected_port() -> None:
    from awsai_demo.contracts import operation
    from awsai_demo.doctor import DoctorCheck, DoctorProbeResult

    port = Mock()
    port.probe.return_value = DoctorProbeResult(
        checks=(DoctorCheck("sts", "ok", "redacted"),),
        operations=(
            operation(
                service="sts",
                operation="GetCallerIdentity",
                execution_target="aws",
                mode="live_identity",
                transport="aws",
                endpoint_url="https://sts.us-east-1.amazonaws.com",
                response_received=True,
            ),
        ),
    )
    report = doctor_result(Settings(), True, probe_port=port)
    assert report["mode"] == "live_identity"


def test_probe_wrapper_preserves_a_skipped_probe() -> None:
    from awsai_demo.contracts import operation
    from awsai_demo.doctor import DoctorCheck, DoctorProbeResult

    port = Mock()
    port.probe.return_value = DoctorProbeResult(
        checks=(DoctorCheck("sts", "blocked", "wall-clock limit exceeded"),),
        operations=(
            operation(
                service="sts",
                operation="GetCallerIdentity",
                execution_target="aws",
                mode="not_run",
                status="blocked",
                response_received=False,
                request_validated=False,
                error_code="timeout",
            ),
        ),
    )
    report = doctor_result(Settings(), True, probe_port=port)
    assert report["mode"] == "not_run"
    assert report["status"] == "blocked"
    assert report["evidence"]["sdk_invoked"] is False
    assert report["evidence"]["network_attempted"] is False
    assert report["evidence"]["aws_executed"] is False


def test_decision_flags_reach_lazy_implementation() -> None:
    spec = get_demo("decision")
    args = build_parser().parse_args(
        [
            "decision",
            "--resume",
            "run",
            "--approve",
            "--approver",
            "Pat",
        ]
    )
    module = ModuleType("decision_fixture")

    def run(**kwargs: object) -> DemoResult:
        assert kwargs["resume"] == "run"
        assert kwargs["approve"] is True
        assert kwargs["approver"] == "Pat"
        assert kwargs["simulate_approval"] is False
        return local_result()

    setattr(module, spec.entrypoint, run)
    execute_demo(spec, args, Settings(), module_loader=lambda _: module)


def test_flags_reach_the_runner() -> None:
    def check(
        _: DemoSpec, __: argparse.Namespace, settings: Settings
    ) -> DemoResult:
        assert settings.model == "fixture-model"
        assert settings.provider == "offline"
        assert settings.trace == "memory"
        assert settings.allow_create
        assert settings.region == "eu-west-1"
        assert settings.aws_profile == "example"
        return local_result()

    assert (
        main(
            [
                "cost",
                "--model",
                "fixture-model",
                "--provider",
                "offline",
                "--trace",
                "memory",
                "--allow-create",
                "--region",
                "eu-west-1",
                "--profile",
                "example",
            ],
            settings_loader=Settings,
            executor=check,
            stdout=io.StringIO(),
        )
        == 0
    )


def test_cli_suppresses_raw_sdk_logs_but_keeps_cost_notices(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from awsai_demo.billing import cost_notice

    def noisy_executor(
        _: DemoSpec, __: argparse.Namespace, ___: Settings
    ) -> DemoResult:
        logging.getLogger("botocore").error(
            "raw SDK diagnostic: %s", "arn:aws:iam::123456789012:user/private"
        )
        cost_notice("Reserved USD 0.000001 before dispatch")
        return local_result()

    assert (
        main(
            ["cost"],
            settings_loader=Settings,
            executor=noisy_executor,
            stdout=io.StringIO(),
        )
        == 0
    )
    assert not caplog.records
    assert "Reserved USD 0.000001" in capsys.readouterr().err
