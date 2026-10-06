"""CLI entry point with lazy demos and sanitized output."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import re
import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Never, cast
from urllib.parse import urlsplit

from awsai_demo.contracts import (
    DemoResult,
    Execution,
    error,
    evidence,
    missing_configuration,
    not_run,
    result,
)
from awsai_demo.lineage import STATUSES, load_lineage
from awsai_demo.redact import quiet_sdk_logging, redact, sanitize_exception
from awsai_demo.registry import (
    DEMOS,
    LANES,
    DemoSpec,
    Lane,
    get_demo,
    list_demos,
)
from awsai_demo.serialization import normalize_timestamps

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence
    from types import ModuleType
    from typing import TextIO

    from awsai_demo.credentials import SessionFactory
    from awsai_demo.doctor import ProbePort
    from awsai_demo.network import GuardMode, NetworkPolicy
    from awsai_demo.runtime import Settings

type DemoExecutor = Callable[
    [DemoSpec, argparse.Namespace, "Settings"], DemoResult
]


class CommandParser(argparse.ArgumentParser):
    """Route parsing errors through the same redaction boundary."""

    def error(self, message: str) -> Never:
        """Route sensitive argument errors to the redaction boundary."""
        raise ValueError(message)


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI without importing any demo or SDK."""
    parser = CommandParser(prog="awsai-demo")
    commands = parser.add_subparsers(dest="command", required=True)
    for demo in DEMOS:
        command = commands.add_parser(demo.name, help=demo.technology)
        _common_arguments(command)
        if demo.name == "decision":
            action = command.add_mutually_exclusive_group()
            action.add_argument("--simulate-approval", action="store_true")
            action.add_argument("--resume", metavar="RUN_ID")
            action.add_argument("--replay", metavar="RUN_ID")
            approval = command.add_mutually_exclusive_group()
            approval.add_argument("--approve", action="store_true")
            approval.add_argument("--deny", action="store_true")
            command.add_argument("--approver")
    catalogue = commands.add_parser("list", help="List the complete catalogue")
    catalogue.add_argument("--lane", choices=LANES)
    _format_argument(catalogue)
    lineage = commands.add_parser(
        "lineage", help="Read sourced lifecycle facts"
    )
    lineage.add_argument("--status", choices=sorted(STATUSES))
    _format_argument(lineage)
    doctor = commands.add_parser("doctor", help="Inspect configuration")
    doctor.add_argument("--probe", action="store_true")
    doctor.add_argument("--region")
    doctor.add_argument("--profile")
    _format_argument(doctor)
    all_command = commands.add_parser(
        "all", help="Run a lane or the catalogue"
    )
    all_command.add_argument("--lane", choices=LANES)
    _common_arguments(all_command)
    cleanup = commands.add_parser(
        "cleanup", help="Inspect owned cleanup intents"
    )
    cleanup.add_argument("--run-id")
    cleanup.add_argument("--execute", action="store_true")
    _format_argument(cleanup)
    return parser


def _format_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format", choices=("pretty", "json"), default="pretty"
    )


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--execution",
        choices=("offline", "emulator", "live"),
        default="offline",
    )
    _format_argument(parser)
    parser.add_argument("--region")
    parser.add_argument("--profile")
    parser.add_argument("--model")
    parser.add_argument("--provider", choices=("offline", "bedrock", "openai"))
    parser.add_argument(
        "--trace", choices=("off", "memory", "otlp"), default="off"
    )
    parser.add_argument("--allow-create", action="store_true")


def _validate_decision(args: argparse.Namespace) -> None:
    if args.command != "decision":
        return
    if args.simulate_approval and args.execution != "offline":
        message = "--simulate-approval is offline only"
        raise ValueError(message)
    if args.approve or args.deny:
        if not args.resume or not args.approver or not args.approver.strip():
            message = (
                "Approval or denial requires --resume and --approver NAME"
            )
            raise ValueError(message)
    elif args.resume or args.approver:
        message = "--resume requires --approve or --deny and --approver NAME"
        raise ValueError(message)


async def _await_result(value: Awaitable[DemoResult]) -> DemoResult:
    return await value


def _missing_live_credentials(
    *,
    demo: str,
    technology: str,
    lane: Lane,
    lifecycle_refs: Sequence[str],
    execution: Execution,
) -> DemoResult:
    """Return not_run when live credentials are missing."""
    from awsai_demo.credentials import (
        LIVE_CREDENTIAL_HEADLINE,
        LIVE_CREDENTIALS_MESSAGE,
        LIVE_CREDENTIALS_NEXT_STEP,
    )

    return missing_configuration(
        demo=demo,
        technology=technology,
        lane=lane,
        lifecycle_refs=lifecycle_refs,
        requested_execution=execution,
        headline=LIVE_CREDENTIAL_HEADLINE,
        message=LIVE_CREDENTIALS_MESSAGE,
        next_steps=[LIVE_CREDENTIALS_NEXT_STEP],
    )


def _blocked_live_demo(
    spec: DemoSpec,
    execution: Execution,
    settings: Settings,
    session_factory: SessionFactory | None,
) -> DemoResult | None:
    """Return not_run when a real live demo has no credentials."""
    if execution != "live":
        return None
    from awsai_demo.credentials import live_credentials_resolved

    if live_credentials_resolved(
        settings=settings,
        session_factory=session_factory,
    ):
        return None
    return _missing_live_credentials(
        demo=spec.name,
        technology=spec.technology,
        lane=spec.lane,
        lifecycle_refs=spec.lifecycle_refs,
        execution=execution,
    )


def _catalogue_missing_credentials(
    *,
    execution: str,
    settings: Settings,
    executor: DemoExecutor,
    session_factory: SessionFactory | None,
) -> bool:
    """Block the real live catalogue when credentials are missing."""
    if executor is not execute_demo or execution != "live":
        return False
    from awsai_demo.credentials import live_credentials_resolved

    return not live_credentials_resolved(
        settings=settings,
        session_factory=session_factory,
    )


def execute_demo(
    spec: DemoSpec,
    args: argparse.Namespace,
    settings: Settings,
    *,
    module_loader: Callable[[str], ModuleType] = importlib.import_module,
    session_factory: SessionFactory | None = None,
) -> DemoResult:
    """Load a demo lazily, reporting catalogue-only entries."""
    from awsai_demo.policy import ExecutionPolicy

    execution = cast("Execution", args.execution)
    blocked = _blocked_live_demo(
        spec,
        execution,
        settings,
        session_factory,
    )
    if blocked is not None:
        return blocked
    try:
        module = module_loader(spec.module)
    except ModuleNotFoundError as exc:
        unavailable = exc.name == spec.module
        return not_run(
            demo=spec.name,
            technology=spec.technology,
            lane=spec.lane,
            lifecycle_refs=spec.lifecycle_refs,
            requested_execution=execution,
            headline="Demo implementation is not available.",
            code="contract_only" if unavailable else "sdk_missing",
            message=(
                "Catalogue entry only; this demo is not implemented yet."
                if unavailable
                else "Install the dependency required by this demo."
            ),
        )
    runner = cast(
        "Callable[..., DemoResult | Awaitable[DemoResult]]",
        getattr(module, spec.entrypoint),
    )
    arguments: dict[str, object] = {
        "execution": execution,
        "settings": settings,
        "policy": ExecutionPolicy(),
    }
    if spec.name == "decision":
        arguments.update(
            {
                name: getattr(args, name, None)
                for name in (
                    "simulate_approval",
                    "resume",
                    "replay",
                    "approve",
                    "deny",
                    "approver",
                )
            }
        )
    try:
        value = runner(**arguments)
        if inspect.isawaitable(value):
            return asyncio.run(_await_result(value))
    except ModuleNotFoundError:
        return not_run(
            demo=spec.name,
            technology=spec.technology,
            lane=spec.lane,
            lifecycle_refs=spec.lifecycle_refs,
            requested_execution=execution,
            headline="A lazy dependency is not installed.",
            code="sdk_missing",
            message="Install the dependency required by this demo.",
        )
    return value


def _network_policy(
    args: argparse.Namespace,
    settings: Settings,
) -> NetworkPolicy | None:
    from awsai_demo.network import NetworkPolicy, registered_loopback_endpoint

    lane = getattr(args, "execution", "offline")
    if (
        lane == "live"
        or getattr(args, "probe", False)
        or (args.command == "cleanup" and args.execute)
    ):
        return None
    endpoints = []
    if lane == "emulator":
        parsed = urlsplit(settings.localstack_endpoint)
        endpoints.append(
            registered_loopback_endpoint(
                "localstack", parsed.hostname or "", parsed.port or 4566
            )
        )
    if getattr(args, "trace", "off") == "otlp":
        parsed = urlsplit(settings.otel_exporter_otlp_endpoint or "")
        if lane != "offline" or parsed.scheme != "http" or parsed.port is None:
            message = "OTLP needs an offline HTTP loopback host and port"
            raise ValueError(message)
        endpoints.append(
            registered_loopback_endpoint(
                "otlp", parsed.hostname or "", parsed.port
            )
        )
    return NetworkPolicy(
        mode=cast("GuardMode", lane), allowed_endpoints=frozenset(endpoints)
    )


def _cleanup(
    args: argparse.Namespace,
    root: Path,
    settings: Settings,
) -> DemoResult:
    from awsai_demo.manifest import ManifestStore

    if args.run_id is not None and not re.fullmatch(
        r"[a-f0-9]{12}", args.run_id
    ):
        message = "run-id must be exactly 12 lowercase hexadecimal characters"
        raise ValueError(message)
    paths = (
        [root / args.run_id / "manifest.json"]
        if args.run_id
        else sorted(root.glob("*/manifest.json"))
    )
    entries = [
        entry for path in paths for entry in ManifestStore(path).list_entries()
    ]
    summary = [
        {"service": e.service, "operation": e.operation, "status": e.status}
        for e in entries
    ]
    if args.execute:
        from awsai_demo.resource_cleanup import execute_cleanup

        if args.run_id is None:
            message = "cleanup --execute requires --run-id"
            raise ValueError(message)
        return execute_cleanup(
            run_id=args.run_id,
            root=root,
            settings=settings,
        )
    return result(
        demo="cleanup",
        technology="Owned resource cleanup",
        lane="operations",
        lifecycle_refs=["iam-sts"],
        requested_execution="offline",
        headline="Cleanup dry run; no resource was changed.",
        data={"entries": summary, "dry_run": True},
    )


def doctor_result(
    settings: Settings,
    probe: bool,
    *,
    probe_port: ProbePort | None = None,
    session_factory: SessionFactory | None = None,
) -> DemoResult:
    """Probe only on request, retaining operation evidence."""
    from awsai_demo.credentials import resolved_credential, select_session
    from awsai_demo.doctor import installed_package_versions, run_doctor

    selected = None
    if probe and probe_port is None:
        selected = select_session(
            execution="live",
            settings=settings,
            session_factory=session_factory,
        )
        if resolved_credential(selected.session) is None:
            return _missing_live_credentials(
                demo="doctor",
                technology="Configuration checks",
                lane="operations",
                lifecycle_refs=["iam-sts"],
                execution="live",
            )
    report = run_doctor(
        settings,
        probe=probe,
        probe_port=probe_port,
        selected_session=selected,
    )
    return result(
        demo="doctor",
        technology="Configuration checks",
        lane="operations",
        lifecycle_refs=["iam-sts"],
        requested_execution="live" if probe else "offline",
        headline="Explicit read-only probes."
        if probe
        else "Configuration only.",
        operations=report.operations,
        evidence=evidence(
            sdk_invoked=any(
                op["request_validated"] for op in report.operations
            ),
            network_attempted=any(
                op["transport"] != "none" for op in report.operations
            ),
            aws_executed=any(
                op["execution_target"] == "aws" and op["response_received"]
                for op in report.operations
            ),
            credential_source=selected.source
            if selected is not None
            else "none",
            region=settings.region,
        ),
        data={
            "probed": report.probed,
            "checks": [asdict(c) for c in report.checks],
            "packages": installed_package_versions(),
        },
    )


def dispatch(
    args: argparse.Namespace,
    settings: Settings,
    *,
    executor: DemoExecutor = execute_demo,
    data_directory: Path | None = None,
    run_directory: Path = Path(".awsai_runs"),
    doctor_runner: Callable[[Settings, bool], DemoResult] = doctor_result,
    session_factory: SessionFactory | None = None,
) -> DemoResult:
    """Execute a utility or delegate to a demo adapter."""
    if args.command == "list":
        return result(
            demo="list",
            technology="Workshop catalogue",
            lane="operations",
            lifecycle_refs=["bedrock"],
            requested_execution="offline",
            headline="30 registered demos; imports are lazy.",
            data=[asdict(spec) for spec in list_demos(args.lane)],
        )
    if args.command == "lineage":
        records = load_lineage(data_directory)
        return result(
            demo="lineage",
            technology="AWS lineage",
            lane="operations",
            lifecycle_refs=["bedrock"],
            requested_execution="offline",
            headline="Dated, sourced lifecycle records.",
            data=[
                asdict(record)
                for record in records
                if args.status is None
                or record.normalized_status == args.status
            ],
        )
    if args.command == "doctor":
        if doctor_runner is doctor_result:
            return doctor_result(
                settings,
                args.probe,
                session_factory=session_factory,
            )
        return doctor_runner(settings, args.probe)
    if args.command == "cleanup":
        return _cleanup(args, run_directory, settings)
    _validate_decision(args)
    if args.command == "all":
        if _catalogue_missing_credentials(
            execution=args.execution,
            settings=settings,
            executor=executor,
            session_factory=session_factory,
        ):
            children = [
                _missing_live_credentials(
                    demo=spec.name,
                    technology=spec.technology,
                    lane=spec.lane,
                    lifecycle_refs=spec.lifecycle_refs,
                    execution="live",
                )
                for spec in list_demos(args.lane)
            ]
            return result(
                demo="all",
                technology="AWS AI workshop",
                lane="operations",
                lifecycle_refs=["bedrock"],
                requested_execution="live",
                headline="Catalogue execution summary.",
                children=children,
            )
        from awsai_demo.billing import (
            SnapshotPrices,
            open_budget_run,
            use_budget_run,
        )
        from awsai_demo.policy import ExecutionPolicy

        budget_scope = (
            use_budget_run(
                open_budget_run(
                    region=settings.region,
                    policy=ExecutionPolicy(),
                    root=run_directory,
                    prices=SnapshotPrices(Path("data/pricing_snapshot.json")),
                    aggregate=True,
                )
            )
            if args.execution == "live"
            else nullcontext()
        )
        with budget_scope:
            children = [
                _checked_demo(
                    spec,
                    args,
                    settings,
                    executor,
                    session_factory=session_factory,
                )
                for spec in list_demos(args.lane)
            ]
        return result(
            demo="all",
            technology="AWS AI workshop",
            lane="operations",
            lifecycle_refs=["bedrock"],
            requested_execution=args.execution,
            headline="Catalogue execution summary.",
            children=children,
        )
    return _checked_demo(
        get_demo(args.command),
        args,
        settings,
        executor,
        session_factory=session_factory,
    )


def _checked_demo(
    spec: DemoSpec,
    args: argparse.Namespace,
    settings: Settings,
    executor: DemoExecutor,
    *,
    session_factory: SessionFactory | None = None,
) -> DemoResult:
    """Isolate execution and serialization failures to one demo."""
    try:
        if executor is execute_demo:
            payload = execute_demo(
                spec,
                args,
                settings,
                session_factory=session_factory,
            )
        else:
            payload = executor(spec, args, settings)
    except Exception as exc:
        return _demo_failure(spec, args, exc)
    try:
        return _json_ready(payload)
    except Exception as exc:
        return _demo_failure(spec, args, exc, prior=payload)


def _demo_failure(
    spec: DemoSpec,
    args: argparse.Namespace,
    exc: Exception,
    *,
    prior: DemoResult | None = None,
) -> DemoResult:
    """Keep validated provenance if demo data cannot be printed."""
    failed = result(
        demo=spec.name,
        technology=spec.technology,
        lane=spec.lane,
        lifecycle_refs=spec.lifecycle_refs,
        requested_execution=args.execution,
        headline="Demo failed before a serializable result was available.",
        status="error",
        error=error("validation_failed", sanitize_exception(exc)),
        data={"operation_evidence_unavailable": True},
    )
    if prior is not None:
        try:
            retained = {
                **failed,
                "operations": prior["operations"],
                "evidence": prior["evidence"],
                "children": prior["children"],
                "mode": prior["mode"],
                "data": {"data_unavailable": True},
            }
            return _json_ready(cast("DemoResult", retained))
        except Exception:
            # Do not present malformed provenance as observed evidence.
            return failed
    return failed


def _json_ready(payload: DemoResult) -> DemoResult:
    """Reject unsupported values without an arbitrary text fallback."""
    if set(payload) != DemoResult.__required_keys__:
        message = "invalid demo result envelope fields"
        raise ValueError(message)
    fields = cast("dict[str, Any]", dict(payload))
    if fields.pop("schema_version") != 1:
        message = "unsupported demo result schema version"
        raise ValueError(message)
    validated = result(**fields)
    for child in validated["children"]:
        _json_ready(child)
    safe = cast("DemoResult", redact(normalize_timestamps(validated)))
    json.dumps(safe, ensure_ascii=True, sort_keys=True, allow_nan=False)
    return safe


def render(payload: DemoResult, output_format: str) -> str:
    """Serialize redacted data as ASCII for Windows terminals."""
    safe = _json_ready(payload)
    text = json.dumps(safe, ensure_ascii=True, sort_keys=True, indent=2)
    if output_format == "json":
        return text
    summary = f"{safe['demo']}: {safe['status']} ({safe['mode']})"
    lines = [summary, safe["headline"]]
    if safe["mode"] == "batch":
        lines.extend(
            f"  {child['demo']:<24} {child['status']:<8} {child['mode']}"
            for child in safe["children"]
        )
    elif safe["demo"] == "list":
        rows = cast("list[dict[str, object]]", safe["data"])
        lines.extend(
            f"  {row['name']!s:<24} {row['lane']!s:<12} {row['technology']}"
            for row in rows
        )
    else:
        lines.append(text)
    return "\n".join(lines).encode("ascii", "backslashreplace").decode()


def main(
    argv: Sequence[str] | None = None,
    *,
    executor: DemoExecutor = execute_demo,
    settings_loader: Callable[[], Settings] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    session_factory: SessionFactory | None = None,
) -> int:
    """Run the CLI, sanitizing every unexpected top-level diagnostic."""
    from awsai_demo.network import network_guard
    from awsai_demo.runtime import load_settings, with_cli_overrides

    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    try:
        args = build_parser().parse_args(argv)
        settings = (
            load_settings if settings_loader is None else settings_loader
        )()
        settings = with_cli_overrides(
            settings,
            region=getattr(args, "region", None),
            profile=getattr(args, "profile", None),
            model=getattr(args, "model", None),
            provider=getattr(args, "provider", None),
            trace=getattr(args, "trace", None),
            allow_create=getattr(args, "allow_create", None),
        )
        policy = _network_policy(args, settings)
        with (
            quiet_sdk_logging(),
            nullcontext() if policy is None else network_guard(policy),
        ):
            payload = dispatch(
                args,
                settings,
                executor=executor,
                session_factory=session_factory,
            )
        out.write(render(payload, args.format) + "\n")
    except Exception as exc:
        diagnostic = sanitize_exception(exc).encode(
            "ascii", "backslashreplace"
        )
        err.write(diagnostic.decode() + "\n")
        return 1
    return int(payload["status"] == "error")


if __name__ == "__main__":
    raise SystemExit(main())
