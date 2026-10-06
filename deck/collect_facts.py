"""Record separate offline and live measurements for the deck."""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from awsai_demo.billing import (
    SnapshotPrices,
    open_budget_run,
    use_budget_run,
)
from awsai_demo.cli import CommandParser, build_parser, dispatch, render
from awsai_demo.network import NetworkPolicy, network_guard
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.redact import quiet_sdk_logging, redact, sanitize_exception
from awsai_demo.runtime import Settings, load_settings, with_cli_overrides
from deck.facts import FactError, load_facts, valid_value

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Sequence
    from typing import TextIO

    from awsai_demo.contracts import DemoResult, OperationOutcome
    from deck.facts import FactDefinition, FactLane

    type Runner = Callable[[argparse.Namespace, Settings], DemoResult]


def _path(value: Any, path: str) -> Any:
    for part in path.split("."):
        value = value[part]
    if value is None:
        message = f"Source produced no value for {path}"
        raise FactError(message)
    return value


def _operations(result: DemoResult) -> list[OperationOutcome]:
    return [
        *result["operations"],
        *(op for child in result["children"] for op in _operations(child)),
    ]


def _answered(
    result: DemoResult, name: str, *, successful: bool = True
) -> OperationOutcome:
    for op in _operations(result):
        if (
            op["operation"] == name
            and op["execution_target"] == "aws"
            and op["response_received"]
            and (not successful or op["status"] == "ok")
        ):
            return op
    message = f"No {'successful ' if successful else ''}AWS {name} response"
    raise FactError(message)


def _table(data: Any, fields: tuple[str, ...]) -> list[list[object]]:
    return [
        [
            ", ".join(value) if isinstance(value, list) else value
            for value in (row[field] for field in fields)
        ]
        for row in data
    ]


def extract_fact(
    lane: str, key: str, definition: FactDefinition, result: DemoResult
) -> object:
    """Extract only observed values using the documented derivations."""
    if result["status"] == "error" or result["mode"] == "not_run":
        message = _reason(result)
        raise FactError(message)
    if lane == "live":
        return _live_value(key, result)
    if definition.how.startswith("data.") and " " not in definition.how:
        return _path(result, definition.how)
    data = cast("Any", result["data"])
    if key.startswith("lineage."):
        status = key.removeprefix("lineage.").removesuffix("_count")
        return (
            len(data)
            if status == "record"
            else sum(row["normalized_status"] == status for row in data)
        )
    if key.startswith("packages."):
        return _path(data, key)
    if key == "demo_count":
        return len(data)
    aliases = {
        "decision.matrix": "responsibility_matrix",
        "policy.denied_reason": "denied_reason",
        "retrieval.injection_retrieved": "injection_retrieved",
        "retrieval.injection_obeyed": "injection_obeyed",
        "retrieval.answerable_citations": "answerable_citations",
        "cost.snapshot_read_at": "read_at",
    }
    if key == "decision.files_read":
        return len(_path(data, "files_read"))
    if key == "mcp.tools":
        return ", ".join(_path(data, "tool_names"))
    if key == "observability.span_names":
        return ", ".join(_path(data, "span_names"))
    if key == "cost.tiers":
        return _table(
            _path(data, "tiers"),
            ("tier", "routing", "usd_per_successful_run"),
        )
    if key == "localstack.advertised":
        return _table(
            _path(data, "advertised"), ("service", "plan", "operations")
        )
    path = aliases.get(key, key.split(".", maxsplit=1)[-1])
    if key == "runtime.ping_status":
        path = key
    return _path(data, path)


def _live_value(key: str, result: DemoResult) -> object:
    data = result["data"]
    if key.startswith("catalog."):
        field = key.removeprefix("catalog.")
        _answered(
            result,
            "ListInferenceProfiles"
            if field == "global_profiles"
            else "ListFoundationModels",
        )
        return _path(data, field)
    if key.startswith("converse."):
        op = _answered(result, "Converse")
        if key == "converse.model":
            return _path(data, "model")
        if key == "converse.estimated_usd":
            return f"{_path(result, 'evidence.estimated_cost_usd'):.6f}"
        field = (
            "input_tokens" if key.endswith("input_tokens") else "output_tokens"
        )
        return _path(op, f"usage.{field}")
    if key.startswith("identity."):
        field = key.removeprefix("identity.").removeprefix("denied_")
        _answered(
            result,
            "GetCallerIdentity"
            if field in {"authenticated", "principal_type"}
            else "ListFoundationModels",
            successful=False,
        )
        return _path(
            data,
            {"probe": "bedrock_probe", "code": "bedrock_probe_code"}.get(
                field, field
            ),
        )
    if key.startswith("guardrail."):
        _answered(result, "ApplyGuardrail")
        return _path(data, key)
    if key == "kendra.list_indices_code":
        _answered(result, "ListIndices", successful=False)
        return _path(data, "list_indices_code")
    _answered(result, "ListAgents")
    return _path(data, "agents.classic_listed")


def _reason(result: DemoResult) -> str:
    if result["error"] is not None:
        return str(result["error"]["message"])
    codes = sorted(
        {
            str(op["error_code"])
            for op in _operations(result)
            if op["error_code"] is not None
        }
    )
    return ", ".join(codes) or f"Source status: {result['status']}"


def source_arguments(
    source: str, *, execution: FactLane, allow_create: bool
) -> argparse.Namespace:
    """Parse source commands without granting implicit writes."""
    tokens = shlex.split(source)
    if "--allow-create" in tokens and not allow_create:
        tokens.remove("--allow-create")
    args = build_parser().parse_args(tokens)
    if getattr(args, "execution", "offline") != execution:
        message = f"Source command does not match {execution}: {source}"
        raise FactError(message)
    return args


def _source_metadata(
    result: DemoResult,
    settings: Settings,
    timestamp: str,
) -> dict[str, Any]:
    data = result["data"]
    return {
        "timestamp": timestamp,
        "mode": result["mode"],
        "status": result["status"],
        "requested_execution": result["requested_execution"],
        "model": data.get("model") if isinstance(data, Mapping) else None,
        "profile": settings.aws_profile,
        "region": settings.region,
        "usage": [
            {"operation": op["operation"], "usage": op["usage"]}
            for op in _operations(result)
            if op["usage"]
        ],
        "value_basis": "estimated" if result["demo"] == "cost" else "measured",
        "cost_basis": "estimated",
    }


def _checked_value(
    lane: str,
    key: str,
    definition: FactDefinition,
    source: DemoResult | str,
) -> object:
    if isinstance(source, str):
        raise FactError(source)
    value = extract_fact(lane, key, definition, source)
    if not valid_value(value, definition.type):
        message = "Source value has the wrong type or contains null"
        raise FactError(message)
    return value


@quiet_sdk_logging()
def collect_facts(
    *,
    root: Path,
    output: Path,
    execution: FactLane = "offline",
    settings: Settings | None = None,
    profile: str | None = None,
    region: str | None = None,
    allow_create: bool = False,
    runner: Runner = dispatch,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    """Refresh one lane under the same aggregate execution budget."""
    store = load_facts(output, root=root)
    base = with_cli_overrides(
        settings or Settings(),
        profile=profile,
        region=region,
        allow_create=allow_create,
    )
    timestamp = clock().isoformat()
    metadata: dict[str, Any] = {
        "fingerprint": store.fingerprint,
        "timestamp": timestamp,
        "sources": {},
        "missing": {},
    }
    values: dict[str, Any] = dict.fromkeys(store.contract[execution])
    cache: dict[str, DemoResult | str] = {}
    budget_scope = (
        use_budget_run(
            open_budget_run(
                region=base.region,
                policy=ExecutionPolicy(),
                aggregate=True,
                root=root / ".awsai_runs",
                prices=SnapshotPrices(root / "data/pricing_snapshot.json"),
            )
        )
        if execution == "live"
        else nullcontext()
    )
    guard = (
        network_guard(NetworkPolicy())
        if execution == "offline"
        else nullcontext()
    )
    with budget_scope, guard:
        for key, definition in store.contract[execution].items():
            if definition.source not in cache:
                try:
                    args = source_arguments(
                        definition.source,
                        execution=execution,
                        allow_create=allow_create,
                    )
                    config = with_cli_overrides(
                        base,
                        profile=getattr(args, "profile", None),
                        allow_create=getattr(args, "allow_create", False),
                    )
                    measured = runner(args, config)
                    safe = cast(
                        "DemoResult", json.loads(render(measured, "json"))
                    )
                    cache[definition.source] = safe
                    metadata["sources"][definition.source] = _source_metadata(
                        safe, config, timestamp
                    )
                except Exception as exc:
                    cache[definition.source] = sanitize_exception(exc)
            source_result = cache[definition.source]
            try:
                values[key] = _checked_value(
                    execution, key, definition, source_result
                )
            except (FactError, KeyError, TypeError, ValueError) as exc:
                metadata["missing"][key] = sanitize_exception(exc)
    store.document[execution] = values
    store.document["metadata"][execution] = metadata
    safe_document = cast("dict[str, Any]", redact(store.document))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(safe_document, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return safe_document


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner = dispatch,
    settings_loader: Callable[[], Settings] = load_settings,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Collect explicitly selected lanes; never run during a build."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    parser = CommandParser(description=__doc__)
    parser.add_argument(
        "--execution", choices=("offline", "live"), default="offline"
    )
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--allow-create", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("deck/facts.json"))
    try:
        args = parser.parse_args(argv)
        document = collect_facts(
            root=Path.cwd(),
            output=args.output,
            execution=args.execution,
            settings=settings_loader(),
            profile=args.profile,
            region=args.region,
            allow_create=args.allow_create,
            runner=runner,
        )
        count = len(document["metadata"][args.execution]["missing"])
        out.write(f"Recorded {args.execution} facts; {count} missing.\n")
    except Exception as exc:
        err.write(sanitize_exception(exc) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
