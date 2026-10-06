"""Capture actual Strands spans with an isolated in-memory exporter.

Lane: operations; lifecycle: agentcore-observability.
Run: uv run awsai-demo observability.
"""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from strands.telemetry.tracer import get_tracer

from awsai_demo import scenario
from awsai_demo.contracts import operation
from awsai_demo.demo_support import (
    build_result,
    contract_only,
    fixture_operation,
    local_operation,
    not_run_operation,
)
from awsai_demo.network import (
    NetworkPolicy,
    active_policy,
    network_guard,
    registered_loopback_endpoint,
)
from awsai_demo.redact import quiet_sdk_logging, redact
from awsai_demo.runtime import Settings
from awsai_demo.strands_agent_demo import PricePilotScriptedModel, build_agent

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from awsai_demo.contracts import DemoResult, Execution, OperationOutcome
    from awsai_demo.policy import ExecutionPolicy

_TRACE_LOCK = RLock()


def capture_spans(
    *,
    agent_builder: Callable[[object], Any] | None = None,
) -> tuple[tuple[ReadableSpan, ...], list[str]]:
    """Collect framework spans and restore its process-wide tracer."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    model = PricePilotScriptedModel()
    # Strands shares one Tracer facade across its span types.
    # Scope its tracer to this run, preserving OTel globals.
    with _TRACE_LOCK:
        tracer = get_tracer()
        previous = tracer.tracer_provider, tracer.tracer
        tracer.tracer_provider = provider
        tracer.tracer = provider.get_tracer("awsai-demo-observability")
        try:
            build: Callable[[object], Any] = (
                build_agent if agent_builder is None else agent_builder
            )
            build(model)(scenario.BRIEF)
            provider.force_flush()
            spans = exporter.get_finished_spans()
        finally:
            tracer.tracer_provider, tracer.tracer = previous
            provider.shutdown()
    return spans, model.calls


def span_tree(spans: Sequence[ReadableSpan]) -> list[dict[str, object]]:
    """Return payload-free names in parent-before-child order."""
    rows: list[dict[str, object]] = []
    ids = {span.context.span_id for span in spans if span.context is not None}

    def visit(parent: int | None, depth: int) -> None:
        for span in sorted(spans, key=lambda item: item.start_time or 0):
            parent_id = span.parent.span_id if span.parent else None
            if parent_id not in ids:
                parent_id = None
            if parent_id == parent:
                rows.append({"name": redact(span.name), "depth": depth})
                if span.context is not None:
                    visit(span.context.span_id, depth + 1)

    visit(None, 0)
    return rows


def export_spans(spans: Sequence[ReadableSpan], endpoint: str) -> bool:
    """Export names and timing to a checked collector."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    safe_spans = [
        ReadableSpan(
            name=str(redact(span.name)),
            context=span.context,
            parent=span.parent,
            start_time=span.start_time,
            end_time=span.end_time,
            kind=span.kind,
        )
        for span in spans
    ]
    exporter = OTLPSpanExporter(
        endpoint=endpoint.rstrip("/") + "/v1/traces",
        timeout=5,
    )
    try:
        return exporter.export(safe_spans) == SpanExportResult.SUCCESS
    finally:
        cast("Callable[[], None]", exporter.shutdown)()


def _export_operation(
    settings: Settings,
    spans: Sequence[ReadableSpan],
    exporter: Callable[[Sequence[ReadableSpan], str], bool],
    *,
    policy_lookup: Callable[[], NetworkPolicy | None] | None = None,
) -> OperationOutcome:
    endpoint = settings.otel_exporter_otlp_endpoint or ""
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("localhost", "127.0.0.1")
        or parsed.port is None
        or parsed.path not in ("", "/")
        or parsed.username is not None
        or parsed.query
        or parsed.fragment
    ):
        return not_run_operation(
            service="opentelemetry",
            operation_name="OTLP export",
            error_code="missing_configuration",
            request_validated=False,
        )
    lookup = active_policy if policy_lookup is None else policy_lookup
    guard = (lookup() or NetworkPolicy()).with_endpoint(
        registered_loopback_endpoint("otlp", parsed.hostname, parsed.port)
    )
    with network_guard(guard):
        succeeded = exporter(spans, endpoint)
    return operation(
        service="opentelemetry",
        operation="OTLP export",
        execution_target="local",
        mode="local_execution",
        effect="export",
        transport="loopback",
        endpoint_url=endpoint,
        status="ok" if succeeded else "error",
        error_code=None if succeeded else "timeout",
        response_received=succeeded,
    )


@quiet_sdk_logging()
def run_observability_demo(
    *,
    execution: Execution = "offline",
    settings: Settings | None = None,
    policy: ExecutionPolicy | None = None,
    exporter: Callable[[Sequence[ReadableSpan], str], bool] = export_spans,
    span_capture: Callable[[], tuple[tuple[ReadableSpan, ...], list[str]]]
    | None = None,
    policy_lookup: Callable[[], NetworkPolicy | None] | None = None,
) -> DemoResult:
    """Observe an agent and optionally export sanitized spans."""
    del policy
    settings = settings or Settings()
    data: dict[str, object] = {}
    if execution == "offline":
        capture = capture_spans if span_capture is None else span_capture
        spans, calls = capture()
        tree = span_tree(spans)
        data = {
            "span_tree": tree,
            "span_names": list(
                dict.fromkeys(str(row["name"]) for row in tree)
            ),
            "span_count": len(spans),
        }
        operations = [
            local_operation(
                service="strands",
                operation_name="Agent with in-memory spans",
            ),
            *(
                fixture_operation(
                    service="strands",
                    operation_name=f"ScriptedModel.stream:{call}",
                    fixture_id="strands-script-price-pilot-v1",
                    effect="none",
                    request_validated=False,
                )
                for call in calls
            ),
        ]
        if settings.trace == "otlp":
            operations.append(
                _export_operation(
                    settings,
                    spans,
                    exporter,
                    policy_lookup=policy_lookup,
                )
            )
    else:
        operations = [
            contract_only(
                service="opentelemetry",
                operation_name="Strands span capture",
            )
        ]
    return build_result(
        demo="observability",
        technology="Agent observability",
        lane="operations",
        lifecycle_refs=("agentcore-observability",),
        execution=execution,
        settings=settings,
        operations=operations,
        headline="Actual Strands span names and parent relationships.",
        data=data,
    )
