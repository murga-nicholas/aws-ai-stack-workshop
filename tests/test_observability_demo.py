from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext
from strands.telemetry.tracer import get_tracer

import awsai_demo.observability_demo as demo
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Sequence


def test_observability_reports_actual_spans_in_tree_order() -> None:
    result = demo.run_observability_demo()
    assert result["status"] == "ok"
    assert result["mode"] == "local_contract"
    assert result["data"]["span_names"] == [
        "invoke_agent Strands Agents",
        "execute_event_loop_cycle",
        "chat",
        "execute_tool price_pilot",
        "execute_tool PilotSummary",
    ]
    assert result["data"]["span_count"] >= 5
    assert result["data"]["span_tree"][0]["depth"] == 0
    assert all(op["transport"] == "none" for op in result["operations"])
    assert (
        demo.run_observability_demo(execution="emulator")["mode"] == "not_run"
    )


def test_tracer_is_restored_after_agent_failure() -> None:
    tracer = get_tracer()
    before = tracer.tracer, tracer.tracer_provider

    def failing_agent(_: object) -> None:
        message = "agent failed"
        raise ValueError(message)

    with pytest.raises(ValueError, match="agent failed"):
        demo.capture_spans(agent_builder=lambda _: failing_agent)
    assert (tracer.tracer, tracer.tracer_provider) == before


def _context(span_id: int) -> SpanContext:
    return SpanContext(trace_id=1, span_id=span_id, is_remote=False)


def test_span_tree_handles_orphans_and_redacts_names() -> None:
    root = ReadableSpan("root", context=_context(1), start_time=20)
    child = ReadableSpan(
        "child", context=_context(2), parent=_context(1), start_time=30
    )
    orphan = ReadableSpan(
        "arn:aws:iam::123456789012:role/private",
        context=_context(3),
        parent=_context(99),
        start_time=10,
    )
    no_context = ReadableSpan("no context")
    tree = demo.span_tree([child, root, orphan, no_context])
    assert [row["depth"] for row in tree] == [0, 0, 0, 1]
    assert tree[1]["name"] == "arn:aws:iam::<account>:<redacted>"


@pytest.mark.parametrize(
    "endpoint",
    [
        None,
        "https://collector.example",
        "http://localhost",
        "http://localhost:4318/bad",
        "http://name@localhost:4318",
        "http://localhost:4318?x=1",
        "http://localhost:4318#x",
    ],
)
def test_otlp_rejects_unregistered_or_remote_endpoint(
    endpoint: str | None,
) -> None:

    def forbidden(_: Sequence[ReadableSpan], __: str) -> bool:
        pytest.fail("unapproved OTLP endpoint called")

    result = demo.run_observability_demo(
        settings=Settings(trace="otlp", otel_exporter_otlp_endpoint=endpoint),
        exporter=forbidden,
        span_capture=lambda: ((), []),
    )
    assert result["operations"][-1]["mode"] == "not_run"
    assert result["operations"][-1]["error_code"] == "missing_configuration"


@pytest.mark.parametrize("succeeded", [False, True])
def test_otlp_export_reports_actual_result(succeeded: bool) -> None:
    result = demo.run_observability_demo(
        settings=Settings(
            trace="otlp", otel_exporter_otlp_endpoint="http://localhost:4318"
        ),
        exporter=lambda _, __: succeeded,
        span_capture=lambda: ((), []),
        policy_lookup=lambda: None,
    )
    op = result["operations"][-1]
    assert op["transport"] == "loopback"
    assert op["status"] == ("ok" if succeeded else "error")
    assert op["response_received"] is succeeded


def test_otlp_real_loopback_export_strips_payload_attributes() -> None:
    received: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received.append(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            self.send_response(200)
            self.end_headers()

        def log_message(self, message: str, *args: object) -> None:
            del message, args

    spans = [
        ReadableSpan(
            "safe span",
            context=_context(1),
            start_time=1,
            end_time=2,
            attributes={"prompt": "PRIVATE-PROMPT"},
        )
    ]
    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}"
            op = demo._export_operation(
                Settings(otel_exporter_otlp_endpoint=endpoint),
                spans,
                demo.export_spans,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert op["status"] == "ok"
    assert len(received) == 1
    assert b"safe span" in received[0]
    assert b"PRIVATE-PROMPT" not in received[0]
