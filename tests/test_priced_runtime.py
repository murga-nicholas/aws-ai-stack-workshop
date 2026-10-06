from __future__ import annotations

import json
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from botocore.exceptions import ClientError

from awsai_demo.bedrock_runtime_demo import run_bedrock_runtime_demo
from awsai_demo.billing import open_budget_run
from awsai_demo.demo_support import AwsResponse
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.priced_runtime import (
    _body_usage,
    _payload_usage,
    _usage_mapping,
)
from awsai_demo.runtime import TESTED_LIVE_BEDROCK_MODEL, Settings

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class Port:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any]]] = []
        self.count: int | None = 40
        self.events: list[dict[str, Any]] = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "priced stream"}}},
        ]
        self.denied = False

    def call(
        self, service: str, operation: str, params: Mapping[str, Any]
    ) -> AwsResponse:
        assert service == "bedrock-runtime"
        self.calls.append((operation, params))
        if operation == "CountTokens":
            payload = {"inputTokens": self.count}
        elif self.denied:
            raise ClientError(
                {
                    "Error": {"Code": "AccessDeniedException"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                operation,
            )
        elif operation == "ConverseStream":
            payload = {"stream": self.events}
        else:
            payload = {
                "output": {
                    "message": {"content": [{"text": "within budget"}]}
                },
                "usage": {"inputTokens": 40, "outputTokens": 8},
            }
        return AwsResponse(
            payload, "https://bedrock-runtime.us-east-1.amazonaws.com"
        )

    stream = call


def run(
    path: Path,
    port: Port,
    *,
    model: str = TESTED_LIVE_BEDROCK_MODEL,
    rate: str = "0.000001",
) -> Any:
    prices = Mock()
    prices.rate.return_value = Decimal(rate)
    policy = ExecutionPolicy()
    budget = open_budget_run(
        region="us-east-1", policy=policy, root=path, prices=prices
    )
    return run_bedrock_runtime_demo(
        execution="live",
        settings=Settings(model=model),
        policy=policy,
        port=port,
        budget_run=budget,
    )


def test_each_complete_request_counted_then_reserved(tmp_path: Path) -> None:
    port = Port()
    report = run(tmp_path, port)
    assert report["mode"] == "live_model"
    assert report["data"]["stream_text"] == "priced stream"
    assert report["data"]["input_tokens"] == 40
    assert report["data"]["output_tokens"] == 8
    for outcome in report["operations"]:
        if outcome["operation"] in {"Converse", "InvokeModel"}:
            assert outcome["status"] == "ok"
            assert outcome["usage"] == {
                "input_tokens": 40,
                "output_tokens": 8,
            }
        if outcome["operation"] == "ConverseStream":
            assert outcome["usage"] is None
    assert report["evidence"]["estimated_cost_usd"] > 0
    assert [name for name, _ in port.calls] == [
        "CountTokens",
        "Converse",
        "CountTokens",
        "Converse",
        "CountTokens",
        "InvokeModel",
        "CountTokens",
        "ConverseStream",
    ]
    assert "toolConfig" in port.calls[2][1]["input"]["converse"]
    assert "body" in port.calls[4][1]["input"]["invokeModel"]
    assert port.calls[0][1]["modelId"] == (
        "anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    assert port.calls[1][1]["modelId"] == TESTED_LIVE_BEDROCK_MODEL
    assert port.calls[1][1]["inferenceConfig"]["maxTokens"] == 512
    counted_body = port.calls[4][1]["input"]["invokeModel"]["body"]
    assert counted_body == port.calls[5][1]["body"]
    assert json.loads(counted_body) == {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Summarize the plan."}],
            }
        ],
    }


def test_missing_count_and_over_budget_never_dispatch(tmp_path: Path) -> None:
    port = Port()
    port.count = None
    report = run(tmp_path, port)
    assert report["status"] == "blocked"
    assert all(name == "CountTokens" for name, _ in port.calls)
    port = Port()
    report = run(tmp_path, port, rate="1")
    assert report["status"] == "blocked"
    assert all(name == "CountTokens" for name, _ in port.calls)


def test_region_routing_provider_specific_skip_and_errors(
    tmp_path: Path,
) -> None:
    for model in ("us.amazon.nova-2-lite-v1:0", "other-provider"):
        port = Port()
        report = run(tmp_path, port, model=model)
        assert report["data"]["model"] == model
    assert "InvokeModel" not in [name for name, _ in port.calls]
    port = Port()
    port.events.append(
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 1}}}
    )
    port.events.append({"validationException": {"message": "fixture"}})
    report = run(tmp_path, port)
    assert report["status"] == "error"
    streamed = next(
        item
        for item in report["operations"]
        if item["operation"] == "ConverseStream"
    )
    assert streamed["usage"] == {"input_tokens": 3, "output_tokens": 1}
    port = Port()
    port.denied = True
    report = run(tmp_path, port)
    assert report["mode"] == "live_service"


def test_usage_helpers_cover_bodies_streams_and_bools() -> None:
    assert _usage_mapping({}) is None
    assert _usage_mapping({"inputTokens": True, "outputTokens": False}) is None
    assert _usage_mapping(
        {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    ) == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    assert _usage_mapping({"inputTokens": 4, "totalTokens": 9}) == {
        "input_tokens": 4,
        "total_tokens": 9,
    }
    assert _body_usage(1) is None
    assert _body_usage(b"\xff") is None
    assert _body_usage(b"not-json") is None
    assert _body_usage("[]") is None
    assert _body_usage('{"usage": 1}') is None
    assert _body_usage('{"note": "none"}') is None
    assert _body_usage('{"usage": {"input_tokens": 7}}') == {"input_tokens": 7}
    assert _payload_usage("Converse", 1) is None
    assert _payload_usage(
        "Converse", {"body": b'{"usage": {"inputTokens": 8}}'}
    ) == {"input_tokens": 8}
    assert _payload_usage(
        "ConverseStream", {"stream": "nope", "usage": {"outputTokens": 2}}
    ) == {"output_tokens": 2}
    assert _payload_usage(
        "ConverseStream",
        {"stream": [1, {"metadata": {}}], "usage": {"output_tokens": 5}},
    ) == {"output_tokens": 5}
    assert _payload_usage(
        "ConverseStream",
        {
            "stream": [
                {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 1}}}
            ]
        },
    ) == {"input_tokens": 3, "output_tokens": 1}
    assert _payload_usage("ConverseStream", {"stream": []}) is None
