"""Count complete requests and reserve each bounded Bedrock dispatch."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from awsai_demo.demo_support import (
    AwsPort,
    AwsStreamPort,
    billable_not_priced,
    contract_only,
    not_run_operation,
    port_operation,
    stream_port_operation,
)
from awsai_demo.policy import PolicyError
from awsai_demo.strands_multiagent_demo import count_tokens_model_id

if TYPE_CHECKING:
    from awsai_demo.billing import BudgetRun
    from awsai_demo.contracts import OperationOutcome, Usage


@dataclass(frozen=True)
class ObservedCount:
    """Exact service count for this one complete serialized request."""

    value: int | None

    def count_tokens(self, model_id: str, payload: bytes) -> int | None:
        """Return the immediately preceding CountTokens outcome."""
        del model_id, payload
        return self.value


def _whole_token(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _usage_mapping(payload: Mapping[str, object]) -> Usage | None:
    usage: Usage = {}
    input_tokens = _whole_token(
        payload.get("input_tokens", payload.get("inputTokens"))
    )
    output_tokens = _whole_token(
        payload.get("output_tokens", payload.get("outputTokens"))
    )
    total_tokens = _whole_token(
        payload.get("total_tokens", payload.get("totalTokens"))
    )
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    return usage or None


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return None


def _body_usage(body: object) -> Usage | None:
    if isinstance(body, bytes):
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return None
    elif isinstance(body, str):
        text = body
    else:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    document = _as_mapping(parsed)
    if document is None:
        return None
    return _nested_usage(document)


def _nested_usage(document: Mapping[str, object]) -> Usage | None:
    nested = _as_mapping(document.get("usage"))
    if nested is None:
        return None
    return _usage_mapping(nested)


def _stream_usage(events: object) -> Usage | None:
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        return None
    for event in events:
        metadata = _as_mapping(event)
        if metadata is None:
            continue
        meta = _as_mapping(metadata.get("metadata"))
        if meta is None:
            continue
        mapped = _nested_usage(meta)
        if mapped is not None:
            return mapped
    return None


def _payload_usage(name: str, payload: object) -> Usage | None:
    document = _as_mapping(payload)
    if document is None:
        return None
    if name == "ConverseStream":
        streamed = _stream_usage(document.get("stream"))
        if streamed is not None:
            return streamed
    direct = _nested_usage(document)
    if direct is not None:
        return direct
    if name == "ConverseStream":
        return None
    return _body_usage(document.get("body"))


def run_priced_runtime(
    run: BudgetRun,
    model_id: str,
    port: AwsPort | AwsStreamPort,
) -> tuple[list[OperationOutcome], dict[str, Any]]:
    """Execute four inference rows with separate reservations."""
    from awsai_demo.bedrock_runtime_demo import (
        converse_stream_params,
        converse_text_params,
        converse_tool_params,
        invoke_model_params,
    )

    budget = run.command("bedrock-runtime")
    operations: list[OperationOutcome] = []
    data: dict[str, Any] = {
        "model": model_id,
        "pricing": "snapshot",
        "run_id": run.run_id,
    }
    rows = (
        ("Converse", converse_text_params(model_id)),
        ("Converse", converse_tool_params(model_id)),
        ("InvokeModel", invoke_model_params(model_id)),
        ("ConverseStream", converse_stream_params(model_id)),
    )
    endpoint = f"https://bedrock-runtime.{run.region}.amazonaws.com"
    routing = (
        "cross-region-global"
        if model_id.startswith("global.")
        else (
            "cross-region"
            if model_id.startswith(("us.", "eu.", "apac."))
            else "in-region"
        )
    )
    for index, (name, params) in enumerate(rows):
        if name == "InvokeModel":
            if not any(
                family in model_id
                for family in ("amazon.nova", "anthropic.claude")
            ):
                operations.append(
                    not_run_operation(
                        service="bedrock-runtime",
                        operation_name=name,
                        error_code="model_unavailable",
                        request_validated=False,
                    )
                )
                continue
            body = json.loads(params["body"])
            if "anthropic_version" in body:
                body["max_tokens"] = run.policy.max_output_tokens
            else:
                body["inferenceConfig"]["maxTokens"] = (
                    run.policy.max_output_tokens
                )
            params["body"] = json.dumps(body).encode()
            count_input = {"invokeModel": {"body": params["body"]}}
            serialized = params["body"]
        else:
            params["inferenceConfig"]["maxTokens"] = (
                run.policy.max_output_tokens
            )
            content = {
                key: value
                for key, value in params.items()
                if key
                in {
                    "messages",
                    "system",
                    "toolConfig",
                    "additionalModelRequestFields",
                }
            }
            count_input = {"converse": content}
            serialized = json.dumps(params).encode()
        count = port_operation(
            port=cast("AwsPort", port),
            service="bedrock-runtime",
            operation_name="CountTokens",
            params={
                "modelId": count_tokens_model_id(model_id),
                "input": count_input,
            },
            execution="live",
            effect="read",
            phase="setup",
            fixture_id="priced-count",
            endpoint_url=endpoint,
        )
        operations.append(count.outcome)
        try:
            raw_count = count.payload.get("inputTokens")
            receipt, counted = budget.reserve_model(
                f"inference-{index}",
                model_id=model_id,
                serialized_request=serialized,
                exact_counter=ObservedCount(raw_count),
                routing=routing,
            )
        except PolicyError:
            operations.append(
                billable_not_priced(
                    service="bedrock-runtime", operation_name=name
                )
            )
            continue
        if name == "ConverseStream":
            response = stream_port_operation(
                port=cast("AwsStreamPort", port),
                service="bedrock-runtime",
                operation_name=name,
                params=params,
                execution="live",
                effect="infer",
                fixture_id="priced-stream",
                endpoint_url=endpoint,
                reserved=True,
            )
            texts = []
            for event in response.payload.get("stream", ()):
                if "contentBlockDelta" in event:
                    texts.append(
                        event["contentBlockDelta"]["delta"].get("text", "")
                    )
                elif any(key.endswith("Exception") for key in event):
                    response.outcome["status"] = "error"
                    response.outcome["mode"] = "live_service"
                    response.outcome["error_code"] = "validation_failed"
            data["stream_text"] = "".join(texts)
        else:
            response = port_operation(
                port=cast("AwsPort", port),
                service="bedrock-runtime",
                operation_name=name,
                params=params,
                execution="live",
                effect="infer",
                fixture_id="priced-inference",
                endpoint_url=endpoint,
                reserved=True,
            )
            if index == 0 and response.outcome["status"] == "ok":
                content = (
                    response.payload.get("output", {})
                    .get("message", {})
                    .get("content", [])
                )
                data["text"] = "".join(
                    block.get("text", "") for block in content
                )
                usage = response.payload.get("usage", {})
                data["input_tokens"] = usage.get("inputTokens")
                data["output_tokens"] = usage.get("outputTokens")
        copied = _payload_usage(name, response.payload)
        if copied is not None:
            response.outcome["usage"] = copied
        response.outcome["reserved_usd"] = float(receipt.amount_usd)
        data[f"counted_input_{index}"] = counted.tokens
        operations.append(response.outcome)
    operations.extend(
        [
            contract_only(
                service="bedrock-runtime", operation_name="StartAsyncInvoke"
            ),
            contract_only(
                service="bedrock", operation_name="CreateModelInvocationJob"
            ),
        ]
    )
    return operations, data
