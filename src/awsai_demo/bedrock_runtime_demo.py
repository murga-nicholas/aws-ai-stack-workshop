"""Amazon Bedrock Runtime demo.

Technology: Amazon Bedrock Runtime.
Lane: models.
Lifecycle refs: bedrock, converse-api, invoke-model-api.
Run:
    uv run awsai-demo bedrock-runtime
    uv run awsai-demo bedrock-runtime --execution emulator
    uv run awsai-demo bedrock-runtime --execution live
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from awsai_demo.demo_support import (
    AwsPort,
    AwsStreamPort,
    billable_not_priced,
    build_default_boto_port,
    build_result,
    contract_only,
    fixture_operation,
    port_operation,
    require_port_not_run,
    unsupported_emulator,
    validate_request,
)
from awsai_demo.runtime import TESTED_LIVE_BEDROCK_MODEL, Execution, Settings
from awsai_demo.scenario import price_pilot
from awsai_demo.stubs import named_fixture_stream, stubbed_client

if TYPE_CHECKING:
    from collections.abc import Mapping

    from awsai_demo.billing import BudgetRun
    from awsai_demo.contracts import (
        CredentialSource,
        DemoResult,
        OperationOutcome,
    )
    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.policy import ExecutionPolicy

_DEMO = "bedrock-runtime"
_TECHNOLOGY = "Amazon Bedrock Runtime"
_REFS = ["bedrock", "converse-api", "invoke-model-api"]
_TEXT_FIXTURE = "bedrock-runtime-converse-text-v1"
_TOOL_FIXTURE = "bedrock-runtime-converse-tool-v1"
_INVOKE_FIXTURE = "bedrock-runtime-invoke-model-v1"
_STREAM_FIXTURE = "bedrock-runtime-converse-stream-v1"
_COUNT_FIXTURE = "bedrock-runtime-count-tokens-v1"
_ASYNC_FIXTURE = "bedrock-runtime-start-async-v1"
_BATCH_FIXTURE = "bedrock-create-model-invocation-job-v1"


class PilotAnswer(BaseModel):
    """Validated shape for the tool-use demo answer."""

    approved: bool
    amount_usd: int


def run_bedrock_runtime_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | AwsStreamPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
    budget_run: BudgetRun | None = None,
) -> DemoResult:
    """Run the Bedrock Runtime core demo in the requested lane."""
    model_id = _runtime_model_id(settings, execution)
    credential_source: CredentialSource | None = None
    if port is None:
        default_port = build_default_boto_port(
            execution=execution,
            settings=settings,
            policy=policy,
            session_factory=session_factory,
            csv_loader=csv_loader,
        )
        if default_port is not None:
            port = default_port.port
            credential_source = default_port.credential_source
    if execution == "offline":
        operations, data = _run_offline(model_id)
    elif execution == "emulator":
        operations, data = _run_emulator(settings, model_id, port)
    else:
        operations, data = _run_live(
            settings,
            model_id,
            port,
            policy=policy,
            budget_run=budget_run,
        )
    return build_result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="models",
        lifecycle_refs=_REFS,
        execution=execution,
        headline="Bedrock Runtime request contracts exercised.",
        operations=operations,
        settings=settings,
        data=data,
        requested_model=model_id,
        observed_model=model_id
        if any(item["response_received"] for item in operations)
        else None,
        credential_source=credential_source,
    )


def _run_offline(
    model_id: str,
) -> tuple[list[OperationOutcome], dict[str, Any]]:
    operations: list[OperationOutcome] = []
    data: dict[str, Any] = {"model": model_id}
    text_params = converse_text_params(model_id)
    tool_params = converse_tool_params(model_id)
    invoke_params = invoke_model_params(model_id)
    stream_params = converse_stream_params(model_id)
    count_params = count_tokens_params(model_id)
    async_params = start_async_invoke_params(model_id)
    batch_params = model_invocation_job_params(model_id)

    with stubbed_client("bedrock-runtime") as stubber:
        client = stubber.client
        stubber.add_response(
            "count_tokens",
            {"inputTokens": 11},
            expected_params=count_params,
        )
        count_response = client.count_tokens(**count_params)
        operations.append(_fixture_count(count_response))

        stubber.add_response(
            "converse",
            _text_converse_response(),
            expected_params=text_params,
        )
        text_response = _converse_request(
            client,
            model_id=model_id,
            messages=text_params["messages"],
            system=text_params["system"],
            inference_config=text_params["inferenceConfig"],
        )
        operations.append(_fixture_converse(_TEXT_FIXTURE, text_response))
        data["text"] = _text_from_converse(text_response)

        stubber.add_response(
            "converse",
            _tool_converse_response(),
            expected_params=tool_params,
        )
        tool_response = client.converse(**tool_params)
        pilot = price_pilot()
        validated = PilotAnswer.model_validate(
            {"approved": False, "amount_usd": int(pilot.total_usd)},
        )
        operations.append(_fixture_converse(_TOOL_FIXTURE, tool_response))
        data["tool_stop_reason"] = tool_response["stopReason"]
        data["tool_result"] = validated.model_dump()

        stubber.add_response(
            "invoke_model",
            _invoke_model_response(),
            expected_params=invoke_params,
        )
        invoke_response = client.invoke_model(**invoke_params)
        operations.append(_fixture_invoke())
        data["invoke_content_type"] = invoke_response["contentType"]

        validate_request(
            service="bedrock-runtime",
            operation_name="ConverseStream",
            params=stream_params,
            fixture_id=_STREAM_FIXTURE,
        )
        stream_response = named_fixture_stream(
            fixture_id=_STREAM_FIXTURE,
            events=_stream_events(),
        )
        operations.append(_fixture_stream())
        data["stream_event_count"] = len(tuple(stream_response))

        stubber.add_response(
            "start_async_invoke",
            {"invocationArn": _ARN_ASYNC},
            expected_params=async_params,
        )
        client.start_async_invoke(**async_params)
        operations.append(_fixture_async())

    with stubbed_client("bedrock") as stubber:
        client = stubber.client
        stubber.add_response(
            "create_model_invocation_job",
            {"jobArn": _ARN_JOB},
            expected_params=batch_params,
        )
        client.create_model_invocation_job(**batch_params)
        operations.append(_fixture_batch())
    return operations, data


def _run_emulator(
    settings: Settings,
    model_id: str,
    port: AwsPort | AwsStreamPort | None,
) -> tuple[list[OperationOutcome], dict[str, Any]]:
    operations: list[OperationOutcome] = []
    data: dict[str, Any] = {"model": model_id}
    if port is None:
        for service, operation_name in _EMULATOR_CALLS:
            operations.append(
                require_port_not_run(
                    service=service,
                    operation_name=operation_name,
                ),
            )
    else:
        for fixture_id, operation_name, params in (
            (_TEXT_FIXTURE, "Converse", converse_text_params(model_id)),
            (_TOOL_FIXTURE, "Converse", converse_tool_params(model_id)),
            (_INVOKE_FIXTURE, "InvokeModel", invoke_model_params(model_id)),
        ):
            result = port_operation(
                port=cast("AwsPort", port),
                service="bedrock-runtime",
                operation_name=operation_name,
                params=params,
                execution="emulator",
                effect="infer",
                fixture_id=fixture_id,
                endpoint_url=settings.localstack_endpoint,
            )
            operations.append(result.outcome)
        data["emulator_endpoint"] = settings.localstack_endpoint
    operations.extend(
        [
            unsupported_emulator(
                service="bedrock-runtime", operation_name="ConverseStream"
            ),
            unsupported_emulator(
                service="bedrock-runtime",
                operation_name="CountTokens",
                phase="setup",
            ),
            unsupported_emulator(
                service="bedrock-runtime", operation_name="StartAsyncInvoke"
            ),
            unsupported_emulator(
                service="bedrock", operation_name="CreateModelInvocationJob"
            ),
        ]
    )
    return operations, data


def _run_live(
    settings: Settings,
    model_id: str,
    port: AwsPort | AwsStreamPort | None,
    *,
    policy: ExecutionPolicy | None = None,
    budget_run: BudgetRun | None = None,
) -> tuple[list[OperationOutcome], dict[str, Any]]:
    from awsai_demo.billing import active_budget_run
    from awsai_demo.policy import ExecutionPolicy, PolicyError
    from awsai_demo.priced_runtime import run_priced_runtime

    try:
        priced_run = budget_run or active_budget_run(
            region=settings.region,
            policy=policy or ExecutionPolicy(),
        )
    except PolicyError:
        priced_run = None
    if priced_run is not None and port is not None:
        return run_priced_runtime(priced_run, model_id, port)
    operations = [
        billable_not_priced(
            service="bedrock-runtime", operation_name="Converse"
        ),
        billable_not_priced(
            service="bedrock-runtime", operation_name="Converse"
        ),
        billable_not_priced(
            service="bedrock-runtime", operation_name="InvokeModel"
        ),
        billable_not_priced(
            service="bedrock-runtime", operation_name="ConverseStream"
        ),
    ]
    if port is None:
        operations.append(
            require_port_not_run(
                service="bedrock-runtime",
                operation_name="CountTokens",
                phase="setup",
            ),
        )
        data: dict[str, Any] = {"model": model_id, "pricing": "missing"}
    else:
        count = port_operation(
            port=cast("AwsPort", port),
            service="bedrock-runtime",
            operation_name="CountTokens",
            params=count_tokens_params(model_id),
            execution="live",
            effect="read",
            phase="setup",
            fixture_id=_COUNT_FIXTURE,
        )
        operations.append(count.outcome)
        data = {
            "model": model_id,
            "pricing": "missing",
            "input_tokens": count.payload.get("inputTokens"),
        }
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


def converse_text_params(model_id: str) -> dict[str, Any]:
    """Return the text Converse request."""
    return {
        "modelId": model_id,
        "messages": _messages("Estimate the proposal risk in one sentence."),
        "system": _system(),
        "inferenceConfig": _inference_config(),
    }


def converse_tool_params(model_id: str) -> dict[str, Any]:
    """Return the tool-enabled Converse request."""
    params = converse_text_params(model_id)
    params["toolConfig"] = {
        "tools": [
            {
                "toolSpec": {
                    "name": "price_pilot",
                    "description": "Return proposal pricing totals.",
                    "inputSchema": {"json": {"type": "object"}},
                }
            }
        ]
    }
    return params


def converse_stream_params(model_id: str) -> dict[str, Any]:
    """Return the ConverseStream request."""
    return converse_text_params(model_id)


def count_tokens_params(model_id: str) -> dict[str, Any]:
    """Return the CountTokens request."""
    from awsai_demo.strands_multiagent_demo import count_tokens_model_id

    params = converse_text_params(model_id)
    return {
        "modelId": count_tokens_model_id(model_id),
        "input": {
            "converse": {
                "messages": params["messages"],
                "system": params["system"],
            }
        },
    }


def invoke_model_params(model_id: str) -> dict[str, Any]:
    """Return the provider's body, retaining Nova for emulation."""
    if "anthropic.claude" in model_id:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": _inference_config()["maxTokens"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize the plan."}
                    ],
                }
            ],
        }
    else:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "Summarize the plan."}],
                }
            ],
            "inferenceConfig": _inference_config(),
        }
    return {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps(body).encode("utf-8"),
    }


def start_async_invoke_params(model_id: str) -> dict[str, Any]:
    """Return a StartAsyncInvoke request shape."""
    return {
        "clientRequestToken": "bedrock-runtime-demo-token",
        "modelId": model_id,
        "modelInput": {"prompt": "offline fixture"},
        "outputDataConfig": {
            "s3OutputDataConfig": {"s3Uri": "s3://example-output/demo/"}
        },
    }


def model_invocation_job_params(model_id: str) -> dict[str, Any]:
    """Return a CreateModelInvocationJob request shape."""
    return {
        "jobName": "awsai-demo-job",
        "roleArn": _ROLE_ARN,
        "clientRequestToken": "bedrock-batch-demo-token",
        "modelId": model_id,
        "inputDataConfig": {
            "s3InputDataConfig": {"s3Uri": "s3://example-input/demo.jsonl"}
        },
        "outputDataConfig": {
            "s3OutputDataConfig": {"s3Uri": "s3://example-output/demo/"}
        },
    }


def _converse_request(
    client: Any,
    *,
    model_id: str,
    messages: list[dict[str, Any]],
    system: list[dict[str, str]],
    inference_config: dict[str, Any],
) -> Mapping[str, Any]:
    # slide: converse-request
    response = client.converse(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig=inference_config,
    )
    # end-slide: converse-request
    return cast("Mapping[str, Any]", response)


def _fixture_count(response: Mapping[str, Any]) -> OperationOutcome:
    return fixture_operation(
        service="bedrock-runtime",
        operation_name="CountTokens",
        phase="setup",
        fixture_id=_COUNT_FIXTURE,
        effect="read",
        usage={"inputTokens": int(response["inputTokens"])},
    )


def _fixture_converse(
    fixture_id: str,
    response: Mapping[str, Any],
) -> OperationOutcome:
    usage = cast("Mapping[str, int]", response.get("usage", {}))
    return fixture_operation(
        service="bedrock-runtime",
        operation_name="Converse",
        fixture_id=fixture_id,
        effect="infer",
        usage=usage,
    )


def _fixture_invoke() -> OperationOutcome:
    return fixture_operation(
        service="bedrock-runtime",
        operation_name="InvokeModel",
        fixture_id=_INVOKE_FIXTURE,
        effect="infer",
    )


def _fixture_stream() -> OperationOutcome:
    return fixture_operation(
        service="bedrock-runtime",
        operation_name="ConverseStream",
        fixture_id=_STREAM_FIXTURE,
        effect="infer",
    )


def _fixture_async() -> OperationOutcome:
    return fixture_operation(
        service="bedrock-runtime",
        operation_name="StartAsyncInvoke",
        fixture_id=_ASYNC_FIXTURE,
        effect="none",
    )


def _fixture_batch() -> OperationOutcome:
    return fixture_operation(
        service="bedrock",
        operation_name="CreateModelInvocationJob",
        fixture_id=_BATCH_FIXTURE,
        effect="none",
    )


def _runtime_model_id(settings: Settings, execution: Execution) -> str:
    if settings.model is not None:
        return settings.model
    if execution == "emulator":
        return _localstack_model_id(settings)
    return TESTED_LIVE_BEDROCK_MODEL


def _localstack_model_id(settings: Settings) -> str:
    configured = getattr(settings, "localstack_bedrock_model", None)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    default = settings.default_bedrock_model
    if default.startswith("ollama."):
        return default
    return f"ollama.{default}"


def _messages(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"text": text}]}]


def _system() -> list[dict[str, str]]:
    return [{"text": "You are a concise AWS AI proposal reviewer."}]


def _inference_config() -> dict[str, Any]:
    return {"maxTokens": 128, "temperature": 0.0}


def _text_converse_response() -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "The risk is within budget."}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
        "metrics": {"latencyMs": 12},
    }


def _tool_converse_response() -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse-price-pilot",
                            "name": "price_pilot",
                            "input": {},
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 22, "outputTokens": 4, "totalTokens": 26},
        "metrics": {"latencyMs": 10},
    }


def _invoke_model_response() -> dict[str, Any]:
    return {
        "body": json.dumps({"outputText": "InvokeModel fixture"}).encode(),
        "contentType": "application/json",
    }


def _stream_events() -> tuple[Mapping[str, Any], ...]:
    return (
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "streamed"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    )


def _text_from_converse(response: Mapping[str, Any]) -> str:
    output = cast("Mapping[str, Any]", response["output"])
    message = cast("Mapping[str, Any]", output["message"])
    content = cast("list[Mapping[str, str]]", message["content"])
    return content[0]["text"]


_EMULATOR_CALLS = (
    ("bedrock-runtime", "Converse"),
    ("bedrock-runtime", "Converse"),
    ("bedrock-runtime", "InvokeModel"),
)
_ROLE_ARN = "arn:aws:iam::123456789012:role/awsai-demo-harness"
_ARN_ASYNC = "arn:aws:bedrock:us-east-1:123456789012:async-invoke/demo"
_ARN_JOB = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/demo"
