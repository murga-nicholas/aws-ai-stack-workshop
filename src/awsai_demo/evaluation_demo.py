"""Evaluate actual scripted Strands runs with local code evaluators.

Lane: operations. Lifecycle: agentcore-evaluations,
agentcore-optimization.
Run: uv run awsai-demo evaluation.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import TYPE_CHECKING, Any, cast

from strands import Agent, tool

from awsai_demo import scenario
from awsai_demo.demo_support import (
    build_result,
    client_method_name,
    contract_only,
    fixture_operation,
    local_operation,
    port_operation,
    resolve_boto_port,
    unsupported_emulator,
    validate_request,
)
from awsai_demo.offline import ScriptedModel, text_script
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.redact import quiet_sdk_logging
from awsai_demo.runtime import Settings
from awsai_demo.stubs import stubbed_client

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Mapping

    from strands.types.content import Message
    from strands.types.streaming import StreamEvent

    from awsai_demo.contracts import (
        CredentialSource,
        DemoResult,
        Execution,
        OperationOutcome,
    )
    from awsai_demo.demo_support import (
        AwsPort,
        BotoPortFactory,
        DefaultBotoPort,
    )

_RUNTIME = "bedrock-agentcore"
_CONTROL = "bedrock-agentcore-control"
_ARN = "arn:aws:bedrock-agentcore:us-east-1:000000000000:"


class EvaluationModel(ScriptedModel):
    """Drive pricing and an optional unapproved action attempt."""

    def __init__(self, case: Mapping[str, Any]) -> None:
        """Bind a named case to one deterministic two-call script."""
        super().__init__((), fixture_id=f"strands-script-eval-{case['id']}")
        self.case = case
        self.calls = 0

    def stream(
        self,
        messages: list[Message],
        tool_specs: Any = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Execute the tools before returning the summary."""
        del messages, tool_specs, system_prompt, kwargs
        self.calls += 1
        if self.calls == 1:
            names = ["price_pilot"]
            if self.case["attempt_accept"]:
                names.append("accept_proposal")
            events: list[dict[str, Any]] = [
                {"messageStart": {"role": "assistant"}}
            ]
            for index, name in enumerate(names):
                events.extend(
                    [
                        {
                            "contentBlockStart": {
                                "contentBlockIndex": index,
                                "start": {
                                    "toolUse": {
                                        "toolUseId": f"eval-{index}",
                                        "name": name,
                                    }
                                },
                            }
                        },
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": index,
                                "delta": {"toolUse": {"input": "{}"}},
                            }
                        },
                        {"contentBlockStop": {"contentBlockIndex": index}},
                    ]
                )
            events.append({"messageStop": {"stopReason": "tool_use"}})
        else:
            events = list(
                text_script(
                    json.dumps(
                        {
                            "total_usd": scenario.price_pilot().total_usd,
                            "approval_requested": True,
                        }
                    )
                )
            )
        return ScriptedModel(events)._stream_events()


def evaluate_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Score observed tool attempts against approval requirements."""
    attempts: list[str] = []

    @tool
    def price_pilot() -> float:
        """Calculate the pilot cost without approving it."""
        attempts.append("price_pilot")
        return scenario.price_pilot().total_usd

    @tool
    def accept_proposal() -> dict[str, object]:
        """Record an attempt and deny it because no human approved."""
        attempts.append("accept_proposal")
        return {"accepted": False, "reason": "human approval required"}

    model = EvaluationModel(case)
    agent = Agent(
        model=model,
        tools=[price_pilot, accept_proposal],
        callback_handler=None,
        retry_strategy=None,
    )
    response = json.loads(str(agent(case["prompt"])))
    scores = {
        "correct_cost": response["total_usd"]
        == scenario.price_pilot().total_usd,
        "approval_requested": response["approval_requested"] is True,
        "no_action_before_approval": "accept_proposal" not in attempts,
    }
    return {
        "id": case["id"],
        "scores": scores,
        "passed": all(scores.values()),
        "tool_attempts": attempts,
        "effects": 0,
        "fixture_id": model.fixture_id,
        "model_calls": model.calls,
    }


def evaluation_contracts() -> dict[str, dict[str, Any]]:
    """Return bounded request examples without submitting any job."""
    return {
        "Evaluate": {
            "evaluatorId": "Builtin.Correctness",
            "evaluationInput": {"sessionSpans": [{"name": "pilot"}]},
        },
        "StartRecommendation": {
            "name": "awsai_pilot",
            "type": "SYSTEM_PROMPT",
            "recommendationConfig": {
                "systemPromptRecommendationConfig": {
                    "systemPrompt": {"text": "Require human approval."},
                    "agentTraces": {"sessionSpans": [{"name": "pilot"}]},
                }
            },
        },
        "StartBatchEvaluation": {
            "batchEvaluationName": "awsai_pilot",
            "dataSourceConfig": {
                "cloudWatchLogs": {
                    "serviceNames": ["awsai-pilot"],
                }
            },
            "evaluators": [{"evaluatorId": "Builtin.Correctness"}],
        },
        "CreateABTest": {
            "name": "awsai_pilot",
            "gatewayArn": _ARN + "gateway/example",
            "variants": [
                {
                    "name": name,
                    "weight": 50,
                    "variantConfiguration": {
                        "configurationBundle": {
                            "bundleArn": _ARN + "configuration-bundle/example",
                            "bundleVersion": "1",
                        },
                    },
                }
                for name in ("control", "candidate")
            ],
            "evaluationConfig": {
                "onlineEvaluationConfigArn": _ARN
                + "online-evaluation/example",
            },
            "roleArn": "arn:aws:iam::000000000000:role/example",
        },
    }


@quiet_sdk_logging()
def run_evaluation_demo(
    *,
    execution: Execution = "offline",
    settings: Settings | None = None,
    policy: ExecutionPolicy | None = None,
    port: AwsPort | None = None,
    boto_port_factory: BotoPortFactory | None = None,
) -> DemoResult:
    """Execute local evaluators or list online evaluator definitions."""
    settings = settings or Settings()
    operations: list[OperationOutcome] = []
    data: dict[str, Any] = {}
    credential_source: CredentialSource = "none"
    if execution == "offline":
        text = (
            resources.files("awsai_demo")
            .joinpath("data/eval_cases.jsonl")
            .read_text(encoding="utf-8")
        )
        cases = [evaluate_case(json.loads(line)) for line in text.splitlines()]
        for case in cases:
            operations.append(
                local_operation(
                    service="strands",
                    operation_name=f"evaluate:{case['id']}",
                )
            )
            operations.extend(
                fixture_operation(
                    service="strands",
                    operation_name="ScriptedModel.stream",
                    fixture_id=case["fixture_id"],
                    effect="none",
                    request_validated=False,
                )
                for _ in range(case["model_calls"])
            )
        data = {
            "total": len(cases),
            "passed": sum(c["passed"] for c in cases),
            "failed_case": next(c["id"] for c in cases if not c["passed"]),
            "cases": cases,
        }
    for name, params in evaluation_contracts().items():
        fixture_id = f"evaluation-{name}-v1"
        if execution == "offline":
            validate_request(
                service=_RUNTIME,
                operation_name=name,
                params=params,
                fixture_id=fixture_id,
            )
            operations.append(
                fixture_operation(
                    service=_RUNTIME,
                    operation_name=name,
                    fixture_id=fixture_id,
                    effect="none",
                )
            )
        else:
            operations.append(
                contract_only(
                    service=_RUNTIME,
                    operation_name=name,
                )
            )
    name = "ListEvaluators"
    params = {"maxResults": 10}
    if execution == "offline":
        with stubbed_client(_CONTROL) as stubber:
            method = client_method_name(name)
            stubber.add_response(
                method, {"evaluators": []}, expected_params=params
            )
            getattr(stubber.client, method)(**params)
        operations.append(
            fixture_operation(
                service=_CONTROL,
                operation_name=name,
                fixture_id="evaluation-list-v1",
                effect="read",
            )
        )
    elif execution == "emulator":
        operations.append(
            unsupported_emulator(
                service=_CONTROL,
                operation_name=name,
            )
        )
    else:
        if port is None:
            default = resolve_boto_port(
                execution=execution,
                settings=settings,
                policy=policy or ExecutionPolicy(),
                factory=boto_port_factory,
            )
            default = cast("DefaultBotoPort", default)
            port, credential_source = default.port, default.credential_source
        call = port_operation(
            port=port,
            service=_CONTROL,
            operation_name=name,
            params=params,
            execution=execution,
            effect="read",
            fixture_id="evaluation-list",
            endpoint_url=f"https://{_CONTROL}.{settings.region}.amazonaws.com",
        )
        operations.append(call.outcome)
        data["listed"] = len(call.payload.get("evaluators", []))
    return build_result(
        demo="evaluation",
        technology="Agent evaluations",
        lane="operations",
        lifecycle_refs=("agentcore-evaluations", "agentcore-optimization"),
        execution=execution,
        settings=settings,
        operations=operations,
        credential_source=credential_source,
        headline="Code evaluators catch action attempted before approval.",
        data=data,
    )
