from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from awsai_demo.billing import open_budget_run
from awsai_demo.demo_support import AwsResponse, BotoAwsPort
from awsai_demo.owned_resources import GUARDRAIL, MEMORY, OwnedResources
from awsai_demo.policy import Charge, ExecutionPolicy, ReservationUnavailable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

RUN_ID = "abcdef123456"
ACCOUNT = "123456789012"
GUARD_ID = "abcdefghijkl"
ARN = f"arn:aws:bedrock:us-east-1:{ACCOUNT}:guardrail/{GUARD_ID}"
NAME = f"awsai-{RUN_ID}-guard"


class Port:
    def __init__(self) -> None:
        self.responses: dict[str, Any] = {
            "GetCallerIdentity": {"Account": ACCOUNT},
            "CreateGuardrail": {"guardrailId": GUARD_ID, "guardrailArn": ARN},
            "GetGuardrail": {
                "guardrailId": GUARD_ID,
                "guardrailArn": ARN,
                "name": NAME,
            },
            "ListGuardrails": {"guardrails": [{"guardrailId": GUARD_ID}]},
            "ListTagsForResource": {
                "tags": [{"key": "run-id", "value": RUN_ID}]
            },
            "DeleteGuardrail": {},
        }
        self.calls: list[str] = []
        self.context: OwnedResources | None = None

    def call(
        self, service: str, operation: str, params: Mapping[str, Any]
    ) -> AwsResponse:
        assert isinstance(params, dict)
        self.calls.append(operation)
        if operation == "CreateGuardrail" and self.context is not None:
            assert self.context.store.list_entries()[0].status == "unknown"
        response = self.responses[operation]
        if isinstance(response, list):
            response = response.pop(0)
        if isinstance(response, Exception):
            raise response
        return AwsResponse(
            response, f"https://{service}.us-east-1.amazonaws.com"
        )


def context(path: Path, port: Port) -> OwnedResources:
    prices = Mock()
    prices.rate.return_value = Decimal("0.0001")
    ctx = OwnedResources(
        run=open_budget_run(
            region="us-east-1",
            policy=ExecutionPolicy(),
            root=path,
            prices=prices,
            run_id=RUN_ID,
        ),
        demo="guardrails",
        port=port,
        operations=[],
    )
    port.context = ctx
    return ctx


def parameters() -> dict[str, Any]:
    return {
        "name": NAME,
        "clientRequestToken": RUN_ID * 3,
        "blockedInputMessaging": "Blocked",
        "blockedOutputsMessaging": "Blocked",
    }


def test_cleanup_has_one_sdk_attempt_per_explicit_retry(
    tmp_path: Path,
) -> None:
    ctx = context(tmp_path, Port())
    client = Mock()
    client.meta.endpoint_url = "https://bedrock.us-east-1.amazonaws.com"
    client.delete_guardrail.return_value = {}
    session = Mock(client=Mock(return_value=client))
    ctx.port = BotoAwsPort(session, "us-east-1")
    response = ctx.call(
        "bedrock",
        "DeleteGuardrail",
        {"guardrailIdentifier": GUARD_ID},
        effect="delete",
        phase="teardown",
    )
    assert response.outcome["status"] == "ok"
    assert (
        session.client.call_args.kwargs["config"].retries["total_max_attempts"]
        == 1
    )


def client_error(status: int = 403) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": "AccessDeniedException", "Message": "denied"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "operation",
    )


def test_owned_create_intent_before_dispatch_and_cascade(
    tmp_path: Path,
) -> None:
    port = Port()
    ctx = context(tmp_path, port)
    with pytest.raises(ValueError, match="identity"):
        ctx.create(GUARDRAIL, parameters())
    assert ctx.authenticate()
    parent = ctx.create(GUARDRAIL, parameters())
    assert parent is not None and ctx.verify(parent, GUARDRAIL)
    child = ctx.intent(
        GUARDRAIL,
        {},
        parent=parent,
        operation="CreateGuardrailVersion",
        token=RUN_ID,
    )
    ctx.store.mark_created(child.entry_id, exact_id="1", child_id="1")
    assert not ctx.cleanup()
    assert all(entry.status == "deleted" for entry in ctx.store.list_entries())
    assert not ctx.cleanup()
    assert "DeleteGuardrail" in port.calls
    assert ctx.operations[-1]["reserved_usd"] == 0
    assert ctx.operations[-1]["phase"] == "teardown"


def test_inference_is_reserved_before_dispatch(tmp_path: Path) -> None:
    port = Port()
    ctx = context(tmp_path, port)
    params = {
        "guardrailIdentifier": GUARD_ID,
        "guardrailVersion": "1",
        "source": "INPUT",
        "content": [{"text": {"text": "hello"}}],
    }
    with pytest.raises(ReservationUnavailable):
        ctx.call("bedrock-runtime", "ApplyGuardrail", params, effect="infer")
    assert not port.calls
    port.responses["ApplyGuardrail"] = {"action": "NONE"}
    answer = ctx.call(
        "bedrock-runtime",
        "ApplyGuardrail",
        params,
        effect="infer",
        charges=(
            Charge("guardrails.content_filter", Decimal(1), "text_unit"),
        ),
    )
    assert answer.outcome["reserved_usd"] == 0.0002
    assert ctx.budget.ledger.snapshot().reserved_usd == Decimal("0.0002")


def test_failed_auth_create_and_unknown_reconciliation(tmp_path: Path) -> None:
    port = Port()
    ctx = context(tmp_path, port)
    port.responses["GetCallerIdentity"] = {}
    assert not ctx.authenticate()
    port.responses["GetCallerIdentity"] = {"Account": ACCOUNT}
    assert ctx.authenticate()
    port.responses["CreateGuardrail"] = client_error()
    assert ctx.create(GUARDRAIL, parameters()) is None
    assert not ctx.cleanup()
    assert ctx.store.list_entries()[0].status == "deleted"


def test_unresolved_and_failed_deletes_remain_durable(tmp_path: Path) -> None:
    port = Port()
    ctx = context(tmp_path, port)
    assert ctx.authenticate()
    parent = ctx.create(GUARDRAIL, parameters())
    port.responses["DeleteGuardrail"] = client_error()
    assert ctx.cleanup()
    assert ctx.store.list_entries()[0].status == "delete_failed"
    port.responses["DeleteGuardrail"] = client_error(404)
    assert not ctx.cleanup()
    other = replace(
        parent, operation="Unknown", entry_id="other", status="created"
    )
    ctx.store.record_intent(
        run_id=RUN_ID,
        account_fingerprint=other.account_fingerprint,
        region=other.region,
        service=other.service,
        operation="Unknown",
        intended_name=NAME,
        client_token=RUN_ID,
        tags={},
        naming_scheme="guardrail",
    )
    assert ctx.cleanup()


def test_ownership_rejects_mismatches_and_wrong_tags(tmp_path: Path) -> None:
    port = Port()
    ctx = context(tmp_path, port)
    assert ctx.authenticate()
    parent = ctx.create(GUARDRAIL, parameters())
    for updates in (
        {"name": "other"},
        {"guardrailArn": "broken"},
        {"guardrailId": "different"},
    ):
        original = port.responses["GetGuardrail"]
        port.responses["GetGuardrail"] = {**original, **updates}
        assert not ctx.verify(parent, GUARDRAIL)
        port.responses["GetGuardrail"] = original
    port.responses["ListTagsForResource"] = {"tags": {}}
    assert ctx.cleanup()
    assert ctx.store.list_entries()[0].status == "unresolved"


def test_reconciliation_pagination_limits_and_failures(tmp_path: Path) -> None:
    port = Port()
    ctx = context(tmp_path, port)
    assert ctx.authenticate()
    entry = ctx.intent(GUARDRAIL, parameters())
    port.responses["ListGuardrails"] = client_error()
    assert ctx.reconcile(entry, GUARDRAIL) is None
    port.responses["ListGuardrails"] = {"guardrails": [], "nextToken": "next"}
    assert ctx.reconcile(entry, GUARDRAIL) is None
    port.responses["ListGuardrails"] = {
        "guardrails": [{"guardrailId": GUARD_ID}]
    }
    port.responses["GetGuardrail"] = {}
    assert ctx.reconcile(entry, GUARDRAIL) is None
    assert ctx.cleanup()
    assert ctx.store.list_entries()[0].status == "unresolved"


def test_memory_nested_shapes_and_map_tags(tmp_path: Path) -> None:
    port = Port()
    ctx = context(tmp_path, port)
    assert ctx.authenticate()
    name = f"awsai_{RUN_ID}_mem"
    memory_id = name + "-1234567890"
    arn = f"arn:aws:bedrock-agentcore:us-east-1:{ACCOUNT}:memory/{memory_id}"
    memory = {"id": memory_id, "arn": arn, "name": name}
    port.responses.update(
        {
            "CreateMemory": {"memory": memory},
            "GetMemory": {"memory": memory},
            "ListMemories": {"memories": [{"id": memory_id}]},
            "ListTagsForResource": {"tags": {"run-id": RUN_ID}},
            "DeleteMemory": {},
        }
    )
    entry = ctx.create(
        MEMORY,
        {
            "name": name,
            "eventExpiryDuration": 7,
            "clientToken": RUN_ID * 3,
        },
    )
    assert entry is not None
    assert ctx.reconcile(replace(entry, exact_id=None), MEMORY) is not None
    assert not ctx.cleanup()
