from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError

from awsai_demo.decision_state import Approval, Effect
from awsai_demo.demo_support import validate_request
from awsai_demo.dynamo_sink import (
    DYNAMO_TABLE_NAME,
    DynamoApprovalSink,
    DynamoSinkError,
    _error_code,
    build_emulator_dynamo_client,
    create_table_request,
    get_item_request,
    put_item_request,
)
from awsai_demo.runtime import Settings
from awsai_demo.scenario import (
    PilotCost,
    action_key,
    price_pilot,
    proposal_version,
)


class FakeDynamoError(Exception):
    def __init__(self, code: object) -> None:
        super().__init__(str(code))
        self.response = {"Error": {"Code": code}}


class FakeDynamo:
    def __init__(
        self,
        *,
        table_exists: bool = False,
        describe_error: object | None = None,
        put_error: object | None = None,
        conflict_without_item: bool = False,
    ) -> None:
        self.table_exists = table_exists
        self.describe_error = describe_error
        self.put_error = put_error
        self.conflict_without_item = conflict_without_item
        self.items: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def describe_table(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(("DescribeTable", kwargs))
        if self.describe_error is not None:
            raise FakeDynamoError(self.describe_error)
        if not self.table_exists:
            code = "ResourceNotFoundException"
            raise FakeDynamoError(code)
        return {"Table": {"TableName": kwargs["TableName"]}}

    def create_table(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(("CreateTable", kwargs))
        self.table_exists = True
        return {"TableDescription": {"TableName": kwargs["TableName"]}}

    def put_item(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(("PutItem", kwargs))
        if self.put_error is not None:
            raise FakeDynamoError(self.put_error)
        key = kwargs["Item"]["action_key"]["S"]
        if key in self.items or self.conflict_without_item:
            code = "ConditionalCheckFailedException"
            raise FakeDynamoError(code)
        self.items[key] = kwargs["Item"]
        return {"ConsumedCapacity": {"TableName": kwargs["TableName"]}}

    def get_item(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(("GetItem", kwargs))
        key = kwargs["Key"]["action_key"]["S"]
        if key not in self.items:
            return {}
        return {"Item": self.items[key]}


def approval_for(cost: PilotCost, approver: str = "Mira") -> Approval:
    return Approval(
        approved=True,
        proposal_version=proposal_version(cost),
        approver=approver,
        source="simulated",
    )


def test_constructor_builds_dummy_client_without_dispatch() -> None:
    sink = DynamoApprovalSink(settings=Settings())
    assert sink.operations == ()

    client = build_emulator_dynamo_client(
        endpoint_url="http://127.0.0.1:4566",
        region_name="eu-west-1",
    )
    assert client.meta.endpoint_url == "http://127.0.0.1:4566"
    assert client.meta.region_name == "us-east-1"

    with pytest.raises(DynamoSinkError):
        build_emulator_dynamo_client(endpoint_url="http://example.com:4566")
    with pytest.raises(DynamoSinkError):
        build_emulator_dynamo_client(endpoint_url="localstack")


def test_commit_creates_table_and_returns_original_on_conflict() -> None:
    cost = price_pilot()
    client = FakeDynamo()
    sink = DynamoApprovalSink(client=client)

    first = sink.commit(cost, approval_for(cost, "Mira"))
    second = sink.commit(cost, approval_for(cost, "Kostya"))

    assert second == first
    assert client.calls[0][0] == "DescribeTable"
    assert [name for name, _kwargs in client.calls] == [
        "DescribeTable",
        "CreateTable",
        "PutItem",
        "DescribeTable",
        "PutItem",
        "GetItem",
    ]
    assert [item["operation"] for item in sink.operations] == [
        "DescribeTable",
        "CreateTable",
        "PutItem",
        "DescribeTable",
        "PutItem",
        "GetItem",
    ]
    assert sink.operations[0]["error_code"] == "ResourceNotFoundException"
    duplicate_put = sink.operations[4]
    assert duplicate_put["status"] == "ok"
    assert duplicate_put["error_code"] == "ConditionalCheckFailedException"


def test_ensure_table_existing_and_non_missing_errors() -> None:
    existing = FakeDynamo(table_exists=True)
    sink = DynamoApprovalSink(client=existing)
    operations = sink.ensure_table()
    assert [item["operation"] for item in operations] == ["DescribeTable"]
    assert [name for name, _kwargs in existing.calls] == ["DescribeTable"]

    failing = DynamoApprovalSink(
        client=FakeDynamo(describe_error="AccessDeniedException"),
    )
    with pytest.raises(FakeDynamoError):
        failing.ensure_table()


def test_find_returns_none_for_absent_effects() -> None:
    sink = DynamoApprovalSink(client=FakeDynamo(table_exists=True))
    assert sink.find(price_pilot()) is None


def test_find_returns_none_when_table_is_absent() -> None:
    class MissingTableReader(FakeDynamo):
        def get_item(self, **kwargs: Any) -> dict[str, object]:
            self.calls.append(("GetItem", kwargs))
            code = "ResourceNotFoundException"
            raise FakeDynamoError(code)

    sink = DynamoApprovalSink(client=MissingTableReader())
    assert sink.find(price_pilot()) is None
    assert sink.operations[0]["error_code"] == "ResourceNotFoundException"


def test_find_reraises_unexpected_get_item_errors() -> None:
    class BrokenReader(FakeDynamo):
        def get_item(self, **kwargs: Any) -> dict[str, object]:
            self.calls.append(("GetItem", kwargs))
            code = "AccessDeniedException"
            raise FakeDynamoError(code)

    sink = DynamoApprovalSink(client=BrokenReader())
    with pytest.raises(FakeDynamoError):
        sink.find(price_pilot())


@pytest.mark.parametrize(
    "response",
    [
        {"Item": "not-a-dict"},
        {"Item": {"audit": "not-a-dict"}},
        {"Item": {"audit": {"S": 7}}},
    ],
)
def test_find_rejects_malformed_items(response: dict[str, object]) -> None:
    class Reader(FakeDynamo):
        def get_item(self, **kwargs: Any) -> dict[str, object]:
            self.calls.append(("GetItem", kwargs))
            return response

    sink = DynamoApprovalSink(client=Reader(table_exists=True))
    with pytest.raises(DynamoSinkError):
        sink.find(price_pilot())


def test_commit_errors_when_conflict_cannot_be_reconciled() -> None:
    cost = price_pilot()
    sink = DynamoApprovalSink(
        client=FakeDynamo(
            table_exists=True,
            conflict_without_item=True,
        ),
    )

    with pytest.raises(DynamoSinkError):
        sink.commit(cost, approval_for(cost))


def test_commit_reraises_unexpected_put_error() -> None:
    cost = price_pilot()
    sink = DynamoApprovalSink(
        client=FakeDynamo(
            table_exists=True,
            put_error="ProvisionedThroughputExceededException",
        ),
    )

    with pytest.raises(FakeDynamoError):
        sink.commit(cost, approval_for(cost))


def test_request_builders_match_dynamodb_shapes() -> None:
    cost = price_pilot()
    effect = Effect(
        action_key=action_key(cost),
        proposal_version=proposal_version(cost),
        approval_id="approval",
        approver="Mira",
        approval_source="simulated",
        approved_at="2026-01-01T00:00:00+00:00",
    )

    validate_request(
        service="dynamodb",
        operation_name="CreateTable",
        params=create_table_request(),
        fixture_id="dynamo-sink-test",
    )
    validate_request(
        service="dynamodb",
        operation_name="GetItem",
        params=get_item_request(cost),
        fixture_id="dynamo-sink-test",
    )
    validate_request(
        service="dynamodb",
        operation_name="PutItem",
        params=put_item_request(effect),
        fixture_id="dynamo-sink-test",
    )
    assert put_item_request(effect)["TableName"] == DYNAMO_TABLE_NAME


def test_error_code_extraction_variants() -> None:
    client_error = ClientError(
        {
            "Error": {"Code": "ConditionalCheckFailedException"},
            "ResponseMetadata": {"HTTPStatusCode": 400},
        },
        "PutItem",
    )
    assert _error_code(client_error) == "ConditionalCheckFailedException"
    assert _error_code(RuntimeError("plain")) == ""
    assert _error_code(FakeDynamoError(7)) == ""

    class BadPayloadError(Exception):
        def __init__(self) -> None:
            super().__init__()
            self.response = {"Error": "bad"}

    assert _error_code(BadPayloadError()) == ""
