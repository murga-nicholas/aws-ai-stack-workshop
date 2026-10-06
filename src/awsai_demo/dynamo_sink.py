"""DynamoDB approval sink for LocalStack decision demos.

Lane: local.
Lifecycle ids: localstack-auth-token, localstack-community-image.
Run: uv run awsai-demo localstack --execution emulator.
"""

from __future__ import annotations

from ipaddress import ip_address
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlsplit

from botocore.exceptions import ClientError

from awsai_demo.contracts import Effect as OperationEffect
from awsai_demo.contracts import OperationOutcome, Phase, operation
from awsai_demo.decision_state import (
    Approval,
    Effect,
    effect_for,
)
from awsai_demo.runtime import DEFAULT_LOCALSTACK_ENDPOINT, Settings
from awsai_demo.scenario import action_key

if TYPE_CHECKING:
    from collections.abc import Mapping

    from awsai_demo.scenario import PilotCost

DYNAMO_TABLE_NAME = "awsai-workshop-decisions"
_FIXED_REGION = "us-east-1"


class DynamoSinkError(RuntimeError):
    """Raised when DynamoDB cannot prove idempotent approval state."""


class DynamoClientPort(Protocol):
    """Subset of the DynamoDB client used by `DynamoApprovalSink`."""

    def describe_table(self, **kwargs: Any) -> Mapping[str, Any]:
        """Describe the decision table."""

    def create_table(self, **kwargs: Any) -> Mapping[str, Any]:
        """Create the decision table."""

    def put_item(self, **kwargs: Any) -> Mapping[str, Any]:
        """Put a decision item conditionally."""

    def get_item(self, **kwargs: Any) -> Mapping[str, Any]:
        """Read a decision item consistently."""


class DynamoApprovalSink:
    """Persist approvals with DynamoDB conditional writes."""

    def __init__(
        self,
        client: DynamoClientPort | None = None,
        *,
        settings: Settings | None = None,
        table_name: str = DYNAMO_TABLE_NAME,
        endpoint_url: str | None = None,
    ) -> None:
        """Store the injected client and loopback endpoint evidence."""
        active_settings = settings or Settings()
        endpoint = endpoint_url or active_settings.localstack_endpoint
        self._client = client or build_emulator_dynamo_client(
            endpoint_url=endpoint,
        )
        self._table_name = table_name
        self._endpoint_url = endpoint
        self._operations: list[OperationOutcome] = []

    @property
    def operations(self) -> tuple[OperationOutcome, ...]:
        """Return operation outcomes collected by this sink."""
        return tuple(self._operations)

    def ensure_table(self) -> tuple[OperationOutcome, ...]:
        """Describe the table and create it when missing."""
        start = len(self._operations)
        try:
            self._client.describe_table(TableName=self._table_name)
            self._record("DescribeTable", effect="read", phase="setup")
        except Exception as exc:
            if _error_code(exc) != "ResourceNotFoundException":
                raise
            self._record(
                "DescribeTable",
                effect="read",
                phase="setup",
                error_code="ResourceNotFoundException",
            )
            self._client.create_table(**create_table_request(self._table_name))
            self._record("CreateTable", effect="write", phase="setup")
        return tuple(self._operations[start:])

    def find(self, cost: PilotCost) -> Effect | None:
        """Read the original approval effect without writing."""
        try:
            response = self._client.get_item(
                **get_item_request(cost, self._table_name),
            )
        except Exception as exc:
            if _error_code(exc) != "ResourceNotFoundException":
                raise
            self._record(
                "GetItem",
                effect="read",
                error_code="ResourceNotFoundException",
            )
            return None
        self._record("GetItem", effect="read")
        item = response.get("Item")
        if item is None:
            return None
        if not isinstance(item, dict):
            _raise_malformed_item()
        audit = item.get("audit")
        if not isinstance(audit, dict):
            _raise_malformed_item()
        raw = audit.get("S")
        if not isinstance(raw, str):
            _raise_malformed_item()
        return Effect.model_validate_json(raw)

    def commit(self, cost: PilotCost, approval: Approval) -> Effect:
        """Write the approval or return the existing effect."""
        prepared = effect_for(cost, approval)
        self.ensure_table()
        try:
            self._client.put_item(
                **put_item_request(prepared, self._table_name),
            )
        except Exception as exc:
            if _error_code(exc) != "ConditionalCheckFailedException":
                raise
            self._record(
                "PutItem",
                effect="write",
                error_code="ConditionalCheckFailedException",
            )
        else:
            self._record("PutItem", effect="write")
            return prepared
        existing = self.find(cost)
        if existing is None:
            message = "Conditional conflict did not return an existing effect"
            raise DynamoSinkError(message)
        return existing

    def _record(
        self,
        operation_name: str,
        *,
        effect: OperationEffect,
        phase: Phase = "main",
        error_code: str | None = None,
    ) -> None:
        self._operations.append(
            operation(
                service="dynamodb",
                operation=operation_name,
                phase=phase,
                execution_target="emulator",
                mode="local_emulator",
                effect=effect,
                transport="loopback",
                endpoint_url=self._endpoint_url,
                error_code=error_code,
            )
        )


def create_table_request(
    table_name: str = DYNAMO_TABLE_NAME,
) -> dict[str, Any]:
    """Return the DynamoDB table request for approval effects."""
    return {
        "TableName": table_name,
        "KeySchema": [{"AttributeName": "action_key", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "action_key", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
    }


def get_item_request(
    cost: PilotCost,
    table_name: str = DYNAMO_TABLE_NAME,
) -> dict[str, Any]:
    """Return a consistent DynamoDB GetItem request for a proposal."""
    return {
        "TableName": table_name,
        "Key": {"action_key": {"S": action_key(cost)}},
        "ConsistentRead": True,
    }


def put_item_request(
    effect: Effect,
    table_name: str = DYNAMO_TABLE_NAME,
) -> dict[str, Any]:
    """Return a conditional DynamoDB PutItem request for an effect."""
    return {
        "TableName": table_name,
        "Item": {
            "action_key": {"S": effect.action_key},
            "proposal_version": {"S": effect.proposal_version},
            "audit": {"S": effect.model_dump_json()},
        },
        "ConditionExpression": "attribute_not_exists(action_key)",
    }


def build_emulator_dynamo_client(
    *,
    endpoint_url: str = DEFAULT_LOCALSTACK_ENDPOINT,
    region_name: str = _FIXED_REGION,
) -> DynamoClientPort:
    """Build a dummy-credential DynamoDB client for LocalStack."""
    del region_name
    _require_loopback_endpoint(endpoint_url)
    import boto3

    session = boto3.Session(
        aws_access_key_id="test",
        aws_secret_access_key="".join(("te", "st")),
        aws_session_token="".join(("te", "st")),
        region_name=_FIXED_REGION,
    )
    return cast(
        "DynamoClientPort",
        session.client(
            "dynamodb",
            endpoint_url=endpoint_url,
            region_name=_FIXED_REGION,
        ),
    )


def _error_code(error: BaseException) -> str:
    if isinstance(error, ClientError):
        return str(error.response.get("Error", {}).get("Code", ""))
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return ""
    payload = response.get("Error")
    if not isinstance(payload, dict):
        return ""
    code = payload.get("Code")
    return code if isinstance(code, str) else ""


def _raise_malformed_item() -> None:
    message = "Stored DynamoDB approval item is malformed"
    raise DynamoSinkError(message)


def _require_loopback_endpoint(endpoint_url: str) -> None:
    parsed = urlsplit(endpoint_url)
    host = parsed.hostname
    if host is None or not _is_loopback_host(host):
        message = "Dynamo approval sink endpoint must be loopback"
        raise DynamoSinkError(message)


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False
