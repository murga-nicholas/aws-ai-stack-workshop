"""LocalStack core demo.

Lane: local.
Lifecycle ids: localstack-auth-token, localstack-community-image.
Run: uv run awsai-demo localstack --execution offline.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
from importlib import metadata
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urljoin, urlsplit, urlunsplit

from botocore.exceptions import ClientError
from botocore.response import StreamingBody

from awsai_demo.contracts import (
    DemoResult,
    Effect,
    ErrorCode,
    Execution,
    OperationOutcome,
    Phase,
    Status,
    error_result,
    evidence,
    missing_configuration,
    not_run,
    operation,
    result,
)
from awsai_demo.demo_support import bounded_boto_config
from awsai_demo.runtime import DEFAULT_LOCALSTACK_ENDPOINT, Settings
from awsai_demo.scenario import price_pilot
from awsai_demo.stubs import stubbed_client

if TYPE_CHECKING:
    from awsai_demo.demo_support import BotoSessionPort
    from awsai_demo.policy import ExecutionPolicy

_DEMO = "localstack"
_TECHNOLOGY = "LocalStack core"
_REFS = ["localstack-auth-token", "localstack-community-image"]
_FIXTURE_ID = "localstack-core-stubber-v1"
_BUCKET = "awsai-workshop-localstack-demo"
_OBJECT_KEY = "notes/price_pilot.txt"
_TABLE = "awsai-workshop-decisions"
_ACTION_KEY = "accept-proposal:localstack-demo"
_FUNCTION = "price_pilot"
_ROLE_ARN = "arn:aws:iam::000000000000:role/lambda"
_FIXED_REGION = "us-east-1"
_HTTP_TIMEOUT_SECONDS = 2.0
_EMULATOR_STEPS: tuple[tuple[str, str, Effect, Phase], ...] = (
    ("localstack", "Health", "read", "setup"),
    ("localstack", "Info", "read", "setup"),
    ("s3", "CreateBucket", "write", "main"),
    ("s3", "PutObject", "write", "main"),
    ("dynamodb", "CreateTable", "write", "main"),
    ("dynamodb", "PutItem", "write", "main"),
    ("dynamodb", "PutItem", "write", "main"),
    ("lambda", "CreateFunction", "write", "main"),
    ("lambda", "Invoke", "write", "main"),
)


class LocalStackUnavailableError(RuntimeError):
    """Raised when injected LocalStack loopback is unavailable."""


class LocalStackPort(Protocol):
    """Injected loopback adapter for LocalStack operations."""

    def health(self) -> Mapping[str, Any]:
        """Read `/_localstack/health`."""

    def info(self) -> Mapping[str, Any]:
        """Read `/_localstack/info`."""

    def create_bucket(self, bucket: str) -> Mapping[str, Any]:
        """Create an S3 bucket in the emulator."""

    def put_object(
        self,
        bucket: str,
        key: str,
        body: bytes,
    ) -> Mapping[str, Any]:
        """Put an S3 object in the emulator."""

    def create_table(self, table_name: str) -> Mapping[str, Any]:
        """Create the DynamoDB decision table if needed."""

    def put_approval_once(
        self,
        table_name: str,
        item: Mapping[str, Any],
    ) -> bool:
        """Return `False` when the conditional write conflicts."""

    def create_function(
        self,
        function_name: str,
        zip_bytes: bytes,
    ) -> Mapping[str, Any]:
        """Create the demo Lambda function."""

    def invoke_function(
        self,
        function_name: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Invoke the demo Lambda function."""


class HttpJsonGet(Protocol):
    """Injected HTTP JSON getter used by LocalStack adapter tests."""

    def __call__(
        self,
        url: str,
        timeout: float,
    ) -> Mapping[str, Any]:
        """Return decoded JSON from a loopback URL."""


@dataclass(frozen=True, slots=True)
class StubberRun:
    """Data returned by the offline LocalStack contract replay."""

    operations: tuple[OperationOutcome, ...]
    data: Mapping[str, Any]


def run_localstack_demo(
    *,
    execution: Execution = "offline",
    settings: Settings | None = None,
    policy: ExecutionPolicy | None = None,
    port: LocalStackPort | None = None,
    auth_token: str | None = None,
    http_get: HttpJsonGet | None = None,
    session: BotoSessionPort | None = None,
) -> DemoResult:
    """Run the LocalStack demo in offline or injected-emulator mode."""
    active_settings = settings or Settings()
    if execution == "offline":
        replay = run_strict_localstack_stubber()
        return _demo_result(
            execution=execution,
            settings=active_settings,
            headline=(
                "Strict offline Stubber replay covered S3, "
                "DynamoDB and Lambda."
            ),
            operations=replay.operations,
            data=replay.data,
        )
    if execution == "emulator":
        token = auth_token or active_settings.localstack_auth_token
        if token is None:
            return missing_configuration(
                demo=_DEMO,
                technology=_TECHNOLOGY,
                lane="local",
                lifecycle_refs=_REFS,
                requested_execution=execution,
                headline="LocalStack auth token is not configured.",
                message="Set LOCALSTACK_AUTH_TOKEN before emulator runs.",
                next_steps=[
                    "Start LocalStack, then rerun with emulator mode.",
                ],
            )
        try:
            active_port = port or LocalStackHttpBotoPort(
                active_settings,
                http_get=http_get,
                session=session,
                config=bounded_boto_config(policy) if policy else None,
            )
        except LocalStackUnavailableError as exc:
            return _emulator_unavailable(
                active_settings.localstack_endpoint,
                str(exc),
            )
        return _run_emulator(active_settings, active_port)
    return _live_not_run()


def run_strict_localstack_stubber(
    *,
    region_name: str = "us-east-1",
) -> StubberRun:
    """Exercise real boto3 clients against strict botocore Stubbers."""
    operations: list[OperationOutcome] = []
    data: dict[str, Any] = {
        "strict_stubber": True,
        "real_writes": False,
        "provenance": "botocore.stub.Stubber",
    }

    with stubbed_client("s3", region_name=region_name) as stubber:
        client = stubber.client
        create_params = create_bucket_params()
        stubber.add_response(
            "create_bucket",
            {"Location": f"/{_BUCKET}"},
            expected_params=create_params,
        )
        client.create_bucket(**create_params)
        operations.append(_fixture("s3", "CreateBucket", effect="write"))

        put_params = put_object_params()
        stubber.add_response(
            "put_object",
            {"ETag": '"price-pilot"'},
            expected_params=put_params,
        )
        client.put_object(**put_params)
        operations.append(_fixture("s3", "PutObject", effect="write"))

    with stubbed_client("dynamodb", region_name=region_name) as stubber:
        client = stubber.client
        table_params = create_table_params()
        stubber.add_response(
            "create_table",
            _create_table_response(),
            expected_params=table_params,
        )
        client.create_table(**table_params)
        operations.append(
            _fixture("dynamodb", "CreateTable", effect="write"),
        )

        approval_params = approval_put_item_params()
        stubber.add_response(
            "put_item",
            {"ConsumedCapacity": {"TableName": _TABLE, "CapacityUnits": 1.0}},
            expected_params=approval_params,
        )
        client.put_item(**approval_params)
        operations.append(_fixture("dynamodb", "PutItem", effect="write"))

        stubber.add_client_error(
            "put_item",
            service_error_code="ConditionalCheckFailedException",
            service_message="approval already exists",
            expected_params=approval_params,
            http_status_code=400,
        )
        second_rejected = _conditional_rejected(client, approval_params)
        operations.append(_fixture("dynamodb", "PutItem", effect="write"))
        data["conditional_second_rejected"] = second_rejected

    with stubbed_client("lambda", region_name=region_name) as stubber:
        client = stubber.client
        create_params = create_function_params()
        stubber.add_response(
            "create_function",
            _create_function_response(),
            expected_params=create_params,
        )
        client.create_function(**create_params)
        operations.append(
            _fixture("lambda", "CreateFunction", effect="write"),
        )

        invoke_params = invoke_function_params()
        stubber.add_response(
            "invoke",
            _invoke_response(),
            expected_params=invoke_params,
        )
        response = client.invoke(**invoke_params)
        operations.append(_fixture("lambda", "Invoke", effect="write"))
        data["lambda_payload"] = _read_payload(response["Payload"])

    return StubberRun(tuple(operations), data)


def build_price_pilot_zip() -> bytes:
    """Return a deterministic ZIP containing the Lambda handler."""
    handler_source = (
        "from awsai_demo.scenario import price_pilot\n"
        "\n"
        "def handler(event, context):\n"
        "    cost = price_pilot(\n"
        "        weeks=int(event.get('weeks', 6)),\n"
        "        budget=float(event.get('budget', 25000)),\n"
        "    )\n"
        "    return {\n"
        "        'weeks': cost.weeks,\n"
        "        'staffing_usd': cost.staffing_usd,\n"
        "        'platform_usd': cost.platform_usd,\n"
        "        'total_usd': cost.total_usd,\n"
        "        'headroom_usd': cost.headroom_usd,\n"
        "        'within_budget': cost.within_budget,\n"
        "    }\n"
    )
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        _write_zip_entry(archive, "awsai_demo/__init__.py", "")
        _write_zip_entry(
            archive,
            "awsai_demo/scenario.py",
            _scenario_source_for_lambda(),
        )
        _write_zip_entry(archive, "price_pilot.py", handler_source)
    return buffer.getvalue()


def create_bucket_params(bucket: str = _BUCKET) -> dict[str, Any]:
    """Return the S3 CreateBucket request used by the demo."""
    return {"Bucket": bucket}


def put_object_params(
    *,
    bucket: str = _BUCKET,
    key: str = _OBJECT_KEY,
    body: bytes = b"price_pilot writes approval only after human approval",
) -> dict[str, Any]:
    """Return the S3 PutObject request used by the demo."""
    return {
        "Bucket": bucket,
        "Key": key,
        "Body": body,
    }


def create_table_params(table_name: str = _TABLE) -> dict[str, Any]:
    """Return the DynamoDB CreateTable request used by the demo."""
    return {
        "TableName": table_name,
        "KeySchema": [{"AttributeName": "action_key", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "action_key", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
    }


def approval_item() -> dict[str, Any]:
    """Return the DynamoDB item written by the conditional demo."""
    return {
        "action_key": {"S": _ACTION_KEY},
        "audit": {
            "S": json.dumps(
                {"source": "localstack", "approved": True},
                sort_keys=True,
            ),
        },
    }


def approval_put_item_params(
    table_name: str = _TABLE,
    *,
    item: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the conditional DynamoDB PutItem request."""
    return {
        "TableName": table_name,
        "Item": approval_item() if item is None else dict(item),
        "ConditionExpression": "attribute_not_exists(action_key)",
    }


def create_function_params(
    *,
    function_name: str = _FUNCTION,
    zip_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Return the Lambda CreateFunction request used by the demo."""
    return {
        "FunctionName": function_name,
        "Runtime": "python3.12",
        "Role": _ROLE_ARN,
        "Handler": "price_pilot.handler",
        "Code": {
            "ZipFile": (
                build_price_pilot_zip() if zip_bytes is None else zip_bytes
            ),
        },
    }


def invoke_function_params(
    *,
    function_name: str = _FUNCTION,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the Lambda Invoke request used by the demo."""
    return {
        "FunctionName": function_name,
        "Payload": json.dumps(
            (
                {"weeks": 6, "budget": 25_000}
                if payload is None
                else dict(payload)
            ),
        ).encode(),
    }


class LocalStackHttpBotoPort:
    """Real LocalStack adapter using loopback HTTP and boto clients."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_get: HttpJsonGet | None = None,
        session: BotoSessionPort | None = None,
        config: object | None = None,
    ) -> None:
        """Create an adapter without dispatching network operations."""
        _require_loopback_endpoint(settings.localstack_endpoint)
        self._endpoint = settings.localstack_endpoint.rstrip("/")
        self._http_get = http_get or _http_json_get
        self._session = session or _localstack_session()
        self._config = config

    def health(self) -> Mapping[str, Any]:
        """Read LocalStack health over loopback HTTP."""
        return self._http_get(
            urljoin(self._endpoint + "/", "_localstack/health"),
            _HTTP_TIMEOUT_SECONDS,
        )

    def info(self) -> Mapping[str, Any]:
        """Read LocalStack info over loopback HTTP."""
        return self._http_get(
            urljoin(self._endpoint + "/", "_localstack/info"),
            _HTTP_TIMEOUT_SECONDS,
        )

    def create_bucket(self, bucket: str) -> Mapping[str, Any]:
        """Create an S3 bucket in LocalStack."""
        try:
            return cast(
                "Mapping[str, Any]",
                self._client("s3").create_bucket(
                    **create_bucket_params(bucket)
                ),
            )
        except ClientError as exc:
            if _client_error_code(exc) not in {
                "BucketAlreadyOwnedByYou",
                "BucketAlreadyExists",
            }:
                raise
            return {"already_exists": True}

    def put_object(
        self,
        bucket: str,
        key: str,
        body: bytes,
    ) -> Mapping[str, Any]:
        """Put an S3 object in LocalStack."""
        return cast(
            "Mapping[str, Any]",
            self._client("s3").put_object(
                **put_object_params(bucket=bucket, key=key, body=body),
            ),
        )

    def create_table(self, table_name: str) -> Mapping[str, Any]:
        """Create a DynamoDB table in LocalStack."""
        try:
            return cast(
                "Mapping[str, Any]",
                self._client("dynamodb").create_table(
                    **create_table_params(table_name),
                ),
            )
        except ClientError as exc:
            if _client_error_code(exc) != "ResourceInUseException":
                raise
            return {"already_exists": True}

    def put_approval_once(
        self,
        table_name: str,
        item: Mapping[str, Any],
    ) -> bool:
        """Write one approval, returning `False` on duplicate."""
        try:
            self._client("dynamodb").put_item(
                **approval_put_item_params(table_name, item=item),
            )
        except ClientError as exc:
            if _client_error_code(exc) != "ConditionalCheckFailedException":
                raise
            return False
        return True

    def create_function(
        self,
        function_name: str,
        zip_bytes: bytes,
    ) -> Mapping[str, Any]:
        """Create the price-pilot Lambda in LocalStack."""
        try:
            client = self._client("lambda")
            response = cast(
                "Mapping[str, Any]",
                client.create_function(
                    **create_function_params(
                        function_name=function_name,
                        zip_bytes=zip_bytes,
                    ),
                ),
            )
            _wait_for_lambda_active(client, function_name)
        except ClientError as exc:
            if _client_error_code(exc) != "ResourceConflictException":
                raise
            return {"already_exists": True}
        else:
            return response

    def invoke_function(
        self,
        function_name: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Invoke the price-pilot Lambda in LocalStack."""
        response = self._client("lambda").invoke(
            **invoke_function_params(
                function_name=function_name,
                payload=payload,
            ),
        )
        body = response.get("Payload")
        if isinstance(body, StreamingBody):
            payload_result = _read_payload(body)
            if "FunctionError" in response:
                payload_result["function_error"] = response["FunctionError"]
            return payload_result
        if hasattr(body, "read"):
            raw = body.read()
            if isinstance(raw, bytes):
                payload_result = cast(
                    "dict[str, Any]",
                    json.loads(raw.decode()),
                )
                if "FunctionError" in response:
                    payload_result["function_error"] = response[
                        "FunctionError"
                    ]
                return payload_result
        return cast("Mapping[str, Any]", response)

    def _client(self, service_name: str) -> Any:
        return self._session.client(
            service_name,
            endpoint_url=self._endpoint,
            region_name=_FIXED_REGION,
            **({"config": self._config} if self._config is not None else {}),
        )


def _run_emulator(
    settings: Settings,
    port: LocalStackPort | None,
) -> DemoResult:
    endpoint = settings.localstack_endpoint
    if port is None:
        return not_run(
            demo=_DEMO,
            technology=_TECHNOLOGY,
            lane="local",
            lifecycle_refs=_REFS,
            requested_execution="emulator",
            headline="LocalStack port is not configured.",
            code="missing_configuration",
            message="No LocalStack port was supplied.",
        )
    operations: list[OperationOutcome] = []
    data: dict[str, Any] = {"real_writes": False}
    try:
        health = port.health()
        operations.append(
            _emulator(
                "localstack",
                "Health",
                endpoint,
                effect="read",
                phase="setup",
            )
        )
        info = port.info()
        operations.append(
            _emulator(
                "localstack",
                "Info",
                endpoint,
                effect="read",
                phase="setup",
            )
        )
        data["localstack"] = _localstack_metadata(health, info)
        port.create_bucket(_BUCKET)
        operations.append(
            _emulator("s3", "CreateBucket", endpoint, effect="write")
        )
        port.put_object(_BUCKET, _OBJECT_KEY, put_object_params()["Body"])
        operations.append(
            _emulator("s3", "PutObject", endpoint, effect="write")
        )
        port.create_table(_TABLE)
        operations.append(
            _emulator("dynamodb", "CreateTable", endpoint, effect="write")
        )
        first = port.put_approval_once(_TABLE, approval_item())
        operations.append(
            _emulator("dynamodb", "PutItem", endpoint, effect="write")
        )
        second = port.put_approval_once(_TABLE, approval_item())
        operations.append(
            _emulator(
                "dynamodb",
                "PutItem",
                endpoint,
                effect="write",
                error_code=None
                if second
                else "ConditionalCheckFailedException",
            )
        )
        port.create_function(_FUNCTION, build_price_pilot_zip())
        operations.append(
            _emulator("lambda", "CreateFunction", endpoint, effect="write")
        )
        lambda_response = port.invoke_function(
            _FUNCTION,
            {"weeks": 6, "budget": 25_000},
        )
        operations.append(
            _emulator("lambda", "Invoke", endpoint, effect="write")
        )
    except (ClientError, LocalStackUnavailableError, OSError) as exc:
        return _emulator_failed_result(
            settings=settings,
            operations=operations,
            data=data,
            step=_failed_step(len(operations)),
            exc=exc,
        )

    data["conditional_first_written"] = first
    data["conditional_second_rejected"] = not second
    data["lambda_payload"] = dict(lambda_response)
    data["lambda_function_error"] = "function_error" in lambda_response
    data["real_writes"] = True
    data["provenance"] = port.__class__.__name__
    return _demo_result(
        execution="emulator",
        settings=settings,
        headline="LocalStack loopback port completed core emulator writes.",
        operations=tuple(operations),
        data=data,
    )


def _live_not_run() -> DemoResult:
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="local",
        lifecycle_refs=_REFS,
        requested_execution="live",
        mode="not_run",
        status="blocked",
        headline="LocalStack core demo is local-only.",
        operations=[],
        evidence=evidence(),
        error=error_result(
            "contract_only",
            "Use offline Stubber replay or emulator mode.",
        ),
    )


def _demo_result(
    *,
    execution: Execution,
    settings: Settings,
    headline: str,
    operations: tuple[OperationOutcome, ...],
    data: Mapping[str, Any],
) -> DemoResult:
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="local",
        lifecycle_refs=_REFS,
        requested_execution=execution,
        headline=headline,
        operations=list(operations),
        evidence=evidence(
            sdk_invoked=any(item["request_validated"] for item in operations),
            network_attempted=any(
                item["transport"] == "loopback" for item in operations
            ),
            aws_executed=False,
            region=settings.region,
            packages=_sdk_packages(),
            fixture_id=next(
                (
                    item["fixture_id"]
                    for item in operations
                    if item["fixture_id"]
                ),
                None,
            ),
            emulator={"endpoint": settings.localstack_endpoint}
            if execution == "emulator"
            else None,
        ),
        data=dict(data),
    )


def _emulator_unavailable(endpoint: str, message: str) -> DemoResult:
    operation_failed = operation(
        service="localstack",
        operation="Health",
        phase="setup",
        execution_target="emulator",
        mode="attempt_failed",
        status="error",
        effect="read",
        transport="loopback",
        endpoint_url=endpoint,
        response_received=False,
        request_validated=True,
        error_code="emulator_unavailable",
    )
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="local",
        lifecycle_refs=_REFS,
        requested_execution="emulator",
        headline="LocalStack loopback endpoint was unavailable.",
        operations=[operation_failed],
        evidence=evidence(
            sdk_invoked=True,
            network_attempted=True,
            packages=_sdk_packages(),
            emulator={"endpoint": endpoint},
        ),
        data={"real_writes": False},
        error=error_result("emulator_unavailable", message),
    )


def _conditional_rejected(
    client: Any,
    params: Mapping[str, Any],
) -> bool:
    try:
        client.put_item(**params)
    except ClientError as exc:
        return _client_error_code(exc) == "ConditionalCheckFailedException"
    return False


def _sdk_packages() -> dict[str, str]:
    return {
        "boto3": metadata.version("boto3"),
        "botocore": metadata.version("botocore"),
    }


def _client_error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def _client_error_status(exc: ClientError) -> int | None:
    metadata = exc.response.get("ResponseMetadata", {})
    status = metadata.get("HTTPStatusCode")
    return status if isinstance(status, int) else None


def _emulator_failed_result(
    *,
    settings: Settings,
    operations: list[OperationOutcome],
    data: Mapping[str, Any],
    step: tuple[str, str, Effect, Phase],
    exc: ClientError | LocalStackUnavailableError | OSError,
) -> DemoResult:
    service, operation_name, effect, phase = step
    failed = (
        _client_error_operation(
            service,
            operation_name,
            settings.localstack_endpoint,
            effect=effect,
            phase=phase,
            exc=exc,
        )
        if isinstance(exc, ClientError)
        else _attempt_failed_operation(
            service,
            operation_name,
            settings.localstack_endpoint,
            effect=effect,
            phase=phase,
        )
    )
    result_code = _result_error_code(exc)
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="local",
        lifecycle_refs=_REFS,
        requested_execution="emulator",
        headline="LocalStack loopback operation did not complete.",
        operations=[*operations, failed],
        evidence=evidence(
            sdk_invoked=True,
            network_attempted=True,
            emulator={"endpoint": settings.localstack_endpoint},
        ),
        data={**dict(data), "real_writes": False},
        error=error_result(result_code, str(exc)),
    )


def _failed_step(completed_operations: int) -> tuple[str, str, Effect, Phase]:
    try:
        return _EMULATOR_STEPS[completed_operations]
    except IndexError:
        return _EMULATOR_STEPS[-1]


def _client_error_operation(
    service: str,
    operation_name: str,
    endpoint: str,
    *,
    effect: Effect,
    phase: Phase,
    exc: ClientError,
) -> OperationOutcome:
    status = _client_error_status(exc)
    return _emulator(
        service,
        operation_name,
        endpoint,
        effect=effect,
        phase=phase,
        status="blocked",
        response_received=True,
        http_status=status,
        error_code=_operation_error_code(exc),
    )


def _attempt_failed_operation(
    service: str,
    operation_name: str,
    endpoint: str,
    *,
    effect: Effect,
    phase: Phase,
) -> OperationOutcome:
    return operation(
        service=service,
        operation=operation_name,
        phase=phase,
        execution_target="emulator",
        mode="attempt_failed",
        status="error",
        effect=effect,
        transport="loopback",
        endpoint_url=endpoint,
        response_received=False,
        request_validated=True,
        error_code="emulator_unavailable",
    )


def _operation_error_code(exc: ClientError) -> str:
    return (
        "entitlement_missing"
        if _is_entitlement_error(exc)
        else _client_error_code(exc)
    )


def _result_error_code(
    exc: ClientError | LocalStackUnavailableError | OSError,
) -> ErrorCode:
    if isinstance(exc, ClientError) and _is_entitlement_error(exc):
        return "entitlement_missing"
    return "emulator_unavailable"


def _is_entitlement_error(exc: ClientError) -> bool:
    code = _client_error_code(exc).lower()
    return (
        _client_error_status(exc) == 403
        or "license" in code
        or "entitlement" in code
        or "authtoken" in code
    )


def _localstack_metadata(
    health: Mapping[str, Any],
    info: Mapping[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    services = health.get("services")
    if isinstance(services, Mapping):
        metadata["services"] = {
            str(name): str(status)
            for name, status in services.items()
            if isinstance(name, str)
        }
    for key in ("version", "image", "digest", "plan"):
        value = info.get(key)
        if isinstance(value, str):
            metadata[key] = value
    edition = info.get("edition")
    if "plan" not in metadata and isinstance(edition, str):
        metadata["plan"] = edition
    return metadata


def _localstack_session() -> BotoSessionPort:
    import boto3

    return cast(
        "BotoSessionPort",
        boto3.Session(
            aws_access_key_id="".join(("te", "st")),
            aws_secret_access_key="".join(("te", "st")),
            aws_session_token="".join(("te", "st")),
            region_name=_FIXED_REGION,
        ),
    )


def _wait_for_lambda_active(client: Any, function_name: str) -> None:
    waiter_factory = getattr(client, "get_waiter", None)
    if not callable(waiter_factory):
        return
    waiter = waiter_factory("function_active_v2")
    waiter.wait(FunctionName=function_name)


def _http_json_get(url: str, timeout: float) -> Mapping[str, Any]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        message = "LocalStack HTTP endpoint must be http(s)"
        raise LocalStackUnavailableError(message)
    if not _is_loopback_host(parsed.hostname):
        message = "LocalStack HTTP endpoint must be loopback"
        raise LocalStackUnavailableError(message)
    connection_class = (
        HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    )
    path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    connection = connection_class(
        parsed.hostname,
        port=parsed.port,
        timeout=timeout,
    )
    try:
        connection.request(
            "GET",
            path,
            headers={"User-Agent": "awsai-workshop/1"},
        )
        response = connection.getresponse()
        payload = response.read()
    finally:
        connection.close()
    parsed = json.loads(payload.decode())
    if not isinstance(parsed, dict):
        message = "LocalStack HTTP endpoint returned non-object JSON"
        raise LocalStackUnavailableError(message)
    return cast("Mapping[str, Any]", parsed)


def _fixture(
    service: str,
    operation_name: str,
    *,
    effect: Effect,
) -> OperationOutcome:
    return operation(
        service=service,
        operation=operation_name,
        execution_target="fixture",
        mode="local_contract",
        effect=effect,
        fixture_id=_FIXTURE_ID,
    )


def _emulator(
    service: str,
    operation_name: str,
    endpoint: str,
    *,
    effect: Effect,
    phase: Phase = "main",
    status: Status = "ok",
    response_received: bool = True,
    http_status: int | None = None,
    error_code: str | None = None,
) -> OperationOutcome:
    return operation(
        service=service,
        operation=operation_name,
        phase=phase,
        execution_target="emulator",
        mode="local_emulator",
        status=status,
        effect=effect,
        transport="loopback",
        endpoint_url=endpoint,
        response_received=response_received,
        http_status=http_status,
        error_code=error_code,
    )


def _create_table_response() -> dict[str, Any]:
    return {
        "TableDescription": {
            "TableName": _TABLE,
            "TableStatus": "ACTIVE",
            "KeySchema": [
                {"AttributeName": "action_key", "KeyType": "HASH"},
            ],
            "AttributeDefinitions": [
                {"AttributeName": "action_key", "AttributeType": "S"},
            ],
            "BillingModeSummary": {"BillingMode": "PAY_PER_REQUEST"},
        },
    }


def _create_function_response() -> dict[str, Any]:
    return {
        "FunctionName": _FUNCTION,
        "FunctionArn": (
            "arn:aws:lambda:us-east-1:000000000000:function:price_pilot"
        ),
        "Runtime": "python3.12",
        "Role": _ROLE_ARN,
        "Handler": "price_pilot.handler",
        "CodeSize": len(build_price_pilot_zip()),
        "Description": "",
        "Timeout": 3,
        "MemorySize": 128,
        "LastModified": "2026-01-01T00:00:00.000+0000",
        "Version": "$LATEST",
    }


def _invoke_response() -> dict[str, Any]:
    payload = json.dumps(_price_pilot_payload()).encode()
    return {
        "StatusCode": 200,
        "ExecutedVersion": "$LATEST",
        "Payload": StreamingBody(BytesIO(payload), len(payload)),
    }


def _read_payload(stream: StreamingBody) -> dict[str, Any]:
    payload = stream.read()
    return cast("dict[str, Any]", json.loads(payload.decode()))


def _price_pilot_payload() -> dict[str, Any]:
    cost = price_pilot()
    return {
        "weeks": cost.weeks,
        "staffing_usd": cost.staffing_usd,
        "platform_usd": cost.platform_usd,
        "total_usd": cost.total_usd,
        "headroom_usd": cost.headroom_usd,
        "within_budget": cost.within_budget,
    }


def _write_zip_entry(
    archive: zipfile.ZipFile,
    name: str,
    source: str,
) -> None:
    info = zipfile.ZipInfo(name)
    info.date_time = (2026, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, source)


def _scenario_source_for_lambda() -> str:
    scenario_path = Path(__file__).with_name("scenario.py")
    return scenario_path.read_text(encoding="utf-8")


def _require_loopback_endpoint(endpoint_url: str) -> None:
    host = urlsplit(endpoint_url).hostname
    if host is None or not _is_loopback_host(host):
        message = "LocalStack endpoint must be loopback"
        raise LocalStackUnavailableError(message)


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        from ipaddress import ip_address

        return ip_address(host).is_loopback
    except ValueError:
        return False


DEFAULT_ENDPOINT = DEFAULT_LOCALSTACK_ENDPOINT
