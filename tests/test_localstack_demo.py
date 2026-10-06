from __future__ import annotations

import json
import sys
import zipfile
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from botocore.exceptions import ClientError
from botocore.response import StreamingBody

import awsai_demo.localstack_demo as localstack_module
from awsai_demo.localstack_demo import (
    LocalStackHttpBotoPort,
    LocalStackUnavailableError,
    _client_error_code,
    _client_error_status,
    _conditional_rejected,
    _failed_step,
    _http_json_get,
    _localstack_metadata,
    _operation_error_code,
    _result_error_code,
    _run_emulator,
    _wait_for_lambda_active,
    approval_item,
    build_price_pilot_zip,
    create_bucket_params,
    create_function_params,
    create_table_params,
    invoke_function_params,
    put_object_params,
    run_localstack_demo,
    run_strict_localstack_stubber,
)
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from pathlib import Path


class RecordingLocalStackPort:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.puts = 0

    def health(self) -> dict[str, object]:
        self.calls.append(("health", ()))
        return {"services": {"s3": "available"}}

    def info(self) -> dict[str, object]:
        self.calls.append(("info", ()))
        return {"edition": "community", "version": "2026.09.0"}

    def create_bucket(self, bucket: str) -> dict[str, object]:
        self.calls.append(("create_bucket", (bucket,)))
        return {"Location": f"/{bucket}"}

    def put_object(
        self,
        bucket: str,
        key: str,
        body: bytes,
    ) -> dict[str, object]:
        self.calls.append(("put_object", (bucket, key, body)))
        return {"ETag": "demo"}

    def create_table(self, table_name: str) -> dict[str, object]:
        self.calls.append(("create_table", (table_name,)))
        return {"TableDescription": {"TableName": table_name}}

    def put_approval_once(
        self,
        table_name: str,
        item: dict[str, object],
    ) -> bool:
        self.puts += 1
        self.calls.append(("put_approval_once", (table_name, item)))
        return self.puts == 1

    def create_function(
        self,
        function_name: str,
        zip_bytes: bytes,
    ) -> dict[str, object]:
        self.calls.append(("create_function", (function_name, zip_bytes)))
        return {"FunctionName": function_name}

    def invoke_function(
        self,
        function_name: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append(("invoke_function", (function_name, payload)))
        return {"accepted": False, "weeks": payload["weeks"]}


class DownLocalStackPort(RecordingLocalStackPort):
    def health(self) -> dict[str, object]:
        message = "localstack is down"
        raise LocalStackUnavailableError(message)


class NoErrorClient:
    def put_item(self, **kwargs: Any) -> dict[str, object]:
        assert kwargs
        return {}


class OtherErrorClient:
    def put_item(self, **kwargs: Any) -> dict[str, object]:
        assert kwargs
        error = {
            "Error": {"Code": "ProvisionedThroughputExceededException"},
            "ResponseMetadata": {"HTTPStatusCode": 400},
        }
        raise ClientError(error, "PutItem")


class FakeBody(BytesIO):
    pass


class FakeS3Client:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def create_bucket(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(f"s3.create_bucket:{kwargs['Bucket']}")
        return {"Location": f"/{kwargs['Bucket']}"}

    def put_object(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(f"s3.put_object:{kwargs['Key']}")
        return {"ETag": "demo"}


class FakeDynamoClient:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.puts = 0

    def create_table(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(f"dynamodb.create_table:{kwargs['TableName']}")
        return {"TableDescription": {"TableName": kwargs["TableName"]}}

    def put_item(self, **kwargs: Any) -> dict[str, object]:
        self.puts += 1
        self.calls.append("dynamodb.put_item")
        if self.puts == 2:
            error = {
                "Error": {"Code": "ConditionalCheckFailedException"},
                "ResponseMetadata": {"HTTPStatusCode": 400},
            }
            raise ClientError(error, "PutItem")
        return {"ConsumedCapacity": {"TableName": kwargs["TableName"]}}


class FakeLambdaClient:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def create_function(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(f"lambda.create_function:{kwargs['FunctionName']}")
        return {"FunctionName": kwargs["FunctionName"]}

    def invoke(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(f"lambda.invoke:{kwargs['FunctionName']}")
        payload = json.dumps(
            {
                "weeks": 6,
                "staffing_usd": 18600.0,
                "platform_usd": 1080.0,
                "total_usd": 19680.0,
                "headroom_usd": 5320.0,
                "within_budget": True,
            },
        ).encode()
        return {"Payload": FakeBody(payload)}


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._clients: dict[str, object] = {}
        self._last_kwargs: dict[str, Any] = {}

    def client(self, service_name: str, **kwargs: Any) -> object:
        assert kwargs["endpoint_url"] == "http://localhost:4566"
        assert kwargs["region_name"] == "us-east-1"
        self._last_kwargs = dict(kwargs)
        if service_name in self._clients:
            return self._clients[service_name]
        if service_name == "s3":
            client = FakeS3Client(self.calls)
        elif service_name == "dynamodb":
            client = FakeDynamoClient(self.calls)
        elif service_name == "lambda":
            client = FakeLambdaClient(self.calls)
        else:
            raise AssertionError(service_name)
        self._clients[service_name] = client
        return client


def http_ok(url: str, timeout: float) -> dict[str, object]:
    assert timeout == 2.0
    if url.endswith("/_localstack/health"):
        return {"services": {"s3": "available"}}
    if url.endswith("/_localstack/info"):
        return {"edition": "community", "version": "2026.09.0"}
    raise AssertionError(url)


def client_error(
    code: str,
    operation_name: str,
    *,
    status: object = 400,
) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation_name,
    )


def test_strict_stubber_exercises_s3_dynamodb_and_lambda() -> None:
    replay = run_strict_localstack_stubber()

    assert replay.data["strict_stubber"] is True
    assert replay.data["real_writes"] is False
    assert replay.data["conditional_second_rejected"] is True
    assert replay.data["lambda_payload"] == {
        "weeks": 6,
        "staffing_usd": 18600.0,
        "platform_usd": 1080.0,
        "total_usd": 19680.0,
        "headroom_usd": 5320.0,
        "within_budget": True,
    }
    assert [item["service"] for item in replay.operations] == [
        "s3",
        "s3",
        "dynamodb",
        "dynamodb",
        "dynamodb",
        "lambda",
        "lambda",
    ]

    demo = run_localstack_demo(execution="offline")
    assert demo["mode"] == "local_contract"
    assert set(demo["evidence"]["packages"]) == {"boto3", "botocore"}
    assert demo["data"]["conditional_second_rejected"] is True
    assert demo["data"]["lambda_payload"]["total_usd"] == 19680.0


def test_emulator_missing_down_and_success_paths() -> None:
    missing = run_localstack_demo(execution="emulator")
    assert missing["mode"] == "not_run"
    assert missing["error"]["code"] == "missing_configuration"

    configured_value = "".join(("configured",))
    settings = Settings(localstack_auth_token=configured_value)

    def down_http(_url: str, _timeout: float) -> dict[str, object]:
        message = "localstack is down"
        raise LocalStackUnavailableError(message)

    no_port = run_localstack_demo(
        execution="emulator",
        settings=settings,
        http_get=down_http,
        session=FakeSession(),
    )
    assert no_port["mode"] == "attempt_failed"
    assert no_port["error"]["code"] == "emulator_unavailable"
    assert no_port["data"] == {"real_writes": False}

    down = run_localstack_demo(
        execution="emulator",
        settings=settings,
        port=DownLocalStackPort(),
    )
    assert down["mode"] == "attempt_failed"
    assert down["error"]["message"] == "localstack is down"

    fake_session = FakeSession()
    built = run_localstack_demo(
        execution="emulator",
        settings=settings,
        http_get=http_ok,
        session=fake_session,
    )
    assert built["mode"] == "local_emulator"
    assert built["data"]["lambda_payload"]["total_usd"] == 19680.0
    assert built["data"]["localstack"] == {
        "services": {"s3": "available"},
        "version": "2026.09.0",
        "plan": "community",
    }
    assert fake_session.calls == [
        "s3.create_bucket:awsai-workshop-localstack-demo",
        "s3.put_object:notes/price_pilot.txt",
        "dynamodb.create_table:awsai-workshop-decisions",
        "dynamodb.put_item",
        "dynamodb.put_item",
        "lambda.create_function:price_pilot",
        "lambda.invoke:price_pilot",
    ]

    port = RecordingLocalStackPort()
    ok = run_localstack_demo(
        execution="emulator",
        settings=settings,
        port=port,
    )
    assert ok["mode"] == "local_emulator"
    assert ok["data"]["real_writes"] is True
    assert ok["data"]["conditional_first_written"] is True
    assert ok["data"]["conditional_second_rejected"] is True
    assert [name for name, _args in port.calls] == [
        "health",
        "info",
        "create_bucket",
        "put_object",
        "create_table",
        "put_approval_once",
        "put_approval_once",
        "create_function",
        "invoke_function",
    ]


def test_live_path_is_contract_only_not_run() -> None:
    demo = run_localstack_demo(execution="live")
    assert demo["mode"] == "not_run"
    assert demo["error"]["code"] == "contract_only"


def test_request_helpers_and_zip_payload_are_deterministic() -> None:
    zip_bytes = build_price_pilot_zip()
    assert zip_bytes == build_price_pilot_zip()
    archive = zipfile.ZipFile(BytesIO(zip_bytes))
    assert archive.namelist() == [
        "awsai_demo/__init__.py",
        "awsai_demo/scenario.py",
        "price_pilot.py",
    ]
    source = archive.read("price_pilot.py").decode()
    assert "def handler" in source

    assert create_bucket_params()["Bucket"].startswith("awsai-workshop")
    assert put_object_params()["Body"].startswith(b"price_pilot")
    assert create_table_params()["BillingMode"] == "PAY_PER_REQUEST"
    assert approval_item()["action_key"]["S"].startswith("accept-proposal")
    assert create_function_params()["Code"]["ZipFile"] == (
        build_price_pilot_zip()
    )
    payload = json.loads(invoke_function_params()["Payload"].decode())
    assert payload == {"weeks": 6, "budget": 25_000}


def test_zip_handler_executes_shared_price_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zip_path = tmp_path / "price_pilot.zip"
    zip_path.write_bytes(build_price_pilot_zip())
    monkeypatch.syspath_prepend(str(zip_path))
    sys.modules.pop("price_pilot", None)

    import price_pilot as handler_module

    response = handler_module.handler({"weeks": 6, "budget": 25_000}, None)

    assert response["staffing_usd"] == 18600.0
    assert response["platform_usd"] == 1080.0
    assert response["total_usd"] == 19680.0
    assert response["headroom_usd"] == 5320.0
    assert response["within_budget"] is True


def test_real_adapter_rejects_non_loopback_endpoint() -> None:
    settings = Settings(localstack_endpoint="http://example.com:4566")
    message = "LocalStack endpoint must be loopback"
    with pytest.raises(LocalStackUnavailableError, match=message):
        LocalStackHttpBotoPort(settings, session=FakeSession())

    configured_value = "".join(("configured",))
    demo = run_localstack_demo(
        execution="emulator",
        settings=Settings(
            localstack_auth_token=configured_value,
            localstack_endpoint="http://example.com:4566",
        ),
    )
    assert demo["mode"] == "attempt_failed"
    assert demo["operations"][0]["operation"] == "Health"
    assert demo["operations"][0]["response_received"] is False


def test_conditional_rejected_helper_codes() -> None:
    assert _conditional_rejected(NoErrorClient(), {"TableName": "t"}) is False
    assert (
        _conditional_rejected(OtherErrorClient(), {"TableName": "t"}) is False
    )
    error = {
        "Error": {"Code": "ConditionalCheckFailedException"},
        "ResponseMetadata": {"HTTPStatusCode": 400},
    }
    assert _client_error_code(ClientError(error, "PutItem")) == (
        "ConditionalCheckFailedException"
    )


def test_emulator_operation_failures_keep_prior_evidence() -> None:
    configured_value = "".join(("configured",))
    settings = Settings(localstack_auth_token=configured_value)

    class DeniedPutObjectPort(RecordingLocalStackPort):
        def put_object(
            self,
            bucket: str,
            key: str,
            body: bytes,
        ) -> dict[str, object]:
            self.calls.append(("put_object", (bucket, key, body)))
            code = "AccessDeniedException"
            operation_name = "PutObject"
            raise client_error(code, operation_name, status=403)

    denied = run_localstack_demo(
        execution="emulator",
        settings=settings,
        port=DeniedPutObjectPort(),
    )
    assert denied["error"]["code"] == "entitlement_missing"
    assert [item["operation"] for item in denied["operations"]] == [
        "Health",
        "Info",
        "CreateBucket",
        "PutObject",
    ]
    failed = denied["operations"][-1]
    assert failed["mode"] == "local_emulator"
    assert failed["response_received"] is True
    assert failed["http_status"] == 403
    assert failed["error_code"] == "entitlement_missing"
    assert denied["data"]["localstack"] == {
        "services": {"s3": "available"},
        "version": "2026.09.0",
        "plan": "community",
    }

    class RefusedBucketPort(RecordingLocalStackPort):
        def create_bucket(self, bucket: str) -> dict[str, object]:
            self.calls.append(("create_bucket", (bucket,)))
            raise OSError

    refused = run_localstack_demo(
        execution="emulator",
        settings=settings,
        port=RefusedBucketPort(),
    )
    failed_attempt = refused["operations"][-1]
    assert failed_attempt["operation"] == "CreateBucket"
    assert failed_attempt["mode"] == "attempt_failed"
    assert failed_attempt["response_received"] is False

    direct = _run_emulator(settings, None)
    assert direct["mode"] == "not_run"
    assert direct["error"]["code"] == "missing_configuration"


def test_real_adapter_handles_existing_resources_and_errors() -> None:
    class SingleClientSession:
        def __init__(self, client: object) -> None:
            self.client_instance = client
            self.kwargs: dict[str, object] | None = None

        def client(self, service_name: str, **kwargs: object) -> object:
            assert service_name
            self.kwargs = dict(kwargs)
            return self.client_instance

    class RaisingClient:
        def __init__(self, code: str) -> None:
            self.code = code

        def create_bucket(self, **_kwargs: object) -> dict[str, object]:
            raise client_error(self.code, "CreateBucket")

        def create_table(self, **_kwargs: object) -> dict[str, object]:
            raise client_error(self.code, "CreateTable")

        def put_item(self, **_kwargs: object) -> dict[str, object]:
            raise client_error(self.code, "PutItem")

        def create_function(self, **_kwargs: object) -> dict[str, object]:
            raise client_error(self.code, "CreateFunction")

    class Waiter:
        def __init__(self) -> None:
            self.waits: list[dict[str, object]] = []

        def wait(self, **kwargs: object) -> None:
            self.waits.append(dict(kwargs))

    class LambdaCreateClient:
        def __init__(self) -> None:
            self.waiter = Waiter()

        def create_function(self, **kwargs: object) -> dict[str, object]:
            return {"FunctionName": kwargs["FunctionName"]}

        def get_waiter(self, name: str) -> Waiter:
            assert name == "function_active_v2"
            return self.waiter

    settings = Settings()
    assert LocalStackHttpBotoPort(
        settings,
        session=SingleClientSession(
            RaisingClient("BucketAlreadyOwnedByYou"),
        ),
    ).create_bucket("demo") == {"already_exists": True}
    with pytest.raises(ClientError):
        LocalStackHttpBotoPort(
            settings,
            session=SingleClientSession(RaisingClient("AccessDenied")),
        ).create_bucket("demo")

    assert LocalStackHttpBotoPort(
        settings,
        session=SingleClientSession(
            RaisingClient("ResourceInUseException"),
        ),
    ).create_table("demo") == {"already_exists": True}
    with pytest.raises(ClientError):
        LocalStackHttpBotoPort(
            settings,
            session=SingleClientSession(RaisingClient("AccessDenied")),
        ).create_table("demo")

    assert (
        LocalStackHttpBotoPort(
            settings,
            session=SingleClientSession(
                RaisingClient("ConditionalCheckFailedException"),
            ),
        ).put_approval_once("demo", approval_item())
        is False
    )
    with pytest.raises(ClientError):
        LocalStackHttpBotoPort(
            settings,
            session=SingleClientSession(RaisingClient("AccessDenied")),
        ).put_approval_once("demo", approval_item())

    assert LocalStackHttpBotoPort(
        settings,
        session=SingleClientSession(
            RaisingClient("ResourceConflictException"),
        ),
    ).create_function("price_pilot", build_price_pilot_zip()) == {
        "already_exists": True,
    }
    with pytest.raises(ClientError):
        LocalStackHttpBotoPort(
            settings,
            session=SingleClientSession(RaisingClient("AccessDenied")),
        ).create_function("price_pilot", build_price_pilot_zip())

    lambda_client = LambdaCreateClient()
    created = LocalStackHttpBotoPort(
        settings,
        session=SingleClientSession(lambda_client),
    ).create_function("price_pilot", build_price_pilot_zip())
    assert created == {"FunctionName": "price_pilot"}
    assert lambda_client.waiter.waits == [{"FunctionName": "price_pilot"}]


def test_real_adapter_invoke_payload_variants() -> None:
    class SingleClientSession:
        def __init__(self, client: object) -> None:
            self.client_instance = client

        def client(self, _service_name: str, **_kwargs: object) -> object:
            return self.client_instance

    class InvokeClient:
        def __init__(self, response: dict[str, object]) -> None:
            self.response = response

        def invoke(self, **_kwargs: object) -> dict[str, object]:
            return self.response

    payload_bytes = b'{"ok": true}'
    streaming = StreamingBody(BytesIO(payload_bytes), len(payload_bytes))
    stream_response = LocalStackHttpBotoPort(
        Settings(),
        session=SingleClientSession(
            InvokeClient(
                {"Payload": streaming, "FunctionError": "Handled"},
            ),
        ),
    ).invoke_function("price_pilot", {})
    assert stream_response == {"ok": True, "function_error": "Handled"}

    payload_without_error = b'{"ok": false}'
    streaming_without_error = StreamingBody(
        BytesIO(payload_without_error),
        len(payload_without_error),
    )
    stream_without_error_response = LocalStackHttpBotoPort(
        Settings(),
        session=SingleClientSession(
            InvokeClient({"Payload": streaming_without_error}),
        ),
    ).invoke_function("price_pilot", {})
    assert stream_without_error_response == {"ok": False}

    body_with_error = FakeBody(b'{"accepted": false}')
    body_response = LocalStackHttpBotoPort(
        Settings(),
        session=SingleClientSession(
            InvokeClient(
                {"Payload": body_with_error, "FunctionError": "Unhandled"},
            ),
        ),
    ).invoke_function("price_pilot", {})
    assert body_response == {
        "accepted": False,
        "function_error": "Unhandled",
    }

    class TextBody:
        def read(self) -> str:
            return "not-bytes"

    raw_response = {"Payload": TextBody(), "StatusCode": 200}
    returned = LocalStackHttpBotoPort(
        Settings(),
        session=SingleClientSession(InvokeClient(raw_response)),
    ).invoke_function("price_pilot", {})
    assert returned == raw_response

    no_body_reader = {"Payload": "plain-text"}
    assert (
        LocalStackHttpBotoPort(
            Settings(),
            session=SingleClientSession(InvokeClient(no_body_reader)),
        ).invoke_function("price_pilot", {})
        == no_body_reader
    )


def test_localstack_http_json_get_validates_loopback_and_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

    class FakeConnection:
        payload = b'{"ok": true}'
        requests: ClassVar[list[tuple[str, str, dict[str, str]]]] = []
        closed = False

        def __init__(
            self,
            host: str,
            *,
            port: int | None,
            timeout: float,
        ) -> None:
            self.host = host
            self.port = port
            self.timeout = timeout

        def request(
            self,
            method: str,
            path: str,
            *,
            headers: dict[str, str],
        ) -> None:
            self.requests.append((method, path, headers))

        def getresponse(self) -> FakeResponse:
            return FakeResponse(self.payload)

        def close(self) -> None:
            self.__class__.closed = True

    monkeypatch.setattr(localstack_module, "HTTPConnection", FakeConnection)
    assert _http_json_get(
        "http://127.0.0.1:4566/_localstack/health?q=1",
        1.5,
    ) == {"ok": True}
    assert FakeConnection.requests == [
        (
            "GET",
            "/_localstack/health?q=1",
            {"User-Agent": "awsai-workshop/1"},
        ),
    ]
    assert FakeConnection.closed is True

    class FakeHttpsConnection(FakeConnection):
        requests: ClassVar[list[tuple[str, str, dict[str, str]]]] = []
        closed = False

    monkeypatch.setattr(
        localstack_module,
        "HTTPSConnection",
        FakeHttpsConnection,
    )
    assert _http_json_get("https://localhost:4566/_localstack/info", 1.0) == {
        "ok": True,
    }
    assert FakeHttpsConnection.closed is True

    with pytest.raises(LocalStackUnavailableError, match="http"):
        _http_json_get("file:///tmp/localstack", 1.0)
    with pytest.raises(LocalStackUnavailableError, match="loopback"):
        _http_json_get("http://example.com:4566/_localstack/health", 1.0)

    class ListPayloadConnection(FakeConnection):
        payload = b"[]"

    monkeypatch.setattr(
        localstack_module,
        "HTTPConnection",
        ListPayloadConnection,
    )
    with pytest.raises(LocalStackUnavailableError, match="non-object"):
        _http_json_get("http://localhost:4566/_localstack/health", 1.0)


def test_localstack_error_and_metadata_helpers() -> None:
    denied = client_error("AccessDeniedException", "PutObject", status=403)
    missing = client_error("LicenseRequired", "PutObject")
    non_int = client_error("AccessDeniedException", "PutObject", status="403")

    assert _client_error_status(denied) == 403
    assert _client_error_status(non_int) is None
    assert _operation_error_code(denied) == "entitlement_missing"
    assert _operation_error_code(missing) == "entitlement_missing"
    assert _operation_error_code(non_int) == "AccessDeniedException"
    assert _result_error_code(denied) == "entitlement_missing"
    assert _result_error_code(OSError("socket closed")) == (
        "emulator_unavailable"
    )
    assert _failed_step(999)[1] == "Invoke"

    assert _localstack_metadata(
        {"services": {"s3": "available", 7: "ignored"}},
        {"edition": "community"},
    ) == {"services": {"s3": "available"}, "plan": "community"}
    assert _localstack_metadata(
        {},
        {
            "version": "2026.09.0",
            "image": "localstack/localstack-pro",
            "digest": "sha256:demo",
            "plan": "pro",
        },
    ) == {
        "version": "2026.09.0",
        "image": "localstack/localstack-pro",
        "digest": "sha256:demo",
        "plan": "pro",
    }


def test_adapter_uses_default_session_and_bounded_policy_config() -> None:
    LocalStackHttpBotoPort(Settings())

    session = FakeSession()
    configured_value = "".join(("configured",))
    demo = run_localstack_demo(
        execution="emulator",
        settings=Settings(localstack_auth_token=configured_value),
        policy=ExecutionPolicy(
            total_max_attempts=3,
            openai_max_retries=2,
            max_wall_seconds=7,
        ),
        http_get=http_ok,
        session=session,
    )

    assert demo["mode"] == "local_emulator"
    assert "config" in session._last_kwargs
    config = session._last_kwargs["config"]
    assert config.retries["total_max_attempts"] == 3
    assert config.connect_timeout == 7


def test_wait_for_lambda_active_without_waiter() -> None:
    class NoWaiter:
        pass

    _wait_for_lambda_active(NoWaiter(), "price_pilot")
