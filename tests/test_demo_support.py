from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from awsai_demo import demo_support
from awsai_demo.demo_support import (
    AwsResponse,
    BotoAwsPort,
    billable_not_priced,
    bounded_boto_config,
    build_default_boto_port,
    build_result,
    client_method_name,
    contract_only,
    fixture_operation,
    local_operation,
    not_run_operation,
    port_operation,
    require_port_not_run,
    stream_port_operation,
    unsupported_emulator,
    validate_request,
)
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from awsai_demo.contracts import OperationOutcome


class FakePort:
    def __init__(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        endpoint_url: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.payload = dict(payload or {})
        self.endpoint_url = endpoint_url
        self.error = error
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []

    def call(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse | Mapping[str, Any]:
        self.calls.append((service, operation_name, params))
        if self.error is not None:
            raise self.error
        if self.endpoint_url is None:
            return self.payload
        return AwsResponse(self.payload, self.endpoint_url)


class FakeStreamPort:
    def __init__(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        endpoint_url: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.events = tuple(events)
        self.endpoint_url = endpoint_url
        self.error = error

    def stream(
        self,
        service: str,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> AwsResponse | Sequence[Mapping[str, Any]]:
        del service, operation_name, params
        if self.error is not None:
            raise self.error
        if self.endpoint_url is None:
            return self.events
        return AwsResponse({"stream": self.events}, self.endpoint_url)


class FakeBotoSession:
    def __init__(self) -> None:
        self.clients: list[tuple[str, dict[str, object]]] = []
        self.client_instance = FakeBotoClient()

    def client(self, service_name: str, **kwargs: object) -> FakeBotoClient:
        self.clients.append((service_name, kwargs))
        return self.client_instance


class FakeBotoClient:
    def __init__(self) -> None:
        self.meta: object = FakeMeta("https://bedrock.us-east-1.amazonaws.com")
        self.params: Mapping[str, Any] | None = None

    def list_foundation_models(self, **kwargs: Any) -> dict[str, object]:
        self.params = kwargs
        return {"modelSummaries": []}

    def converse_stream(self, **kwargs: Any) -> dict[str, object]:
        self.params = kwargs
        return {"stream": []}


class FakeMeta:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url


class ScalarBotoSession(FakeBotoSession):
    def __init__(self) -> None:
        self.clients: list[tuple[str, dict[str, object]]] = []
        self.client_instance = ScalarBotoClient()


class ScalarBotoClient(FakeBotoClient):
    def list_foundation_models(self, **kwargs: Any) -> Any:
        self.params = kwargs
        return "not-a-mapping"


class NoEndpointBotoSession(FakeBotoSession):
    def __init__(self) -> None:
        self.clients: list[tuple[str, dict[str, object]]] = []
        self.client_instance = NoEndpointBotoClient()


class NoEndpointBotoClient(FakeBotoClient):
    def __init__(self) -> None:
        self.meta = object()
        self.params: Mapping[str, Any] | None = None


class ErrorBotoSession:
    def __init__(self) -> None:
        self.clients: list[tuple[str, dict[str, object]]] = []
        self.client_instance = ErrorBotoClient()

    def client(self, service_name: str, **kwargs: object) -> ErrorBotoClient:
        self.clients.append((service_name, kwargs))
        return self.client_instance


class ErrorBotoClient:
    def __init__(self) -> None:
        self.meta = FakeMeta("https://bedrock.us-east-1.amazonaws.com")

    def list_foundation_models(self, **kwargs: Any) -> dict[str, object]:
        del kwargs
        raise ClientError(
            {
                "Error": {"Code": "AccessDeniedException"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "ListFoundationModels",
        )


def test_client_method_name_and_request_validation() -> None:
    assert client_method_name("ListFoundationModels") == (
        "list_foundation_models"
    )
    validate_request(
        service="bedrock",
        operation_name="ListFoundationModels",
        params={},
        fixture_id="list-models",
    )


def test_operation_factories_cover_contract_variants() -> None:
    usage = {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}
    fixture = fixture_operation(
        service="bedrock-runtime",
        operation_name="Converse",
        fixture_id="fx",
        effect="infer",
        usage=usage,
    )
    local = local_operation(service="data/lineage.yaml", operation_name="Join")
    skipped = not_run_operation(
        service="bedrock",
        operation_name="ListFoundationModels",
        error_code="missing_configuration",
        request_validated=False,
        status="error",
    )

    assert fixture["usage"] == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
    }
    assert local["mode"] == "local_execution"
    assert skipped["status"] == "error"
    assert (
        billable_not_priced(
            service="bedrock-runtime",
            operation_name="Converse",
        )["error_code"]
        == "budget_exceeded"
    )
    assert (
        require_port_not_run(
            service="bedrock",
            operation_name="ListFoundationModels",
        )["error_code"]
        == "missing_configuration"
    )
    assert (
        unsupported_emulator(
            service="bedrock",
            operation_name="ListInferenceProfiles",
        )["error_code"]
        == "not_supported_by_emulator"
    )
    assert (
        contract_only(
            service="bedrock",
            operation_name="CreateModelInvocationJob",
        )["error_code"]
        == "contract_only"
    )
    output_only = fixture_operation(
        service="bedrock-runtime",
        operation_name="Converse",
        fixture_id="fx-output",
        effect="infer",
        usage={"outputTokens": 2},
    )
    assert output_only["usage"] == {"output_tokens": 2}
    input_only = fixture_operation(
        service="bedrock-runtime",
        operation_name="CountTokens",
        fixture_id="fx-input",
        effect="read",
        usage={"inputTokens": 4},
    )
    assert input_only["usage"] == {"input_tokens": 4}


def test_port_operation_success_modes_and_build_result() -> None:
    params = {"modelIdentifier": "amazon.nova-2-lite-v1:0"}
    live = port_operation(
        port=FakePort(
            {"modelDetails": {"modelId": "amazon.nova-2-lite-v1:0"}},
            endpoint_url="https://bedrock.us-east-1.amazonaws.com",
        ),
        service="bedrock",
        operation_name="GetFoundationModel",
        params=params,
        execution="live",
        effect="read",
        fixture_id="get-model",
    )
    emulator = port_operation(
        port=FakePort(
            {"modelDetails": {}},
            endpoint_url="http://localhost:4566",
        ),
        service="bedrock",
        operation_name="GetFoundationModel",
        params=params,
        execution="emulator",
        effect="read",
        fixture_id="get-model",
    )
    offline = port_operation(
        port=FakePort({"modelDetails": {}}),
        service="bedrock",
        operation_name="GetFoundationModel",
        params=params,
        execution="offline",
        effect="read",
        fixture_id="get-model",
    )

    assert live.outcome["mode"] == "live_service"
    assert emulator.outcome["mode"] == "local_emulator"
    assert offline.outcome["mode"] == "local_contract"
    result = build_result(
        demo="demo",
        technology="Tech",
        lane="models",
        lifecycle_refs=["ref"],
        execution="emulator",
        headline="ok",
        operations=[emulator.outcome],
        settings=Settings(localstack_endpoint="http://localhost:4566"),
        result_error=("validation_failed", "redacted public message"),
    )
    assert result["evidence"]["emulator"] == {
        "endpoint": "http://localhost:4566"
    }
    assert result["error"] == {
        "code": "validation_failed",
        "message": "redacted public message",
    }


def test_port_operation_refuses_unreserved_live_billable_calls() -> None:
    params = {
        "modelId": "amazon.nova-2-lite-v1:0",
        "messages": [],
    }

    with pytest.raises(ValueError, match="prior reservation"):
        port_operation(
            port=FakePort(
                endpoint_url=(
                    "https://bedrock-runtime.us-east-1.amazonaws.com"
                )
            ),
            service="bedrock-runtime",
            operation_name="Converse",
            params=params,
            execution="live",
            effect="infer",
            fixture_id="converse",
        )

    reserved = port_operation(
        port=FakePort(
            endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
        ),
        service="bedrock-runtime",
        operation_name="Converse",
        params=params,
        execution="live",
        effect="infer",
        fixture_id="converse",
        reserved=True,
    )
    assert reserved.outcome["mode"] == "live_model"


def test_port_operation_classifies_client_errors() -> None:
    denied = ClientError(
        {
            "Error": {"Code": "AccessDeniedException"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        "ListFoundationModels",
    )
    throttled = ClientError(
        {
            "Error": {"Code": "ThrottlingException"},
            "ResponseMetadata": {"HTTPStatusCode": 429},
        },
        "ListFoundationModels",
    )
    params: dict[str, Any] = {}
    live = port_operation(
        port=FakePort(error=denied),
        service="bedrock",
        operation_name="ListFoundationModels",
        params=params,
        execution="live",
        effect="read",
        fixture_id="list-models",
        endpoint_url="https://bedrock.us-east-1.amazonaws.com",
    )
    emulator = port_operation(
        port=FakePort(error=throttled),
        service="bedrock",
        operation_name="ListFoundationModels",
        params=params,
        execution="emulator",
        effect="read",
        fixture_id="list-models",
        endpoint_url="http://localhost:4566",
    )

    assert live.outcome["error_code"] == "authorization_denied"
    assert live.outcome["http_status"] == 403
    assert emulator.outcome["error_code"] == "ThrottlingException"
    assert emulator.outcome["mode"] == "local_emulator"


def test_port_operation_classifies_attempt_failures() -> None:
    params: dict[str, Any] = {}
    live = port_operation(
        port=FakePort(
            error=ReadTimeoutError(
                endpoint_url="https://bedrock.example",
                error="slow",
            )
        ),
        service="bedrock",
        operation_name="ListFoundationModels",
        params=params,
        execution="live",
        effect="read",
        fixture_id="list-models",
        endpoint_url="https://bedrock.example",
    )
    emulator = port_operation(
        port=FakePort(error=OSError("down")),
        service="bedrock",
        operation_name="ListFoundationModels",
        params=params,
        execution="emulator",
        effect="read",
        fixture_id="list-models",
        endpoint_url="http://localhost:4566",
    )

    assert live.outcome["mode"] == "attempt_failed"
    assert live.outcome["error_code"] == "timeout"
    assert live.outcome["response_received"] is False
    assert emulator.outcome["error_code"] == "emulator_unavailable"


def test_stream_port_operation_normalizes_sequences_and_responses() -> None:
    params = {
        "modelId": "amazon.nova-2-lite-v1:0",
        "messages": [],
    }
    events = [
        {
            "messageStart": {"role": "assistant"},
            "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
        }
    ]
    sequence = stream_port_operation(
        port=FakeStreamPort(events),
        service="bedrock-runtime",
        operation_name="ConverseStream",
        params=params,
        execution="offline",
        effect="infer",
        fixture_id="stream",
    )
    response = stream_port_operation(
        port=FakeStreamPort(
            events,
            endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        ),
        service="bedrock-runtime",
        operation_name="ConverseStream",
        params=params,
        execution="live",
        effect="infer",
        fixture_id="stream",
        reserved=True,
    )

    expected = {
        "stream": (
            {
                "messageStart": {"role": "assistant"},
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
        )
    }
    assert sequence.payload == expected
    assert response.payload == expected
    assert response.outcome["mode"] == "live_model"


def test_subscription_required_is_blocked_with_stable_code() -> None:
    called = port_operation(
        port=FakePort(
            error=ClientError(
                {
                    "Error": {"Code": "SubscriptionRequiredException"},
                    "ResponseMetadata": {"HTTPStatusCode": 400},
                },
                "TranslateText",
            ),
            endpoint_url="https://translate.us-east-1.amazonaws.com",
        ),
        service="translate",
        operation_name="TranslateText",
        params={
            "Text": "hello",
            "SourceLanguageCode": "en",
            "TargetLanguageCode": "es",
        },
        execution="live",
        effect="infer",
        fixture_id="translation-subscription",
        reserved=True,
    )

    assert called.outcome["mode"] == "live_service"
    assert called.outcome["status"] == "blocked"
    assert called.outcome["error_code"] == "subscription_required"
    assert called.outcome["http_status"] == 400
    assert called.outcome["response_received"] is True


def test_stream_port_operation_classifies_errors() -> None:
    params = {
        "modelId": "amazon.nova-2-lite-v1:0",
        "messages": [],
    }
    denied = ClientError(
        {
            "Error": {"Code": "AccessDeniedException"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        "ConverseStream",
    )
    client_error = stream_port_operation(
        port=FakeStreamPort(
            [],
            endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            error=denied,
        ),
        service="bedrock-runtime",
        operation_name="ConverseStream",
        params=params,
        execution="live",
        effect="infer",
        fixture_id="stream",
        reserved=True,
    )
    failed = stream_port_operation(
        port=FakeStreamPort([], error=OSError("down")),
        service="bedrock-runtime",
        operation_name="ConverseStream",
        params=params,
        execution="emulator",
        effect="infer",
        fixture_id="stream",
        endpoint_url="http://localhost:4566",
    )

    assert client_error.outcome["error_code"] == "authorization_denied"
    assert failed.outcome["mode"] == "attempt_failed"
    assert failed.outcome["error_code"] == "emulator_unavailable"


def test_boto_aws_port_dispatches_with_bounded_config() -> None:
    session = FakeBotoSession()
    config = bounded_boto_config(ExecutionPolicy(max_wall_seconds=5))
    port = BotoAwsPort(
        session=session,
        region_name="us-east-1",
        config=config,
    )

    response = port.call("bedrock", "ListFoundationModels", {})

    assert response.payload == {"modelSummaries": []}
    assert response.endpoint_url == "https://bedrock.us-east-1.amazonaws.com"
    assert session.clients[0] == (
        "bedrock",
        {"region_name": "us-east-1", "config": config},
    )


def test_boto_port_normalizes_real_stubber_timestamp_responses() -> None:
    from awsai_demo.model_lifecycle_demo import _get_model_response
    from awsai_demo.stubs import stubbed_client

    with stubbed_client("bedrock") as stubber:
        params = {"modelIdentifier": "amazon.nova-2-lite-v1:0"}
        stubber.add_response(
            "get_foundation_model",
            _get_model_response(),
            expected_params=params,
        )

        class StubbedSession:
            def client(self, service_name: str, **kwargs: Any) -> Any:
                assert service_name == "bedrock"
                assert kwargs["region_name"] == "us-east-1"
                return stubber.client

        port = BotoAwsPort(session=StubbedSession(), region_name="us-east-1")
        response = port.call("bedrock", "GetFoundationModel", params)

    assert response.payload["modelDetails"]["modelLifecycle"] == {
        "status": "ACTIVE",
        "startOfLifeTime": "2025-12-02T00:00:00+00:00",
    }


def test_boto_aws_port_stream_and_loopback_validation() -> None:
    session = FakeBotoSession()
    port = BotoAwsPort(
        session=session,
        region_name="us-east-1",
        endpoint_url="http://localhost:4566",
        loopback_only=True,
    )
    response = port.stream(
        "bedrock-runtime",
        "ConverseStream",
        {"modelId": "amazon.nova-2-lite-v1:0", "messages": []},
    )

    assert response.payload == {"stream": []}
    assert session.clients[0][1]["endpoint_url"] == "http://localhost:4566"

    with pytest.raises(ValueError, match="loopback"):
        BotoAwsPort(
            session=session,
            region_name="us-east-1",
            endpoint_url="https://example.com",
            loopback_only=True,
        )
    with pytest.raises(ValueError, match="loopback"):
        BotoAwsPort(
            session=session,
            region_name="us-east-1",
            loopback_only=True,
        )
    with pytest.raises(ValueError, match="loopback"):
        BotoAwsPort(
            session=session,
            region_name="us-east-1",
            endpoint_url="not a url",
            loopback_only=True,
        )


def test_boto_aws_port_rejects_non_mapping_payload() -> None:
    port = BotoAwsPort(
        session=ScalarBotoSession(),
        region_name="us-east-1",
    )

    with pytest.raises(TypeError, match="non-mapping"):
        port.call("bedrock", "ListFoundationModels", {})


def test_boto_aws_port_uses_fallback_endpoint_without_client_meta() -> None:
    port = BotoAwsPort(
        session=NoEndpointBotoSession(),
        region_name="us-east-1",
        endpoint_url="https://bedrock.example",
    )

    response = port.call("bedrock", "ListFoundationModels", {})

    assert response.endpoint_url == "https://bedrock.example"


def test_build_default_boto_port_selects_expected_sessions() -> None:
    created: list[dict[str, str]] = []

    def factory(**kwargs: str) -> FakeBotoSession:
        created.append(kwargs)
        return FakeBotoSession()

    assert (
        build_default_boto_port(
            execution="offline",
            settings=Settings(),
            policy=ExecutionPolicy(),
            session_factory=factory,
        )
        is None
    )
    assert (
        build_default_boto_port(
            execution="emulator",
            settings=Settings(),
            policy=ExecutionPolicy(),
            session_factory=factory,
        )
        is None
    )
    emulator = build_default_boto_port(
        execution="emulator",
        settings=Settings(localstack_auth_token="token"),  # noqa: S106
        policy=ExecutionPolicy(max_wall_seconds=3),
        session_factory=factory,
    )
    live = build_default_boto_port(
        execution="live",
        settings=Settings(aws_profile="demo"),
        policy=ExecutionPolicy(),
        session_factory=factory,
    )

    assert emulator is not None
    assert emulator.credential_source == "none"
    assert live is not None
    assert live.credential_source == "profile"
    assert created[0]["region_name"] == "us-east-1"
    assert created[1] == {"profile_name": "demo", "region_name": "us-east-1"}


def test_build_result_uses_only_explicit_credential_source() -> None:
    operations: list[OperationOutcome] = [
        fixture_operation(
            service="svc",
            operation_name="op",
            fixture_id="fixture",
            effect="none",
            request_validated=False,
        )
    ]
    implicit_result = build_result(
        demo="demo",
        technology="Tech",
        lane="models",
        lifecycle_refs=["ref"],
        execution="offline",
        headline="ok",
        operations=operations,
        settings=Settings(aws_profile="profile"),
    )
    explicit_result = build_result(
        demo="demo",
        technology="Tech",
        lane="models",
        lifecycle_refs=["ref"],
        execution="offline",
        headline="ok",
        operations=operations,
        settings=Settings(aws_creds_file_path=Path("synthetic.csv")),
        credential_source="csv_file",
    )

    assert implicit_result["evidence"]["credential_source"] == "none"
    assert explicit_result["evidence"]["credential_source"] == "csv_file"


def test_boto_aws_port_attaches_endpoint_to_client_errors() -> None:
    port = BotoAwsPort(
        session=ErrorBotoSession(),
        region_name="us-east-1",
    )

    result = port_operation(
        port=port,
        service="bedrock",
        operation_name="ListFoundationModels",
        params={},
        execution="live",
        effect="read",
        fixture_id="list-models",
    )

    assert result.outcome["error_code"] == "authorization_denied"
    assert result.outcome["endpoint_url"] == (
        "https://bedrock.us-east-1.amazonaws.com"
    )


def test_attach_endpoint_ignores_absent_endpoint() -> None:
    exc = RuntimeError("no endpoint")

    demo_support._attach_endpoint(exc, None)

    assert not hasattr(exc, "_awsai_endpoint_url")
