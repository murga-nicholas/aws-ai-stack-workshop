from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from botocore.exceptions import ParamValidationError
from botocore.model import ShapeResolver

from awsai_demo.stubs import (
    named_fixture_stream,
    stubbed_client,
    validate_operation_request,
    validate_response_timestamps,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def test_strict_stubber_requires_expected_params_and_asserts_empty() -> None:
    with stubbed_client("sts") as stubber:
        stubber.add_response(
            "get_caller_identity",
            {
                "UserId": "AIDACKCEVSQ6C2EXAMPLE",
                "Account": "123456789012",
                "Arn": "arn:aws:iam::123456789012:user/demo",
            },
            expected_params={},
        )

        response = stubber.client.get_caller_identity()

    assert response["Account"] == "123456789012"


def test_strict_stubber_detects_pending_response() -> None:
    with pytest.raises(AssertionError), stubbed_client("sts") as stubber:
        stubber.add_response(
            "get_caller_identity",
            {
                "UserId": "AIDACKCEVSQ6C2EXAMPLE",
                "Account": "123456789012",
                "Arn": "arn:aws:iam::123456789012:user/demo",
            },
            expected_params={},
        )


def test_strict_stubber_can_return_client_error() -> None:
    with stubbed_client("sts") as stubber:
        stubber.add_client_error(
            "get_caller_identity",
            service_error_code="AccessDenied",
            service_message="denied",
            expected_params={},
            http_status_code=403,
        )
        with pytest.raises(Exception) as exc_info:
            stubber.client.get_caller_identity()

    assert "AccessDenied" in str(exc_info.value)


def test_validate_operation_request_returns_evidence() -> None:
    evidence = validate_operation_request(
        service_name="bedrock-runtime",
        operation_name="CountTokens",
        params={
            "modelId": "amazon.test",
            "input": {"converse": {"messages": []}},
        },
        fixture_id="count-v1",
    )

    assert evidence.fixture_id == "count-v1"
    assert evidence.service == "bedrock-runtime"
    assert evidence.operation == "CountTokens"
    assert evidence.request_validated is True
    assert evidence.boto3_version
    assert evidence.botocore_version


def test_validate_operation_request_rejects_bad_params() -> None:
    with pytest.raises(ParamValidationError):
        validate_operation_request(
            service_name="bedrock-runtime",
            operation_name="CountTokens",
            params={"input": {}},
            fixture_id="bad-v1",
        )


def test_validate_operation_request_rejects_input_for_empty_shape() -> None:
    with pytest.raises(ValueError, match="does not accept input"):
        validate_operation_request(
            service_name="acm",
            operation_name="GetAccountConfiguration",
            params={"extra": "value"},
            fixture_id="empty-input-v1",
        )


def test_validate_operation_request_accepts_empty_shape_without_params() -> (
    None
):
    evidence = validate_operation_request(
        service_name="acm",
        operation_name="GetAccountConfiguration",
        params={},
        fixture_id="empty-input-v1",
    )

    assert evidence.request_validated is True


def test_named_fixture_stream_replays_events() -> None:
    events: list[Mapping[str, object]] = [{"chunk": {"bytes": b"x"}}]

    stream = named_fixture_stream(fixture_id="stream-v1", events=events)

    assert stream.fixture_id == "stream-v1"
    assert list(stream) == events


def test_named_fixture_stream_requires_id() -> None:
    with pytest.raises(ValueError):
        named_fixture_stream(fixture_id="", events=[])


@pytest.mark.parametrize("timestamp", [1_700_000_000, "2026-01-01T00:00:00Z"])
def test_stubber_rejects_timestamp_fixtures_the_real_parser_cannot_return(
    timestamp: object,
) -> None:
    from awsai_demo.model_lifecycle_demo import _list_profiles_response

    response = _list_profiles_response()
    response["inferenceProfileSummaries"][0]["createdAt"] = timestamp

    with (
        stubbed_client("bedrock") as stubber,
        pytest.raises(TypeError, match=r"\[0\].createdAt.*requires datetime"),
    ):
        stubber.add_response(
            "list_inference_profiles", response, expected_params={}
        )


def test_timestamp_shape_validation_handles_maps_and_absent_output() -> None:
    shapes = ShapeResolver(
        {
            "TimestampMap": {
                "type": "map",
                "key": {"shape": "String"},
                "value": {"shape": "Timestamp"},
            },
            "String": {"type": "string"},
            "Timestamp": {"type": "timestamp"},
        }
    )
    shape = shapes.get_shape_by_name("TimestampMap")
    validate_response_timestamps(
        shape, {"created": datetime(2026, 1, 1, tzinfo=UTC)}, path="Records"
    )
    validate_response_timestamps(None, {}, path="EmptyOperation")

    with pytest.raises(TypeError, match=r"Records\[created\]"):
        validate_response_timestamps(
            shape, {"created": "2026-01-01"}, path="Records"
        )
