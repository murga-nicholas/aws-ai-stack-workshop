"""Strict botocore Stubber utilities and event-stream fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

_DUMMY_ACCESS_KEY_ID = "AKIDEXAMPLE"
_DUMMY_SECRET_PARTS = ("wJalrXUtnFEMI/K7MDENG", "+bPxRfiCYEXAMPLEKEY")
_DUMMY_SESSION_PARTS = ("offline", "session", "marker")


@dataclass(frozen=True)
class RequestValidationEvidence:
    """Evidence that a request matched a botocore service model."""

    fixture_id: str
    service: str
    operation: str
    boto3_version: str
    botocore_version: str
    request_validated: bool = True


@dataclass(frozen=True)
class FixtureStream:
    """Named replay stream for event-stream APIs."""

    fixture_id: str
    events: tuple[Mapping[str, object], ...]

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        """Replay events in fixture order."""
        return iter(self.events)


class StrictStubber:
    """A Stubber wrapper requiring expected params and final checks."""

    def __init__(self, client: Any) -> None:
        """Wrap a real dummy-credential boto3 client."""
        from botocore.stub import Stubber

        self.client = client
        self._stubber = Stubber(client)

    def __enter__(self) -> Self:
        """Activate the underlying Stubber."""
        self._stubber.activate()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Verify pending responses before leaving the context."""
        try:
            self.assert_no_pending_responses()
        finally:
            self._stubber.deactivate()

    def add_response(
        self,
        operation_name: str,
        service_response: Mapping[str, Any],
        *,
        expected_params: Mapping[str, Any],
    ) -> None:
        """Add a response with explicit expected params."""
        api_name = self.client.meta.method_to_api_mapping[operation_name]
        output_shape = self.client.meta.service_model.operation_model(
            api_name
        ).output_shape
        validate_response_timestamps(
            output_shape, service_response, path=api_name
        )
        self._stubber.add_response(
            operation_name,
            service_response,
            expected_params=dict(expected_params),
        )

    def add_client_error(
        self,
        operation_name: str,
        *,
        service_error_code: str,
        service_message: str,
        expected_params: Mapping[str, Any],
        http_status_code: int = 400,
    ) -> None:
        """Add a client error, also requiring expected params."""
        self._stubber.add_client_error(
            operation_name,
            service_error_code=service_error_code,
            service_message=service_message,
            http_status_code=http_status_code,
            expected_params=dict(expected_params),
        )

    def assert_no_pending_responses(self) -> None:
        """Assert every registered response was consumed."""
        self._stubber.assert_no_pending_responses()


def make_client(service_name: str, *, region_name: str = "us-east-1") -> Any:
    """Create a real boto3 client with fixed dummy credentials."""
    import boto3

    session = boto3.Session(
        aws_access_key_id=_DUMMY_ACCESS_KEY_ID,
        aws_secret_access_key="".join(_DUMMY_SECRET_PARTS),
        aws_session_token="-".join(_DUMMY_SESSION_PARTS),
        region_name=region_name,
    )
    return session.client(service_name, region_name=region_name)


def validate_response_timestamps(shape: Any, value: Any, *, path: str) -> None:
    """Require real parsed-SDK timestamp types in response fixtures.

    Botocore accepts strings and epoch numbers when validating Stubber
    responses, although its response parser returns datetime objects.
    Follow the service shape so date-looking string fields stay strings;
    ignore unmodeled response metadata.
    """
    if shape is None:
        return
    if shape.type_name == "timestamp":
        if not isinstance(value, datetime):
            msg = f"{path}: response timestamp fixture requires datetime"
            raise TypeError(msg)
    elif shape.type_name == "structure":
        for name, member in shape.members.items():
            if name in value:
                validate_response_timestamps(
                    member, value[name], path=f"{path}.{name}"
                )
    elif shape.type_name == "list":
        for index, item in enumerate(value):
            validate_response_timestamps(
                shape.member, item, path=f"{path}[{index}]"
            )
    elif shape.type_name == "map":
        for key, item in value.items():
            validate_response_timestamps(
                shape.value, item, path=f"{path}[{key}]"
            )


def stubbed_client(
    service_name: str,
    *,
    region_name: str = "us-east-1",
) -> StrictStubber:
    """Create a strict Stubber around a dummy-credential client."""
    return StrictStubber(make_client(service_name, region_name=region_name))


def validate_operation_request(
    *,
    service_name: str,
    operation_name: str,
    params: Mapping[str, Any],
    fixture_id: str,
) -> RequestValidationEvidence:
    """Validate a request against botocore shapes."""
    from botocore.loaders import Loader
    from botocore.model import ServiceModel
    from botocore.validate import validate_parameters

    service_model = ServiceModel(
        Loader().load_service_model(service_name, "service-2"),
        service_name=service_name,
    )
    operation_model = service_model.operation_model(operation_name)
    input_shape = operation_model.input_shape
    if input_shape is None:
        if params:
            msg = f"{service_name}.{operation_name} does not accept input"
            raise ValueError(msg)
    else:
        validate_parameters(dict(params), input_shape)
    return RequestValidationEvidence(
        fixture_id=fixture_id,
        service=service_name,
        operation=operation_name,
        boto3_version=metadata.version("boto3"),
        botocore_version=metadata.version("botocore"),
    )


def named_fixture_stream(
    *,
    fixture_id: str,
    events: Iterable[Mapping[str, object]],
) -> FixtureStream:
    """Return a named fixture stream for replay."""
    if not fixture_id:
        msg = "fixture_id must be non-empty"
        raise ValueError(msg)
    return FixtureStream(fixture_id=fixture_id, events=tuple(events))
