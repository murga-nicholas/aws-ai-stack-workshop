"""Keep tests offline and detached from the machine's credentials."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use dummy values and disable real identity discovery."""
    for name in (
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_CREDS_FILE_PATH",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_ENDPOINT_URL",
        "OPENAI_API_KEY",
        "LOCALSTACK_AUTH_TOKEN",
        "LOCALSTACK_ENDPOINT",
        "DEFAULT_BEDROCK_MODEL",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in {
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_SESSION_TOKEN": "test",
        "AWS_EC2_METADATA_DISABLED": "true",
    }.items():
        monkeypatch.setenv(name, value)


@pytest.fixture(autouse=True)
def offline_network(request: pytest.FixtureRequest) -> Iterator[None]:
    """Deny egress outside the guard's own activation tests."""
    from awsai_demo.network import NetworkPolicy, network_guard

    if request.node.path.name == "test_network.py":
        yield
    else:
        with network_guard(NetworkPolicy()):
            yield
