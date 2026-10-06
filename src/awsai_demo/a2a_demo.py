"""Run a local Strands A2A exchange and Agent Registry contracts.

Lane: agents; lifecycle: a2a, agent-registry.
Run: uv run awsai-demo a2a.
"""

from __future__ import annotations

import asyncio
import socket
import time
from threading import Thread
from typing import TYPE_CHECKING, Any, cast

import httpx
import uvicorn
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart
from strands import Agent
from strands.multiagent.a2a import A2AServer

from awsai_demo import scenario
from awsai_demo.contracts import operation
from awsai_demo.demo_support import (
    build_result,
    contract_only,
    fixture_operation,
    not_run_operation,
    port_operation,
    resolve_boto_port,
    unsupported_emulator,
    validate_request,
)
from awsai_demo.network import (
    NetworkPolicy,
    active_policy,
    network_guard,
    registered_loopback_endpoint,
)
from awsai_demo.offline import ScriptedModel, text_script
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.redact import quiet_sdk_logging
from awsai_demo.runtime import Settings
from awsai_demo.stubs import stubbed_client

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from awsai_demo.contracts import (
        CredentialSource,
        DemoResult,
        Execution,
        OperationOutcome,
        Phase,
    )
    from awsai_demo.demo_support import (
        AwsPort,
        BotoPortFactory,
        DefaultBotoPort,
    )

_FIXTURE = "strands-script-a2a-v1"


def build_a2a_server(port: int) -> A2AServer:
    """Build a real A2AServer using only a scripted provider."""

    def agent_factory(context_id: str) -> Agent:
        del context_id
        return Agent(
            name="Pilot adviser",
            description="Prices a pilot; requires approval.",
            model=ScriptedModel(
                text_script(
                    "The pilot costs 19680 USD. Human approval is required."
                ),
                fixture_id=_FIXTURE,
            ),
            callback_handler=None,
            retry_strategy=None,
        )

    return A2AServer(
        agent_factory=agent_factory,
        host="127.0.0.1",
        port=port,
        enable_a2a_compliant_streaming=True,
    )


async def exchange(endpoint: str) -> dict[str, Any]:
    """Discover the card and send a task through the A2A SDK."""
    async with httpx.AsyncClient(timeout=10, trust_env=False) as http:
        card = await A2ACardResolver(http, endpoint).get_agent_card()
        client = ClientFactory(
            ClientConfig(
                streaming=False,
                httpx_client=http,
            )
        ).create(card)
        message = Message(
            message_id="awsai-pilot-task",
            role=Role.user,
            parts=[Part(root=TextPart(text=scenario.BRIEF))],
        )
        updates = []
        async for item in client.send_message(message):
            task = item[0] if isinstance(item, tuple) else item
            updates.append(task.model_dump(mode="json", exclude_none=True))
        return {
            "card_name": card.name,
            "card_url": card.url,
            "task_updates": updates,
        }


def local_exchange(
    *,
    exchanger: Callable[[str], Coroutine[Any, Any, dict[str, Any]]]
    | None = None,
) -> tuple[str, dict[str, Any]]:
    """Serve on registered loopback and close the listener afterward.

    exchanger replaces the A2A client exchange in tests.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        endpoint = f"http://127.0.0.1:{port}"
        guard = (active_policy() or NetworkPolicy()).with_endpoint(
            registered_loopback_endpoint("a2a", "127.0.0.1", port)
        )
        server = uvicorn.Server(
            uvicorn.Config(
                build_a2a_server(port).to_starlette_app(),
                log_config=None,
                access_log=False,
                lifespan="off",
            )
        )
        thread = Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        with network_guard(guard):
            thread.start()
            try:
                wait_started(server)
                send = exchange if exchanger is None else exchanger
                data: dict[str, Any] = asyncio.run(send(endpoint))
            finally:
                server.should_exit = True
                thread.join(timeout=10)
    return endpoint, data


def wait_started(server: uvicorn.Server, *, timeout: float = 10) -> None:
    """Bound readiness waiting without sending speculative requests."""
    deadline = time.monotonic() + timeout
    while not server.started:
        if time.monotonic() >= deadline:
            message = "Local A2A server did not start within its deadline"
            raise TimeoutError(message)
        time.sleep(0.01)


@quiet_sdk_logging()
def run_a2a_demo(
    *,
    execution: Execution = "offline",
    settings: Settings | None = None,
    policy: ExecutionPolicy | None = None,
    port: AwsPort | None = None,
    boto_port_factory: BotoPortFactory | None = None,
) -> DemoResult:
    """Keep local protocol execution separate from registry requests."""
    settings = settings or Settings()
    operations: list[OperationOutcome] = []
    data: dict[str, Any] = {}
    source: CredentialSource = "none"
    registry_params = {
        "registryIds": [settings.registry_id or "registry-demo"],
        "searchQuery": "customer support",
        "maxResults": 10,
    }
    if execution == "offline":
        endpoint, data = local_exchange()
        for name, phase in (
            ("A2AServer", "setup"),
            ("GET /.well-known/agent-card.json", "main"),
            ("message/send", "main"),
        ):
            operations.append(
                operation(
                    service="a2a",
                    operation=name,
                    phase=cast("Phase", phase),
                    execution_target="local",
                    mode="local_execution",
                    transport="loopback",
                    endpoint_url=endpoint,
                    response_received=True,
                )
            )
        operations.append(
            fixture_operation(
                service="strands",
                operation_name="ScriptedModel.stream",
                fixture_id=_FIXTURE,
                effect="none",
                request_validated=False,
            )
        )
        with stubbed_client("agent-registry") as stubber:
            stubber.add_response(
                "search_discoverable_registry_records",
                {"registryRecords": []},
                expected_params=registry_params,
            )
            stubber.client.search_discoverable_registry_records(
                **registry_params
            )
        operations.append(
            fixture_operation(
                service="agent-registry",
                operation_name="SearchDiscoverableRegistryRecords",
                fixture_id="a2a-registry-v1",
                effect="read",
            )
        )
        validate_request(
            service="bedrock-agentcore",
            operation_name="GetAgentCard",
            params={
                "agentRuntimeArn": "arn:aws:bedrock-agentcore:us-east-1:"
                "000000000000:runtime/example-1234567890"
            },
            fixture_id="a2a-get-card-v1",
        )
        operations.append(
            fixture_operation(
                service="bedrock-agentcore",
                operation_name="GetAgentCard",
                fixture_id="a2a-get-card-v1",
                effect="none",
            )
        )
    else:
        operations.append(
            contract_only(
                service="bedrock-agentcore",
                operation_name="GetAgentCard",
            )
        )
        name = "SearchDiscoverableRegistryRecords"
        if execution == "emulator":
            operations.append(
                unsupported_emulator(
                    service="agent-registry",
                    operation_name=name,
                )
            )
        elif settings.registry_id is None:
            operations.append(
                not_run_operation(
                    service="agent-registry",
                    operation_name=name,
                    error_code="missing_configuration",
                    request_validated=False,
                    optional=True,
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
                port = default.port
                source = default.credential_source
            call = port_operation(
                port=port,
                service="agent-registry",
                operation_name=name,
                params=registry_params,
                execution=execution,
                effect="read",
                fixture_id="a2a-registry-v1",
                endpoint_url=f"https://agent-registry.{settings.region}.amazonaws.com",
            )
            operations.append(call.outcome)
            data["registry_records"] = len(
                call.payload.get("registryRecords", [])
            )
    return build_result(
        demo="a2a",
        technology="A2A and AWS Agent Registry",
        lane="agents",
        lifecycle_refs=("a2a", "agent-registry"),
        execution=execution,
        settings=settings,
        operations=operations,
        credential_source=source,
        headline="Local A2A discovery and a scripted agent task.",
        data=data,
    )
