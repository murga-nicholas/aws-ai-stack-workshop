"""Serve and consume the scenario tools over local MCP stdio.

Lane: agents; lifecycle: mcp, aws-mcp-server.
Run: uv run awsai-demo mcp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from awsai_demo.agentcore_gateway_demo import run_mcp_round_trip
from awsai_demo.demo_support import (
    build_result,
    contract_only,
    local_operation,
)
from awsai_demo.runtime import Settings

if TYPE_CHECKING:
    from awsai_demo.contracts import DemoResult, Execution
    from awsai_demo.policy import ExecutionPolicy


def run_mcp_demo(
    *,
    execution: Execution = "offline",
    settings: Settings | None = None,
    policy: ExecutionPolicy | None = None,
) -> DemoResult:
    """Exercise the actual MCP server and client without AWS calls."""
    del policy
    data: dict[str, object] = {
        "managed_server": "AWS MCP Server is described, not called.",
    }
    if execution == "offline":
        data.update(run_mcp_round_trip())
        operations = [
            local_operation(service="mcp", operation_name="MCPServer stdio"),
            local_operation(
                service="strands", operation_name="MCPClient list/call"
            ),
        ]
    else:
        operations = [
            contract_only(service="mcp", operation_name="MCPServer stdio")
        ]
    return build_result(
        demo="mcp",
        technology="Model Context Protocol",
        lane="agents",
        lifecycle_refs=("mcp", "aws-mcp-server"),
        execution=execution,
        settings=settings or Settings(),
        headline="MCP tools use a real local stdio round trip.",
        operations=operations,
        data=data,
    )
