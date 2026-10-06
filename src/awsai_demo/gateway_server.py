"""Local stdio MCP server for the AgentCore Gateway demo.

Lane: agents; lifecycle: agentcore-gateway, agentcore-policy.
Run: uv run awsai-demo agentcore-gateway --execution offline.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mcp.server.mcpserver import MCPServer

from awsai_demo import scenario


def build_server() -> MCPServer:
    """Build the local MCP server without opening sockets."""
    server = MCPServer(
        "awsai-demo-gateway",
        title="AWS AI demo gateway",
        description="Local MCP tools for the workshop pilot scenario.",
    )
    server.add_tool(price_pilot, name="price_pilot")
    server.add_tool(accept_proposal, name="accept_proposal")
    return server


def price_pilot() -> dict[str, Any]:
    """Return the deterministic pilot price and proposal version."""
    cost = scenario.price_pilot()
    payload = asdict(cost)
    payload["proposal_version"] = scenario.proposal_version(cost)
    return payload


def accept_proposal(
    *,
    approved: bool,
    proposal_version: str,
    approver: str,
) -> dict[str, object]:
    """Return an acceptance record for a matching approved proposal."""
    cost = scenario.price_pilot()
    expected = scenario.proposal_version(cost)
    accepted = approved and proposal_version == expected
    return {
        "accepted": accepted,
        "approver": approver if accepted else None,
        "proposal_version": proposal_version,
        "expected_version": expected,
    }


def main() -> None:
    """Run the server over stdio for Strands MCPClient."""
    build_server().run("stdio")


if __name__ == "__main__":
    main()
