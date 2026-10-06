from awsai_demo.mcp_demo import run_mcp_demo
from awsai_demo.runtime import Settings


def test_mcp_uses_actual_stdio_client_and_server() -> None:
    result = run_mcp_demo()
    assert result["status"] == "ok"
    assert result["mode"] == "local_execution"
    assert result["data"]["tool_names"] == ("accept_proposal", "price_pilot")
    assert result["data"]["price_total"] == 19680
    assert all(op["transport"] == "none" for op in result["operations"])
    assert not result["evidence"]["aws_executed"]


def test_mcp_managed_endpoint_is_not_invoked_live() -> None:
    result = run_mcp_demo(execution="live", settings=Settings())
    assert result["mode"] == "not_run"
    assert result["operations"][0]["error_code"] == "contract_only"
    assert not result["evidence"]["network_attempted"]
