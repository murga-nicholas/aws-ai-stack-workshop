from __future__ import annotations

import re
from pathlib import Path

import pytest

from awsai_demo.registry import get_demo

MARKERS = (
    ("bedrock-runtime", "converse-request"),
    ("strands-agent", "agent"),
    ("agentcore-harness", "create-harness"),
    ("agentcore-gateway", "cedar-policy"),
    ("decision", "interrupt"),
    ("knowledge-bases", "untrusted"),
    ("bedrock-openai", "openai-client"),
    ("guardrails", "apply-guardrail"),
    ("agentcore-runtime", "app"),
)
REQUIRED_CODE = {
    "openai-client": ("OpenAI(", "api_key=", "chat.completions.create("),
    "apply-guardrail": ("apply_guardrail(", '["action"]'),
    "app": (
        "BedrockAgentCoreApp()",
        "@app.entrypoint",
        "invoke_runtime_agent(",
    ),
}


@pytest.mark.parametrize(("demo", "marker"), MARKERS)
def test_core_slide_markers_fit_the_source_viewport(
    demo: str, marker: str
) -> None:
    spec = get_demo(demo)
    filename = spec.module.rsplit(".", 1)[-1] + ".py"
    path = (
        Path(__file__).resolve().parents[1] / "src" / "awsai_demo" / filename
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    start = [
        i
        for i, line in enumerate(lines)
        if line.strip() == f"# slide: {marker}"
    ]
    end = [
        i
        for i, line in enumerate(lines)
        if line.strip() == f"# end-slide: {marker}"
    ]
    assert len(start) == len(end) == 1, path
    assert end[0] > start[0]
    snippet = lines[start[0] + 1 : end[0]]
    assert 0 < len(snippet) <= 14, (path, len(snippet))
    nonempty = [line for line in snippet if line.strip()]
    indent = min(len(line) - len(line.lstrip()) for line in nonempty)
    assert all(len(line[indent:]) <= 72 for line in snippet), path
    assert re.search(r"\w", "\n".join(snippet))
    for required in REQUIRED_CODE.get(marker, ()):
        assert required in "\n".join(snippet), (path, required)
