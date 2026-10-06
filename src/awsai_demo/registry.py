"""The complete workshop catalogue, independent of demo imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type Lane = Literal[
    "models", "agents", "data", "platform", "operations", "local"
]
LANES: tuple[Lane, ...] = (
    "models",
    "agents",
    "data",
    "platform",
    "operations",
    "local",
)


@dataclass(frozen=True)
class DemoSpec:
    """Catalogue metadata that needs no SDK imports."""

    name: str
    technology: str
    lane: Lane
    lifecycle_refs: tuple[str, ...]
    core: bool = False

    @property
    def module(self) -> str:
        """Return the lazily imported implementation module name."""
        return "awsai_demo." + self.name.replace("-", "_") + "_demo"

    @property
    def entrypoint(self) -> str:
        """Return the public function expected in that module."""
        return "run_" + self.name.replace("-", "_") + "_demo"


DEMOS: tuple[DemoSpec, ...] = (
    DemoSpec(
        "bedrock-runtime",
        "Bedrock inference",
        "models",
        ("bedrock", "converse-api", "invoke-model-api"),
        True,
    ),
    DemoSpec(
        "bedrock-openai",
        "Bedrock OpenAI-compatible APIs",
        "models",
        ("bedrock-mantle", "bedrock-runtime-openai", "bedrock-api-keys"),
    ),
    DemoSpec(
        "model-lifecycle",
        "Bedrock model lifecycle",
        "models",
        (
            "bedrock-model-lifecycle",
            "titan-text",
            "nova-v1",
            "nova-2",
            "bedrock-inference-profiles",
        ),
        True,
    ),
    DemoSpec(
        "guardrails",
        "Bedrock Guardrails",
        "models",
        ("guardrails", "comprehend-prompt-safety"),
    ),
    DemoSpec(
        "bedrock-flows",
        "Prompts and Flows",
        "models",
        ("bedrock-flows", "prompt-management"),
    ),
    DemoSpec("nova-act", "Amazon Nova Act", "models", ("nova-act",)),
    DemoSpec(
        "model-customization",
        "Model customization",
        "models",
        ("model-customization",),
    ),
    DemoSpec(
        "agents-classic",
        "Bedrock Agents Classic",
        "agents",
        ("agents-classic", "agents-classic-multi-agent"),
    ),
    DemoSpec(
        "strands-agent", "Strands Agents", "agents", ("strands-agents",), True
    ),
    DemoSpec(
        "strands-multiagent",
        "Strands multi-agent",
        "agents",
        (
            "strands-multiagent",
            "agents-classic-multi-agent",
            "agent-squad",
            "multi-agent-orchestrator",
        ),
    ),
    DemoSpec(
        "agentcore-runtime",
        "AgentCore Runtime",
        "agents",
        ("agentcore-runtime",),
    ),
    DemoSpec(
        "agentcore-harness",
        "AgentCore harness",
        "agents",
        ("agentcore-harness", "agents-classic"),
        True,
    ),
    DemoSpec(
        "agentcore-memory", "AgentCore Memory", "agents", ("agentcore-memory",)
    ),
    DemoSpec(
        "agentcore-gateway",
        "AgentCore Gateway and Policy",
        "agents",
        ("agentcore-gateway", "agentcore-policy"),
        True,
    ),
    DemoSpec(
        "agentcore-identity",
        "AgentCore Identity",
        "agents",
        ("agentcore-identity",),
    ),
    DemoSpec(
        "agentcore-tools",
        "AgentCore tools",
        "agents",
        (
            "agentcore-code-interpreter",
            "agentcore-browser",
            "agentcore-web-search",
            "agentcore-payments",
        ),
    ),
    DemoSpec(
        "mcp", "Model Context Protocol", "agents", ("mcp", "aws-mcp-server")
    ),
    DemoSpec(
        "a2a", "A2A and Agent Registry", "agents", ("a2a", "agent-registry")
    ),
    DemoSpec(
        "decision",
        "Human-approved pilot decision",
        "agents",
        ("strands-agents", "agentcore-harness", "agents-classic"),
        True,
    ),
    DemoSpec(
        "knowledge-bases",
        "Bedrock Knowledge Bases",
        "data",
        ("knowledge-bases", "managed-knowledge-base", "kendra"),
        True,
    ),
    DemoSpec("kendra", "Amazon Kendra", "data", ("kendra",)),
    DemoSpec("s3-vectors", "Amazon S3 Vectors", "data", ("s3-vectors",)),
    DemoSpec(
        "ai-services",
        "AWS AI services",
        "data",
        (
            "comprehend",
            "translate",
            "polly",
            "transcribe",
            "textract",
            "rekognition-maintenance-features",
            "bedrock-data-automation",
        ),
    ),
    DemoSpec(
        "sagemaker-ai",
        "Amazon SageMaker AI",
        "platform",
        (
            "sagemaker-ai",
            "sagemaker-ai-maintenance-features",
            "sagemaker-profiler",
        ),
    ),
    DemoSpec("aws-identity", "AWS IAM and STS", "operations", ("iam-sts",)),
    DemoSpec(
        "evaluation",
        "Agent evaluations",
        "operations",
        ("agentcore-evaluations", "agentcore-optimization"),
    ),
    DemoSpec(
        "observability",
        "Agent observability",
        "operations",
        ("agentcore-observability",),
    ),
    DemoSpec(
        "cost",
        "Cost per successful task",
        "operations",
        ("bedrock-service-tiers", "bedrock-inference-profiles"),
    ),
    DemoSpec(
        "localstack",
        "LocalStack core",
        "local",
        ("localstack-auth-token", "localstack-community-image"),
        True,
    ),
    DemoSpec(
        "localstack-ai",
        "LocalStack AI services",
        "local",
        ("localstack-bedrock", "localstack-ai-services"),
    ),
)


def list_demos(lane: str | None = None) -> tuple[DemoSpec, ...]:
    """List every registered demo, optionally restricted to one lane."""
    if lane is not None and lane not in LANES:
        message = f"Unknown lane: {lane}"
        raise ValueError(message)
    return tuple(demo for demo in DEMOS if lane is None or demo.lane == lane)


def get_demo(name: str) -> DemoSpec:
    """Find a demo without importing its implementation."""
    for demo in DEMOS:
        if demo.name == name:
            return demo
    message = f"Unknown demo: {name}"
    raise ValueError(message)
