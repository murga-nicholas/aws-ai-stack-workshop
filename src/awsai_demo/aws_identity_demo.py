"""Authentication and authorization are separate AWS checks.

Technology: AWS IAM and STS.
Lane: operations.
Lifecycle refs: iam-sts.
Run: uv run awsai-demo aws-identity
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from awsai_demo.demo_support import build_default_boto_port, build_result
from awsai_demo.service_rows import ServiceRow

if TYPE_CHECKING:
    from awsai_demo.contracts import DemoResult
    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.demo_support import AwsPort
    from awsai_demo.policy import ExecutionPolicy
    from awsai_demo.runtime import Execution, Settings


def principal_type(arn: str) -> str:
    """Expose only the principal category, never the resource name."""
    parts = arn.split(":", maxsplit=5)
    if len(parts) != 6:
        return "unknown"
    return parts[5].split("/", maxsplit=1)[0]


def run_aws_identity_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Probe STS and Bedrock with the same selected identity."""
    default = (
        None
        if port is not None
        else build_default_boto_port(
            execution=execution,
            settings=settings,
            policy=policy,
            session_factory=session_factory,
            csv_loader=csv_loader,
        )
    )
    port = port if default is None else default.port
    rows = (
        ServiceRow(
            "sts",
            "GetCallerIdentity",
            {},
            {
                "Account": "123456789012",
                "Arn": "arn:aws:iam::123456789012:user/offline-user",
                "UserId": "fixture-user",
            },
            emulator=True,
        ),
        ServiceRow(
            "bedrock", "ListFoundationModels", {}, {"modelSummaries": []}
        ),
    )
    identity, probe = [
        row.run(
            execution=execution,
            settings=settings,
            policy=policy,
            port=port,
            command="aws-identity",
        )
        for row in rows
    ]
    arn = str(identity.payload.get("Arn", ""))
    data = {
        "authenticated": identity.outcome["status"] == "ok",
        "principal_type": principal_type(arn),
        "principal_arn": arn or None,
        "bedrock_probe": probe.outcome["status"],
        "bedrock_probe_code": probe.outcome["error_code"],
    }
    return build_result(
        demo="aws-identity",
        technology="AWS IAM and STS",
        lane="operations",
        lifecycle_refs=["iam-sts"],
        execution=execution,
        headline="STS and Bedrock authorization have separate outcomes.",
        operations=[identity.outcome, probe.outcome],
        settings=settings,
        data=data,
        credential_source=None
        if default is None
        else default.credential_source,
    )
