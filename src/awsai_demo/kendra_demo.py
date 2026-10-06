"""Kendra maintenance lineage and explicit retrieval-filter migration.

Technology: Amazon Kendra and Managed Knowledge Bases.
Lane: data.
Lifecycle refs: kendra, managed-knowledge-base.
Run: uv run awsai-demo kendra
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from awsai_demo.demo_support import (
    build_default_boto_port,
    build_result,
    local_operation,
)
from awsai_demo.service_rows import ServiceRow

if TYPE_CHECKING:
    from collections.abc import Mapping

    from awsai_demo.contracts import DemoResult
    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.demo_support import AwsPort
    from awsai_demo.policy import ExecutionPolicy
    from awsai_demo.runtime import Execution, Settings


def _validate_filter(kendra: Mapping[str, Any]) -> None:
    if set(kendra) != {"EqualsTo"}:
        message = "Only EqualsTo is supported by this example"
        raise ValueError(message)
    if set(kendra["EqualsTo"]["Value"]) != {"StringValue"}:
        message = "The example requires a StringValue"
        raise ValueError(message)


# slide: translate
def translate_filter(kendra: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the demonstrated string equality filter to BMKB."""
    _validate_filter(kendra)
    match = kendra["EqualsTo"]
    key, value = match["Key"], match["Value"]["StringValue"]
    return {"equals": {"key": key, "value": value}}


# end-slide: translate


def run_kendra_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Validate Retrieve and show the string equality filter mapping."""
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
    attribute = {
        "EqualsTo": {"Key": "department", "Value": {"StringValue": "support"}}
    }
    rows = (
        ServiceRow(
            "kendra",
            "Retrieve",
            {
                "IndexId": "00000000-0000-4000-8000-000000000001",
                "QueryText": "support pilot budget",
                "AttributeFilter": attribute,
                "PageSize": 1,
            },
            {"ResultItems": []},
            effect="none",
            live=False,
        ),
        ServiceRow(
            "kendra",
            "ListIndices",
            {"MaxResults": 10},
            {"IndexConfigurationSummaryItems": []},
        ),
    )
    responses = [
        row.run(
            execution=execution,
            settings=settings,
            policy=policy,
            port=port,
            command="kendra",
        )
        for row in rows
    ]
    operations = [response.outcome for response in responses]
    data: dict[str, Any] = {
        "indices_listed": len(
            responses[1].payload.get("IndexConfigurationSummaryItems", [])
        )
    }
    if responses[1].aws_error_code:
        data["list_indices_code"] = responses[1].aws_error_code
    if execution != "emulator":
        data["translated_request"] = {
            "knowledgeBaseId": "ABCDEFGHIJ",
            "retrievalQuery": {"text": "support pilot budget"},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "filter": translate_filter(attribute),
                    "numberOfResults": 1,
                }
            },
        }
        from awsai_demo.demo_support import validate_request

        validate_request(
            service="bedrock-agent-runtime",
            operation_name="Retrieve",
            params=data["translated_request"],
            fixture_id="kendra-bmkb-filter-v1",
        )
        operations.append(
            local_operation(
                service="local", operation_name="TranslateRetrievalFilter"
            )
        )
    return build_result(
        demo="kendra",
        technology="Amazon Kendra",
        lane="data",
        lifecycle_refs=["kendra", "managed-knowledge-base"],
        execution=execution,
        headline="A string equality filter maps to Bedrock retrieval.",
        operations=operations,
        settings=settings,
        data=data,
        credential_source=None
        if default is None
        else default.credential_source,
    )
