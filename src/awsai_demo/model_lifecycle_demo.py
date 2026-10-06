"""Amazon Bedrock model catalog demo.

Technology: Amazon Bedrock model catalog.
Lane: models.
Lifecycle refs: bedrock-model-lifecycle, titan-text, nova-v1,
nova-2, bedrock-inference-profiles.
Run:
    uv run awsai-demo model-lifecycle
    uv run awsai-demo model-lifecycle --execution emulator
    uv run awsai-demo model-lifecycle --execution live
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from awsai_demo.demo_support import (
    AwsPort,
    build_default_boto_port,
    build_result,
    fixture_operation,
    local_operation,
    port_operation,
    require_port_not_run,
    unsupported_emulator,
)
from awsai_demo.serialization import normalize_timestamps
from awsai_demo.stubs import stubbed_client

if TYPE_CHECKING:
    from collections.abc import Sequence

    from awsai_demo.contracts import (
        CredentialSource,
        DemoResult,
        OperationOutcome,
    )
    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.policy import ExecutionPolicy
    from awsai_demo.runtime import Execution, Settings

_DEMO = "model-lifecycle"
_TECHNOLOGY = "Amazon Bedrock model catalog"
_REFS = [
    "bedrock-model-lifecycle",
    "titan-text",
    "nova-v1",
    "nova-2",
    "bedrock-inference-profiles",
]
_LIST_FIXTURE = "model-lifecycle-list-foundation-models-v1"
_GET_FIXTURE = "model-lifecycle-get-foundation-model-v1"
_PROFILE_FIXTURE = "model-lifecycle-list-inference-profiles-v1"
_MODEL_ID = "amazon.nova-2-lite-v1:0"
_NOVA_MODEL_ARN = (
    "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-lite-v1:0"
)
_TITAN_MODEL_ARN = (
    "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-lite-v1"
)
_GLOBAL_PROFILE_ARN = (
    "arn:aws:bedrock:us-east-1:123456789012:inference-profile/"
    "global.amazon.nova-2-lite-v1:0"
)
_US_PROFILE_ARN = (
    "arn:aws:bedrock:us-east-1:123456789012:inference-profile/"
    "us.amazon.nova-2-lite-v1:0"
)


def run_model_lifecycle_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    port: AwsPort | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Run the model lifecycle catalog demo."""
    credential_source: CredentialSource | None = None
    if port is None:
        default_port = build_default_boto_port(
            execution=execution,
            settings=settings,
            policy=policy,
            session_factory=session_factory,
            csv_loader=csv_loader,
        )
        if default_port is not None:
            port = default_port.port
            credential_source = default_port.credential_source
    if execution == "offline":
        operations, data = _run_offline()
    elif execution == "emulator":
        operations, data = _run_emulator(settings, port)
    else:
        operations, data = _run_live(settings, port)
    return build_result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="models",
        lifecycle_refs=_REFS,
        execution=execution,
        headline="Model catalog lifecycle statuses were counted.",
        operations=operations,
        settings=settings,
        data=data,
        credential_source=credential_source,
    )


def _run_offline() -> tuple[list[OperationOutcome], dict[str, Any]]:
    operations: list[OperationOutcome] = []
    with stubbed_client("bedrock") as stubber:
        client = stubber.client
        stubber.add_response(
            "list_foundation_models",
            _list_models_response(),
            expected_params={},
        )
        list_response = client.list_foundation_models()
        operations.append(
            fixture_operation(
                service="bedrock",
                operation_name="ListFoundationModels",
                fixture_id=_LIST_FIXTURE,
                effect="read",
            )
        )

        get_params = get_foundation_model_params()
        stubber.add_response(
            "get_foundation_model",
            _get_model_response(),
            expected_params=get_params,
        )
        get_response = client.get_foundation_model(**get_params)
        operations.append(
            fixture_operation(
                service="bedrock",
                operation_name="GetFoundationModel",
                fixture_id=_GET_FIXTURE,
                effect="read",
            )
        )

        stubber.add_response(
            "list_inference_profiles",
            _list_profiles_response(),
            expected_params={},
        )
        profiles_response = client.list_inference_profiles()
        operations.append(
            fixture_operation(
                service="bedrock",
                operation_name="ListInferenceProfiles",
                fixture_id=_PROFILE_FIXTURE,
                effect="read",
            )
        )
    operations.append(
        local_operation(
            service="data/lineage.yaml",
            operation_name="JoinModels",
        ),
    )
    return operations, _catalog_data(
        list_response,
        get_response,
        profiles_response,
    )


def _run_emulator(
    settings: Settings,
    port: AwsPort | None,
) -> tuple[list[OperationOutcome], dict[str, Any]]:
    operations: list[OperationOutcome] = []
    list_payload: Mapping[str, Any] = {"modelSummaries": []}
    get_payload: Mapping[str, Any] = {"modelDetails": {}}
    if port is None:
        operations.extend(
            [
                require_port_not_run(
                    service="bedrock", operation_name="ListFoundationModels"
                ),
                require_port_not_run(
                    service="bedrock", operation_name="GetFoundationModel"
                ),
            ]
        )
    else:
        listed = port_operation(
            port=port,
            service="bedrock",
            operation_name="ListFoundationModels",
            params={},
            execution="emulator",
            effect="read",
            fixture_id=_LIST_FIXTURE,
            endpoint_url=settings.localstack_endpoint,
        )
        operations.append(listed.outcome)
        list_payload = listed.payload
        got = port_operation(
            port=port,
            service="bedrock",
            operation_name="GetFoundationModel",
            params=get_foundation_model_params(_localstack_model_id(settings)),
            execution="emulator",
            effect="read",
            fixture_id=_GET_FIXTURE,
            endpoint_url=settings.localstack_endpoint,
        )
        operations.append(got.outcome)
        get_payload = got.payload
    operations.append(
        unsupported_emulator(
            service="bedrock", operation_name="ListInferenceProfiles"
        )
    )
    operations.append(
        local_operation(
            service="data/lineage.yaml",
            operation_name="JoinModels",
        ),
    )
    return operations, _catalog_data(list_payload, get_payload, {})


def _run_live(
    settings: Settings,
    port: AwsPort | None,
) -> tuple[list[OperationOutcome], dict[str, Any]]:
    operations: list[OperationOutcome] = []
    list_payload: Mapping[str, Any] = {"modelSummaries": []}
    get_payload: Mapping[str, Any] = {"modelDetails": {}}
    profiles_payload: Mapping[str, Any] = {"inferenceProfileSummaries": []}
    if port is None:
        operations.extend(
            [
                require_port_not_run(
                    service="bedrock", operation_name="ListFoundationModels"
                ),
                require_port_not_run(
                    service="bedrock", operation_name="GetFoundationModel"
                ),
                require_port_not_run(
                    service="bedrock", operation_name="ListInferenceProfiles"
                ),
            ]
        )
    else:
        for operation_name, params, fixture_id in (
            ("ListFoundationModels", {}, _LIST_FIXTURE),
            (
                "GetFoundationModel",
                get_foundation_model_params(),
                _GET_FIXTURE,
            ),
            ("ListInferenceProfiles", {}, _PROFILE_FIXTURE),
        ):
            called = port_operation(
                port=port,
                service="bedrock",
                operation_name=operation_name,
                params=params,
                execution="live",
                effect="read",
                fixture_id=fixture_id,
            )
            operations.append(called.outcome)
            if operation_name == "ListFoundationModels":
                list_payload = called.payload
            elif operation_name == "GetFoundationModel":
                get_payload = called.payload
            else:
                profiles_payload = called.payload
    operations.append(
        local_operation(
            service="data/lineage.yaml",
            operation_name="JoinModels",
        ),
    )
    read_success = any(
        item["operation"] == "ListFoundationModels" and item["status"] == "ok"
        for item in operations
    )
    profile_success = any(
        item["operation"] == "ListInferenceProfiles" and item["status"] == "ok"
        for item in operations
    )
    data = (
        _catalog_data(list_payload, get_payload, profiles_payload)
        if read_success
        else {}
    )
    if read_success and not profile_success:
        for key in (
            "global_profiles",
            "us_profiles",
            "eu_profiles",
            "apac_profiles",
        ):
            data.pop(key, None)
    data["region"] = settings.region
    data["read_at"] = datetime.now(UTC).date().isoformat()
    data["catalog_read_succeeded"] = read_success
    return operations, data


def get_foundation_model_params(model_id: str = _MODEL_ID) -> dict[str, str]:
    """Return GetFoundationModel params for the foundation-model id."""
    return {"modelIdentifier": model_id}


def _localstack_model_id(settings: Settings) -> str:
    configured = getattr(settings, "localstack_bedrock_model", None)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    default = settings.default_bedrock_model
    if default.startswith("ollama."):
        return default
    return f"ollama.{default}"


def _catalog_data(
    list_payload: Mapping[str, Any],
    get_payload: Mapping[str, Any],
    profiles_payload: Mapping[str, Any],
) -> dict[str, Any]:
    models = cast(
        "Sequence[Mapping[str, Any]]",
        list_payload.get("modelSummaries", ()),
    )
    providers = {
        str(item.get("providerName", ""))
        for item in models
        if item.get("providerName")
    }
    statuses = [_status(item.get("modelLifecycle")) for item in models]
    profiles = cast(
        "Sequence[Mapping[str, Any]]",
        profiles_payload.get("inferenceProfileSummaries", ()),
    )
    details = cast("Mapping[str, Any]", get_payload.get("modelDetails", {}))
    return {
        "model_count": len(models),
        "active_count": statuses.count("ACTIVE"),
        "legacy_count": statuses.count("LEGACY"),
        "provider_count": len(providers),
        "global_profiles": _profile_prefix_count(profiles, "global."),
        "us_profiles": _profile_prefix_count(profiles, "us."),
        "eu_profiles": _profile_prefix_count(profiles, "eu."),
        "apac_profiles": _profile_prefix_count(profiles, "apac."),
        "foundation_model": details.get("modelId", _MODEL_ID),
        "foundation_lifecycle": normalize_timestamps(
            details.get("modelLifecycle", {})
        ),
        "lineage_model_records": _lineage_model_count(),
    }


def _status(lifecycle: object) -> str:
    if isinstance(lifecycle, Mapping):
        return str(lifecycle.get("status", ""))
    return ""


def _profile_prefix_count(
    profiles: Sequence[Mapping[str, Any]],
    prefix: str,
) -> int:
    return sum(
        1
        for item in profiles
        if str(item.get("inferenceProfileId", "")).startswith(prefix)
    )


def _lineage_model_count() -> int:
    path = Path("data/lineage.yaml")
    if not path.exists():
        return 0
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    records = loaded.get("records", []) if isinstance(loaded, Mapping) else []
    return sum(
        1
        for item in records
        if isinstance(item, Mapping) and item.get("entity_type") == "model"
    )


def _list_models_response() -> dict[str, Any]:
    return {
        "modelSummaries": [
            {
                "modelArn": _NOVA_MODEL_ARN,
                "modelId": _MODEL_ID,
                "modelName": "Nova 2 Lite",
                "providerName": "Amazon",
                "inputModalities": ["TEXT"],
                "outputModalities": ["TEXT"],
                "responseStreamingSupported": True,
                "customizationsSupported": [],
                "inferenceTypesSupported": ["ON_DEMAND"],
                "modelLifecycle": {"status": "ACTIVE"},
            },
            {
                "modelArn": _TITAN_MODEL_ARN,
                "modelId": "amazon.titan-text-lite-v1",
                "modelName": "Titan Text Lite",
                "providerName": "Amazon",
                "inputModalities": ["TEXT"],
                "outputModalities": ["TEXT"],
                "customizationsSupported": [],
                "inferenceTypesSupported": ["ON_DEMAND"],
                "modelLifecycle": {"status": "LEGACY"},
            },
        ]
    }


def _get_model_response() -> dict[str, Any]:
    return {
        "modelDetails": {
            "modelArn": _NOVA_MODEL_ARN,
            "modelId": _MODEL_ID,
            "modelName": "Nova 2 Lite",
            "providerName": "Amazon",
            "inputModalities": ["TEXT"],
            "outputModalities": ["TEXT"],
            "responseStreamingSupported": True,
            "customizationsSupported": [],
            "inferenceTypesSupported": ["ON_DEMAND"],
            "modelLifecycle": {
                "status": "ACTIVE",
                "startOfLifeTime": datetime(2025, 12, 2, tzinfo=UTC),
            },
        }
    }


def _list_profiles_response() -> dict[str, Any]:
    return {
        "inferenceProfileSummaries": [
            {
                "inferenceProfileName": "Global Nova 2 Lite",
                "description": "global routing",
                "createdAt": datetime(2025, 12, 2, tzinfo=UTC),
                "updatedAt": datetime(2025, 12, 2, tzinfo=UTC),
                "inferenceProfileArn": _GLOBAL_PROFILE_ARN,
                "inferenceProfileId": "global.amazon.nova-2-lite-v1:0",
                "models": [{"modelArn": _NOVA_MODEL_ARN}],
                "status": "ACTIVE",
                "type": "SYSTEM_DEFINED",
            },
            {
                "inferenceProfileName": "US Nova 2 Lite",
                "description": "us routing",
                "createdAt": datetime(2025, 12, 2, tzinfo=UTC),
                "updatedAt": datetime(2025, 12, 2, tzinfo=UTC),
                "inferenceProfileArn": _US_PROFILE_ARN,
                "inferenceProfileId": "us.amazon.nova-2-lite-v1:0",
                "models": [{"modelArn": _NOVA_MODEL_ARN}],
                "status": "ACTIVE",
                "type": "SYSTEM_DEFINED",
            },
        ]
    }
