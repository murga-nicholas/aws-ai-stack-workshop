"""Bedrock OpenAI-compatible API demo.

Technology: Amazon Bedrock OpenAI-compatible APIs.
Lane: models.
Lifecycle refs: bedrock-mantle, bedrock-runtime-openai,
bedrock-api-keys.
Run:
    uv run awsai-demo bedrock-openai
    uv run awsai-demo bedrock-openai --execution live
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module, metadata
from typing import TYPE_CHECKING, Any, Protocol, cast

import httpx

from awsai_demo import billing
from awsai_demo.contracts import (
    CredentialSource,
    DemoResult,
    ErrorCode,
    Evidence,
    Execution,
    OperationOutcome,
    Usage,
    evidence,
    operation,
    result,
)
from awsai_demo.credentials import SelectedSession, select_session
from awsai_demo.demo_support import (
    AwsPort,
    build_default_boto_port,
    not_run_operation,
    port_operation,
)
from awsai_demo.policy import (
    ExecutionPolicy,
    PolicyError,
    PricedBudget,
    Reservation,
    TokenCounter,
)
from awsai_demo.redact import sanitize_exception

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from awsai_demo.credentials import CsvLoader, SessionFactory
    from awsai_demo.runtime import Settings

_DEMO = "bedrock-openai"
_TECHNOLOGY = "Bedrock OpenAI-compatible APIs"
_REFS = ["bedrock-mantle", "bedrock-runtime-openai", "bedrock-api-keys"]
_TOKEN_FIXTURE = "bedrock-openai-token-v1"  # noqa: S105
_CHAT_FIXTURE = "bedrock-openai-chat-completions-v1"
_RESPONSES_FIXTURE = "bedrock-openai-responses-v1"
_CATALOG_FIXTURE = "bedrock-openai-list-foundation-models-v1"
_DEFAULT_CHAT_MODEL = "openai.gpt-oss-20b-1:0"
_DEFAULT_RESPONSES_MODEL = "us.openai.gpt-5.6-sol/global"
_OPENAI_API_SUPPORT_SOURCE = (
    "https://docs.aws.amazon.com/bedrock/latest/userguide/"
    "inference-responses-api.html"
)
_OPENAI_API_SUPPORT = {
    "openai.gpt-oss-20b-1:0": frozenset({"CHAT_COMPLETIONS"}),
    "openai.gpt-oss-120b-1:0": frozenset({"CHAT_COMPLETIONS"}),
}


@dataclass(frozen=True)
class OpenAITextResult:
    """Normalized text and token usage from an OpenAI call."""

    text: str
    usage: Usage | None


class OpenAIClientPort(Protocol):
    """Small OpenAI-compatible surface used by the demo."""

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
    ) -> OpenAITextResult:
        """Run Chat Completions and return normalized data."""

    def response(
        self,
        *,
        model: str,
        input_text: str,
        max_output_tokens: int,
    ) -> OpenAITextResult:
        """Run Responses and return normalized data."""


class OpenAIClientFactory(Protocol):
    """Factory that creates a configured OpenAI SDK wrapper."""

    def __call__(
        self,
        *,
        api_key: str,
        base_url: str,
        policy: ExecutionPolicy,
    ) -> OpenAIClientPort:
        """Create an OpenAI-compatible client."""


class TokenProvider(Protocol):
    """Callable that mints a short-term Bedrock API key."""

    def __call__(self, selected: SelectedSession, region: str) -> str:
        """Return a bearer token without contacting AWS services."""


class OpenAISdkPort:
    """Adapter around the real OpenAI Python SDK."""

    def __init__(
        self,
        client: object,
        *,
        chat_create: Callable[..., object] | None = None,
    ) -> None:
        """Store an SDK client."""
        self._client = client
        self._chat_create = chat_create

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
    ) -> OpenAITextResult:
        """Run Chat Completions through the SDK."""
        creator = self._chat_create
        if creator is None:
            creator = cast("Any", self._client).chat.completions.create
        raw_chat = creator(
            model=model,
            messages=[dict(item) for item in messages],
            max_tokens=max_tokens,
        )
        chat = cast("Any", raw_chat)
        choice = chat.choices[0]
        text = str(choice.message.content or "")
        return OpenAITextResult(text=text, usage=_chat_usage(chat.usage))

    def response(
        self,
        *,
        model: str,
        input_text: str,
        max_output_tokens: int,
    ) -> OpenAITextResult:
        """Run Responses through the SDK."""
        response = cast("Any", self._client).responses.create(
            model=model,
            input=input_text,
            max_output_tokens=max_output_tokens,
        )
        return OpenAITextResult(
            text=str(response.output_text),
            usage=_responses_usage(response.usage),
        )


class StaticTokenCounter:
    """Exact counter for tests or pre-measured fixtures."""

    def __init__(self, tokens: int) -> None:
        """Store one deterministic token count."""
        self.tokens = tokens

    def count_tokens(
        self,
        model_id: str,
        serialized_request: bytes,
    ) -> int:
        """Return the configured count."""
        del model_id, serialized_request
        return self.tokens


def run_bedrock_openai_demo(
    *,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy,
    client_factory: OpenAIClientFactory | None = None,
    budget: PricedBudget | None = None,
    catalog_port: AwsPort | None = None,
    token_provider: TokenProvider | None = None,
    exact_counter: TokenCounter | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> DemoResult:
    """Run the Bedrock OpenAI-compatible API demo."""
    chat_model = _chat_model(settings)
    responses_model = _responses_model(settings)
    if execution == "offline":
        operations, data = _run_offline(
            policy=policy,
            client_factory=client_factory or _mock_client_factory,
            chat_model=chat_model,
            responses_model=responses_model,
        )
        credential_source: CredentialSource = "none"
    elif execution == "emulator":
        operations, data = _run_emulator()
        credential_source = "none"
    else:
        operations, data, credential_source = _run_live(
            settings=settings,
            policy=policy,
            client_factory=client_factory or default_openai_client,
            budget=budget,
            catalog_port=catalog_port,
            token_provider=token_provider or generate_bedrock_api_key,
            exact_counter=exact_counter,
            session_factory=session_factory,
            csv_loader=csv_loader,
            chat_model=chat_model,
            responses_model=responses_model,
        )
    evidence_model = _evidence_model(data, chat_model)
    return result(
        demo=_DEMO,
        technology=_TECHNOLOGY,
        lane="models",
        lifecycle_refs=_REFS,
        requested_execution=execution,
        headline="OpenAI SDK calls are pointed at Bedrock Runtime.",
        operations=operations,
        evidence=_openai_evidence(
            operations,
            settings=settings,
            credential_source=credential_source,
            requested_model=evidence_model,
            observed_model=_observed_model(operations, evidence_model),
        ),
        data=data,
    )


def default_openai_client(
    *,
    api_key: str,
    base_url: str,
    policy: ExecutionPolicy,
    http_client: object | None = None,
) -> OpenAIClientPort:
    """Create the production OpenAI SDK adapter for Bedrock."""
    from openai import OpenAI

    # slide: openai-client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        max_retries=policy.openai_max_retries,
        timeout=min(policy.max_wall_seconds, 30),
        http_client=cast("Any", http_client),
    )

    def chat_request(**kwargs: object) -> object:
        return client.chat.completions.create(  # type: ignore
            **kwargs,
        )

    # end-slide: openai-client
    return OpenAISdkPort(client, chat_create=chat_request)


def generate_bedrock_api_key(selected: SelectedSession, region: str) -> str:
    """Generate a short-term Bedrock API key from a selected session."""
    module = import_module("aws_bedrock_token_generator")
    generator_type = cast("Any", module).BedrockTokenGenerator
    credentials = _credentials_from_session(selected.session)
    return str(generator_type().get_token(credentials, region))


def chat_completion_request(
    model: str,
    *,
    max_tokens: int = 64,
) -> dict[str, object]:
    """Return the Chat Completions request body."""
    return {"model": model, "messages": _messages(), "max_tokens": max_tokens}


def responses_request(
    model: str,
    *,
    max_output_tokens: int = 64,
) -> dict[str, object]:
    """Return the Responses request body."""
    return {
        "model": model,
        "input": "Summarize why Bedrock API keys help migrations.",
        "max_output_tokens": max_output_tokens,
    }


def bedrock_openai_base_url(region: str) -> str:
    """Return the recommended OpenAI-compatible Bedrock Runtime URL."""
    return f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1"


def _run_offline(
    *,
    policy: ExecutionPolicy,
    client_factory: OpenAIClientFactory,
    chat_model: str,
    responses_model: str,
) -> tuple[list[OperationOutcome], dict[str, object]]:
    base_url = bedrock_openai_base_url("us-east-1")
    output_limit = _request_limit(policy)
    client = client_factory(
        api_key="bedrock-api-key-offline-fixture",
        base_url=base_url,
        policy=policy,
    )
    chat = client.chat_completion(
        model=chat_model,
        messages=cast("Sequence[Mapping[str, str]]", _messages()),
        max_tokens=output_limit,
    )
    response_body = responses_request(
        responses_model,
        max_output_tokens=output_limit,
    )
    response = client.response(
        model=responses_model,
        input_text=str(response_body["input"]),
        max_output_tokens=output_limit,
    )
    operations = [
        _token_fixture_operation(),
        _openai_fixture_operation(
            operation_name="chat.completions.create",
            fixture_id=_CHAT_FIXTURE,
            usage=chat.usage,
        ),
        _openai_fixture_operation(
            operation_name="responses.create",
            fixture_id=_RESPONSES_FIXTURE,
            usage=response.usage,
        ),
    ]
    return operations, {
        "base_url": base_url,
        "chat_model": chat_model,
        "responses_model": responses_model,
        "chat_text": chat.text,
        "responses_text": response.text,
        "auth_source": "fixture_token",
    }


def _run_emulator() -> tuple[list[OperationOutcome], dict[str, object]]:
    operations = [
        not_run_operation(
            service="aws-bedrock-token-generator",
            operation_name="provide_token",
            phase="setup",
            error_code="not_supported_by_emulator",
            request_validated=False,
        ),
        not_run_operation(
            service="openai",
            operation_name="chat.completions.create",
            error_code="not_supported_by_emulator",
            request_validated=False,
        ),
        not_run_operation(
            service="openai",
            operation_name="responses.create",
            error_code="not_supported_by_emulator",
            request_validated=False,
        ),
    ]
    return operations, {"emulator": "not_supported"}


def _run_live(
    *,
    settings: Settings,
    policy: ExecutionPolicy,
    client_factory: OpenAIClientFactory,
    budget: PricedBudget | None,
    catalog_port: AwsPort | None,
    token_provider: TokenProvider,
    exact_counter: TokenCounter | None,
    session_factory: SessionFactory | None,
    csv_loader: CsvLoader | None,
    chat_model: str,
    responses_model: str,
) -> tuple[list[OperationOutcome], dict[str, object], CredentialSource]:
    base_url = bedrock_openai_base_url(settings.region)
    operations: list[OperationOutcome] = []
    data: dict[str, object] = {
        "base_url": base_url,
        "chat_model": chat_model,
        "responses_model": responses_model,
        "capability_source": _OPENAI_API_SUPPORT_SOURCE,
    }
    selected: SelectedSession | None = None
    api_key: str | None = None
    credential_source: CredentialSource = "none"
    if catalog_port is None:
        default_port = cast(
            "Any",
            build_default_boto_port(
                execution="live",
                settings=settings,
                policy=policy,
                session_factory=session_factory,
                csv_loader=csv_loader,
            ),
        )
        catalog_port = default_port.port
        credential_source = default_port.credential_source
        selected = SelectedSession(
            session=default_port.port.session,
            source=default_port.credential_source,
            region=default_port.port.region_name,
        )
    catalog_outcome, catalog_payload = _list_openai_catalog(
        settings=settings,
        catalog_port=catalog_port,
    )
    operations.append(catalog_outcome)
    selected_chat_model = _select_live_model(
        settings=settings,
        catalog_payload=catalog_payload,
        api="CHAT_COMPLETIONS",
    )
    selected_responses_model = _select_live_model(
        settings=settings,
        catalog_payload=catalog_payload,
        api="RESPONSES",
    )
    data["chat_model"] = selected_chat_model or chat_model
    data["responses_model"] = selected_responses_model or responses_model

    for name, fixture_id, model, api in (
        (
            "chat.completions.create",
            _CHAT_FIXTURE,
            selected_chat_model,
            "CHAT_COMPLETIONS",
        ),
        (
            "responses.create",
            _RESPONSES_FIXTURE,
            selected_responses_model,
            "RESPONSES",
        ),
    ):
        if model is None:
            candidate = settings.model or selected_chat_model
            known_gap = _known_capability_gap(candidate, api, catalog_payload)
            operations.append(
                _model_unavailable_operation(
                    name,
                    "not_supported_by_model"
                    if known_gap
                    else "model_unavailable",
                )
            )
            if known_gap:
                gaps = cast(
                    "dict[str, object]", data.setdefault("capability_gaps", {})
                )
                gaps[name] = {"model": candidate, "api": api}
            continue
        output_limit = _request_limit(policy)
        request_body = (
            chat_completion_request(model, max_tokens=output_limit)
            if name == "chat.completions.create"
            else responses_request(model, max_output_tokens=output_limit)
        )
        outcome, state = _reserve_and_call_model(
            operation_name=name,
            fixture_id=fixture_id,
            model=model,
            request_body=request_body,
            settings=settings,
            policy=policy,
            budget=budget,
            exact_counter=exact_counter,
            client_factory=client_factory,
            token_provider=token_provider,
            session_factory=session_factory,
            csv_loader=csv_loader,
            selected=selected,
            api_key=api_key,
            base_url=base_url,
        )
        _append_token_operations(operations, state.token_operation)
        selected = state.selected or selected
        api_key = state.api_key or api_key
        operations.append(outcome)
        if state.pricing_error is not None:
            pricing_errors = cast(
                "dict[str, str]", data.setdefault("pricing_errors", {})
            )
            pricing_errors[name] = state.pricing_error
        if state.text is not None:
            key = "chat_text" if name.startswith("chat.") else "responses_text"
            data[key] = state.text
    if not _has_token_operation(operations) and not _has_live_model(
        operations
    ):
        operations.insert(0, _token_not_run(_token_skip_code(operations)))
    data["budget"] = (
        "reserved" if _reserved_total(operations) else "not_reserved"
    )
    if selected is not None:
        credential_source = selected.source
    return operations, data, credential_source


def _list_openai_catalog(
    *,
    settings: Settings,
    catalog_port: AwsPort | None,
) -> tuple[OperationOutcome, Mapping[str, Any]]:
    if catalog_port is None:
        return (
            not_run_operation(
                service="bedrock",
                operation_name="ListFoundationModels",
                phase="setup",
                error_code="missing_configuration",
                request_validated=False,
            ),
            {},
        )
    listed = port_operation(
        port=catalog_port,
        service="bedrock",
        operation_name="ListFoundationModels",
        params={"byProvider": "OpenAI"},
        execution="live",
        effect="read",
        fixture_id=_CATALOG_FIXTURE,
        phase="setup",
        endpoint_url=f"https://bedrock.{settings.region}.amazonaws.com",
    )
    return listed.outcome, listed.payload


def _select_live_model(
    *,
    settings: Settings,
    catalog_payload: Mapping[str, Any],
    api: str,
) -> str | None:
    requested = settings.model
    candidates: list[str] = []
    for item in _openai_model_summaries(catalog_payload):
        model_id = str(item.get("modelId", ""))
        if requested is not None and model_id != requested:
            continue
        if _summary_supports_api(item, api):
            candidates.append(model_id)
    if not candidates:
        return None
    if requested is not None:
        return candidates[0]
    if api == "CHAT_COMPLETIONS" and _DEFAULT_CHAT_MODEL in candidates:
        return _DEFAULT_CHAT_MODEL
    return sorted(candidates)[0]


def _openai_model_summaries(
    catalog_payload: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    return tuple(
        item
        for item in cast(
            "Sequence[Mapping[str, Any]]",
            catalog_payload.get("modelSummaries", ()),
        )
        if str(item.get("providerName", "OpenAI")) == "OpenAI"
    )


def _summary_supports_api(summary: Mapping[str, Any], api: str) -> bool:
    model_id = str(summary.get("modelId", ""))
    return api in _OPENAI_API_SUPPORT.get(model_id, frozenset())


@dataclass(frozen=True)
class _LiveCallState:
    token_operation: tuple[OperationOutcome, ...]
    selected: SelectedSession | None
    api_key: str | None
    text: str | None
    pricing_error: str | None = None


def _reserve_and_call_model(
    *,
    operation_name: str,
    fixture_id: str,
    model: str,
    request_body: Mapping[str, object],
    settings: Settings,
    policy: ExecutionPolicy,
    budget: PricedBudget | None,
    exact_counter: TokenCounter | None,
    client_factory: OpenAIClientFactory,
    token_provider: TokenProvider,
    session_factory: SessionFactory | None,
    csv_loader: CsvLoader | None,
    selected: SelectedSession | None,
    api_key: str | None,
    base_url: str,
) -> tuple[OperationOutcome, _LiveCallState]:
    del fixture_id
    try:
        active_selected = selected or select_session(
            execution="live",
            settings=settings,
            session_factory=session_factory,
            csv_loader=csv_loader,
        )
        active_key = api_key or token_provider(
            active_selected, settings.region
        )
    except (RuntimeError, ValueError, TypeError):
        return (
            not_run_operation(
                service="openai",
                operation_name=operation_name,
                error_code="missing_configuration",
                request_validated=False,
            ),
            _LiveCallState(
                token_operation=()
                if api_key is not None
                else (_token_not_run("missing_configuration"),),
                selected=selected,
                api_key=api_key,
                text=None,
            ),
        )
    try:
        active_budget = budget or billing.active_budget_run(
            region=settings.region,
            policy=policy,
        ).command(_DEMO)
        reservation, _count = active_budget.reserve_model(
            operation_name,
            model_id=model,
            serialized_request=_serialized(request_body),
            exact_counter=exact_counter,
        )
    except PolicyError as exc:
        return (
            not_run_operation(
                service="openai",
                operation_name=operation_name,
                error_code="budget_exceeded",
                request_validated=False,
            ),
            _LiveCallState(
                token_operation=()
                if api_key is not None
                else (_token_live_operation(),),
                selected=active_selected,
                api_key=active_key,
                text=None,
                pricing_error=sanitize_exception(exc),
            ),
        )
    client = client_factory(
        api_key=active_key,
        base_url=base_url,
        policy=policy,
    )
    try:
        if operation_name == "chat.completions.create":
            response = client.chat_completion(
                model=model,
                messages=cast(
                    "Sequence[Mapping[str, str]]", request_body["messages"]
                ),
                max_tokens=cast("int", request_body["max_tokens"]),
            )
        else:
            response = client.response(
                model=model,
                input_text=cast("str", request_body["input"]),
                max_output_tokens=cast(
                    "int", request_body["max_output_tokens"]
                ),
            )
    except Exception as exc:
        failure = _openai_failure_operation(
            exc=exc,
            operation_name=operation_name,
            endpoint_url=base_url,
            reservation=reservation,
        )
        if failure is None:
            raise
        return (
            failure,
            _LiveCallState(
                token_operation=()
                if api_key is not None
                else (_token_live_operation(),),
                selected=active_selected,
                api_key=active_key,
                text=None,
            ),
        )
    return (
        _openai_live_operation(
            operation_name=operation_name,
            endpoint_url=base_url,
            usage=response.usage,
            reservation=reservation,
        ),
        _LiveCallState(
            token_operation=()
            if api_key is not None
            else (_token_live_operation(),),
            selected=active_selected,
            api_key=active_key,
            text=response.text,
        ),
    )


def _token_fixture_operation() -> OperationOutcome:
    return operation(
        service="aws-bedrock-token-generator",
        operation="provide_token",
        phase="setup",
        execution_target="fixture",
        mode="local_contract",
        effect="none",
        transport="none",
        fixture_id=_TOKEN_FIXTURE,
    )


def _token_live_operation() -> OperationOutcome:
    return operation(
        service="aws-bedrock-token-generator",
        operation="provide_token",
        phase="setup",
        execution_target="local",
        mode="local_execution",
        effect="none",
        transport="none",
        response_received=True,
        request_validated=True,
    )


def _token_not_run(code: str) -> OperationOutcome:
    return not_run_operation(
        service="aws-bedrock-token-generator",
        operation_name="provide_token",
        phase="setup",
        error_code=cast("Any", code),
        request_validated=False,
    )


def _openai_fixture_operation(
    *,
    operation_name: str,
    fixture_id: str,
    usage: Usage | None,
) -> OperationOutcome:
    return operation(
        service="openai",
        operation=operation_name,
        execution_target="fixture",
        mode="local_contract",
        effect="infer",
        transport="none",
        response_received=True,
        request_validated=True,
        fixture_id=fixture_id,
        usage=usage,
    )


def _openai_live_operation(
    *,
    operation_name: str,
    endpoint_url: str,
    usage: Usage | None,
    reservation: Reservation,
) -> OperationOutcome:
    return operation(
        service="openai",
        operation=operation_name,
        execution_target="aws",
        mode="live_model",
        effect="infer",
        transport="aws",
        endpoint_url=endpoint_url,
        response_received=True,
        request_validated=True,
        usage=usage,
        reserved_usd=float(reservation.amount_usd),
    )


def _openai_blocked_operation(
    *,
    operation_name: str,
    endpoint_url: str,
    reservation: Reservation,
    http_status: int | None,
    error_code: str,
    status: str = "blocked",
) -> OperationOutcome:
    return operation(
        service="openai",
        operation=operation_name,
        execution_target="aws",
        mode="live_service",
        status=cast("Any", status),
        effect="infer",
        transport="aws",
        endpoint_url=endpoint_url,
        response_received=True,
        request_validated=True,
        http_status=http_status,
        error_code=cast("Any", error_code),
        reserved_usd=float(reservation.amount_usd),
    )


def _openai_attempt_failed_operation(
    *,
    operation_name: str,
    endpoint_url: str,
    reservation: Reservation,
) -> OperationOutcome:
    return operation(
        service="openai",
        operation=operation_name,
        execution_target="aws",
        mode="attempt_failed",
        status="blocked",
        effect="infer",
        transport="aws",
        endpoint_url=endpoint_url,
        response_received=False,
        request_validated=True,
        error_code="timeout",
        reserved_usd=float(reservation.amount_usd),
    )


def _known_capability_gap(
    model_id: str | None, api: str, catalog: Mapping[str, Any]
) -> bool:
    """Require both a documented restriction and an observed model."""
    if model_id is None or model_id not in _OPENAI_API_SUPPORT:
        return False
    supported = _OPENAI_API_SUPPORT[model_id]
    return api not in supported and any(
        item.get("modelId") == model_id
        for item in _openai_model_summaries(catalog)
    )


def _model_unavailable_operation(
    operation_name: str, code: ErrorCode = "model_unavailable"
) -> OperationOutcome:
    return not_run_operation(
        service="openai",
        operation_name=operation_name,
        error_code=code,
        request_validated=False,
    )


def _openai_evidence(
    operations: Sequence[OperationOutcome],
    *,
    settings: Settings,
    credential_source: CredentialSource,
    requested_model: str,
    observed_model: str | None,
) -> Evidence:
    cost = _reserved_total(operations)
    return evidence(
        sdk_invoked=any(
            item["mode"] != "not_run" or item["request_validated"]
            for item in operations
        ),
        network_attempted=any(
            item["transport"] in {"aws", "external", "loopback"}
            for item in operations
        ),
        aws_executed=any(
            item["execution_target"] == "aws" and item["response_received"]
            for item in operations
        ),
        provider="bedrock-openai",
        region=settings.region,
        fixture_id=next(
            (item["fixture_id"] for item in operations if item["fixture_id"]),
            None,
        ),
        requested_model=requested_model,
        observed_model=observed_model,
        credential_source=credential_source,
        packages=_package_versions(),
        estimated_cost_usd=cost,
        cost_basis="estimated" if cost is not None else None,
    )


def _package_versions() -> dict[str, str]:
    names = ("openai", "httpx", "aws-bedrock-token-generator")
    return {name: metadata.version(name) for name in names}


def _evidence_model(data: Mapping[str, object], fallback: str) -> str:
    candidate = data.get("chat_model")
    return candidate if isinstance(candidate, str) else fallback


def _openai_failure_operation(
    *,
    exc: Exception,
    operation_name: str,
    endpoint_url: str,
    reservation: Reservation,
) -> OperationOutcome | None:
    if _is_openai_attempt_failure(exc):
        return _openai_attempt_failed_operation(
            operation_name=operation_name,
            endpoint_url=endpoint_url,
            reservation=reservation,
        )
    status = _openai_http_status(exc)
    if status is not None:
        code = _openai_http_error_code(status, exc)
        return _openai_blocked_operation(
            operation_name=operation_name,
            endpoint_url=endpoint_url,
            reservation=reservation,
            http_status=status,
            error_code=code,
            status="error" if code == "validation_failed" else "blocked",
        )
    if _is_openai_model_unavailable(exc):
        return _openai_blocked_operation(
            operation_name=operation_name,
            endpoint_url=endpoint_url,
            reservation=reservation,
            http_status=None,
            error_code="model_unavailable",
        )
    return None


def _openai_http_status(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _openai_http_error_code(status: int, exc: Exception) -> str:
    if status in {400, 404} and _is_openai_model_unavailable(exc):
        return "model_unavailable"
    if status == 403:
        return "authorization_denied"
    if status == 429:
        return "throttled"
    return "validation_failed"


def _is_openai_attempt_failure(exc: Exception) -> bool:
    if isinstance(exc, OSError):
        return True
    name = type(exc).__name__
    return name in {"APIConnectionError", "APITimeoutError"}


def _is_openai_model_unavailable(exc: Exception) -> bool:
    marker = f"{type(exc).__name__} {exc}".lower()
    return "unsupported" in marker or "not supported" in marker


def _observed_model(
    operations: Sequence[OperationOutcome],
    model: str,
) -> str | None:
    if _has_live_model(operations):
        return model
    return None


def _has_live_model(operations: Sequence[OperationOutcome]) -> bool:
    return any(
        item["mode"] == "live_model" and item["status"] == "ok"
        for item in operations
    )


def _append_token_operations(
    operations: list[OperationOutcome],
    token_operations: Sequence[OperationOutcome],
) -> None:
    if token_operations and not _has_token_operation(operations):
        operations.extend(token_operations)


def _has_token_operation(operations: Sequence[OperationOutcome]) -> bool:
    return any(
        item["service"] == "aws-bedrock-token-generator" for item in operations
    )


def _token_skip_code(operations: Sequence[OperationOutcome]) -> str:
    if any(
        item["error_code"] == "missing_configuration" for item in operations
    ):
        return "missing_configuration"
    return "model_unavailable"


def _reserved_total(operations: Sequence[OperationOutcome]) -> float | None:
    total = sum(item["reserved_usd"] or 0.0 for item in operations)
    return total or None


def _mock_client_factory(
    *,
    api_key: str,
    base_url: str,
    policy: ExecutionPolicy,
) -> OpenAIClientPort:
    del policy
    from openai import OpenAI

    return OpenAISdkPort(
        OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            http_client=cast(
                "Any",
                httpx.Client(
                    transport=httpx.MockTransport(_mock_openai_response),
                ),
            ),
        ),
    )


def _mock_openai_response(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/chat/completions"):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-bedrock-demo",
                "object": "chat.completion",
                "created": 1_760_000_000,
                "model": _DEFAULT_CHAT_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "Bedrock accepted the OpenAI chat call."
                            ),
                        },
                        "finish_reason": "stop",
                    },
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 6,
                    "total_tokens": 18,
                },
            },
        )
    return httpx.Response(
        200,
        json={
            "id": "resp-bedrock-demo",
            "object": "response",
            "created_at": 1_760_000_000,
            "status": "completed",
            "model": _DEFAULT_RESPONSES_MODEL,
            "output": [
                {
                    "id": "msg-bedrock-demo",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Responses works through Bedrock too.",
                            "annotations": [],
                        },
                    ],
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        },
    )


def _chat_usage(raw: object) -> Usage | None:
    if raw is None:
        return None
    usage: Usage = {}
    if (value := getattr(raw, "prompt_tokens", None)) is not None:
        usage["input_tokens"] = int(value)
    if (value := getattr(raw, "completion_tokens", None)) is not None:
        usage["output_tokens"] = int(value)
    if (value := getattr(raw, "total_tokens", None)) is not None:
        usage["total_tokens"] = int(value)
    return usage


def _responses_usage(raw: object) -> Usage | None:
    if raw is None:
        return None
    usage: Usage = {}
    if (value := getattr(raw, "input_tokens", None)) is not None:
        usage["input_tokens"] = int(value)
    if (value := getattr(raw, "output_tokens", None)) is not None:
        usage["output_tokens"] = int(value)
    if (value := getattr(raw, "total_tokens", None)) is not None:
        usage["total_tokens"] = int(value)
    return usage


def _credentials_from_session(session: object) -> object:
    direct = getattr(session, "get_credentials", None)
    if callable(direct):
        credentials = direct()
    else:
        internal = getattr(session, "_session", None)
        internal_get = getattr(internal, "get_credentials", None)
        credentials = internal_get() if callable(internal_get) else None
    if credentials is None:
        message = "selected AWS session did not provide credentials"
        raise RuntimeError(message)
    return credentials


def _chat_model(settings: Settings) -> str:
    return settings.model or _DEFAULT_CHAT_MODEL


def _responses_model(settings: Settings) -> str:
    return settings.model or _DEFAULT_RESPONSES_MODEL


def _request_limit(policy: ExecutionPolicy) -> int:
    return min(64, policy.max_output_tokens)


def _messages() -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": "Summarize the support pilot migration in one line.",
        },
    ]


def _serialized(body: Mapping[str, object]) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode(
        "utf-8",
    )
