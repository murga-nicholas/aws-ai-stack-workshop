"""Lazy provider factory for offline, Bedrock, emulator, and OpenAI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from awsai_demo.offline import ScriptedModel, text_script
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import (
    DEFAULT_LOCALSTACK_ENDPOINT,
    TESTED_LIVE_BEDROCK_MODEL,
    Execution,
    ProviderName,
    Settings,
)

if TYPE_CHECKING:
    from awsai_demo.credentials import SelectedSession

ProviderExecution = Literal["offline", "emulator", "live"]
_STRANDS_PACKAGE = "strands-agents"


class SdkMissingError(RuntimeError):
    """Raised when an optional provider SDK is not installed."""

    def __init__(self, package: str) -> None:
        """Name the missing package without leaking configuration."""
        super().__init__(f"Optional SDK is not installed: {package}")
        self.package = package


SdkMissing = SdkMissingError


@dataclass(frozen=True)
class ModelSelection:
    """A created model object and evidence provider label."""

    model: object
    provider: str
    model_id: str | None


def create_model(
    *,
    provider: ProviderName,
    execution: Execution,
    settings: Settings,
    policy: ExecutionPolicy | None = None,
    selected_session: SelectedSession | None = None,
    model_id: str | None = None,
) -> ModelSelection:
    """Create the selected model lazily."""
    chosen_policy = policy or ExecutionPolicy()
    if provider == "offline" and execution == "offline":
        return ModelSelection(
            model=ScriptedModel(text_script("Offline fixture response.")),
            provider="offline",
            model_id=None,
        )
    if execution == "offline" or provider == "offline":
        msg = f"Unsupported provider/execution combination: {provider}/offline"
        raise ValueError(msg)
    requested_model = _requested_model(
        execution=execution,
        settings=settings,
        model_id=model_id,
    )
    if provider == "bedrock" and execution == "emulator":
        return _create_emulator_model(
            settings=settings,
            policy=chosen_policy,
            model_id=requested_model,
            selected_session=selected_session,
        )
    if provider == "bedrock":
        if selected_session is None:
            msg = "Bedrock provider requires a selected session"
            raise ValueError(msg)
        return _create_bedrock_model(
            selected_session=selected_session,
            policy=chosen_policy,
            model_id=requested_model,
        )
    if provider == "openai" and execution == "live":
        return _create_openai_model(
            settings=settings,
            policy=chosen_policy,
            model_id=_requested_openai_model(
                settings=settings,
                model_id=model_id,
            ),
        )
    msg = f"Unsupported provider/execution combination: {provider}/{execution}"
    raise ValueError(msg)


def _create_bedrock_model(
    *,
    selected_session: SelectedSession,
    policy: ExecutionPolicy,
    model_id: str,
) -> ModelSelection:
    try:
        from botocore.config import Config

        from awsai_demo.bounded_bedrock import BoundedBedrockModel
    except ModuleNotFoundError as exc:
        raise SdkMissingError(_STRANDS_PACKAGE) from exc
    return ModelSelection(
        model=BoundedBedrockModel(
            model_id=model_id,
            boto_session=selected_session.session,
            boto_client_config=Config(
                retries={
                    "total_max_attempts": policy.total_max_attempts,
                    "mode": "standard",
                },
                connect_timeout=min(policy.max_wall_seconds, 30),
                read_timeout=min(policy.max_wall_seconds, 30),
            ),
            max_tokens=policy.max_output_tokens,
        ),
        provider="bedrock",
        model_id=model_id,
    )


def _create_emulator_model(
    *,
    settings: Settings,
    policy: ExecutionPolicy,
    model_id: str,
    selected_session: SelectedSession | None,
) -> ModelSelection:
    if selected_session is None:
        msg = "Bedrock emulator provider requires a selected session"
        raise ValueError(msg)
    try:
        from botocore.config import Config
        from strands.models.bedrock import BedrockModel
    except ModuleNotFoundError as exc:
        raise SdkMissingError(_STRANDS_PACKAGE) from exc
    endpoint_url = settings.localstack_endpoint or DEFAULT_LOCALSTACK_ENDPOINT
    local_model_id = (
        model_id if model_id.startswith("ollama.") else f"ollama.{model_id}"
    )
    return ModelSelection(
        model=BedrockModel(
            model_id=local_model_id,
            endpoint_url=endpoint_url,
            boto_session=selected_session.session,
            boto_client_config=Config(
                retries={
                    "total_max_attempts": policy.total_max_attempts,
                    "mode": "standard",
                },
                connect_timeout=min(policy.max_wall_seconds, 30),
                read_timeout=min(policy.max_wall_seconds, 30),
            ),
            streaming=False,
            use_native_token_count=False,
            max_tokens=policy.max_output_tokens,
        ),
        provider="bedrock-emulator",
        model_id=local_model_id,
    )


def _create_openai_model(
    *,
    settings: Settings,
    policy: ExecutionPolicy,
    model_id: str | None,
) -> ModelSelection:
    if settings.openai_api_key is None:
        msg = "OPENAI_API_KEY is required for provider openai"
        raise ValueError(msg)
    if model_id is None:
        msg = "OpenAI provider requires --model"
        raise ValueError(msg)
    try:
        from strands.models.openai import OpenAIModel
    except ModuleNotFoundError as exc:
        raise SdkMissingError(_STRANDS_PACKAGE) from exc
    return ModelSelection(
        model=OpenAIModel(
            client_args={
                "api_key": settings.openai_api_key,
                "max_retries": policy.openai_max_retries,
            },
            model_id=model_id,
            params={"max_tokens": policy.max_output_tokens},
        ),
        provider="openai",
        model_id=model_id,
    )


def _requested_model(
    *,
    execution: Execution,
    settings: Settings,
    model_id: str | None,
) -> str:
    if model_id is not None:
        return model_id
    if settings.model is not None:
        return settings.model
    if execution == "emulator":
        return settings.default_bedrock_model
    return TESTED_LIVE_BEDROCK_MODEL


def _requested_openai_model(
    *,
    settings: Settings,
    model_id: str | None,
) -> str | None:
    if model_id is not None:
        return model_id
    return settings.model
