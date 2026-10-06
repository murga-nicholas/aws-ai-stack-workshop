"""Runtime settings and policy helpers for offline-first demos."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

Execution = Literal["offline", "emulator", "live"]
ProviderName = Literal["offline", "bedrock", "openai"]
TraceMode = Literal["off", "memory", "otlp"]

DEFAULT_REGION = "us-east-1"
DEFAULT_LOCALSTACK_ENDPOINT = "http://localhost:4566"
DEFAULT_OLLAMA_MODEL = "qwen2.5:0.5b"
DEFAULT_BEDROCK_MODEL = DEFAULT_OLLAMA_MODEL
TESTED_LIVE_BEDROCK_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


@dataclass(frozen=True)
class Settings:
    """Environment-derived configuration."""

    region: str = DEFAULT_REGION
    model: str | None = None
    provider: ProviderName | None = None
    trace: TraceMode = "off"
    allow_create: bool = False
    aws_profile: str | None = None
    aws_creds_file_path: Path | None = None
    harness_role_arn: str | None = None
    guardrail_id: str | None = None
    guardrail_version: str | None = None
    registry_id: str | None = None
    otel_exporter_otlp_endpoint: str | None = None
    localstack_auth_token: str | None = field(default=None, repr=False)
    localstack_endpoint: str = DEFAULT_LOCALSTACK_ENDPOINT
    default_bedrock_model: str = DEFAULT_BEDROCK_MODEL
    openai_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        """Build settings from an environment mapping."""
        source = environ if environ is not None else os.environ
        return cls(
            region=region_from_env(source),
            aws_profile=_blank_to_none(source.get("AWS_PROFILE")),
            aws_creds_file_path=_optional_path(
                source.get("AWS_CREDS_FILE_PATH")
            ),
            harness_role_arn=_blank_to_none(
                source.get("AWSAI_HARNESS_ROLE_ARN")
            ),
            guardrail_id=_blank_to_none(source.get("AWSAI_GUARDRAIL_ID")),
            guardrail_version=_blank_to_none(
                source.get("AWSAI_GUARDRAIL_VERSION"),
            ),
            registry_id=_blank_to_none(source.get("AWSAI_REGISTRY_ID")),
            otel_exporter_otlp_endpoint=_blank_to_none(
                source.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
            ),
            localstack_auth_token=_blank_to_none(
                source.get("LOCALSTACK_AUTH_TOKEN"),
            ),
            localstack_endpoint=_non_blank(
                source.get("LOCALSTACK_ENDPOINT"),
                DEFAULT_LOCALSTACK_ENDPOINT,
            ),
            default_bedrock_model=_non_blank(
                source.get("DEFAULT_BEDROCK_MODEL"),
                DEFAULT_BEDROCK_MODEL,
            ),
            openai_api_key=_blank_to_none(source.get("OPENAI_API_KEY")),
        )


def load_settings(
    *,
    env_file: str | Path | None = ".env",
    environ: MutableMapping[str, str] | None = None,
) -> Settings:
    """Load dotenv without overriding real env."""
    target = environ if environ is not None else os.environ
    if env_file is not None:
        load_env_file(env_file, environ=target)
    return Settings.from_env(target)


def load_env_file(
    path: str | Path = ".env",
    *,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Read dotenv assignments without overriding existing values."""
    target = environ if environ is not None else os.environ
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return {}
    parsed = parse_env_text(dotenv_path.read_text(encoding="utf-8-sig"))
    applied: dict[str, str] = {}
    for key, value in parsed.items():
        if key not in target:
            target[key] = value
            applied[key] = value
    return applied


def parse_env_text(text: str) -> dict[str, str]:
    """Parse simple dotenv content into key/value pairs."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not _ENV_KEY_RE.fullmatch(key):
            msg = f"Invalid .env line for key {key!r}"
            raise ValueError(msg)
        values[key] = _strip_env_value(value.strip())
    return values


def region_from_env(
    environ: Mapping[str, str],
    *,
    cli_region: str | None = None,
) -> str:
    """Apply CLI, AWS_REGION, AWS_DEFAULT_REGION, default precedence."""
    return _non_blank(
        cli_region,
        _non_blank(
            environ.get("AWS_REGION"),
            _non_blank(environ.get("AWS_DEFAULT_REGION"), DEFAULT_REGION),
        ),
    )


def with_cli_overrides(
    settings: Settings,
    *,
    region: str | None = None,
    profile: str | None = None,
    model: str | None = None,
    provider: ProviderName | None = None,
    trace: TraceMode | None = None,
    allow_create: bool | None = None,
) -> Settings:
    """Return settings with CLI region/profile overrides applied."""
    return Settings(
        region=_non_blank(region, settings.region),
        model=_blank_to_none(model) or settings.model,
        provider=provider or settings.provider,
        trace=trace or settings.trace,
        allow_create=(
            settings.allow_create if allow_create is None else allow_create
        ),
        aws_profile=_blank_to_none(profile) or settings.aws_profile,
        aws_creds_file_path=settings.aws_creds_file_path,
        harness_role_arn=settings.harness_role_arn,
        guardrail_id=settings.guardrail_id,
        guardrail_version=settings.guardrail_version,
        registry_id=settings.registry_id,
        otel_exporter_otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        localstack_auth_token=settings.localstack_auth_token,
        localstack_endpoint=settings.localstack_endpoint,
        default_bedrock_model=settings.default_bedrock_model,
        openai_api_key=settings.openai_api_key,
    )


_ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _non_blank(value: str | None, default: str) -> str:
    return _blank_to_none(value) or default


def _optional_path(value: str | None) -> Path | None:
    stripped = _blank_to_none(value)
    if stripped is None:
        return None
    return Path(stripped)
