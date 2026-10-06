import re
from pathlib import Path

import pytest

from awsai_demo.runtime import (
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_LOCALSTACK_ENDPOINT,
    DEFAULT_REGION,
    Settings,
    load_env_file,
    load_settings,
    parse_env_text,
    region_from_env,
    with_cli_overrides,
)


def test_parse_env_text_handles_comments_exports_and_quotes() -> None:
    parsed = parse_env_text(
        """
        # comment
        export AWS_REGION = us-east-2
        OPENAI_API_KEY="sk-test"
        LOCALSTACK_AUTH_TOKEN='token'
        """,
    )

    assert parsed == {
        "AWS_REGION": "us-east-2",
        "OPENAI_API_KEY": "sk-test",
        "LOCALSTACK_AUTH_TOKEN": "token",
    }


def test_parse_env_text_rejects_malformed_line() -> None:
    with pytest.raises(ValueError, match=re.escape("Invalid .env line")):
        parse_env_text("1BAD=value")


def test_load_env_file_preserves_real_environment(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "AWS_REGION=us-west-2\nDEFAULT_BEDROCK_MODEL=amazon.test\n",
        encoding="utf-8",
    )
    environ = {"AWS_REGION": "eu-central-1"}

    applied = load_env_file(env_path, environ=environ)

    assert applied == {"DEFAULT_BEDROCK_MODEL": "amazon.test"}
    assert environ["AWS_REGION"] == "eu-central-1"
    assert environ["DEFAULT_BEDROCK_MODEL"] == "amazon.test"


def test_load_env_file_missing_file_is_noop(tmp_path: Path) -> None:
    environ: dict[str, str] = {}

    assert load_env_file(tmp_path / "missing.env", environ=environ) == {}
    assert environ == {}


def test_settings_from_env_applies_defaults_and_blank_values() -> None:
    settings = Settings.from_env({"AWS_REGION": "", "LOCALSTACK_ENDPOINT": ""})

    assert settings.region == DEFAULT_REGION
    assert settings.localstack_endpoint == DEFAULT_LOCALSTACK_ENDPOINT
    assert settings.default_bedrock_model == DEFAULT_BEDROCK_MODEL
    assert settings.aws_profile is None


def test_settings_from_env_reads_all_known_values() -> None:
    environ = {
        "AWS_REGION": "us-east-2",
        "AWS_PROFILE": "demo",
        "AWS_CREDS_FILE_PATH": "creds.csv",
        "AWSAI_HARNESS_ROLE_ARN": "arn",
        "AWSAI_GUARDRAIL_ID": "guard",
        "AWSAI_GUARDRAIL_VERSION": "1",
        "AWSAI_REGISTRY_ID": "reg",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector",
        "LOCALSTACK_AUTH_TOKEN": "configured-value",
        "LOCALSTACK_ENDPOINT": "http://localhost:4566",
        "DEFAULT_BEDROCK_MODEL": "model",
        "OPENAI_API_KEY": "key",
    }
    settings = Settings.from_env(environ)

    assert settings.region == "us-east-2"
    assert settings.aws_profile == "demo"
    assert settings.aws_creds_file_path == Path("creds.csv")
    assert settings.harness_role_arn == "arn"
    assert settings.guardrail_id == "guard"
    assert settings.guardrail_version == "1"
    assert settings.registry_id == "reg"
    assert settings.otel_exporter_otlp_endpoint == "http://collector"
    assert settings.localstack_auth_token == environ["LOCALSTACK_AUTH_TOKEN"]
    assert settings.localstack_endpoint == "http://localhost:4566"
    assert settings.default_bedrock_model == "model"
    assert settings.openai_api_key == "key"


def test_region_from_env_precedence() -> None:
    environ = {"AWS_REGION": "us-west-2", "AWS_DEFAULT_REGION": "us-east-2"}

    assert region_from_env(environ, cli_region="eu-west-1") == "eu-west-1"
    assert region_from_env(environ) == "us-west-2"
    assert region_from_env({"AWS_DEFAULT_REGION": "us-east-2"}) == "us-east-2"
    assert region_from_env({}) == DEFAULT_REGION


def test_load_settings_reads_file_then_builds_settings(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("AWS_DEFAULT_REGION=us-east-2\n", encoding="utf-8")

    settings = load_settings(env_file=env_path, environ={})

    assert settings.region == "us-east-2"


def test_load_settings_can_skip_env_file() -> None:
    settings = load_settings(
        env_file=None,
        environ={"AWS_DEFAULT_REGION": "eu-west-1"},
    )

    assert settings.region == "eu-west-1"


def test_with_cli_overrides_replaces_region_and_profile() -> None:
    settings = Settings(region="us-east-1", aws_profile="env-profile")

    updated = with_cli_overrides(
        settings,
        region="us-west-2",
        profile="cli-profile",
        model="model-id",
        provider="bedrock",
        trace="memory",
        allow_create=True,
    )

    assert updated.region == "us-west-2"
    assert updated.aws_profile == "cli-profile"
    assert updated.model == "model-id"
    assert updated.provider == "bedrock"
    assert updated.trace == "memory"
    assert updated.allow_create is True
    assert updated.default_bedrock_model == settings.default_bedrock_model
