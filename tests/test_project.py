from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV_KEYS = {
    "AWS_ACCESS_KEY_ID",
    "AWS_CREDS_FILE_PATH",
    "AWS_DEFAULT_REGION",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWSAI_GUARDRAIL_ID",
    "AWSAI_GUARDRAIL_VERSION",
    "AWSAI_HARNESS_ROLE_ARN",
    "AWSAI_REGISTRY_ID",
    "DEFAULT_BEDROCK_MODEL",
    "LOCALSTACK_AUTH_TOKEN",
    "LOCALSTACK_ENDPOINT",
    "OPENAI_API_KEY",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
}


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())


def test_pyproject_foundation_packaging_script_and_coverage() -> None:
    config = _pyproject()

    assert config["project"]["name"] == "aws-ai-stack-workshop"
    assert config["project"]["requires-python"] == ">=3.12,<3.14"
    assert config["project"]["scripts"] == {
        "awsai-demo": "awsai_demo.cli:main",
    }
    assert config["build-system"] == {
        "requires": ["hatchling"],
        "build-backend": "hatchling.build",
    }
    wheel = config["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["src/awsai_demo", "deck"]
    assert wheel["force-include"] == {
        "data/lineage.yaml": "awsai_demo/data/lineage.yaml",
        "data/components.yaml": "awsai_demo/data/components.yaml",
        "data/aws_ai_stack_notes.md": "awsai_demo/data/aws_ai_stack_notes.md",
    }

    pytest_options = config["tool"]["pytest"]["ini_options"]
    assert pytest_options["testpaths"] == ["tests"]
    assert pytest_options["xfail_strict"] is True
    for required in (
        "--strict-config",
        "--strict-markers",
        "--basetemp=.pytest_tmp",
        "--cov",
        "--cov-report=term-missing",
    ):
        assert required in pytest_options["addopts"]

    coverage_run = config["tool"]["coverage"]["run"]
    assert coverage_run["branch"] is True
    assert coverage_run["parallel"] is True
    assert coverage_run["source"] == ["awsai_demo", "scripts", "deck"]
    assert coverage_run["patch"] == ["subprocess"]

    coverage_report = config["tool"]["coverage"]["report"]
    assert coverage_report["fail_under"] == 100
    assert coverage_report["show_missing"] is True
    assert coverage_report["skip_covered"] is True


def test_env_example_documents_every_user_facing_var_blank() -> None:
    lines = (PROJECT_ROOT / ".env.example").read_text().splitlines()
    assignments = dict(
        line.split("=", maxsplit=1)
        for line in lines
        if line and not line.startswith("#")
    )

    assert set(assignments) == EXPECTED_ENV_KEYS
    assert all(value == "" for value in assignments.values())


def test_localstack_compose_uses_pinned_loopback_configuration() -> None:
    text = (PROJECT_ROOT / "docker-compose.localstack.yml").read_text()
    match = re.search(r"^\s*image:\s*(\S+)\s*$", text, flags=re.MULTILINE)

    assert match is not None
    image = match.group(1)
    assert image.startswith("localstack/localstack:2026.09.0")
    assert ":latest" not in image
    assert "127.0.0.1:4566:4566" in text
    assert "${LOCALSTACK_AUTH_TOKEN:?Set your LocalStack auth token}" in text
    assert "SERVICES" not in text
    assert "@sha256:" in image or ("TODO" in text and "digest" in text)


def test_curated_data_sources_exist_for_packaging() -> None:
    assert (PROJECT_ROOT / "data" / "lineage.yaml").is_file()
    assert (PROJECT_ROOT / "data" / "components.yaml").is_file()


def test_repository_python_has_no_coverage_escape_pragmas() -> None:
    for folder in ("src", "scripts", "deck"):
        for path in (PROJECT_ROOT / folder).rglob("*.py"):
            assert "pragma: no cover" not in path.read_text(encoding="utf-8")
            assert "pragma: no branch" not in path.read_text(encoding="utf-8")
