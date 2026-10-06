from __future__ import annotations

import importlib.util
import io
import json
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.stub import Stubber

from awsai_demo.runtime import Settings

SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "snapshot_prices.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "awsai_snapshot_prices_under_test",
    SNAPSHOT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main
snapshot_prices = _MODULE.snapshot_prices


class StubbedSessionFactory:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.calls: list[dict[str, str]] = []
        self.session = StubbedSession(client)

    def __call__(self, **kwargs: str) -> StubbedSession:
        self.calls.append(kwargs)
        return self.session


class StubbedSession:
    def __init__(self, client: Any) -> None:
        self.client_instance = client
        self.client_calls: list[tuple[str, dict[str, object]]] = []

    def client(self, service_name: str, **kwargs: object) -> Any:
        self.client_calls.append((service_name, kwargs))
        return self.client_instance


def test_snapshot_prices_cli_uses_profile_and_stubbed_pricing(
    tmp_path: Path,
    capsys: Any,
) -> None:
    client = _pricing_client()
    factory = StubbedSessionFactory(client)
    output = tmp_path / "pricing_snapshot.json"
    with Stubber(client) as stubber:
        _stub_translate(stubber)
        exit_code = main(
            ["--output", str(output), "--profile", "demo"],
            session_factory=factory,
            settings_loader=_settings,
            clock=lambda: datetime(2026, 10, 6, tzinfo=UTC),
        )

    payload = json.loads(output.read_text(encoding="utf-8"))
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert factory.calls == [
        {"profile_name": "demo", "region_name": "us-east-1"}
    ]
    assert factory.session.client_calls == [
        ("pricing", {"region_name": "us-east-1"})
    ]
    assert payload["requested_regions"] == ["us-east-1"]
    assert payload["rows"][0]["feature"] == "translate.characters"
    assert payload["rows"][0]["unit"] == "character"
    assert payload["rows"][0]["raw_unit"] == "1K characters"
    assert summary["rows"] == 1


def test_snapshot_prices_function_accepts_explicit_regions(
    tmp_path: Path,
) -> None:
    client = _pricing_client()
    factory = StubbedSessionFactory(client)
    output = tmp_path / "snapshot.json"
    with Stubber(client) as stubber:
        _stub_translate(stubber, region="eu-west-1")
        snapshot = snapshot_prices(
            regions=["eu-west-1"],
            output=output,
            session_factory=factory,
            settings=Settings(),
            clock=lambda: "fixed",
        )

    assert snapshot.requested_regions == ("eu-west-1",)
    assert snapshot.read_at == "fixed"
    assert output.exists()


def test_snapshot_prices_cli_sanitizes_parse_errors() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(["--bad"], stdout=stdout, stderr=stderr)

    assert code == 1
    assert stdout.getvalue() == ""
    assert "ValueError" in stderr.getvalue()


def test_optional_review_output_is_explicit_and_gitignored(
    tmp_path: Path,
) -> None:
    client = _pricing_client()
    output = tmp_path / "snapshot.json"
    review = tmp_path / "unmatched.pricing-review.json"
    with Stubber(client) as stubber:
        _stub_translate(stubber)
        code = main(
            ["--output", str(output), "--review-output", str(review)],
            session_factory=StubbedSessionFactory(client),
            settings_loader=_settings,
            stdout=io.StringIO(),
        )
    assert code == 0
    assert json.loads(review.read_text(encoding="utf-8"))["rows"] == []
    assert (
        main(
            ["--review-output", str(tmp_path / "unsafe.json")],
            stderr=io.StringIO(),
        )
        == 1
    )


def test_snapshot_prices_module_help_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["snapshot_prices.py", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(SNAPSHOT_PATH), run_name="__main__")

    assert excinfo.value.code == 0


def _settings() -> Settings:
    return Settings()


def _pricing_client() -> Any:
    session = boto3.Session(
        aws_access_key_id="AKIDEXAMPLE",
        aws_secret_access_key="secret",  # noqa: S106
        aws_session_token="token",  # noqa: S106
        region_name="us-east-1",
    )
    return session.client("pricing", region_name="us-east-1")


def _stub_translate(stubber: Stubber, *, region: str = "us-east-1") -> None:
    stubber.add_response(
        "describe_services",
        {"Services": [{"ServiceCode": "AmazonTranslate"}]},
        {"MaxResults": 100},
    )
    stubber.add_response(
        "get_products",
        {
            "PriceList": [
                json.dumps(
                    {
                        "product": {
                            "sku": "TRANS",
                            "productFamily": "AI",
                            "attributes": {
                                "servicecode": "AmazonTranslate",
                                "regionCode": region,
                                "usagetype": "TranslateChars",
                            },
                        },
                        "terms": {
                            "OnDemand": {
                                "TRANS.term": {
                                    "priceDimensions": {
                                        "TRANS.rate": {
                                            "unit": "1K characters",
                                            "pricePerUnit": {"USD": "0.015"},
                                            "beginRange": "0",
                                            "endRange": "Inf",
                                            "description": (
                                                "Translate characters"
                                            ),
                                        }
                                    }
                                }
                            }
                        },
                    }
                )
            ]
        },
        {
            "ServiceCode": "AmazonTranslate",
            "FormatVersion": "aws_v1",
            "Filters": [
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region}
            ],
            "MaxResults": 100,
        },
    )
