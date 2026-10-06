from __future__ import annotations

from decimal import Decimal
from io import BytesIO
from typing import Any

import pytest
from botocore.exceptions import ClientError

from awsai_demo.ai_services_demo import run_ai_services_demo, service_rows
from awsai_demo.aws_identity_demo import principal_type, run_aws_identity_demo
from awsai_demo.billing import open_budget_run, use_budget_run
from awsai_demo.demo_support import AwsResponse
from awsai_demo.kendra_demo import run_kendra_demo, translate_filter
from awsai_demo.policy import ExecutionPolicy, ReservationUnavailable
from awsai_demo.runtime import Settings
from awsai_demo.sagemaker_ai_demo import run_sagemaker_ai_demo
from awsai_demo.service_rows import ServiceRow


class Prices:
    def __init__(self, missing: bool = False) -> None:
        self.missing = missing
        self.calls: list[tuple[str, str]] = []

    def rate(self, feature: str, *, unit: str, **_: Any) -> Decimal:
        self.calls.append((feature, unit))
        if self.missing:
            message = "missing verified price"
            raise ReservationUnavailable(message)
        return Decimal("0.000001")


class Port:
    def __init__(self, denied: str = "", identity: bool = True) -> None:
        self.denied = denied
        self.identity = identity
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.audio = BytesIO(b"123456")

    def call(
        self, service: str, operation_name: str, params: Any
    ) -> AwsResponse:
        self.calls.append((service, operation_name, dict(params)))
        if operation_name == self.denied:
            raise ClientError(
                {
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": "arn:aws:iam::123456789012:user/secret",
                    },
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                operation_name,
            )
        payloads: dict[str, Any] = {
            "GetCallerIdentity": {
                "Arn": (
                    "arn:aws:sts::123456789012:"
                    "assumed-role/private-role/person"
                ),
                "Account": "123456789012",
            }
            if self.identity
            else {},
            "ListFoundationModels": {"modelSummaries": []},
            "ListIndices": {
                "IndexConfigurationSummaryItems": [{"Name": "one"}]
            },
            "ListEndpoints": {"Endpoints": [{}]},
            "ListModels": {"Models": [{}, {}]},
            "SynthesizeSpeech": {"AudioStream": self.audio},
            "DetectPiiEntities": {"Entities": [{"Type": "EMAIL"}]},
            "DetectSentiment": {"Sentiment": "NEUTRAL"},
            "TranslateText": {"TranslatedText": "aprobacion"},
        }
        endpoint = (
            "http://localhost:4566"
            if service in {"s3", "transcribe", "textract", "sagemaker"}
            else f"https://{service}.us-east-1.amazonaws.com"
        )
        return AwsResponse(payloads.get(operation_name, {}), endpoint)


@pytest.mark.parametrize(
    "runner",
    [
        run_kendra_demo,
        run_sagemaker_ai_demo,
        run_aws_identity_demo,
        run_ai_services_demo,
    ],
)
def test_offline_and_emulator_missing(runner: Any) -> None:
    offline = runner(
        execution="offline", settings=Settings(), policy=ExecutionPolicy()
    )
    assert offline["status"] == "ok"
    blocked = runner(
        execution="emulator", settings=Settings(), policy=ExecutionPolicy()
    )
    assert blocked["status"] == "blocked"


def test_kendra_translation_and_live() -> None:
    attribute = {
        "EqualsTo": {"Key": "team", "Value": {"StringValue": "support"}}
    }
    assert translate_filter(attribute) == {
        "equals": {"key": "team", "value": "support"}
    }
    for invalid in ({}, {"EqualsTo": {"Key": "x", "Value": {"LongValue": 1}}}):
        with pytest.raises(ValueError):
            translate_filter(invalid)
    port = Port()
    result = run_kendra_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=port,
    )
    assert result["data"]["indices_listed"] == 1
    assert [item[1] for item in port.calls] == ["ListIndices"]


@pytest.mark.parametrize("execution", ["live", "emulator"])
def test_sagemaker_list_only(execution: Any) -> None:
    port = Port()
    if execution == "live":

        class LivePort(Port):
            def call(self, *args: Any, **kwargs: Any) -> AwsResponse:
                response = super().call(*args, **kwargs)
                return AwsResponse(
                    response.payload,
                    "https://sagemaker.us-east-1.amazonaws.com",
                )

        port = LivePort()
    result = run_sagemaker_ai_demo(
        execution=execution,
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=port,
    )
    assert result["data"]["models_listed"] == 2
    assert [item[1] for item in port.calls] == ["ListEndpoints", "ListModels"]


def test_identity_success_denial_and_redaction() -> None:
    port = Port(denied="ListFoundationModels")
    result = run_aws_identity_demo(
        execution="live",
        settings=Settings(aws_profile="aila-sandbox"),
        policy=ExecutionPolicy(),
        port=port,
    )
    assert result["data"]["authenticated"] is True
    assert result["data"]["principal_type"] == "assumed-role"
    assert result["data"]["bedrock_probe"] == "blocked"
    assert result["data"]["bedrock_probe_code"] == "authorization_denied"
    assert result["operations"][0]["mode"] == "live_identity"
    assert result["operations"][1]["mode"] == "live_service"
    assert "private-role" not in str(result) and "123456789012" not in str(
        result
    )
    assert principal_type("malformed") == "unknown"
    missing = run_aws_identity_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=Port(identity=False),
    )
    assert missing["data"]["principal_arn"] is None


class Session:
    def __init__(self, port: Port) -> None:
        self.port = port
        self.names: list[str] = []

    def client(self, service: str, **kwargs: Any) -> Any:
        self.names.append(service)
        owner = self

        class Client:
            meta = type(
                "Meta",
                (),
                {
                    "endpoint_url": kwargs.get(
                        "endpoint_url",
                        f"https://{service}.us-east-1.amazonaws.com",
                    )
                },
            )()

            def __getattr__(self, method: str) -> Any:
                operation = "".join(part.title() for part in method.split("_"))

                def call(**params: Any) -> Any:
                    return dict(
                        owner.port.call(service, operation, params).payload
                    )

                return call

        return Client()


@pytest.mark.parametrize(
    "runner",
    [
        run_kendra_demo,
        run_sagemaker_ai_demo,
        run_aws_identity_demo,
        run_ai_services_demo,
    ],
)
def test_default_adapter_with_injected_session(
    runner: Any, tmp_path: Any
) -> None:
    session = Session(Port())
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        prices=Prices(),
        root=tmp_path,
    )
    with use_budget_run(run):
        result = runner(
            execution="live",
            settings=Settings(aws_profile="aila-sandbox"),
            policy=run.policy,
            session_factory=lambda **_: session,
        )
    assert (
        result["evidence"]["credential_source"] == "profile"
        or result["mode"] == "batch"
    )
    assert session.names


def test_identity_emulator_sts_only() -> None:
    # Endpoint evidence must identify a loopback response.
    session = Session(Port())
    result = run_aws_identity_demo(
        execution="emulator",
        settings=Settings(localstack_auth_token="fixture"),  # noqa: S106
        policy=ExecutionPolicy(),
        session_factory=lambda **_: session,
    )
    assert result["operations"][0]["mode"] == "local_emulator"
    assert session.names == ["sts"]


def test_ai_services_shared_reservations_and_summaries(tmp_path: Any) -> None:
    prices = Prices()
    port = Port()
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        prices=prices,
        root=tmp_path,
    )
    with use_budget_run(run):
        result = run_ai_services_demo(
            execution="live", settings=Settings(), policy=run.policy, port=port
        )
    assert result["mode"] == "batch" and len(result["children"]) == 7
    assert [item[1] for item in port.calls] == [
        "DetectPiiEntities",
        "DetectSentiment",
        "TranslateText",
        "SynthesizeSpeech",
    ]
    children = {
        child["demo"].split("/")[-1]: child for child in result["children"]
    }
    assert children["polly"]["data"]["audio_bytes"] == 6 and port.audio.closed
    assert children["comprehend"]["data"]["text_characters"] == 200
    assert children["comprehend"]["data"]["pii_types"] == ["EMAIL"]
    assert len(run.command("ai-services").ledger.snapshot().reservations) == 4
    assert len(prices.calls) == 4


def test_ai_unpriced_and_denied_paths(tmp_path: Any) -> None:
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        prices=Prices(missing=True),
        root=tmp_path,
    )
    port = Port()
    with use_budget_run(run):
        result = run_ai_services_demo(
            execution="live", settings=Settings(), policy=run.policy, port=port
        )
    assert not port.calls
    assert (
        result["children"][0]["operations"][0]["error_code"]
        == "budget_exceeded"
    )
    run = open_budget_run(
        region="us-east-1",
        policy=ExecutionPolicy(),
        prices=Prices(),
        root=tmp_path,
    )
    with use_budget_run(run):
        denied = run_ai_services_demo(
            execution="live",
            settings=Settings(),
            policy=run.policy,
            port=Port(denied="SynthesizeSpeech"),
        )
    assert "audio_bytes" not in denied["children"][2]["data"]


@pytest.mark.parametrize("denied", ["", "CreateBucket"])
def test_ai_emulator_prepares_media(denied: str) -> None:
    port = Port(denied=denied)
    result = run_ai_services_demo(
        execution="emulator",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=port,
    )
    child = result["children"][3]
    assert child["operations"][0]["phase"] == "setup"
    starts = [
        call for call in port.calls if call[1] == "StartTranscriptionJob"
    ]
    assert bool(starts) is (not denied)
    if starts:
        assert starts[0][2]["Media"]["MediaFileUri"].endswith("silence.wav")


def test_service_row_direct_missing_and_emulator_unpriced() -> None:
    row = service_rows()["comprehend"][0]
    blocked = row.run(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=None,
        command="ai-services",
    )
    assert blocked.outcome["error_code"] == "missing_configuration"
    row = ServiceRow("sts", "GetCallerIdentity", {}, {}, emulator=True)

    class LoopbackPort(Port):
        def call(self, *args: Any, **kwargs: Any) -> AwsResponse:
            answer = super().call(*args, **kwargs)
            return AwsResponse(answer.payload, "http://localhost:4566")

    answer = row.run(
        execution="emulator",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=LoopbackPort(),
        command="identity",
    )
    assert answer.outcome["mode"] == "local_emulator"


def test_ai_missing_snapshot_never_calls_port(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    port = Port()
    answer = run_ai_services_demo(
        execution="live",
        settings=Settings(),
        policy=ExecutionPolicy(),
        port=port,
    )
    assert answer["status"] == "blocked" and not port.calls
