from importlib import metadata
from pathlib import Path

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from awsai_demo.contracts import ContractError, result
from awsai_demo.credentials import SelectedSession
from awsai_demo.doctor import (
    AwsProbePort,
    DoctorCheck,
    DoctorProbeResult,
    DoctorReport,
    ProbeCall,
    ProbePort,
    inspect_configuration,
    installed_package_versions,
    run_doctor,
)
from awsai_demo.policy import ExecutionPolicy
from awsai_demo.runtime import Settings


class FakeProbe:
    def __init__(self) -> None:
        self.called = False

    def probe(self, settings: Settings) -> DoctorProbeResult:
        self.called = True
        return DoctorProbeResult(
            checks=(DoctorCheck("sts", "ok", f"region {settings.region}"),),
            operations=(),
        )


class FakeAwsSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []

    def client(self, service_name: str, **kwargs: object) -> "FakeAwsClient":
        region = str(kwargs["region_name"])
        self.calls.append((service_name, region, kwargs["config"]))
        return FakeAwsClient(service_name, region)


class FakeAwsClient:
    def __init__(self, service_name: str, region: str) -> None:
        self._service_name = service_name
        self.meta = FakeMeta(f"https://{service_name}.{region}.example")

    def get_caller_identity(self) -> dict[str, object]:
        return {"Account": "123456789012"}

    def list_foundation_models(self) -> dict[str, object]:
        return {"modelSummaries": [{}, {}]}

    def list_agent_runtimes(self) -> dict[str, object]:
        return {"agentRuntimes": []}

    def list_recommendations(self) -> dict[str, object]:
        return {"recommendations": []}

    def list_vector_buckets(self) -> dict[str, object]:
        return {"vectorBuckets": []}

    def list_workflow_definitions(self) -> dict[str, object]:
        return {"workflowDefinitions": []}

    def list_sessions(self) -> dict[str, object]:
        return {"sessionSummaries": []}


class FakeMeta:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url


class TimeoutSession(FakeAwsSession):
    def client(self, service_name: str, **kwargs: object) -> "TimeoutClient":
        region = str(kwargs["region_name"])
        self.calls.append((service_name, region, kwargs["config"]))
        return TimeoutClient(service_name, region)


class TimeoutClient(FakeAwsClient):
    def list_foundation_models(self) -> dict[str, object]:
        raise ReadTimeoutError(
            endpoint_url=self.meta.endpoint_url, error="slow"
        )


class DeniedSession(FakeAwsSession):
    def client(self, service_name: str, **kwargs: object) -> "DeniedClient":
        region = str(kwargs["region_name"])
        self.calls.append((service_name, region, kwargs["config"]))
        return DeniedClient(service_name, region)


class DeniedClient(FakeAwsClient):
    def list_foundation_models(self) -> dict[str, object]:
        raise ClientError(
            {
                "Error": {"Code": "AccessDeniedException"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "ListFoundationModels",
        )


class DeniedStsSession(FakeAwsSession):
    def client(self, service_name: str, **kwargs: object) -> "DeniedStsClient":
        region = str(kwargs["region_name"])
        self.calls.append((service_name, region, kwargs["config"]))
        return DeniedStsClient(service_name, region)


class DeniedStsClient(FakeAwsClient):
    def get_caller_identity(self) -> dict[str, object]:
        raise ClientError(
            {
                "Error": {"Code": "AccessDenied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "GetCallerIdentity",
        )


class BrokenSession(FakeAwsSession):
    def client(self, service_name: str, **kwargs: object) -> "BrokenClient":
        region = str(kwargs["region_name"])
        self.calls.append((service_name, region, kwargs["config"]))
        return BrokenClient(service_name, region)


class BrokenClient(FakeAwsClient):
    def list_foundation_models(self) -> dict[str, object]:
        message = "unexpected"
        raise RuntimeError(message)


class NoEndpointSession(FakeAwsSession):
    def client(
        self, service_name: str, **kwargs: object
    ) -> "NoEndpointClient":
        region = str(kwargs["region_name"])
        self.calls.append((service_name, region, kwargs["config"]))
        return NoEndpointClient()


class NoEndpointClient:
    meta = object()

    def list_foundation_models(self) -> dict[str, object]:
        return {"modelSummaries": []}


class ScalarResponseSession(FakeAwsSession):
    def client(
        self, service_name: str, **kwargs: object
    ) -> "ScalarResponseClient":
        region = str(kwargs["region_name"])
        self.calls.append((service_name, region, kwargs["config"]))
        return ScalarResponseClient(service_name, region)


class ScalarResponseClient(FakeAwsClient):
    def list_foundation_models(self) -> object:
        return "ok"


def test_inspect_configuration_reports_presence_without_secret_values() -> (
    None
):
    report = inspect_configuration(
        Settings(
            region="us-west-2",
            aws_creds_file_path=None,
            localstack_endpoint=(
                "http://user:secret@localhost:4566/path?token=abc"
            ),
            openai_api_key="secret-key",
        ),
    )

    details = {check.name: check.detail for check in report.checks}
    assert details["region"] == "us-west-2"
    assert details["credential_source"] == "default_chain"
    assert details["localstack_endpoint"] == "http://localhost:4566"
    assert details["openai_api_key"] == "configured"
    assert "secret-key" not in repr(report)
    assert "secret" not in repr(report)
    assert report.status == "warning"


def test_inspect_configuration_sanitizes_endpoint_without_port() -> None:
    report = inspect_configuration(
        Settings(localstack_endpoint="http://localhost"),
    )

    details = {check.name: check.detail for check in report.checks}
    assert details["localstack_endpoint"] == "http://localhost"


def test_inspect_configuration_hides_unparseable_endpoint() -> None:
    report = inspect_configuration(
        Settings(localstack_endpoint="not a url"),
    )

    details = {check.name: check.detail for check in report.checks}
    assert details["localstack_endpoint"] == "configured"


def test_inspect_configuration_reports_profile_and_csv_sources() -> None:
    profile_report = inspect_configuration(Settings(aws_profile="demo"))
    csv_report = inspect_configuration(
        Settings(aws_creds_file_path=Path("synthetic.csv")),
    )

    profile_details = {
        check.name: check.detail for check in profile_report.checks
    }
    csv_details = {check.name: check.detail for check in csv_report.checks}
    assert profile_details["credential_source"] == "profile"
    assert csv_details["credential_source"] == "csv_file"


def test_run_doctor_without_probe_does_not_call_port() -> None:
    probe = FakeProbe()

    report = run_doctor(Settings(), probe=False, probe_port=probe)

    assert probe.called is False
    assert report.probed is False


def test_run_doctor_probe_without_port_is_blocked() -> None:
    report = run_doctor(Settings(), probe=True)

    assert report.probed is False
    assert report.status == "blocked"
    assert report.checks[-1].name == "probe_port"


def test_run_doctor_uses_injected_probe_only_when_requested() -> None:
    probe = FakeProbe()

    report = run_doctor(
        Settings(region="us-east-2"), probe=True, probe_port=probe
    )

    assert probe.called is True
    assert report.probed is True
    assert report.checks[-1] == DoctorCheck("sts", "ok", "region us-east-2")
    assert report.operations == ()


def test_run_doctor_builds_default_probe_from_selected_session() -> None:
    session = FakeAwsSession()
    selected = SelectedSession(
        session=session, source="none", region="us-east-2"
    )
    report = run_doctor(
        Settings(region="us-east-2"),
        probe=True,
        selected_session=selected,
    )

    assert report.probed is True
    assert session.calls[0][:2] == ("sts", "us-east-2")
    assert report.operations[0]["mode"] == "live_identity"
    assert report.operations[1]["mode"] == "live_service"
    assert report.operations[1]["endpoint_url"] == (
        "https://bedrock.us-east-2.example"
    )
    assert session.calls[0][2].retries["total_max_attempts"] == 2


def test_aws_probe_port_respects_wall_clock_before_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeAwsSession()
    selected = SelectedSession(
        session=session, source="none", region="us-east-1"
    )
    port = AwsProbePort(
        selected,
        calls=(ProbeCall("bedrock", "ListFoundationModels", "live_service"),),
        policy=ExecutionPolicy(max_wall_seconds=120),
    )
    times = iter((0.0, 121.0))
    monkeypatch.setattr(
        "awsai_demo.doctor.time.monotonic", lambda: next(times)
    )

    result = port.probe(Settings())

    assert session.calls == []
    assert result.checks[0].status == "blocked"
    assert result.operations[0]["mode"] == "not_run"
    assert result.operations[0]["transport"] == "none"
    assert result.operations[0]["response_received"] is False
    assert result.operations[0]["request_validated"] is True


def test_doctor_report_assembles_result_for_timeout_before_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeAwsSession()
    selected = SelectedSession(
        session=session, source="none", region="us-east-1"
    )
    port = AwsProbePort(
        selected,
        calls=(ProbeCall("bedrock", "ListFoundationModels", "live_service"),),
        policy=ExecutionPolicy(max_wall_seconds=120),
    )
    times = iter((0.0, 121.0))
    monkeypatch.setattr(
        "awsai_demo.doctor.time.monotonic", lambda: next(times)
    )

    report = run_doctor(Settings(), probe=True, probe_port=port)
    built = result(
        demo="doctor",
        technology="doctor",
        lane="operations",
        lifecycle_refs=["bedrock"],
        requested_execution="live",
        headline="doctor probe",
        operations=report.operations,
    )

    assert built["mode"] == "not_run"
    assert built["operations"][0]["transport"] == "none"


def test_doctor_report_assembles_result_for_timeout_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeAwsSession()
    selected = SelectedSession(
        session=session, source="none", region="us-east-1"
    )
    port = AwsProbePort(
        selected,
        calls=(
            ProbeCall("sts", "GetCallerIdentity", "live_identity"),
            ProbeCall("bedrock", "ListFoundationModels", "live_service"),
        ),
        policy=ExecutionPolicy(max_wall_seconds=120),
    )
    times = iter((0.0, 0.0, 0.0, 121.0))
    monkeypatch.setattr(
        "awsai_demo.doctor.time.monotonic", lambda: next(times)
    )

    report = run_doctor(Settings(), probe=True, probe_port=port)
    built = result(
        demo="doctor",
        technology="doctor",
        lane="operations",
        lifecycle_refs=["bedrock"],
        requested_execution="live",
        headline="doctor probe",
        operations=report.operations,
    )

    assert built["mode"] == "live_identity"
    assert built["operations"][0]["mode"] == "live_identity"
    assert built["operations"][1]["mode"] == "not_run"


def test_aws_probe_port_rejects_success_without_endpoint() -> None:
    selected = SelectedSession(
        session=NoEndpointSession(),
        source="none",
        region="us-east-1",
    )
    port = AwsProbePort(
        selected,
        calls=(ProbeCall("bedrock", "ListFoundationModels", "live_service"),),
    )

    with pytest.raises(ContractError, match="non-loopback endpoint"):
        port.probe(Settings())


def test_aws_probe_port_reports_non_mapping_response() -> None:
    selected = SelectedSession(
        session=ScalarResponseSession(),
        source="none",
        region="us-east-1",
    )
    port = AwsProbePort(
        selected,
        calls=(ProbeCall("bedrock", "ListFoundationModels", "live_service"),),
    )

    result = port.probe(Settings())

    assert result.checks[0].detail == "ListFoundationModels: response"


def test_aws_probe_port_records_timeout_attempt_failed() -> None:
    selected = SelectedSession(
        session=TimeoutSession(),
        source="none",
        region="us-east-1",
    )
    port = AwsProbePort(
        selected,
        calls=(ProbeCall("bedrock", "ListFoundationModels", "live_service"),),
    )

    result = port.probe(Settings())

    assert result.checks[0].status == "blocked"
    assert result.operations[0]["mode"] == "attempt_failed"
    assert result.operations[0]["response_received"] is False


def test_aws_probe_port_records_403_as_blocked_response() -> None:
    selected = SelectedSession(
        session=DeniedSession(),
        source="none",
        region="us-east-1",
    )
    port = AwsProbePort(
        selected,
        calls=(ProbeCall("bedrock", "ListFoundationModels", "live_service"),),
    )

    result = port.probe(Settings())

    assert result.checks[0].status == "blocked"
    assert result.operations[0]["mode"] == "live_service"
    assert result.operations[0]["status"] == "blocked"
    assert result.operations[0]["response_received"] is True
    assert result.operations[0]["http_status"] == 403
    assert result.operations[0]["error_code"] == "authorization_denied"


def test_aws_probe_port_records_denied_sts_as_live_service() -> None:
    selected = SelectedSession(
        session=DeniedStsSession(),
        source="none",
        region="us-east-1",
    )
    port = AwsProbePort(
        selected,
        calls=(ProbeCall("sts", "GetCallerIdentity", "live_identity"),),
    )

    result = port.probe(Settings())

    assert result.operations[0]["mode"] == "live_service"
    assert result.operations[0]["service"] == "sts"
    assert result.operations[0]["response_received"] is True


def test_aws_probe_port_reraises_unknown_exception() -> None:
    selected = SelectedSession(
        session=BrokenSession(),
        source="none",
        region="us-east-1",
    )
    port = AwsProbePort(
        selected,
        calls=(ProbeCall("bedrock", "ListFoundationModels", "live_service"),),
    )

    with pytest.raises(RuntimeError, match="unexpected"):
        port.probe(Settings())


def test_aws_probe_port_rejects_required_input_shape() -> None:
    selected = SelectedSession(
        session=FakeAwsSession(),
        source="none",
        region="us-east-1",
    )

    with pytest.raises(ValueError, match="requires input"):
        AwsProbePort(
            selected,
            calls=(
                ProbeCall("bedrock-runtime", "CountTokens", "live_service"),
            ),
        )


def test_doctor_report_status_rank() -> None:
    assert DoctorReport((DoctorCheck("a", "ok", ""),), False).status == "ok"
    assert (
        DoctorReport((DoctorCheck("a", "warning", ""),), False).status
        == "warning"
    )
    assert (
        DoctorReport((DoctorCheck("a", "blocked", ""),), False).status
        == "blocked"
    )


def test_fake_probe_conforms_to_protocol() -> None:
    probe: ProbePort = FakeProbe()

    assert probe.probe(Settings()).checks[0].name == "sts"


def test_doctor_exposes_installed_versions_without_probing() -> None:
    from awsai_demo.cli import doctor_result

    report = doctor_result(Settings(), False)

    assert report["data"]["packages"] == {
        "boto3": metadata.version("boto3"),
        "strands_agents": metadata.version("strands-agents"),
        "bedrock_agentcore": metadata.version("bedrock-agentcore"),
        "a2a_sdk": metadata.version("a2a-sdk"),
        "mcp": metadata.version("mcp"),
    }
    assert report["operations"] == []
    assert report["data"]["probed"] is False


def test_installed_versions_mark_missing_distribution_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_version = metadata.version

    def missing_mcp(distribution: str) -> str:
        if distribution == "mcp":
            raise metadata.PackageNotFoundError(distribution)
        return read_version(distribution)

    monkeypatch.setattr(metadata, "version", missing_mcp)

    versions = installed_package_versions()

    assert versions["mcp"] is None
    assert versions["boto3"] == read_version("boto3")
