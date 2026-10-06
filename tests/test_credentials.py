from pathlib import Path

import pytest

from awsai_demo.credentials import (
    CsvCredentials,
    SelectedSession,
    boto3_session,
    parse_iam_console_csv,
    select_session,
    session_as_type,
)
from awsai_demo.runtime import Settings


class RecordingSessionFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def __call__(self, **kwargs: str) -> dict[str, str]:
        self.calls.append(kwargs)
        return kwargs


def test_offline_uses_dummy_credentials_without_csv_loader() -> None:
    factory = RecordingSessionFactory()
    selected = select_session(
        execution="offline",
        settings=Settings(aws_profile="real", aws_creds_file_path=Path("x")),
        session_factory=factory,
        csv_loader=lambda path: pytest.fail(f"unexpected read {path}"),
    )

    assert selected.source == "none"
    assert factory.calls == [
        {
            "aws_access_key_id": "AKIDEXAMPLE",
            "aws_secret_access_key": (
                "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
            ),
            "aws_session_token": "offline-session-marker",
            "region_name": "us-east-1",
        },
    ]


def test_emulator_also_uses_dummy_credentials() -> None:
    factory = RecordingSessionFactory()

    selected = select_session(
        execution="emulator",
        settings=Settings(region="us-west-2"),
        session_factory=factory,
    )

    assert selected.source == "none"
    assert selected.region == "us-east-1"
    assert factory.calls[0]["region_name"] == "us-east-1"


def test_live_cli_profile_wins_over_environment_profile() -> None:
    factory = RecordingSessionFactory()

    selected = select_session(
        execution="live",
        settings=Settings(aws_profile="env-profile"),
        cli_profile="cli-profile",
        session_factory=factory,
    )

    assert selected.source == "profile"
    assert factory.calls == [
        {"profile_name": "cli-profile", "region_name": "us-east-1"},
    ]


def test_live_env_profile_wins_over_csv() -> None:
    factory = RecordingSessionFactory()

    selected = select_session(
        execution="live",
        settings=Settings(
            aws_profile="env-profile",
            aws_creds_file_path=Path("creds.csv"),
        ),
        session_factory=factory,
        csv_loader=lambda path: pytest.fail(f"unexpected read {path}"),
    )

    assert selected.source == "profile"
    assert factory.calls == [
        {"profile_name": "env-profile", "region_name": "us-east-1"},
    ]


def test_live_csv_source_uses_injected_loader_only() -> None:
    factory = RecordingSessionFactory()
    csv_text = (
        "\ufeffAccess key ID,Secret access key,Session token\n"
        "AKIAIOSFODNN7EXAMPLE,secret,configured\n"
    )

    selected = select_session(
        execution="live",
        settings=Settings(
            region="us-east-2",
            aws_creds_file_path=Path("synthetic.csv"),
        ),
        session_factory=factory,
        csv_loader=lambda _path: csv_text,
    )

    assert selected.source == "csv_file"
    assert factory.calls == [
        {
            "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "aws_secret_access_key": "secret",
            "aws_session_token": "configured",
            "region_name": "us-east-2",
        },
    ]


def test_live_csv_source_can_read_synthetic_fixture(
    tmp_path: Path,
) -> None:
    factory = RecordingSessionFactory()
    csv_path = tmp_path / "synthetic.csv"
    csv_path.write_text(
        "Access key ID,Secret access key\nAKIAIOSFODNN7EXAMPLE,secret\n",
        encoding="utf-8",
    )

    selected = select_session(
        execution="live",
        settings=Settings(aws_creds_file_path=csv_path),
        session_factory=factory,
    )

    assert selected.source == "csv_file"
    assert factory.calls[0] == {
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "secret",
        "region_name": "us-east-1",
    }


def test_live_default_chain_is_last() -> None:
    factory = RecordingSessionFactory()

    selected = select_session(
        execution="live",
        settings=Settings(region="us-east-2"),
        session_factory=factory,
    )

    assert selected.source == "default_chain"
    assert factory.calls == [{"region_name": "us-east-2"}]


def test_parse_iam_console_csv_accepts_case_and_spacing() -> None:
    parsed = parse_iam_console_csv(
        " access KEY id , secret ACCESS key \n key , secret \n",
    )

    assert parsed == CsvCredentials("key", "secret")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Access key ID,Secret access key\n",
        "Access key ID,Secret access key\none,two\nthree,four\n",
        "Access key ID,Other\none,two\n",
    ],
)
def test_parse_iam_console_csv_rejects_invalid_shapes(text: str) -> None:
    with pytest.raises(ValueError):
        parse_iam_console_csv(text)


def test_session_as_type_casts_selected_session() -> None:
    selected = SelectedSession(
        session={"ok": "yes"}, source="none", region="x"
    )

    assert session_as_type(selected, dict)["ok"] == "yes"


def test_session_as_type_rejects_wrong_type() -> None:
    selected = SelectedSession(session=object(), source="none", region="x")

    with pytest.raises(TypeError, match="wrong type"):
        session_as_type(selected, dict)


def test_selected_session_and_csv_credentials_repr_redacts_values() -> None:
    selected = SelectedSession(
        session={"secret": "value"}, source="none", region="x"
    )
    credentials = CsvCredentials("access", "secret", "token")

    assert "secret" not in repr(selected)
    assert "access" not in repr(credentials)
    assert "token" not in repr(credentials)


def test_boto3_session_factory_imports_lazily() -> None:
    session = boto3_session(region_name="us-east-1")

    assert session.region_name == "us-east-1"
