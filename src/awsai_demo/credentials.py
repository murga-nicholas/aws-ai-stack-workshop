"""Credential-source selection without live credential probes.

Live runs still ask the selected session for credentials locally
before any service call. That lookup does not contact AWS.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from botocore.exceptions import NoCredentialsError, PartialCredentialsError

if TYPE_CHECKING:
    from awsai_demo.contracts import CredentialSource, Execution
    from awsai_demo.runtime import Settings

from awsai_demo.runtime import DEFAULT_REGION

SessionKwargs = dict[str, str]
LIVE_CREDENTIAL_HEADLINE = "Live credentials are not configured."
LIVE_CREDENTIALS_MESSAGE = (
    "No credentials resolved from --profile, AWS_PROFILE, "
    "AWS_CREDS_FILE_PATH, default chain."
)
LIVE_CREDENTIALS_NEXT_STEP = (
    "Set --profile, AWS_PROFILE, AWS_CREDS_FILE_PATH, "
    "or the default chain, then retry the live lane."
)
_DUMMY_ACCESS_KEY_ID = "AKIDEXAMPLE"
_DUMMY_SECRET_PARTS = ("wJalrXUtnFEMI/K7MDENG", "+bPxRfiCYEXAMPLEKEY")
_DUMMY_SESSION_PARTS = ("offline", "session", "marker")


class SessionFactory(Protocol):
    """Factory protocol for boto3-like sessions."""

    def __call__(self, **kwargs: str) -> object:
        """Create and return a session object."""


CsvLoader = Callable[[Path], str]


@dataclass(frozen=True)
class SelectedSession:
    """A selected session and its final credential source."""

    session: object = field(repr=False)
    source: CredentialSource
    region: str


@dataclass(frozen=True)
class CsvCredentials:
    """Credentials parsed from IAM console CSV text."""

    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    session_token: str | None = field(default=None, repr=False)


def select_session(
    *,
    execution: Execution,
    settings: Settings,
    cli_profile: str | None = None,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> SelectedSession:
    """Select one credential source according to the workshop rules."""
    factory = session_factory or boto3_session
    if execution == "offline":
        return SelectedSession(
            session=_dummy_session(factory, settings.region),
            source="none",
            region=settings.region,
        )
    if execution == "emulator":
        return SelectedSession(
            session=_dummy_session(factory, DEFAULT_REGION),
            source="none",
            region=DEFAULT_REGION,
        )
    if cli_profile:
        return _profile_session(factory, cli_profile, settings.region)
    if settings.aws_profile:
        return _profile_session(factory, settings.aws_profile, settings.region)
    if settings.aws_creds_file_path is not None:
        loader = csv_loader or _read_csv_file
        credentials = parse_iam_console_csv(
            loader(settings.aws_creds_file_path)
        )
        return SelectedSession(
            session=factory(
                aws_access_key_id=credentials.access_key_id,
                aws_secret_access_key=credentials.secret_access_key,
                **_session_token_arg(credentials),
                region_name=settings.region,
            ),
            source="csv_file",
            region=settings.region,
        )
    return SelectedSession(
        session=factory(region_name=settings.region),
        source="default_chain",
        region=settings.region,
    )


def parse_iam_console_csv(text: str) -> CsvCredentials:
    """Parse one IAM console CSV row."""
    reader = csv.DictReader(text.lstrip("\ufeff").splitlines())
    if reader.fieldnames is None:
        msg = "CSV credentials must include a header row"
        raise ValueError(msg)
    rows = list(reader)
    if len(rows) != 1:
        msg = "CSV credentials must contain exactly one data row"
        raise ValueError(msg)
    normalized = {
        _normalize_header(key): (value or "").strip()
        for key, value in rows[0].items()
        if key is not None
    }
    access_key_id = normalized.get("accesskeyid", "")
    secret_access_key = normalized.get("secretaccesskey", "")
    if not access_key_id or not secret_access_key:
        msg = "CSV credentials require Access key ID and Secret access key"
        raise ValueError(msg)
    session_token = normalized.get("sessiontoken") or None
    return CsvCredentials(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
    )


def boto3_session(**kwargs: str) -> object:
    """Create a boto3 Session lazily so offline imports stay light."""
    import boto3

    return boto3.Session(**kwargs)


def _credential_getter(session: object) -> Callable[[], object] | None:
    direct = getattr(session, "get_credentials", None)
    if callable(direct):
        return cast("Callable[[], object]", direct)
    internal = getattr(session, "_session", None)
    nested = getattr(internal, "get_credentials", None)
    if callable(nested):
        return cast("Callable[[], object]", nested)
    return None


def _profile_session(
    factory: SessionFactory,
    profile_name: str,
    region: str,
) -> SelectedSession:
    return SelectedSession(
        session=factory(profile_name=profile_name, region_name=region),
        source="profile",
        region=region,
    )


def _dummy_session(factory: SessionFactory, region: str) -> object:
    return factory(
        aws_access_key_id=_DUMMY_ACCESS_KEY_ID,
        aws_secret_access_key="".join(_DUMMY_SECRET_PARTS),
        aws_session_token="-".join(_DUMMY_SESSION_PARTS),
        region_name=region,
    )


def _session_token_arg(credentials: CsvCredentials) -> SessionKwargs:
    if credentials.session_token is None:
        return {}
    return {"aws_session_token": credentials.session_token}


def _normalize_header(value: str) -> str:
    return "".join(value.strip().lower().split())


def _read_csv_file(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def resolved_credential(session: object) -> object | None:
    """Return local credentials, or None when the source has none.

    ``NoCredentialsError`` and ``PartialCredentialsError`` count as
    none. Any other exception propagates. This does not call AWS.
    """
    getter = _credential_getter(session)
    if getter is None:
        return None
    try:
        credential = getter()
    except NoCredentialsError:
        return None
    except PartialCredentialsError:
        return None
    return credential


def live_credentials_resolved(
    *,
    settings: Settings,
    session_factory: SessionFactory | None = None,
    csv_loader: CsvLoader | None = None,
) -> bool:
    """Return whether the selected live source resolved credentials.

    Selection follows ``--profile``, ``AWS_PROFILE``,
    ``AWS_CREDS_FILE_PATH``, then the default chain. The check is
    local and does not call AWS.
    """
    selected = select_session(
        execution="live",
        settings=settings,
        session_factory=session_factory,
        csv_loader=csv_loader,
    )
    return resolved_credential(selected.session) is not None


def session_as_type[T](selected: SelectedSession, expected_type: type[T]) -> T:
    """Return a selected session typed for an adapter boundary."""
    if not isinstance(selected.session, expected_type):
        msg = "selected session has the wrong type"
        raise TypeError(msg)
    return selected.session
