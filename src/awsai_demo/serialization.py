"""Normalize SDK timestamps without hiding unsupported result values."""

from collections.abc import Mapping
from datetime import datetime
from typing import Any


def normalize_timestamps(value: Any) -> Any:
    """Copy nested containers and encode datetimes as ISO 8601 strings.

    Leave unknown objects for the final JSON boundary to reject. Demos
    must still consume streams and binary payloads explicitly.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {key: normalize_timestamps(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_timestamps(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_timestamps(item) for item in value)
    return value
