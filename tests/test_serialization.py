import json
from datetime import UTC, datetime, timedelta, timezone
from types import MappingProxyType

import pytest

from awsai_demo.serialization import normalize_timestamps


def test_nested_timestamps_preserve_offsets_and_containers() -> None:
    utc_value = datetime(2025, 10, 15, tzinfo=UTC)
    offset = timezone(timedelta(hours=3))
    local_value = datetime(2025, 10, 15, 3, 1, 2, 123, tzinfo=offset)
    source = MappingProxyType(
        {"records": [{"started": utc_value}], "events": (local_value, None)}
    )

    normalized = normalize_timestamps(source)

    assert normalized == {
        "records": [{"started": "2025-10-15T00:00:00+00:00"}],
        "events": ("2025-10-15T03:01:02.000123+03:00", None),
    }
    assert source["records"][0]["started"] is utc_value
    assert "2025-10-15" in json.dumps(normalized)


def test_unknown_sdk_objects_are_not_silently_stringified() -> None:
    unsupported = object()
    normalized = normalize_timestamps({"value": unsupported})

    assert normalized["value"] is unsupported
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(normalized)
