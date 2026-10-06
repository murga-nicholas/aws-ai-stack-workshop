from __future__ import annotations

import pytest

from awsai_demo.lineage import load_lineage
from awsai_demo.registry import DEMOS, get_demo, list_demos


def test_complete_catalogue_and_lazy_paths() -> None:
    assert len(DEMOS) == 30
    assert len({demo.name for demo in DEMOS}) == 30
    assert sum(demo.core for demo in DEMOS) == 8
    assert list_demos() == DEMOS
    records = {record.id for record in load_lineage()}
    for demo in DEMOS:
        assert set(demo.lifecycle_refs) <= records
        assert get_demo(demo.name) == demo
        assert demo.module.startswith("awsai_demo.")
        assert demo.entrypoint.endswith("_demo")
    assert all(demo.lane == "models" for demo in list_demos("models"))


def test_unknown_catalogue_input() -> None:
    with pytest.raises(ValueError, match="Unknown demo"):
        get_demo("missing")
    with pytest.raises(ValueError, match="Unknown lane"):
        list_demos("missing")
