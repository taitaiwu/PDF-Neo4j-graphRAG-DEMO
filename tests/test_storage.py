import json

import pytest

from manual_graphrag.storage import read_json, write_json


def test_json_round_trip_adds_schema_version(tmp_path) -> None:
    target = write_json(tmp_path / "config.json", {"name": "測試"})
    assert read_json(target) == {"schema_version": 1, "name": "測試"}


def test_read_json_rejects_unknown_schema(tmp_path) -> None:
    target = tmp_path / "config.json"
    target.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(target)
