import json
import importlib
import types
import sys


def test_load_cards_from_json(tmp_path, monkeypatch):
    data = {"two of clubs": 1, "ace of clubs": 0}
    json_path = tmp_path / "model.json"
    json_path.write_text(json.dumps(data))

    dummy_streamlit = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "streamlit", dummy_streamlit)

    module = importlib.import_module("streamlit_app")
    labels = module.load_cards(json_path)
    assert labels == ["Ace of Clubs", "Two of Clubs"]
