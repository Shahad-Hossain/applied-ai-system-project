from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def test_app_loads_without_errors():
    at = run_app()
    assert not at.exception
    assert at.title[0].value.endswith("VibeMatch 2.0")
    assert not at.session_state["query"]


def test_app_shows_recommendations_for_a_request():
    at = run_app()
    at.text_input[0].set_value("something calm for studying").run()
    assert not at.exception
    text = " ".join(e.label for e in at.expander)
    assert "Library Rain" in text
    assert any("Confidence" == m.label for m in at.metric)


def test_app_shows_message_for_unknown_request():
    at = run_app()
    at.text_input[0].set_value("asdfgh qwerty").run()
    assert not at.exception
    assert any("couldn't find" in w.value for w in at.warning)


def test_gemini_checkbox_is_disabled_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("src.rag.load_env", lambda *a, **k: None)  # ignore a real .env on this machine
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert at.checkbox[0].disabled
