"""Tests for building the chat model against OpenRouter."""

import pytest

from src.modules import llm


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENROUTER_BASE_URL",
        "LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_a_missing_key_is_a_clear_error():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        llm.build_model()


def test_it_points_at_openrouter_not_openai(monkeypatch):
    """The OpenAI client is reused; only the base URL makes it OpenRouter."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    model = llm.build_model()
    assert "openrouter.ai" in str(model.openai_api_base)
    assert model.model_name == "google/gemini-3-flash-preview"


def test_the_model_is_swappable_without_a_code_change(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-3.7-flash")
    assert llm.build_model().model_name == "google/gemini-3.7-flash"
    assert llm.model_label() == "google/gemini-3.7-flash"


def test_temperature_defaults_low_for_repeatable_summaries(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert llm.build_model().temperature == llm.DEFAULT_TEMPERATURE

    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    assert llm.build_model().temperature == 0.0


def test_the_key_is_not_leaked_into_headers(monkeypatch):
    """Attribution headers are public; the key belongs in Authorization only."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    model = llm.build_model()
    assert "sk-or-secret" not in str(model.default_headers)


def test_openrouter_attribution_headers_are_set(monkeypatch):
    """OpenRouter attributes usage to these; they carry no credentials."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    headers = llm.build_model().default_headers
    assert headers["X-Title"] == "news-agent"
    assert headers["HTTP-Referer"].startswith("https://")
