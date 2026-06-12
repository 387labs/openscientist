"""Tests for the ModelProfile abstraction and context-window resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from openscientist import models
from openscientist.models import (
    ModelProfile,
    _known_context_tokens,
    _ollama_http_base,
    resolve_model_profile,
)


def _settings(**provider_kw) -> SimpleNamespace:
    defaults = dict(
        provider_id="ollama",
        model=None,
        ollama_model="gpt-oss:120b",
        ollama_base_url="http://host:11434/v1",
        model_context_tokens=None,
    )
    defaults.update(provider_kw)
    return SimpleNamespace(provider=SimpleNamespace(**defaults))


def test_ollama_http_base_strips_v1():
    assert _ollama_http_base("http://host:11434/v1") == "http://host:11434"
    assert _ollama_http_base("http://host:11434/v1/") == "http://host:11434"
    assert _ollama_http_base("http://host:11434") == "http://host:11434"


def test_known_context_tokens_prefix_match():
    assert _known_context_tokens("claude-sonnet-4-6") == 200_000
    assert _known_context_tokens("gpt-4o-mini") == 128_000
    assert _known_context_tokens("totally-unknown-model") is None


def test_explicit_override_wins():
    with patch.object(models, "get_settings", return_value=_settings(model_context_tokens=65536)):
        profile = resolve_model_profile()
    assert profile.context_window_tokens == 65536


def test_ollama_probe_used_when_no_override():
    with (
        patch.object(models, "get_settings", return_value=_settings()),
        patch.object(models, "_probe_ollama_context_tokens", return_value=131072) as probe,
    ):
        profile = resolve_model_profile()
    assert profile.context_window_tokens == 131072
    assert profile.id == "gpt-oss:120b"
    probe.assert_called_once_with("http://host:11434/v1", "gpt-oss:120b")


def test_falls_back_to_known_table_then_default():
    # Non-ollama provider with a known model: table is used (no probe).
    s = _settings(provider_id="anthropic", model="claude-sonnet-4-6")
    with patch.object(models, "get_settings", return_value=s):
        assert resolve_model_profile().context_window_tokens == 200_000

    # Unknown model, no override, non-ollama: conservative default.
    s2 = _settings(provider_id="anthropic", model="mystery-model")
    with patch.object(models, "get_settings", return_value=s2):
        assert resolve_model_profile().context_window_tokens == models._DEFAULT_CONTEXT_TOKENS


def test_probe_returns_none_falls_through_to_default():
    with (
        patch.object(models, "get_settings", return_value=_settings(ollama_model="weird:tag")),
        patch.object(models, "_probe_ollama_context_tokens", return_value=None),
    ):
        profile = resolve_model_profile()
    assert profile.context_window_tokens == models._DEFAULT_CONTEXT_TOKENS


def test_profile_is_frozen():
    p = ModelProfile(id="m", context_window_tokens=1000)
    try:
        p.context_window_tokens = 2  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ModelProfile should be frozen")
