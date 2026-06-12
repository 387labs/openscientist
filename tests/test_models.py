"""Tests for the ModelProfile abstraction and context-window resolution."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from openscientist import models
from openscientist.models import (
    ModelProfile,
    _known_context_tokens,
    _ollama_http_base,
    _probe_ollama_context_tokens,
    resolve_model_profile,
)


def _resp(payload: dict) -> MagicMock:
    """A fake requests.Response returning ``payload`` from .json()."""
    r = MagicMock()
    r.raise_for_status.return_value = None
    r.json.return_value = payload
    return r


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


# --- _probe_ollama_context_tokens (the parsing the budget depends on) ---------


def test_probe_reads_loaded_context_from_api_ps():
    # The loaded model's runtime context (num_ctx) comes from /api/ps.
    payload = {"models": [{"name": "gpt-oss:120b", "context_length": 131072}]}
    with patch.object(models.requests, "get", return_value=_resp(payload)) as get:
        assert _probe_ollama_context_tokens("http://h:11434/v1", "gpt-oss:120b") == 131072
    get.assert_called_once_with("http://h:11434/api/ps", timeout=5)


def test_probe_matches_model_name_prefix():
    # An /api/ps name with a quant suffix still matches the configured id.
    payload = {"models": [{"name": "gpt-oss:120b-q4", "context_length": 40000}]}
    with patch.object(models.requests, "get", return_value=_resp(payload)):
        assert _probe_ollama_context_tokens("http://h:11434/v1", "gpt-oss:120b") == 40000


def test_probe_falls_back_to_api_show_when_not_loaded():
    # Model not loaded -> /api/ps empty -> /api/show model_info context_length.
    with (
        patch.object(models.requests, "get", return_value=_resp({"models": []})),
        patch.object(
            models.requests,
            "post",
            return_value=_resp({"model_info": {"gptoss.context_length": 131072}}),
        ) as post,
    ):
        assert _probe_ollama_context_tokens("http://h:11434/v1", "gpt-oss:120b") == 131072
    post.assert_called_once()


def test_probe_returns_none_on_empty_responses():
    with (
        patch.object(models.requests, "get", return_value=_resp({"models": []})),
        patch.object(models.requests, "post", return_value=_resp({"model_info": {}})),
    ):
        assert _probe_ollama_context_tokens("http://h:11434/v1", "gpt-oss:120b") is None


def test_probe_returns_none_on_connection_error():
    with (
        patch.object(models.requests, "get", side_effect=requests.ConnectionError("down")),
        patch.object(models.requests, "post", side_effect=requests.ConnectionError("down")),
    ):
        assert _probe_ollama_context_tokens("http://h:11434/v1", "gpt-oss:120b") is None


def test_ollama_probe_failure_logs_warning(caplog):
    # A failed probe must not be silent: it collapses the prompt budget.
    with (
        patch.object(models, "get_settings", return_value=_settings(ollama_model="weird:tag")),
        patch.object(models, "_probe_ollama_context_tokens", return_value=None),
        caplog.at_level(logging.WARNING, logger="openscientist.models"),
    ):
        profile = resolve_model_profile()
    assert profile.context_window_tokens == models._DEFAULT_CONTEXT_TOKENS
    assert any("Could not probe" in r.message for r in caplog.records)
