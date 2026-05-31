"""Tests for `OpenAIDirectProvider` (the first CodexCompatible provider)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from openscientist.providers.base import CodexCompatible
from openscientist.providers.openai import OpenAIDirectProvider


@pytest.fixture(autouse=True)
def _auth_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic auth: no API key, but a fake codex home with auth.json so
    construction validates the same way in CI as locally."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    home = tmp_path_factory.mktemp("codexhome")
    (home / ".codex").mkdir()
    (home / ".codex" / "auth.json").write_text("{}")
    monkeypatch.setattr("openscientist.providers.openai.Path.home", lambda: home)


def test_is_codex_compatible() -> None:
    assert isinstance(OpenAIDirectProvider(), CodexCompatible)


def test_identity_and_codex_hooks() -> None:
    p = OpenAIDirectProvider()
    assert p.id == "openai"
    assert p.display_name == "OpenAI API"
    assert p.codex_model_provider_id() == "openai"
    assert p.codex_config_overrides() == []


def test_model_name_is_none_without_explicit_model() -> None:
    # No forced default: codex picks the account/config default.
    assert OpenAIDirectProvider().codex_model_name() is None


def test_codex_sdk_env_reflects_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    assert OpenAIDirectProvider().codex_sdk_env() == {}
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert OpenAIDirectProvider().codex_sdk_env() == {"OPENAI_API_KEY": "sk-test"}


def test_valid_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert OpenAIDirectProvider().validate_required_config() == []


def test_valid_with_codex_auth_json(tmp_path: Path) -> None:
    auth = tmp_path / ".codex" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text("{}")
    with patch("openscientist.providers.openai.Path.home", return_value=tmp_path):
        assert OpenAIDirectProvider().validate_required_config() == []


def test_valid_with_codex_auth_host_path(tmp_path: Path) -> None:
    # In the agent container there is no ~/.codex and no key, but the runner
    # provisions auth into the per-job CODEX_HOME and signals it via this path.
    with patch("openscientist.providers.openai.Path.home", return_value=tmp_path):
        with patch("openscientist.providers.openai.get_settings") as mock_settings:
            mock_settings.return_value.provider.codex_auth_host_path = "/host/auth.json"
            assert OpenAIDirectProvider().validate_required_config() == []


def test_invalid_without_any_auth(tmp_path: Path) -> None:
    # No API key, no ~/.codex/auth.json -> Provider.__init__ validates and raises.
    with patch("openscientist.providers.openai.Path.home", return_value=tmp_path):
        with pytest.raises(ValueError, match="auth"):
            OpenAIDirectProvider()


def test_get_cost_info_unavailable() -> None:
    info = OpenAIDirectProvider().get_cost_info()
    assert info.total_spend_usd is None
    assert info.recent_spend_usd is None


def test_get_provider_selects_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """`provider_id="openai"` resolves to OpenAIDirectProvider via the factory."""
    from openscientist.providers import get_provider
    from openscientist.settings import clear_settings_cache

    monkeypatch.setenv("OPENSCIENTIST_PROVIDER", "openai")
    clear_settings_cache()
    try:
        assert isinstance(get_provider(), OpenAIDirectProvider)
    finally:
        clear_settings_cache()
