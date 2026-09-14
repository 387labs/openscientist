"""llama.cpp context-window probe (``GET /props``). Shared behavior is in ``test_self_hosted_providers.py``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from openscientist.providers import llamacpp as llamacpp_mod
from openscientist.providers.llamacpp import _probe_llamacpp_context_tokens


def _resp(payload: object) -> MagicMock:
    """A fake requests.Response returning ``payload`` from .json()."""
    r = MagicMock()
    r.raise_for_status.return_value = None
    r.json.return_value = payload
    return r


def test_probe_reads_n_ctx_from_props() -> None:
    payload = {"default_generation_settings": {"n_ctx": 8192}}
    with patch.object(llamacpp_mod.requests, "get", return_value=_resp(payload)) as get:
        assert _probe_llamacpp_context_tokens("http://h:8080/v1", None) == 8192
    # /props is at the server root, not under /v1.
    get.assert_called_once_with("http://h:8080/props", headers={}, timeout=5)


def test_probe_strips_v1_and_trailing_slash() -> None:
    payload = {"default_generation_settings": {"n_ctx": 4096}}
    with patch.object(llamacpp_mod.requests, "get", return_value=_resp(payload)) as get:
        assert _probe_llamacpp_context_tokens("http://h:8080/v1/", None) == 4096
    get.assert_called_once_with("http://h:8080/props", headers={}, timeout=5)


def test_probe_targets_props_at_root_for_a_base_without_v1() -> None:
    """A base without ``/v1`` (the proxy) still probes ``/props`` at the root."""
    payload = {"default_generation_settings": {"n_ctx": 16384}}
    with patch.object(llamacpp_mod.requests, "get", return_value=_resp(payload)) as get:
        assert _probe_llamacpp_context_tokens("http://openscientist:8081", None) == 16384
    get.assert_called_once_with("http://openscientist:8081/props", headers={}, timeout=5)


def test_probe_authenticates_against_a_keyed_server() -> None:
    """A server started with --api-key 401s an unauthenticated probe."""
    payload = {"default_generation_settings": {"n_ctx": 8192}}
    with patch.object(llamacpp_mod.requests, "get", return_value=_resp(payload)) as get:
        assert _probe_llamacpp_context_tokens("http://h:8080/v1", "lk-secret") == 8192
    get.assert_called_once_with(
        "http://h:8080/props",
        headers={"authorization": "Bearer lk-secret"},
        timeout=5,
    )


def test_probe_returns_none_when_generation_settings_absent() -> None:
    """A non-llama.cpp OpenAI server omits the field, which must not crash."""
    with patch.object(llamacpp_mod.requests, "get", return_value=_resp({"model_path": "x"})):
        assert _probe_llamacpp_context_tokens("http://h:8080/v1", None) is None


def test_probe_returns_none_when_n_ctx_absent() -> None:
    payload: dict[str, dict[str, int]] = {"default_generation_settings": {}}
    with patch.object(llamacpp_mod.requests, "get", return_value=_resp(payload)):
        assert _probe_llamacpp_context_tokens("http://h:8080/v1", None) is None


def test_probe_returns_none_on_unexpected_payload_shape() -> None:
    with patch.object(llamacpp_mod.requests, "get", return_value=_resp(["not", "a", "dict"])):
        assert _probe_llamacpp_context_tokens("http://h:8080/v1", None) is None


def test_probe_returns_none_on_non_numeric_n_ctx() -> None:
    payload = {"default_generation_settings": {"n_ctx": "lots"}}
    with patch.object(llamacpp_mod.requests, "get", return_value=_resp(payload)):
        assert _probe_llamacpp_context_tokens("http://h:8080/v1", None) is None


def test_probe_returns_none_on_connection_error() -> None:
    with patch.object(llamacpp_mod.requests, "get", side_effect=requests.ConnectionError("down")):
        assert _probe_llamacpp_context_tokens("http://h:8080/v1", None) is None
