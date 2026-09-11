"""vLLM context-window probe (``GET /v1/models`` ``max_model_len``). Shared behavior is in ``test_self_hosted_providers.py``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from openscientist.providers import vllm as vllm_mod
from openscientist.providers.vllm import _probe_vllm_context_tokens


def _resp(payload: object) -> MagicMock:
    """A fake requests.Response returning ``payload`` from .json()."""
    r = MagicMock()
    r.raise_for_status.return_value = None
    r.json.return_value = payload
    return r


def test_probe_reads_max_model_len_for_the_served_model() -> None:
    payload = {
        "data": [
            {"id": "other", "max_model_len": 4096},
            {"id": "Qwen/Qwen3-32B", "max_model_len": 40960},
        ]
    }
    with patch.object(vllm_mod.requests, "get", return_value=_resp(payload)) as get:
        assert _probe_vllm_context_tokens("http://h:8000/v1", "Qwen/Qwen3-32B", None) == 40960
    get.assert_called_once_with("http://h:8000/v1/models", headers={}, timeout=5)


def test_probe_strips_a_trailing_slash_from_the_base_url() -> None:
    payload = {"data": [{"id": "m", "max_model_len": 8192}]}
    with patch.object(vllm_mod.requests, "get", return_value=_resp(payload)) as get:
        assert _probe_vllm_context_tokens("http://h:8000/v1/", "m", None) == 8192
    get.assert_called_once_with("http://h:8000/v1/models", headers={}, timeout=5)


def test_probe_authenticates_against_a_keyed_server() -> None:
    """A server started with --api-key 401s an unauthenticated probe."""
    payload = {"data": [{"id": "m", "max_model_len": 8192}]}
    with patch.object(vllm_mod.requests, "get", return_value=_resp(payload)) as get:
        assert _probe_vllm_context_tokens("http://h:8000/v1", "m", "vk-secret") == 8192
    get.assert_called_once_with(
        "http://h:8000/v1/models",
        headers={"authorization": "Bearer vk-secret"},
        timeout=5,
    )


def test_probe_falls_back_to_the_only_served_model() -> None:
    """A one-model server answers even when the configured id spells it differently."""
    payload = {"data": [{"id": "served-alias", "max_model_len": 32768}]}
    with patch.object(vllm_mod.requests, "get", return_value=_resp(payload)):
        assert _probe_vllm_context_tokens("http://h:8000/v1", "Qwen/Qwen3-32B", None) == 32768


def test_probe_returns_none_when_several_models_and_no_match() -> None:
    payload = {"data": [{"id": "a", "max_model_len": 1}, {"id": "b", "max_model_len": 2}]}
    with patch.object(vllm_mod.requests, "get", return_value=_resp(payload)):
        assert _probe_vllm_context_tokens("http://h:8000/v1", "c", None) is None


def test_probe_returns_none_when_max_model_len_absent() -> None:
    """Older or non-vLLM OpenAI servers omit the field, which must not crash."""
    with patch.object(vllm_mod.requests, "get", return_value=_resp({"data": [{"id": "m"}]})):
        assert _probe_vllm_context_tokens("http://h:8000/v1", "m", None) is None


def test_probe_returns_none_on_unexpected_payload_shape() -> None:
    with patch.object(vllm_mod.requests, "get", return_value=_resp(["not", "a", "dict"])):
        assert _probe_vllm_context_tokens("http://h:8000/v1", "m", None) is None


def test_probe_returns_none_on_non_numeric_max_model_len() -> None:
    payload = {"data": [{"id": "m", "max_model_len": "lots"}]}
    with patch.object(vllm_mod.requests, "get", return_value=_resp(payload)):
        assert _probe_vllm_context_tokens("http://h:8000/v1", "m", None) is None


def test_probe_returns_none_on_connection_error() -> None:
    with patch.object(vllm_mod.requests, "get", side_effect=requests.ConnectionError("down")):
        assert _probe_vllm_context_tokens("http://h:8000/v1", "m", None) is None
