"""Model abstraction: intrinsic properties of an LLM, independent of who serves it.

A ``Provider`` is *who* hosts a model (Ollama, Anthropic, Vertex). An ``AbstractAgent``
is *how* we drive it. This module is the third axis: *what* model is in use and its
limits. The property that matters today is the usable context window, which is a
property of the model and its deployment -- NOT of the provider. The same Ollama
server can serve a 4K-context model and a 131072-context one side by side, and the
effective window for a self-hosted model is whatever ``num_ctx`` the deployment
allocates (e.g. ``OLLAMA_CONTEXT_LENGTH``), which can be well below the model's
trained maximum.

``resolve_model_profile`` therefore reads the *real* runtime value where it can:

1. an explicit operator override (``OPENSCIENTIST_MODEL_CONTEXT_TOKENS``),
2. a live probe of a self-hosted Ollama deployment (the actual ``num_ctx``),
3. a small table of known API models,
4. a conservative default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from openscientist.settings import get_settings

logger = logging.getLogger(__name__)

# Conservative fallback when nothing else resolves: small enough that prompt
# budgeting trims aggressively rather than risking a silent context overflow.
_DEFAULT_CONTEXT_TOKENS = 8192

# Known API models (trained windows). Self-hosted models are probed instead,
# because their deployment can cap the window below the trained maximum.
_KNOWN_CONTEXT_TOKENS: dict[str, int] = {
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-3-5-sonnet": 200_000,
    "gpt-4o": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-5": 400_000,
}


@dataclass(frozen=True)
class ModelProfile:
    """Intrinsic, provider-independent properties of an LLM."""

    id: str
    context_window_tokens: int
    # Room to grow: max_output_tokens, supports_tool_use, vision, reasoning, ...


def _ollama_http_base(base_url: str) -> str:
    """The Ollama HTTP root from its OpenAI-compatible base URL.

    ``OLLAMA_BASE_URL`` is the OpenAI-compat endpoint (``.../v1``); the native
    ``/api/*`` routes live one level up.
    """
    return base_url.rstrip("/").removesuffix("/v1").rstrip("/")


def _probe_ollama_context_tokens(base_url: str, model_id: str) -> int | None:
    """Read the actual runtime context window of a loaded Ollama model.

    ``/api/ps`` reports ``context_length`` for currently-loaded models, which
    reflects the deployment's ``num_ctx`` (e.g. ``OLLAMA_CONTEXT_LENGTH``) --
    the number we must budget against. Falls back to ``/api/show`` (the model's
    trained maximum) when the model is not currently loaded. Returns None on any
    failure so the caller can fall back further.
    """
    root = _ollama_http_base(base_url)
    try:
        resp = requests.get(f"{root}/api/ps", timeout=5)
        resp.raise_for_status()
        for m in resp.json().get("models", []):
            name = m.get("name", "")
            if (name == model_id or name.startswith(model_id)) and m.get("context_length"):
                return int(m["context_length"])
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.debug("Ollama /api/ps probe failed: %s", exc)

    try:
        resp = requests.post(f"{root}/api/show", json={"name": model_id}, timeout=5)
        resp.raise_for_status()
        info = resp.json().get("model_info", {})
        for key, value in info.items():
            if key.endswith("context_length") and value:
                return int(value)
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.debug("Ollama /api/show probe failed: %s", exc)

    return None


def _known_context_tokens(model_id: str) -> int | None:
    """Look up a known API model's window by longest-matching prefix."""
    matches = [ctx for prefix, ctx in _KNOWN_CONTEXT_TOKENS.items() if model_id.startswith(prefix)]
    return max(matches) if matches else None


def resolve_model_profile(model_id: str | None = None) -> ModelProfile:
    """Resolve the active model's profile (chiefly its usable context window).

    Resolution order: explicit ``OPENSCIENTIST_MODEL_CONTEXT_TOKENS`` override,
    live Ollama probe (for the ``ollama`` provider), known-model table, then a
    conservative default.
    """
    provider = get_settings().provider
    if not model_id:
        model_id = provider.model
        if not model_id and provider.provider_id == "ollama":
            model_id = provider.ollama_model
        model_id = model_id or "unknown"

    override = provider.model_context_tokens
    if override:
        return ModelProfile(id=model_id, context_window_tokens=int(override))

    if provider.provider_id == "ollama":
        probed = _probe_ollama_context_tokens(provider.ollama_base_url, model_id)
        if probed:
            return ModelProfile(id=model_id, context_window_tokens=probed)

    known = _known_context_tokens(model_id)
    if known:
        return ModelProfile(id=model_id, context_window_tokens=known)

    logger.info(
        "No context window known for model %s; defaulting to %d tokens",
        model_id,
        _DEFAULT_CONTEXT_TOKENS,
    )
    return ModelProfile(id=model_id, context_window_tokens=_DEFAULT_CONTEXT_TOKENS)
