"""Ollama provider (drives the Codex agent against a local model).

Routes the Codex agent at a locally hosted Ollama server through its
OpenAI-compatible Responses endpoint (default
``http://localhost:11434/v1``), which serves open-weight models such as
``gpt-oss:20b`` with tool calling. Ollama is local and keyless, so codex
is told the provider needs no OpenAI auth (``requires_openai_auth =
false``) and no API key is sent.

Because the server runs on the host, the base URL must be reachable from
wherever the agent runs. In-process on the dev box ``localhost`` works
directly. From inside the agent container, point ``OLLAMA_BASE_URL`` at
the host (for example ``http://host.docker.internal:11434/v1``).
"""

from __future__ import annotations

import logging
import os

import requests

from openscientist.models import ModelProfile, probed_model_profile
from openscientist.providers.base import (
    LLM_PROXY_URL_ENV,
    CodexCompatible,
    CostInfo,
    LlmUpstream,
    OmpModelCatalog,
    self_hosted_codex_provider_table,
    self_hosted_omp_model_catalog,
)
from openscientist.settings import ProviderSettings, get_settings

logger = logging.getLogger(__name__)

#: Served when OPENSCIENTIST_MODEL is unset. Ollama can hold several models and
#: selects one per request, so a default keeps it usable with no configuration.
_DEFAULT_MODEL = "gpt-oss:20b"


def _ollama_http_base(base_url: str) -> str:
    """The Ollama HTTP root from its OpenAI-compatible base URL.

    ``OLLAMA_BASE_URL`` is the OpenAI-compat endpoint (``.../v1``). The native
    ``/api/*`` routes live one level up.
    """
    return base_url.rstrip("/").removesuffix("/v1").rstrip("/")


def _probe_ollama_context_tokens(base_url: str, model_id: str) -> int | None:
    """Read the served context window of a loaded Ollama model, or None.

    ``/api/ps`` reports ``context_length`` for currently-loaded models, which
    reflects the deployment's ``num_ctx`` (e.g. ``OLLAMA_CONTEXT_LENGTH``), the
    number we must budget against. A cold model gets None rather than the
    ``/api/show`` trained maximum: both callers run before the first model call,
    so the trained figure would be the usual answer rather than a rare fallback,
    and it over-budgets a server launched smaller (262144 reported against a
    32768 deployment). ``probed_model_profile`` turns None into a conservative
    budget plus a warning naming ``OPENSCIENTIST_MODEL_CONTEXT_TOKENS``, which
    is the number the operator can actually act on.
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

    return None


class OllamaProvider(CodexCompatible):
    """Local Ollama server as a Codex backend (open-weight models)."""

    @property
    def id(self) -> str:
        return "ollama"

    display_name = "Ollama (local)"

    @classmethod
    def container_env(
        cls, provider: ProviderSettings, *, gcp_credentials_container_path: str | None = None
    ) -> dict[str, str]:
        # The model is forwarded generically as OPENSCIENTIST_MODEL.
        return {"OLLAMA_BASE_URL": provider.ollama_base_url}

    def harness_env(self, *, proxy: str | None) -> dict[str, str]:
        if proxy:
            return super().harness_env(proxy=proxy)
        # Local, keyless: point an OpenAI-family harness straight at Ollama.
        s = get_settings().provider
        return {"OPENAI_BASE_URL": s.ollama_base_url, "OPENAI_API_KEY": "ollama"}

    def validate_required_config(self) -> list[str]:
        # Local and keyless: the base URL and model both have defaults, so
        # there is nothing the operator must supply for the provider to
        # construct. Reachability is a runtime concern, surfaced by the run.
        return []

    def get_cost_info(self, lookback_hours: int = 24) -> CostInfo:
        # Local inference has no per-call API cost. Report zero spend so the
        # budget checks pass cleanly rather than warning on unavailable data.
        return CostInfo(
            provider_name=self.display_name,
            total_spend_usd=0.0,
            recent_spend_usd=0.0,
            recent_period_hours=lookback_hours,
            data_lag_note="Local Ollama inference incurs no API cost.",
        )

    def llm_upstream(self) -> LlmUpstream | None:
        # Keyless: forward to Ollama with no injected auth.
        return LlmUpstream(get_settings().provider.ollama_base_url, {})

    def proxy_env_overrides(self, *, proxy_base_url: str, placeholder: str) -> dict[str, str]:
        # Codex sends the placeholder so the proxy authenticates the container.
        return {"OPENAI_API_KEY": placeholder, LLM_PROXY_URL_ENV: proxy_base_url}

    def codex_config_overrides(self) -> list[str]:
        proxy = os.environ.get(LLM_PROXY_URL_ENV)
        return self_hosted_codex_provider_table(
            provider_id=self.codex_model_provider_id(),
            name=self.display_name,
            base_url=proxy or get_settings().provider.ollama_base_url,
            # Ollama has no credential of its own, so only the proxy keys it.
            keyed=bool(proxy),
        )

    def codex_model_name(self) -> str | None:
        # Ollama holds several models and picks one per request, so a default
        # here keeps it zero-config. It lives with the provider rather than as a
        # second env var that OPENSCIENTIST_MODEL would silently override.
        return get_settings().provider.model or _DEFAULT_MODEL

    def model_profile(self) -> ModelProfile:
        # A self-hosted window is whatever num_ctx the deployment allocates, so
        # probe the live server rather than trusting the model's trained maximum.
        s = get_settings().provider
        return probed_model_profile(
            model_id=self.effective_model_name(),
            override=s.model_context_tokens,
            probe=lambda mid: _probe_ollama_context_tokens(s.ollama_base_url, mid),
            server_name="Ollama",
            provider_logger=logger,
        )

    def probe_context_window(self) -> int | None:
        # Probe the local server directly so the launcher can inject the window.
        model = self.effective_model_name()
        if not model:
            return None
        return _probe_ollama_context_tokens(get_settings().provider.ollama_base_url, model)

    def codex_model_provider_id(self) -> str:
        # Not "ollama": codex reserves that id for its built-in provider.
        return "ollama-local"

    def codex_sdk_env(self) -> dict[str, str]:
        # Keyless: nothing to forward into the codex child environment.
        return {}

    def omp_model_catalog(self, *, context_window: int) -> OmpModelCatalog | None:
        model_id = self.effective_model_name()
        if not model_id:
            return None
        proxy = os.environ.get(LLM_PROXY_URL_ENV)
        # Ollama is keyless, so a credential exists only when the proxy is
        # active and the runner exported the job placeholder.
        key = os.environ.get("OPENAI_API_KEY") if proxy else None
        return self_hosted_omp_model_catalog(
            provider_id=self.id,
            name=self.display_name,
            base_url=proxy or get_settings().provider.ollama_base_url,
            model_id=model_id,
            context_window=context_window,
            api_key=key,
        )
