"""Azure OpenAI Service provider (drives the Codex agent).

Routes the Codex agent at an Azure-hosted OpenAI deployment. Azure selects
the model by the deployment name in the URL path, not by the model name in
the request body, so the deployment is configured separately from the model.
Authentication is an ``AZURE_OPENAI_API_KEY`` sent as a Bearer token.

This is distinct from ``FoundryProvider``, which is ``ClaudeCompatible`` and
serves Anthropic models through Azure AI Foundry.
"""

from __future__ import annotations

import os

from openscientist.providers.base import CodexCompatible, CostInfo
from openscientist.settings import get_settings


class AzureOpenAIProvider(CodexCompatible):
    """Azure OpenAI Service as a Codex backend (OpenAI models via Azure)."""

    @property
    def id(self) -> str:
        return "azure-openai"

    @property
    def display_name(self) -> str:
        return "Azure OpenAI Service"

    def validate_required_config(self) -> list[str]:
        s = get_settings().provider
        errors: list[str] = []
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            errors.append("AZURE_OPENAI_API_KEY is required for the Azure OpenAI provider.")
        if not s.azure_openai_resource:
            errors.append(
                "AZURE_OPENAI_RESOURCE is required (the <resource> in "
                "https://<resource>.openai.azure.com)."
            )
        if not s.azure_openai_deployment:
            errors.append(
                "AZURE_OPENAI_DEPLOYMENT is required (the deployment name configured in Azure)."
            )
        return errors

    def get_cost_info(self, lookback_hours: int = 24) -> CostInfo:
        # Azure OpenAI spend is tracked through Azure Cost Management, which the
        # foundry provider already wires up. Report unavailable here.
        return CostInfo(
            provider_name=self.display_name,
            total_spend_usd=None,
            recent_spend_usd=None,
            recent_period_hours=lookback_hours,
            data_lag_note="Azure OpenAI per-key cost tracking is not available.",
        )

    def _base_url(self) -> str:
        # Azure serves the Responses API on the OpenAI v1 surface
        # (.../openai/v1/responses), not under /deployments/<name>/. Codex
        # appends "/responses" to base_url, and the deployment is sent as the
        # model name in the request body (see codex_model_name).
        return f"https://{get_settings().provider.azure_openai_resource}.openai.azure.com/openai/v1"

    def codex_config_overrides(self) -> list[str]:
        # A [model_providers.azure-openai] TOML table. The key is sent as a
        # Bearer token (env_key). api-version is optional on the v1 surface, so
        # it is pinned only when the operator configures one.
        s = get_settings().provider
        # Codex only supports wire_api = "responses" (the chat wire was removed).
        # stream_max_retries makes codex reconnect through Azure's intermittent
        # streaming disconnects (a known Azure-side timeout, openai/codex#9936),
        # which it otherwise treats as a fatal "stream disconnected" error.
        lines = [
            "[model_providers.azure-openai]",
            'name = "Azure OpenAI Service"',
            f'base_url = "{self._base_url()}"',
            'env_key = "AZURE_OPENAI_API_KEY"',
            'wire_api = "responses"',
            f"stream_max_retries = {s.azure_openai_stream_max_retries}",
        ]
        if s.azure_openai_api_version:
            lines.append(f'query_params = {{ "api-version" = "{s.azure_openai_api_version}" }}')
        return lines

    def codex_model_name(self) -> str | None:
        # The deployment name is sent as the request-body model, which is how
        # Azure routes on the v1 surface. Default to the deployment unless
        # OPENSCIENTIST_MODEL is set.
        s = get_settings().provider
        return s.model or s.azure_openai_deployment

    def codex_model_provider_id(self) -> str:
        return "azure-openai"

    def codex_sdk_env(self) -> dict[str, str]:
        key = os.environ.get("AZURE_OPENAI_API_KEY")
        return {"AZURE_OPENAI_API_KEY": key} if key else {}
