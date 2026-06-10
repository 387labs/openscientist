"""
Agent factory for OpenScientist.

`get_agent(config)` instantiates the configured provider and returns the
agent that drives its compatibility family: `ClaudeCompatible` providers
get a `ClaudeCodeAgent`, `CodexCompatible` providers a `CodexAgent`.
"""

from __future__ import annotations

import logging
from typing import Any

from openscientist.agent.base import AbstractAgent, AgentBackend, AgentConfig
from openscientist.agent.claude_code_agent import ClaudeCodeAgent
from openscientist.providers.anthropic import AnthropicProvider
from openscientist.providers.azure_openai import AzureOpenAIProvider
from openscientist.providers.base import ClaudeCompatible, CodexCompatible, Provider
from openscientist.providers.bedrock import BedrockProvider
from openscientist.providers.cborg import CborgProvider
from openscientist.providers.foundry import FoundryProvider
from openscientist.providers.ollama import OllamaProvider
from openscientist.providers.openai import OpenAIDirectProvider
from openscientist.providers.vertex import VertexProvider
from openscientist.settings import get_settings

logger = logging.getLogger(__name__)

_PROVIDER_REGISTRY: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider,
    "cborg": CborgProvider,
    "vertex": VertexProvider,
    "bedrock": BedrockProvider,
    "foundry": FoundryProvider,
    "openai": OpenAIDirectProvider,
    "azure-openai": AzureOpenAIProvider,
    "ollama": OllamaProvider,
}


def _instantiate_provider(provider_id: str) -> Provider:
    """Construct the provider registered under `provider_id`."""
    cls = _PROVIDER_REGISTRY.get(provider_id.lower())
    if cls is None:
        valid = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise ValueError(f"Unknown provider {provider_id!r}. Valid options: {valid}")
    return cls()


def backend_for_provider(provider: Provider) -> AgentBackend:
    """Return the agent backend that drives a provider instance.

    The single source of truth for the provider -> backend mapping; callers
    that only have a provider id (e.g. the UI) use ``backend_for_provider_id``.
    """
    return AgentBackend.CODEX if isinstance(provider, CodexCompatible) else AgentBackend.CLAUDE_CODE


def agent_class_for_provider(provider: Provider) -> type[AbstractAgent[Any]]:
    """Return the agent class that drives a provider instance.

    Lets a caller reach the backend's classmethods (e.g. ``chat_system_prompt``)
    before constructing the agent. ``ClaudeCompatible`` is checked first, the
    same precedence ``build_agent`` uses.
    """
    if isinstance(provider, CodexCompatible) and not isinstance(provider, ClaudeCompatible):
        from openscientist.agent.codex_agent import CodexAgent

        return CodexAgent
    return ClaudeCodeAgent


def backend_for_provider_id(provider_id: str) -> AgentBackend:
    """Return the agent backend for a provider id without instantiating it.

    Lets the UI label a past job from its stored provider id without
    constructing the (maybe unconfigured) provider. Unknown ids fall back to
    the Claude backend.
    """
    cls = _PROVIDER_REGISTRY.get(provider_id.lower())
    if cls is not None and issubclass(cls, CodexCompatible):
        return AgentBackend.CODEX
    return AgentBackend.CLAUDE_CODE


def agent_class_for_provider_id(provider_id: str) -> type[AbstractAgent[Any]]:
    """Return the agent class for a provider id without instantiating anything.

    Lets the web/orchestrator process (where no agent instance exists) reach a
    backend's classmethods, e.g. ``provision_host_prelaunch`` before the agent
    container is launched. Unknown ids fall back to the Claude agent.
    """
    cls = _PROVIDER_REGISTRY.get(provider_id.lower())
    if cls is not None and issubclass(cls, CodexCompatible):
        # Deferred import: the codex SDK is only needed on the codex path.
        from openscientist.agent.codex_agent import CodexAgent

        return CodexAgent
    return ClaudeCodeAgent


def build_agent(config: AgentConfig, provider: Provider) -> AbstractAgent[Provider]:
    """Construct the agent that drives an explicit provider instance.

    Shared by `get_agent` (which resolves the provider from settings) and the
    chat path (which already holds a provider and needs a single build).
    ClaudeCompatible is checked first: a hypothetical multi-family provider
    prefers the mature Claude path until a real hybrid case appears.
    """
    if isinstance(provider, ClaudeCompatible):
        logger.info("Using ClaudeCodeAgent with provider %s", provider.id)
        return ClaudeCodeAgent(config, provider, model_override=config.model_override)
    if isinstance(provider, CodexCompatible):
        # Deferred import: the codex SDK is only needed on the codex path, so
        # environments without it (e.g. images that ship only the Claude SDK)
        # can still import the factory.
        from openscientist.agent.codex_agent import CodexAgent

        logger.info("Using CodexAgent with provider %s", provider.id)
        return CodexAgent(config, provider)
    raise ValueError(
        f"Provider {type(provider).__name__} does not implement a known agent "
        "compatibility family (ClaudeCompatible or CodexCompatible)."
    )


def get_agent(config: AgentConfig) -> AbstractAgent[Provider]:
    """Return the agent for the configured provider.

    The active provider is selected by `settings.provider.provider_id`. The
    agent class is chosen by the provider's compatibility family.
    """
    return build_agent(config, _instantiate_provider(get_settings().provider.provider_id))
