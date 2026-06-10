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
from openscientist.providers import provider_class
from openscientist.providers.base import ClaudeCompatible, CodexCompatible, Provider
from openscientist.settings import get_settings

logger = logging.getLogger(__name__)


def _instantiate_provider(provider_id: str) -> Provider:
    """Construct the provider registered under `provider_id` (validates auth)."""
    return provider_class(provider_id)()


def _agent_class_for_provider_class(cls: type[Provider]) -> type[AbstractAgent[Any]]:
    """The one provider-family -> agent-class dispatch.

    Every other resolver derives from this, so adding a new agent family is a
    single edit here. ClaudeCompatible is preferred when a provider somehow
    implements both families (hypothetical, no real provider does).
    """
    if issubclass(cls, ClaudeCompatible):
        return ClaudeCodeAgent
    if issubclass(cls, CodexCompatible):
        # Deferred import: the codex SDK is only needed on the codex path, so
        # environments without it (e.g. images shipping only the Claude SDK)
        # can still import the factory.
        from openscientist.agent.codex_agent import CodexAgent

        return CodexAgent
    raise ValueError(
        f"Provider {cls.__name__} does not implement a known agent "
        "compatibility family (ClaudeCompatible or CodexCompatible)."
    )


def agent_class_for_provider(provider: Provider) -> type[AbstractAgent[Any]]:
    """The agent class that drives a provider instance."""
    return _agent_class_for_provider_class(type(provider))


def agent_class_for_provider_id(provider_id: str) -> type[AbstractAgent[Any]]:
    """The agent class for a provider id without instantiating anything.

    Lets the web/orchestrator process (no agent instance) reach a backend's
    classmethods, e.g. ``provision_host_prelaunch`` before the agent container
    launches. An unknown id falls back to the Claude agent (UI labelling).
    """
    try:
        cls = provider_class(provider_id)
    except ValueError:
        return ClaudeCodeAgent
    return _agent_class_for_provider_class(cls)


def backend_for_provider(provider: Provider) -> AgentBackend:
    """The agent backend that drives a provider instance."""
    return agent_class_for_provider(provider).backend


def backend_for_provider_id(provider_id: str) -> AgentBackend:
    """The agent backend for a provider id without instantiating it (UI)."""
    return agent_class_for_provider_id(provider_id).backend


def build_agent(config: AgentConfig, provider: Provider) -> AbstractAgent[Provider]:
    """Construct the agent that drives an explicit provider instance.

    Shared by `get_agent` (provider resolved from settings) and the chat path
    (which already holds a provider and needs a single build). The agent reads
    any per-run model override from `config.model_override`.
    """
    agent_cls = agent_class_for_provider(provider)
    logger.info("Using %s with provider %s", agent_cls.__name__, provider.id)
    return agent_cls(config, provider)


def get_agent(config: AgentConfig) -> AbstractAgent[Provider]:
    """Return the agent for the configured provider.

    The active provider is selected by `settings.provider.provider_id`; its
    compatibility family chooses the agent class.
    """
    return build_agent(config, _instantiate_provider(get_settings().provider.provider_id))
