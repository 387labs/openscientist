"""Provider base hierarchy (two-axis Provider x Agent model).

A `Provider` is a model-hosting service (Anthropic, Vertex, Bedrock,
OpenAI, ...). The `ClaudeCompatible` and `CodexCompatible` marker
subclasses carry the wire-format-specific methods that let a provider
be driven by the Claude Code agent or the Codex agent respectively.
Nothing consumes these yet; concrete providers migrate onto them in
later PRs.
"""

from __future__ import annotations

import abc


class Provider(abc.ABC):
    """A model-hosting service. Family-specific behavior lives on the
    marker subclasses below."""

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """Stable identifier used by the factory selector."""

    @property
    @abc.abstractmethod
    def display_name(self) -> str: ...

    @abc.abstractmethod
    def validate_required_config(self) -> list[str]:
        """Return a list of error strings if the provider is
        misconfigured; empty list otherwise."""


class ClaudeCompatible(Provider, abc.ABC):
    """Provider that speaks the Anthropic Messages API and can be driven
    by the Claude Code agent."""

    @abc.abstractmethod
    def claude_sdk_env(self) -> dict[str, str]:
        """Environment variables the claude-agent-sdk CLI must see
        (e.g., ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, AWS_REGION)."""

    @abc.abstractmethod
    def claude_model_name(self) -> str:
        """Model name to pass to ClaudeAgentOptions.model."""


class CodexCompatible(Provider, abc.ABC):
    """Provider that speaks the OpenAI Responses API and can be driven by
    the Codex agent."""

    @abc.abstractmethod
    def codex_config_overrides(self) -> list[str]:
        """`key=value` entries for AppServerConfig.config_overrides."""

    @abc.abstractmethod
    def codex_model_name(self) -> str:
        """Model name to pass to thread_start(model=...)."""

    @abc.abstractmethod
    def codex_model_provider_id(self) -> str:
        """The model_providers.<id> key to pass to
        thread_start(model_provider=...)."""
