"""Tests for the agent factory: provider registry and family dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from openscientist.agent.base import AgentConfig
from openscientist.agent.claude_code_agent import ClaudeCodeAgent
from openscientist.agent.factory import (
    _PROVIDER_REGISTRY,
    _instantiate_provider,
    get_agent,
)
from openscientist.providers.anthropic import AnthropicProvider
from openscientist.providers.base_v2 import ClaudeCompatible, Provider


class _ClaudeStub(ClaudeCompatible):
    @property
    def id(self) -> str:
        return "claude-stub"

    @property
    def display_name(self) -> str:
        return "Claude Stub"

    def validate_required_config(self) -> list[str]:
        return []

    def claude_sdk_env(self) -> dict[str, str]:
        return {}

    def claude_model_name(self) -> str:
        return "stub-model"


class _FamilylessProvider(Provider):
    """A provider that implements no agent compatibility family."""

    @property
    def id(self) -> str:
        return "familyless"

    @property
    def display_name(self) -> str:
        return "Familyless"

    def validate_required_config(self) -> list[str]:
        return []


def test_registry_maps_known_ids() -> None:
    assert _PROVIDER_REGISTRY["anthropic"] is AnthropicProvider
    assert set(_PROVIDER_REGISTRY) == {"anthropic", "cborg", "vertex", "bedrock", "foundry"}


def test_instantiate_provider_unknown_id_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider 'nope'"):
        _instantiate_provider("nope")


def test_get_agent_returns_claude_code_agent(tmp_path: Path) -> None:
    provider = _ClaudeStub()
    config = AgentConfig(job_dir=tmp_path)
    with patch("openscientist.agent.factory._instantiate_provider", return_value=provider):
        agent = get_agent(config)

    assert isinstance(agent, ClaudeCodeAgent)
    assert agent.config is config
    assert agent.provider is provider


def test_get_agent_rejects_provider_without_family(tmp_path: Path) -> None:
    with patch(
        "openscientist.agent.factory._instantiate_provider",
        return_value=_FamilylessProvider(),
    ):
        with pytest.raises(ValueError, match="compatibility family"):
            get_agent(AgentConfig(job_dir=tmp_path))
