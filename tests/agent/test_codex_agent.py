"""Runtime tests for the `CodexAgent` skeleton.

The real `run_iteration`/`shutdown` land in a later change; here we only
assert the scaffold: it is an `AbstractAgent`, constructs against a
`CodexCompatible` provider, and its abstract methods raise
`NotImplementedError` until implemented.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openscientist.agent.base import AbstractAgent, AgentConfig, TokenUsage
from openscientist.agent.codex_agent import CodexAgent
from tests.helpers import StubCodexProvider


def _agent(tmp_path: Path) -> CodexAgent:
    return CodexAgent(AgentConfig(job_dir=tmp_path), StubCodexProvider())


def test_codex_agent_is_abstract_agent_subclass() -> None:
    assert issubclass(CodexAgent, AbstractAgent)


def test_construct_exposes_config_and_provider(tmp_path: Path) -> None:
    config = AgentConfig(job_dir=tmp_path)
    provider = StubCodexProvider()
    agent = CodexAgent(config, provider)
    assert agent.config is config
    assert agent.provider is provider
    assert agent.total_tokens == TokenUsage()


async def test_run_iteration_raises_not_implemented(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    with pytest.raises(NotImplementedError):
        await agent.run_iteration("hello")


async def test_shutdown_raises_not_implemented(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    with pytest.raises(NotImplementedError):
        await agent.shutdown()
