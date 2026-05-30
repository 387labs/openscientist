"""Tests for `CodexAgent` — config/AGENTS.md wiring and the run loop.

The codex SDK is mocked (we patch `codex_agent.Codex`), so no codex
binary or network is required; the tests exercise the agent's own logic:
the written `config.toml`/`AGENTS.md`, thread lifecycle, usage mapping,
and failure handling.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from openai_codex_sdk import Usage

from openscientist.agent.base import AbstractAgent, AgentConfig, TokenUsage
from openscientist.agent.codex_agent import CodexAgent
from tests.helpers import StubCodexProvider


class _Provider(StubCodexProvider):
    """Codex stub with a realistic provider config + auth env."""

    def codex_config_overrides(self) -> list[str]:
        return ["[model_providers.openai]", 'base_url = "https://api.openai.com/v1"']

    def codex_model_provider_id(self) -> str:
        return "openai"

    def codex_model_name(self) -> str:
        return "gpt-test"

    def codex_sdk_env(self) -> dict[str, str]:
        return {"OPENAI_API_KEY": "sk-secret"}


def _agent(tmp_path: Path, **cfg_kwargs: object) -> CodexAgent:
    config = AgentConfig(job_dir=tmp_path, **cfg_kwargs)  # type: ignore[arg-type]
    return CodexAgent(config, _Provider())


def _turn(
    *, final: str = "done", items: list[object] | None = None, usage: Usage | None = None
) -> SimpleNamespace:
    if items is None:
        items = [SimpleNamespace(type="agent_message"), SimpleNamespace(type="command_execution")]
    return SimpleNamespace(items=items, final_response=final, usage=usage)


def _patch_codex(turn: SimpleNamespace) -> tuple[MagicMock, MagicMock]:
    """Return (MockCodex, thread) with `thread.run` returning `turn`."""
    mock_codex_cls = MagicMock(name="Codex")
    thread = MagicMock(name="Thread")
    thread.run = AsyncMock(return_value=turn)
    mock_codex_cls.return_value.start_thread.return_value = thread
    return mock_codex_cls, thread


# ── scaffold (unchanged contract) ──────────────────────────────────────


def test_codex_agent_is_abstract_agent_subclass() -> None:
    assert issubclass(CodexAgent, AbstractAgent)


def test_construct_exposes_config_and_provider(tmp_path: Path) -> None:
    config = AgentConfig(job_dir=tmp_path)
    provider = _Provider()
    agent = CodexAgent(config, provider)
    assert agent.config is config
    assert agent.provider is provider
    assert agent.total_tokens == TokenUsage()


# ── run loop ───────────────────────────────────────────────────────────


async def test_run_iteration_success(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    turn = _turn(
        final="the answer",
        usage=Usage(input_tokens=100, cached_input_tokens=20, output_tokens=50),
    )
    mock_codex_cls, thread = _patch_codex(turn)

    with patch("openscientist.agent.codex_agent.Codex", mock_codex_cls):
        result = await agent.run_iteration("hello")

    assert result.success is True
    assert result.output == "the answer"
    # one non-agent_message item -> one tool call
    assert result.tool_calls == 1
    assert result.transcript == []
    thread.run.assert_awaited_once_with("hello")


async def test_usage_subtraction_accumulates(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    turn = _turn(usage=Usage(input_tokens=100, cached_input_tokens=20, output_tokens=50))
    mock_codex_cls, _ = _patch_codex(turn)

    with patch("openscientist.agent.codex_agent.Codex", mock_codex_cls):
        await agent.run_iteration("hi")

    tu = agent.total_tokens
    assert tu.input_tokens == 80  # 100 - 20 cached
    assert tu.cache_read_tokens == 20
    assert tu.output_tokens == 50
    assert tu.cache_write_tokens == 0
    assert tu.reasoning_tokens == 0


def test_usage_from_payload_math() -> None:
    tu = CodexAgent._usage_from_payload(
        Usage(input_tokens=30, cached_input_tokens=12, output_tokens=7)
    )
    assert tu == TokenUsage(
        input_tokens=18,
        output_tokens=7,
        cache_read_tokens=12,
        cache_write_tokens=0,
        reasoning_tokens=0,
    )


async def test_run_iteration_writes_config_and_agents_md(tmp_path: Path) -> None:
    agent = _agent(tmp_path, system_prompt="do science", use_hypotheses=True)
    mock_codex_cls, _ = _patch_codex(_turn(usage=None))

    with patch("openscientist.agent.codex_agent.Codex", mock_codex_cls):
        await agent.run_iteration("go")

    cfg = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text())
    assert cfg["model_provider"] == "openai"
    assert cfg["model_providers"]["openai"]["base_url"] == "https://api.openai.com/v1"
    mcp = cfg["mcp_servers"]["openscientist-tools"]
    assert mcp["args"] == ["-m", "openscientist_tools"]
    assert mcp["env"]["OPENSCIENTIST_JOB_DIR"] == str(tmp_path)
    assert mcp["env"]["OPENSCIENTIST_USE_HYPOTHESES"] == "1"
    assert (tmp_path / "AGENTS.md").read_text() == "do science"

    # CODEX_HOME points the child at the per-job config home + provider auth
    env = mock_codex_cls.call_args.args[0]["env"]
    assert env["CODEX_HOME"] == str(tmp_path / ".codex")
    assert env["OPENAI_API_KEY"] == "sk-secret"


async def test_no_agents_md_without_system_prompt(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    mock_codex_cls, _ = _patch_codex(_turn(usage=None))
    with patch("openscientist.agent.codex_agent.Codex", mock_codex_cls):
        await agent.run_iteration("go")
    assert not (tmp_path / "AGENTS.md").exists()


async def test_thread_is_reused_then_reset(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    mock_codex_cls, _ = _patch_codex(_turn(usage=None))

    with patch("openscientist.agent.codex_agent.Codex", mock_codex_cls):
        await agent.run_iteration("one")
        await agent.run_iteration("two")  # reuses the cached thread
        assert mock_codex_cls.call_count == 1
        await agent.run_iteration("three", reset_session=True)  # rebuilds
        assert mock_codex_cls.call_count == 2


async def test_run_failure_returns_error_and_clears_thread(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    mock_codex_cls = MagicMock(name="Codex")
    thread = MagicMock(name="Thread")
    thread.run = AsyncMock(side_effect=RuntimeError("boom"))
    mock_codex_cls.return_value.start_thread.return_value = thread

    with patch("openscientist.agent.codex_agent.Codex", mock_codex_cls):
        result = await agent.run_iteration("go")
        assert result.success is False
        assert "boom" in result.error
        assert result.output == ""
        # thread cleared -> next call rebuilds
        thread.run = AsyncMock(return_value=_turn(usage=None))
        mock_codex_cls.return_value.start_thread.return_value = thread
        await agent.run_iteration("retry")
        assert mock_codex_cls.call_count == 2


async def test_shutdown_is_noop(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    mock_codex_cls, _ = _patch_codex(_turn(usage=None))
    with patch("openscientist.agent.codex_agent.Codex", mock_codex_cls):
        await agent.run_iteration("go")
    assert agent._thread is not None
    await agent.shutdown()
    assert agent._thread is None
