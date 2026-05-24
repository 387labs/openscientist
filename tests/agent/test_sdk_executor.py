"""Unit tests for `SDKAgentExecutor` cutover to the standalone MCP."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from openscientist.agent.sdk_executor import SDKAgentExecutor


def _make_executor(
    tmp_path: Path,
    *,
    data_file: Path | None = None,
    data_files: list[Path] | None = None,
    use_hypotheses: bool = False,
    model_override: str | None = None,
) -> SDKAgentExecutor:
    return SDKAgentExecutor(
        job_dir=tmp_path,
        data_file=data_file,
        system_prompt="test prompt",
        use_hypotheses=use_hypotheses,
        data_files=data_files,
        model_override=model_override,
    )


def test_build_options_uses_stdio_spec_for_openscientist_tools(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    options = executor._build_options()

    mcp_servers = options.mcp_servers
    assert isinstance(mcp_servers, dict)
    cfg = cast(dict[str, Any], mcp_servers["openscientist-tools"])
    assert cfg["type"] == "stdio"
    assert cfg["command"] == sys.executable
    assert cfg["args"] == ["-m", "openscientist_tools"]
    # The env dict carries the per-job overlays to the subprocess; missing it
    # would break the standalone MCP server at startup.
    assert "env" in cfg
    assert cfg["env"]["OPENSCIENTIST_JOB_ID"] == tmp_path.name
    assert cfg["env"]["OPENSCIENTIST_JOB_DIR"] == str(tmp_path)


def test_subprocess_env_passes_through_unrelated_openscientist_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`OPENSCIENTIST_*` env vars set by the parent (provider, model,
    executor settings, etc.) must reach the subprocess unmodified."""
    monkeypatch.setenv("OPENSCIENTIST_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENSCIENTIST_MODEL", "claude-sonnet-test")
    monkeypatch.setenv("OPENSCIENTIST_EXECUTOR_IMAGE", "custom-executor:latest")
    monkeypatch.setenv("OPENSCIENTIST_EXECUTOR_TIMEOUT", "180")

    env = _make_executor(tmp_path)._build_subprocess_env()
    assert env["OPENSCIENTIST_PROVIDER"] == "anthropic"
    assert env["OPENSCIENTIST_MODEL"] == "claude-sonnet-test"
    assert env["OPENSCIENTIST_EXECUTOR_IMAGE"] == "custom-executor:latest"
    assert env["OPENSCIENTIST_EXECUTOR_TIMEOUT"] == "180"


def test_subprocess_env_includes_job_id_and_job_dir(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    env = executor._build_subprocess_env()

    assert env["OPENSCIENTIST_JOB_ID"] == tmp_path.name
    assert env["OPENSCIENTIST_JOB_DIR"] == str(tmp_path)


def test_subprocess_env_inherits_critical_parent_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/db")
    monkeypatch.setenv("OPENSCIENTIST_SECRET_KEY", "test-key")
    monkeypatch.setenv("PATH", "/test/bin:/usr/bin")

    executor = _make_executor(tmp_path)
    env = executor._build_subprocess_env()

    assert env["DATABASE_URL"] == "postgresql+asyncpg://test/db"
    assert env["OPENSCIENTIST_SECRET_KEY"] == "test-key"
    assert env["PATH"] == "/test/bin:/usr/bin"


@pytest.mark.parametrize("use_hypotheses,expected", [(True, "1"), (False, "0")])
def test_subprocess_env_use_hypotheses_flag(
    tmp_path: Path, use_hypotheses: bool, expected: str
) -> None:
    executor = _make_executor(tmp_path, use_hypotheses=use_hypotheses)
    assert executor._build_subprocess_env()["OPENSCIENTIST_USE_HYPOTHESES"] == expected


def test_subprocess_env_data_file_optional(tmp_path: Path) -> None:
    data_file = tmp_path / "primary.csv"
    data_file.write_text("col\n1\n")

    with_file = _make_executor(tmp_path, data_file=data_file)._build_subprocess_env()
    assert with_file["OPENSCIENTIST_DATA_FILE"] == str(data_file)

    without_file = _make_executor(tmp_path, data_file=None)._build_subprocess_env()
    assert "OPENSCIENTIST_DATA_FILE" not in without_file


def test_subprocess_env_data_files_pathsep_joined(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    executor = _make_executor(tmp_path, data_files=[a, b])
    env = executor._build_subprocess_env()
    assert env["OPENSCIENTIST_DATA_FILES"] == f"{a}{os.pathsep}{b}"


def test_subprocess_env_data_files_empty_unsets_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENSCIENTIST_DATA_FILES", "/should/be/removed")
    executor = _make_executor(tmp_path, data_files=None)
    env = executor._build_subprocess_env()
    assert "OPENSCIENTIST_DATA_FILES" not in env


def test_build_options_passes_cwd_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _make_executor(tmp_path, model_override="custom-model")
    options = executor._build_options()

    assert options.cwd == str(tmp_path)
    assert options.model == "custom-model"


def test_init_does_not_call_build_tool_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cutover removed the eager `build_tool_list` call from __init__.

    Tools now live in the standalone subprocess; constructing the executor
    must not trigger the in-process tool registry.
    """

    def _boom(*_a: object, **_kw: object) -> None:
        raise AssertionError("build_tool_list must not be called from __init__")

    monkeypatch.setattr("openscientist.tools.registry.build_tool_list", _boom)
    _make_executor(tmp_path)  # would raise if build_tool_list ran


async def test_built_spec_spawns_subprocess_that_lists_all_tools(
    tmp_path: Path,
    test_database_url: str,
    _apply_migrations_once: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: the spec the executor builds must actually spawn a working
    `openscientist_tools` subprocess whose tool list matches what the agent
    is supposed to see. Bypasses the SDK and connects directly via MCP
    stdio so we exercise the spec wiring without needing an LLM."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    monkeypatch.setenv("DATABASE_URL", test_database_url)

    executor = _make_executor(tmp_path, use_hypotheses=True)
    options = executor._build_options()
    mcp_servers = options.mcp_servers
    assert isinstance(mcp_servers, dict)
    cfg = cast(dict[str, Any], mcp_servers["openscientist-tools"])

    params = StdioServerParameters(
        command=cfg["command"],
        args=cfg["args"],
        env=cfg["env"],
        cwd=str(tmp_path),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}

    # The standalone server registers all 13 agent tools plus the
    # `ping` smoke tool. With `use_hypotheses=True` (set on the spec env),
    # add_hypothesis and update_hypothesis are included.
    assert "ping" in names
    assert {
        "read_document",
        "search_pubmed",
        "set_status",
        "set_job_title",
        "save_iteration_summary",
        "set_consensus_answer",
        "update_knowledge_state",
        "add_hypothesis",
        "update_hypothesis",
        "execute_code",
    } <= names
