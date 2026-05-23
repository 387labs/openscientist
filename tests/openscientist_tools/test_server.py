"""End-to-end stdio tests for the standalone tools MCP server."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent


def _server_env(job_dir: Path, **overrides: str) -> dict[str, str]:
    """Build a clean env dict for the spawned subprocess.

    Strips any pre-existing ``OPENSCIENTIST_*`` keys from the parent
    env so tests are hermetic, then applies sane defaults plus the
    caller's overrides. Pass ``OPENSCIENTIST_JOB_DIR=""`` to drop a
    required var.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENSCIENTIST_")}
    env["OPENSCIENTIST_JOB_ID"] = "test-job-001"
    env["OPENSCIENTIST_JOB_DIR"] = str(job_dir)
    for key, value in overrides.items():
        if value == "":
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _server_params(env: dict[str, str]) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "openscientist_tools"],
        env=env,
    )


async def test_lists_only_ping_tool(tmp_path: Path) -> None:
    async with stdio_client(_server_params(_server_env(tmp_path))) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            assert names == ["ping"]
            (ping_tool,) = tools.tools
            assert "message" in ping_tool.inputSchema["properties"]


async def test_ping_returns_state_bound_job_id(tmp_path: Path) -> None:
    async with stdio_client(_server_params(_server_env(tmp_path))) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("ping", {"message": "world"})
            (block,) = result.content
            assert isinstance(block, TextContent)
            assert block.text == "pong: world from job test-job-001"


def test_missing_required_env_var_fails(tmp_path: Path) -> None:
    env = _server_env(tmp_path, OPENSCIENTIST_JOB_DIR="")
    proc = subprocess.run(
        [sys.executable, "-m", "openscientist_tools"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode != 0
    assert "job_dir" in proc.stderr.lower()
