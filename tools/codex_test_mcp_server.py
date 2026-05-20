#!/usr/bin/env python3
"""Minimal MCP server for capturing Codex ``mcp_tool_call`` fixtures.

Two tools over stdio: ``echo`` for the success path, ``fail`` for
the failure path. Register in ``~/.codex/config.toml``::

    [mcp_servers.openscientist-fixture]
    command = "uv"
    args = [
        "run", "--project", "/path/to/this/repo",
        "python", "/path/to/this/repo/tools/codex_test_mcp_server.py",
    ]
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp: FastMCP = FastMCP("openscientist-fixture")


@mcp.tool()
def echo(text: str) -> str:
    """Return ``text`` verbatim. Always succeeds."""
    return text


@mcp.tool()
def fail(reason: str) -> str:
    """Always raises ``RuntimeError(reason)``."""
    raise RuntimeError(reason)


if __name__ == "__main__":
    mcp.run()
