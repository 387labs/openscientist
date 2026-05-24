"""Tests for `openscientist.agent.mcp_specs`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openscientist.agent import (
    HttpMcpServerSpec,
    McpServerSpec,
    StdioMcpServerSpec,
)


def test_stdio_minimal_construction_and_sdk_config() -> None:
    spec = StdioMcpServerSpec(name="tools", command="python")
    assert spec.name == "tools"
    assert spec.command == "python"
    assert spec.args == ()
    assert spec.env is None
    assert spec.cwd is None
    assert spec.to_sdk_config() == {"type": "stdio", "command": "python"}


def test_stdio_full_construction_and_sdk_config() -> None:
    spec = StdioMcpServerSpec(
        name="tools",
        command="python",
        args=("-m", "openscientist_tools"),
        env={"OPENSCIENTIST_JOB_ID": "abc"},
        cwd="/jobs/abc",
    )
    cfg = spec.to_sdk_config()
    assert cfg == {
        "type": "stdio",
        "command": "python",
        "args": ["-m", "openscientist_tools"],
        "env": {"OPENSCIENTIST_JOB_ID": "abc"},
    }
    # cwd lives on the spec for the launcher, not in the SDK config.
    assert "cwd" not in cfg
    assert spec.cwd == "/jobs/abc"


def test_stdio_is_frozen() -> None:
    spec = StdioMcpServerSpec(name="tools", command="python")
    with pytest.raises(ValidationError):
        spec.command = "bash"  # type: ignore[misc]


def test_stdio_json_round_trip() -> None:
    spec = StdioMcpServerSpec(
        name="tools",
        command="python",
        args=("-m", "openscientist_tools"),
        env={"K": "V"},
        cwd="/tmp",
    )
    restored = StdioMcpServerSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec


def test_http_minimal_construction_and_sdk_config() -> None:
    spec = HttpMcpServerSpec(name="remote", url="https://example.com/mcp")
    assert spec.to_sdk_config() == {"type": "http", "url": "https://example.com/mcp"}


def test_http_with_headers_and_sdk_config() -> None:
    spec = HttpMcpServerSpec(
        name="remote",
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer token-123"},
    )
    assert spec.to_sdk_config() == {
        "type": "http",
        "url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer token-123"},
    }


def test_http_is_frozen() -> None:
    spec = HttpMcpServerSpec(name="remote", url="https://example.com/mcp")
    with pytest.raises(ValidationError):
        spec.url = "https://other.example.com/mcp"  # type: ignore[misc]


def test_http_json_round_trip() -> None:
    spec = HttpMcpServerSpec(
        name="remote",
        url="https://example.com/mcp",
        headers={"X-Trace": "abc"},
    )
    restored = HttpMcpServerSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec


def test_stdio_requires_command() -> None:
    with pytest.raises(ValidationError) as excinfo:
        StdioMcpServerSpec(name="tools")  # type: ignore[call-arg]
    assert "command" in str(excinfo.value).lower()


def test_http_requires_url() -> None:
    with pytest.raises(ValidationError) as excinfo:
        HttpMcpServerSpec(name="remote")  # type: ignore[call-arg]
    assert "url" in str(excinfo.value).lower()


def test_union_alias_accepts_both() -> None:
    def _accept(spec: McpServerSpec) -> str:
        return spec.name

    stdio: McpServerSpec = StdioMcpServerSpec(name="stdio-one", command="python")
    http: McpServerSpec = HttpMcpServerSpec(name="http-one", url="https://x/mcp")
    assert _accept(stdio) == "stdio-one"
    assert _accept(http) == "http-one"
    assert isinstance(stdio, StdioMcpServerSpec)
    assert isinstance(http, HttpMcpServerSpec)
