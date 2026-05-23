"""Shared pytest fixtures for spawning the standalone tools MCP server."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from mcp.client.stdio import StdioServerParameters


@pytest.fixture
def server_env() -> Callable[..., dict[str, str]]:
    """Return a builder for a clean spawn-env dict.

    Strips any pre-existing ``OPENSCIENTIST_*`` keys from the parent
    env so tests are hermetic, then applies sane defaults plus the
    caller's overrides. Pass ``OPENSCIENTIST_JOB_DIR=""`` to drop a
    required var.
    """

    def _build(job_dir: Path, **overrides: str) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("OPENSCIENTIST_")}
        env["OPENSCIENTIST_JOB_ID"] = "test-job-001"
        env["OPENSCIENTIST_JOB_DIR"] = str(job_dir)
        for key, value in overrides.items():
            if value == "":
                env.pop(key, None)
            else:
                env[key] = value
        return env

    return _build


@pytest.fixture
def server_params() -> Callable[[dict[str, str]], StdioServerParameters]:
    def _build(env: dict[str, str]) -> StdioServerParameters:
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", "openscientist_tools"],
            env=env,
        )

    return _build
