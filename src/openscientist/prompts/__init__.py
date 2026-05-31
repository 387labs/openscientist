"""Prompt templates for the OpenScientist orchestrator.

`get_system_prompt(agent_backend=...)` selects the Claude or Codex
variant. The shared bodies live in `common`. The backend fragments and
per-job doc generators live in `claude` / `codex`.
"""

from typing import Literal

from openscientist.prompts.claude import generate_job_claude_md, get_claude_system_prompt
from openscientist.prompts.codex import generate_job_agents_md, get_codex_system_prompt
from openscientist.prompts.common import (
    build_discovery_prompt,
    format_skills_list,
    get_enabled_skills,
)

AgentBackend = Literal["claude_code", "codex"]


def get_system_prompt(*, agent_backend: AgentBackend = "claude_code") -> str:
    """Return the system prompt for the given agent backend."""
    if agent_backend == "codex":
        return get_codex_system_prompt()
    return get_claude_system_prompt()


__all__ = [
    "AgentBackend",
    "build_discovery_prompt",
    "format_skills_list",
    "generate_job_agents_md",
    "generate_job_claude_md",
    "get_enabled_skills",
    "get_system_prompt",
]
