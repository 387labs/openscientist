"""Generic agent base parameterised over the provider family.

`AbstractAgent[P: Provider]` ties an agent runtime to the provider
family it can drive: `ClaudeCodeAgent(AbstractAgent[ClaudeCompatible])`
and `CodexAgent(AbstractAgent[CodexCompatible])` (added later) cannot be
constructed with a mismatched provider, and mypy rejects the mismatch at
check-time. Nothing inherits this yet.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

from openscientist.agent.mcp_specs import McpServerSpec
from openscientist.agent.protocol import IterationResult, TokenUsage
from openscientist.providers.base_v2 import Provider
from openscientist.transcript import TranscriptEntry

__all__ = [
    "AbstractAgent",
    "AgentConfig",
    "IterationResult",
    "TokenUsage",
    "TranscriptEntry",
]


@dataclass(frozen=True)
class AgentConfig:
    """Backend-agnostic agent configuration."""

    job_dir: Path
    data_file: Path | None = None
    system_prompt: str | None = None
    use_hypotheses: bool = False
    data_files: tuple[Path, ...] = ()
    mcp_servers: tuple[McpServerSpec, ...] = ()


class AbstractAgent[P: Provider](abc.ABC):
    """Agent runtime parameterised over the provider family it accepts."""

    def __init__(self, config: AgentConfig, provider: P) -> None:
        self._config = config
        self._provider = provider
        self._token_usage = TokenUsage()

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def provider(self) -> P:
        return self._provider

    @property
    def total_tokens(self) -> TokenUsage:
        return self._token_usage

    @abc.abstractmethod
    async def run_iteration(
        self, prompt: str, *, reset_session: bool = False
    ) -> IterationResult: ...

    @abc.abstractmethod
    async def shutdown(self) -> None: ...
