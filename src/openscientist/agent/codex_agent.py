"""Codex agent backend (skeleton).

`CodexAgent` drives the Codex CLI via the ``openai-codex-sdk``. This is a
scaffold: the abstract methods raise ``NotImplementedError`` and gain real
implementations in a later change. The generic bound ties it to
``CodexCompatible`` providers, so mypy rejects pairing it with a
Claude-only provider.
"""

from __future__ import annotations

from openscientist.agent.base import AbstractAgent, IterationResult
from openscientist.providers.base import CodexCompatible


class CodexAgent(AbstractAgent[CodexCompatible]):
    """Agent that drives the Codex CLI (``openai-codex-sdk``)."""

    async def run_iteration(self, prompt: str, *, reset_session: bool = False) -> IterationResult:
        raise NotImplementedError("CodexAgent.run_iteration is not implemented yet")

    async def shutdown(self) -> None:
        raise NotImplementedError("CodexAgent.shutdown is not implemented yet")
