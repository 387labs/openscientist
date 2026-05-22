"""
AgentExecutor protocol definition for OpenScientist.

Defines the interface that all agent executors must implement,
allowing the orchestrator to be decoupled from any specific provider
or execution strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from openscientist.transcript import TranscriptEntry


@dataclass
class TokenUsage:
    """Normalized token usage across all iterations.

    Each token is counted in exactly one category; the categories are
    additive and non-overlapping. Total token count equals the sum of
    all five fields. This invariant lets cost functions multiply each
    field by its per-category rate and sum, without double-counting.

    Backend SDKs that report hierarchical counts (e.g., OpenAI's
    ``input_tokens`` includes ``cached_input_tokens``) must subtract
    sub-categories from totals at the agent boundary before populating
    this dataclass.
    """

    input_tokens: int = 0
    """Fresh, uncached input tokens."""

    output_tokens: int = 0
    """Visible (non-reasoning) output tokens."""

    cache_write_tokens: int = 0
    """Tokens written to a provider-side prompt cache. Anthropic only."""

    cache_read_tokens: int = 0
    """Tokens served from a provider-side prompt cache."""

    reasoning_tokens: int = 0
    """Internal reasoning tokens (o-series; Anthropic extended thinking)."""

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.reasoning_tokens += other.reasoning_tokens
        return self


@dataclass(frozen=True)
class IterationResult:
    """Result of a single agent iteration."""

    success: bool
    output: str
    tool_calls: int
    transcript: list[TranscriptEntry]
    error: str = ""


@runtime_checkable
class AgentExecutor(Protocol):
    """
    Protocol for agent executors.

    All executors must implement run_iteration, shutdown, and total_tokens.
    This protocol is @runtime_checkable, enabling isinstance() checks in tests.

    Usage::

        executor: AgentExecutor = get_agent_executor(...)
        result = await executor.run_iteration(prompt, reset_session=True)
        print(result.output)
        await executor.shutdown()
    """

    async def run_iteration(
        self,
        prompt: str,
        *,
        reset_session: bool = False,
    ) -> IterationResult:
        """
        Run a single discovery iteration.

        Args:
            prompt: User prompt for this iteration
            reset_session: If True, clear session history before running

        Returns:
            IterationResult with success flag, output text, and token counts
        """
        ...

    async def shutdown(self) -> None:
        """Release all resources held by this executor."""
        ...

    @property
    def total_tokens(self) -> TokenUsage:
        """Cumulative token usage across all iterations."""
        ...
