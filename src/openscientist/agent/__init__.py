"""
Agent package for OpenScientist.

Provides the AgentExecutor protocol and the ClaudeCodeAgent implementation
(backed by claude-agent-sdk).  Use get_agent() from agent.factory to get the
agent for the configured provider.
"""

from openscientist.agent.mcp_specs import (
    HttpMcpServerSpec,
    McpServerSpec,
    StdioMcpServerSpec,
)
from openscientist.agent.protocol import AgentExecutor, IterationResult, TokenUsage

__all__ = [
    "AgentExecutor",
    "HttpMcpServerSpec",
    "IterationResult",
    "McpServerSpec",
    "StdioMcpServerSpec",
    "TokenUsage",
]
