"""Claude Code agent prompt variants.

The fragments are identity substitutions over the shared bodies, so the
Claude system prompt and ``CLAUDE.md`` are byte-identical to the
pre-split versions.
"""

from openscientist.prompts.common import BackendFragments, build_job_doc, build_system_prompt

CLAUDE_FRAGMENTS = BackendFragments(
    skills_location="`.claude/skills/`",
    builtin_read_tool="Claude's built-in `Read` tool",
    builtin_read_tool_short="Claude's `Read` tool",
)


def get_claude_system_prompt() -> str:
    """System prompt for the Claude Code agent."""
    return build_system_prompt(CLAUDE_FRAGMENTS)


def generate_job_claude_md(*, use_hypotheses: bool = False, phenix_available: bool = False) -> str:
    """The per-job ``CLAUDE.md`` content for the Claude Code agent."""
    return build_job_doc(
        use_hypotheses=use_hypotheses,
        phenix_available=phenix_available,
        frags=CLAUDE_FRAGMENTS,
    )
