"""Codex agent prompt variants.

The Codex agent reads its instructions from ``AGENTS.md`` and has no
``.claude/`` directory, so the fragments drop Claude-specific paths and
the ``Read`` tool name. Skill on-disk delivery for Codex is not wired yet
(later work), so the doc refers to skills generically.
"""

from openscientist.prompts.common import BackendFragments, build_job_doc, build_system_prompt

CODEX_FRAGMENTS = BackendFragments(
    skills_location="the `skills/` directory provided to you",
    builtin_read_tool="the built-in file-reading tool",
    builtin_read_tool_short="the built-in file-reading tool",
)


def get_codex_system_prompt() -> str:
    """System prompt for the Codex agent."""
    return build_system_prompt(CODEX_FRAGMENTS)


def generate_job_agents_md(*, use_hypotheses: bool = False, phenix_available: bool = False) -> str:
    """The per-job ``AGENTS.md`` content for the Codex agent."""
    return build_job_doc(
        use_hypotheses=use_hypotheses,
        phenix_available=phenix_available,
        frags=CODEX_FRAGMENTS,
    )
