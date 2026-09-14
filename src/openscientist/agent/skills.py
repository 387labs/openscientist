"""SKILL.md rendering shared by the agents that materialise skills."""

from __future__ import annotations

from openscientist.database.models import Skill


def _yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_skill_md(skill: Skill) -> str:
    """Render a skill as an Agent Skills ``SKILL.md`` (frontmatter + body).

    The ``name`` is capped at 64 chars and ``description`` collapsed to one line
    (max 1024) with a non-empty fallback, since some providers drop a skill whose
    description is empty.
    """
    name = f"{skill.category}--{skill.slug}"[:64]
    description = " ".join((skill.description or "").split())[:1024]
    if not description:
        description = f"{skill.category} skill: {skill.name}"
    frontmatter = f"---\nname: {_yaml_quote(name)}\ndescription: {_yaml_quote(description)}\n---\n"
    return frontmatter + skill.content
