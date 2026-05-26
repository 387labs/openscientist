"""Type-system tests for the `AbstractAgent[P: Provider]` bound.

These run mypy as a subprocess on small snippets because the project's
mypy config does not enable `warn_unused_ignores`, so inline
`# type: ignore` assertions would be toothless. The subprocess approach
directly checks that the generic bound is enforced at check-time.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_VALID_SUBCLASS = """
from openscientist.agent.base import AbstractAgent, IterationResult
from openscientist.providers.base_v2 import ClaudeCompatible


class MyAgent(AbstractAgent[ClaudeCompatible]):
    async def run_iteration(
        self, prompt: str, *, reset_session: bool = False
    ) -> IterationResult:
        return IterationResult(success=True, output="", tool_calls=0, transcript=[])

    async def shutdown(self) -> None:
        return None
"""


def _run_mypy(code: str, tmp_path: Path) -> tuple[int, str]:
    snippet = tmp_path / "snippet.py"
    snippet.write_text(textwrap.dedent(code))
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", str(snippet)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    return result.returncode, result.stdout + result.stderr


def test_bound_rejects_non_provider_type_arg(tmp_path: Path) -> None:
    code = """
        from openscientist.agent.base import AbstractAgent

        x: type[AbstractAgent[int]] | None = None
    """
    rc, out = _run_mypy(code, tmp_path)
    assert rc != 0
    assert "type-var" in out


def test_accepts_claude_compatible_subclass(tmp_path: Path) -> None:
    rc, out = _run_mypy(_VALID_SUBCLASS, tmp_path)
    assert rc == 0, out


def test_rejects_subclass_missing_abstract_method(tmp_path: Path) -> None:
    code = """
        from openscientist.agent.base import AbstractAgent, AgentConfig, IterationResult
        from openscientist.providers.base_v2 import ClaudeCompatible
        from pathlib import Path


        class Incomplete(AbstractAgent[ClaudeCompatible]):
            async def run_iteration(
                self, prompt: str, *, reset_session: bool = False
            ) -> IterationResult:
                return IterationResult(success=True, output="", tool_calls=0, transcript=[])
            # shutdown omitted


        def _make(config: AgentConfig, provider: ClaudeCompatible) -> None:
            Incomplete(config, provider)
    """
    rc, out = _run_mypy(code, tmp_path)
    assert rc != 0
    assert "abstract" in out.lower()
