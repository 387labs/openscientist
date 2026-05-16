"""Tests for the TranscriptEntry discriminated union."""

from typing import assert_never

import pytest
from pydantic import ValidationError

from openscientist.transcript import (
    AssistantText,
    FileChange,
    Reasoning,
    ShellExecution,
    ToolCall,
    ToolResult,
    TranscriptAdapter,
    TranscriptEntry,
)

# ---- Per-variant round-trip tests ---------------------------------------------------------------


def test_assistant_text_roundtrip() -> None:
    entry = AssistantText(text="hello world")
    raw = entry.model_dump(mode="json")
    assert raw == {"type": "assistant_text", "text": "hello world"}
    assert AssistantText.model_validate(raw) == entry


def test_tool_call_roundtrip() -> None:
    entry = ToolCall(id="call_1", tool="execute_code", arguments={"code": "print(1)"})
    raw = entry.model_dump(mode="json")
    assert raw == {
        "type": "tool_call",
        "id": "call_1",
        "tool": "execute_code",
        "arguments": {"code": "print(1)"},
    }
    assert ToolCall.model_validate(raw) == entry


def test_tool_result_roundtrip() -> None:
    entry = ToolResult(call_id="call_1", output="1\n", success=True, duration_ms=42)
    raw = entry.model_dump(mode="json")
    assert raw == {
        "type": "tool_result",
        "call_id": "call_1",
        "output": "1\n",
        "success": True,
        "duration_ms": 42,
    }
    assert ToolResult.model_validate(raw) == entry


def test_tool_result_duration_optional() -> None:
    entry = ToolResult(call_id="call_2", output="", success=False)
    assert entry.duration_ms is None


def test_shell_execution_roundtrip() -> None:
    entry = ShellExecution(id="sh_1", command="ls", output="a\nb\n", exit_code=0)
    raw = entry.model_dump(mode="json")
    assert raw == {
        "type": "shell_execution",
        "id": "sh_1",
        "command": "ls",
        "output": "a\nb\n",
        "exit_code": 0,
    }
    assert ShellExecution.model_validate(raw) == entry


def test_file_change_roundtrip() -> None:
    entry = FileChange(
        id="fc_1",
        path="src/foo.py",
        kind="edit",
        diff="@@ -1 +1 @@\n-old\n+new\n",
        success=True,
    )
    raw = entry.model_dump(mode="json")
    assert raw == {
        "type": "file_change",
        "id": "fc_1",
        "path": "src/foo.py",
        "kind": "edit",
        "diff": "@@ -1 +1 @@\n-old\n+new\n",
        "success": True,
    }
    assert FileChange.model_validate(raw) == entry


def test_file_change_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        FileChange(id="fc_2", path="x", kind="delete", success=True)  # type: ignore[arg-type]


def test_reasoning_roundtrip() -> None:
    entry = Reasoning(text="thinking", summary="step 1")
    raw = entry.model_dump(mode="json")
    assert raw == {"type": "reasoning", "text": "thinking", "summary": "step 1"}
    assert Reasoning.model_validate(raw) == entry


# ---- Discriminated-union dispatch tests ---------------------------------------------------------


def test_typeadapter_parses_each_variant_to_correct_class() -> None:
    entries: list[TranscriptEntry] = [
        AssistantText(text="hi"),
        ToolCall(id="c1", tool="t", arguments={}),
        ToolResult(call_id="c1", output="ok", success=True),
        ShellExecution(id="s1", command="echo", output="echo\n", exit_code=0),
        FileChange(id="f1", path="x", kind="write", success=True),
        Reasoning(text="r"),
    ]
    raw = TranscriptAdapter.dump_python(entries, mode="json")
    parsed = TranscriptAdapter.validate_python(raw)
    assert parsed == entries
    assert [type(p) for p in parsed] == [
        AssistantText,
        ToolCall,
        ToolResult,
        ShellExecution,
        FileChange,
        Reasoning,
    ]


def test_typeadapter_rejects_unknown_discriminator() -> None:
    with pytest.raises(ValidationError):
        TranscriptAdapter.validate_python([{"type": "nope"}])


def test_typeadapter_rejects_missing_discriminator() -> None:
    with pytest.raises(ValidationError):
        TranscriptAdapter.validate_python([{"text": "missing type field"}])


# ---- Exhaustiveness test ------------------------------------------------------------------------


def _describe(entry: TranscriptEntry) -> str:
    """Match against every variant.

    Adding a new variant to the union without adding a case below makes
    mypy --strict reject the ``assert_never`` call, so this acts as a
    compile-time exhaustiveness check whenever the suite type-checks.
    """
    match entry:
        case AssistantText():
            return "assistant_text"
        case ToolCall():
            return "tool_call"
        case ToolResult():
            return "tool_result"
        case ShellExecution():
            return "shell_execution"
        case FileChange():
            return "file_change"
        case Reasoning():
            return "reasoning"
        case _:
            assert_never(entry)


def test_match_exhaustiveness_covers_every_variant() -> None:
    """Sanity-check the exhaustiveness helper runs for every variant."""
    samples: list[TranscriptEntry] = [
        AssistantText(text=""),
        ToolCall(id="", tool="", arguments={}),
        ToolResult(call_id="", output="", success=True),
        ShellExecution(id="", command="", output=""),
        FileChange(id="", path="", kind="write", success=True),
        Reasoning(text=""),
    ]
    assert [_describe(s) for s in samples] == [
        "assistant_text",
        "tool_call",
        "tool_result",
        "shell_execution",
        "file_change",
        "reasoning",
    ]
