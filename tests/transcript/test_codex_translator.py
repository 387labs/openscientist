"""Exhaustive tests for the Codex :class:`TranscriptDeserializer`.

Covers happy-path translation for every supported item type, no-drop
key coverage across the real ``codex exec --json`` fixtures, envelope
event handling (``thread.started`` / ``turn.completed`` /
``turn.failed`` / ``error`` / ``item.started`` / ``item.updated``), and
the :class:`UnknownEntry` + WARNING contract for unrecognised shapes.
"""

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from openscientist.transcript import (
    CODEX,
    AssistantText,
    CodexAgent,
    CodexDeserializer,
    CollabAgentToolCall,
    FileChange,
    Plan,
    Reasoning,
    ShellExecution,
    TaskNotification,
    TaskStarted,
    ToolCall,
    ToolResult,
    TranscriptAdapter,
    TranscriptDeserializer,
    UnknownEntry,
    UserPrompt,
    WebSearch,
)


def _envelope(thread_id: str = "thread-x") -> list[dict[str, Any]]:
    """A minimal envelope of ``thread.started`` plus ``turn.started``.

    The caller appends ``item.completed`` events and a closing
    ``turn.completed`` event.
    """
    return [
        {"type": "thread.started", "thread_id": thread_id},
        {"type": "turn.started"},
    ]


def _item_completed(item: dict[str, Any]) -> dict[str, Any]:
    return {"type": "item.completed", "item": item}


def _closing() -> dict[str, Any]:
    return {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}


# ---- Protocol surface --------------------------------------------------------------------------


class TestCodexDeserializerSurface:
    """Pins the public surface: :data:`CODEX` + the Protocol it satisfies."""

    def test_module_singleton_is_a_codex_deserializer(self) -> None:
        assert isinstance(CODEX, CodexDeserializer)

    def test_module_singleton_satisfies_the_protocol(self) -> None:
        assert isinstance(CODEX, TranscriptDeserializer)

    def test_codex_agent_is_a_marker(self) -> None:
        from openscientist.transcript import AgentMarker

        assert issubclass(CodexAgent, AgentMarker)


# ---- Envelope events ---------------------------------------------------------------------------


class TestEnvelopeEvents:
    """Envelope events that aren't items still have their fields preserved."""

    def test_thread_started_becomes_task_started_with_thread_metadata(self) -> None:
        out = CODEX.deserialize([{"type": "thread.started", "thread_id": "thr-001"}, _closing()])
        assert isinstance(out[0], TaskStarted)
        assert out[0].task_id == "thr-001"
        assert out[0].task_type == "thread"
        assert out[0].session_id == "thr-001"

    def test_turn_started_emits_nothing(self) -> None:
        out = CODEX.deserialize(
            [
                {"type": "thread.started", "thread_id": "thr-002"},
                {"type": "turn.started"},
                _closing(),
            ]
        )
        # TaskStarted (thread) + TaskNotification (turn.completed). The
        # turn.started event has no fields to drop.
        assert [type(e).__name__ for e in out] == ["TaskStarted", "TaskNotification"]

    def test_turn_completed_becomes_task_notification_with_usage(self) -> None:
        out = CODEX.deserialize(
            [
                {"type": "thread.started", "thread_id": "thr-003"},
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 100, "output_tokens": 25},
                },
            ]
        )
        notif = out[-1]
        assert isinstance(notif, TaskNotification)
        assert notif.status == "completed"
        assert notif.usage == {"input_tokens": 100, "output_tokens": 25}
        assert notif.task_id == "thr-003"

    def test_turn_failed_becomes_task_notification_with_error_message(self) -> None:
        out = CODEX.deserialize(
            [
                {"type": "thread.started", "thread_id": "thr-004"},
                {"type": "turn.failed", "error": {"message": "rate limited"}},
            ]
        )
        notif = out[-1]
        assert isinstance(notif, TaskNotification)
        assert notif.status == "failed"
        assert notif.summary == "rate limited"

    def test_top_level_error_becomes_task_notification(self) -> None:
        out = CODEX.deserialize(
            [
                {"type": "thread.started", "thread_id": "thr-005"},
                {"type": "error", "message": "stream interrupted"},
            ]
        )
        notif = out[-1]
        assert isinstance(notif, TaskNotification)
        assert notif.status == "failed"
        assert notif.summary == "stream interrupted"

    def test_item_started_and_updated_are_skipped(self) -> None:
        events = _envelope("thr-006") + [
            {
                "type": "item.started",
                "item": {
                    "id": "i1",
                    "type": "command_execution",
                    "command": "ls",
                    "aggregated_output": "",
                    "exit_code": None,
                    "status": "in_progress",
                },
            },
            {
                "type": "item.updated",
                "item": {
                    "id": "i1",
                    "type": "command_execution",
                    "command": "ls",
                    "aggregated_output": "a\n",
                    "exit_code": None,
                    "status": "in_progress",
                },
            },
            _item_completed(
                {
                    "id": "i1",
                    "type": "command_execution",
                    "command": "ls",
                    "aggregated_output": "a\nb\n",
                    "exit_code": 0,
                    "status": "completed",
                }
            ),
            _closing(),
        ]
        out = CODEX.deserialize(events)
        # Only the completed item produces a ShellExecution entry. The
        # started and updated events are superseded.
        shells = [e for e in out if isinstance(e, ShellExecution)]
        assert len(shells) == 1
        assert shells[0].output == "a\nb\n"
        assert shells[0].exit_code == 0


# ---- Per-item type translation -----------------------------------------------------------------


class TestAgentMessageItem:
    def test_emits_assistant_text(self) -> None:
        events = (
            _envelope()
            + [_item_completed({"id": "i1", "type": "agent_message", "text": "hello"})]
            + [_closing()]
        )
        out = CODEX.deserialize(events)
        msgs = [e for e in out if isinstance(e, AssistantText)]
        assert msgs == [AssistantText(id="i1", text="hello", raw=msgs[0].raw)]
        assert msgs[0].raw["_thread_id"] == "thread-x"


class TestReasoningItem:
    def test_emits_reasoning(self) -> None:
        events = (
            _envelope()
            + [_item_completed({"id": "r1", "type": "reasoning", "text": "step by step"})]
            + [_closing()]
        )
        out = CODEX.deserialize(events)
        reasons = [e for e in out if isinstance(e, Reasoning)]
        assert len(reasons) == 1
        assert reasons[0].text == "step by step"
        assert reasons[0].id == "r1"


class TestCommandExecutionItem:
    def test_completed_emits_shell_execution_with_aggregated_output(self) -> None:
        item = {
            "id": "c1",
            "type": "command_execution",
            "command": "ls /tmp",
            "aggregated_output": "a\nb\n",
            "exit_code": 0,
            "status": "completed",
        }
        out = CODEX.deserialize(_envelope() + [_item_completed(item), _closing()])
        shells = [e for e in out if isinstance(e, ShellExecution)]
        assert len(shells) == 1
        assert shells[0].command == "ls /tmp"
        assert shells[0].output == "a\nb\n"
        assert shells[0].exit_code == 0
        assert shells[0].status == "completed"

    def test_missing_aggregated_output_becomes_empty_string(self) -> None:
        item = {
            "id": "c2",
            "type": "command_execution",
            "command": "true",
            "exit_code": 0,
            "status": "completed",
        }
        out = CODEX.deserialize(_envelope() + [_item_completed(item), _closing()])
        shells = [e for e in out if isinstance(e, ShellExecution)]
        assert shells[0].output == ""


class TestFileChangeItem:
    def test_one_change_emits_one_file_change(self) -> None:
        item = {
            "id": "f1",
            "type": "file_change",
            "changes": [{"path": "/tmp/foo", "kind": "add"}],
            "status": "completed",
        }
        out = CODEX.deserialize(_envelope() + [_item_completed(item), _closing()])
        fcs = [e for e in out if isinstance(e, FileChange)]
        assert len(fcs) == 1
        assert fcs[0].path == "/tmp/foo"
        assert fcs[0].kind == "create"
        assert fcs[0].success is True

    def test_multiple_changes_split_into_multiple_entries_sharing_parent_id(
        self,
    ) -> None:
        item = {
            "id": "f2",
            "type": "file_change",
            "changes": [
                {"path": "/a", "kind": "add"},
                {"path": "/b", "kind": "update"},
                {"path": "/c", "kind": "delete"},
            ],
            "status": "completed",
        }
        out = CODEX.deserialize(_envelope() + [_item_completed(item), _closing()])
        fcs = [e for e in out if isinstance(e, FileChange)]
        assert [fc.path for fc in fcs] == ["/a", "/b", "/c"]
        assert [fc.kind for fc in fcs] == ["create", "edit", "delete"]
        assert {fc.id for fc in fcs} == {"f2"}

    def test_failed_status_sets_success_false(self) -> None:
        item = {
            "id": "f3",
            "type": "file_change",
            "changes": [{"path": "/x", "kind": "add"}],
            "status": "failed",
        }
        out = CODEX.deserialize(_envelope() + [_item_completed(item), _closing()])
        fcs = [e for e in out if isinstance(e, FileChange)]
        assert fcs[0].success is False

    def test_unknown_kind_becomes_unknown_entry(self) -> None:
        item = {
            "id": "f4",
            "type": "file_change",
            "changes": [{"path": "/x", "kind": "scramble"}],
            "status": "completed",
        }
        out = CODEX.deserialize(_envelope() + [_item_completed(item), _closing()])
        # Other entries (TaskStarted, TaskNotification) are still emitted;
        # the failing change becomes an UnknownEntry with the offending
        # change preserved.
        unknowns = [e for e in out if isinstance(e, UnknownEntry)]
        assert len(unknowns) == 1
        assert "scramble" in str(unknowns[0].raw)


class TestMcpToolCallItem:
    def test_success_emits_paired_tool_call_and_tool_result(self) -> None:
        item = {
            "id": "m1",
            "type": "mcp_tool_call",
            "server": "shandy-tools",
            "tool": "execute_code",
            "arguments": {"code": "print(1)", "language": "python"},
            "result": {
                "content": [{"type": "text", "text": "1\n"}],
                "structured_content": {"value": 1},
            },
            "error": None,
            "status": "completed",
        }
        out = CODEX.deserialize(_envelope() + [_item_completed(item), _closing()])
        calls = [e for e in out if isinstance(e, ToolCall)]
        results = [e for e in out if isinstance(e, ToolResult)]
        assert len(calls) == 1 and len(results) == 1
        assert calls[0].id == results[0].call_id == "m1"
        assert calls[0].tool == "execute_code"
        assert calls[0].server == "shandy-tools"
        assert calls[0].arguments == {"code": "print(1)", "language": "python"}
        assert results[0].success is True
        assert results[0].output == "1\n"
        assert results[0].structured_content == {"value": 1}

    def test_failure_emits_tool_result_with_error_message(self) -> None:
        item = {
            "id": "m2",
            "type": "mcp_tool_call",
            "server": "shandy-tools",
            "tool": "execute_code",
            "arguments": {},
            "result": None,
            "error": {"message": "exited with code 2"},
            "status": "failed",
        }
        out = CODEX.deserialize(_envelope() + [_item_completed(item), _closing()])
        results = [e for e in out if isinstance(e, ToolResult)]
        assert results[0].success is False
        assert results[0].error_message == "exited with code 2"


class TestCollabToolCallItem:
    def test_emits_collab_agent_tool_call(self) -> None:
        item = {
            "id": "ct1",
            "type": "collab_tool_call",
            "tool": "spawn_agent",
            "sender_thread_id": "parent-thread",
            "receiver_thread_ids": ["child-thread"],
            "prompt": "investigate this",
            "agents_states": {"child-thread": {"status": "completed", "message": "done"}},
            "status": "completed",
        }
        out = CODEX.deserialize(_envelope() + [_item_completed(item), _closing()])
        collabs = [e for e in out if isinstance(e, CollabAgentToolCall)]
        assert len(collabs) == 1
        assert collabs[0].prompt == "investigate this"
        assert collabs[0].agents_states == {
            "child-thread": {"status": "completed", "message": "done"}
        }


class TestWebSearchItem:
    def test_emits_web_search(self) -> None:
        item = {
            "id": "w1",
            "type": "web_search",
            "query": "claude code memory",
            "action": {"type": "search", "query": "claude code memory"},
        }
        out = CODEX.deserialize(_envelope() + [_item_completed(item), _closing()])
        searches = [e for e in out if isinstance(e, WebSearch)]
        assert len(searches) == 1
        assert searches[0].query == "claude code memory"
        assert searches[0].action == {"type": "search", "query": "claude code memory"}


class TestTodoListItem:
    def test_emits_plan_with_rendered_checklist(self) -> None:
        item = {
            "id": "t1",
            "type": "todo_list",
            "items": [
                {"text": "step one", "completed": True},
                {"text": "step two", "completed": False},
            ],
        }
        out = CODEX.deserialize(_envelope() + [_item_completed(item), _closing()])
        plans = [e for e in out if isinstance(e, Plan)]
        assert len(plans) == 1
        assert plans[0].text == "- [x] step one\n- [ ] step two"


class TestErrorItem:
    def test_emits_task_notification_with_failed_status(self) -> None:
        item = {"id": "e1", "type": "error", "message": "non-fatal hiccup"}
        out = CODEX.deserialize(_envelope() + [_item_completed(item), _closing()])
        notifs = [e for e in out if isinstance(e, TaskNotification)]
        # One from the error item, one from turn.completed.
        assert any(n.summary == "non-fatal hiccup" and n.status == "failed" for n in notifs)


class TestUserMessageItem:
    def test_emits_user_prompt(self) -> None:
        item = {"id": "u1", "type": "user_message", "text": "explain that again"}
        out = CODEX.deserialize(_envelope() + [_item_completed(item), _closing()])
        prompts = [e for e in out if isinstance(e, UserPrompt)]
        assert prompts == [UserPrompt(id="u1", text="explain that again", raw=prompts[0].raw)]


# ---- Unknown handling -------------------------------------------------------------------------


class TestUnknownHandling:
    def test_unknown_envelope_type_becomes_unknown_entry(self) -> None:
        out = CODEX.deserialize(
            [
                {"type": "thread.started", "thread_id": "thr-u"},
                {"type": "thread.surprising", "payload": "?"},
                _closing(),
            ]
        )
        unknowns = [e for e in out if isinstance(e, UnknownEntry)]
        assert len(unknowns) == 1
        assert unknowns[0].source == "codex"
        assert unknowns[0].raw["type"] == "thread.surprising"

    def test_unknown_item_type_becomes_unknown_entry(self) -> None:
        events = (
            _envelope()
            + [_item_completed({"id": "x1", "type": "novel_item", "extra": 1})]
            + [_closing()]
        )
        out = CODEX.deserialize(events)
        unknowns = [e for e in out if isinstance(e, UnknownEntry)]
        assert len(unknowns) == 1
        assert unknowns[0].raw["_item"]["type"] == "novel_item"

    def test_item_completed_without_dict_item_becomes_unknown_entry(self) -> None:
        events = _envelope() + [{"type": "item.completed", "item": "not-a-dict"}, _closing()]
        out = CODEX.deserialize(events)
        unknowns = [e for e in out if isinstance(e, UnknownEntry)]
        assert len(unknowns) == 1

    def test_non_dict_event_becomes_unknown_entry(self) -> None:
        out = CODEX.deserialize([{"type": "thread.started", "thread_id": "x"}, "not-a-dict"])  # type: ignore[list-item]
        unknowns = [e for e in out if isinstance(e, UnknownEntry)]
        assert len(unknowns) == 1

    def test_unknown_emits_warning_via_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="openscientist.transcript"):
            CODEX.deserialize([{"type": "future.event_kind"}])
        assert any("UnknownEntry" in rec.message for rec in caplog.records)


# ---- Real-fixture sweep ------------------------------------------------------------------------


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "transcripts" / "codex" / "exec_jsonl"
EXEC_FIXTURES = sorted(FIXTURE_DIR.glob("*.jsonl.gz"))


def _load_exec_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


class TestRealFixtureSweep:
    @pytest.mark.parametrize("path", EXEC_FIXTURES, ids=lambda p: p.stem)
    def test_real_fixture_translates_without_unknown(self, path: Path) -> None:
        entries = CODEX.deserialize(_load_exec_jsonl(path))
        unknowns = [e for e in entries if isinstance(e, UnknownEntry)]
        assert not unknowns, f"{path.name}: produced UnknownEntry. Sample: {unknowns[0].raw}"

    @pytest.mark.parametrize("path", EXEC_FIXTURES, ids=lambda p: p.stem)
    def test_real_fixture_roundtrips_through_typeadapter(self, path: Path) -> None:
        entries = CODEX.deserialize(_load_exec_jsonl(path))
        raw = TranscriptAdapter.dump_python(entries, mode="json")
        assert TranscriptAdapter.validate_python(raw) == entries

    @pytest.mark.parametrize("path", EXEC_FIXTURES, ids=lambda p: p.stem)
    def test_real_fixture_begins_with_task_started_and_ends_with_task_notification(
        self, path: Path
    ) -> None:
        entries = CODEX.deserialize(_load_exec_jsonl(path))
        assert isinstance(entries[0], TaskStarted)
        assert entries[0].task_type == "thread"
        assert isinstance(entries[-1], TaskNotification)


# ---- No-drop key coverage on real fixtures ----------------------------------------------------


def _flatten_source_keys(node: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else key
            sub = _flatten_source_keys(value, child)
            if sub:
                paths.extend(sub)
            else:
                paths.append(child)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            sub = _flatten_source_keys(value, f"{prefix}[{index}]")
            if sub:
                paths.extend(sub)
            else:
                paths.append(f"{prefix}[{index}]")
    else:
        if prefix:
            paths.append(prefix)
    return paths


def _flatten_entry_values(node: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(node, dict):
        for value in node.values():
            values |= _flatten_entry_values(value)
    elif isinstance(node, list):
        for value in node:
            values |= _flatten_entry_values(value)
    elif node is None:
        pass
    elif isinstance(node, bool):
        values.add(str(node))
    else:
        values.add(repr(node))
    return values


class TestNoDropCoverage:
    """The translator must surface every distinct source value somewhere
    in the produced entries, either as a typed field or under
    ``raw``. The check is conservative. It walks distinct VALUES so
    snake_case keys vs typed fields do not false-fail when only the
    naming differs.
    """

    @pytest.mark.parametrize("path", EXEC_FIXTURES, ids=lambda p: p.stem)
    def test_real_fixture_loses_no_source_value(self, path: Path) -> None:
        events = _load_exec_jsonl(path)
        entries = CODEX.deserialize(events)
        dumped = TranscriptAdapter.dump_python(entries, mode="python")
        produced = _flatten_entry_values(dumped)
        source: set[str] = set()
        for event in events:
            source |= _flatten_entry_values(event)
        # Values consumed as type discriminators / structural enums that
        # don't surface verbatim in the typed output (because they're
        # mapped to a different typed name).
        consumed_discriminators: set[str] = {
            "'thread.started'",
            "'turn.started'",
            "'turn.completed'",
            "'turn.failed'",
            "'item.started'",
            "'item.updated'",
            "'item.completed'",
            "'error'",
            "'agent_message'",
            "'reasoning'",
            "'command_execution'",
            "'file_change'",
            "'mcp_tool_call'",
            "'collab_tool_call'",
            "'web_search'",
            "'todo_list'",
            "'user_message'",
            # PatchChangeKind values get remapped to FileChange.kind variants.
            "'add'",
            "'update'",
            "'delete'",
            # ``item.started`` and ``item.updated`` carry intermediate
            # lifecycle status values that are deliberately superseded
            # by the matching ``item.completed`` whose status is the
            # final one. The no-drop contract applies to the final
            # logical state, not to transient streaming snapshots.
            "'in_progress'",
            # ``WebSearchAction.type`` ``"other"`` appears on the
            # early ``item.started`` snapshot of a ``web_search``
            # before the action type resolves to ``"search"`` or
            # ``"openPage"`` (and so on) by the time the item
            # completes. The completed action survives.
            "'other'",
            # ``False`` is the per-entry ``completed`` flag on a
            # fresh ``todo_list`` item. The final ``item.completed``
            # snapshot has all entries flipped to ``True``. The
            # structured items list (including any historical
            # ``False`` entries) survives in the produced Plan's
            # ``raw["_todo_items"]``.
            "False",
        }
        missing = source - produced - consumed_discriminators
        assert not missing, (
            f"{path.name}: source values dropped by translator: {sorted(missing)[:10]}"
        )


# ---- Coverage gate ------------------------------------------------------------------------------


# Codex variants we expect at least one REAL fixture under
# exec_jsonl/ to exercise. Locked here so that accidentally breaking
# real-data coverage for one of them surfaces as a test failure.
# Variants not in this list (Reasoning, CollabAgentToolCall,
# UserPrompt) currently have synthetic-only coverage.
_CODEX_REAL_FIXTURE_VARIANTS: list[type[Any]] = sorted(
    [
        AssistantText,
        ShellExecution,
        FileChange,
        ToolCall,
        ToolResult,
        WebSearch,
        Plan,
        TaskStarted,
        TaskNotification,
    ],
    key=lambda c: c.__name__,
)


@pytest.mark.parametrize(
    "variant",
    _CODEX_REAL_FIXTURE_VARIANTS,
    ids=lambda c: c.__name__,
)
def test_each_target_variant_has_at_least_one_real_fixture(variant: type[Any]) -> None:
    """A variant once exercised by a real fixture must stay covered."""
    for path in EXEC_FIXTURES:
        entries = CODEX.deserialize(_load_exec_jsonl(path))
        if any(isinstance(e, variant) for e in entries):
            return
    pytest.fail(f"No real Codex exec_jsonl fixture produces a {variant.__name__} entry.")
