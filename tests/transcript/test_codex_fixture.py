"""Structure-only sanity checks for the Codex transcript fixtures.

Covers both ``codex/exec_jsonl/`` (real captures consumed by
:data:`CODEX`) and ``codex/sdk_v2/`` (synthetic anchor for the
``openai-codex`` SDK v2 shape).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "transcripts" / "codex"
EXEC_JSONL_DIR = FIXTURE_ROOT / "exec_jsonl"
SDK_V2_FIXTURE = FIXTURE_ROOT / "sdk_v2" / "synthetic_all_variants.json.gz"

# ---- exec_jsonl real captures ------------------------------------------------

EXEC_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "thread.started",
        "turn.started",
        "turn.completed",
        "turn.failed",
        "item.started",
        "item.updated",
        "item.completed",
        "error",
    }
)

# Snake_case ``ThreadItemDetails.type`` discriminators emitted by the
# CLI. See ``codex-rs/exec/src/exec_events.rs::ThreadItemDetails``.
EXEC_ITEM_TYPES: frozenset[str] = frozenset(
    {
        "agent_message",
        "reasoning",
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "collab_tool_call",
        "web_search",
        "plan",
        "todo_list",
        "user_message",
        "error",
    }
)


def _load_exec_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert isinstance(obj, dict)
            events.append(obj)
    return events


EXEC_FIXTURES = sorted(EXEC_JSONL_DIR.glob("*.jsonl.gz"))


class TestCodexExecJsonlFixtures:
    """Real ``codex exec --json`` captures, gzipped."""

    def test_exec_jsonl_dir_has_real_captures(self) -> None:
        assert EXEC_FIXTURES, "exec_jsonl/ should contain at least one capture"

    @pytest.mark.parametrize("path", EXEC_FIXTURES, ids=lambda p: p.stem)
    def test_exec_fixture_loads_as_jsonl(self, path: Path) -> None:
        events = _load_exec_jsonl(path)
        assert events, f"{path.name} is empty"

    @pytest.mark.parametrize("path", EXEC_FIXTURES, ids=lambda p: p.stem)
    def test_exec_fixture_uses_only_known_event_types(self, path: Path) -> None:
        events = _load_exec_jsonl(path)
        seen = {e.get("type") for e in events}
        unknown = seen - EXEC_EVENT_TYPES
        assert not unknown, (
            f"{path.name} carries unknown ThreadEvent types: {sorted(map(str, unknown))}"
        )

    @pytest.mark.parametrize("path", EXEC_FIXTURES, ids=lambda p: p.stem)
    def test_exec_fixture_starts_with_thread_started(self, path: Path) -> None:
        events = _load_exec_jsonl(path)
        assert events[0].get("type") == "thread.started"
        assert isinstance(events[0].get("thread_id"), str)

    @pytest.mark.parametrize("path", EXEC_FIXTURES, ids=lambda p: p.stem)
    def test_exec_fixture_ends_with_turn_terminal(self, path: Path) -> None:
        events = _load_exec_jsonl(path)
        terminal = {"turn.completed", "turn.failed", "error"}
        assert events[-1].get("type") in terminal, (
            f"{path.name} ends with {events[-1].get('type')!r}, expected one of {terminal}"
        )

    @pytest.mark.parametrize("path", EXEC_FIXTURES, ids=lambda p: p.stem)
    def test_exec_fixture_items_use_only_known_item_types(self, path: Path) -> None:
        events = _load_exec_jsonl(path)
        item_types = {
            e["item"]["type"]
            for e in events
            if e.get("type", "").startswith("item.") and isinstance(e.get("item"), dict)
        }
        unknown = item_types - EXEC_ITEM_TYPES
        assert not unknown, (
            f"{path.name} carries unknown ThreadItemDetails.type values: "
            f"{sorted(map(str, unknown))}"
        )

    @pytest.mark.parametrize("path", EXEC_FIXTURES, ids=lambda p: p.stem)
    def test_exec_fixture_item_lifecycle_pairs(self, path: Path) -> None:
        """Every ``item.started`` must be eventually followed by an
        ``item.completed`` with the same id (within the same turn)."""
        events = _load_exec_jsonl(path)
        started_ids: set[str] = set()
        completed_ids: set[str] = set()
        for e in events:
            etype = e.get("type")
            item = e.get("item") if isinstance(e.get("item"), dict) else None
            if etype == "item.started" and item:
                started_ids.add(item["id"])
            elif etype == "item.completed" and item:
                completed_ids.add(item["id"])
        # Every started id reaches completion. The reverse is not required
        # (some items, like the final agent_message, complete without a
        # prior started event).
        unfinished = started_ids - completed_ids
        assert not unfinished, (
            f"{path.name} has item.started events with no matching "
            f"item.completed: {sorted(unfinished)}"
        )

    def test_exec_jsonl_coverage_across_captures(self) -> None:
        """The bundled captures, taken together, must exercise more than
        just ``agent_message`` so the future translator has shell- and
        file-mutation shapes to test against."""
        seen: set[str] = set()
        for path in EXEC_FIXTURES:
            for e in _load_exec_jsonl(path):
                if e.get("type", "").startswith("item.") and isinstance(e.get("item"), dict):
                    seen.add(e["item"]["type"])
        # The current captures cover these three item types. Tightening the
        # set is the right move when we add more captures later.
        required = {"agent_message", "command_execution", "file_change"}
        missing = required - seen
        assert not missing, f"exec_jsonl fixtures are missing real captures for: {sorted(missing)}"


# ---- sdk_v2 synthetic fixture ------------------------------------------------

# Wire-format ``ThreadItem.type`` discriminators declared by the
# openai-codex v2 schema. See ``v2_all.py``.
SDK_V2_VARIANT_TYPES: frozenset[str] = frozenset(
    {
        "userMessage",
        "hookPrompt",
        "agentMessage",
        "plan",
        "reasoning",
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "dynamicToolCall",
        "collabAgentToolCall",
        "webSearch",
        "imageView",
        "imageGeneration",
        "enteredReviewMode",
        "exitedReviewMode",
        "contextCompaction",
    }
)


def _load_sdk_v2_fixture() -> list[dict[str, Any]]:
    with gzip.open(SDK_V2_FIXTURE, "rt", encoding="utf-8") as f:
        loaded = json.load(f)
    assert isinstance(loaded, list)
    return loaded


class TestCodexSdkV2SyntheticFixture:
    def test_fixture_file_exists(self) -> None:
        assert SDK_V2_FIXTURE.is_file(), (
            "synthetic SDK v2 Codex fixture is checked in for the future "
            "Codex translator PR but is missing on disk"
        )

    def test_fixture_loads_as_valid_json(self) -> None:
        items = _load_sdk_v2_fixture()
        assert items, "synthetic SDK v2 Codex fixture is empty"
        for index, item in enumerate(items):
            assert isinstance(item, dict), f"entry {index} is not a dict"
            assert "type" in item, f"entry {index} is missing the type discriminator"
            assert "id" in item or item.get("type") in {"contextCompaction"}, (
                f"entry {index} of type {item.get('type')!r} is missing an id"
            )

    def test_fixture_covers_every_documented_variant(self) -> None:
        items = _load_sdk_v2_fixture()
        seen = {entry["type"] for entry in items}
        missing = SDK_V2_VARIANT_TYPES - seen
        assert not missing, f"synthetic SDK v2 fixture is missing variants: {sorted(missing)}"

    def test_fixture_uses_only_documented_variants(self) -> None:
        items = _load_sdk_v2_fixture()
        seen = {entry["type"] for entry in items}
        unexpected = seen - SDK_V2_VARIANT_TYPES
        assert not unexpected, (
            "synthetic SDK v2 fixture introduced a variant the schema does "
            f"not document: {sorted(unexpected)}"
        )

    def test_fixture_ids_are_unique_within_their_type(self) -> None:
        items = _load_sdk_v2_fixture()
        by_type: dict[str, list[str]] = {}
        for entry in items:
            by_type.setdefault(entry["type"], []).append(entry.get("id", ""))
        for variant_type, ids in by_type.items():
            assert len(ids) == len(set(ids)), f"duplicate id within {variant_type!r} entries: {ids}"
