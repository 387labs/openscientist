"""Tests for the omp :class:`TranscriptDeserializer`.

Covers happy-path translation for each omp role and block type, the
:class:`UnknownEntry` contract, no-drop accounting for unrecognised keys,
and a fixture sweep over the committed ``agent_end_messages.json`` capture.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from openscientist.transcript import (
    OMP,
    AgentMarker,
    AssistantText,
    OmpAgent,
    OmpDeserializer,
    Reasoning,
    ToolCall,
    ToolResult,
    TranscriptAdapter,
    TranscriptDeserializer,
    UnknownEntry,
    UserPrompt,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "transcripts" / "omp"


# ---- Protocol surface -----------------------------------------------------------------------


class TestOmpDeserializerSurface:
    def test_module_singleton_is_an_omp_deserializer(self) -> None:
        assert isinstance(OMP, OmpDeserializer)

    def test_module_singleton_satisfies_the_protocol(self) -> None:
        assert isinstance(OMP, TranscriptDeserializer)

    def test_omp_agent_is_a_marker(self) -> None:
        assert issubclass(OmpAgent, AgentMarker)


# ---- Happy-path per role/block type ---------------------------------------------------------


class TestUserMessage:
    def test_emits_user_prompt_joining_text_blocks(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello world"}],
                "attribution": "user",
                "timestamp": 1000,
            }
        ]
        out = OMP.deserialize(msgs)
        assert len(out) == 1
        assert isinstance(out[0], UserPrompt)
        assert out[0].text == "hello world"

    def test_multiple_text_blocks_joined_with_newline(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "line one"},
                    {"type": "text", "text": "line two"},
                ],
            }
        ]
        out = OMP.deserialize(msgs)
        assert isinstance(out[0], UserPrompt)
        assert out[0].text == "line one\nline two"

    def test_string_content_becomes_user_prompt(self) -> None:
        msgs = [{"role": "user", "content": "direct string"}]
        out = OMP.deserialize(msgs)
        assert isinstance(out[0], UserPrompt)
        assert out[0].text == "direct string"


class TestAssistantTextBlock:
    def test_emits_assistant_text_with_model(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
                "model": "claude-haiku-4-5",
                "api": "anthropic-messages",
                "stopReason": "stop",
            }
        ]
        out = OMP.deserialize(msgs)
        assert len(out) == 1
        assert isinstance(out[0], AssistantText)
        assert out[0].text == "Done."
        assert out[0].model == "claude-haiku-4-5"

    def test_assistant_message_metadata_in_raw(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "model": "m",
                "api": "anthropic-messages",
                "provider": "anthropic",
                "stopReason": "stop",
            }
        ]
        out = OMP.deserialize(msgs)
        assert isinstance(out[0], AssistantText)
        raw = out[0].raw
        assert raw["api"] == "anthropic-messages"
        assert raw["provider"] == "anthropic"
        assert raw["stopReason"] == "stop"


class TestThinkingBlock:
    def test_emits_reasoning_with_signature(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "step by step",
                        "thinkingSignature": "sig-abc",
                    }
                ],
                "model": "m",
            }
        ]
        out = OMP.deserialize(msgs)
        assert len(out) == 1
        assert isinstance(out[0], Reasoning)
        assert out[0].text == "step by step"
        assert out[0].signature == "sig-abc"

    def test_missing_thinking_text_becomes_unknown(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinkingSignature": "sig"}],
                "model": "m",
            }
        ]
        out = OMP.deserialize(msgs)
        assert len(out) == 1
        assert isinstance(out[0], UnknownEntry)
        assert out[0].source == "omp"


class TestToolCallBlock:
    def test_emits_tool_call(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "tc-1",
                        "name": "bash",
                        "arguments": {"command": "ls"},
                    }
                ],
                "model": "m",
            }
        ]
        out = OMP.deserialize(msgs)
        assert len(out) == 1
        assert isinstance(out[0], ToolCall)
        assert out[0].id == "tc-1"
        assert out[0].tool == "bash"
        assert out[0].arguments == {"command": "ls"}

    def test_intent_extra_field_preserved_in_raw(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "tc-2",
                        "name": "bash",
                        "arguments": {},
                        "intent": "do something",
                    }
                ],
                "model": "m",
            }
        ]
        out = OMP.deserialize(msgs)
        assert isinstance(out[0], ToolCall)
        assert out[0].raw["_block_extras"]["intent"] == "do something"


class TestToolResultMessage:
    def test_emits_tool_result_success(self) -> None:
        msgs = [
            {
                "role": "toolResult",
                "toolCallId": "tc-1",
                "toolName": "bash",
                "content": [{"type": "text", "text": "output here"}],
                "isError": False,
                "timestamp": 9999,
            }
        ]
        out = OMP.deserialize(msgs)
        assert len(out) == 1
        assert isinstance(out[0], ToolResult)
        assert out[0].call_id == "tc-1"
        assert out[0].output == "output here"
        assert out[0].success is True

    def test_is_error_true_sets_success_false(self) -> None:
        msgs = [
            {
                "role": "toolResult",
                "toolCallId": "tc-x",
                "toolName": "bash",
                "content": [{"type": "text", "text": "boom"}],
                "isError": True,
            }
        ]
        out = OMP.deserialize(msgs)
        assert isinstance(out[0], ToolResult)
        assert out[0].success is False

    def test_tool_result_metadata_in_raw(self) -> None:
        msgs = [
            {
                "role": "toolResult",
                "toolCallId": "tc-1",
                "toolName": "bash",
                "content": [{"type": "text", "text": "x"}],
                "isError": False,
                "details": {"wallTimeMs": 31},
                "timestamp": 1000,
            }
        ]
        out = OMP.deserialize(msgs)
        assert isinstance(out[0], ToolResult)
        assert out[0].raw["toolName"] == "bash"
        assert out[0].raw["details"] == {"wallTimeMs": 31}
        assert out[0].raw["timestamp"] == 1000


# ---- Unknown handling -----------------------------------------------------------------------


class TestUnknownHandling:
    def test_unknown_role_becomes_unknown_entry(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="openscientist.transcript"):
            out = OMP.deserialize([{"role": "system", "text": "ignored"}])
        assert len(out) == 1
        assert isinstance(out[0], UnknownEntry)
        assert out[0].source == "omp"
        assert out[0].raw["role"] == "system"

    def test_unknown_role_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="openscientist.transcript"):
            OMP.deserialize([{"role": "bogus"}])
        assert any("UnknownEntry" in rec.message for rec in caplog.records)

    def test_unknown_block_type_becomes_unknown_entry(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "novelBlockType", "data": 42}],
                "model": "m",
            }
        ]
        out = OMP.deserialize(msgs)
        assert len(out) == 1
        assert isinstance(out[0], UnknownEntry)
        assert out[0].source == "omp"

    def test_unknown_block_type_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="openscientist.transcript"):
            OMP.deserialize(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "future_type"}],
                        "model": "m",
                    }
                ]
            )
        assert any("UnknownEntry" in rec.message for rec in caplog.records)

    def test_non_dict_message_becomes_unknown_entry(self) -> None:
        out = OMP.deserialize(["not-a-dict"])  # type: ignore[list-item]
        assert len(out) == 1
        assert isinstance(out[0], UnknownEntry)


# ---- No-drop accounting --------------------------------------------------------------------


class TestNoDrop:
    def test_extra_key_on_user_message_in_raw(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
                "attribution": "human",
                "timestamp": 12345,
                "_extra_unrecognised": "preserved",
            }
        ]
        out = OMP.deserialize(msgs)
        assert isinstance(out[0], UserPrompt)
        assert out[0].raw["_extra_unrecognised"] == "preserved"
        assert out[0].raw["attribution"] == "human"
        assert out[0].raw["timestamp"] == 12345

    def test_extra_key_on_assistant_message_in_raw(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "model": "m",
                "usage": {"input": 1, "output": 1},
                "_unrecognised_msg_field": "msg-level",
            }
        ]
        out = OMP.deserialize(msgs)
        assert isinstance(out[0], AssistantText)
        assert out[0].raw["_unrecognised_msg_field"] == "msg-level"
        assert out[0].raw["usage"] == {"input": 1, "output": 1}

    def test_extra_key_on_block_preserved_in_raw(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi", "_block_extra": "block-level"}],
                "model": "m",
            }
        ]
        out = OMP.deserialize(msgs)
        assert isinstance(out[0], AssistantText)
        assert out[0].raw["_block_extras"]["_block_extra"] == "block-level"


# ---- Fixture sweep -------------------------------------------------------------------------


class TestFixtureSweep:
    def test_fixture_exists(self) -> None:
        path = FIXTURE_DIR / "agent_end_messages.json"
        assert path.exists(), f"fixture not found: {path}"

    def test_entry_types_in_order(self) -> None:
        data: list[dict[str, Any]] = json.loads(
            (FIXTURE_DIR / "agent_end_messages.json").read_text()
        )
        entries = OMP.deserialize(data)
        types = [type(e).__name__ for e in entries]
        assert types == [
            "UserPrompt",
            "Reasoning",
            "ToolCall",
            "ToolResult",
            "Reasoning",
            "AssistantText",
        ], f"unexpected entry type sequence: {types}"

    def test_tool_call_name_is_bash(self) -> None:
        data: list[dict[str, Any]] = json.loads(
            (FIXTURE_DIR / "agent_end_messages.json").read_text()
        )
        entries = OMP.deserialize(data)
        calls = [e for e in entries if isinstance(e, ToolCall)]
        assert len(calls) == 1
        assert calls[0].tool == "bash"

    def test_tool_result_success_is_true(self) -> None:
        data: list[dict[str, Any]] = json.loads(
            (FIXTURE_DIR / "agent_end_messages.json").read_text()
        )
        entries = OMP.deserialize(data)
        results = [e for e in entries if isinstance(e, ToolResult)]
        assert len(results) == 1
        assert results[0].success is True

    def test_round_trips_through_transcript_adapter(self) -> None:
        data: list[dict[str, Any]] = json.loads(
            (FIXTURE_DIR / "agent_end_messages.json").read_text()
        )
        entries = OMP.deserialize(data)
        raw = TranscriptAdapter.dump_json(entries)
        restored = TranscriptAdapter.validate_json(raw)
        assert restored == entries

    def test_no_unknown_entries_in_fixture(self) -> None:
        data: list[dict[str, Any]] = json.loads(
            (FIXTURE_DIR / "agent_end_messages.json").read_text()
        )
        entries = OMP.deserialize(data)
        unknowns = [e for e in entries if isinstance(e, UnknownEntry)]
        assert not unknowns, f"produced UnknownEntry: {unknowns[0].raw}"
