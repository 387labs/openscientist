"""Tests for transcript parsing utilities."""

from openscientist.transcript import (
    AssistantText,
    ToolCall,
    ToolResult,
    TranscriptEntry,
)
from openscientist.webapp_components.utils.transcript_parser import (
    extract_usage_summary,
    get_action_description,
    parse_transcript_actions,
)


class TestGetActionDescription:
    def test_explicit_description_takes_precedence(self) -> None:
        call = ToolCall(
            id="t1",
            tool="openscientist__load_data",
            arguments={"description": "Loading test dataset"},
        )
        assert get_action_description(call) == "Loading test dataset"

    def test_search_pubmed_fallback(self) -> None:
        call = ToolCall(
            id="t1",
            tool="openscientist__search_pubmed",
            arguments={"query": "cancer treatment"},
        )
        assert get_action_description(call) == "Search: cancer treatment"

    def test_update_knowledge_state_fallback(self) -> None:
        call = ToolCall(
            id="t1",
            tool="openscientist__update_knowledge_state",
            arguments={"title": "Important finding"},
        )
        assert get_action_description(call) == "Finding: Important finding"

    def test_save_iteration_summary_truncates_long_summaries(self) -> None:
        call = ToolCall(
            id="t1",
            tool="openscientist__save_iteration_summary",
            arguments={
                "summary": "This is a very long summary that should be truncated to fifty characters"
            },
        )
        result = get_action_description(call)
        assert result.startswith("Summary: ")
        assert result.endswith("...")
        assert "This is a very long summary" in result

    def test_execute_code_fallback(self) -> None:
        call = ToolCall(
            id="t1",
            tool="openscientist__execute_code",
            arguments={"code": "print('hello')"},
        )
        assert get_action_description(call) == "Code execution"

    def test_skill_fallback(self) -> None:
        call = ToolCall(id="t1", tool="Skill", arguments={"skill": "review"})
        assert get_action_description(call) == "Skill: review"

    def test_tool_name_short_form_when_no_fallback_matches(self) -> None:
        call = ToolCall(id="t1", tool="openscientist__custom_tool", arguments={})
        assert get_action_description(call) == "custom_tool"

    def test_empty_arguments(self) -> None:
        call = ToolCall(id="t1", tool="execute_code", arguments={})
        assert get_action_description(call) == "Code execution"


class TestParseTranscriptActions:
    def test_empty_transcript(self) -> None:
        assert parse_transcript_actions([]) == []

    def test_single_successful_action(self) -> None:
        transcript: list[TranscriptEntry] = [
            ToolCall(
                id="tool_123",
                tool="openscientist__load_data",
                arguments={"description": "Loading data", "path": "data.csv"},
            ),
            ToolResult(call_id="tool_123", output="Successfully loaded 100 rows", success=True),
        ]
        actions = parse_transcript_actions(transcript)
        assert len(actions) == 1
        assert actions[0]["tool_name"] == "openscientist__load_data"
        assert actions[0]["short_name"] == "load_data"
        assert actions[0]["description"] == "Loading data"
        assert actions[0]["input"]["path"] == "data.csv"
        assert actions[0]["result"] == "Successfully loaded 100 rows"
        assert actions[0]["success"] is True

    def test_failed_action_takes_success_from_typed_field(self) -> None:
        """The typed `ToolResult.success` flag is the source of truth.

        No more inferring success from substrings in the result text.
        """
        transcript: list[TranscriptEntry] = [
            ToolCall(
                id="tool_456",
                tool="openscientist__analyze",
                arguments={"method": "correlation"},
            ),
            ToolResult(
                call_id="tool_456",
                output="something that does not contain the word fail",
                success=False,
            ),
        ]
        actions = parse_transcript_actions(transcript)
        assert len(actions) == 1
        assert actions[0]["success"] is False

    def test_non_openscientist_tools_filtered(self) -> None:
        transcript: list[TranscriptEntry] = [
            ToolCall(id="tool_read", tool="Read", arguments={"file": "test.txt"}),
            ToolCall(
                id="tool_openscientist",
                tool="openscientist__analyze",
                arguments={"method": "test"},
            ),
            ToolResult(call_id="tool_read", output="file content", success=True),
            ToolResult(call_id="tool_openscientist", output="analysis result", success=True),
        ]
        actions = parse_transcript_actions(transcript)
        assert len(actions) == 1
        assert actions[0]["tool_name"] == "openscientist__analyze"

    def test_multiple_actions_in_source_order(self) -> None:
        transcript: list[TranscriptEntry] = [
            ToolCall(id="t1", tool="openscientist__load_data", arguments={"path": "data1.csv"}),
            ToolResult(call_id="t1", output="Success", success=True),
            ToolCall(id="t2", tool="openscientist__analyze", arguments={"method": "test"}),
            ToolResult(call_id="t2", output="Error: Failed to analyze", success=False),
        ]
        actions = parse_transcript_actions(transcript)
        assert len(actions) == 2
        assert actions[0]["success"] is True
        assert actions[1]["success"] is False

    def test_missing_tool_result_defaults_to_empty_and_success(self) -> None:
        transcript: list[TranscriptEntry] = [
            ToolCall(id="tool_missing", tool="openscientist__test", arguments={}),
        ]
        actions = parse_transcript_actions(transcript)
        assert len(actions) == 1
        assert actions[0]["result"] == ""
        assert actions[0]["success"] is True

    def test_bare_tool_names_resolve_via_known_set(self) -> None:
        """Bare names like `execute_code` and `search_pubmed` are recognised."""
        transcript: list[TranscriptEntry] = [
            ToolCall(
                id="t1",
                tool="execute_code",
                arguments={"code": "print('hello')", "description": "Test run"},
            ),
            ToolCall(id="t2", tool="search_pubmed", arguments={"query": "cancer therapy"}),
        ]
        actions = parse_transcript_actions(transcript)
        assert len(actions) == 2
        assert actions[0]["tool_name"] == "execute_code"
        assert actions[0]["short_name"] == "execute_code"
        assert actions[0]["description"] == "Test run"
        assert actions[1]["tool_name"] == "search_pubmed"
        assert actions[1]["description"] == "Search: cancer therapy"

    def test_bare_unknown_tools_filtered(self) -> None:
        transcript: list[TranscriptEntry] = [
            ToolCall(id="t1", tool="Read", arguments={"file": "test.txt"}),
            ToolCall(id="t2", tool="Bash", arguments={"command": "ls"}),
            ToolCall(id="t3", tool="execute_code", arguments={"code": "print(1)"}),
        ]
        actions = parse_transcript_actions(transcript)
        assert len(actions) == 1
        assert actions[0]["tool_name"] == "execute_code"

    def test_assistant_text_entries_ignored(self) -> None:
        """Non-tool entries do not produce actions."""
        transcript: list[TranscriptEntry] = [
            AssistantText(text="Let me analyse the data."),
            ToolCall(id="t1", tool="execute_code", arguments={"code": "print(1)"}),
            ToolResult(call_id="t1", output="1\n", success=True),
            AssistantText(text="Done."),
        ]
        actions = parse_transcript_actions(transcript)
        assert len(actions) == 1
        assert actions[0]["tool_name"] == "execute_code"


class TestExtractUsageSummary:
    def test_empty_transcript(self) -> None:
        summary = extract_usage_summary([])
        assert summary.tool_counts == {}
        assert summary.mcp_tool_calls == 0
        assert summary.skills_used == []

    def test_counts_code_executions_and_pubmed_searches(self) -> None:
        transcript: list[TranscriptEntry] = [
            ToolCall(id="t1", tool="execute_code", arguments={"code": "1"}),
            ToolCall(id="t2", tool="execute_code", arguments={"code": "2"}),
            ToolCall(id="t3", tool="search_pubmed", arguments={"query": "x"}),
            ToolCall(id="t4", tool="update_knowledge_state", arguments={"title": "y"}),
        ]
        summary = extract_usage_summary(transcript)
        assert summary.code_executions == 2
        assert summary.pubmed_searches == 1
        assert summary.findings_recorded == 1
        assert summary.mcp_tool_calls == 4

    def test_collects_skills_in_order_without_duplicates(self) -> None:
        transcript: list[TranscriptEntry] = [
            ToolCall(id="t1", tool="Skill", arguments={"skill": "review"}),
            ToolCall(id="t2", tool="Skill", arguments={"skill": "debug"}),
            ToolCall(id="t3", tool="Skill", arguments={"skill": "review"}),
        ]
        summary = extract_usage_summary(transcript)
        assert summary.skill_invocations == ["review", "debug", "review"]
        assert summary.skills_used == ["review", "debug"]
