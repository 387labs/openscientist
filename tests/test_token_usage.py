"""Tests for TokenUsage arithmetic and SDK usage parsing."""

from openscientist.agent.claude_code_agent import ClaudeCodeAgent
from openscientist.agent.protocol import TokenUsage


def test_add_accumulates_all_fields() -> None:
    a = TokenUsage(
        input_tokens=1,
        output_tokens=2,
        cache_write_tokens=3,
        cache_read_tokens=4,
        reasoning_tokens=5,
    )
    b = TokenUsage(
        input_tokens=10,
        output_tokens=20,
        cache_write_tokens=30,
        cache_read_tokens=40,
        reasoning_tokens=50,
    )
    assert a + b == TokenUsage(
        input_tokens=11,
        output_tokens=22,
        cache_write_tokens=33,
        cache_read_tokens=44,
        reasoning_tokens=55,
    )


def test_iadd_accumulates_all_fields() -> None:
    acc = TokenUsage()
    acc += TokenUsage(
        input_tokens=1,
        output_tokens=2,
        cache_write_tokens=3,
        cache_read_tokens=4,
        reasoning_tokens=5,
    )
    acc += TokenUsage(
        input_tokens=10,
        output_tokens=20,
        cache_write_tokens=30,
        cache_read_tokens=40,
        reasoning_tokens=50,
    )
    assert acc == TokenUsage(
        input_tokens=11,
        output_tokens=22,
        cache_write_tokens=33,
        cache_read_tokens=44,
        reasoning_tokens=55,
    )


def test_anthropic_payload_maps_to_additive_fields() -> None:
    """Anthropic's usage shape is already additive; only field-name remap is needed."""
    usage = {
        "input_tokens": 100,
        "output_tokens": 200,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 20,
    }
    assert ClaudeCodeAgent._usage_from_payload(usage) == TokenUsage(
        input_tokens=100,
        output_tokens=200,
        cache_write_tokens=10,
        cache_read_tokens=20,
        reasoning_tokens=0,
    )


def test_anthropic_object_payload_maps_to_additive_fields() -> None:
    class _UsageObj:
        input_tokens = 100
        output_tokens = 200
        cache_creation_input_tokens = 10
        cache_read_input_tokens = 20

    assert ClaudeCodeAgent._usage_from_payload(_UsageObj()) == TokenUsage(
        input_tokens=100,
        output_tokens=200,
        cache_write_tokens=10,
        cache_read_tokens=20,
        reasoning_tokens=0,
    )


def test_anthropic_payload_reasoning_is_zero() -> None:
    """Anthropic's API does not expose extended-thinking tokens separately."""
    usage = {"input_tokens": 100, "output_tokens": 200}
    assert ClaudeCodeAgent._usage_from_payload(usage).reasoning_tokens == 0
