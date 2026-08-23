"""Tests for LLM pricing normalization and fallback cost math."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from openscientist.providers import pricing as pricing_mod


@pytest.fixture(autouse=True)
def _reset_pricing_cache() -> Iterator[None]:
    """Keep pricing module cache isolated across tests."""
    previous_cache = pricing_mod._cache
    previous_fetched_at = pricing_mod._cache_fetched_at
    pricing_mod._cache = {}
    pricing_mod._cache_fetched_at = 0.0
    try:
        yield
    finally:
        pricing_mod._cache = previous_cache
        pricing_mod._cache_fetched_at = previous_fetched_at


def test_normalize_model_name_strips_bedrock_and_vertex_affixes() -> None:
    assert (
        pricing_mod.normalize_model_name("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        == "claude-sonnet-4-5"
    )
    assert pricing_mod.normalize_model_name("claude-sonnet-4-5@20250929") == "claude-sonnet-4-5"
    assert pricing_mod.normalize_model_name("claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_estimate_cost_usd_uses_fallback_rates_for_normalized_ids() -> None:
    # Force empty-cache fallback path without hitting the network.
    with patch(
        "openscientist.providers.pricing.requests.get",
        side_effect=RuntimeError("network disabled"),
    ):
        bedrock_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        cost = pricing_mod.estimate_cost_usd(bedrock_id, input_tokens=1000, output_tokens=500)

    expected = 3e-6 * 1000 + 15e-6 * 500
    assert cost == pytest.approx(expected)
    assert cost > 0.0


def test_estimate_cost_usd_returns_zero_for_unknown_model() -> None:
    with patch(
        "openscientist.providers.pricing.requests.get",
        side_effect=RuntimeError("network disabled"),
    ):
        assert pricing_mod.estimate_cost_usd("missing-model-xyz", 10, 10) == 0.0
