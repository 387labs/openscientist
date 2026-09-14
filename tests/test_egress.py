"""Tests for the air-gapped egress allowlist dispatcher.

The dispatcher is tested in isolation over an AirgapPosture. Which posture each
provider reports is covered by the provider tests.
"""

from types import SimpleNamespace
from typing import cast

import pytest

from openscientist.job_container.egress import (
    AirgapProviderError,
    derive_egress_allowlist,
    format_egress_allowlist,
)
from openscientist.providers.base import AirgapEgress, AirgapPosture
from openscientist.settings import Settings

BROKER = ("openscientist", 8082)
PROXY = ("openscientist", 8081)


@pytest.fixture(autouse=True)
def _fixed_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "openscientist.job_container.egress.container_broker_base_url",
        lambda: "http://openscientist:8082",
    )
    monkeypatch.setattr(
        "openscientist.llm_proxy.container_proxy_base_url",
        lambda: "http://openscientist:8081",
    )


def _settings(database_url: str = "postgresql+asyncpg://u:p@postgres:5432/db") -> Settings:
    return cast(
        Settings,
        SimpleNamespace(database=SimpleNamespace(effective_database_url=database_url)),
    )


def test_proxy_posture_allows_db_broker_and_proxy() -> None:
    entries = derive_egress_allowlist(_settings(), AirgapPosture(AirgapEgress.PROXY))
    assert ("postgres", 5432) in entries
    assert BROKER in entries
    assert PROXY in entries


def test_proxy_posture_excludes_any_provider_endpoint() -> None:
    entries = derive_egress_allowlist(_settings(), AirgapPosture(AirgapEgress.PROXY))
    hosts = {host for host, _ in entries}
    assert hosts == {"postgres", "openscientist"}


def test_direct_posture_allows_endpoints_not_proxy() -> None:
    posture = AirgapPosture(
        AirgapEgress.DIRECT,
        direct_endpoints=(("bedrock-runtime.us-east-1.amazonaws.com", 443),),
    )
    entries = derive_egress_allowlist(_settings(), posture)
    assert ("postgres", 5432) in entries
    assert BROKER in entries
    assert ("bedrock-runtime.us-east-1.amazonaws.com", 443) in entries
    assert PROXY not in entries


def test_direct_posture_multiple_endpoints() -> None:
    posture = AirgapPosture(
        AirgapEgress.DIRECT,
        direct_endpoints=(
            ("us-east5-aiplatform.googleapis.com", 443),
            ("oauth2.googleapis.com", 443),
        ),
    )
    entries = derive_egress_allowlist(_settings(), posture)
    assert ("us-east5-aiplatform.googleapis.com", 443) in entries
    assert ("oauth2.googleapis.com", 443) in entries


def test_unsupported_posture_refused() -> None:
    posture = AirgapPosture(AirgapEgress.UNSUPPORTED, reason="Foo cannot be air-gapped.")
    with pytest.raises(AirgapProviderError, match="cannot be air-gapped"):
        derive_egress_allowlist(_settings(), posture)


def test_postgres_default_port_when_absent() -> None:
    entries = derive_egress_allowlist(
        _settings(database_url="postgresql+asyncpg://u:p@db-host/db"),
        AirgapPosture(AirgapEgress.PROXY),
    )
    assert ("db-host", 5432) in entries


def test_deduplicates_entries() -> None:
    posture = AirgapPosture(AirgapEgress.DIRECT, direct_endpoints=(("openscientist", 8082),))
    entries = derive_egress_allowlist(_settings(), posture)
    assert entries.count(("openscientist", 8082)) == 1


def test_format_egress_allowlist() -> None:
    rendered = format_egress_allowlist([("postgres", 5432), ("openscientist", 8081)])
    assert rendered == "postgres:5432,openscientist:8081"
