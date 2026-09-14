"""search_pubmed dispatch: NCBI by default, local corpus when air-gapped."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import openscientist.literature as literature
from openscientist.settings import AirgapSettings


def _airgap(enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(airgap=SimpleNamespace(enabled=enabled))


def test_airgap_setting_defaults_off_and_parses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENSCIENTIST_AIRGAPPED", raising=False)
    assert AirgapSettings().enabled is False
    monkeypatch.setenv("OPENSCIENTIST_AIRGAPPED", "1")
    assert AirgapSettings().enabled is True


def test_search_pubmed_uses_local_when_airgapped(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_local(query: str, max_results: int = 10) -> list[dict[str, str]]:
        calls["args"] = (query, max_results)
        return [{"pmid": "1", "title": "t", "abstract": "a", "authors": "x", "year": "2020"}]

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("NCBI must not be contacted in air-gapped mode")

    monkeypatch.setattr(literature, "get_settings", lambda: _airgap(True))
    monkeypatch.setattr("openscientist.pubmed_mirror.query.search_local_sync", fake_local)
    monkeypatch.setattr(literature.requests, "get", boom)

    papers = literature.search_pubmed("cancer", max_results=3)

    assert calls["args"] == ("cancer", 3)
    assert papers[0]["pmid"] == "1"


def test_search_pubmed_uses_ncbi_when_not_airgapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom_local(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("local corpus must not be queried when air-gap is off")

    esearch = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"esearchresult": {"idlist": []}},
    )

    monkeypatch.setattr(literature, "get_settings", lambda: _airgap(False))
    monkeypatch.setattr("openscientist.pubmed_mirror.query.search_local_sync", boom_local)
    monkeypatch.setattr(literature.requests, "get", lambda *a, **k: esearch)

    assert literature.search_pubmed("cancer") == []
