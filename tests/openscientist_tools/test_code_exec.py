"""Tests for the standalone `execute_code` tool."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent
from sqlalchemy import delete

from openscientist.database import AsyncSessionLocal
from openscientist.database.models.job import Job
from openscientist.knowledge_state import KnowledgeState
from openscientist_tools import code_exec as code_exec_module
from openscientist_tools.code_exec import execute_code
from openscientist_tools.state import STATE


def _docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        client.images.get("openscientist-executor:latest")
        return True
    except Exception:
        return False


DOCKER_REQUIRED = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon and openscientist-executor:latest image required",
)


@asynccontextmanager
async def _spawned_for_job(
    server_env: Callable[..., dict[str, str]],
    server_params: Callable[[dict[str, str]], StdioServerParameters],
    tmp_path: Path,
    test_database_url: str,
    job_id: UUID,
    *,
    env_overrides: dict[str, str] | None = None,
) -> AsyncGenerator[ClientSession, None]:
    async with AsyncSessionLocal(thread_safe=True) as setup:
        setup.add(
            Job(
                id=job_id,
                research_question="code_exec subprocess test",
                llm_provider="mock",
                llm_config={"model": "mock-model-v1"},
                status="pending",
            )
        )
        await setup.commit()
    try:
        env = server_env(tmp_path, OPENSCIENTIST_JOB_ID=str(job_id))
        env["DATABASE_URL"] = test_database_url
        env["OPENSCIENTIST_SECRET_KEY"] = os.environ["OPENSCIENTIST_SECRET_KEY"]
        if env_overrides:
            env.update(env_overrides)
        params = server_params(env)
        params.cwd = str(tmp_path)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    finally:
        async with AsyncSessionLocal(thread_safe=True) as cleanup:
            await cleanup.execute(delete(Job).where(Job.id == job_id))
            await cleanup.commit()


def _text(result: object) -> str:
    blocks = result.content  # type: ignore[attr-defined]
    (block,) = blocks
    assert isinstance(block, TextContent)
    return block.text


@pytest.fixture(autouse=True)
def _state_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(STATE, "job_id", "test-job-uuid")


@pytest.fixture
def state_job_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point STATE.job_dir at tmp_path."""
    monkeypatch.setattr(STATE, "job_dir", tmp_path)
    monkeypatch.setattr(STATE, "data_file", None)
    monkeypatch.setattr(STATE, "data_files", ())
    return tmp_path


@pytest.fixture(autouse=True)
def _clear_data_caches() -> None:
    """Module-level data caches leak across tests; reset each run."""
    code_exec_module._DATA_CACHE.clear()
    code_exec_module._DATA_LOADED.clear()
    code_exec_module._DATA_ERROR.clear()


def _mock_container_manager(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]) -> MagicMock:
    """Patch `get_container_manager` to return a mock whose execute_code
    returns ``result``. Returns the mock so call args can be inspected."""
    fake_mgr = MagicMock()
    fake_mgr.execute_code.return_value = result
    monkeypatch.setattr(code_exec_module, "get_container_manager", lambda: fake_mgr)
    return fake_mgr


# ----- In-process branch coverage -----


def test_execute_code_python_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    state_job_dir: Path,
    patched_ks_persistence: KnowledgeState,
) -> None:
    fake_mgr = _mock_container_manager(
        monkeypatch,
        {"success": True, "output": "hello world", "execution_time": 0.5, "plots": []},
    )

    result = execute_code("print('hello world')")

    assert "hello world" in result
    assert "successfully" in result.lower() or "✅" in result
    fake_mgr.execute_code.assert_called_once()
    call_kwargs = fake_mgr.execute_code.call_args.kwargs
    assert call_kwargs["language"] == "python"
    assert call_kwargs["timeout"] == 60

    last_log = patched_ks_persistence.data["analysis_log"][-1]
    assert last_log["action"] == "execute_code"
    assert last_log["success"] is True


def test_execute_code_rust_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    state_job_dir: Path,
    patched_ks_persistence: KnowledgeState,
) -> None:
    fake_mgr = _mock_container_manager(
        monkeypatch,
        {"success": True, "output": "42", "execution_time": 1.2, "plots": []},
    )

    execute_code('fn main(){println!("42");}', language="rust")

    call_kwargs = fake_mgr.execute_code.call_args.kwargs
    assert call_kwargs["language"] == "rust"
    assert call_kwargs["timeout"] == 300
    assert "data_files" not in call_kwargs
    assert "data_path" not in call_kwargs


def test_execute_code_sparql_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    state_job_dir: Path,
    patched_ks_persistence: KnowledgeState,
) -> None:
    fake_mgr = _mock_container_manager(
        monkeypatch,
        {"success": True, "output": "rows: 0", "execution_time": 0.3, "plots": []},
    )

    execute_code("SELECT * WHERE { ?s ?p ?o } LIMIT 1", language="sparql")

    call_kwargs = fake_mgr.execute_code.call_args.kwargs
    assert call_kwargs["language"] == "sparql"
    assert call_kwargs["timeout"] == 60


def test_execute_code_invalid_language_returns_error(
    monkeypatch: pytest.MonkeyPatch,
    state_job_dir: Path,
    patched_ks_persistence: KnowledgeState,
) -> None:
    result = execute_code("anything", language="lisp")
    assert result == (
        "❌ ERROR: Unsupported language 'lisp'. Supported: 'python', 'rust', 'sparql'"
    )
    assert patched_ks_persistence.data["analysis_log"] == []


def test_execute_code_failure_returns_error_format(
    monkeypatch: pytest.MonkeyPatch,
    state_job_dir: Path,
    patched_ks_persistence: KnowledgeState,
) -> None:
    _mock_container_manager(
        monkeypatch,
        {
            "success": False,
            "output": "partial output here",
            "error": "ZeroDivisionError: division by zero",
            "traceback": "Traceback (most recent call last):\n  ...",
            "execution_time": 0.1,
            "plots": [],
        },
    )

    result = execute_code("print(1/0)")
    assert "❌" in result or "failed" in result.lower()
    assert "ZeroDivisionError" in result or "division by zero" in result

    last_log = patched_ks_persistence.data["analysis_log"][-1]
    assert last_log["success"] is False


def test_execute_code_python_with_data_files(
    monkeypatch: pytest.MonkeyPatch,
    state_job_dir: Path,
    patched_ks_persistence: KnowledgeState,
) -> None:
    (state_job_dir / "a.csv").write_text("col\n1\n2\n")
    (state_job_dir / "b.csv").write_text("col\n3\n4\n")
    monkeypatch.setattr(STATE, "data_files", (state_job_dir / "a.csv", state_job_dir / "b.csv"))

    fake_mgr = _mock_container_manager(
        monkeypatch,
        {"success": True, "output": "", "execution_time": 0.1, "plots": []},
    )

    execute_code("pass")

    call_kwargs = fake_mgr.execute_code.call_args.kwargs
    assert call_kwargs["data_path"] == str(state_job_dir / "a.csv")
    assert len(call_kwargs["data_files"]) == 2
    assert {f["name"] for f in call_kwargs["data_files"]} == {"a.csv", "b.csv"}


def test_execute_code_python_data_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    state_job_dir: Path,
    patched_ks_persistence: KnowledgeState,
) -> None:
    missing = state_job_dir / "missing.csv"
    monkeypatch.setattr(STATE, "data_file", missing)
    fake_mgr = _mock_container_manager(
        monkeypatch,
        {"success": True, "output": "", "execution_time": 0, "plots": []},
    )

    def _raise_load(_path: Path) -> None:
        raise FileNotFoundError(str(_path))

    monkeypatch.setattr(code_exec_module, "load_data_file", _raise_load)
    # `_ensure_data_loaded` looks at file size before calling load_data_file,
    # so make sure the file exists so we hit the load branch.
    missing.write_text("dummy")

    result = execute_code("print('hi')")
    assert "❌ ERROR: Cannot execute code" in result
    assert "data file failed to load" in result
    fake_mgr.execute_code.assert_not_called()


def test_execute_code_cache_hits_on_second_call(
    monkeypatch: pytest.MonkeyPatch,
    state_job_dir: Path,
    patched_ks_persistence: KnowledgeState,
) -> None:
    data_file = state_job_dir / "data.csv"
    data_file.write_text("col\n1\n2\n")
    monkeypatch.setattr(STATE, "data_file", data_file)

    call_count = {"n": 0}

    def _counting_loader(_path: Path) -> object:
        call_count["n"] += 1
        fake = MagicMock()
        fake.shape = (2, 1)
        return fake

    monkeypatch.setattr(code_exec_module, "load_data_file", _counting_loader)
    _mock_container_manager(
        monkeypatch,
        {"success": True, "output": "", "execution_time": 0, "plots": []},
    )

    execute_code("pass")
    execute_code("pass")

    assert call_count["n"] == 1


# ----- Full-call subprocess + real-DB + real-Docker -----


@DOCKER_REQUIRED
async def test_execute_python_via_subprocess(
    tmp_path: Path,
    server_env: Callable[..., dict[str, str]],
    server_params: Callable[[dict[str, str]], StdioServerParameters],
    test_database_url: str,
    _apply_migrations_once: None,
) -> None:
    job_id = uuid4()
    async with _spawned_for_job(
        server_env, server_params, tmp_path, test_database_url, job_id
    ) as mcp:
        response = await mcp.call_tool(
            "execute_code",
            {
                "code": "print('hello world')",
                "language": "python",
                "description": "smoke test description",
            },
        )
        text = _text(response)
        assert "hello world" in text

        reloaded = KnowledgeState.load_from_database_sync(str(job_id))
        last_log = reloaded.data["analysis_log"][-1]
        assert last_log["action"] == "execute_code"
        assert last_log["success"] is True
        assert last_log["description"] == "smoke test description"


@DOCKER_REQUIRED
async def test_execute_rust_via_subprocess(
    tmp_path: Path,
    server_env: Callable[..., dict[str, str]],
    server_params: Callable[[dict[str, str]], StdioServerParameters],
    test_database_url: str,
    _apply_migrations_once: None,
) -> None:
    job_id = uuid4()
    async with _spawned_for_job(
        server_env, server_params, tmp_path, test_database_url, job_id
    ) as mcp:
        response = await mcp.call_tool(
            "execute_code",
            {
                "code": 'fn main() { println!("hi from rust"); }',
                "language": "rust",
            },
        )
        text = _text(response)
        assert "hi from rust" in text

        reloaded = KnowledgeState.load_from_database_sync(str(job_id))
        last_log = reloaded.data["analysis_log"][-1]
        assert last_log["action"] == "execute_code"
        assert last_log["success"] is True


@DOCKER_REQUIRED
async def test_execute_python_with_plot_via_subprocess(
    tmp_path: Path,
    server_env: Callable[..., dict[str, str]],
    server_params: Callable[[dict[str, str]], StdioServerParameters],
    test_database_url: str,
    _apply_migrations_once: None,
) -> None:
    job_id = uuid4()
    code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.plot([1, 2, 3], [1, 4, 9])\n"
        "plt.title('subprocess test plot')\n"
        "plt.show()\n"
    )

    # The standalone tool writes plots to STATE.job_dir/provenance.
    # _spawned_for_job uses tmp_path as STATE.job_dir.
    async with _spawned_for_job(
        server_env, server_params, tmp_path, test_database_url, job_id
    ) as mcp:
        response = await mcp.call_tool(
            "execute_code",
            {"code": code, "language": "python"},
        )
        text = _text(response)
        assert "plot" in text.lower() or "✅" in text

        plots = sorted((tmp_path / "provenance").glob("plot_*.png"))
        assert plots, "no plot files were written by the executor"

        reloaded = KnowledgeState.load_from_database_sync(str(job_id))
        last_log = reloaded.data["analysis_log"][-1]
        assert last_log["success"] is True
        assert len(last_log["plots"]) >= 1


@DOCKER_REQUIRED
async def test_execute_python_with_data_file_via_subprocess(
    tmp_path: Path,
    server_env: Callable[..., dict[str, str]],
    server_params: Callable[[dict[str, str]], StdioServerParameters],
    test_database_url: str,
    _apply_migrations_once: None,
) -> None:
    """Mount a CSV into the executor and read it from Python end-to-end."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "sample.csv"
    csv_path.write_text("value\nsentinel-42\nother\n")

    job_id = uuid4()
    code = (
        "import pandas as pd\n"
        "df = pd.read_csv(data_files[0]['path'])\n"
        "print('rows:', len(df))\n"
        "print('first:', df.iloc[0]['value'])\n"
    )

    async with _spawned_for_job(
        server_env,
        server_params,
        tmp_path,
        test_database_url,
        job_id,
        env_overrides={"OPENSCIENTIST_DATA_FILES": str(csv_path)},
    ) as mcp:
        response = await mcp.call_tool(
            "execute_code",
            {"code": code, "language": "python"},
        )
        text = _text(response)
        assert "rows: 2" in text
        assert "first: sentinel-42" in text

        reloaded = KnowledgeState.load_from_database_sync(str(job_id))
        last_log = reloaded.data["analysis_log"][-1]
        assert last_log["success"] is True


@DOCKER_REQUIRED
async def test_execute_python_with_multiple_data_files_via_subprocess(
    tmp_path: Path,
    server_env: Callable[..., dict[str, str]],
    server_params: Callable[[dict[str, str]], StdioServerParameters],
    test_database_url: str,
    _apply_migrations_once: None,
) -> None:
    """Mount two CSVs via os.pathsep-joined env and read both inside the container."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.csv").write_text("value\nalpha\n")
    (data_dir / "b.csv").write_text("value\nbravo\n")

    job_id = uuid4()
    joined = os.pathsep.join([str(data_dir / "a.csv"), str(data_dir / "b.csv")])
    code = (
        "import pandas as pd\n"
        "print('count:', len(data_files))\n"
        "for entry in data_files:\n"
        "    df = pd.read_csv(entry['path'])\n"
        "    print('cell:', df.iloc[0]['value'])\n"
    )

    async with _spawned_for_job(
        server_env,
        server_params,
        tmp_path,
        test_database_url,
        job_id,
        env_overrides={"OPENSCIENTIST_DATA_FILES": joined},
    ) as mcp:
        response = await mcp.call_tool(
            "execute_code",
            {"code": code, "language": "python"},
        )
        text = _text(response)
        assert "count: 2" in text
        assert "cell: alpha" in text
        assert "cell: bravo" in text


@DOCKER_REQUIRED
async def test_execute_sparql_via_subprocess(
    tmp_path: Path,
    server_env: Callable[..., dict[str, str]],
    server_params: Callable[[dict[str, str]], StdioServerParameters],
    test_database_url: str,
    _apply_migrations_once: None,
) -> None:
    """Run a real SPARQL query against Wikidata end-to-end."""
    job_id = uuid4()
    code = (
        "# ENDPOINT: https://query.wikidata.org/sparql\n"
        "SELECT ?item WHERE { ?item wdt:P31 wd:Q5 } LIMIT 1\n"
    )

    async with _spawned_for_job(
        server_env, server_params, tmp_path, test_database_url, job_id
    ) as mcp:
        response = await mcp.call_tool(
            "execute_code",
            {"code": code, "language": "sparql"},
        )
        text = _text(response)
        # Wikidata returns at least one row; the formatter emits an item URI.
        assert "item" in text.lower() or "wikidata" in text.lower()

        reloaded = KnowledgeState.load_from_database_sync(str(job_id))
        last_log = reloaded.data["analysis_log"][-1]
        assert last_log["action"] == "execute_code"
        assert last_log["success"] is True


@DOCKER_REQUIRED
async def test_execute_python_failure_via_subprocess(
    tmp_path: Path,
    server_env: Callable[..., dict[str, str]],
    server_params: Callable[[dict[str, str]], StdioServerParameters],
    test_database_url: str,
    _apply_migrations_once: None,
) -> None:
    job_id = uuid4()
    async with _spawned_for_job(
        server_env, server_params, tmp_path, test_database_url, job_id
    ) as mcp:
        response = await mcp.call_tool(
            "execute_code",
            {"code": "print(1 / 0)", "language": "python"},
        )
        text = _text(response)
        assert "ZeroDivisionError" in text or "division by zero" in text

        reloaded = KnowledgeState.load_from_database_sync(str(job_id))
        last_log = reloaded.data["analysis_log"][-1]
        assert last_log["action"] == "execute_code"
        assert last_log["success"] is False
