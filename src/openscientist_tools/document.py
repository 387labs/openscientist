"""Standalone `read_document` tool."""

from __future__ import annotations

from pathlib import Path

from openscientist.document_reader import read_document as _backend_read_document
from openscientist_tools.server import mcp
from openscientist_tools.state import STATE


@mcp.tool()
def read_document(file_path: str) -> str:
    """Read text content from a PDF, Word, or Excel document.

    Absolute paths are used as-is. Relative paths resolve to
    ``<STATE.job_dir>/data/<basename(file_path)>``.
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = STATE.job_dir / "data" / path.name
    if not path.exists():
        data_dir = STATE.job_dir / "data"
        if data_dir.exists():
            available = [f.name for f in data_dir.iterdir() if f.is_file()]
            if available:
                return (
                    f"❌ File not found: {file_path}\n\n"
                    f"Available files in data directory:\n"
                    + "\n".join(f"  - {name}" for name in available)
                )
        return f"❌ File not found: {file_path}"
    try:
        return _backend_read_document(path)
    except Exception as e:
        return f"❌ Failed to read document: {e}"
