"""Stdio entry point: ``python -m openscientist_tools``."""

from __future__ import annotations

import sys
from typing import Any

import mcp.types as types

from openscientist_tools.server import mcp
from openscientist_tools.state import STATE

HANDSHAKE_MARKER = "mcp_handshake"


def _install_handshake_marker() -> None:
    """Record each tools/list. Wraps the ``request_handlers`` entry because
    ``FastMCP`` binds its own ``list_tools`` there at construction."""
    marker = STATE.job_dir / ".omp" / HANDSHAKE_MARKER
    handlers = mcp._mcp_server.request_handlers
    original = handlers.get(types.ListToolsRequest)
    if original is None:
        print("no ListToolsRequest handler, handshake marker disabled", file=sys.stderr)
        return

    async def record_then_list(*args: Any, **kwargs: Any) -> types.ServerResult:
        # After the await, so the marker attests delivery.
        result = await original(*args, **kwargs)
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        except OSError as error:
            print(f"could not write handshake marker: {error}", file=sys.stderr)
        return result

    handlers[types.ListToolsRequest] = record_then_list


if __name__ == "__main__":
    _install_handshake_marker()
    mcp.run()
