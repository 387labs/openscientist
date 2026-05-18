"""Shared helpers used by every backend translator."""

import logging
from typing import Any, Literal

from pydantic import BaseModel

from openscientist.transcript.variants import UnknownEntry

logger = logging.getLogger("openscientist.transcript")


def wrapper_overlay(entry_extras: dict[str, Any], message_extras: dict[str, Any]) -> dict[str, Any]:
    """Combine entry-level and message-level leftover keys into an overlay
    propagated onto every entry produced from a single source message."""
    overlay: dict[str, Any] = {}
    if entry_extras:
        overlay["_entry_extras"] = entry_extras
    if message_extras:
        overlay["_message_extras"] = message_extras
    return overlay


def block_extras_or_none(extras: dict[str, Any]) -> dict[str, Any] | None:
    return extras if extras else None


def merge_overlay(overlay: dict[str, Any], block_extras: dict[str, Any] | None) -> dict[str, Any]:
    """Combine wrapper overlay with block-level leftover keys."""
    merged = dict(overlay)
    if block_extras:
        merged["_block_extras"] = block_extras
    return merged


def full_block_overlay(block: dict[str, object], overlay: dict[str, Any]) -> dict[str, Any]:
    """Build a raw dict for ``UnknownEntry`` that captures the offending block
    plus the surrounding wrapper context, so no data is lost."""
    full = dict(overlay)
    full["_block"] = dict(block)
    return full


def unknown(
    source: Literal["claude", "codex"],
    raw: dict[str, Any],
    reason: str,
) -> UnknownEntry:
    """Construct an :class:`UnknownEntry` and log a WARNING.

    Per the no-drop contract, every translator MUST go through this
    helper when emitting an :class:`UnknownEntry`, so unrecognised
    shapes surface in operational logs.
    """
    logger.warning(
        "%s transcript translator: %s. Stored as UnknownEntry",
        source,
        reason,
    )
    return UnknownEntry(source=source, raw=raw)


def unknown_block(
    source: Literal["claude", "codex"],
    overlay: dict[str, Any],
    block: BaseModel,
    reason: str,
) -> UnknownEntry:
    """``unknown`` variant for content blocks: serialises the block via
    ``model_dump`` into ``raw['_block']``."""
    raw = dict(overlay)
    raw["_block"] = block.model_dump(mode="json")
    return unknown(source, raw, reason)


def safe_str(value: str | None, key: str) -> str:
    """Substitute an empty string for a missing required field, after
    logging a WARNING. Used by translators where dropping the entry would
    be worse than producing one with an empty field."""
    if value is None:
        logger.warning(
            "transcript translator: missing required field %s. Substituting empty string",
            key,
        )
        return ""
    return value


def coerce_tool_result_content(content: object) -> str:
    """Flatten an Anthropic Messages API ``tool_result.content`` field to a
    single string.

    The API allows ``content`` to be either a plain ``str`` or a list of
    ``{"type": "text", "text": "..."}`` blocks. Any other shape is
    best-effort-stringified so the translator never raises on
    legitimate-looking input.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    if content is None:
        return ""
    return str(content)
