"""Async client for HermesAgent KB (search_api at localhost:8765).

Memory is scoped per user: room_id = "{platform}:{user_id}"
Both user_id and chat_id are stored as metadata for flexible future queries.
"""

from __future__ import annotations

import os

import httpx
from loguru import logger

_KB_BASE = os.environ.get("KB_BASE_URL", "http://localhost:8765")
_KB_TIMEOUT = httpx.Timeout(connect=1.0, read=5.0, write=3.0, pool=1.0)

# Shared persistent client — avoids per-request connection setup overhead.
_client = httpx.AsyncClient(timeout=_KB_TIMEOUT)


def room_id_for(platform: str, user_id: str) -> str:
    """Canonical room_id scoped to a unique user: "telegram:111222333"."""
    return f"{platform}:{user_id}"


async def fetch_recent(room_id: str, limit: int = 3) -> list[dict]:
    """Return the *limit* most recent KB messages for *room_id*, oldest-first.

    Used as the chronological anchor in the hybrid context strategy.
    Returns [] silently on any connectivity error.
    """
    try:
        resp = await _client.get(
            f"{_KB_BASE}/recent",
            params={"room": room_id, "limit": limit, "type": "agent_message"},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return list(reversed(results))  # /recent is newest-first; reverse for prompt
    except Exception as exc:
        logger.debug("kb_client.fetch_recent room={} err={}: {}", room_id, type(exc).__name__, exc)
        return []


async def search_room(room_id: str, query: str, limit: int = 3) -> list[dict]:
    """Semantic search within *room_id* — returns top *limit* relevant entries.

    Used as the topic-depth component in the hybrid context strategy.
    Results are in score-descending order (most relevant first).
    Returns [] silently on any connectivity error.
    """
    try:
        resp = await _client.post(
            f"{_KB_BASE}/kb/search-room",
            json={"query": query, "room_id": room_id, "limit": limit},
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as exc:
        logger.debug("kb_client.search_room room={} err={}: {}", room_id, type(exc).__name__, exc)
        return []


async def store(
    room_id: str,
    content: str,
    *,
    sender: str,
    user_id: str,
    chat_id: str,
    platform: str,
) -> None:
    """Store one message turn to the KB for *room_id*. Fire-and-forget safe.

    sender: "user" for incoming messages, "claude" for assistant replies.
    Both user_id and chat_id are stored in metadata for flexible future queries:
      - filter by room_id   → all memory for this user across chats
      - filter by chat_id   → all messages in a specific group
      - filter by both      → one user's messages in one group
    """
    if not content.strip():
        return
    try:
        await _client.post(
            f"{_KB_BASE}/kb/store",
            json={
                "content": content,
                "content_type": "agent_message",
                "metadata": {
                    "from": sender,
                    "room_id": room_id,
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "platform": platform,
                    "direction": "inbound" if sender == "user" else "outbound",
                },
            },
        )
    except Exception as exc:
        logger.debug("kb_client.store room={} sender={} err={}: {}", room_id, sender, type(exc).__name__, exc)


def dedupe(entries: list[dict]) -> list[dict]:
    """Deduplicate by entry id, preserving order (first occurrence wins)."""
    seen: set[str] = set()
    out: list[dict] = []
    for e in entries:
        eid = e.get("id")
        if eid is None:
            logger.warning("kb_client.dedupe: entry missing 'id', including as-is")
            out.append(e)
            continue
        key = str(eid)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def build_context_prefix(messages: list[dict], max_chars: int = 3000) -> str:
    """Format KB entries as a plain-text context block for prompt injection.

    Accepts results from either fetch_recent or search_room (same shape).
    Truncates to *max_chars* from the end (keeps most recent/relevant content).
    Returns empty string if no usable messages.
    """
    lines: list[str] = []
    for msg in messages:
        meta = msg.get("metadata") or {}
        sender = meta.get("from", "")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        label = "User" if sender == "user" else "Assistant"
        lines.append(f"{label}: {content[:600]}")

    if not lines:
        return ""

    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[-max_chars:]
        cut = block.find("\n")
        if cut != -1:
            block = block[cut + 1:]

    return f"[Conversation history]\n{block}\n---"
