"""Claude subprocess provider.

Spawns 'claude -p' for real Claude responses (Claude Code subscription auth).
Falls back to Ollama on any subprocess failure.

Tool definitions are injected into the system prompt as XML+JSON; tool calls are
parsed from the response via <tool_call>{...}</tool_call> tags and converted to
proper Anthropic tool_use SSE blocks so hermes's AIAgent can execute them.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from asyncio.subprocess import PIPE
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from providers.base import BaseProvider, ProviderConfig
from providers.model_listing import model_infos_from_ids

_CLAUDE_MODEL_MAP: dict[str, str] = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}

_OLLAMA_MODEL_MAP: dict[str, str] = {
    "haiku": "qwen2.5:0.5b",
    "sonnet": "gemma4:latest",
    "opus": "gemma4:latest",
}

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_SUBPROCESS_TIMEOUT = 120  # seconds


class ClaudeSubprocessProvider(BaseProvider):
    """Spawns 'claude -p' for Claude-quality responses; Ollama fallback on failure."""

    async def cleanup(self) -> None:
        pass

    async def list_model_ids(self) -> frozenset[str]:
        return frozenset(_CLAUDE_MODEL_MAP.values())

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        try:
            async for chunk in self._subprocess_stream(request, input_tokens):
                yield chunk
        except Exception as exc:
            logger.warning(
                f"claude_subprocess failed ({type(exc).__name__}: {str(exc)[:120]}), falling back to Ollama"
            )
            async for chunk in self._ollama_stream(request, input_tokens, request_id=request_id):
                yield chunk

    async def _subprocess_stream(self, request: Any, input_tokens: int) -> AsyncIterator[str]:
        system_text = _extract_system_text(request.system)
        if request.tools:
            tools_json = json.dumps(
                [t.model_dump(exclude_none=True) for t in request.tools], indent=2
            )
            system_text = (system_text or "") + (
                "\n\n<tools>\n"
                + tools_json
                + "\n</tools>\n"
                "When calling a tool, emit exactly:\n"
                "<tool_call>{\"name\":\"TOOL_NAME\",\"input\":{...}}</tool_call>\n"
                "You may emit multiple tool calls. Add explanatory text outside the tags."
            )

        prompt = _build_conversation_text(request.messages)
        claude_model = _resolve_claude_model(request.model)

        cmd = [
            "claude", "-p",
            "--model", claude_model,
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--max-turns", "10",
        ]
        if system_text:
            cmd += ["--system-prompt", system_text]
        cmd.append(prompt)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=_clean_env(),
            stdin=asyncio.subprocess.DEVNULL,  # prevent 3s stdin wait in daemon
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_SUBPROCESS_TIMEOUT)

        if proc.returncode != 0:
            raise RuntimeError(f"claude exit {proc.returncode}: {stderr.decode()[:300]}")

        text = stdout.decode()
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        parts = _parse_content(text)
        has_tool_use = any(p["type"] == "tool_use" for p in parts)

        yield (
            f"event: message_start\ndata: "
            + json.dumps({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": request.model,
                    "stop_reason": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            })
            + "\n\n"
        )

        for idx, part in enumerate(parts):
            if part["type"] == "text" and part["text"]:
                yield f"event: content_block_start\ndata: " + json.dumps({"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}}) + "\n\n"
                yield f"event: content_block_delta\ndata: " + json.dumps({"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": part["text"]}}) + "\n\n"
                yield f"event: content_block_stop\ndata: " + json.dumps({"type": "content_block_stop", "index": idx}) + "\n\n"
            elif part["type"] == "tool_use":
                tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                yield f"event: content_block_start\ndata: " + json.dumps({"type": "content_block_start", "index": idx, "content_block": {"type": "tool_use", "id": tool_id, "name": part["name"], "input": {}}}) + "\n\n"
                yield f"event: content_block_delta\ndata: " + json.dumps({"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": json.dumps(part["input"])}}) + "\n\n"
                yield f"event: content_block_stop\ndata: " + json.dumps({"type": "content_block_stop", "index": idx}) + "\n\n"

        stop_reason = "tool_use" if has_tool_use else "end_turn"
        yield f"event: message_delta\ndata: " + json.dumps({"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": 0}}) + "\n\n"
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    async def _ollama_stream(
        self, request: Any, input_tokens: int, *, request_id: str | None = None
    ) -> AsyncIterator[str]:
        from providers.ollama import OllamaProvider

        ollama_model = _resolve_ollama_model(request.model)
        ollama_request = request.model_copy(update={"model": ollama_model})
        config = ProviderConfig(api_key="ollama", base_url="http://localhost:11434", max_concurrency=5)
        provider = OllamaProvider(config)
        try:
            async for chunk in provider.stream_response(ollama_request, input_tokens, request_id=request_id):
                yield chunk
        finally:
            await provider.cleanup()


def _extract_system_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts = []
    for block in system:
        text = block.text if hasattr(block, "text") else (block.get("text", "") if isinstance(block, dict) else "")
        if text:
            parts.append(text)
    return "\n".join(parts)


def _build_conversation_text(messages: list[Any]) -> str:
    parts = []
    for msg in messages:
        role = str(getattr(msg, "role", "user")).capitalize()
        content = msg.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        texts.append(block.text)
                    elif block.type == "tool_result":
                        rc = block.content
                        texts.append(f"[Tool result: {rc if isinstance(rc, str) else json.dumps(rc)}]")
                    elif block.type == "tool_use":
                        texts.append(f"[Called {block.name}: {json.dumps(block.input)}]")
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
            text = "\n".join(texts)
        else:
            text = str(content)
        parts.append(f"{role}: {text}")
    return "\n\n".join(parts)


def _parse_content(text: str) -> list[dict[str, Any]]:
    """Split response text into text blocks and tool_use blocks."""
    parts: list[dict[str, Any]] = []
    last_end = 0
    for m in _TOOL_CALL_RE.finditer(text):
        if m.start() > last_end:
            chunk = text[last_end : m.start()].strip()
            if chunk:
                parts.append({"type": "text", "text": chunk})
        try:
            data = json.loads(m.group(1).strip())
            parts.append({"type": "tool_use", "name": data.get("name", ""), "input": data.get("input", {})})
        except json.JSONDecodeError:
            parts.append({"type": "text", "text": m.group(0)})
        last_end = m.end()
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            parts.append({"type": "text", "text": remaining})
    return parts


def _resolve_claude_model(model_name: str) -> str:
    name_lower = model_name.lower()
    for key, claude_model in _CLAUDE_MODEL_MAP.items():
        if key in name_lower:
            return claude_model
    return "claude-haiku-4-5"


def _resolve_ollama_model(model_name: str) -> str:
    name_lower = model_name.lower()
    for key, ollama_model in _OLLAMA_MODEL_MAP.items():
        if key in name_lower:
            return ollama_model
    return "qwen2.5:0.5b"


def _clean_env() -> dict[str, str]:
    """Return clean env: strip FCC mock credentials so subprocess uses real Claude auth."""
    env = dict(os.environ)
    for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CONFIG_DIR"):
        env.pop(key, None)
    return env
