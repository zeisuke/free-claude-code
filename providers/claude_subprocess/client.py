"""Claude subprocess provider.

Spawns 'claude -p --output-format stream-json' for real Claude responses using
Claude Code subscription auth. The agent loop (tool calls, WebSearch, Read, Bash)
runs inside the subprocess; FCC waits for the final 'result' event and returns
it as a single Anthropic SSE text block.

Falls back to Ollama on any subprocess failure.
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
    "haiku": "llama3.1:8b",
    "sonnet": "llama3.1:8b",
    "opus": "llama3.1:8b",
}

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_SUBPROCESS_TIMEOUT = 120  # seconds


async def _is_fcc_active() -> bool:
    """Check if FCC fallback mode is active via search_api /fcc-status.

    Returns False on any error (fail open — assume inactive, try claude -p).
    """
    try:
        import urllib.request as _ur
        import json as _j
        loop = asyncio.get_event_loop()

        def _check():
            with _ur.urlopen("http://localhost:8765/fcc-status", timeout=1.0) as _r:
                return _j.loads(_r.read()).get("fcc_active", False)

        return await loop.run_in_executor(None, _check)
    except Exception:
        return False


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
        if await _is_fcc_active():
            logger.info("claude_subprocess: FCC active — routing to Ollama")
            async for chunk in self._ollama_stream(request, input_tokens, request_id=request_id):
                yield chunk
            return
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
        prompt, image_paths = _build_conversation_text(request.messages)
        claude_model = _resolve_claude_model(request.model)

        final_system = system_text or ""
        if image_paths:
            final_system = (
                final_system
                + "\n\nIMAGE INSTRUCTION: image files are available in the added directories. "
                "Use the Read tool to load and view them before responding."
            ).strip()

        # --output-format json: claude -p runs the full agent loop (WebSearch, Read, Bash)
        # internally and returns a single JSON result when done. No XML injection needed;
        # --allowed-tools passes native Claude Code tools directly.
        #
        # cwd must be a neutral directory with no .claude/settings.json to avoid loading
        # SessionStart hooks (which connect to Hermes KB and MCP servers, adding 20-30s).
        _fcc_cwd = "/tmp/hermes_fcc"
        os.makedirs(_fcc_cwd, exist_ok=True)

        cmd = [
            "claude", "-p",
            "--model", claude_model,
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--output-format", "json",
            "--max-turns", "5",
            "--allowed-tools", "WebSearch,WebFetch,Read",
        ]
        if final_system:
            cmd += ["--system-prompt", final_system]
        seen_dirs: set[str] = set()
        for img_path in image_paths:
            img_dir = os.path.dirname(img_path)
            if img_dir and img_dir not in seen_dirs:
                cmd += ["--add-dir", img_dir]
                seen_dirs.add(img_dir)
        cmd.append(prompt)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=_clean_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=PIPE,
            stderr=PIPE,
            cwd=_fcc_cwd,
        )

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield (
            "event: message_start\ndata: "
            + json.dumps({
                "type": "message_start",
                "message": {
                    "id": msg_id, "type": "message", "role": "assistant",
                    "content": [], "model": request.model,
                    "stop_reason": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            })
            + "\n\n"
        )

        # claude -p --output-format json blocks until the full agent loop completes,
        # then writes a single JSON object. Yield keepalives every 5s while waiting.
        communicate_task = asyncio.ensure_future(proc.communicate())
        elapsed = 0
        while not communicate_task.done():
            done, _ = await asyncio.wait({communicate_task}, timeout=5.0)
            if not done:
                yield ": keepalive\n\n"
                elapsed += 5
                if elapsed >= _SUBPROCESS_TIMEOUT:
                    communicate_task.cancel()
                    raise RuntimeError(f"claude subprocess timeout after {_SUBPROCESS_TIMEOUT}s")

        stdout_bytes, stderr_bytes = communicate_task.result()
        if proc.returncode != 0:
            raise RuntimeError(f"claude exit {proc.returncode}: {stderr_bytes.decode()[:300]}")

        try:
            result_obj = json.loads(stdout_bytes.decode())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"claude json parse error: {exc}") from exc

        # Extract final text from result object
        result_text = result_obj.get("result", "")
        if not result_text and result_obj.get("is_error"):
            raise RuntimeError(f"claude error result: {result_obj.get('subtype','unknown')}")
        if not result_text:
            raise RuntimeError("claude subprocess: empty result")

        yield (
            "event: content_block_start\ndata: "
            + json.dumps({"type": "content_block_start", "index": 0,
                          "content_block": {"type": "text", "text": ""}})
            + "\n\n"
        )
        yield (
            "event: content_block_delta\ndata: "
            + json.dumps({"type": "content_block_delta", "index": 0,
                          "delta": {"type": "text_delta", "text": result_text}})
            + "\n\n"
        )
        yield (
            "event: content_block_stop\ndata: "
            + json.dumps({"type": "content_block_stop", "index": 0})
            + "\n\n"
        )
        yield (
            "event: message_delta\ndata: "
            + json.dumps({"type": "message_delta",
                          "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                          "usage": {"output_tokens": 0}})
            + "\n\n"
        )
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    async def _ollama_stream(
        self, request: Any, input_tokens: int, *, request_id: str | None = None
    ) -> AsyncIterator[str]:
        from providers.ollama import OllamaProvider
        import json as _json_o, uuid as _uuid_o

        ollama_model = _resolve_ollama_model(request.model)

        # Check for image blocks — route to moondream via Ollama /api/chat (supports vision).
        # Standard OllamaProvider uses /v1/messages which does NOT support image blocks.
        def _extract_images_and_text(msgs):
            """Return (ollama_chat_messages, has_images) for Ollama /api/chat format."""
            out, has_imgs = [], False
            for msg in (msgs or []):
                role = str(getattr(msg, "role", "user"))
                content = msg.content
                texts, imgs = [], []
                if isinstance(content, list):
                    for block in content:
                        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                        if btype == "text":
                            texts.append(getattr(block, "text", "") or block.get("text", ""))
                        elif btype == "image":
                            source = getattr(block, "source", None) or (block.get("source") if isinstance(block, dict) else None)
                            data = getattr(source, "data", None) or (source.get("data") if isinstance(source, dict) else None)
                            if data:
                                imgs.append(data)
                        elif btype == "image_url":
                            iu = getattr(block, "image_url", None) or (block.get("image_url") if isinstance(block, dict) else None)
                            url = getattr(iu, "url", "") or (iu.get("url", "") if isinstance(iu, dict) else "")
                            if url.startswith("data:"):
                                _, _, d = url.partition(",")
                                imgs.append(d)
                elif isinstance(content, str):
                    texts.append(content)
                entry: dict = {"role": role, "content": " ".join(texts)}
                if imgs:
                    entry["images"] = imgs
                    has_imgs = True
                out.append(entry)
            return out, has_imgs

        chat_messages, has_images = _extract_images_and_text(request.messages)

        if has_images:
            # Use moondream via Ollama /api/chat for vision (supports image blocks natively)
            import urllib.request as _ur_v, json as _j_v, uuid as _uid_v
            system_text = _extract_system_text(request.system)
            if system_text:
                chat_messages = [{"role": "system", "content": system_text}] + chat_messages
            body = _j_v.dumps({"model": "moondream:latest", "messages": chat_messages, "stream": False}).encode()
            req_v = _ur_v.Request("http://localhost:11434/api/chat", data=body,
                                  headers={"Content-Type": "application/json"}, method="POST")
            try:
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, lambda: _j_v.loads(_ur_v.urlopen(req_v, timeout=60).read()))
                text = resp.get("message", {}).get("content", "")
                logger.info("claude_subprocess: moondream vision response len=%d", len(text))
            except Exception as _ve:
                logger.warning("moondream vision failed: %s", _ve)
                text = ""
            mid = f"msg_{_uid_v.uuid4().hex[:24]}"
            yield f"event: message_start\ndata: " + _j_v.dumps({"type":"message_start","message":{"id":mid,"type":"message","role":"assistant","content":[],"model":request.model,"stop_reason":None,"usage":{"input_tokens":input_tokens,"output_tokens":0}}}) + "\n\n"
            if text:
                yield "event: content_block_start\ndata: " + _j_v.dumps({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}) + "\n\n"
                yield "event: content_block_delta\ndata: " + _j_v.dumps({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}}) + "\n\n"
                yield "event: content_block_stop\ndata: " + _j_v.dumps({"type":"content_block_stop","index":0}) + "\n\n"
            yield "event: message_delta\ndata: " + _j_v.dumps({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":None},"usage":{"output_tokens":0}}) + "\n\n"
            yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            return

        # Strip tools from Ollama request — Ollama's /v1/messages doesn't support hermes tool schemas.
        ollama_request = request.model_copy(update={"model": ollama_model, "tools": None})
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


def _extract_image_path(block: Any) -> str | None:
    """Extract or save an image block to disk; return local file path or None.

    Handles both Anthropic format (type=image, source=...) and
    OpenAI format (type=image_url, image_url.url=data:... or file://...).
    """
    import base64 as _b64, hashlib as _hs

    block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)

    # OpenAI-style image_url block: {"type": "image_url", "image_url": {"url": "data:..."}}
    if block_type == "image_url":
        img_url_obj = getattr(block, "image_url", None) or (block.get("image_url") if isinstance(block, dict) else None)
        url = getattr(img_url_obj, "url", None) or (img_url_obj.get("url") if isinstance(img_url_obj, dict) else None) or ""
        if url.startswith("file://"):
            return url[7:]
        if url.startswith("data:"):
            try:
                header, data = url.split(",", 1)
                media_type = header.split(";")[0].split(":")[1] if ":" in header else "image/jpeg"
                ext = media_type.split("/")[-1] if "/" in media_type else "jpg"
                tmp_dir = "/tmp/fcc_images"
                os.makedirs(tmp_dir, exist_ok=True)
                fname = f"img_{_hs.md5(data[:100].encode()).hexdigest()[:12]}.{ext}"
                fpath = os.path.join(tmp_dir, fname)
                if not os.path.exists(fpath):
                    with open(fpath, "wb") as _f:
                        _f.write(_b64.b64decode(data))
                return fpath
            except Exception:
                return None
        return None

    # Anthropic-style image block: {"type": "image", "source": {"type": "base64", ...}}
    source = getattr(block, "source", None) or (block.get("source") if isinstance(block, dict) else None)
    if not source:
        return None
    src_type = getattr(source, "type", None) or (source.get("type") if isinstance(source, dict) else None)
    if src_type == "url":
        url = getattr(source, "url", "") or (source.get("url", "") if isinstance(source, dict) else "")
        if url.startswith("file://"):
            return url[7:]
    elif src_type == "base64":
        data = getattr(source, "data", "") or (source.get("data", "") if isinstance(source, dict) else "")
        media_type = getattr(source, "media_type", "image/jpeg") or (source.get("media_type", "image/jpeg") if isinstance(source, dict) else "image/jpeg")
        ext = media_type.split("/")[-1] if "/" in media_type else "jpg"
        tmp_dir = "/tmp/fcc_images"
        os.makedirs(tmp_dir, exist_ok=True)
        fname = f"img_{_hs.md5(data[:100].encode()).hexdigest()[:12]}.{ext}"
        fpath = os.path.join(tmp_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "wb") as _f:
                _f.write(_b64.b64decode(data))
        return fpath
    return None


def _build_conversation_text(messages: list[Any]) -> tuple[str, list[str]]:
    """Returns (conversation_text, image_paths) where image_paths are local files to add.

    Extracts images from:
    1. API-level image blocks (Anthropic format or OpenAI image_url format)
    2. [image: /path] text references in message content
    """
    import re as _re
    _IMAGE_REF_RE = _re.compile(r'\[image:\s*(/[^\]]+)\]')

    parts = []
    image_paths: list[str] = []
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
                    elif block.type == "image":
                        img_path = _extract_image_path(block)
                        if img_path:
                            texts.append(f"[image: {img_path}]")
                            image_paths.append(img_path)
                    elif block.type == "tool_result":
                        rc = block.content
                        texts.append(f"[Tool result: {rc if isinstance(rc, str) else json.dumps(rc)}]")
                    elif block.type == "tool_use":
                        texts.append(f"[Called {block.name}: {json.dumps(block.input)}]")
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "image":
                        img_path = _extract_image_path(block)
                        if img_path:
                            texts.append(f"[image: {img_path}]")
                            image_paths.append(img_path)
            text = "\n".join(texts)
        else:
            text = str(content)
        # Extract [image: /path] text references
        for _m in _IMAGE_REF_RE.finditer(text):
            _img_path = _m.group(1).strip()
            if os.path.isfile(_img_path) and _img_path not in image_paths:
                image_paths.append(_img_path)
        parts.append(f"{role}: {text}")
    return "\n\n".join(parts), image_paths


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
