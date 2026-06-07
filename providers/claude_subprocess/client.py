"""Claude subprocess provider.

Spawns 'claude -p --output-format stream-json' for real Claude responses using
Claude Code subscription auth. The agent loop (tool calls, WebSearch, Read, Bash)
runs inside the subprocess; FCC waits for the final 'result' event and returns
it as a single Anthropic SSE text block.

Falls back to Ollama only when FCC is active (all models in cooldown) or credit limit exceeded.
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

_OLLAMA_MODEL_DEFAULT = "qwen3.5:9b"
_ollama_model_cache: tuple[float, str] | None = None  # (expires_at, model)
_OLLAMA_MODEL_CACHE_TTL = 30.0  # seconds


def _fetch_ollama_model() -> str:
    """Read current Ollama text model from search_api /fcc-status (30s cache)."""
    global _ollama_model_cache
    import time as _time_m
    now = _time_m.monotonic()
    if _ollama_model_cache and now < _ollama_model_cache[0]:
        return _ollama_model_cache[1]
    try:
        import urllib.request as _ur_m, json as _j_m
        with _ur_m.urlopen("http://localhost:8765/fcc-status", timeout=2) as r:
            model = _j_m.loads(r.read()).get("ollama_model") or _OLLAMA_MODEL_DEFAULT
    except Exception:
        model = (_ollama_model_cache[1] if _ollama_model_cache else _OLLAMA_MODEL_DEFAULT)
    _ollama_model_cache = (now + _OLLAMA_MODEL_CACHE_TTL, model)
    return model

# Matches <tool_call>...</tool_call> and malformed variants like /tool_call>{...}
_TOOL_CALL_RE = re.compile(
    r"(?:<|/)tool_call>?\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL
)

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
        _fcc_now = await _is_fcc_active()
        if _fcc_now:
            logger.info("claude_subprocess: FCC active — routing to Ollama")
            async for chunk in self._ollama_stream(request, input_tokens, request_id=request_id):
                yield chunk
            return
        async for chunk in self._subprocess_stream(request, input_tokens):
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
            "--max-turns", "15",
            "--allowed-tools", "Read,Edit,Write,Glob,Grep,WebSearch,WebFetch,Bash,Agent,mcp__hermes-kb__*",
        ]
        if final_system:
            cmd += ["--system-prompt", final_system]
        seen_dirs: set[str] = set()
        for img_path in image_paths:
            img_dir = os.path.dirname(img_path)
            if img_dir and img_dir not in seen_dirs:
                cmd += ["--add-dir", img_dir]
                seen_dirs.add(img_dir)
        logger.debug(
            f"subprocess: model={claude_model!r} prompt_len={len(prompt)} "
            f"prompt_start={prompt[:120]!r}"
        )
        # Prompt is sent via stdin; claude -p ignores the CLI argument when stdin is
        # /dev/null (v2.1.163 bug: checks stdin first, errors if empty regardless of arg).
        prompt_bytes = prompt.encode()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=_clean_env(),
            stdin=asyncio.subprocess.PIPE,
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
        communicate_task = asyncio.ensure_future(proc.communicate(prompt_bytes))
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

        # Try stdout first — claude exits 1 on some non-fatal conditions (e.g. error_max_turns)
        # but still writes JSON to stdout. Only fall back to stderr error if stdout is unusable.
        try:
            result_obj = json.loads(stdout_bytes.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            result_obj = {}

        # Extract final text from result object
        result_text = result_obj.get("result", "")

        if proc.returncode != 0 and not result_text:
            subtype = result_obj.get("subtype", "")
            if subtype == "error_max_turns":
                # Return a graceful message rather than letting hermes-agent see empty content
                result_text = "[任务超过最大步骤数，请简化请求后重试。]"
            else:
                raise RuntimeError(f"claude exit {proc.returncode}: {stderr_bytes.decode()[:300]}")

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

        # qwen3.5:9b supports native vision + tools — no moondream routing needed.
        import urllib.request as _ur_t, json as _j_t, uuid as _uid_t

        # 0. Compress claude -p built-in system prompt (38K+ tokens → ≤12K chars ~3K tokens).
        #    Compress, NOT replace — keep tool descriptions and core behavior rules.
        #    Strategy: keep first _SYS_KEEP_CHARS chars (contains role + tools) + Jarvis addendum.
        _CLAUDE_CODE_MARKERS = (
            "Claude Code", "claude code", "CLAUDE CODE",
            "agentic coding", "software engineering tasks",
            "tool_use_id", "bash_20250124", "computer_use",
        )
        _SYS_KEEP_CHARS = 6000   # ~1.5K tokens: role + key tools, leaves room for messages + KV cache
        raw_system = _extract_system_text(request.system)
        _is_claude_code_prompt = any(m in raw_system for m in _CLAUDE_CODE_MARKERS)
        # Defined here so both if/else branches can use it
        _STDOUT_RULE = (
            "\nBash commands MUST print results to stdout. "
            "Never redirect output to files (no >, >>, tee). "
            "Keep commands simple — do not add extra error handling or logging."
        )

        def _soul_head() -> str:
            """Load first 600 chars of SOUL.md from disk (preferred) or fallback."""
            import pathlib as _pl2
            _sp = _pl2.Path.home() / ".hermes" / "SOUL.md"
            if _sp.exists():
                txt = _sp.read_text()
                pos = txt.find("# Jarvis")
                if pos >= 0:
                    return txt[pos:pos + 600].strip()
            return "You are Jarvis. You HAVE tools: Bash, Read, Write, WebSearch, WebFetch. Execute tasks directly. Call Bash when asked to run a command."

        if _is_claude_code_prompt:
            # Use ONLY the SOUL.md head — do NOT include Claude Code tool descriptions.
            # The first fcc-claude call may not have SOUL.md appended yet; always load from disk.
            _soul_start = raw_system.find("# Jarvis")
            if _soul_start >= 0:
                _effective_system = raw_system[_soul_start:_soul_start + 600].strip() + _STDOUT_RULE
            else:
                # First call: SOUL.md not in raw_system yet — load from disk
                _effective_system = _soul_head() + _STDOUT_RULE
            logger.info(
                "claude_subprocess: using SOUL.md head %d→%d chars",
                len(raw_system), len(_effective_system),
            )
        else:
            # Non claude-code prompt: extract from raw_system or load from disk
            _soul_start2 = raw_system.find("# Jarvis") if raw_system else -1
            if _soul_start2 >= 0:
                _effective_system = raw_system[_soul_start2:_soul_start2 + 600].strip() + _STDOUT_RULE
            else:
                _effective_system = _soul_head() + _STDOUT_RULE

        # 0.1 Compress conversation history if too large for local model context.
        #     - Keep recent KEEP_TURNS messages in full (recent tool results matter most)
        #     - Compress older messages → readable 1-line summaries
        #     - Archive full content → Hindsight bank for on-demand retrieval
        #     Reduces hallucination vs silent truncation: model knows what was compressed.
        _CTX_TOKEN_LIMIT = 20000   # target max tokens for messages (~80K chars)
        _KEEP_TURNS = 6            # keep last 6 messages (≈3 tool call rounds) in full
        _SUMMARY_CHARS = 200       # max chars per compressed message summary

        def _get_field(m, key, default=""):
            """Get a field from either a pydantic object or a dict."""
            if isinstance(m, dict):
                return m.get(key, default)
            return getattr(m, key, default)

        def _estimate_tokens(msgs):
            return sum(len(str(_get_field(m, "content", ""))) for m in msgs) // 4

        def _msg_summary(msg):
            role = _get_field(msg, "role", "?")
            content = str(_get_field(msg, "content", ""))
            preview = content.replace("\n", " ")[:_SUMMARY_CHARS]
            if len(content) > _SUMMARY_CHARS:
                preview += "…"
            return f"[{role}]: {preview}"

        _msgs = list(request.messages or [])
        if _estimate_tokens(_msgs) > _CTX_TOKEN_LIMIT and len(_msgs) > _KEEP_TURNS:
            _to_archive = _msgs[:-_KEEP_TURNS]
            _to_keep   = _msgs[-_KEEP_TURNS:]

            # Build readable summary of archived messages
            _archive_lines = [_msg_summary(m) for m in _to_archive]
            _archive_text  = "\n".join(_archive_lines)

            # Archive full content to KB (search_api /kb/store) for reliable retrieval.
            # Uses Hermes-KB-Token from keychain — same path used by all KB writes.
            _retrieval_hint = "Earlier context compressed. Use search_kb or hindsight_recall to retrieve if needed."
            try:
                import subprocess as _sp
                # Run blocking calls in executor to avoid blocking asyncio event loop
                def _archive_to_kb():
                    _kb_token = _sp.check_output(
                        ["security", "find-generic-password", "-s", "Hermes-KB-Token", "-w"],
                        stderr=_sp.DEVNULL,
                    ).decode().strip()
                    _kb_body = _j_t.dumps({
                        "content": f"[FCC session context archive — request_id: {request_id}]\n{_archive_text}",
                        "content_type": "fcc_session",
                        "metadata": {
                            "source": "fcc_compress",
                            "request_id": request_id or "unknown",
                            "archived_turns": len(_to_archive),
                        },
                    }).encode()
                    _kb_req = _ur_t.Request(
                        "http://localhost:8765/kb/store",
                        data=_kb_body,
                        headers={"Content-Type": "application/json",
                                 "Authorization": f"Bearer {_kb_token}"},
                    )
                    return _j_t.loads(_ur_t.urlopen(_kb_req, timeout=5).read())
                _kb_resp = await asyncio.get_event_loop().run_in_executor(None, _archive_to_kb)
                _kb_id = _kb_resp.get("id", "?")
                _retrieval_hint = (
                    f"Earlier context archived to KB (id={_kb_id}). "
                    "Use search_kb or hindsight_recall to retrieve if needed."
                )
                logger.info("claude_subprocess: archived %d msgs to KB id=%s", len(_to_archive), _kb_id)
            except Exception as _ae:
                logger.warning("claude_subprocess: KB archive failed: %s", _ae)

            # Replace old messages with compressed summary + retrieval hint
            _summary_msg = {
                "role": "user",
                "content": (
                    f"[Earlier conversation compressed — {len(_to_archive)} messages archived]\n"
                    f"{_archive_text}\n\n"
                    f"{_retrieval_hint}"
                ),
            }
            _msgs = [_summary_msg] + _to_keep
            logger.info(
                "claude_subprocess: compressed %d→%d messages (est %d tokens saved)",
                len(_to_archive) + _KEEP_TURNS,
                len(_msgs),
                _estimate_tokens(_to_archive),
            )

        # 1. Convert tools: Anthropic → OpenAI function calling format (qwen3.5 native)
        #    Use minimal schemas — the Claude Code full schemas are 10K+ chars each,
        #    consuming the entire context window and preventing qwen3.5 from thinking.
        _CORE_TOOLS = {"Bash", "Read", "Write", "WebSearch", "WebFetch"}
        _MINIMAL_SCHEMAS = {
            "Bash":      {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command to run"}}, "required": ["command"]},
            "Read":      {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
            "Write":     {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]},
            "WebSearch": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            "WebFetch":  {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        }
        _MINIMAL_DESCS = {
            "Bash":      "Execute shell commands. Output goes to stdout.",
            "Read":      "Read file contents.",
            "Write":     "Write content to a file.",
            "WebSearch": "Search the web.",
            "WebFetch":  "Fetch a URL.",
        }
        ollama_tools = None
        if request.tools:
            ollama_tools = []
            for t in request.tools:
                name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else "")
                if name not in _CORE_TOOLS:
                    continue
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": _MINIMAL_DESCS.get(name, name),
                        "parameters": _MINIMAL_SCHEMAS.get(name, {"type": "object", "properties": {}}),
                    },
                })
            if not ollama_tools:
                ollama_tools = None

        # 2. Build /api/chat messages with native vision + tool support.
        #    Anthropic format → Ollama /api/chat format:
        #      image blocks      → "images" list (base64)
        #      tool_use blocks   → assistant message with tool_calls
        #      tool_result block → role=tool message
        def _convert_block_list(role, blocks):
            """Convert Anthropic content block list to ≥1 Ollama messages."""
            texts, images, tool_calls_blk, tool_results = [], [], [], []
            for block in blocks:
                btype = (getattr(block, "type", None) or
                         (block.get("type") if isinstance(block, dict) else None))
                if btype == "text":
                    _t = getattr(block, "text", "") or block.get("text", "")
                    if role == "user":
                        _t = _strip_reminders(_t)
                    if _t:
                        texts.append(_t)
                elif btype in ("image", "image_url"):
                    # Extract base64 data
                    if btype == "image":
                        src = getattr(block, "source", None) or block.get("source", {})
                        data = getattr(src, "data", None) or (src.get("data") if isinstance(src, dict) else None)
                    else:
                        iu = getattr(block, "image_url", None) or block.get("image_url", {})
                        url = getattr(iu, "url", "") or (iu.get("url", "") if isinstance(iu, dict) else "")
                        _, _, data = url.partition(",") if url.startswith("data:") else ("", "", "")
                    if data:
                        images.append(data)
                elif btype == "tool_use":
                    tid = getattr(block, "id", None) or block.get("id", f"call_{_uid_t.uuid4().hex[:8]}")
                    name = getattr(block, "name", "") or block.get("name", "")
                    inp = getattr(block, "input", {}) or block.get("input", {})
                    # Ollama /api/chat expects arguments as dict (not JSON string like OpenAI)
                    if isinstance(inp, str):
                        try:
                            inp = _j_t.loads(inp)
                        except Exception:
                            inp = {"_raw": inp}
                    tool_calls_blk.append({
                        "id": tid, "type": "function",
                        "function": {"name": name, "arguments": inp},
                    })
                elif btype == "tool_result":
                    tid = getattr(block, "tool_use_id", None) or block.get("tool_use_id", "")
                    rc = getattr(block, "content", "") or block.get("content", "")
                    if isinstance(rc, list):
                        rc = " ".join(
                            (getattr(p, "text", "") or p.get("text", ""))
                            for p in rc if hasattr(p, "text") or isinstance(p, dict)
                        )
                    tool_results.append({"role": "tool", "tool_call_id": tid, "content": str(rc)})

            out = []
            if tool_results:
                out.extend(tool_results)
            elif tool_calls_blk:
                msg: dict = {"role": "assistant", "tool_calls": tool_calls_blk}
                if texts:
                    msg["content"] = " ".join(texts)
                out.append(msg)
            else:
                entry: dict = {"role": role, "content": " ".join(texts)}
                if images:
                    entry["images"] = images
                out.append(entry)
            return out

        import re as _re_sr
        _SYSTEM_REMINDER_RE = _re_sr.compile(
            r"<system-reminder>.*?</system-reminder>", _re_sr.DOTALL
        )
        # Hindsight injects conversation summaries like "对话摘要：\n用户：...\nJarvis：FAKE_RESULT"
        # These contain hallucinated results from previous sessions and create feedback loops.
        _CONV_SUMMARY_RE = _re_sr.compile(
            r"对话摘要[：:].+?(?=\n\n|\Z)", _re_sr.DOTALL
        )
        # Claude Code adds "LANGUAGE INSTRUCTION: ..." prefix to user messages.
        _LANG_INSTR_RE = _re_sr.compile(
            r"^LANGUAGE INSTRUCTION:[^\n]*\n*", _re_sr.MULTILINE
        )
        # Hindsight injects evaluation blocks like "判断：\n1. 用户是否明确..."
        # These cause qwen3.5 to write to USER.md instead of executing the task.
        _JUDGE_BLOCK_RE = _re_sr.compile(
            r"判断[：:]\s*\n.*?(?=\n\n|\Z)", _re_sr.DOTALL
        )

        # Track multi-step bash commands so we can inject them sequentially
        _pending_bash_steps: list[str] = []

        def _split_bash_steps(text: str) -> tuple[str, list[str]]:
            """If text is 'run bash: cmd1; cmd2', return ('run bash: cmd1', ['cmd2']).
            Splits on ; and && so qwen3.5 executes one step at a time."""
            import re as _re_bash
            m = _re_bash.search(r'run bash:\s*(.+)', text, _re_bash.IGNORECASE)
            if not m:
                return text, []
            full_cmd = m.group(1).strip()
            # Split on ; or && preserving the rest
            parts = _re_bash.split(r'\s*(?:;|&&)\s*', full_cmd)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) <= 1:
                return text, []
            first_step = text[:m.start(1)] + parts[0]
            remaining = parts[1:]
            return first_step, remaining

        def _strip_reminders(text: str) -> str:
            """Remove injected hook content from user messages before sending to Ollama."""
            result = _SYSTEM_REMINDER_RE.sub("", text)
            result = _CONV_SUMMARY_RE.sub("", result)
            result = _LANG_INSTR_RE.sub("", result)
            result = _JUDGE_BLOCK_RE.sub("", result)
            return result.strip()

        ollama_messages = []
        _first_user_seen = False
        if _effective_system:
            ollama_messages.append({"role": "system", "content": _effective_system})
        for msg in _msgs:
            role = str(getattr(msg, "role", "user"))
            content = msg.content
            if isinstance(content, list):
                converted = _convert_block_list(role, content)
                # Apply bash split to first NON-EMPTY user message from list branch.
                # Only set _first_user_seen if we actually found non-empty content;
                # empty converted messages (all blocks stripped) must NOT consume the flag.
                if role == "user" and not _first_user_seen:
                    for i, cm in enumerate(converted):
                        if cm.get("role") == "user" and cm.get("content"):
                            _first_user_seen = True
                            cm["content"], steps = _split_bash_steps(cm["content"])
                            _pending_bash_steps.extend(steps)
                            break
                ollama_messages.extend(converted)
            elif isinstance(content, str):
                cleaned = _strip_reminders(content) if role == "user" else content
                if role == "user" and not _first_user_seen and cleaned:
                    _first_user_seen = True
                    cleaned, _pending_bash_steps[:] = _split_bash_steps(cleaned)
                if cleaned:
                    ollama_messages.append({"role": role, "content": cleaned})
            else:
                cleaned = _strip_reminders(str(content)) if role == "user" else str(content)
                if cleaned:
                    ollama_messages.append({"role": role, "content": cleaned})

        # Global compression: if total ollama_messages tokens > threshold, compress.
        # Threshold chosen to leave headroom for model response + KV cache overhead.
        _OLLAMA_TOKEN_LIMIT = 16000  # ~16K tokens safe for 24K context window
        _total_ollama_tokens = sum(len(str(m.get("content", ""))) for m in ollama_messages) // 4
        if _total_ollama_tokens > _OLLAMA_TOKEN_LIMIT:
            # Keep system + last 4 messages; summarise the rest
            _sys_msgs = [m for m in ollama_messages if m.get("role") == "system"]
            _non_sys  = [m for m in ollama_messages if m.get("role") != "system"]
            _keep = _non_sys[-4:] if len(_non_sys) > 4 else _non_sys
            _dropped = _non_sys[:-4] if len(_non_sys) > 4 else []
            if _dropped:
                _summary = " | ".join(
                    f"[{m['role']}]: {str(m.get('content',''))[:100]}…" for m in _dropped[:6]
                )
                _trunc_msg = {"role": "user",
                              "content": f"[Earlier context compressed: {_summary}]"}
                ollama_messages = _sys_msgs + [_trunc_msg] + _keep
            else:
                ollama_messages = _sys_msgs + _keep
            _new_tokens = sum(len(str(m.get("content",""))) for m in ollama_messages) // 4
            logger.info("ollama compress: %d→%d msgs, ~%d→%d tokens",
                        len(_non_sys) + len(_sys_msgs), len(ollama_messages),
                        _total_ollama_tokens, _new_tokens)

        # Track last tool result for post-processing (see below after _stream_collect)
        _last_tool_result = None
        for _m in reversed(ollama_messages):
            if _m.get("role") == "tool":
                _last_tool_result = str(_m.get("content", ""))
                break

        # If the last tool result contains an error, inject a "continue" user message.
        # qwen3.5 needs explicit guidance to proceed after tool failure (unlike Claude
        # which infers this from context). Without this, qwen3.5 stops after the error.
        _ERROR_MARKERS = ("no such file", "not found", "error", "failed", "exit code: 1",
                          "permission denied", "command not found", "enoent", "exception")
        if _last_tool_result and any(m in _last_tool_result.lower() for m in _ERROR_MARKERS):
            if _pending_bash_steps:
                next_cmd = _pending_bash_steps.pop(0)
                ollama_messages.append({
                    "role": "user",
                    "content": f"The previous command failed. Now run: {next_cmd}"
                })
                logger.info("claude_subprocess: injected next step after failure: %s", next_cmd)
            else:
                ollama_messages.append({
                    "role": "user",
                    "content": "The previous command failed. Continue with the remaining steps of the task."
                })
                logger.info("claude_subprocess: injected continue message after tool failure")

        # think=False disables CoT overhead for qwen3/qwen3.5 models (3-24x speed gain)
        # think=True required for reliable tool calling decisions.
        # Without thinking, qwen3.5 uses pattern matching and skips tools for "known" commands.
        # Speed penalty is acceptable: tool calling accuracy > inference speed.
        api_body: dict = {
            "model": ollama_model,
            "messages": ollama_messages,
            "stream": False,
            "think": True,
        }
        if ollama_tools:
            api_body["tools"] = ollama_tools

        _final_tokens = sum(len(str(m.get("content",""))) for m in ollama_messages) // 4
        _tool_names = [t["function"]["name"] for t in (ollama_tools or [])]
        logger.info("ollama request: model=%s msgs=%d est_tokens=%d tools=%s",
                    ollama_model, len(ollama_messages), _final_tokens, _tool_names)

        # Use stream=False: qwen3.5's XML <tool_call> format requires full response for
        # Ollama's qwen3.5 parser to detect tool calls. stream=True splits XML across
        # chunks causing tool calls to be missed. The pool probe is satisfied by the
        # message_start event yielded above (before this Ollama call).
        api_body["stream"] = False
        api_req = _ur_t.Request("http://localhost:11434/api/chat",
                                data=_j_t.dumps(api_body).encode(),
                                headers={"Content-Type": "application/json"})

        def _stream_collect():
            tool_calls, texts = [], []
            with _ur_t.urlopen(api_req, timeout=120) as _r:
                raw = _r.read()
            # stream=False: single JSON response
            try:
                _d = _j_t.loads(raw)
            except Exception as _pe:
                logger.warning("claude_subprocess: Ollama JSON parse error: %s raw=%s", _pe, raw[:200])
                return tool_calls, ""
            _msg = _d.get("message", {})
            _tcs = _msg.get("tool_calls")
            if _tcs:
                tool_calls.extend(_tcs)
            _content = _msg.get("content", "")
            if _content:
                texts.append(_content)
            return tool_calls, "".join(texts)

        mid = f"msg_{_uid_t.uuid4().hex[:24]}"

        # 4. Emit message_start BEFORE calling Ollama so the pool probe gets first chunk
        #    within its 30s timeout window. Ollama can take 40+ seconds to respond.
        yield "event: message_start\ndata: " + _j_t.dumps({
            "type": "message_start", "message": {
                "id": mid, "type": "message", "role": "assistant",
                "content": [], "model": request.model, "stop_reason": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            }
        }) + "\n\n"

        # Now call Ollama (after yielding message_start so pool probe succeeds)
        try:
            tool_calls_out, text_out = await asyncio.get_event_loop().run_in_executor(
                None, _stream_collect
            )
        except Exception as _re:
            logger.warning("claude_subprocess: Ollama /api/chat failed: %s", _re)
            tool_calls_out, text_out = [], ""

        # 5a'. If we have a tool_result in context and qwen3.5 returned no new tool_calls,
        #      use the tool_result as the output. This handles two cases:
        #      (a) qwen3.5 wraps result in commentary → replace with raw tool_result
        #      (b) qwen3.5 returns empty text after tool execution → use tool_result directly
        if _last_tool_result and not tool_calls_out:
            text_out = _last_tool_result
            logger.info("claude_subprocess: using tool_result as output (text_out was: %r)", text_out[:50] if text_out else "empty")

        # 5a. Check for XML tool calls embedded in text.
        #     qwen3.5 uses Claude Code tool descriptions which teach XML <tool_call> format.
        #     If no JSON tool_calls found but text contains XML tags, extract them here.
        if text_out and not tool_calls_out:
            _parsed = _parse_content(text_out)
            _xml_tools = [p for p in _parsed if p.get("type") == "tool_use"]
            if _xml_tools:
                for _p in _xml_tools:
                    tool_calls_out.append({
                        "id": f"toolu_{_uid_t.uuid4().hex[:24]}",
                        "function": {
                            "name": _p.get("name", ""),
                            "arguments": _p.get("input", {}),
                        }
                    })
                text_out = " ".join(
                    _p.get("text", "") for _p in _parsed if _p.get("type") == "text"
                ).strip()
                logger.info("claude_subprocess: extracted %d XML tool calls from text", len(_xml_tools))

        block_idx = 0

        # 5. Emit tool_use blocks if model made tool calls.
        #    Preserve Ollama's original tool call ID so round-trip back to Ollama is consistent.
        for tc in tool_calls_out:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            tool_args = fn.get("arguments", {})
            # Use Ollama's own id (call_xxx) — if absent fall back to generated toolu_xxx
            tool_id = tc.get("id") or f"toolu_{_uid_t.uuid4().hex[:24]}"
            args_json = _j_t.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
            yield "event: content_block_start\ndata: " + _j_t.dumps({
                "type": "content_block_start", "index": block_idx,
                "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}},
            }) + "\n\n"
            yield "event: content_block_delta\ndata: " + _j_t.dumps({
                "type": "content_block_delta", "index": block_idx,
                "delta": {"type": "input_json_delta", "partial_json": args_json},
            }) + "\n\n"
            yield "event: content_block_stop\ndata: " + _j_t.dumps({
                "type": "content_block_stop", "index": block_idx,
            }) + "\n\n"
            block_idx += 1

        # 6. Emit text block if model returned text
        if text_out:
            yield "event: content_block_start\ndata: " + _j_t.dumps({
                "type": "content_block_start", "index": block_idx,
                "content_block": {"type": "text", "text": ""},
            }) + "\n\n"
            yield "event: content_block_delta\ndata: " + _j_t.dumps({
                "type": "content_block_delta", "index": block_idx,
                "delta": {"type": "text_delta", "text": text_out},
            }) + "\n\n"
            yield "event: content_block_stop\ndata: " + _j_t.dumps({
                "type": "content_block_stop", "index": block_idx,
            }) + "\n\n"
            block_idx += 1

        stop_reason = "tool_use" if tool_calls_out else "end_turn"
        yield "event: message_delta\ndata: " + _j_t.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 0},
        }) + "\n\n"
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'


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
    return _fetch_ollama_model()


def _clean_env() -> dict[str, str]:
    """Return clean env: strip FCC mock credentials so subprocess uses real Claude auth."""
    env = dict(os.environ)
    for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CONFIG_DIR"):
        env.pop(key, None)
    return env
