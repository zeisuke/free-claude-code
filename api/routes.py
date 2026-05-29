"""FastAPI route handlers."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count
from core.trace import trace_event
from providers.registry import ProviderRegistry

from . import dependencies
from .dependencies import get_settings, require_api_key
from .gateway_model_ids import gateway_model_id, no_thinking_gateway_model_id
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import ModelResponse, ModelsListResponse
from .services import ClaudeProxyService

router = APIRouter()

DISCOVERED_MODEL_CREATED_AT = "1970-01-01T00:00:00Z"


SUPPORTED_CLAUDE_MODELS = [
    ModelResponse(
        id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-haiku-4-20250514",
        display_name="Claude Haiku 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        created_at="2024-02-29T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        created_at="2024-10-22T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku",
        created_at="2024-03-07T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        created_at="2024-10-22T00:00:00Z",
    ),
]


def get_proxy_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ClaudeProxyService:
    """Build the request service for route handlers."""
    pool = getattr(request.app.state, "pool", None)
    registry = getattr(request.app.state, "provider_registry", None)
    return ClaudeProxyService(
        settings,
        provider_getter=lambda provider_type: dependencies.resolve_provider(
            provider_type, app=request.app, settings=settings
        ),
        token_counter=get_token_count,
        pool=pool,
        provider_registry=registry if isinstance(registry, ProviderRegistry) else None,
    )


def _probe_response(allow: str) -> Response:
    """Return an empty success response for compatibility probes."""
    return Response(status_code=204, headers={"Allow": allow})


def _discovered_model_response(model_id: str, *, display_name: str) -> ModelResponse:
    return ModelResponse(
        id=model_id,
        display_name=display_name,
        created_at=DISCOVERED_MODEL_CREATED_AT,
    )


def _append_unique_model(
    models: list[ModelResponse], seen: set[str], model: ModelResponse
) -> None:
    if model.id in seen:
        return
    seen.add(model.id)
    models.append(model)


def _append_provider_model_variants(
    models: list[ModelResponse],
    seen: set[str],
    provider_model_ref: str,
    *,
    supports_thinking: bool | None = None,
) -> None:
    if supports_thinking is not False:
        _append_unique_model(
            models,
            seen,
            _discovered_model_response(
                gateway_model_id(provider_model_ref),
                display_name=provider_model_ref,
            ),
        )
    _append_unique_model(
        models,
        seen,
        _discovered_model_response(
            no_thinking_gateway_model_id(provider_model_ref),
            display_name=f"{provider_model_ref} (no thinking)",
        ),
    )


def _build_models_list_response(
    settings: Settings, provider_registry: ProviderRegistry | None
) -> ModelsListResponse:
    models: list[ModelResponse] = []
    seen: set[str] = set()

    for ref in settings.configured_chat_model_refs():
        supports_thinking = None
        if provider_registry is not None:
            supports_thinking = provider_registry.cached_model_supports_thinking(
                ref.provider_id, ref.model_id
            )
        _append_provider_model_variants(
            models,
            seen,
            ref.model_ref,
            supports_thinking=supports_thinking,
        )

    if provider_registry is not None:
        blocklist = settings.model_blocklist
        for model_info in provider_registry.cached_prefixed_model_infos():
            if model_info.model_id in blocklist:
                continue
            _append_provider_model_variants(
                models,
                seen,
                model_info.model_id,
                supports_thinking=model_info.supports_thinking,
            )

    for model in SUPPORTED_CLAUDE_MODELS:
        _append_unique_model(models, seen, model)

    return ModelsListResponse(
        data=models,
        first_id=models[0].id if models else None,
        has_more=False,
        last_id=models[-1].id if models else None,
    )


# =============================================================================
# Routes
# =============================================================================
@router.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    service: ClaudeProxyService = Depends(get_proxy_service),
    _auth=Depends(require_api_key),
):
    """Create a message — streaming by default, JSON for stream=false."""
    response = service.create_message(request_data)
    if request_data.stream is not True:  # None (omitted) or False → non-streaming JSON
        from api.services import _accumulate_sse_to_json
        # StreamingResponse wraps the async generator; extract it for accumulation
        stream = getattr(response, "body_iterator", None) or response
        return await _accumulate_sse_to_json(stream)
    return response


@router.api_route("/v1/messages", methods=["HEAD", "OPTIONS"])
async def probe_messages(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the messages endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request_data: TokenCountRequest,
    service: ClaudeProxyService = Depends(get_proxy_service),
    _auth=Depends(require_api_key),
):
    """Count tokens for a request."""
    return service.count_tokens(request_data)


@router.api_route("/v1/messages/count_tokens", methods=["HEAD", "OPTIONS"])
async def probe_count_tokens(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the token count endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.get("/")
async def root(settings: Settings = Depends(get_settings)):
    """Root endpoint — no auth required."""
    return {
        "status": "ok",
        "provider": settings.provider_type,
        "model": settings.model,
    }


@router.api_route("/", methods=["HEAD", "OPTIONS"])
async def probe_root():
    """Respond to unauthenticated local compatibility probes for the root endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/admin", response_class=HTMLResponse)
async def admin(settings: Settings = Depends(get_settings)):
    """Admin dashboard — no auth required."""
    tiers = [
        ("Default", settings.model),
        ("Opus",    getattr(settings, "model_opus",   None) or "—"),
        ("Sonnet",  getattr(settings, "model_sonnet", None) or "—"),
        ("Haiku",   getattr(settings, "model_haiku",  None) or "—"),
    ]
    rows = "".join(
        f"<tr><td>{tier}</td><td><code>{model}</code></td></tr>"
        for tier, model in tiers
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Free Claude Code — Admin</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:system-ui,sans-serif;background:#0f0f0f;color:#e0e0e0;padding:2rem}}
    h1{{font-size:1.4rem;font-weight:600;margin-bottom:1.5rem;color:#fff}}
    .card{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:1.25rem;margin-bottom:1.25rem}}
    .card h2{{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;color:#888;margin-bottom:.75rem}}
    table{{width:100%;border-collapse:collapse;font-size:.9rem}}
    td{{padding:.45rem .6rem;border-bottom:1px solid #222}}
    tr:last-child td{{border:none}}
    td:first-child{{color:#888;width:90px}}
    code{{color:#7ec8e3;font-size:.85rem}}
    .badge{{display:inline-block;padding:.15rem .5rem;border-radius:4px;font-size:.75rem;font-weight:600}}
    .ok{{background:#1a3a1a;color:#4caf50}}
    #models-body td{{font-size:.8rem}}
    .refresh{{float:right;font-size:.75rem;cursor:pointer;border:none;background:none;color:#666}}
    .refresh:hover{{color:#aaa}}
  </style>
</head>
<body>
  <h1>⚡ Free Claude Code &nbsp;<span style="font-weight:400;color:#555;font-size:.9rem">Admin</span></h1>

  <div class="card">
    <h2>Status</h2>
    <table>
      <tr><td>Server</td><td><span class="badge ok">running</span></td></tr>
      <tr><td>Provider</td><td><code>{settings.provider_type}</code></td></tr>
      <tr><td>Thinking</td><td><code>{"on" if getattr(settings, "enable_model_thinking", False) else "off"}</code></td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Model Routing</h2>
    <table>{rows}</table>
  </div>

  <div class="card">
    <h2>Available Models <button class="refresh" onclick="loadModels()">↻ refresh</button></h2>
    <table><thead><tr style="color:#555;font-size:.75rem">
      <th style="text-align:left;padding:.3rem .6rem">Model</th>
      <th style="text-align:left;padding:.3rem .6rem">Provider</th>
    </tr></thead>
    <tbody id="models-body"><tr><td colspan="2" style="color:#555">loading…</td></tr></tbody></table>
  </div>

  <div class="card">
    <h2>Pool Status <button class="refresh" onclick="loadPool()">↻ refresh</button></h2>
    <div id="pool-status-header" style="font-size:.8rem;color:#888;margin-bottom:.5rem">loading…</div>
    <table><thead><tr style="color:#555;font-size:.75rem">
      <th style="text-align:left;padding:.3rem .6rem">Rank</th>
      <th style="text-align:left;padding:.3rem .6rem">Model</th>
      <th style="text-align:left;padding:.3rem .6rem">Status</th>
      <th style="text-align:left;padding:.3rem .6rem">Cooldown</th>
    </tr></thead>
    <tbody id="pool-body"><tr><td colspan="4" style="color:#555">loading…</td></tr></tbody></table>
  </div>

  <div class="card">
    <h2>Endpoints</h2>
    <table>
      <tr><td>Proxy</td><td><code>POST /v1/messages</code></td></tr>
      <tr><td>Models</td><td><code>GET /v1/models</code> &nbsp;<span style="color:#555;font-size:.8rem">(Bearer freecc)</span></td></tr>
      <tr><td>Health</td><td><code>GET /health</code></td></tr>
      <tr><td>Pool</td><td><code>GET /pool/status</code> &nbsp;<span style="color:#555;font-size:.8rem">(Bearer freecc)</span></td></tr>
    </table>
  </div>

  <script>
    async function loadModels() {{
      const tbody = document.getElementById('models-body');
      tbody.innerHTML = '<tr><td colspan="2" style="color:#555">loading…</td></tr>';
      try {{
        const r = await fetch('/v1/models', {{headers:{{'Authorization':'Bearer freecc'}}}});
        const data = await r.json();
        tbody.innerHTML = data.data.slice(0, 30).map(m => {{
          const parts = m.display_name.split('/');
          const provider = parts[0] || '—';
          return `<tr><td>${{m.display_name}}</td><td style="color:#888">${{provider}}</td></tr>`;
        }}).join('');
      }} catch(e) {{
        tbody.innerHTML = `<tr><td colspan="2" style="color:#e57373">${{e.message}}</td></tr>`;
      }}
    }}

    async function loadPool() {{
      const header = document.getElementById('pool-status-header');
      const tbody = document.getElementById('pool-body');
      tbody.innerHTML = '<tr><td colspan="4" style="color:#555">loading…</td></tr>';
      try {{
        const r = await fetch('/pool/status', {{headers:{{'Authorization':'Bearer freecc'}}}});
        const data = await r.json();
        if (!data.enabled) {{
          header.textContent = 'Pool disabled';
          tbody.innerHTML = '<tr><td colspan="4" style="color:#555">No pool configured</td></tr>';
          return;
        }}
        const lastUsed = data.last_used && data.last_used > 0
          ? new Date(data.last_used * 1000).toLocaleTimeString()
          : '—';
        header.innerHTML = `Hot: <strong style="color:#4caf50">${{data.hot_count}}</strong> &nbsp; Cooling: <strong style="color:#7ec8e3">${{data.cooling_count}}</strong> &nbsp; Last used: ${{lastUsed}}`;
        tbody.innerHTML = data.model_pool.map(m => {{
          const isCooling = m.status === 'cooling';
          const statusBadge = isCooling
            ? '<span style="color:#7ec8e3">❄️ cooling</span>'
            : '<span style="color:#4caf50">🟢 hot</span>';
          const remaining = isCooling
            ? `${{Math.ceil(m.cooldown_remaining_s)}}s`
            : '—';
          const modelShort = m.model.replace('open_router/', '');
          return `<tr><td style="color:#888">${{m.rank}}</td><td style="font-size:.78rem"><code>${{modelShort}}</code></td><td>${{statusBadge}}</td><td style="color:#888;font-size:.8rem">${{remaining}}</td></tr>`;
        }}).join('');
      }} catch(e) {{
        header.textContent = '';
        tbody.innerHTML = `<tr><td colspan="4" style="color:#e57373">${{e.message}}</td></tr>`;
      }}
    }}

    loadModels();
    loadPool();
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}




@router.get("/pool/status")
async def pool_status(
    request: Request,
    _auth=Depends(require_api_key),
):
    """Return circuit-breaker pool status."""
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        return {"enabled": False, "model_pool": [], "hot_count": 0, "cooling_count": 0, "last_used": None}
    status = pool.get_status()
    status["enabled"] = True
    return status


@router.api_route("/health", methods=["HEAD", "OPTIONS"])
async def probe_health():
    """Respond to compatibility probes for the health endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models(
    request: Request,
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_api_key),
):
    """List the model ids this proxy advertises to Claude-compatible clients."""
    trace_event(stage="ingress", event="api.models.list", source="api")
    registry = getattr(request.app.state, "provider_registry", None)
    provider_registry = registry if isinstance(registry, ProviderRegistry) else None
    return _build_models_list_response(settings, provider_registry)


@router.post("/stop")
async def stop_cli(request: Request, _auth=Depends(require_api_key)):
    """Stop all CLI sessions and pending tasks."""
    handler = getattr(request.app.state, "message_handler", None)
    if not handler:
        # Fallback if messaging not initialized
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            logger.info("STOP_CLI: source=cli_manager cancelled_count=N/A")
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await handler.stop_all_tasks()
    trace_event(
        stage="ingress",
        event="api.cli.stop_via_handler",
        source="api",
        cancelled_nodes=count,
    )
    logger.info("STOP_CLI: source=handler cancelled_count={}", count)
    return {"status": "stopped", "cancelled_count": count}
