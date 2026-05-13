"""
File-backed circuit breaker for the model pool.

Penalises failing/hanging models with a cooldown and rotates to the next
available model on each pick_models() call. State is written to a JSON file
so cooldowns survive process restarts and are shared across workers.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_COOLDOWN_429 = 15 * 60   # rate-limited  → 15 min
_COOLDOWN_HANG = 5 * 60   # timeout/hang  → 5 min

_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "model_cooldowns.json"
_lock = threading.Lock()
_mem: dict[str, float] = {}  # model_ref → expiry timestamp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read() -> dict:
    try:
        return json.loads(_STATE_PATH.read_text())
    except Exception:
        return {"cooldowns": {}}


def _write(cooldowns: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"cooldowns": cooldowns}))
    tmp.replace(_STATE_PATH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pick_models(pool: list[str]) -> list[str]:
    """Return pool reordered: hot models first, cooling models appended at end."""
    if not pool:
        return []
    now = time.time()
    with _lock:
        file_state = _read()
        for m, exp in file_state["cooldowns"].items():
            if exp > now:
                _mem[m] = max(_mem.get(m, 0), exp)
            elif m in _mem and _mem[m] <= now:
                del _mem[m]
        hot  = [m for m in pool if _mem.get(m, 0) <= now]
        cold = [m for m in pool if _mem.get(m, 0) >  now]
    return hot + cold


def penalise(model: str, *, is_429: bool) -> None:
    """Mark model as cooling. Persisted to file immediately."""
    ttl = _COOLDOWN_429 if is_429 else _COOLDOWN_HANG
    exp = time.time() + ttl
    with _lock:
        _mem[model] = exp
        state = _read()
        state["cooldowns"][model] = exp
        _write(state["cooldowns"])


def mark_used(model: str) -> None:
    """Clear any cooldown after a successful response."""
    with _lock:
        _mem.pop(model, None)
        state = _read()
        state["cooldowns"].pop(model, None)
        _write(state["cooldowns"])


def get_status(pool: list[str]) -> list[dict]:
    """Return pool status dicts for diagnostics."""
    from datetime import datetime, timezone
    now = time.time()
    with _lock:
        cooldowns = dict(_mem)
    result = []
    for m in pool:
        exp = cooldowns.get(m, 0)
        cooling = exp > now
        result.append({
            "model": m,
            "status": "cooling" if cooling else "hot",
            "cooldown_remaining_s": int(exp - now) if cooling else None,
            "cooldown_until": datetime.fromtimestamp(exp, tz=timezone.utc).isoformat() if cooling else None,
        })
    return result
