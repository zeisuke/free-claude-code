"""Model pool with cooldown-based circuit breaking for free-tier providers."""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_POOL: list[str] = [
    "open_router/openai/gpt-oss-120b:free",
    "open_router/openai/ring-2.6-1t:free",
    "open_router/qwen/qwen3-coder:free",
    "open_router/qwen/qwen3-next-80b-a3b-instruct:free",
    "open_router/google/gemma-4-31b-it:free",
    "open_router/meta-llama/llama-3.3-70b-instruct:free",
    "open_router/openai/gpt-oss-20b:free",
    "open_router/microsoft/mai-ds-r1:free",
    "open_router/google/gemma-4-26b-a4b-it:free",
    "open_router/nousresearch/hermes-3-llama-3.1-405b:free",
    "open_router/z-ai/glm-4.5-air:free",
    "open_router/minimax/minimax-m2.5:free",
]

_COOLDOWN_429 = 15 * 60      # rate-limited — back off 15 min
_COOLDOWN_TIMEOUT = 3 * 60   # first-token hang — back off 3 min
_COOLDOWN_5XX = 5 * 60       # server error — back off 5 min

# State file lives alongside this module's package root
_STATE_PATH = Path(__file__).resolve().parent.parent / "pool_state.json"


def _read_file_state() -> dict:
    try:
        return json.loads(_STATE_PATH.read_text())
    except Exception:
        return {}


def _write_file_state(state: dict) -> None:
    try:
        tmp = _STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(_STATE_PATH)
    except Exception:
        pass


@dataclass
class _ModelState:
    model: str
    rank: int
    cooldown_until: float = 0.0
    last_used: float = 0.0

    def is_cooling(self) -> bool:
        return time.time() < self.cooldown_until

    def cooldown_remaining(self) -> float:
        return max(0.0, self.cooldown_until - time.time())


class ModelPool:
    def __init__(self, models: list[str]) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, _ModelState] = {
            model: _ModelState(model=model, rank=i)
            for i, model in enumerate(models)
        }
        self._last_used_model: str = ""
        self._load_persisted_state()

    def _load_persisted_state(self) -> None:
        saved = _read_file_state()
        if not saved:
            return
        now = time.time()
        cooldowns = saved.get("cooldowns", {})
        for model, exp in cooldowns.items():
            if model in self._states and exp > now:
                self._states[model].cooldown_until = exp
        self._last_used_model = saved.get("last_used_model", "")
        # Restore last_used timestamp so get_status() returns correct last_used_model
        if self._last_used_model and self._last_used_model in self._states:
            self._states[self._last_used_model].last_used = saved.get("last_used_ts", time.time())

    def _save_state(self) -> None:
        cooldowns = {
            m: s.cooldown_until
            for m, s in self._states.items()
            if s.cooldown_until > time.time()
        }
        _write_file_state({
            "cooldowns": cooldowns,
            "last_used_model": self._last_used_model,
            "last_used_ts": self._states[self._last_used_model].last_used
            if self._last_used_model and self._last_used_model in self._states else 0.0,
        })

    def pick(self) -> list[str]:
        """Return pool with hot models first, cooling models appended at end (never omitted)."""
        with self._lock:
            hot = []
            cooling = []
            for state in sorted(self._states.values(), key=lambda s: s.rank):
                if state.is_cooling():
                    cooling.append(state.model)
                else:
                    hot.append(state.model)
            return hot + cooling

    def penalise(self, model: str, *, is_429: bool = False, is_timeout: bool = False) -> None:
        """Penalise a model. is_429 > is_timeout > generic in cooldown duration."""
        with self._lock:
            state = self._states.get(model)
            if state is None:
                return
            if is_429:
                duration = _COOLDOWN_429
            elif is_timeout:
                duration = _COOLDOWN_TIMEOUT
            else:
                duration = _COOLDOWN_5XX
            state.cooldown_until = time.time() + duration
            self._save_state()

    def mark_used(self, model: str) -> None:
        """Record a successful completion. Clears any existing cooldown."""
        with self._lock:
            state = self._states.get(model)
            if state is None:
                return
            state.cooldown_until = 0.0
            state.last_used = time.time()
            self._last_used_model = model
            self._save_state()

    def get_status(self) -> dict:
        """Return dict with: model_pool (list of per-model dicts), hot_count, cooling_count, last_used_model."""
        with self._lock:
            model_pool = []
            hot_count = 0
            cooling_count = 0

            for state in sorted(self._states.values(), key=lambda s: s.rank):
                cooling = state.is_cooling()
                status = "cooling" if cooling else "hot"
                if cooling:
                    cooling_count += 1
                else:
                    hot_count += 1
                model_pool.append({
                    "model": state.model,
                    "rank": state.rank,
                    "status": status,
                    "cooldown_remaining_s": state.cooldown_remaining(),
                    "cooldown_until": state.cooldown_until,
                })

            return {
                "model_pool": model_pool,
                "hot_count": hot_count,
                "cooling_count": cooling_count,
                "last_used_model": self._last_used_model,
            }


def make_pool(models: list[str]) -> ModelPool:
    """Construct a :class:`ModelPool` from an explicit model list."""
    return ModelPool(models)
