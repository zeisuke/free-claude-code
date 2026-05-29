"""Claude subprocess provider — real Claude via 'claude -p', Ollama fallback."""

from .client import ClaudeSubprocessProvider

__all__ = ["ClaudeSubprocessProvider"]
