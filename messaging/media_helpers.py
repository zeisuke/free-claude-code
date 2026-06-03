"""Media analysis helpers for the free-claude-code proxy.

Provides image, video, audio, and document analysis using the local Ollama
endpoint (gemma4:latest for vision, qwen3:8b for text tasks).
"""

from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

_OLLAMA_BASE = "http://localhost:11434/v1"
_VISION_MODEL = "gemma4:latest"
_OLLAMA_TIMEOUT = 120.0


async def analyze_image_ollama(image_path: Path, prompt: str) -> str:
    """Describe an image via Ollama moondream vision. Returns plain-text description."""
    import httpx

    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    suffix = image_path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".webp": "image/webp", ".gif": "image/gif"}
    mime = mime_map.get(suffix, "image/jpeg")
    data_url = f"data:{mime};base64,{b64}"

    payload = {
        "model": _VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT) as client:
            resp = await client.post(
                f"{_OLLAMA_BASE}/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer no-key-required"},
            )
            resp.raise_for_status()
            data_resp = resp.json()
            return data_resp["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Ollama image analysis failed: {}", exc)
        return f"[Image analysis failed: {exc}]"


def extract_video_frames(
    video_path: Path,
    max_frames: int = 6,
    interval_secs: float = 5.0,
) -> list[Path]:
    """Extract JPEG frames from a video using ffmpeg. Returns list of frame paths."""
    frame_dir = Path(tempfile.mkdtemp(prefix="fcc_video_frames_"))
    pattern = str(frame_dir / "frame_%03d.jpg")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps=1/{interval_secs},scale=768:-2",
        "-frames:v", str(max_frames),
        "-q:v", "3",
        pattern, "-y", "-loglevel", "error",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return sorted(frame_dir.glob("frame_*.jpg"))[:max_frames]
    except FileNotFoundError:
        logger.warning("ffmpeg not found — cannot extract video frames")
        return []
    except Exception as exc:
        logger.warning("ffmpeg frame extraction failed: {}", exc)
        return []


async def analyze_video_frames(video_path: Path, prompt: str) -> str:
    """Analyze a video by extracting frames and describing each via vision."""
    frames = extract_video_frames(video_path)
    if not frames:
        return "[Video analysis failed — ffmpeg unavailable or file unreadable]"

    parts = []
    for idx, frame in enumerate(frames):
        try:
            frame_prompt = (
                f"This is frame {idx + 1} of {len(frames)} from a video "
                f"(sampled every 5 seconds). {prompt}"
            )
            desc = await analyze_image_ollama(frame, frame_prompt)
            parts.append(f"[Frame {idx + 1} @ ~{idx * 5}s] {desc}")
        except Exception as exc:
            logger.warning("Frame {} analysis error: {}", idx + 1, exc)
        finally:
            try:
                frame.unlink(missing_ok=True)
            except Exception:
                pass

    return "\n\n".join(parts) if parts else "[Video: no frames could be analyzed]"


def extract_pdf_text(pdf_path: Path, max_chars: int = 8000) -> str:
    """Extract plain text from a PDF using pypdf. Returns truncated text or error note."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n\n".join(t.strip() for t in pages if t.strip())
        if not text:
            return "[PDF contained no extractable text]"
        if len(text) > max_chars:
            return text[:max_chars] + "\n\n[truncated]"
        return text
    except ImportError:
        return "[PDF extraction unavailable — pypdf not installed]"
    except Exception as exc:
        return f"[PDF extraction failed: {exc}]"
