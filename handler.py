"""RunPod serverless handler for Music Flamingo (nvidia/music-flamingo-hf).

Cold start (~30-90s on A100): downloads the model weights to local disk (cached
on subsequent warm starts in the same worker) and loads them onto the GPU.
Warm calls (~3-10s): just runs inference on the provided audio.

Request payload (event["input"]):
    {
        # One of these is required:
        "audio_b64": "<base64-encoded wav/mp3>",  # preferred: simplest, no external storage
        "audio_url": "https://.../file.wav",      # alternative: any URL the worker can fetch
        "audio_format": "wav",                    # optional hint, default "wav"

        "prompt": "Critique this melodic techno mix's low end.",  # required
        "max_new_tokens": 512,                                    # optional, default 512
        "system_prompt": "You are an expert producer...",         # optional, prepended
    }

Response: {"output": {"text": "..."}}
Errors: {"error": "..."}

The base64 path is simpler for end-to-end use because the MCP tool can just
read the local WAV, encode it, and POST without any external storage. RunPod's
synchronous `/runsync` endpoint accepts payloads up to ~10 MB which covers any
bounced region under ~30 s at 48 kHz / 24-bit stereo.
"""

from __future__ import annotations

import base64
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import requests
import runpod
import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

MODEL_ID = os.environ.get("MUSIC_FLAMINGO_MODEL_ID", "nvidia/music-flamingo-hf")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS_DEFAULT", "512"))
DOWNLOAD_TIMEOUT_SEC = 60
MAX_AUDIO_BYTES = 250 * 1024 * 1024  # 250 MB hard cap so we don't blow disk

_MODEL: Any = None
_PROCESSOR: Any = None
_DEVICE: torch.device | None = None


def _load_model() -> None:
    """Load Music Flamingo on first request. Subsequent requests reuse it."""
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None:
        return

    started = time.time()
    print(f"[handler] loading {MODEL_ID}…", flush=True)
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
    _MODEL = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # uses _DEVICE under the hood; "auto" handles multi-GPU
    ).eval()
    print(
        f"[handler] model loaded in {time.time() - started:.1f}s on {_DEVICE}",
        flush=True,
    )


def _decode_b64(audio_b64: str, audio_format: str) -> Path:
    """Decode an inline base64 audio payload to a tempfile."""
    print(f"[handler] decoding inline audio ({len(audio_b64)} b64 chars, format={audio_format})", flush=True)
    raw = base64.b64decode(audio_b64)
    if len(raw) > MAX_AUDIO_BYTES:
        raise ValueError(f"Decoded audio too large: {len(raw)} bytes (max {MAX_AUDIO_BYTES}).")
    ext = "." + audio_format.lstrip(".")
    fd, path_str = tempfile.mkstemp(suffix=ext, prefix="mf-input-")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    path = Path(path_str)
    print(f"[handler] decoded {len(raw)} bytes to {path}", flush=True)
    return path


def _download(audio_url: str) -> Path:
    """Pull the audio file to local /tmp. Reject anything > MAX_AUDIO_BYTES."""
    print(f"[handler] downloading {audio_url}", flush=True)
    with requests.get(audio_url, stream=True, timeout=DOWNLOAD_TIMEOUT_SEC) as r:
        r.raise_for_status()
        length = int(r.headers.get("Content-Length", "0"))
        if length and length > MAX_AUDIO_BYTES:
            raise ValueError(f"Audio file too large: {length} bytes (max {MAX_AUDIO_BYTES}).")
        # Guess extension from URL or default to .wav
        ext = Path(audio_url.split("?", 1)[0]).suffix or ".wav"
        fd, path_str = tempfile.mkstemp(suffix=ext, prefix="mf-input-")
        os.close(fd)
        path = Path(path_str)
        total = 0
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_AUDIO_BYTES:
                    path.unlink(missing_ok=True)
                    raise ValueError(f"Audio file too large: > {MAX_AUDIO_BYTES} bytes (streamed).")
                f.write(chunk)
        print(f"[handler] downloaded {total} bytes to {path}", flush=True)
        return path


def _critique(audio_path: Path, prompt: str, max_new_tokens: int, system_prompt: str | None) -> str:
    """Run a single Music Flamingo critique on an on-disk audio file."""
    assert _MODEL is not None and _PROCESSOR is not None

    content: list[dict[str, Any]] = []
    if system_prompt:
        content.append({"type": "text", "text": system_prompt})
    content.append({"type": "text", "text": prompt})
    content.append({"type": "audio", "path": str(audio_path)})

    conversation = [{"role": "user", "content": content}]
    inputs = _PROCESSOR.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(_MODEL.device)

    with torch.no_grad():
        outputs = _MODEL.generate(**inputs, max_new_tokens=max_new_tokens)
    text = _PROCESSOR.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    return (text[0] if text else "").strip()


def handler(event: dict[str, Any]) -> dict[str, Any]:
    """RunPod entrypoint. event["input"] is our request dict."""
    started = time.time()
    payload = event.get("input", {}) or {}
    audio_url = payload.get("audio_url")
    audio_b64 = payload.get("audio_b64")
    audio_format = (payload.get("audio_format") or "wav").lstrip(".")
    prompt = payload.get("prompt")
    max_new_tokens = int(payload.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS)
    system_prompt = payload.get("system_prompt")

    if not audio_url and not audio_b64:
        return {"error": "Missing required field: provide either audio_b64 or audio_url."}
    if not prompt:
        return {"error": "Missing required field: prompt"}

    audio_path: Path | None = None
    try:
        _load_model()
        audio_path = (
            _decode_b64(audio_b64, audio_format) if audio_b64 else _download(audio_url)
        )
        text = _critique(audio_path, prompt, max_new_tokens, system_prompt)
        return {
            "output": {
                "text": text,
                "elapsedSec": round(time.time() - started, 2),
                "model": MODEL_ID,
            }
        }
    except Exception as exc:
        print(f"[handler] error: {exc}\n{traceback.format_exc()}", flush=True)
        return {"error": str(exc), "elapsedSec": round(time.time() - started, 2)}
    finally:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass


# Called at module top-level (not under `if __name__ == "__main__"`) so RunPod's
# static "is this a serverless worker?" scanner finds it. The container's
# CMD always runs this file with python directly, so unconditional start is safe.
runpod.serverless.start({"handler": handler})
