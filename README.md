# Music Flamingo serverless worker

RunPod serverless worker that exposes [nvidia/music-flamingo-hf](https://huggingface.co/nvidia/music-flamingo-hf) as a per-call HTTP endpoint.

## Files

- `handler.py` — RunPod entrypoint. Loads MF on cold start, takes `{audio_url, prompt}` and returns `{text}`.
- `Dockerfile` — based on `runpod/pytorch:2.5.0-py3.11-cuda12.4.1`, layers MF deps on top.
- `requirements.txt` — Python deps.

## Deployment (the easy path, no Docker locally needed)

1. Push this folder (`scripts/audio-critique/runpod-worker/`) to a public GitHub repo.
2. In RunPod console → Serverless → New Endpoint → GitHub repo
3. Choose this repo, branch `main`, root dir = the worker directory.
4. GPU: A6000 48GB or A100 80GB. Active workers: 0. Max workers: 1 (raise later).
5. Idle timeout: 300s (keeps it warm between consecutive listens).
6. After deploy, RunPod builds the image and gives you a URL like:
       https://api.runpod.ai/v2/<endpoint-id>/run

7. Put the endpoint ID + your RunPod API key into the parent project's `.env`:
       MUSIC_FLAMINGO_RUNPOD_ENDPOINT_ID=...
       RUNPOD_API_KEY=...

8. The `audio_critique` MCP tool reads those at startup and calls your endpoint.

## First cold start

Without `MUSIC_FLAMINGO_PREDOWNLOAD=1`, the very first cold start downloads
~16 GB of weights from Hugging Face (~3-5 min on RunPod's network). Subsequent
cold starts reuse the cached weights via `/runpod-volume` and complete in
10-15 s.

To pre-bake the model into the image (faster first call, slower image build
+ slower worker pulls), uncomment the relevant `RUN` block in the Dockerfile.

## Request / response shape

POST `/run` (or `/runsync` for blocking):

    {
      "input": {
        "audio_url": "https://...presigned.../bounce.wav",
        "prompt": "Critique this melodic techno mix's low end.",
        "max_new_tokens": 512,
        "system_prompt": "You are an expert melodic techno mixing engineer."
      }
    }

Response:

    {
      "output": {
        "text": "The kick fundamental sits at ~55 Hz and is …",
        "elapsedSec": 8.4,
        "model": "nvidia/music-flamingo-hf"
      }
    }

Errors propagate as `{"error": "..."}` at the top level (no `output`).
