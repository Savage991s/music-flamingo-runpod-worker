# Music Flamingo serverless worker — runs on RunPod GPU instances.
#
# Base image: RunPod's official PyTorch image with CUDA 12.4 + PyTorch 2.5.
# Includes torch, torchaudio, basic system deps, and the `runpod` Python SDK.
# We layer the model code + our handler on top.
#
# Cold-start optimization:
#   - The model (~16 GB) is NOT baked into the image. Doing so would push the
#     image past 30 GB, slow every pull, and lock us to a specific MF revision.
#   - Instead the handler downloads weights from HF on first call and caches
#     them under /runpod-volume (RunPod persists this between cold starts on
#     the same endpoint). After the first cold start, subsequent ones reuse
#     the cache and finish in ~10-15 s.
#   - To pre-warm: set MUSIC_FLAMINGO_PREDOWNLOAD=1 in the endpoint env and
#     uncomment the RUN line below. This bakes weights into the image
#     (faster first-call, larger image, slower cold-pulls on new workers).

# PyTorch 2.8 / CUDA 12.8 — the git-main transformers we install in
# requirements.txt imports torch.distributed.tensor.parallel.ParallelStyle,
# which only exists in torch >= 2.6. PyTorch 2.4 throws NameError at import.
# This is the newest verified runpod/pytorch tag on Docker Hub.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# System deps for librosa/soundfile + a small set of audio codecs so we can
# decode whatever WAV/MP3/FLAC the caller throws at us without surprises.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps — cached layer separate from handler code so iteration on
# handler.py doesn't re-resolve the wheel set.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Tell HF + transformers to cache models on the persistent volume so cold
# starts after the first one are fast.
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1

COPY handler.py /app/handler.py

# Uncomment to bake the model into the image (see note above):
# ARG MUSIC_FLAMINGO_PREDOWNLOAD=0
# RUN if [ "$MUSIC_FLAMINGO_PREDOWNLOAD" = "1" ]; then \
#       python -c "from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor; \
#                  AutoProcessor.from_pretrained('nvidia/music-flamingo-hf'); \
#                  AudioFlamingo3ForConditionalGeneration.from_pretrained('nvidia/music-flamingo-hf')"; \
#     fi

CMD ["python", "-u", "/app/handler.py"]
