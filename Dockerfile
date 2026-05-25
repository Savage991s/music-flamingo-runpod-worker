# Music Flamingo serverless worker — runs on RunPod GPU instances.
#
# Base image: runpod/base — RunPod's recommended starter for serverless
# workers. It ships with Python 3.11 + pip + the runpod CLI tools but is
# otherwise barebones: NO pre-installed PyTorch, NO sshd entrypoint that
# fights with serverless. We install torch + the model stack on top.
#
# Why not runpod/pytorch:* — those images are tuned for interactive Pods
# and ship an ENTRYPOINT that starts sshd / Jupyter. In serverless that
# entrypoint runs instead of the handler, and the worker exits with code 1
# before any Python is reached. Clearing the ENTRYPOINT in Dockerfile also
# strips CUDA library setup that the entrypoint relied on, producing the
# same silent crash. The clean fix is to start from runpod/base.

FROM runpod/base:1.0.3-cuda1290-ubuntu2404

WORKDIR /app

# System-level audio codecs so librosa / soundfile can decode any wav/mp3/
# flac/aiff the caller throws at us.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first with the explicit CUDA wheel index, then the rest
# of the stack. Order matters — some packages (accelerate, transformers
# with native extensions) link against torch at install time.
# cu128 wheels are forward-compatible with CUDA 12.9 runtime in the base
# image.
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Tell HF + transformers to cache models on the persistent worker volume so
# only the very first cold start pays the ~16 GB MF download.
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
