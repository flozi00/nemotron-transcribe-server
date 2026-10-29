# Nemotron-3.5-Transcribe-30B-A4B-NVFP4 on vLLM 0.32.0 + NeMo Speech, behind a
# parakeet-compatible front end.
#
# The model card's runtime recipe, verbatim: vLLM 0.32.0 and NeMo Speech main
# (which ships the `nemo_speechlm` vLLM plugin that registers
# Nemotron3_5TranscribeForConditionalGeneration), plus a CUDA 13 toolkit on
# PATH because FlashInfer JIT-compiles the quantized attention kernels on first
# use. The devel base is what provides that nvcc; the engine itself comes from
# NeMo's own lock file, so torch/vLLM versions are the ones NVIDIA validated
# rather than whatever the base image happened to carry.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/nemo/.venv \
    PATH=/usr/local/bin:/opt/nemo/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv

# NeMo Speech main, installed exactly as the model card prescribes: the locked
# dependency set for the `asr` + `vllm` extras, then the project itself without
# deps (it is an editable checkout, not a wheel).
RUN git clone --depth 1 https://github.com/NVIDIA-NeMo/Speech.git /opt/nemo
WORKDIR /opt/nemo
RUN uv venv --python 3.13 /opt/nemo/.venv \
    && uv sync --extra asr --extra vllm --locked --no-install-project \
    && uv pip install --python /opt/nemo/.venv/bin/python --no-deps -e . \
    && uv pip install --python /opt/nemo/.venv/bin/python --no-deps \
        av==19.0.2 opencv-python-headless==5.0.0.93

# The parakeet-compatible front end and the entrypoint that splits the serving
# port (shim) from the engine port (vLLM).
COPY shim/parakeet_shim.py /opt/shim/parakeet_shim.py
COPY bin/vllm /usr/local/bin/vllm
RUN chmod 0755 /usr/local/bin/vllm

# FlashInfer's JIT cache lives here; a mounted cache PVC keeps it across
# restarts instead of recompiling the quantized kernels every cold start.
ENV FLASHINFER_WORKSPACE_BASE=/cache/flashinfer

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/vllm"]
CMD ["serve", "--help"]
