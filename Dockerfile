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

# FlashInfer's prebuilt sm120f cubins. Without this the engine JIT-compiles
# `fp4_gemm_cutlass_sm120` on first use — the log says "several minutes", and
# nvcc's peak RSS does not fit the pod's resource share (0.875 x node capacity
# / node_gpus), so the engine is OOM-killed mid-compile and the container
# restarts in a loop. The wheel is version-locked to the flashinfer-python the
# lock file resolves (0.7.0.post1) and to cu130; a mismatch is ignored by
# FlashInfer and silently falls back to the JIT, so the pin is load-bearing.
RUN uv pip install --python /opt/nemo/.venv/bin/python --no-deps \
        --index-url https://flashinfer.ai/whl/cu130 \
        "flashinfer-jit-cache==0.7.0.post1+cu130" \
        "flashinfer-jit-cache-sm120f==0.7.0.post1+cu130"

# The parakeet-compatible front end and the entrypoint that splits the serving
# port (shim) from the engine port (vLLM).
COPY shim/parakeet_shim.py /opt/shim/parakeet_shim.py
COPY bin/vllm /usr/local/bin/vllm
RUN chmod 0755 /usr/local/bin/vllm

# Fail closed: a missing provider means the engine would JIT-compile at
# startup, which is exactly the OOM loop this image exists to avoid.
RUN /opt/nemo/.venv/bin/python -c "\
import flashinfer_jit_cache as c; \
p = c.get_jit_cache_providers(); \
assert p, 'no flashinfer jit-cache provider installed'; \
assert any('sm120' in a for prov in p for a in prov.cuda_architectures), \
    'no sm120 provider: %r' % [sorted(x.cuda_architectures) for x in p]; \
print('flashinfer jit-cache providers:', [sorted(x.cuda_architectures) for x in p])"

# No JIT cache directory: the sm120f cubins are prebuilt into the image, so
# there is nothing to compile and nothing to persist across restarts.

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/vllm"]
CMD ["serve", "--help"]
