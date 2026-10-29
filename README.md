# nemotron-transcribe-server

A drop-in replacement for `parakeet-server`: one container that serves
**NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4** on vLLM 0.32.0 + NeMo Speech,
answering both the LLM API (`/v1/chat/completions`) and parakeet's
transcription API (`/v1/audio/transcriptions`) on the same port.

The served model id and the hostname stay the parakeet ones, so clients do not
change: `primeline/parakeet-primeline` on `parakeet-primeline.pl-ai.net`.

## Why an image at all

The checkpoint is an out-of-tree vLLM model (`model_type: nemo_speechlm`,
architecture `Nemotron3_5TranscribeForConditionalGeneration`). It loads only
with the `nemo_speechlm` plugin that ships in **NeMo Speech main**, and the
quantized attention kernels are JIT-compiled by FlashInfer, which needs `nvcc`
from a **CUDA 13** toolkit on `PATH`. Neither is in `vllm/vllm-openai:v0.32.0`,
so the image is built from the CUDA 13 devel base and installs NeMo Speech with
its own lock file — the exact recipe from the model card.

## What the shim does

vLLM has no `/v1/audio/transcriptions` route (v0.32.0 has no speech-to-text
entrypoint at all); the model is transcribed through `/v1/chat/completions`
with the audio inline. `shim/parakeet_shim.py` therefore:

- listens on the port the console probes and moves vLLM to an internal port
  (`bin/vllm` rewrites `--port`);
- translates a parakeet multipart request into a chat completion with the audio
  as a `data:` URL, `skip_special_tokens=false` and
  `mm_processor_kwargs.capture_ctc_timestamps=true`;
- reshapes the answer into parakeet-server's schema — `text`, `segments`,
  `language`, `duration`, `transcription_time`, `queue_wait_time`, `task` for
  `response_format=json`, and `task`/`duration`/`text`/`segments` with
  `speaker`/`start`/`end` for `response_format=diarized_json`;
- proxies every other route (`/v1/models`, `/v1/chat/completions`, …) straight
  through, and answers `/health` and `/ready` in parakeet's shape.

Diarization comes from the checkpoint's own speaker encoder: the transcript
carries `<spk:N>` tags, and the CTC alignment times each speaker run.

## Build

CI builds and pushes `ghcr.io/flozi00/nemotron-transcribe-server:main` on every
push to `main` (`.github/workflows/build-image.yml`). The build is large — a
CUDA 13 devel base plus the full NeMo Speech dependency set.

## Run

```bash
docker run --gpus all -p 8080:8080 \
  -e VLLM_PLUGINS=nemo_speechlm \
  -e VLLM_MAX_AUDIO_DECODE_DURATION_S=3600 \
  -e VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel \
  ghcr.io/flozi00/nemotron-transcribe-server:main \
  serve /models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4 \
    --served-model-name primeline/parakeet-primeline \
    --trust-remote-code --dtype bfloat16 --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85 --max-model-len 65536 \
    --max-num-batched-tokens 4096 --max-num-seqs 64 \
    --enable-chunked-prefill --no-enable-prefix-caching \
    --use-replayssm --mamba-backend flashinfer --mamba-cache-mode none \
    --mm-processor-cache-gb 0 --no-async-scheduling \
    --limit-mm-per-prompt '{"audio":{"count":1,"length":57600000}}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":8}' \
    --reasoning-parser nemotron_v3 --trust-request-mm-kwargs \
    --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware \
    --host 0.0.0.0 --port 8080
```

`--port` is the *public* port: the wrapper starts the shim there and runs the
engine on 8000.

## Test

```bash
python3 test_shim.py     # real shim process against a fake vLLM upstream
```
