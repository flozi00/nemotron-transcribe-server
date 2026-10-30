# Rollback — `parakeet-primeline.pl-ai.net`

The prod hostname and the served model id (`primeline/parakeet-primeline`) are
unchanged by the Nemotron swap. Only the pool behind them changed:

| | before | after |
|---|---|---|
| pool | `maas-inference/parakeet-primeline-v2` | `maas-inference/parakeet-primeline-v3` |
| runtime | Parakeet ASR (`ghcr.io/flozi00/parakeet-server`) | vLLM 0.32.0 + NeMo Speech (`ghcr.io/flozi00/nemotron-transcribe-server`) |
| node | `maas-inference-worker-03` (RTX 5090, 32 GB) | `maas-inference-worker-01` (RTX PRO 6000, 96 GB) |
| route | HTTPRoute → Service (no InferencePool) | HTTPRoute → Service (`direct_route: true`) |

## Why v2 was deleted

`parakeet-primeline-v2` was removed after v3 passed the checks below. Its
Deployment cannot be edited through the console any more (the console refuses
to round-trip a redacted `HF_TOKEN`), so a rollback re-creates it from the spec
recorded here rather than scaling the old object back up.

## Re-create v2 (the rollback)

`deploy_pool` with this spec — the values are the live ones read off the
Deployment before it was deleted:

```json
{
  "namespace": "maas-inference",
  "name": "parakeet-primeline-v2",
  "runtime": "parakeet",
  "image": "ghcr.io/flozi00/parakeet-server@sha256:f8ef39a709931fd080920bff26ecd050128349848f86ded831c51cb8b3fc00b8",
  "model": "primeline/parakeet-primeline",
  "gpus": 1,
  "replicas": 1,
  "cpu_millis": 7000,
  "memory_mib": 44029,
  "shm_mib": 1024,
  "gpu_selectors": [{"key": "nvidia.com/gpu.product", "value": "NVIDIA-GeForce-RTX-5090"}],
  "hostname": "parakeet-primeline.pl-ai.net",
  "env": [
    {"name": "DIARIZATION_ENABLED", "value": "true"},
    {"name": "DIARIZATION_DEVICE", "value": "cuda"},
    {"name": "TRAINING_REPO_ID", "value": "flozi00/parakeet-primeline-online-v2"},
    {"name": "HF_TOKEN", "value": "<the token from the model-weights secret — re-enter it>"},
    {"name": "PORT", "value": "8000"},
    {"name": "NEMO_REPO_ID", "value": "primeline/parakeet-primeline"},
    {"name": "ASR_MODEL_NAME", "value": "primeline/parakeet-primeline"}
  ]
}
```

The checkpoint it served was `runs/8dfee81c1181413ebad21e0272177c70/model.nemo`
(global step 28, German WER 5.14 % on the 50-item suite), selected by
`latest.json` in `flozi00/parakeet-primeline-online-v2`. The server reads
`latest.json` only at startup, so pointing it back at an older run needs a pod
restart.

Before attaching the hostname, detach it from v3 (`delete_route` on
`parakeet-primeline-v3`) — two HTTPRoutes cannot claim one hostname.

## What was verified before the swap

Same 50 items / 797 words, `benchmarks/asr.py`, same seed:

| | parakeet v2 | nemotron v3 |
|---|---|---|
| WER | 5.14 % (41/797) | **3.89 % (31/797)** |
| common_voice_19_0 | 3.53 % (11/312) | 3.85 % (12/312) |
| multilingual librispeech | 6.19 % (30/485) | **3.71 % (18/485)** |
| latency p50 | 363 ms | 418 ms |
| latency p90 | 449 ms | 657 ms |
| latency max | 529 ms | 1907 ms |

Quality is the reason for the swap (owner, 2026-10-30: quality outranks
latency). The tail latency is worse and that is accepted, not hidden.

Endpoints checked on `https://parakeet-primeline.pl-ai.net` after the swap:
`/health` 200, `/v1/models` lists `primeline/parakeet-primeline`,
`POST /v1/audio/transcriptions` (multipart) 200, `POST /v1/chat/completions`
with an inline audio data URL 200, `response_format=diarized_json` 200.
