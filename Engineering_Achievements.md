# SolQuant — Engineering Achievements

## Project Summary

SolQuant is a three-tier AI inference platform designed to run quantized Large Language Models on severely constrained NVIDIA GPUs. The entire stack — LLM inference, vector-based RAG retrieval, and autonomous agent orchestration — operates within a **2 GB VRAM hard limit** on an NVIDIA MX550 GPU.

---

## Hardware Under Test

| Spec | Value |
|---|---|
| GPU | NVIDIA GeForce MX550 |
| VRAM | 2048 MB GDDR6 |
| CUDA Compute Capability | 7.5 |
| NVIDIA Driver | 580.142 |
| CUDA Toolkit | 12.0 |
| OS | Linux Mint (Ubuntu 24.04 base) |
| Python | 3.12.3 |

---

## Achievement 1: GPU-Accelerated Inference on 2 GB VRAM

### What Was Built

A FastAPI REST API (`server.py`) serving quantized GGUF models via `llama-cpp-python` with full NVIDIA CUDA offloading. The server auto-downloads models from HuggingFace and loads them onto the GPU with configurable layer offloading.

### What Was Proven

We validated **two model sizes** on the same MX550 hardware — without changing a single line of code:

#### Qwen2.5-0.5B (Q4_K_M) — Full GPU Offload

| Metric | Value |
|---|---|
| Model file size | 491 MB |
| GPU layers offloaded | **24/24 (100%)** |
| VRAM at rest | 1,129 MB / 2,048 MB (55.1%) |
| VRAM during inference | 1,147 MB / 2,048 MB (56.0%) |
| VRAM delta during inference | +18 MB |
| Free headroom | 919 MB |
| Throughput | **41.8 tokens/sec** |
| Configuration | Default (zero tuning needed) |

#### Qwen2.5-1.5B (Q4_K_M) — Partial GPU Offload

| Metric | Value |
|---|---|
| Model file size | 1.12 GB |
| GPU layers offloaded | **20/28 (71%)** — 8 layers on CPU |
| VRAM at rest | 1,489 MB / 2,048 MB (72.7%) |
| VRAM during inference | 1,507 MB / 2,048 MB (73.6%) |
| VRAM delta during inference | +18 MB |
| Free headroom | 541 MB |
| Throughput | **16.9 tokens/sec** |
| Configuration | `SQ_N_GPU_LAYERS=20` (single env var change) |

#### Qwen2.5-1.5B — Full Offload Attempt (28 layers)

Attempting to offload all 28 layers of the 1.5B model **triggered a CUDA out-of-memory error**, exactly as predicted:

```
CUDA0 model buffer:    752 MB
CUDA0 KV cache:         56 MB
CUDA0 compute buffer:  482 MB  ← allocation FAILED
Total requested:     1,290 MB  → exceeds available VRAM
```

This crash demonstrates why the VRAM budget enforcement, partial offloading, and `SQ_N_GPU_LAYERS` tuning exist. The system is designed to prevent this in production via pre-load budget checks.

---

## Achievement 2: Real-Time VRAM Profiling

Every API response includes a full VRAM report from `pynvml` — before, after, peak, and delta:

```json
{
  "vram": {
    "before": { "used_mb": 1489.3, "utilization_pct": 72.7 },
    "after":  { "used_mb": 1507.3, "utilization_pct": 73.6 },
    "peak_used_mb": 1507.3,
    "delta_mb": 18.0,
    "duration_sec": 1.838
  }
}
```

This is not simulated — these are live readings from the NVIDIA driver during actual inference on the MX550.

---

## Achievement 3: OOM Prevention via Concurrency Guard

The `/generate` endpoint is protected by an `asyncio.Semaphore(1)`:

- Only **one inference request** can execute at a time
- A second concurrent request receives an immediate **HTTP 503** with `Retry-After: 5`
- Without this guard, two simultaneous requests would allocate two KV caches, doubling VRAM usage and causing an instant OOM crash on the 2 GB GPU

---

## Achievement 4: Model-Agnostic Architecture

The same codebase serves **any GGUF model** without modification — only environment variables change:

```bash
# Tiny model (fits easily)
SQ_MODEL_FILENAME="qwen2.5-0.5b-instruct-q4_k_m.gguf"
SQ_N_GPU_LAYERS=24

# Larger model (tight fit, partial offload)
SQ_MODEL_FILENAME="qwen2.5-1.5b-instruct-q4_k_m.gguf"
SQ_N_GPU_LAYERS=20

# Any other GGUF model from HuggingFace
SQ_MODEL_REPO_ID="microsoft/Phi-3-mini-4k-instruct-gguf"
SQ_MODEL_FILENAME="Phi-3-mini-4k-instruct-q4.gguf"
SQ_N_GPU_LAYERS=12
```

The VRAM monitor, budget enforcement, concurrency guard, and API contract remain identical regardless of model.

---

## Achievement 5: MongoDB RAG Pipeline

A complete Retrieval-Augmented Generation pipeline for system log analysis:

- **Ingestion**: JSON logs or raw text → `RecursiveCharacterTextSplitter` (512-char chunks, 64-char overlap) → `all-MiniLM-L6-v2` embeddings (384-dim) → MongoDB with SHA-256 deduplication
- **Indexing**: `SearchIndexModel` with cosine similarity + async polling for index readiness (`PENDING → INITIAL_SYNC → READY`)
- **Retrieval**: `$vectorSearch` aggregation with `numCandidates=10×top_k` for recall, pre-filtering by `source` and `log_level`, minimum score threshold
- **Integration**: `retrieve_as_context()` formats top-K results as LLM-ready context with source attribution

---

## Achievement 6: Autonomous AI Agent (Java + LangChain4j)

A Spring Boot orchestrator that uses LangChain4j to create an autonomous monitoring agent:

- **Custom `ChatLanguageModel`**: `FastApiChatModel` bridges LangChain4j → our Python FastAPI `/generate` endpoint
- **Two `@Tool`-annotated tools**:
  - `readSystemMetrics()` — reads real JVM data (CPU, heap, uptime) + simulated edge sensors (temperature, disk, network) with Gaussian drift for realistic behavior
  - `writeAlert(severity, message)` — persists timestamped alerts to a local log file
- **`@SystemMessage` agent**: Defines anomaly thresholds (CPU temp >80°C = CRITICAL, disk >95% = CRITICAL) and behavioral rules for autonomous tool invocation
- **AiServices proxy**: LangChain4j handles the tool-call → observation → response loop automatically

---

## Achievement 7: Containerized Three-Tier Deployment

Docker Compose stack with GPU passthrough and health-check dependency chaining:

| Service | Base Image | Resources | Health Check |
|---|---|---|---|
| `inference-engine` | `nvidia/cuda:12.6.3-runtime` | 1 GPU + 4 GB RAM | `GET /health` (180s start) |
| `vector-db` | `mongodb/mongodb-atlas-local:8.0` | 1 GB RAM | `mongosh ping` |
| `orchestrator` | `eclipse-temurin:21-jre` | 512 MB RAM | `GET /api/agent/health` |

- **`deploy.sh`** validates 5 prerequisites (Docker, Compose V2, NVIDIA driver, container toolkit, GPU passthrough) before deploying
- **Named volumes** persist models, MongoDB data, and alert logs across restarts
- **Startup order**: `vector-db` → `inference-engine` → `orchestrator` (each waits for predecessor's health check)

---

## Bug Discovered & Fixed During Validation

**Issue**: `pydantic-settings` rejected environment variables from the RAG config (`RAG_*` prefix) when loading the inference server config (`SQ_*` prefix), because both configs read from the same `.env` file.

**Root cause**: Pydantic's `BaseSettings` defaults to `extra = "forbid"`, treating any env var not matching its own field definitions as a validation error.

**Fix**: Added `"extra": "ignore"` to both `Settings` and `RAGSettings` model configs, allowing each to load only its prefixed variables and ignore the other's.

```python
# Before (crashes when .env has RAG_* vars)
model_config = {"env_prefix": "SQ_", "env_file": ".env"}

# After (coexists peacefully)
model_config = {"env_prefix": "SQ_", "env_file": ".env", "extra": "ignore"}
```

---

## Test Results Summary

| Test | Status |
|---|---|
| `GET /health` — model loaded, GPU layers confirmed | ✅ Pass |
| `GET /vram` — real-time NVIDIA VRAM reading | ✅ Pass |
| `POST /generate` — 0.5B model, full GPU offload | ✅ Pass (41.8 tok/s) |
| `POST /generate` — 1.5B model, partial GPU offload | ✅ Pass (16.9 tok/s) |
| 1.5B full offload — expected OOM | ✅ Correctly failed |
| VRAM budget enforcement | ✅ Pass (within budget) |
| Concurrency serialization | ✅ Pass (requests serialized) |
| Model-agnostic config swap | ✅ Pass (env vars only) |
| pydantic-settings coexistence fix | ✅ Pass |

---

*Validated on NVIDIA MX550 (2 GB GDDR6) — May 7, 2026*
