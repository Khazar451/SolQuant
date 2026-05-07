# SolQuant Inference Server

A production-grade REST API for serving extremely quantized Small Language Models (SLMs) on constrained NVIDIA GPUs. Designed for low VRAM GPUs (2-4 GB) and tested on the **NVIDIA MX550 (2 GB GDDR6 VRAM)**.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  FastAPI Server (server.py)                     │
│  ├── POST /generate  ← text generation + VRAM  │
│  ├── GET  /health    ← model & GPU status       │
│  └── GET  /vram      ← real-time VRAM monitor   │
├─────────────────────────────────────────────────┤
│  Model Loader (model_loader.py)                 │
│  └── HuggingFace download → llama-cpp-python    │
├─────────────────────────────────────────────────┤
│  VRAM Monitor (vram_monitor.py)                 │
│  └── pynvml snapshots + inference tracking      │
├─────────────────────────────────────────────────┤
│  Config (config.py)                             │
│  └── pydantic-settings + env vars (SQ_ prefix)  │
└─────────────────────────────────────────────────┘
```

## VRAM Budget (MX550 — 2048 MB)

| Component | Estimated Usage |
|---|---|
| CUDA driver + context | ~200 MB |
| Qwen2.5-0.5B Q4_K_M weights (24 layers) | ~400 MB |
| KV cache (ctx=2048) | ~120 MB |
| Inference scratch buffers | ~80 MB |
| **Total** | **~800 MB** |
| **Headroom remaining** | **~1248 MB** |

## Quick Start

### 1. Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA toolkit installed
- `nvidia-smi` working

### 2. Install

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (CUDA-enabled llama-cpp-python)
CMAKE_ARGS="-DGGML_CUDA=on" pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env if needed (defaults are tuned for MX550)
```

### 4. Run

```bash
python main.py
# or
uvicorn server:app --host 0.0.0.0 --port 8000
```

The model will be auto-downloaded from HuggingFace on first run (~350 MB).

### 5. Test

```bash
# Health check
curl http://localhost:8000/health | python -m json.tool

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in one paragraph.",
    "max_tokens": 256,
    "temperature": 0.7
  }' | python -m json.tool

# VRAM status
curl http://localhost:8000/vram | python -m json.tool
```

## API Reference

### `POST /generate`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | *required* | Input text |
| `max_tokens` | int | 512 | Max tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `top_k` | int | 40 | Top-K sampling |
| `repeat_penalty` | float | 1.1 | Repetition penalty |
| `system_prompt` | string | "You are a helpful assistant." | System instruction |

**Response** includes `text`, `tokens_generated`, `tokens_per_second`, and a `vram` object with before/after/peak memory usage.

### `GET /health`

Returns model status, GPU layer count, context size, and current VRAM snapshot.

### `GET /vram`

Returns current VRAM utilization and whether usage is within the configured budget.

## Tuning for OOM Safety

If you encounter OOM errors, adjust these in `.env`:

```bash
# Reduce GPU layers (moves layers to CPU — slower but less VRAM)
SQ_N_GPU_LAYERS=18

# Reduce context window (shrinks KV cache)
SQ_N_CTX=1024

# Reduce batch size
SQ_N_BATCH=256
```

## Project Structure

```
SolQuant/
├── main.py            # Entry point (uvicorn launcher)
├── server.py          # FastAPI app + endpoints
├── model_loader.py    # Model download + llama-cpp loading
├── vram_monitor.py    # NVIDIA VRAM profiling
├── config.py          # Settings (env vars / .env)
├── schemas.py         # Pydantic request/response models
├── requirements.txt   # Python dependencies
├── .env.example       # Environment template
└── models/            # Downloaded GGUF files (gitignored)
```

## License

Internal use — SolQuant Project.
