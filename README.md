 SolQuant
SolQuant is a three-tier AI edge stack for low-VRAM systems, combining:
- a **Python FastAPI inference engine** (llama-cpp + GGUF),
- a **MongoDB Atlas Local vector store** for RAG,
- and a **Java Spring Boot agent orchestrator** (LangChain4j).
The project is tuned for constrained NVIDIA hardware (tested on MX550-class GPUs) and is designed for system monitoring and alert-driven agent workflows.
## Architecture
```text
┌──────────────────────────────────────────────────────────────┐
│ Orchestrator (Spring Boot, Java 21)                         │
│ - POST /api/agent/chat                                      │
│ - GET  /api/agent/health                                    │
│ Uses LangChain4j + tools for metrics and alerting           │
└───────────────┬──────────────────────────────────────────────┘
                │ HTTP
                ▼
┌──────────────────────────────────────────────────────────────┐
│ Inference Engine (FastAPI, Python)                          │
│ - POST /generate                                             │
│ - GET  /health                                               │
│ - GET  /vram                                                 │
│ Loads quantized GGUF model via llama-cpp-python             │
└───────────────┬──────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ Vector DB (MongoDB Atlas Local)                             │
│ Stores embeddings and supports vector search for RAG         │
└──────────────────────────────────────────────────────────────┘
```
## Repository Structure
```text
.
├── agent-controller/       # Spring Boot orchestrator
├── rag/                    # RAG ingestion, embeddings, retrieval, DB helpers
├── server.py               # FastAPI app and endpoints
├── model_loader.py         # Model download/load lifecycle
├── vram_monitor.py         # NVML-based VRAM monitoring
├── schemas.py              # API schemas
├── config.py               # Inference config (SQ_*)
├── docker-compose.yml      # Full stack orchestration
├── deploy.sh               # Guided deployment script
└── .env.example            # Environment template
```
## Core Components
### 1) Inference Engine (Python)
- FastAPI API for text generation and VRAM observability.
- Automatic GGUF model download from Hugging Face.
- Concurrency protection to avoid multi-request GPU OOM.
- Tuned defaults for low-VRAM operation.
Endpoints:
- `POST /generate`
- `GET /health`
- `GET /vram`
### 2) Agent Controller (Java / LangChain4j)
- Exposes an AI monitoring agent over HTTP.
- Bridges to the Python inference engine through a custom chat model.
- Includes tool-enabled workflows:
  - `readSystemMetrics` (CPU/RAM/disk/temp report)
  - `writeAlert` (INFO/WARNING/CRITICAL alert logging)
Endpoints:
- `POST /api/agent/chat`
- `GET /api/agent/health`
### 3) RAG Pipeline (Python)
- Log ingestion and chunking.
- Embedding generation (all-MiniLM-L6-v2).
- MongoDB vector index creation and retrieval utilities.
## Quick Start (Recommended: Docker)
### Prerequisites
- Docker Engine + Docker Compose v2
- NVIDIA drivers installed
- NVIDIA Container Toolkit configured
- `nvidia-smi` working
### Deploy
```bash
chmod +x deploy.sh
./deploy.sh
```
Useful commands:
```bash
./deploy.sh --status   # service status + endpoints
./deploy.sh --logs     # stream logs
./deploy.sh --build    # rebuild images without cache
./deploy.sh --down     # stop stack
```
## Local Run (Inference Engine Only)
```bash
python -m venv .venv
source .venv/bin/activate
CMAKE_ARGS="-DGGML_CUDA=on" pip install -r requirements.txt
cp .env.example .env
python main.py
```
## API Examples
### Inference Health
```bash
curl http://localhost:8000/health
```
### Text Generation
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize current edge system status.",
    "max_tokens": 256,
    "temperature": 0.3
  }'
```
### Agent Chat
```bash
curl -X POST http://localhost:8081/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"How healthy is the system right now?"}'
```
## Configuration
Copy `.env.example` to `.env` and adjust values as needed.
Key groups:
- `SQ_*` — inference model/runtime settings
- `RAG_*` — MongoDB + embedding + retrieval settings
- `SOLQUANT_*` — orchestrator-to-inference integration settings (Docker/env)
## Development Notes
- Java module targets **Java 21** (`agent-controller/pom.xml`).
- Maven tests/build for `agent-controller` require a Java 21 toolchain.
- Inference defaults are optimized for low-VRAM cards; reduce `SQ_N_GPU_LAYERS`, `SQ_N_CTX`, or `SQ_N_BATCH` if you encounter OOM.
## License
Internal use — SolQuant project.
