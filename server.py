"""
SolQuant Inference Server — FastAPI Application
=================================================
REST API for serving quantized SLMs on constrained NVIDIA GPUs.

Endpoints:
  POST /generate  — Text generation with VRAM profiling
  GET  /health    — Health check + model status
  GET  /vram      — Current VRAM utilization

Target hardware: NVIDIA MX550 (2 GB GDDR6)
Target model:    Qwen2.5-0.5B-Instruct (Q4_K_M)
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from model_loader import load_model, get_model, unload_model
from schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    VRAMResponse,
)
from vram_monitor import (
    get_vram_snapshot,
    log_vram_status,
    check_vram_budget,
    track_inference_vram,
    shutdown as vram_shutdown,
)

# ── Logging setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("solquant.server")


# ── Application lifespan ────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    logger.info("=" * 60)
    logger.info("SolQuant Inference Server starting")
    logger.info(f"  Model:      {settings.model_repo_id}")
    logger.info(f"  Quant:      {settings.model_filename}")
    logger.info(f"  GPU layers: {settings.n_gpu_layers}")
    logger.info(f"  Context:    {settings.n_ctx} tokens")
    logger.info(f"  VRAM budget:{settings.vram_budget_mb} MB")
    logger.info("=" * 60)

    try:
        load_model()
    except RuntimeError as e:
        logger.critical(f"Failed to load model: {e}")
        raise

    yield

    logger.info("Shutting down...")
    unload_model()
    vram_shutdown()


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="SolQuant Inference Server",
    description=(
        "Low-VRAM inference server for quantized Small Language Models. "
        "Designed for NVIDIA MX550 (2 GB VRAM) using llama-cpp-python."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Concurrency guard ────────────────────────────────────────────────────────
# llama-cpp-python is NOT thread-safe and concurrent inference would allocate
# a second KV cache, instantly doubling VRAM usage → OOM on a 2 GB GPU.
# A semaphore(1) serialises all /generate calls.  We use try_acquire so that
# a second request gets an immediate 503 instead of silently queuing.

_inference_semaphore = asyncio.Semaphore(1)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate text from a prompt",
    tags=["Inference"],
)
async def generate(req: GenerateRequest):
    """
    Generate text using the loaded SLM.

    The prompt is wrapped in Qwen's ChatML format:
      <|im_start|>system\n{system_prompt}<|im_end|>
      <|im_start|>user\n{prompt}<|im_end|>
      <|im_start|>assistant

    Concurrency: Only one inference runs at a time. If the model is already
    busy, the caller receives HTTP 503 immediately (no queuing).

    VRAM is profiled before and after inference. If usage exceeds the
    configured budget, a warning is logged but the response is still returned.
    """
    # ── Acquire inference lock (non-blocking) ───────────────────────
    if _inference_semaphore.locked():
        logger.warning("Rejected /generate — inference already in progress")
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is busy processing another request. "
                "Concurrent inference is disabled to prevent GPU OOM. "
                "Please retry shortly."
            ),
            headers={"Retry-After": "5"},
        )

    async with _inference_semaphore:
        llm = get_model()

        # ── Format prompt in ChatML (Qwen2.5 native format) ─────────
        formatted_prompt = (
            f"<|im_start|>system\n{req.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{req.prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        logger.info(
            f"Generating: max_tokens={req.max_tokens}, "
            f"temp={req.temperature}, prompt_len={len(req.prompt)} chars"
        )

        # ── Pre-inference VRAM check ────────────────────────────────
        if not check_vram_budget(settings.vram_budget_mb):
            logger.warning(
                "VRAM near/over budget before inference — proceeding anyway"
            )

        # ── Run inference with VRAM tracking ────────────────────────
        vram_report = None
        t_start = time.perf_counter()

        with track_inference_vram() as vram_container:
            try:
                result = llm(
                    formatted_prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    repeat_penalty=req.repeat_penalty,
                    stop=settings.stop_tokens,
                    echo=False,
                )
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Inference error: {str(e)}",
                )

        t_elapsed = time.perf_counter() - t_start

        # ── Extract results ─────────────────────────────────────────
        generated_text = result["choices"][0]["text"].strip()
        usage = result.get("usage", {})
        tokens_prompt = usage.get("prompt_tokens", 0)
        tokens_generated = usage.get("completion_tokens", 0)
        tps = tokens_generated / t_elapsed if t_elapsed > 0 else 0.0

        # ── Build VRAM report ───────────────────────────────────────
        if vram_container.get("report"):
            vram_report = vram_container["report"].to_dict()

        logger.info(
            f"Done: {tokens_generated} tokens in {t_elapsed:.2f}s "
            f"({tps:.1f} tok/s)"
        )

        return GenerateResponse(
            text=generated_text,
            tokens_generated=tokens_generated,
            tokens_prompt=tokens_prompt,
            generation_time_sec=round(t_elapsed, 3),
            tokens_per_second=round(tps, 1),
            vram=vram_report,
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health():
    """Check server health, model status, and VRAM utilization."""
    model_loaded = False
    try:
        get_model()
        model_loaded = True
    except RuntimeError:
        pass

    snap = get_vram_snapshot()

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=settings.model_filename,
        gpu_layers=settings.n_gpu_layers,
        context_size=settings.n_ctx,
        vram=snap.to_dict() if snap else None,
    )


@app.get(
    "/vram",
    response_model=VRAMResponse,
    summary="VRAM utilization",
    tags=["System"],
)
async def vram_status():
    """Get current GPU VRAM usage and budget status."""
    snap = get_vram_snapshot()
    within_budget = check_vram_budget(settings.vram_budget_mb)

    return VRAMResponse(
        available=snap is not None,
        snapshot=snap.to_dict() if snap else None,
        budget_mb=settings.vram_budget_mb,
        within_budget=within_budget,
    )
