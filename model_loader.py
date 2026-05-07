"""
SolQuant Inference Server — Model Loader
==========================================
Downloads (if needed) and loads the GGUF model into llama-cpp-python
with precise GPU layer offloading for the MX550 2GB VRAM budget.

Memory budget breakdown (MX550, 2048 MB total):
  ┌─────────────────────────────────────────┐
  │  CUDA driver + context overhead  ~200 MB│
  │  Model weights (Q4_K_M, 24 layers)~400 MB│
  │  KV cache (ctx=2048)            ~120 MB │
  │  Inference scratch buffers       ~80 MB │
  │  ─────────────────────────────────────  │
  │  Total estimated               ~800 MB  │
  │  Remaining headroom            ~1248 MB │
  └─────────────────────────────────────────┘
"""

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from config import settings
from vram_monitor import log_vram_status, check_vram_budget

logger = logging.getLogger("solquant.loader")

# Module-level singleton
_model: Llama | None = None


def _download_model() -> Path:
    """Download the GGUF model file from HuggingFace Hub if not cached."""
    model_dir = settings.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    local_path = model_dir / settings.model_filename

    if local_path.exists():
        size_mb = local_path.stat().st_size / (1024 ** 2)
        logger.info(f"Model already cached: {local_path} ({size_mb:.0f} MB)")
        return local_path

    logger.info(
        f"Downloading {settings.model_repo_id}/{settings.model_filename} ..."
    )
    downloaded = hf_hub_download(
        repo_id=settings.model_repo_id,
        filename=settings.model_filename,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    path = Path(downloaded)
    size_mb = path.stat().st_size / (1024 ** 2)
    logger.info(f"Download complete: {path} ({size_mb:.0f} MB)")
    return path


def load_model() -> Llama:
    """
    Load the model into llama-cpp-python with MX550-safe parameters.

    Key GPU memory controls:
      - n_gpu_layers: number of transformer layers offloaded to VRAM
      - n_ctx: context length (affects KV cache size in VRAM)
      - n_batch: prompt processing batch size (affects scratch buffer)
      - flash_attn: reduces KV cache memory when supported

    Returns:
        Loaded Llama model instance.

    Raises:
        RuntimeError: If VRAM budget is exceeded after loading.
    """
    global _model
    if _model is not None:
        logger.info("Model already loaded, returning cached instance")
        return _model

    # Pre-load VRAM snapshot
    log_vram_status("pre-load")

    model_path = _download_model()

    logger.info(
        f"Loading model with GPU offload: "
        f"n_gpu_layers={settings.n_gpu_layers}, "
        f"n_ctx={settings.n_ctx}, "
        f"n_batch={settings.n_batch}"
    )

    _model = Llama(
        model_path=str(model_path),
        n_gpu_layers=settings.n_gpu_layers,
        n_ctx=settings.n_ctx,
        n_batch=settings.n_batch,
        n_threads=settings.n_threads,
        # Memory optimizations for constrained VRAM
        use_mmap=True,          # memory-map weights (reduces RAM, GPU gets layers)
        use_mlock=False,        # don't lock pages — we rely on mmap
        verbose=True,           # log llama.cpp internals for debugging
    )

    # Post-load VRAM check
    snap = log_vram_status("post-load")
    if snap and snap.used_mb > settings.vram_budget_mb:
        logger.critical(
            f"VRAM BUDGET EXCEEDED after model load! "
            f"Used: {snap.used_mb:.0f} MB > Budget: {settings.vram_budget_mb} MB. "
            f"Reduce n_gpu_layers or n_ctx."
        )
        raise RuntimeError(
            f"Model loaded but exceeds VRAM budget "
            f"({snap.used_mb:.0f}/{settings.vram_budget_mb} MB). "
            f"Adjust SQ_N_GPU_LAYERS or SQ_N_CTX."
        )

    if not check_vram_budget(settings.vram_budget_mb):
        logger.warning("VRAM usage near budget limit after model load")

    logger.info("✓ Model loaded successfully within VRAM budget")
    return _model


def get_model() -> Llama:
    """Get the loaded model instance. Raises if not loaded."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model


def unload_model():
    """Release model resources."""
    global _model
    if _model is not None:
        del _model
        _model = None
        logger.info("Model unloaded")
        log_vram_status("post-unload")
