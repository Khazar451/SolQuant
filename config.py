"""
SolQuant Inference Server — Configuration
==========================================
All tunables for the MX550 2GB VRAM hard constraint.

The MX550 has 2048 MB GDDR6. We reserve ~200 MB for CUDA context and driver
overhead, leaving ~1848 MB for model weights + KV cache.

Qwen2.5-0.5B Q4_K_M ≈ 400 MB on disk, ~500 MB resident in VRAM with full
offload (24 layers).  This leaves ample room for KV cache even at ctx=2048.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Server configuration loaded from environment variables or .env file."""

    # ── Model identity ──────────────────────────────────────────────────
    model_repo_id: str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    model_filename: str = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    model_dir: Path = Path("./models")

    # ── GPU offload tuning ──────────────────────────────────────────────
    # Qwen2.5-0.5B has 24 transformer layers.
    # At Q4_K_M each layer is ~16 MB → 24 × 16 = ~384 MB for weights.
    # KV cache at ctx=2048 with GQA adds ~120 MB.
    # Total ≈ 504 MB — well within the 1848 MB budget.
    n_gpu_layers: int = 24  # offload ALL layers to GPU
    n_ctx: int = 2048       # context window (tokens)
    n_batch: int = 512      # prompt eval batch size
    n_threads: int = 4      # CPU threads for any remaining work

    # ── VRAM safety ─────────────────────────────────────────────────────
    vram_total_mb: int = 2048      # MX550 physical VRAM
    vram_budget_mb: int = 1600     # conservative budget (leave 448 MB headroom)
    vram_warn_threshold: float = 0.85  # warn at 85% of total VRAM

    # ── Generation defaults ─────────────────────────────────────────────
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop_tokens: list[str] = ["<|im_end|>", "<|endoftext|>"]

    # ── Server ──────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    model_config = {"env_prefix": "SQ_", "env_file": ".env"}


settings = Settings()
