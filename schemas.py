"""
SolQuant Inference Server — Request/Response Schemas
=====================================================
Pydantic models for the API contract.
"""

from pydantic import BaseModel, Field
from config import settings


# ── /generate endpoint ───────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    """Request body for text generation."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="The input prompt for the model.",
        examples=["Explain quantum computing in simple terms."],
    )
    max_tokens: int = Field(
        default=settings.max_tokens,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=settings.temperature,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 = deterministic.",
    )
    top_p: float = Field(
        default=settings.top_p,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold.",
    )
    top_k: int = Field(
        default=settings.top_k,
        ge=0,
        le=100,
        description="Top-K sampling. 0 = disabled.",
    )
    repeat_penalty: float = Field(
        default=settings.repeat_penalty,
        ge=0.0,
        le=3.0,
        description="Repetition penalty factor.",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response (SSE). Not yet implemented.",
    )
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt for chat-style formatting.",
    )


class GenerateResponse(BaseModel):
    """Response body for text generation."""

    text: str = Field(description="Generated text output.")
    tokens_generated: int = Field(description="Number of tokens produced.")
    tokens_prompt: int = Field(description="Number of tokens in the prompt.")
    generation_time_sec: float = Field(
        description="Wall-clock time for generation."
    )
    tokens_per_second: float = Field(
        description="Generation throughput (tokens/sec)."
    )
    vram: dict | None = Field(
        default=None,
        description="VRAM usage report for this inference call.",
    )


# ── /health endpoint ────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str
    gpu_layers: int
    context_size: int
    vram: dict | None = None


# ── /vram endpoint ──────────────────────────────────────────────────────────


class VRAMResponse(BaseModel):
    """Current VRAM status."""

    available: bool
    snapshot: dict | None = None
    budget_mb: int
    within_budget: bool
