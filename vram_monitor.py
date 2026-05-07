"""
SolQuant Inference Server — VRAM Monitor
==========================================
Real-time NVIDIA GPU memory profiling using pynvml.
Provides pre/post inference snapshots and continuous monitoring.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

logger = logging.getLogger("solquant.vram")

# ── pynvml initialisation ───────────────────────────────────────────────────

_nvml_available = False
_handle = None

try:
    import pynvml

    pynvml.nvmlInit()
    # Use device 0 (MX550 is typically the only discrete GPU)
    _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _device_name = pynvml.nvmlDeviceGetName(_handle)
    if isinstance(_device_name, bytes):
        _device_name = _device_name.decode("utf-8")
    _nvml_available = True
    logger.info(f"VRAM monitor initialised — GPU: {_device_name}")
except Exception as e:
    logger.warning(f"pynvml unavailable ({e}). VRAM monitoring disabled.")


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class VRAMSnapshot:
    """A point-in-time VRAM reading."""

    timestamp: float
    total_mb: float
    used_mb: float
    free_mb: float
    utilization_pct: float

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "total_mb": round(self.total_mb, 1),
            "used_mb": round(self.used_mb, 1),
            "free_mb": round(self.free_mb, 1),
            "utilization_pct": round(self.utilization_pct, 1),
        }


@dataclass
class InferenceVRAMReport:
    """VRAM usage delta across a single inference call."""

    before: VRAMSnapshot
    after: VRAMSnapshot
    peak_used_mb: float
    delta_mb: float
    duration_sec: float

    def to_dict(self) -> dict:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "peak_used_mb": round(self.peak_used_mb, 1),
            "delta_mb": round(self.delta_mb, 1),
            "duration_sec": round(self.duration_sec, 3),
        }


# ── Public API ───────────────────────────────────────────────────────────────


def get_vram_snapshot() -> VRAMSnapshot | None:
    """Take a single VRAM reading. Returns None if NVML is unavailable."""
    if not _nvml_available or _handle is None:
        return None

    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(_handle)
        total_mb = mem_info.total / (1024 ** 2)
        used_mb = mem_info.used / (1024 ** 2)
        free_mb = mem_info.free / (1024 ** 2)
        util_pct = (used_mb / total_mb) * 100 if total_mb > 0 else 0.0

        return VRAMSnapshot(
            timestamp=time.time(),
            total_mb=total_mb,
            used_mb=used_mb,
            free_mb=free_mb,
            utilization_pct=util_pct,
        )
    except pynvml.NVMLError as e:
        logger.error(f"NVML read failed: {e}")
        return None


def check_vram_budget(budget_mb: float) -> bool:
    """Return True if current VRAM usage is within budget."""
    snap = get_vram_snapshot()
    if snap is None:
        logger.warning("Cannot verify VRAM budget — NVML unavailable")
        return True  # optimistic fallback

    within_budget = snap.used_mb <= budget_mb
    if not within_budget:
        logger.critical(
            f"VRAM OVER BUDGET! Used: {snap.used_mb:.0f} MB / "
            f"Budget: {budget_mb:.0f} MB"
        )
    return within_budget


def log_vram_status(label: str = "status") -> VRAMSnapshot | None:
    """Log current VRAM state with a human-readable label."""
    snap = get_vram_snapshot()
    if snap:
        logger.info(
            f"[VRAM {label}] "
            f"Used: {snap.used_mb:.0f}/{snap.total_mb:.0f} MB "
            f"({snap.utilization_pct:.1f}%) | "
            f"Free: {snap.free_mb:.0f} MB"
        )
    return snap


@contextmanager
def track_inference_vram() -> Generator[dict, None, None]:
    """
    Context manager that captures VRAM before/after an inference call.

    Usage:
        with track_inference_vram() as report_container:
            result = llm(prompt)
        report = report_container["report"]  # InferenceVRAMReport
    """
    container: dict = {"report": None}
    before = get_vram_snapshot()
    peak = before.used_mb if before else 0.0
    t_start = time.time()

    try:
        yield container
    finally:
        after = get_vram_snapshot()
        duration = time.time() - t_start

        if before and after:
            # The peak is approximated as max(before, after).
            # For true peak tracking we'd need a polling thread,
            # but for an MX550 with tight budgets this is sufficient.
            peak = max(before.used_mb, after.used_mb)
            delta = after.used_mb - before.used_mb

            report = InferenceVRAMReport(
                before=before,
                after=after,
                peak_used_mb=peak,
                delta_mb=delta,
                duration_sec=duration,
            )
            container["report"] = report

            level = logging.WARNING if peak > 1700 else logging.INFO
            logger.log(
                level,
                f"[VRAM inference] "
                f"Before: {before.used_mb:.0f} MB → "
                f"After: {after.used_mb:.0f} MB | "
                f"Δ: {delta:+.0f} MB | "
                f"Peak: {peak:.0f} MB | "
                f"Duration: {duration:.2f}s",
            )


def shutdown():
    """Clean up NVML resources."""
    if _nvml_available:
        try:
            pynvml.nvmlShutdown()
            logger.info("NVML shutdown complete")
        except pynvml.NVMLError:
            pass
