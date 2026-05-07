"""
SolQuant Inference Server — Entry Point
========================================
Run with:  python main.py
   or:     uvicorn server:app --host 0.0.0.0 --port 8000
"""

import uvicorn
from config import settings


def main():
    uvicorn.run(
        "server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,  # disable in production
        workers=1,     # single worker — one model instance in VRAM
    )


if __name__ == "__main__":
    main()
