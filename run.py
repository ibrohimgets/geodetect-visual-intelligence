#!/usr/bin/env python
"""
Start the application.

    python run.py

Then open http://127.0.0.1:8000 in a browser.

Pass --reload while developing to restart on file changes, and --port to move
off 8000 if something else is already there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Put backend/ on the import path so `app.main` resolves whichever directory
# you happen to launch this from.
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aerial detection and geo-referencing pipeline")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    import uvicorn

    print()
    print("  Aerial Detection & Geo-Referencing Pipeline")
    print(f"  ---> http://{args.host}:{args.port}")
    print(f"  API docs at http://{args.host}:{args.port}/docs")
    print()
    print("  First start downloads the YOLOv8 weights (~6 MB) if they are not")
    print("  cached in models/ yet, so give it a moment.")
    print()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
