"""``python -m src.engine`` starts the headless engine server."""

from __future__ import annotations

from src.engine.server import main

if __name__ == "__main__":
    raise SystemExit(main())
