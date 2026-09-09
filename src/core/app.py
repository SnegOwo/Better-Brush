"""QApplication bootstrap."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.utils.logging_config import configure_logging


def run() -> int:
    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("Better Brush")
    app.setApplicationVersion("0.1.0")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
