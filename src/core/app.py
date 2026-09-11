"""QApplication bootstrap."""

from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.ui.theme import STYLESHEET
from src.utils.logging_config import configure_logging


def run() -> int:
    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("Better Brush")
    app.setApplicationVersion("0.1.0")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    # Welcome launcher once the window is visible (blocking modal on purpose).
    QTimer.singleShot(200, window.show_welcome)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
