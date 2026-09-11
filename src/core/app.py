"""QApplication bootstrap, with a minimal splash while the app starts."""

from __future__ import annotations

import sys

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from src.ui.assets import logo_path
from src.ui.main_window import MainWindow
from src.ui.theme import STYLESHEET
from src.utils.logging_config import configure_logging, init_stdio


def _splash_pixmap() -> QPixmap:
    """Small, quiet splash: dark card, app logo, one loading line."""
    pix = QPixmap(300, 168)
    pix.fill(QColor(18, 18, 23))
    painter = QPainter(pix)

    logo = logo_path()
    if logo.is_file():
        icon = QPixmap(str(logo)).scaled(
            64, 64, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap((pix.width() - icon.width()) // 2, 22, icon)

    title_font = QFont()
    title_font.setPointSize(15)
    title_font.setBold(True)
    painter.setFont(title_font)
    painter.setPen(QColor(230, 230, 234))
    painter.drawText(0, 116, pix.width(), 32, Qt.AlignmentFlag.AlignCenter, "Better Brush")

    sub_font = QFont()
    sub_font.setPointSize(9)
    painter.setFont(sub_font)
    painter.setPen(QColor(154, 154, 164))
    painter.drawText(0, 142, pix.width(), 22, Qt.AlignmentFlag.AlignCenter, "Starting…")
    painter.end()
    return pix


def run() -> int:
    init_stdio()
    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("Better Brush")
    app.setApplicationVersion("0.1.0")
    app.setStyleSheet(STYLESHEET)

    splash = QSplashScreen(_splash_pixmap())
    try:
        splash.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        splash.show()
        app.processEvents()
    except Exception:  # noqa: BLE001 - a splash must never block startup
        splash = None

    window = MainWindow()
    window.show()

    if splash is not None:
        QTimer.singleShot(450, lambda: splash.finish(window))
    # Welcome launcher once the window is visible (blocking modal on purpose).
    QTimer.singleShot(200, window.show_welcome)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())