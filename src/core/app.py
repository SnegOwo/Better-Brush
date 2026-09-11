"""QApplication bootstrap with an instant, animated splash.

The splash is shown before any heavy (torch/torchvision/SAM2) import runs, then
those imports happen on a background thread while the splash animates. That way
the window appears on screen immediately after double-click, instead of after
several blank seconds.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

from src.ui.assets import logo_path
from src.ui.theme import STYLESHEET
from src.utils.logging_config import configure_logging, init_stdio

logger = logging.getLogger(__name__)

_BG = QColor(16, 16, 21)
_TRACK = QColor(46, 46, 54)
_FILL = QColor(0, 180, 240)
_TITLE = QColor(232, 232, 238)
_SUB = QColor(150, 150, 162)


class SplashScreen(QWidget):
    """Frameless, always-on-top loading card with a soft 'liquid' feel."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setFixedSize(360, 224)
        self._text = "Starting…"
        self._frac = -1.0  # <0 → indeterminate pulse

    def set_state(self, text: str, frac: float = -1.0) -> None:
        self._text = text
        self._frac = frac
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _BG)

        # Logo
        logo = logo_path()
        if logo.is_file():
            icon = QPixmap(str(logo)).scaled(
                72, 72, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap((self.width() - icon.width()) // 2, 28, icon)

        # Title
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(_TITLE)
        painter.drawText(
            0, 112, self.width(), 34, Qt.AlignmentFlag.AlignCenter, "Better Brush"
        )

        # Status
        sub_font = QFont()
        sub_font.setPointSize(9)
        painter.setFont(sub_font)
        painter.setPen(_SUB)
        painter.drawText(
            0, 146, self.width(), 20, Qt.AlignmentFlag.AlignCenter, self._text
        )

        # Progress bar: indeterminate pulse, or a filled fraction if known
        bar_w, bar_h = 220, 3
        bar_x = (self.width() - bar_w) // 2
        bar_y = 174
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_TRACK)
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 2, 2)
        painter.setBrush(_FILL)
        x = time.monotonic()
        if self._frac < 0:
            # Smooth left → right → left pulse (phase from wall clock)
            phase = (x % 1.6) / 1.6
            seg = bar_w // 3
            offset = int(phase * bar_w)
            painter.drawRoundedRect(bar_x + offset - seg, bar_y, seg, bar_h, 2, 2)
        else:
            painter.drawRoundedRect(
                bar_x, bar_y, int(bar_w * max(0.0, min(1.0, self._frac))), bar_h, 2, 2
            )
        painter.end()


def _load_heavy_deps() -> None:
    """Warm the model stack off the GUI thread while the splash animates."""
    import src.models.sam2_manager  # noqa: F401  (torch, SAM2)
    import src.processing.video_reader  # noqa: F401  (OpenCV)
    import src.hardware.detector  # noqa: F401     (torchvision, PyNvML)
    import src.export.exporter  # noqa: F401        (ffmpeg helper)


def run() -> int:
    init_stdio()
    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("Better Brush")
    app.setApplicationVersion("0.1.0")
    app.setStyleSheet(STYLESHEET)

    splash = SplashScreen()
    try:
        splash.show()
        app.processEvents()
    except Exception:  # noqa: BLE001 - a splash must never block startup
        splash = None

    if splash is not None:
        # Load the heavy stack in the background, animating meanwhile.
        deps = threading.Thread(target=_load_heavy_deps, daemon=True)
        deps.start()
        while deps.is_alive():
            if splash is not None:
                splash.set_state("Loading AI models…")
            app.processEvents()
            time.sleep(0.02)
        try:
            deps.join(timeout=30)
        except Exception:  # noqa: BLE001
            logger.warning("Background module load did not finish cleanly")

    # Warm imports are done — building the window is now fast.
    from src.ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    window.raise_()

    if splash is not None:
        # 'Liquid' handoff: fade the splash out over the now-visible window.
        def _fade(step: int = 0) -> None:
            if step >= 8:
                splash.close()
                return
            splash.setWindowOpacity(max(0.0, 1.0 - step / 8.0))
            splash.set_state("Welcome")
            QTimer.singleShot(35, lambda: _fade(step + 1))

        QTimer.singleShot(120, _fade)

    # Welcome launcher once the window is visible (blocking modal on purpose).
    QTimer.singleShot(200, window.show_welcome)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())