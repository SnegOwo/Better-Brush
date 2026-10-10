"""Styled export progress dialog with percentage, ETA, and cancel button."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class ExportProgressDialog(QDialog):
    """Modal progress dialog for export tasks.

    Displays a styled progress bar with percentage, frame count, ETA,
    and a Cancel button.  Follows the app's dark theme.
    """

    cancel_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Exporting…")
        self.setModal(True)
        self.setFixedSize(380, 170)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        self._cancelled = False

        self.setStyleSheet(
            "QDialog { background: #1e1e22; border: 1px solid #333; border-radius: 6px; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 14)
        root.setSpacing(10)

        # ── Title ─────────────────────────────────────────────────────────
        self._lbl_title = QLabel("Exporting…")
        self._lbl_title.setStyleSheet(
            "color: #e6e6ea; font-size: 14px; font-weight: 600; background: transparent;"
        )
        root.addWidget(self._lbl_title)

        # ── Stage + frame count ───────────────────────────────────────────
        self._lbl_stage = QLabel("")
        self._lbl_stage.setStyleSheet(
            "color: #9a9aa4; font-size: 11px; background: transparent;"
        )
        root.addWidget(self._lbl_stage)

        # ── Progress bar ──────────────────────────────────────────────────
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._bar.setStyleSheet(
            "QProgressBar { background: #2a2a30; border: none; border-radius: 4px; }"
            "QProgressBar::chunk { background: #3b82f6; border-radius: 4px; }"
        )
        root.addWidget(self._bar)

        # ── Info row (percent + ETA) ──────────────────────────────────────
        info = QHBoxLayout()
        info.setSpacing(0)

        self._lbl_pct = QLabel("0 %")
        self._lbl_pct.setStyleSheet(
            "color: #e6e6ea; font-size: 22px; font-weight: 700; background: transparent;"
        )
        info.addWidget(self._lbl_pct)
        info.addStretch()

        eta_col = QVBoxLayout()
        eta_col.setSpacing(0)
        self._lbl_elapsed = QLabel("Elapsed  --")
        self._lbl_elapsed.setStyleSheet(
            "color: #6b6b74; font-size: 10px; background: transparent;"
        )
        eta_col.addWidget(self._lbl_elapsed, alignment=Qt.AlignmentFlag.AlignRight)
        self._lbl_eta = QLabel("Remaining  --")
        self._lbl_eta.setStyleSheet(
            "color: #6b6b74; font-size: 10px; background: transparent;"
        )
        eta_col.addWidget(self._lbl_eta, alignment=Qt.AlignmentFlag.AlignRight)
        info.addLayout(eta_col)
        root.addLayout(info)

        # ── Cancel button ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setFixedWidth(80)
        self._btn_cancel.setStyleSheet(
            "QPushButton { background: #2e2e33; color: #d0d0d6; border: none;"
            "  border-radius: 4px; padding: 5px 12px; font-size: 11px; }"
            "QPushButton:hover { background: #3a3a40; }"
            "QPushButton:pressed { background: #45454d; }"
        )
        self._btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._btn_cancel)
        root.addLayout(btn_row)

    # ── Public API ─────────────────────────────────────────────────────────

    def update_progress(
        self,
        current: int,
        total: int,
        stage: str = "",
        elapsed: float = 0.0,
    ) -> None:
        """Update the bar and labels with new progress information."""
        pct = int(current / total * 100) if total > 0 else 0
        self._bar.setValue(pct)
        self._lbl_pct.setText(f"{pct} %")

        # Stage + frame count
        parts: list[str] = []
        if stage:
            parts.append(stage)
        if total > 0:
            parts.append(f"{current:,} / {total:,} frames")
        self._lbl_stage.setText("  ·  ".join(parts))

        # Elapsed
        mins, secs = divmod(int(elapsed), 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            self._lbl_elapsed.setText(f"Elapsed  {hrs}h {mins}m {secs}s")
        else:
            self._lbl_elapsed.setText(f"Elapsed  {mins}m {secs}s")

        # ETA
        if current > 0 and total > 0:
            eta_secs = elapsed * (total - current) / current
            eta_mins, eta_secs_i = divmod(int(eta_secs), 60)
            eta_hrs, eta_mins = divmod(eta_mins, 60)
            if eta_hrs > 0:
                self._lbl_eta.setText(
                    f"Remaining  {eta_hrs}h {eta_mins}m {eta_secs_i}s"
                )
            else:
                self._lbl_eta.setText(f"Remaining  {eta_mins}m {eta_secs_i}s")
        else:
            self._lbl_eta.setText("Remaining  —")

    def set_finished(self, success: bool = True, cancelled: bool = False) -> None:
        """Mark the export as finished and auto-close.

        A cancelled run is not a failure: it keeps its last progress value and
        says so, so the dialog never claims "Export failed" for a run the user
        stopped on purpose.
        """
        if cancelled:
            self._lbl_title.setText("Export cancelled")
            self._lbl_stage.setText("")
            self._lbl_eta.setText("")
        else:
            if success:
                self._lbl_title.setText("Export complete")
                self._lbl_stage.setText("")
                self._lbl_eta.setText("")
            else:
                self._lbl_title.setText("Export failed")
            self._bar.setValue(100 if success else 0)
        self._btn_cancel.setEnabled(True)
        self._btn_cancel.setText("Close")
        self._btn_cancel.clicked.disconnect()
        self._btn_cancel.clicked.connect(self.accept)

    # ── Internals ──────────────────────────────────────────────────────────

    def _on_cancel(self) -> None:
        self._cancelled = True
        self._btn_cancel.setEnabled(False)
        self._lbl_stage.setText("Cancelling…")
        self.cancel_requested.emit()

    @property
    def cancelled(self) -> bool:
        return self._cancelled
