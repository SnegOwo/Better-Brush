"""Professional dark theme for Better Brush.

Color palette inspired by DaVinci Resolve / Premiere Pro.
No purple, no gradients, no AI SaaS vibes.
"""

from __future__ import annotations

# ── Color tokens ──────────────────────────────────────────────────────────
BG_DARKEST = "#111114"
BG_DARK = "#1a1a1e"
BG_CANVAS = "#141418"
BG_PANEL = "#202024"
BG_PANEL_HEADER = "#28282c"
BG_CONTROL = "#2a2a2f"
BG_CONTROL_HOVER = "#333338"
BG_CONTROL_ACTIVE = "#3d3d44"
BG_SELECTED = "#264f78"

TEXT_PRIMARY = "#d4d4d8"
TEXT_SECONDARY = "#8b8b94"
TEXT_MUTED = "#555560"

ACCENT = "#4a9eff"
ACCENT_HOVER = "#60adff"
POSITIVE = "#4ade80"
NEGATIVE = "#f87171"
WARNING = "#f59e0b"
INFO = "#38bdf8"

BORDER = "#2e2e33"
BORDER_LIGHT = "#3a3a40"

MASK_RGBA = (0, 180, 240, 110)
PLAYHEAD = "#ef4444"
KEYFRAME = "#f59e0b"
MASKED_BAND = "#2563eb"

# ── Stylesheet ────────────────────────────────────────────────────────────
STYLESHEET = r"""
/* ── Base ──────────────────────────────────────────────────────────────── */
QMainWindow {
    background-color: #1a1a1e;
    color: #d4d4d8;
}
QWidget {
    color: #d4d4d8;
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", sans-serif;
    font-size: 12px;
}

/* ── Menu bar ──────────────────────────────────────────────────────────── */
QMenuBar {
    background-color: #1a1a1e;
    border-bottom: 1px solid #2e2e33;
    padding: 1px 2px;
}
QMenuBar::item {
    background: transparent;
    padding: 4px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #2a2a2f;
}
QMenu {
    background-color: #222226;
    border: 1px solid #2e2e33;
    padding: 4px 0px;
}
QMenu::item {
    padding: 5px 28px 5px 12px;
}
QMenu::item:selected {
    background-color: #35354a;
}
QMenu::separator {
    height: 1px;
    background: #2e2e33;
    margin: 4px 8px;
}

/* ── Buttons ───────────────────────────────────────────────────────────── */
QPushButton {
    background-color: #2a2a2f;
    color: #d4d4d8;
    border: 1px solid #3a3a40;
    border-radius: 4px;
    padding: 5px 12px;
    min-height: 18px;
}
QPushButton:hover {
    background-color: #333338;
    border-color: #4a4a50;
}
QPushButton:pressed {
    background-color: #3d3d44;
}
QPushButton:checked {
    background-color: #264f78;
    border-color: #4a9eff;
    color: #e8e8ec;
}
QPushButton:disabled {
    background-color: #1e1e22;
    color: #555560;
    border-color: #28282c;
}
QPushButton[cssClass="accent"] {
    background-color: #264f78;
    border-color: #4a9eff;
    color: #e8e8ec;
}
QPushButton[cssClass="accent"]:hover {
    background-color: #2d5f8f;
}
QPushButton[cssClass="danger"] {
    background-color: #3a1a1a;
    border-color: #f87171;
    color: #f87171;
}
QPushButton[cssClass="danger"]:hover {
    background-color: #4a2020;
}
QPushButton[cssClass="section-header"] {
    background-color: transparent;
    border: none;
    border-radius: 0px;
    padding: 6px 8px;
    text-align: left;
    font-weight: 600;
    font-size: 11px;
    color: #8b8b94;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QPushButton[cssClass="section-header"]:hover {
    color: #b0b0b8;
    background-color: transparent;
}
QPushButton[cssClass="section-header"]:checked {
    color: #d4d4d8;
    background-color: transparent;
}
QPushButton[cssClass="transport"] {
    background-color: #222226;
    border: 1px solid #2e2e33;
    border-radius: 4px;
    padding: 4px 10px;
    min-width: 32px;
}
QPushButton[cssClass="transport"]:hover {
    background-color: #2a2a2f;
    border-color: #3a3a40;
}
QPushButton[cssClass="transport"]:checked {
    background-color: #1a3a5c;
    border-color: #4a9eff;
}

/* ── Combo box ─────────────────────────────────────────────────────────── */
QComboBox {
    background-color: #2a2a2f;
    color: #d4d4d8;
    border: 1px solid #3a3a40;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 20px;
}
QComboBox:hover {
    border-color: #4a4a50;
}
QComboBox:focus {
    border-color: #4a9eff;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #222226;
    color: #d4d4d8;
    border: 1px solid #2e2e33;
    selection-background-color: #35354a;
    outline: none;
    padding: 2px;
}

/* ── Spin box ──────────────────────────────────────────────────────────── */
QSpinBox {
    background-color: #2a2a2f;
    color: #d4d4d8;
    border: 1px solid #3a3a40;
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 18px;
}
QSpinBox:hover {
    border-color: #4a4a50;
}
QSpinBox:focus {
    border-color: #4a9eff;
}
QSpinBox::up-button, QSpinBox::down-button {
    background-color: #2a2a2f;
    border: none;
    width: 14px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #333338;
}

/* ── Check box ─────────────────────────────────────────────────────────── */
QCheckBox {
    spacing: 6px;
    color: #d4d4d8;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #4a4a50;
    border-radius: 3px;
    background-color: #2a2a2f;
}
QCheckBox::indicator:checked {
    background-color: #4a9eff;
    border-color: #4a9eff;
}
QCheckBox::indicator:hover {
    border-color: #60adff;
}

/* ── Labels ────────────────────────────────────────────────────────────── */
QLabel {
    background: transparent;
    color: #d4d4d8;
}
QLabel[color="secondary"] {
    color: #8b8b94;
}
QLabel[color="muted"] {
    color: #555560;
}
QLabel[color="accent"] {
    color: #4a9eff;
}

/* ── Splitter ──────────────────────────────────────────────────────────── */
QSplitter::handle {
    background-color: #2e2e33;
}
QSplitter::handle:horizontal {
    width: 1px;
}
QSplitter::handle:vertical {
    height: 1px;
}
QSplitter::handle:hover {
    background-color: #4a9eff;
}

/* ── Status bar ────────────────────────────────────────────────────────── */
QStatusBar {
    background-color: #171719;
    color: #8b8b94;
    border-top: 1px solid #2e2e33;
    font-size: 11px;
    padding: 2px 8px;
}
QStatusBar::item {
    border: none;
}

/* ── Scroll bars ───────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: #3a3a40;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background-color: #505058;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background-color: #3a3a40;
    border-radius: 4px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #505058;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}

/* ── Slider ────────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    background: #2a2a2f;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #d4d4d8;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
}
QSlider::sub-page:horizontal {
    background: #4a9eff;
    border-radius: 2px;
}

/* ── Message box / dialogs ─────────────────────────────────────────────── */
QMessageBox, QDialog {
    background-color: #222226;
}
QMessageBox QLabel, QDialog QLabel {
    color: #d4d4d8;
}

/* ── Tooltips ──────────────────────────────────────────────────────────── */
QToolTip {
    background-color: #2a2a2f;
    color: #d4d4d8;
    border: 1px solid #3a3a40;
    padding: 4px 8px;
    border-radius: 4px;
}
"""
