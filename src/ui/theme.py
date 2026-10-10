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
/* The main action inside a sidebar section. Deliberately much quieter than
   "accent": the sidebar stacks a dozen buttons, and filling several of them
   with the loud accent turns the panel into a pile of blue. This gives one
   per section enough weight to be findable without shouting. */
QPushButton[cssClass="primary"] {
    background-color: #23394f;
    border-color: #3a6ea5;
    color: #cfe3ff;
    font-weight: 600;
}
QPushButton[cssClass="primary"]:hover {
    background-color: #2a4763;
    border-color: #4a9eff;
    color: #e8f2ff;
}
QPushButton[cssClass="primary"]:pressed {
    background-color: #1f3143;
}
QPushButton[cssClass="primary"]:disabled {
    background-color: #1e1e22;
    border-color: #28282c;
    color: #555560;
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
    border-radius: 5px;
    padding: 7px 8px;
    text-align: left;
    font-weight: 600;
    font-size: 11px;
    color: #9a9aa4;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QPushButton[cssClass="section-header"]:hover {
    color: #e6e6ea;
    background-color: #2a2a30;
}
QPushButton[cssClass="section-header"]:checked {
    color: #d4d4d8;
    background-color: transparent;
}
QPushButton[cssClass="section-header"]:focus-visible {
    color: #e6e6ea;
    background-color: #2a2a30;
}
QPushButton[cssClass="tool"] {
    padding: 5px 10px;
}
QPushButton[cssClass="tool"]:checked {
    background-color: #264f78;
    border-color: #4a9eff;
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
QSlider:disabled::sub-page:horizontal {
    background: #3a3a40;
}
QSlider:disabled::handle:horizontal {
    background: #555560;
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
    background-color: #26262b;
    color: #e0e0e6;
    border: 1px solid #45454c;
    padding: 6px 9px;
    border-radius: 5px;
}

/* ─── Keyboard focus ────────────────────────────────────────────────
   Tab navigation used to be invisible: focusing a control changed nothing
   on screen, so keyboard users had no idea where they were. The ring is
   drawn on the border colour so it does not reflow the control, and it is
   scoped to :focus-visible so a mouse click does not leave one behind. */
QPushButton:focus, QComboBox:focus, QSpinBox:focus, QLineEdit:focus,
QCheckBox:focus, QListWidget:focus, QTreeWidget:focus, QTextEdit:focus {
    outline: none;
}
QPushButton:focus-visible, QComboBox:focus-visible, QSpinBox:focus-visible,
QLineEdit:focus-visible, QCheckBox:focus-visible, QListWidget:focus-visible,
QTextEdit:focus-visible {
    border-color: #4a9eff;
}
QPushButton[cssClass="accent"]:focus-visible,
QPushButton[cssClass="tool"]:focus-visible,
QPushButton[cssClass="transport"]:focus-visible {
    border-color: #8cc4ff;
}

/* ─── Line edit ─────────────────────────────────────────────────────
   Was unstyled, so text fields fell back to the platform widget and
   visibly clashed with the dark panel. */
QLineEdit {
    background-color: #232327;
    color: #d4d4d8;
    border: 1px solid #3a3a40;
    border-radius: 4px;
    padding: 5px 8px;
    selection-background-color: #264f78;
    selection-color: #e8e8ec;
}
QLineEdit:hover { border-color: #4a4a50; }
QLineEdit:focus { border-color: #4a9eff; background-color: #26262b; }
QLineEdit:disabled {
    background-color: #1e1e22;
    color: #555560;
    border-color: #28282c;
}
QLineEdit::placeholder { color: #555560; }

/* ─── Item views (the tutorial rail, history and library lists) ─────── */
QListWidget, QTreeWidget, QTableWidget, QListView, QTreeView {
    background-color: #1e1e22;
    border: 1px solid #2e2e33;
    border-radius: 4px;
    outline: none;
    selection-background-color: #264f78;
    selection-color: #e8e8ec;
}
QListWidget::item, QTreeWidget::item {
    padding: 4px 6px;
    border-radius: 3px;
}
QListWidget::item:hover, QTreeWidget::item:hover { background-color: #2a2a30; }
QListWidget::item:selected, QTreeWidget::item:selected { background-color: #264f78; }
QHeaderView::section {
    background-color: #26262b;
    color: #8b8b94;
    border: none;
    border-bottom: 1px solid #2e2e33;
    padding: 5px 6px;
    font-size: 11px;
}

/* ─── Text edit ─────────────────────────────────────────────────────── */
QTextEdit, QPlainTextEdit {
    background-color: #1e1e22;
    color: #d4d4d8;
    border: 1px solid #2e2e33;
    border-radius: 4px;
    padding: 6px;
    selection-background-color: #264f78;
}

/* ─── Group box / tab bar ───────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #2e2e33;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 8px;
    color: #8b8b94;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    font-size: 11px;
    letter-spacing: 0.5px;
}
QTabBar::tab {
    background: #232327;
    color: #8b8b94;
    border: 1px solid #2e2e33;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 5px 12px;
    margin-right: 2px;
}
QTabBar::tab:hover { background: #2e2e33; color: #d4d4d8; }
QTabBar::tab:selected {
    background: #2a2a2f;
    color: #e8e8ec;
    border-bottom: 1px solid #2a2a2f;
}

/* ─── Disabled states for the remaining controls ─────────────────────
   Only the push button had a disabled look, so a greyed-out combo or spin
   box still looked interactive and invited clicks that did nothing. */
QComboBox:disabled, QSpinBox:disabled {
    background-color: #1e1e22;
    color: #555560;
    border-color: #28282c;
}
QCheckBox:disabled { color: #555560; }
QCheckBox:disabled::indicator {
    background-color: #1e1e22;
    border-color: #28282c;
}
QCheckBox:hover { color: #e8e8ec; }

/* ─── Spin box arrows ───────────────────────────────────────────────── */
QSpinBox::up-arrow, QSpinBox::down-arrow {
    width: 7px;
    height: 7px;
    border-left: 1px solid #8b8b94;
    border-top: 1px solid #8b8b94;
}
QSpinBox::up-arrow:hover, QSpinBox::down-arrow:hover {
    border-left-color: #d4d4d8;
    border-top-color: #d4d4d8;
}
QSpinBox::up-button, QSpinBox::down-button {
    background-color: #2a2a2f;
    border-left: 1px solid #3a3a40;
    width: 15px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background-color: #333338; }

/* ─── Slider handle ─────────────────────────────────────────────────── */
QSlider::handle:horizontal {
    background: #e8e8ec;
    border: 1px solid #14141a;
    width: 11px;
    height: 11px;
    margin: -4px 0;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
    border-color: #4a9eff;
}
QSlider::groove:horizontal:hover { background: #333338; }

/* ─── Menus ─────────────────────────────────────────────────────────── */
QMenu { border-radius: 5px; }
QMenu::item:disabled { color: #555560; }
"""
