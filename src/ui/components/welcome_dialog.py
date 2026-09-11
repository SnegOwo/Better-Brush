"""Welcome screen shown on app start: recent projects + quick start."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.project import ProjectManager, format_relative_time
from src.ui.assets import logo_path

_LOGO_SIZE = 128


class WelcomeDialog(QDialog):
    """Minimal dark launcher with recent projects and quick actions.

    After ``exec()`` the chosen action is available on ``choice``:
      ``("blank", None)``            - start empty
      ``("import", Path)``           - import a video
      ``("open", Path)``             - open a saved .bbproj
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        recent: Optional[list[dict]] = None,
    ) -> None:
        super().__init__(parent)
        self.choice: tuple[str, Optional[Path]] = ("blank", None)

        self.setWindowTitle("Better Brush")
        self.setModal(True)
        self.setFixedWidth(560)
        self.setMinimumHeight(460)

        self._recent = list(recent or ProjectManager.load_recent())

        self._build_ui()
        if self._recent:
            self._fill_recent()

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 32, 40, 28)
        root.setSpacing(10)
        root.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Logo
        logo = QLabel()
        pix = logo_pixmap(_LOGO_SIZE)
        if not pix.isNull():
            logo.setPixmap(pix)
            logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            logo.setText("BB")
            logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo.setStyleSheet(
                "color: #4a9eff; font-size: 40px; font-weight: 700;"
            )
        root.addWidget(logo)

        title = QLabel("Better Brush")
        title.setStyleSheet(
            "font-size: 22px; font-weight: 700; color: #d4d4d8;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel("AI-powered rotoscoping")
        subtitle.setStyleSheet("color: #8b8b94; font-size: 12px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(subtitle)

        root.addSpacing(8)

        # Recent projects
        if self._recent:
            head = QLabel("RECENT PROJECTS")
            head.setStyleSheet(
                "color: #555560; font-size: 10px; font-weight: 600; "
                "letter-spacing: 1px;"
            )
            root.addWidget(head)
            self._list = QListWidget()
            self._list.setFixedHeight(150)
            self._list.setStyleSheet(
                "QListWidget { background: transparent; border: 1px solid #2e2e33;"
                " border-radius: 6px; padding: 4px; }"
                "QListWidget::item { padding: 6px 8px; border-radius: 4px; color: #d4d4d8; }"
                "QListWidget::item:hover { background: #28282c; }"
                "QListWidget::item:selected { background: #264f78; }"
            )
            self._list.itemClicked.connect(self._on_recent_clicked)
            root.addWidget(self._list)

        root.addSpacing(6)

        # Actions
        actions = QHBoxLayout()
        actions.setSpacing(8)

        self._btn_blank = QPushButton("Start Blank")
        self._btn_blank.setFixedHeight(34)
        self._btn_blank.clicked.connect(self._on_blank)
        actions.addWidget(self._btn_blank)

        self._btn_import = QPushButton("Import Video")
        self._btn_import.setFixedHeight(34)
        self._btn_import.setProperty("cssClass", "accent")
        self._btn_import.clicked.connect(self._on_import)
        actions.addWidget(self._btn_import)

        root.addLayout(actions)

        # Secondary: open an existing project file
        link_row = QHBoxLayout()
        self._btn_open_project = QPushButton("Open a saved project...")
        self._btn_open_project.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #4a9eff;"
            " font-size: 11px; padding: 2px; }"
            "QPushButton:hover { color: #60adff; }"
        )
        self._btn_open_project.clicked.connect(self._on_open_project)
        link_row.addWidget(self._btn_open_project, alignment=Qt.AlignmentFlag.AlignCenter)
        root.addLayout(link_row)

        # Thin divider + tutorial link
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(
            "background: #2e2e33; color: #2e2e33; max-height: 1px;"
        )
        root.addWidget(divider)

        self._btn_tutorial = QPushButton("Quick tutorial (first time here?)")
        self._btn_tutorial.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #8b8b94;"
            " font-size: 11px; padding: 2px; }"
            "QPushButton:hover { color: #d4d4d8; }"
        )
        self._btn_tutorial.clicked.connect(self._on_tutorial)
        root.addWidget(self._btn_tutorial, alignment=Qt.AlignmentFlag.AlignCenter)

    # ── Recent list ───────────────────────────────────────────────────────
    def _fill_recent(self) -> None:
        for entry in self._recent:
            name = entry.get("video_name") or Path(entry.get("project", "")).stem
            when = format_relative_time(entry.get("saved_at", ""))
            item = QListWidgetItem(f"{name}    {when}")
            item.setData(Qt.UserRole, entry.get("project"))
            self._list.addItem(item)

    def _on_recent_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            self.choice = ("open", Path(path))
            self.accept()

    # ── Actions ───────────────────────────────────────────────────────────
    def _on_blank(self) -> None:
        self.choice = ("blank", None)
        self.accept()

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)",
        )
        if not path:
            return
        self.choice = ("import", Path(path))
        self.accept()

    def _on_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Better Brush Project (*.bbproj);;All Files (*)"
        )
        if not path:
            return
        self.choice = ("open", Path(path))
        self.accept()

    def _on_tutorial(self) -> None:
        # Avoid circular import at module load time.
        from src.ui.components.tutorial_dialog import TutorialDialog

        TutorialDialog(self).exec()


def logo_pixmap(size: int = _LOGO_SIZE) -> QPixmap:
    """Scaled logo pixmap, or a null pixmap when the asset is missing."""
    from PySide6.QtCore import QSize

    pix = QPixmap(str(logo_path()))
    if pix.isNull():
        return pix
    return pix.scaled(
        QSize(int(size), int(size)),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )