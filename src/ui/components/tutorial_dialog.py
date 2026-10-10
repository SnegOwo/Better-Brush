"""Quick step-by-step tutorial dialog for first-time users."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_STEPS = [
    (
        "1 · Import a video",
        "Drag and drop a video onto the window, or use File → Open Video (Ctrl+O).",
    ),
    (
        "2 · Choose a Preset (AMV / Human / Object)",
        (
            "In the Content panel, pick a preset that matches your footage:\n"
            "• AMV — anime / cartoon outlines, emphasises hard edges.\n"
            "• Human — faces and people, preserves skin detail.\n"
            "• Object — general objects, good starting point for anything."
        ),
    ),
    (
        "3 · Add points on your subject",
        (
            "With the Points tool, left-click places a green marker (include) on the "
            "subject you want to cut out. Right-click places a red marker (exclude) "
            "on stuff SAM 2 should ignore."
        ),
    ),
    (
        "4 · Create the first mask",
        (
            "Press Segment Frame. SAM 2 turns your points into a crisp alpha matte "
            "for the current frame."
        ),
    ),
    (
        "5 · Propagate across the timeline",
        (
            "Use the Prop buttons (◀ Prop / Prop ▶) to step one frame at a time, "
            "or click Propagate to End to extend the mask through the entire video. "
            "Post-process masks then cleans edges and reduces flicker; adjust Edge "
            "smoothness or press Apply to re-run the current shot. If the sky is included, "
            "right-click it before re-segmenting. Per-shot profiles can override these settings."
        ),
    ),
    (
        "6 · Fix edges and color spill",
        (
            "Open Inspect → Shot Profiles (Ctrl+Shift+P) for the current shot.\n"
            "Leave Edge softness at 2 — it anti-aliases the exported matte, which "
            "is what stops edges looking like a staircase, and removes the "
            "background colour from edge pixels.\n"
            "For a colour fringe, set Spill cleaner to Despill or Alpha key, press "
            "Pick, then click the fringe on the preview inside the dialog. Press Set "
            "Area and drag a box to clean only part of frame, or Protect to shield a "
            "box — use that when something on the subject is genuinely the spill "
            "colour, like blue hair under a blue sky.\n"
            "Tick Fix colour cast only if the tint is unwanted light; it will "
            "neutralise an intentional backdrop, and needs a small Area box over the "
            "background to work from."
        ),
    ),
    (
        "7 · Refine with the brush",
        (
            "Press B to switch to the Brush tool: paint to add to the mask. "
            "Hold Shift while painting to erase. Use [ and ] to resize the brush. "
            "Space or P plays and pauses the clip."
        ),
    ),
    (
        "8 · Export and save",
        (
            "Ctrl+E opens the export dialog — H.264 MP4 with a solid "
            "background (green-screen, blue, etc.) is great for compositing. "
            "Ctrl+S saves your project, keeping every point and mask for later."
        ),
    ),
    (
        "9 · Library & recode",
        (
            "File → Media Library (Ctrl+L) shows all your exports. "
            "Right-click any video to recode it to a different format (MP4 → MOV ProRes, "
            "WebM, etc.) without re-rendering the mask."
        ),
    ),
    (
        "10 · Inspect matte quality",
        (
            "Open Inspect → Matte QC Inspector (Ctrl+Shift+M) after creating a mask. "
            "The left pane shows the untouched source frame; the right pane shows the current "
            "matte with QC overlays for hair/detail, background spill, holes, and edge flicker. "
            "Link views mirrors zoom and panning, while QC Coach answers typed complaints and "
            "suggests settings you can apply to the current shot."
        ),
    ),
    (
        "11 · Set a shot profile",
        (
            "Open Inspect → Shot Profiles (Ctrl+Shift+P) and choose the current "
            "object/shot. Give that shot its own model, edge smoothness, temporal "
            "cleanup, detail preservation, edge softness and colour-spill settings, "
            "or make the profile the object default. The colour controls live in the "
            "same dialog, with a preview of the current frame so you can sample and "
            "draw boxes without leaving it."
        ),
    ),
    (
        "12 · Check GPU headroom",
        (
            "In the right-side Hardware section or Inspect menu (Ctrl+Shift+G), choose Open "
            "Performance Monitor to see live GPU use, VRAM, temperature, thermal pressure, an "
            "estimated render time, and a warning before a model is likely to run out of memory."
        ),
    ),
    (
        "13 · Queue and recover work",
        (
            "Export → Batch Render Queue saves multiple .bbproj projects, presets, progress, and "
            "destinations between sessions. Pause, resume, or retry failed renders. Inspect → "
            "Version History (Ctrl+Shift+H) keeps recoverable checkpoints; Create Checkpoint "
            "saves one manually."
        ),
    ),
]

_STEP_ACTIONS: dict[int, tuple[str, str]] = {
    0: ("open_video", "Open video…"),
    3: ("segment", "Segment current frame"),
    4: ("propagate", "Propagate to end"),
    7: ("export", "Open export dialog"),
    8: ("library", "Open media library"),
    9: ("matte_qc", "Open Matte QC Inspector"),
    10: ("shot_profiles", "Open Shot Profiles"),
    11: ("gpu_monitor", "Open Performance Monitor"),
    12: ("queue", "Open Batch Render Queue"),
}


#: Per-step callout: (where to click, shortcut or hint shown on hover).
_STEP_META: dict[int, tuple[str, str]] = {
    0: ("Drag onto the window, or File → Open Video", "Ctrl+O"),
    1: ("Right sidebar → Content", "AMV / Human / Object"),
    2: ("Canvas, Points tool", "Left = include · Right = exclude"),
    3: ("Right sidebar → Segment", "Segment Frame"),
    4: ("Right sidebar → Propagation", "Propagate to End"),
    5: ("Inspect → Shot Profiles", "Ctrl+Shift+P"),
    6: ("Canvas, Brush tool", "B · Shift = erase · [ ] = size"),
    7: ("File → Export", "Ctrl+E"),
    8: ("File → Media Library", "Ctrl+L"),
    9: ("Inspect → Matte QC Inspector", "Ctrl+Shift+M"),
    10: ("Inspect → Shot Profiles", "Ctrl+Shift+P"),
    11: ("Inspect → GPU Performance Monitor", "Ctrl+Shift+G"),
    12: ("Export → Batch Render Queue", "Inspect → History: Ctrl+Shift+H"),
}


class TutorialDialog(QDialog):
    """Interactive walkthrough with navigation progress and try-it actions."""

    action_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("How Better Brush works")
        self.setModal(True)
        self.setMinimumSize(760, 520)
        self.resize(880, 600)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(14)

        # ── Step rail: hover a step to preview it, click to jump ──────────
        self._rail = QListWidget()
        self._rail.setFixedWidth(228)
        self._rail.setSpacing(1)
        # Long step names must elide rather than force a horizontal scrollbar.
        self._rail.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._rail.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._rail.setWordWrap(False)
        self._rail.setStyleSheet(
            "QListWidget { background: #1a1a1e; border: 1px solid #2e2e33;"
            " border-radius: 6px; outline: none; }"
            "QListWidget::item { color: #8b8b94; padding: 7px 9px; border-radius: 4px; }"
            "QListWidget::item:hover { background: #26262c; color: #d4d4d8; }"
            "QListWidget::item:selected { background: #2f3a4d; color: #ffffff; }"
        )
        for index, (title, _body) in enumerate(_STEPS):
            item = QListWidgetItem(title.split(" · ", 1)[-1])
            item.setToolTip(self._step_tooltip(index))
            self._rail.addItem(item)
        # Hovering previews a step without leaving the one being read.
        self._rail.itemEntered.connect(self._on_rail_hovered)
        self._rail.currentRowChanged.connect(self._on_rail_selected)
        outer.addWidget(self._rail)

        # ── Detail column ────────────────────────────────────────────────
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        self._title = QLabel("")
        self._title.setStyleSheet("font-size: 17px; font-weight: 600; color: #e6e6ea;")
        self._title.setWordWrap(True)
        root.addWidget(self._title)

        # Callout: exactly where to click, with a hover for the shortcut.
        self._where = QLabel("")
        self._where.setWordWrap(True)
        self._where.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._where.setStyleSheet(
            "color: #cfe3ff; font-size: 11px; background: #1d2530;"
            " border: 1px solid #2c3a4d; border-left: 3px solid #3b82f6;"
            " border-radius: 5px; padding: 8px 10px;"
        )
        root.addWidget(self._where)

        self._body = QLabel("")
        self._body.setWordWrap(True)
        self._body.setStyleSheet("color: #a0a0a8; font-size: 12px; line-height: 1.5;")
        root.addWidget(self._body)

        self._count = QLabel("")
        self._count.setStyleSheet("color: #555560; font-size: 10px;")
        root.addWidget(self._count)

        self._progress = QProgressBar()
        self._progress.setRange(1, len(_STEPS))
        self._progress.setValue(1)
        self._progress.setTextVisible(True)
        self._progress.setFormat(f"1 / {len(_STEPS)} · current step")
        self._progress.setFixedHeight(18)
        root.addWidget(self._progress)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #6b6b74; font-size: 10px;")
        root.addWidget(self._hint)

        actions = QHBoxLayout()
        actions.addStretch()
        self._btn_try = QPushButton("Try it now")
        self._btn_try.setToolTip("Open the related editor action and close this guide")
        self._btn_try.clicked.connect(self._try_it)
        actions.addWidget(self._btn_try)
        self._btn_prev = QPushButton("Back")
        self._btn_prev.setToolTip("Previous step (Left arrow)")
        self._btn_prev.clicked.connect(self._go_prev)
        actions.addWidget(self._btn_prev)
        self._btn_next = QPushButton("Next")
        self._btn_next.setProperty("cssClass", "accent")
        self._btn_next.setToolTip("Next step (Right arrow)")
        self._btn_next.clicked.connect(self._go_next)
        actions.addWidget(self._btn_next)
        self._btn_skip = QPushButton("Close")
        self._btn_skip.setToolTip("Close this guide (Esc)")
        self._btn_skip.clicked.connect(self.reject)
        actions.addWidget(self._btn_skip)

        root.addLayout(actions)
        outer.addLayout(root, stretch=1)

        self._step = 0
        self._show_step()

    # ── Rail helpers ──────────────────────────────────────────────────────
    def _step_tooltip(self, index: int) -> str:
        """Hover text for a rail entry: the step, where, and the shortcut.

        The body is trimmed to its first line - a tooltip is a preview, and
        dumping the whole step into it produces an unreadable wall of text.
        """
        title, body = _STEPS[index]
        where, shortcut = _STEP_META.get(index, ("", ""))
        first = body.split("\n")[0].strip()
        if len(first) > 110:
            first = first[:107].rstrip() + "…"
        lines = [title, "", first]
        if where:
            lines += ["", f"Where: {where}"]
        if shortcut:
            lines.append(f"Shortcut: {shortcut}")
        return "\n".join(lines)

    def _on_rail_hovered(self, row: int) -> None:
        """Hovering a step previews it without discarding the current one."""
        if row < 0 or row == self._step:
            return
        title, _body = _STEPS[row]
        where, shortcut = _STEP_META.get(row, ("", ""))
        preview = f"{title}"
        if where:
            preview += f"\n{where}"
        if shortcut:
            preview += f"  ({shortcut})"
        self._hint.setText(f"Hover preview → {preview}")

    def _on_rail_selected(self, row: int) -> None:
        if row >= 0 and row != self._step:
            self._step = row
            self._show_step()

    # ── Navigation ────────────────────────────────────────────────────────
    def _show_step(self) -> None:
        title, body = _STEPS[self._step]
        self._title.setText(title)
        self._body.setText(body)
        where, shortcut = _STEP_META.get(self._step, ("", ""))
        if where:
            self._where.setText(
                f"WHERE:  {where}" + (f"      ·      {shortcut}" if shortcut else "")
            )
            self._where.setToolTip(
                f"{where}" + (f"\nShortcut: {shortcut}" if shortcut else "")
            )
            self._where.setVisible(True)
        else:
            self._where.setVisible(False)
        self._count.setText(f"Step {self._step + 1} of {len(_STEPS)}")
        self._progress.setValue(self._step + 1)
        self._progress.setFormat(f"{self._step + 1} / {len(_STEPS)} · current step")
        # Keep the rail in step without re-entering _on_rail_selected.
        if self._rail.currentRow() != self._step:
            self._rail.blockSignals(True)
            self._rail.setCurrentRow(self._step)
            self._rail.blockSignals(False)
        action = _STEP_ACTIONS.get(self._step)
        if action is None:
            self._btn_try.setVisible(False)
            self._hint.setText("Use Next to continue through the workflow.")
        else:
            self._btn_try.setVisible(True)
            self._btn_try.setText(action[1])
            self._hint.setText(
                "Try the action now, or continue with Next."
                "  ·  Hover a step on the left to preview it."
            )
        self._btn_prev.setEnabled(self._step > 0)
        last = self._step == len(_STEPS) - 1
        self._btn_next.setText("Finish" if last else "Next")
        self._btn_skip.setVisible(not last)

    def _try_it(self) -> None:
        action = _STEP_ACTIONS.get(self._step)
        if action is not None:
            self.action_requested.emit(action[0])
        self.accept()

    def _go_prev(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._show_step()

    def _go_next(self) -> None:
        if self._step < len(_STEPS) - 1:
            self._step += 1
            self._show_step()
        else:
            self.accept()

    def keyPressEvent(self, event) -> None:
        """Arrow keys walk the guide; Esc closes it."""
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Backspace):
            self._go_prev()
        elif key == Qt.Key.Key_Right:
            self._go_next()
        else:
            super().keyPressEvent(event)
