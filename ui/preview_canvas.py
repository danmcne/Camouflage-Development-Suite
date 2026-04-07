"""
PreviewCanvas – a QLabel-based live preview that always shows the current pattern
superimposed on the active background.  Supports zoom, opacity slider, and drag.
"""
from __future__ import annotations
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

from utils.rendering import bgr_to_qpixmap, superimpose


class PreviewCanvas(QWidget):
    """Live preview pane: pattern on background with opacity and zoom controls."""

    request_generate = pyqtSignal()   # emitted when user clicks "Regenerate"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pattern: np.ndarray | None = None
        self._background: np.ndarray | None = None
        self._alpha: float = 1.0

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Title bar
        title_row = QHBoxLayout()
        lbl = QLabel("Live Preview")
        lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        title_row.addWidget(lbl)
        title_row.addStretch()

        regen_btn = QPushButton("⟳ Regenerate")
        regen_btn.setFixedHeight(26)
        regen_btn.setToolTip("Generate a new pattern with current settings")
        regen_btn.clicked.connect(self.request_generate)
        title_row.addWidget(regen_btn)
        root.addLayout(title_row)

        # Image label
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._img_label.setMinimumSize(256, 256)
        self._img_label.setStyleSheet(
            "background: #1a1a1a; border: 1px solid #444; border-radius: 4px;"
        )
        self._img_label.setText("No pattern generated yet")
        root.addWidget(self._img_label)

        # Opacity slider
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacity:"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.setToolTip("Pattern opacity over background")
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        op_row.addWidget(self._opacity_slider)
        self._opacity_label = QLabel("100%")
        self._opacity_label.setFixedWidth(36)
        op_row.addWidget(self._opacity_label)
        root.addLayout(op_row)

        # Fitness readout (updated externally)
        self._fitness_label = QLabel("Fitness: –")
        self._fitness_label.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(self._fitness_label)

    # ── public API ────────────────────────────────────────────────────────────

    def set_pattern(self, img: np.ndarray):
        self._pattern = img
        self._refresh()

    def set_background(self, img: np.ndarray | None):
        self._background = img
        self._refresh()

    def set_fitness(self, scores: dict):
        parts = [f"{k}: {v:.3f}" for k, v in scores.items()]
        self._fitness_label.setText("  |  ".join(parts))

    def clear(self):
        self._pattern = None
        self._img_label.setText("No pattern generated yet")
        self._fitness_label.setText("Fitness: –")

    # ── internals ─────────────────────────────────────────────────────────────

    def _on_opacity_changed(self, value: int):
        self._alpha = value / 100.0
        self._opacity_label.setText(f"{value}%")
        self._refresh()

    def _refresh(self):
        if self._pattern is None:
            return

        available = self._img_label.size()
        max_dim = min(available.width(), available.height(), 1024)
        max_dim = max(max_dim, 128)

        if self._background is not None:
            composite = superimpose(self._pattern, self._background, self._alpha)
        else:
            composite = self._pattern

        pixmap = bgr_to_qpixmap(composite, max_size=max_dim)
        self._img_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()
