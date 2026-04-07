"""
ColorPanel – Tab 1.

Changes from v2:
  • When colour count changes after image extraction, palette is recalculated
    from the stored source image (not padded with random colours).
  • When colour count changes from a preset, new slots are filled with
    perceptually similar variants (via palette.resize_to).
  • Added Warm Urban and Woodland presets.
  • Presets load at 8 colours by default; spin box updates accordingly.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QPushButton, QColorDialog,
    QFileDialog, QFrame, QSizePolicy, QScrollArea,
    QToolButton, QGroupBox, QMessageBox,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, pyqtSignal

from core.palette import ColorPalette
from config.defaults import APP


class SwatchWidget(QFrame):
    color_clicked = pyqtSignal(int)
    lock_toggled  = pyqtSignal(int, bool)

    def __init__(self, index: int, hex_color: str, locked: bool = False, parent=None):
        super().__init__(parent)
        self.index  = index
        self._hex   = hex_color
        self._locked= locked
        self.setFixedSize(66, 74)
        self.setFrameShape(QFrame.Shape.Box)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._color_btn = QPushButton()
        self._color_btn.setFixedHeight(44)
        self._color_btn.setToolTip("Click to change colour")
        self._color_btn.clicked.connect(lambda: self.color_clicked.emit(self.index))
        layout.addWidget(self._color_btn)

        self._lock_btn = QToolButton()
        self._lock_btn.setCheckable(True)
        self._lock_btn.setChecked(locked)
        self._lock_btn.setFixedSize(62, 20)
        self._lock_btn.toggled.connect(self._on_lock)
        layout.addWidget(self._lock_btn)

        self._apply_style()

    def _apply_style(self):
        self._color_btn.setStyleSheet(
            f"background-color:{self._hex}; border:2px solid #555; border-radius:3px;"
        )
        self._lock_btn.setText("🔒 Lock" if self._locked else "🔓 Free")
        self._lock_btn.setChecked(self._locked)

    def set_color(self, h: str):
        self._hex = h
        self._apply_style()

    def set_locked(self, v: bool):
        self._locked = v
        self._apply_style()

    def _on_lock(self, checked: bool):
        self._locked = checked
        self._apply_style()
        self.lock_toggled.emit(self.index, checked)


class ColorPanel(QWidget):
    palette_changed = pyqtSignal(object)   # emits ColorPalette

    def __init__(self, parent=None):
        super().__init__(parent)
        self._palette  = ColorPalette.military_preset()
        self._swatches: list[SwatchWidget] = []
        self._build_ui()
        self._rebuild_swatches()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Number of colours:"))
        self._spin = QSpinBox()
        self._spin.setRange(2, APP["max_palette_colors"])
        self._spin.setValue(len(self._palette))
        self._spin.valueChanged.connect(self._on_count_changed)
        count_row.addWidget(self._spin)
        count_row.addStretch()
        root.addLayout(count_row)

        # Swatch scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(210)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._swatch_container = QWidget()
        self._swatch_grid = QGridLayout(self._swatch_container)
        self._swatch_grid.setSpacing(5)
        scroll.setWidget(self._swatch_container)
        root.addWidget(scroll)

        # Presets
        preset_group = QGroupBox("Presets")
        pg = QGridLayout(preset_group)
        presets = [
            ("🌿 Military",   "military_preset"),
            ("🏜 Desert",     "desert_preset"),
            ("🏙 Urban",      "urban_preset"),
            ("🧱 Warm Urban", "warm_urban_preset"),
            ("🌲 Woodland",   "woodland_preset"),
            ("❄ Arctic",     "arctic_preset"),
            ("🎲 Random",     None),
        ]
        for i, (label, method) in enumerate(presets):
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda _, m=method: self._load_preset(m))
            pg.addWidget(btn, i // 2, i % 2)
        root.addWidget(preset_group)

        # Extract from image
        extract_group = QGroupBox("Extract from image")
        el = QHBoxLayout(extract_group)
        self._extract_btn = QPushButton("📂 Choose image…")
        self._extract_btn.clicked.connect(self._extract_from_image)
        el.addWidget(self._extract_btn)
        self._source_label = QLabel("No image selected")
        self._source_label.setStyleSheet("color:#888; font-size:10px;")
        self._source_label.setWordWrap(True)
        el.addWidget(self._source_label, 1)
        root.addWidget(extract_group)

        root.addStretch()

    # ── swatch management ─────────────────────────────────────────────────────

    def _rebuild_swatches(self):
        for sw in self._swatches:
            sw.deleteLater()
        self._swatches.clear()

        cols = 5
        for i, hex_color in enumerate(self._palette):
            sw = SwatchWidget(i, hex_color, self._palette.is_locked(i))
            sw.color_clicked.connect(self._on_swatch_click)
            sw.lock_toggled.connect(self._on_lock_toggle)
            self._swatch_grid.addWidget(sw, i // cols, i % cols)
            self._swatches.append(sw)

    def _on_swatch_click(self, index: int):
        current = QColor(self._palette[index])
        chosen  = QColorDialog.getColor(current, self, "Choose colour")
        if chosen.isValid():
            h = chosen.name().upper()
            self._palette.set_color(index, h)
            self._swatches[index].set_color(h)
            self.palette_changed.emit(self._palette)

    def _on_lock_toggle(self, index: int, locked: bool):
        self._palette.set_locked(index, locked)

    # ── count change ──────────────────────────────────────────────────────────

    def _on_count_changed(self, n: int):
        if self._palette.source_image:
            # Recalculate from the original image with new count
            try:
                new_pal = ColorPalette.from_image_kmeans(
                    self._palette.source_image, n_colors=n
                )
                # Re-apply any locked colours (overwrite corresponding slots)
                for i in range(min(len(self._palette), len(new_pal))):
                    if self._palette.is_locked(i):
                        new_pal.set_color(i, self._palette[i])
                        new_pal.set_locked(i, True)
                self._palette = new_pal
            except Exception:
                self._palette.resize_to(n)
        else:
            self._palette.resize_to(n)

        self._spin.blockSignals(True)
        self._spin.setValue(len(self._palette))
        self._spin.blockSignals(False)
        self._rebuild_swatches()
        self.palette_changed.emit(self._palette)

    # ── presets ───────────────────────────────────────────────────────────────

    def _load_preset(self, method: str | None):
        if method is None:
            self._palette = ColorPalette.random(self._spin.value())
        else:
            self._palette = getattr(ColorPalette, method)()
        self._palette.source_image = None
        self._source_label.setText("No image selected")
        self._spin.blockSignals(True)
        self._spin.setValue(len(self._palette))
        self._spin.blockSignals(False)
        self._rebuild_swatches()
        self.palette_changed.emit(self._palette)

    # ── image extraction ──────────────────────────────────────────────────────

    def _extract_from_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if not path:
            return
        try:
            n = self._spin.value()
            self._palette = ColorPalette.from_image_kmeans(path, n_colors=n)
            self._source_label.setText(path.split("/")[-1])
            self._rebuild_swatches()
            self.palette_changed.emit(self._palette)
        except Exception as e:
            QMessageBox.warning(self, "Extraction failed", str(e))

    # ── public API ────────────────────────────────────────────────────────────

    def get_palette(self) -> ColorPalette:
        return self._palette

    def set_palette(self, palette: ColorPalette):
        self._palette = palette
        self._spin.blockSignals(True)
        self._spin.setValue(len(palette))
        self._spin.blockSignals(False)
        self._rebuild_swatches()
