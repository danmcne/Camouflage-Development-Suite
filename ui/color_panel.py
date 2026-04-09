"""
ColorPanel – Tab 1.

Now hosts two palettes via an inner QTabWidget:
  • Layer 1  – used by the primary generator
  • Layer 2  – used by the second generator layer (optional)

Signals:
  palette_changed(palette, layer)  – layer 0 or 1
"""
from __future__ import annotations
import random

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QPushButton, QColorDialog,
    QFileDialog, QFrame, QSizePolicy, QScrollArea,
    QToolButton, QGroupBox, QMessageBox, QTabWidget,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, pyqtSignal

from core.palette import ColorPalette
from config.defaults import APP


# ── swatch widget ─────────────────────────────────────────────────────────────

class SwatchWidget(QFrame):
    color_clicked = pyqtSignal(int)
    lock_toggled  = pyqtSignal(int, bool)

    def __init__(self, index, hex_color, locked=False, parent=None):
        super().__init__(parent)
        self.index   = index
        self._hex    = hex_color
        self._locked = locked
        self.setFixedSize(66, 74)
        self.setFrameShape(QFrame.Shape.Box)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2,2,2,2); lay.setSpacing(2)

        self._btn = QPushButton()
        self._btn.setFixedHeight(44)
        self._btn.setToolTip("Click to change colour")
        self._btn.clicked.connect(lambda: self.color_clicked.emit(self.index))
        lay.addWidget(self._btn)

        self._lock = QToolButton()
        self._lock.setCheckable(True)
        self._lock.setChecked(locked)
        self._lock.setFixedSize(62, 20)
        self._lock.toggled.connect(self._on_lock)
        lay.addWidget(self._lock)
        self._apply()

    def _apply(self):
        self._btn.setStyleSheet(
            f"background-color:{self._hex};border:2px solid #555;border-radius:3px;")
        self._lock.setText("🔒 Lock" if self._locked else "🔓 Free")
        self._lock.setChecked(self._locked)

    def set_color(self, h):  self._hex = h;  self._apply()
    def set_locked(self, v): self._locked = v; self._apply()

    def _on_lock(self, c):
        self._locked = c; self._apply()
        self.lock_toggled.emit(self.index, c)


# ── single-palette editor ─────────────────────────────────────────────────────

class _PaletteEditor(QWidget):
    """Self-contained editor for one ColorPalette."""

    changed = pyqtSignal(object)   # emits ColorPalette

    def __init__(self, palette: ColorPalette, label: str = "Palette", parent=None):
        super().__init__(parent)
        self._palette  = palette
        self._swatches: list[SwatchWidget] = []
        self._label    = label
        self._build()
        self._rebuild_swatches()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6,6,6,6)
        root.setSpacing(6)

        # Count row
        cr = QHBoxLayout()
        cr.addWidget(QLabel("Colours:"))
        self._spin = QSpinBox()
        self._spin.setRange(2, APP["max_palette_colors"])
        self._spin.setValue(len(self._palette))
        self._spin.valueChanged.connect(self._on_count)
        cr.addWidget(self._spin); cr.addStretch()
        root.addLayout(cr)

        # Swatch scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(200)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sw_container = QWidget()
        self._sw_grid = QGridLayout(self._sw_container)
        self._sw_grid.setSpacing(4)
        scroll.setWidget(self._sw_container)
        root.addWidget(scroll)

        # Presets
        pg = QGroupBox("Presets")
        pl = QGridLayout(pg)
        presets = [
            ("🌿 Military",   "military_preset"),
            ("🏜 Desert",     "desert_preset"),
            ("🏙 Urban",      "urban_preset"),
            ("🧱 Warm Urban", "warm_urban_preset"),
            ("🌲 Woodland",   "woodland_preset"),
            ("❄ Arctic",     "arctic_preset"),
            ("❄🌑 Cool Hi",  "cool_contrast_preset"),
            ("☀🌑 Warm Hi",  "warm_contrast_preset"),
            ("🎲 Random",     None),
        ]
        for i, (lbl, m) in enumerate(presets):
            b = QPushButton(lbl); b.setFixedHeight(26)
            b.clicked.connect(lambda _, mm=m: self._load_preset(mm))
            pl.addWidget(b, i//2, i%2)
        root.addWidget(pg)

        # Image extract
        eg = QGroupBox("Extract from image")
        el = QHBoxLayout(eg)
        eb = QPushButton("📂 Choose image…")
        eb.clicked.connect(self._extract)
        el.addWidget(eb)
        self._src_lbl = QLabel("No image")
        self._src_lbl.setStyleSheet("color:#888;font-size:10px;")
        self._src_lbl.setWordWrap(True)
        el.addWidget(self._src_lbl, 1)
        root.addWidget(eg)
        root.addStretch()

    def _rebuild_swatches(self):
        for sw in self._swatches: sw.deleteLater()
        self._swatches.clear()
        cols = 5
        for i, h in enumerate(self._palette):
            sw = SwatchWidget(i, h, self._palette.is_locked(i))
            sw.color_clicked.connect(self._on_click)
            sw.lock_toggled.connect(self._on_lock)
            self._sw_grid.addWidget(sw, i//cols, i%cols)
            self._swatches.append(sw)

    def _on_click(self, idx):
        chosen = QColorDialog.getColor(QColor(self._palette[idx]), self, "Choose colour")
        if chosen.isValid():
            h = chosen.name().upper()
            self._palette.set_color(idx, h)
            self._swatches[idx].set_color(h)
            self.changed.emit(self._palette)

    def _on_lock(self, idx, locked):
        self._palette.set_locked(idx, locked)

    def _on_count(self, n):
        if self._palette.source_image:
            try:
                new = ColorPalette.from_image_kmeans(self._palette.source_image, n)
                for i in range(min(len(self._palette), len(new))):
                    if self._palette.is_locked(i):
                        new.set_color(i, self._palette[i]); new.set_locked(i, True)
                self._palette = new
            except Exception:
                self._palette.resize_to(n)
        else:
            self._palette.resize_to(n)
        self._spin.blockSignals(True)
        self._spin.setValue(len(self._palette))
        self._spin.blockSignals(False)
        self._rebuild_swatches()
        self.changed.emit(self._palette)

    def _load_preset(self, method):
        if method is None:
            self._palette = ColorPalette.random(self._spin.value())
        else:
            self._palette = getattr(ColorPalette, method)()
        self._palette.source_image = None
        self._src_lbl.setText("No image")
        self._spin.blockSignals(True)
        self._spin.setValue(len(self._palette))
        self._spin.blockSignals(False)
        self._rebuild_swatches()
        self.changed.emit(self._palette)

    def _extract(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if not path: return
        try:
            self._palette = ColorPalette.from_image_kmeans(path, self._spin.value())
            self._src_lbl.setText(path.split("/")[-1])
            self._rebuild_swatches()
            self.changed.emit(self._palette)
        except Exception as e:
            QMessageBox.warning(self, "Extraction failed", str(e))

    def get_palette(self) -> ColorPalette: return self._palette

    def set_palette(self, palette: ColorPalette):
        self._palette = palette
        self._spin.blockSignals(True)
        self._spin.setValue(len(palette))
        self._spin.blockSignals(False)
        self._rebuild_swatches()


# ── outer panel with two tabs ─────────────────────────────────────────────────

class ColorPanel(QWidget):
    # layer: 0 = layer1 palette, 1 = layer2 palette
    palette_changed = pyqtSignal(object, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ed1 = _PaletteEditor(ColorPalette.military_preset(), "Layer 1")
        self._ed2 = _PaletteEditor(ColorPalette.urban_preset(),    "Layer 2")

        tabs = QTabWidget()
        tabs.addTab(self._ed1, "Layer 1")
        tabs.addTab(self._ed2, "Layer 2")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(tabs)

        self._ed1.changed.connect(lambda p: self.palette_changed.emit(p, 0))
        self._ed2.changed.connect(lambda p: self.palette_changed.emit(p, 1))

    def get_palette(self, layer=0) -> ColorPalette:
        return self._ed1.get_palette() if layer == 0 else self._ed2.get_palette()

    def set_palette(self, palette: ColorPalette, layer=0):
        ed = self._ed1 if layer == 0 else self._ed2
        ed.set_palette(palette)
