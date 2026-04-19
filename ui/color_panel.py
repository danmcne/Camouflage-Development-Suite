"""
ColorPanel – redesigned palette editor with multi-image support.

Each layer palette editor (_PaletteEditor) provides:
  • Swatch grid – click to pick colour via dialog; locked swatches shown with padlock.
  • Selected swatch – highlighted in blue; receives colour from image picker.
  • Pick mode – hover over any image tab to see a magnifier ring showing the
    colour under the cursor; click to send that colour to the selected swatch.
    Locked swatches are skipped.
  • K-means – samples pixels from ALL uploaded images at once; locked colours
    are preserved and restored after extraction.
  • Image tabs – up to 10 source images plus a "Generator Output" tab for
    reference or picking colours from the current pattern.
  • Presets – compact combobox + Apply button.
  • Lock – blocks both dialog edit and pick-mode / k-means updates.
"""
from __future__ import annotations
import os
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QPushButton, QColorDialog,
    QFileDialog, QFrame, QSizePolicy, QScrollArea,
    QToolButton, QGroupBox, QMessageBox, QTabWidget,
    QComboBox, QSplitter,
)
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint

from core.palette import ColorPalette
from config.defaults import APP


# ── helpers ───────────────────────────────────────────────────────────────────

def _np_to_qpixmap(img_rgb: np.ndarray) -> QPixmap:
    """Convert H×W×3 uint8 RGB numpy array to QPixmap."""
    h, w = img_rgb.shape[:2]
    qimg  = QImage(img_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ── image picker widget ───────────────────────────────────────────────────────

class ImagePickerWidget(QWidget):
    """
    Displays an image.  In pick_mode, the cursor shows a coloured ring
    matching the pixel underneath; clicking emits color_picked(QColor).
    """
    color_picked = pyqtSignal(object)   # QColor

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img_np: np.ndarray | None = None   # H×W×3 RGB uint8
        self._pixmap: QPixmap | None    = None
        self._pick_mode  = False
        self._hover_pos: QPoint | None  = None
        self._hover_color: QColor | None= None
        self.setMouseTracking(True)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ── public ────────────────────────────────────────────────────────────────

    def set_image_from_path(self, path: str) -> bool:
        import cv2
        img = cv2.imread(path)
        if img is None:
            return False
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._load_rgb(rgb)

    def set_image_from_array(self, img_bgr: np.ndarray) -> bool:
        import cv2
        if img_bgr is None:
            self._img_np = None; self._pixmap = None; self.update(); return False
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self._load_rgb(rgb)

    def set_pick_mode(self, enabled: bool):
        self._pick_mode = enabled
        self._hover_pos  = None
        self._hover_color= None
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
        self.update()

    def has_image(self) -> bool:
        return self._img_np is not None

    # ── internal ──────────────────────────────────────────────────────────────

    def _load_rgb(self, rgb: np.ndarray) -> bool:
        self._img_np = rgb
        self._pixmap = _np_to_qpixmap(rgb)
        self.update()
        return True

    def _display_rect(self) -> QRect:
        if self._pixmap is None:
            return QRect(0, 0, self.width(), self.height())
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if pw == 0 or ph == 0:
            return QRect(0, 0, ww, wh)
        scale = min(ww / pw, wh / ph)
        dw = int(pw * scale); dh = int(ph * scale)
        x = (ww - dw) // 2;   y = (wh - dh) // 2
        return QRect(x, y, dw, dh)

    def _color_at(self, pos: QPoint) -> QColor | None:
        if self._img_np is None:
            return None
        dr = self._display_rect()
        if not dr.contains(pos):
            return None
        rx = pos.x() - dr.x(); ry = pos.y() - dr.y()
        ih, iw = self._img_np.shape[:2]
        px = max(0, min(int(rx / dr.width()  * iw), iw - 1))
        py = max(0, min(int(ry / dr.height() * ih), ih - 1))
        r, g, b = self._img_np[py, px]
        return QColor(int(r), int(g), int(b))

    # ── Qt events ─────────────────────────────────────────────────────────────

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        if self._pixmap:
            dr = self._display_rect()
            p.drawPixmap(dr, self._pixmap)
        else:
            p.fillRect(0, 0, W, H, QColor(40, 40, 40))
            p.setPen(QColor(120, 120, 120))
            p.drawText(QRect(0, 0, W, H), Qt.AlignmentFlag.AlignCenter,
                       "No image\n(add images or generate a pattern)")

        # Magnifier ring
        if self._pick_mode and self._hover_pos and self._hover_color:
            cx, cy = self._hover_pos.x(), self._hover_pos.y()
            R = 16
            # Filled coloured disc
            p.setBrush(QBrush(self._hover_color))
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawEllipse(cx - R, cy - R, R * 2, R * 2)
            # Crosshair lines outside the disc
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.drawLine(cx - R - 6, cy, cx - R - 1, cy)
            p.drawLine(cx + R + 1, cy, cx + R + 6, cy)
            p.drawLine(cx, cy - R - 6, cx, cy - R - 1)
            p.drawLine(cx, cy + R + 1, cx, cy + R + 6)
        p.end()

    def mouseMoveEvent(self, ev):
        pos = ev.position().toPoint()
        if self._pick_mode:
            self._hover_pos   = pos
            self._hover_color = self._color_at(pos)
            self.update()

    def leaveEvent(self, ev):
        self._hover_pos = None; self.update()

    def mousePressEvent(self, ev):
        if self._pick_mode and ev.button() == Qt.MouseButton.LeftButton:
            c = self._color_at(ev.position().toPoint())
            if c:
                self.color_picked.emit(c)


# ── swatch widget ─────────────────────────────────────────────────────────────

class SwatchWidget(QFrame):
    color_clicked = pyqtSignal(int)
    lock_toggled  = pyqtSignal(int, bool)

    def __init__(self, index, hex_color, locked=False, parent=None):
        super().__init__(parent)
        self.index    = index
        self._hex     = hex_color
        self._locked  = locked
        self._selected= False
        self.setFixedSize(66, 78)
        self.setFrameShape(QFrame.Shape.Box)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2); lay.setSpacing(2)

        self._btn = QPushButton()
        self._btn.setFixedHeight(44)
        self._btn.setToolTip("Click to change colour (or pick from image in pick mode)")
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
        border = "3px solid #4488ff" if self._selected else "2px solid #555"
        self._btn.setStyleSheet(
            f"background-color:{self._hex};border:{border};border-radius:3px;")
        self._lock.setText("🔒 Lock" if self._locked else "🔓 Free")
        self._lock.setChecked(self._locked)

    def set_color(self, h: str):
        self._hex = h; self._apply()

    def set_locked(self, v: bool):
        self._locked = v; self._apply()

    def set_selected(self, v: bool):
        self._selected = v; self._apply()

    def _on_lock(self, c):
        self._locked = c; self._apply()
        self.lock_toggled.emit(self.index, c)


# ── palette editor ────────────────────────────────────────────────────────────

class _PaletteEditor(QWidget):
    """Self-contained editor for one ColorPalette."""
    changed = pyqtSignal(object)   # emits ColorPalette

    _PRESETS = [
        ("🌿 Military",   "military_preset"),
        ("🏜  Desert",    "desert_preset"),
        ("🏙  Urban",     "urban_preset"),
        ("🧱 Warm Urban", "warm_urban_preset"),
        ("🌲 Woodland",   "woodland_preset"),
        ("❄  Arctic",    "arctic_preset"),
        ("❄🌑 Cool Hi",  "cool_contrast_preset"),
        ("☀🌑 Warm Hi",  "warm_contrast_preset"),
        ("🎲 Random",     None),
    ]

    def __init__(self, palette: ColorPalette, label: str = "Palette", parent=None):
        super().__init__(parent)
        self._palette   = palette
        self._swatches: list[SwatchWidget] = []
        self._selected  = 0        # which swatch receives picked colours
        self._pick_mode = False
        self._source_images: list[str] = []      # paths of uploaded images
        self._pickers:  list[ImagePickerWidget] = []  # one per source image tab
        self._out_picker: ImagePickerWidget | None = None  # generator output tab
        self._label     = label
        self._build_ui()
        self._rebuild_swatches()

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── row 1: colour count + preset ─────────────────────────────────────
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Colours:"))
        self._spin = QSpinBox()
        self._spin.setRange(2, APP["max_palette_colors"])
        self._spin.setValue(len(self._palette))
        self._spin.valueChanged.connect(self._on_count_changed)
        r1.addWidget(self._spin)
        r1.addSpacing(8)
        r1.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        for lbl, _ in self._PRESETS:
            self._preset_combo.addItem(lbl)
        r1.addWidget(self._preset_combo, 1)
        apply_btn = QPushButton("Apply"); apply_btn.setFixedWidth(46)
        apply_btn.clicked.connect(self._apply_preset)
        r1.addWidget(apply_btn)
        root.addLayout(r1)

        # ── swatch grid (scrollable) ──────────────────────────────────────────
        self._sw_container = QWidget()
        self._sw_grid = QGridLayout(self._sw_container)
        self._sw_grid.setSpacing(3)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(175)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._sw_container)
        root.addWidget(scroll)

        # ── row 2: pick / k-means / add images / clear ───────────────────────
        r2 = QHBoxLayout()
        self._pick_btn = QPushButton("🎯 Pick from image")
        self._pick_btn.setCheckable(True)
        self._pick_btn.setToolTip(
            "Toggle pick mode: hover over an image tab to see the colour under "
            "the cursor; click to set the selected swatch to that colour. "
            "Locked swatches are skipped.")
        self._pick_btn.toggled.connect(self._on_pick_toggled)
        r2.addWidget(self._pick_btn, 1)

        km_btn = QPushButton("📊 K-means")
        km_btn.setToolTip("Run k-means clustering over all uploaded images to "
                          "extract a palette. Locked swatches are preserved.")
        km_btn.clicked.connect(self._run_kmeans)
        r2.addWidget(km_btn)
        root.addLayout(r2)

        r3 = QHBoxLayout()
        add_btn = QPushButton("📂 Add images…")
        add_btn.setToolTip("Add one or more source images (up to 10 total).")
        add_btn.clicked.connect(self._add_images)
        r3.addWidget(add_btn)
        clr_btn = QPushButton("🗑 Clear")
        clr_btn.setToolTip("Remove all uploaded source images.")
        clr_btn.clicked.connect(self._clear_images)
        r3.addWidget(clr_btn)
        self._img_count_lbl = QLabel("No images")
        self._img_count_lbl.setStyleSheet("color:#888;font-size:10px;")
        r3.addWidget(self._img_count_lbl, 1)
        root.addLayout(r3)

        # ── image tabs (generator output + source images) ─────────────────────
        self._img_tabs = QTabWidget()
        self._img_tabs.setMinimumHeight(190)
        self._img_tabs.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Expanding)

        # Generator output tab (index 0)
        self._out_picker = ImagePickerWidget()
        self._out_picker.color_picked.connect(self._on_color_picked)
        self._img_tabs.addTab(self._out_picker, "Output")
        root.addWidget(self._img_tabs, 1)

    # ── swatches ──────────────────────────────────────────────────────────────

    def _rebuild_swatches(self):
        for sw in self._swatches:
            sw.deleteLater()
        self._swatches.clear()
        cols = 5
        self._selected = max(0, min(self._selected, len(self._palette) - 1))
        for i, h in enumerate(self._palette):
            sw = SwatchWidget(i, h, self._palette.is_locked(i))
            sw.set_selected(i == self._selected)
            sw.color_clicked.connect(self._on_swatch_clicked)
            sw.lock_toggled.connect(self._on_lock_toggled)
            self._sw_grid.addWidget(sw, i // cols, i % cols)
            self._swatches.append(sw)

    def _select_swatch(self, idx: int):
        if idx == self._selected:
            return
        if 0 <= self._selected < len(self._swatches):
            self._swatches[self._selected].set_selected(False)
        self._selected = idx
        if 0 <= idx < len(self._swatches):
            self._swatches[idx].set_selected(True)

    # ── swatch events ─────────────────────────────────────────────────────────

    def _on_swatch_clicked(self, idx: int):
        self._select_swatch(idx)
        if self._palette.is_locked(idx):
            return
        if not self._pick_mode:
            # Open colour dialog
            chosen = QColorDialog.getColor(
                QColor(self._palette[idx]), self, "Choose colour")
            if chosen.isValid():
                h = chosen.name().upper()
                self._palette.set_color(idx, h)
                self._swatches[idx].set_color(h)
                self.changed.emit(self._palette)

    def _on_lock_toggled(self, idx: int, locked: bool):
        self._palette.set_locked(idx, locked)

    # ── image picking ─────────────────────────────────────────────────────────

    def _on_pick_toggled(self, enabled: bool):
        self._pick_mode = enabled
        self._pick_btn.setText(
            "🎯 Pick ON — click image" if enabled else "🎯 Pick from image")
        self._out_picker.set_pick_mode(enabled)
        for pk in self._pickers:
            pk.set_pick_mode(enabled)

    def _on_color_picked(self, color: QColor):
        """Receive a picked colour and apply to selected (unlocked) swatch."""
        idx = self._selected
        # If the selected swatch is locked, find next unlocked
        n = len(self._palette)
        for offset in range(n):
            candidate = (idx + offset) % n
            if not self._palette.is_locked(candidate):
                idx = candidate
                self._select_swatch(idx)
                break
        else:
            return  # all locked

        h = color.name().upper()
        self._palette.set_color(idx, h)
        self._swatches[idx].set_color(h)
        # Auto-advance to next unlocked swatch
        for offset in range(1, n):
            candidate = (idx + offset) % n
            if not self._palette.is_locked(candidate):
                self._select_swatch(candidate)
                break
        self.changed.emit(self._palette)

    # ── image management ──────────────────────────────────────────────────────

    def _add_images(self):
        remaining = 10 - len(self._source_images)
        if remaining <= 0:
            QMessageBox.information(self, "Limit reached",
                                    "Maximum 10 source images. Remove some first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add source images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if not paths:
            return
        paths = paths[:remaining]
        for path in paths:
            if path not in self._source_images:
                self._source_images.append(path)
        self._rebuild_image_tabs()

    def _clear_images(self):
        self._source_images.clear()
        self._rebuild_image_tabs()

    def _rebuild_image_tabs(self):
        """Sync image tabs with current _source_images list."""
        # Remove all tabs except Output (index 0)
        while self._img_tabs.count() > 1:
            w = self._img_tabs.widget(1)
            self._img_tabs.removeTab(1)
            if w: w.deleteLater()
        self._pickers.clear()

        for i, path in enumerate(self._source_images):
            pk = ImagePickerWidget()
            pk.set_image_from_path(path)
            pk.set_pick_mode(self._pick_mode)
            pk.color_picked.connect(self._on_color_picked)
            name = os.path.basename(path)
            if len(name) > 12:
                name = name[:10] + "…"
            self._img_tabs.addTab(pk, f"Img {i+1}")
            self._img_tabs.setTabToolTip(i + 1, path)
            self._pickers.append(pk)

        n = len(self._source_images)
        self._img_count_lbl.setText(
            f"{n}/10 image{'s' if n!=1 else ''}" if n else "No images")

    # ── k-means ───────────────────────────────────────────────────────────────

    def _run_kmeans(self):
        if not self._source_images:
            QMessageBox.information(self, "No images",
                "Add source images first (📂 Add images…).")
            return
        n = self._spin.value()
        try:
            new_pal = ColorPalette.from_images_kmeans(self._source_images, n)
        except Exception as e:
            QMessageBox.warning(self, "K-means failed", str(e))
            return

        # Restore locked colours
        for i in range(min(len(self._palette), len(new_pal))):
            if self._palette.is_locked(i):
                new_pal.set_color(i, self._palette[i])
                new_pal.set_locked(i, True)

        self._palette = new_pal
        self._spin.blockSignals(True)
        self._spin.setValue(len(self._palette))
        self._spin.blockSignals(False)
        self._rebuild_swatches()
        self.changed.emit(self._palette)

    # ── count / preset ────────────────────────────────────────────────────────

    def _on_count_changed(self, n: int):
        # If we have source images, re-run k-means at new count
        if self._source_images:
            try:
                new_pal = ColorPalette.from_images_kmeans(self._source_images, n)
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
        self.changed.emit(self._palette)

    def _apply_preset(self):
        idx = self._preset_combo.currentIndex()
        _, method = self._PRESETS[idx]
        if method is None:
            new_pal = ColorPalette.random(self._spin.value())
        else:
            new_pal = getattr(ColorPalette, method)()
        # Restore locked colours
        for i in range(min(len(self._palette), len(new_pal))):
            if self._palette.is_locked(i):
                new_pal.set_color(i, self._palette[i])
                new_pal.set_locked(i, True)
        self._palette = new_pal
        self._spin.blockSignals(True)
        self._spin.setValue(len(self._palette))
        self._spin.blockSignals(False)
        self._rebuild_swatches()
        self.changed.emit(self._palette)

    # ── public API ────────────────────────────────────────────────────────────

    def set_preview(self, img_bgr: np.ndarray):
        """Update the Generator Output tab with the current pattern."""
        if self._out_picker is not None:
            self._out_picker.set_image_from_array(img_bgr)

    def get_palette(self) -> ColorPalette:
        return self._palette

    def set_palette(self, palette: ColorPalette):
        self._palette = palette
        self._spin.blockSignals(True)
        self._spin.setValue(len(palette))
        self._spin.blockSignals(False)
        self._rebuild_swatches()


# ── outer panel with L1 / L2 tabs ────────────────────────────────────────────

class ColorPanel(QWidget):
    palette_changed = pyqtSignal(object, int)   # (ColorPalette, layer 0 or 1)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ed1 = _PaletteEditor(ColorPalette.military_preset(), "Layer 1")
        self._ed2 = _PaletteEditor(ColorPalette.urban_preset(),    "Layer 2")

        tabs = QTabWidget()
        tabs.addTab(self._ed1, "Layer 1")
        tabs.addTab(self._ed2, "Layer 2")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(tabs)

        self._ed1.changed.connect(lambda p: self.palette_changed.emit(p, 0))
        self._ed2.changed.connect(lambda p: self.palette_changed.emit(p, 1))

    def get_palette(self, layer: int = 0) -> ColorPalette:
        return self._ed1.get_palette() if layer == 0 else self._ed2.get_palette()

    def set_palette(self, palette: ColorPalette, layer: int = 0):
        ed = self._ed1 if layer == 0 else self._ed2
        ed.set_palette(palette)

    def set_preview_image(self, img_bgr: np.ndarray):
        """
        Called by main_window after a pattern is generated.
        Updates the Generator Output tab in BOTH palette editors.
        """
        self._ed1.set_preview(img_bgr)
        self._ed2.set_preview(img_bgr)
