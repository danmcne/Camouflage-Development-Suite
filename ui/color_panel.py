"""
ColorPanel – palette editor with separate full-size image viewer.

Architecture
────────────
PaletteImageViewer  – standalone widget (placed in MainWindow's right panel
                      when the Palette tab is active).  Contains:
                        • "Output" tab  – mirrors the current generator output
                        • "Img N" tabs  – user-uploaded source images
                      Emits color_picked(QColor) when in pick mode.
                      Also owns the Add / Clear controls.

_PaletteEditor      – swatch grid, colour count, preset picker, pick-mode
                      toggle, k-means button.  All image display is delegated
                      to PaletteImageViewer.  Emits:
                        • changed(ColorPalette)
                        • pick_mode_toggled(bool)
                        • kmeans_requested(int)

ColorPanel          – outer widget with Layer 1 / Layer 2 tabs.
                      Owns the PaletteImageViewer and wires everything together:
                        • routes color_picked → active editor
                        • routes pick_mode_toggled → viewer
                        • handles kmeans_requested using viewer's image list
                        • exposes image_viewer property for MainWindow
"""
from __future__ import annotations
import os
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QPushButton, QColorDialog,
    QFileDialog, QFrame, QSizePolicy, QScrollArea,
    QToolButton, QMessageBox, QTabWidget, QComboBox,
)
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint

from core.palette import ColorPalette
from config.defaults import APP


# ── helpers ───────────────────────────────────────────────────────────────────

def _np_to_qpixmap(img_rgb: np.ndarray) -> QPixmap:
    h, w = img_rgb.shape[:2]
    img_rgb = np.ascontiguousarray(img_rgb)
    qimg    = QImage(img_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ── image picker (full-size display + magnifier) ──────────────────────────────

class ImagePickerWidget(QWidget):
    """
    Displays one image scaled-to-fit.  In pick mode the cursor carries a
    coloured disc (magnifier) showing the pixel colour underneath; click
    to emit color_picked(QColor).
    """
    color_picked = pyqtSignal(object)   # QColor

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img_np: np.ndarray | None = None   # H×W×3 RGB uint8
        self._pixmap: QPixmap | None    = None
        self._pick_mode   = False
        self._hover_pos: QPoint | None   = None
        self._hover_color: QColor | None = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ── public ────────────────────────────────────────────────────────────────

    def set_image_from_path(self, path: str) -> bool:
        import cv2
        img = cv2.imread(path)
        if img is None:
            return False
        return self._load_rgb(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def set_image_from_array(self, img_bgr: np.ndarray | None) -> bool:
        import cv2
        if img_bgr is None:
            self._img_np = None; self._pixmap = None; self.update(); return False
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        return self._load_rgb(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    def set_pick_mode(self, enabled: bool):
        self._pick_mode = enabled
        self._hover_pos = None; self._hover_color = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled
                       else Qt.CursorShape.ArrowCursor)
        self.update()

    # ── private ───────────────────────────────────────────────────────────────

    def _load_rgb(self, rgb: np.ndarray) -> bool:
        self._img_np = rgb
        self._pixmap = _np_to_qpixmap(rgb)
        self.update(); return True

    def _display_rect(self) -> QRect:
        if not self._pixmap:
            return QRect(0, 0, self.width(), self.height())
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if pw == 0 or ph == 0:
            return QRect(0, 0, ww, wh)
        scale  = min(ww / pw, wh / ph)
        dw, dh = int(pw * scale), int(ph * scale)
        return QRect((ww - dw) // 2, (wh - dh) // 2, dw, dh)

    def _color_at(self, pos: QPoint) -> QColor | None:
        if self._img_np is None: return None
        dr = self._display_rect()
        if not dr.contains(pos): return None
        ih, iw = self._img_np.shape[:2]
        px = max(0, min(int((pos.x()-dr.x()) / dr.width()  * iw), iw-1))
        py = max(0, min(int((pos.y()-dr.y()) / dr.height() * ih), ih-1))
        r, g, b = self._img_np[py, px]
        return QColor(int(r), int(g), int(b))

    # ── Qt events ─────────────────────────────────────────────────────────────

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        if self._pixmap:
            p.drawPixmap(self._display_rect(), self._pixmap)
        else:
            p.fillRect(0, 0, W, H, QColor(45, 45, 45))
            p.setPen(QColor(120, 120, 120))
            p.drawText(QRect(0, 0, W, H), Qt.AlignmentFlag.AlignCenter,
                       "No image loaded")
        if self._pick_mode and self._hover_pos and self._hover_color:
            cx, cy = self._hover_pos.x(), self._hover_pos.y()
            R = 18
            p.setBrush(QBrush(self._hover_color))
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawEllipse(cx - R, cy - R, R*2, R*2)
            p.setPen(QPen(QColor(255, 255, 255), 1))
            for dx, dy, ex, ey in [(-R-8,0,-R-2,0),(R+2,0,R+8,0),
                                    (0,-R-8,0,-R-2),(0,R+2,0,R+8)]:
                p.drawLine(cx+dx, cy+dy, cx+ex, cy+ey)
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
            if c: self.color_picked.emit(c)


# ── palette image viewer (goes in MainWindow's right panel) ───────────────────

class PaletteImageViewer(QWidget):
    """
    Full-size image viewer: 'Output' tab + up to 10 uploaded source images.
    Placed in the right panel of MainWindow when the Palette tab is active.
    Emits color_picked(QColor) whenever the user clicks in pick mode.
    """
    color_picked = pyqtSignal(object)   # QColor

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_images: list[str] = []
        self._pickers: list[ImagePickerWidget] = []
        self._pick_mode = False
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4); root.setSpacing(4)

        # Controls row
        ctrl = QHBoxLayout()
        self._add_btn = QPushButton("📂 Add images…")
        self._add_btn.setToolTip("Add source images for colour picking / k-means (max 10).")
        self._add_btn.clicked.connect(self._add_images)
        ctrl.addWidget(self._add_btn)
        self._clr_btn = QPushButton("🗑 Clear all")
        self._clr_btn.setToolTip("Remove all uploaded source images.")
        self._clr_btn.clicked.connect(self._clear_images)
        ctrl.addWidget(self._clr_btn)
        self._count_lbl = QLabel("No source images")
        self._count_lbl.setStyleSheet("color:#888;font-size:10px;")
        ctrl.addWidget(self._count_lbl, 1)
        root.addLayout(ctrl)

        # Tab widget: Output + source images
        self._tabs = QTabWidget()
        self._out_picker = ImagePickerWidget()
        self._out_picker.color_picked.connect(self.color_picked)
        self._tabs.addTab(self._out_picker, "Generator Output")
        root.addWidget(self._tabs, 1)

    # ── public ────────────────────────────────────────────────────────────────

    def set_pick_mode(self, enabled: bool):
        self._pick_mode = enabled
        self._out_picker.set_pick_mode(enabled)
        for pk in self._pickers:
            pk.set_pick_mode(enabled)

    def set_preview(self, img_bgr: np.ndarray):
        """Update the Generator Output tab with the current pattern."""
        self._out_picker.set_image_from_array(img_bgr)

    def get_source_images(self) -> list[str]:
        return list(self._source_images)

    # ── image management ──────────────────────────────────────────────────────

    def _add_images(self):
        remaining = 10 - len(self._source_images)
        if remaining <= 0:
            QMessageBox.information(self, "Limit reached",
                                    "Maximum 10 source images. Clear some first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add source images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if not paths: return
        paths = paths[:remaining]
        for path in paths:
            if path not in self._source_images:
                self._source_images.append(path)
        self._rebuild_tabs()

    def _clear_images(self):
        self._source_images.clear()
        self._rebuild_tabs()

    def _rebuild_tabs(self):
        while self._tabs.count() > 1:
            w = self._tabs.widget(1)
            self._tabs.removeTab(1)
            if w: w.deleteLater()
        self._pickers.clear()
        for i, path in enumerate(self._source_images):
            pk = ImagePickerWidget()
            pk.set_image_from_path(path)
            pk.set_pick_mode(self._pick_mode)
            pk.color_picked.connect(self.color_picked)
            name = os.path.basename(path)
            if len(name) > 14: name = name[:12] + "…"
            self._tabs.addTab(pk, f"Img {i+1}")
            self._tabs.setTabToolTip(i+1, path)
            self._pickers.append(pk)
        n = len(self._source_images)
        self._count_lbl.setText(
            f"{n}/10 source image{'s' if n!=1 else ''}" if n else "No source images")


# ── swatch ────────────────────────────────────────────────────────────────────

class SwatchWidget(QFrame):
    color_clicked = pyqtSignal(int)
    lock_toggled  = pyqtSignal(int, bool)

    def __init__(self, index, hex_color, locked=False, parent=None):
        super().__init__(parent)
        self.index     = index
        self._hex      = hex_color
        self._locked   = locked
        self._selected = False
        self.setFixedSize(66, 78)
        self.setFrameShape(QFrame.Shape.Box)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2); lay.setSpacing(2)
        self._btn = QPushButton()
        self._btn.setFixedHeight(44)
        self._btn.setToolTip("Click to edit colour (or pick from image in pick mode).")
        self._btn.clicked.connect(lambda: self.color_clicked.emit(self.index))
        lay.addWidget(self._btn)
        self._lock = QToolButton()
        self._lock.setCheckable(True); self._lock.setChecked(locked)
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

    def set_color(self, h: str):      self._hex = h;  self._apply()
    def set_locked(self, v: bool):    self._locked = v; self._apply()
    def set_selected(self, v: bool):  self._selected = v; self._apply()

    def _on_lock(self, c):
        self._locked = c; self._apply(); self.lock_toggled.emit(self.index, c)


# ── palette editor ────────────────────────────────────────────────────────────

class _PaletteEditor(QWidget):
    """
    Swatch grid + controls.  Image display is handled externally by
    PaletteImageViewer; this widget emits signals for ColorPanel to act on.
    """
    changed          = pyqtSignal(object)   # ColorPalette
    pick_mode_toggled= pyqtSignal(bool)
    kmeans_requested = pyqtSignal(int)      # n_colors

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

    def __init__(self, palette: ColorPalette, parent=None):
        super().__init__(parent)
        self._palette   = palette
        self._swatches: list[SwatchWidget] = []
        self._selected  = 0
        self._pick_mode = False
        self._build()
        self._rebuild_swatches()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(6)

        # Colour count + preset
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Colours:"))
        self._spin = QSpinBox()
        self._spin.setRange(2, APP["max_palette_colors"])
        self._spin.setValue(len(self._palette))
        self._spin.valueChanged.connect(self._on_count_changed)
        r1.addWidget(self._spin)
        r1.addSpacing(6)
        r1.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        for lbl, _ in self._PRESETS:
            self._preset_combo.addItem(lbl)
        r1.addWidget(self._preset_combo, 1)
        apply_btn = QPushButton("Apply"); apply_btn.setFixedWidth(46)
        apply_btn.clicked.connect(self._apply_preset)
        r1.addWidget(apply_btn)
        root.addLayout(r1)

        # Swatch scroll
        self._sw_container = QWidget()
        self._sw_grid = QGridLayout(self._sw_container)
        self._sw_grid.setSpacing(3)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(175)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._sw_container)
        root.addWidget(scroll)

        # Pick + K-means row
        r2 = QHBoxLayout()
        self._pick_btn = QPushButton("🎯 Pick from image")
        self._pick_btn.setCheckable(True)
        self._pick_btn.setToolTip(
            "Switch to the Palette tab's image viewer and hover over an image. "
            "The magnifier shows the colour under the cursor; click to send it "
            "to the selected swatch (locked swatches are skipped).")
        self._pick_btn.toggled.connect(self._on_pick_toggled)
        r2.addWidget(self._pick_btn, 1)
        km_btn = QPushButton("📊 K-means")
        km_btn.setToolTip(
            "Extract palette by k-means over all uploaded source images. "
            "Locked swatches are preserved.")
        km_btn.clicked.connect(lambda: self.kmeans_requested.emit(self._spin.value()))
        r2.addWidget(km_btn)
        root.addLayout(r2)
        root.addStretch()
        
        

    # ── swatches ──────────────────────────────────────────────────────────────

    def _rebuild_swatches(self):
        for sw in self._swatches: sw.deleteLater()
        self._swatches.clear()
        self._selected = max(0, min(self._selected, len(self._palette) - 1))
        cols = 5
        for i, h in enumerate(self._palette):
            sw = SwatchWidget(i, h, self._palette.is_locked(i))
            sw.set_selected(i == self._selected)
            sw.color_clicked.connect(self._on_swatch_clicked)
            sw.lock_toggled.connect(self._on_lock_toggled)
            self._sw_grid.addWidget(sw, i // cols, i % cols)
            self._swatches.append(sw)

    def _select_swatch(self, idx: int):
        if idx == self._selected: return
        if 0 <= self._selected < len(self._swatches):
            self._swatches[self._selected].set_selected(False)
        self._selected = idx
        if 0 <= idx < len(self._swatches):
            self._swatches[idx].set_selected(True)

    def _on_swatch_clicked(self, idx: int):
        self._select_swatch(idx)
        if self._palette.is_locked(idx) or self._pick_mode:
            return
        chosen = QColorDialog.getColor(QColor(self._palette[idx]), self, "Choose colour")
        if chosen.isValid():
            h = chosen.name().upper()
            self._palette.set_color(idx, h)
            self._swatches[idx].set_color(h)
            self.changed.emit(self._palette)

    def _on_lock_toggled(self, idx: int, locked: bool):
        self._palette.set_locked(idx, locked)

    def _on_pick_toggled(self, enabled: bool):
        self._pick_mode = enabled
        self._pick_btn.setText(
            "🎯 Pick ON — click image" if enabled else "🎯 Pick from image")
        self.pick_mode_toggled.emit(enabled)

    # ── public: receive a colour from the image viewer ────────────────────────

    def receive_color(self, color: QColor):
        """Called by ColorPanel when the image viewer emits color_picked."""
        idx = self._selected
        n   = len(self._palette)
        # Find first unlocked swatch from selected onwards
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
        # Advance to next unlocked
        for offset in range(1, n):
            candidate = (idx + offset) % n
            if not self._palette.is_locked(candidate):
                self._select_swatch(candidate)
                break
        self.changed.emit(self._palette)

    # ── count / preset ────────────────────────────────────────────────────────

    def _on_count_changed(self, n: int):
        # ColorPanel handles re-running k-means; here just resize
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

    def get_palette(self) -> ColorPalette: return self._palette

    def set_palette(self, palette: ColorPalette):
        self._palette = palette
        self._spin.blockSignals(True)
        self._spin.setValue(len(palette))
        self._spin.blockSignals(False)
        self._rebuild_swatches()

    def apply_new_palette(self, new_pal: ColorPalette):
        """Apply a freshly-extracted palette, preserving locked swatches."""
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

    def get_pick_mode(self) -> bool: return self._pick_mode

    def reset_pick_btn(self):
        """Uncheck pick button without emitting signal (called on tab switch)."""
        self._pick_btn.blockSignals(True)
        self._pick_btn.setChecked(False)
        self._pick_btn.setText("🎯 Pick from image")
        self._pick_btn.blockSignals(False)
        self._pick_mode = False


# ── outer panel ───────────────────────────────────────────────────────────────

class ColorPanel(QWidget):
    palette_changed  = pyqtSignal(object, int)   # (ColorPalette, layer 0|1)
    wants_image_view = pyqtSignal(bool)           # True = show viewer, False = hide

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ed1 = _PaletteEditor(ColorPalette.military_preset())
        self._ed2 = _PaletteEditor(ColorPalette.urban_preset())

        # Shared image viewer (placed in MainWindow's right panel)
        self._viewer = PaletteImageViewer()
        self._viewer.color_picked.connect(self._on_color_picked)

        # Wire editor signals
        self._ed1.changed.connect(lambda p: self.palette_changed.emit(p, 0))
        self._ed2.changed.connect(lambda p: self.palette_changed.emit(p, 1))
        self._ed1.pick_mode_toggled.connect(self._on_pick_mode_toggled)
        self._ed2.pick_mode_toggled.connect(self._on_pick_mode_toggled)
        self._ed1.kmeans_requested.connect(self._on_kmeans_requested)
        self._ed2.kmeans_requested.connect(self._on_kmeans_requested)

        # Inner L1 / L2 tab widget
        self._layer_tabs = QTabWidget()
        self._layer_tabs.addTab(self._ed1, "Layer 1")
        self._layer_tabs.addTab(self._ed2, "Layer 2")
        self._layer_tabs.currentChanged.connect(self._on_layer_tab_changed)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._layer_tabs)

    # ── routing ───────────────────────────────────────────────────────────────

    def _active_editor(self) -> _PaletteEditor:
        return self._ed1 if self._layer_tabs.currentIndex() == 0 else self._ed2

    def _on_color_picked(self, color: QColor):
        self._active_editor().receive_color(color)

    def _on_pick_mode_toggled(self, enabled: bool):
        self._viewer.set_pick_mode(enabled)

    def _on_layer_tab_changed(self, _idx: int):
        # When switching L1↔L2, turn off pick mode on both editors and viewer
        for ed in (self._ed1, self._ed2):
            if ed.get_pick_mode():
                ed.reset_pick_btn()
        self._viewer.set_pick_mode(False)

    def _on_kmeans_requested(self, n_colors: int):
        paths = self._viewer.get_source_images()
        if not paths:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No images",
                "Add source images first using '📂 Add images…' in the image viewer.")
            return
        try:
            new_pal = ColorPalette.from_images_kmeans(paths, n_colors)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "K-means failed", str(e))
            return
        self._active_editor().apply_new_palette(new_pal)

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def image_viewer(self) -> PaletteImageViewer:
        """The full-size image viewer widget; placed in MainWindow's right panel."""
        return self._viewer

    def set_preview_image(self, img_bgr: np.ndarray):
        """Called by MainWindow after each generation."""
        self._viewer.set_preview(img_bgr)

    def get_palette(self, layer: int = 0) -> ColorPalette:
        return self._ed1.get_palette() if layer == 0 else self._ed2.get_palette()

    def set_palette(self, palette: ColorPalette, layer: int = 0):
        ed = self._ed1 if layer == 0 else self._ed2
        ed.set_palette(palette)
