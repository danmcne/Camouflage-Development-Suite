"""
MainWindow – wires dual palettes, inspect mode, layer-2 palette, and shape folder hint.
"""
from __future__ import annotations
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
    QTabWidget, QFileDialog, QMessageBox, QApplication,
    QDialog, QVBoxLayout, QFormLayout, QComboBox,
    QDialogButtonBox, QLabel,
)
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence

from ui.color_panel      import ColorPanel
from ui.generator_panel  import GeneratorPanel
from ui.evolution_panel  import EvolutionPanel
from ui.preview_canvas   import PreviewCanvas
from evolution.background_manager import BackgroundManager
from core.palette        import ColorPalette
from config.defaults     import APP
from utils.image_ops     import EXPORT_SIZES


# ── layer blend helper ────────────────────────────────────────────────────────

def _blend_layers(base, overlay, mode, opacity):
    """
    Composite overlay onto base (both BGR or BGRA numpy arrays).
    overlay alpha channel is respected when present.
    Returns BGR uint8.
    """
    if overlay.ndim == 3 and overlay.shape[2] == 4:
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        ov    = overlay[:, :, :3]
    else:
        alpha = np.ones((*overlay.shape[:2], 1), dtype=np.float32)
        ov    = overlay

    b = base.astype(np.float32) / 255.0
    o = ov.astype(np.float32) / 255.0

    if base.shape[:2] != ov.shape[:2]:
        o     = cv2.resize(o,     (base.shape[1], base.shape[0]))
        alpha = cv2.resize(alpha[:, :, 0], (base.shape[1], base.shape[0]))[:, :, np.newaxis]

    if   mode == "multiply":   blended = b * o
    elif mode == "screen":     blended = 1 - (1-b) * (1-o)
    elif mode == "overlay":    blended = np.where(b < 0.5, 2*b*o, 1-2*(1-b)*(1-o))
    elif mode == "soft_light": blended = (1-2*o)*b*b + 2*o*b
    else:                      blended = o

    eff = alpha * opacity
    return (np.clip(b*(1-eff) + blended*eff, 0, 1) * 255).astype(np.uint8)


# ── generate worker ───────────────────────────────────────────────────────────

class _GenerateWorker(QObject):
    finished = pyqtSignal(object)   # emits BGR or BGRA ndarray
    error    = pyqtSignal(str)

    def __init__(self, gen_name, params, colors_rgb, size, second=None):
        super().__init__()
        self._gen_name   = gen_name
        self._params     = params
        self._colors_rgb = colors_rgb
        self._size       = size
        self._second     = second   # dict with generator/params/blend/opacity/palette

    def run(self):
        try:    self._do_run()
        except Exception:
            import traceback; self.error.emit(traceback.format_exc())

    def _do_run(self):
        from generators import get_generator
        gen = get_generator(self._gen_name)
        img = gen.generate(self._size[0], self._size[1], self._colors_rgb, self._params)

        if self._second:
            gen2    = get_generator(self._second["generator"])
            colors2 = self._second["palette"].as_rgb()
            img2    = gen2.generate(self._size[0], self._size[1],
                                    colors2, self._second["params"])
            # Blend L2 over L1 (L1 alpha already stripped for base)
            base = (cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    if img.ndim == 3 and img.shape[2] == 4 else img)
            img  = _blend_layers(base, img2,
                                 self._second["blend"], self._second["opacity"])

        # Preserve BGRA if no second layer; callers that can't handle alpha
        # will composite themselves.  The preview and export both handle BGRA.
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        self.finished.emit(img)


# ── export dialog ─────────────────────────────────────────────────────────────

class _ExportDialog(QDialog):
    def __init__(self, has_alpha: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export options")
        self.setModal(True)

        layout = QVBoxLayout(self)
        form   = QFormLayout()

        self._size_combo = QComboBox()
        for label in EXPORT_SIZES:
            self._size_combo.addItem(label)
        self._size_combo.setCurrentText("2048 × 2048")
        form.addRow("Size:", self._size_combo)

        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(["PNG", "JPEG", "TIFF"])
        if not has_alpha:
            self._fmt_combo.setCurrentText("PNG")
        form.addRow("Format:", self._fmt_combo)

        if has_alpha:
            note = QLabel("ℹ PNG and TIFF preserve transparency. JPEG composites over white.")
            note.setWordWrap(True)
            note.setStyleSheet("color:#aaa;font-size:10px;")
            layout.addWidget(note)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_size(self) -> tuple[int, int]:
        return EXPORT_SIZES[self._size_combo.currentText()]

    def selected_format(self) -> str:
        return self._fmt_combo.currentText()


# ── main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camouflage Development Tool")
        self.resize(1280, 820)
        self.setMinimumSize(900, 600)

        self._current_pattern: np.ndarray | None = None   # BGR or BGRA
        self._bg_manager  = BackgroundManager()
        self._gen_thread: QThread | None = None
        self._normal_sizes = [400, 880]

        self._build_ui()
        self._build_menu()
        self._wire_signals()

        self.statusBar().showMessage("Ready.  Choose a generator and click Generate.")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4); root.setSpacing(0)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._tabs = QTabWidget()
        self._tabs.setMinimumWidth(320); self._tabs.setMaximumWidth(500)

        self._color_panel = ColorPanel()
        self._gen_panel   = GeneratorPanel()
        self._evo_panel   = EvolutionPanel(self._bg_manager)

        self._tabs.addTab(self._color_panel, "🎨 Palette")
        self._tabs.addTab(self._gen_panel,   "⚙ Generator")
        self._tabs.addTab(self._evo_panel,   "🧬 Evolution")

        self._splitter.addWidget(self._tabs)
        self._preview = PreviewCanvas()
        self._splitter.addWidget(self._preview)
        self._splitter.setStretchFactor(0, 0); self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes(self._normal_sizes)
        root.addWidget(self._splitter)

    def _build_menu(self):
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        ae = QAction("&Export pattern…", self); ae.setShortcut(QKeySequence("Ctrl+S"))
        ae.triggered.connect(self._export_pattern); fm.addAction(ae)
        fm.addSeparator()
        aq = QAction("&Quit", self); aq.setShortcut(QKeySequence("Ctrl+Q"))
        aq.triggered.connect(QApplication.quit); fm.addAction(aq)

        bm = mb.addMenu("&Backgrounds")
        af = QAction("Add background &folder…", self)
        af.triggered.connect(self._add_bg_folder); bm.addAction(af)
        ai = QAction("Add background &image…", self)
        ai.triggered.connect(self._add_bg_file);  bm.addAction(ai)

        hm = mb.addMenu("&Help")
        ab = QAction("&About", self); ab.triggered.connect(self._show_about)
        hm.addAction(ab)

    def _wire_signals(self):
        self._color_panel.palette_changed.connect(self._on_palette_changed)

        self._gen_panel.generate_requested.connect(self._on_generate_requested)
        self._gen_panel.params_changed.connect(
            lambda name, params: self._evo_panel.set_seed_params(name, params))
        self._preview.request_generate.connect(
            lambda: self._on_generate_requested(
                self._gen_panel.get_generator_name(),
                self._gen_panel.get_params()))

        # Inject L2 provider so evolution panel reads generator panel's L2 config
        self._evo_panel._layer2_provider = self._gen_panel.get_second_layer_config

        self._evo_panel.candidate_chosen.connect(self._on_candidate_chosen)
        self._evo_panel.wants_fullwidth.connect(self._on_evo_fullwidth)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._gen_panel.set_palette(self._color_panel.get_palette(0), 0)
        self._gen_panel.set_palette(self._color_panel.get_palette(1), 1)
        self._evo_panel.set_palette(self._color_panel.get_palette(0))

    # ── tab management ────────────────────────────────────────────────────────

    def _on_tab_changed(self, idx):
        evo_idx = self._tabs.indexOf(self._evo_panel)
        if idx == evo_idx: self._evo_panel.on_tab_activated()
        else:              self._evo_panel.on_tab_deactivated()

    def _on_evo_fullwidth(self, expand):
        if expand:
            self._normal_sizes = self._splitter.sizes()
            total = sum(self._splitter.sizes())
            self._splitter.setSizes([total, 0])
            self._tabs.setMaximumWidth(16777215)
            self._preview.setVisible(False)
        else:
            self._preview.setVisible(True)
            self._tabs.setMaximumWidth(500)
            self._splitter.setSizes(self._normal_sizes)

    # ── palette routing ───────────────────────────────────────────────────────

    def _on_palette_changed(self, palette: ColorPalette, layer: int):
        self._gen_panel.set_palette(palette, layer)
        if layer == 0:
            self._evo_panel.set_palette(palette)

    # ── generate ─────────────────────────────────────────────────────────────

    def _on_generate_requested(self, gen_name: str, params: dict):
        if self._gen_thread is not None:
            try:
                if self._gen_thread.isRunning():
                    return
            except RuntimeError:
                self._gen_thread = None

        colors_rgb = self._color_panel.get_palette(0).as_rgb()
        second     = self._gen_panel.get_second_layer_config()

        self._gen_thread = QThread()
        self._worker     = _GenerateWorker(
            gen_name, params, colors_rgb, APP["preview_size"], second=second)

        self._worker.moveToThread(self._gen_thread)
        self._gen_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_generate_done)
        self._worker.error.connect(self._on_generate_error)
        self._worker.finished.connect(self._gen_thread.quit)
        self._worker.error.connect(self._gen_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._gen_thread.finished.connect(self._gen_thread.deleteLater)
        self._gen_thread.finished.connect(self._on_thread_finished)
        self._gen_thread.start()

        note = f" + {second['generator']} [{second['blend']}]" if second else ""
        self.statusBar().showMessage(f"Generating {gen_name}{note}…")

    def _on_thread_finished(self):
        self._gen_thread = None
        self._worker     = None

    def _on_generate_done(self, img: np.ndarray):
        self._current_pattern = img   # keep BGR or BGRA as-is

        # For preview, composite BGRA over white so the canvas looks correct
        preview_img = self._bgra_to_bgr_for_display(img)
        self._preview.set_pattern(preview_img)

        # Update colour panel "Generator Output" tabs for manual colour picking
        self._color_panel.set_preview_image(preview_img)

        bg = self._bg_manager.get_active()
        self._preview.set_background(bg)
        if bg is not None:
            from core.fitness import composite_fitness
            scores = composite_fitness(preview_img, bg)
            self._preview.set_fitness(scores)
            self.statusBar().showMessage(
                f"Done — colour={scores['color']:.3f}  texture={scores['texture']:.3f}  "
                f"disruption={scores['disruption']:.3f}  total={scores['total']:.3f}")
        else:
            has_alpha = img.ndim == 3 and img.shape[2] == 4
            self.statusBar().showMessage(
                f"Done.{'  (transparent layer — add background for fitness.)'
                        if has_alpha else '  Add a background to see fitness.'}")

    def _on_generate_error(self, msg: str):
        self.statusBar().showMessage("Generation failed — see console.")
        print("=== Generate error ===\n", msg)

    @staticmethod
    def _bgra_to_bgr_for_display(img: np.ndarray) -> np.ndarray:
        """Composite BGRA over a white background for display/fitness use."""
        if img.ndim == 3 and img.shape[2] == 4:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            a   = img[:, :, 3:4].astype(np.float32) / 255.0
            bgr = img[:, :, :3].astype(np.float32)
            return (bgr * a + 255.0 * (1.0 - a)).clip(0, 255).astype(np.uint8)
        return img

    # ── candidate chosen (from evolution) ────────────────────────────────────

    def _on_candidate_chosen(self, img: np.ndarray, scores: dict,
                             gen_name: str, params: dict,
                             gen_name2: str, params2: dict):
        self._current_pattern = img
        preview_img = self._bgra_to_bgr_for_display(img)
        self._preview.set_pattern(preview_img)
        bg = self._bg_manager.get_active()
        self._preview.set_background(bg)
        self._preview.set_fitness(scores)

        self._gen_panel.load_pattern(gen_name, params)
        if gen_name2:
            self._gen_panel.load_pattern_layer2(gen_name2, params2)

        l2_tag = f" + {gen_name2}" if gen_name2 else ""
        self.statusBar().showMessage(
            f"Candidate [{gen_name}{l2_tag}] — total={scores.get('total',0):.3f}  "
            f"(Generator tab updated)")

    # ── backgrounds ──────────────────────────────────────────────────────────

    def _add_bg_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select background folder")
        if folder:
            self._bg_manager.add_folder(folder)
            self.statusBar().showMessage(f"{len(self._bg_manager)} background(s) loaded.")
            self._preview.set_background(self._bg_manager.get_active())

    def _add_bg_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select background image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if path:
            self._bg_manager.add_file(path)
            self._bg_manager.set_active(len(self._bg_manager) - 1)
            self._preview.set_background(self._bg_manager.get_active())

    # ── export ────────────────────────────────────────────────────────────────

    def _export_pattern(self):
        if self._current_pattern is None:
            QMessageBox.information(self, "Nothing to export", "Generate a pattern first.")
            return

        has_alpha = (self._current_pattern.ndim == 3 and
                     self._current_pattern.shape[2] == 4 and
                     self._current_pattern[:, :, 3].min() < 255)

        dlg = _ExportDialog(has_alpha, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        size   = dlg.selected_size()
        fmt    = dlg.selected_format()
        ext    = {"PNG": ".png", "JPEG": ".jpg", "TIFF": ".tiff"}[fmt]
        filter_str = {"PNG":  "PNG (*.png)",
                      "JPEG": "JPEG (*.jpg *.jpeg)",
                      "TIFF": "TIFF (*.tiff *.tif)"}[fmt]

        path, _ = QFileDialog.getSaveFileName(
            self, "Export pattern", f"pattern{ext}", filter_str)
        if not path:
            return

        from utils.image_ops import export_image
        export_image(self._current_pattern, path, size=size)
        self.statusBar().showMessage(f"Exported {size[0]}×{size[1]} {fmt} → {path}")

    def _show_about(self):
        QMessageBox.about(self, "Camouflage Development Tool",
            "<b>Camouflage Dev Tool</b> v0.5<br><br>"
            "Generators: Noise · Blur-Sharp · Gray-Scott RD · L-System · "
            "Recursive Fractal · Urban Geometric · Collage · Dazzle · Plaid<br>"
            "All outputs seamlessly tileable (toroidal).<br>"
            "Dual palette (Layer 1 / Layer 2). Two-layer evolution.<br>"
            "Export: PNG (with alpha) · JPEG · TIFF at up to 8192×8192.<br>"
            "Built with PyQt6 · NumPy · OpenCV · scikit-learn")
