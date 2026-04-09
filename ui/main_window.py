"""
MainWindow – wires dual palettes, inspect mode, layer-2 palette, and shape folder hint.
"""
from __future__ import annotations
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
    QTabWidget, QFileDialog, QMessageBox, QApplication,
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


# ── blend helper ─────────────────────────────────────────────────────────────

def _blend_layers(base, overlay, mode, opacity):
    if overlay.ndim == 3 and overlay.shape[2] == 4:
        alpha = overlay[:,:,3:4].astype(np.float32)/255.0
        ov    = overlay[:,:,:3]
    else:
        alpha = np.ones((*overlay.shape[:2],1), dtype=np.float32)
        ov    = overlay

    b = base.astype(np.float32)/255.0
    o = ov.astype(np.float32)/255.0

    if base.shape[:2] != ov.shape[:2]:
        o     = cv2.resize(o, (base.shape[1], base.shape[0]))
        alpha = cv2.resize(alpha[:,:,0], (base.shape[1], base.shape[0]))[:,:,np.newaxis]

    if   mode == "multiply":   blended = b*o
    elif mode == "screen":     blended = 1-(1-b)*(1-o)
    elif mode == "overlay":    blended = np.where(b<0.5, 2*b*o, 1-2*(1-b)*(1-o))
    elif mode == "soft_light": blended = (1-2*o)*b*b + 2*o*b
    else:                      blended = o

    eff = alpha * opacity
    return (np.clip(b*(1-eff)+blended*eff, 0, 1)*255).astype(np.uint8)


# ── generate worker ───────────────────────────────────────────────────────────

class _GenerateWorker(QObject):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, gen_name, params, colors_rgb, size, second=None):
        super().__init__()
        self._gen_name   = gen_name
        self._params     = params
        self._colors_rgb = colors_rgb
        self._size       = size
        self._second     = second

    def run(self):
        try:    
            self._do_run()
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
            img = _blend_layers(
                cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.ndim==3 and img.shape[2]==4 else img,
                img2, self._second["blend"], self._second["opacity"])

        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.finished.emit(img)


# ── main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camouflage Development Tool")
        self.resize(1280, 820)
        self.setMinimumSize(900, 600)

        self._current_pattern: np.ndarray | None = None
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
        root = QHBoxLayout(central); root.setContentsMargins(4,4,4,4); root.setSpacing(0)

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
        self._splitter.setStretchFactor(0,0); self._splitter.setStretchFactor(1,1)
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
        # Dual palette → generator panel + evolution panel
        self._color_panel.palette_changed.connect(self._on_palette_changed)

        # Generator panel signals
        self._gen_panel.generate_requested.connect(self._on_generate_requested)
        self._gen_panel.params_changed.connect(
            lambda name, params: self._evo_panel.set_seed_params(name, params))
        self._preview.request_generate.connect(
            lambda: self._on_generate_requested(
                self._gen_panel.get_generator_name(),
                self._gen_panel.get_params()))

        # Evolution: candidate chosen → preview + (if inspect) → generator panel
        self._evo_panel.candidate_chosen.connect(self._on_candidate_chosen)
        self._evo_panel.wants_fullwidth.connect(self._on_evo_fullwidth)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Push initial palettes
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
        self._worker = _GenerateWorker(
            gen_name, params, colors_rgb, APP["preview_size"], second=second
        )

        self._worker.moveToThread(self._gen_thread)

        self._gen_thread.started.connect(self._worker.run)

        self._worker.finished.connect(self._on_generate_done)
        self._worker.error.connect(self._on_generate_error)

        self._worker.finished.connect(self._gen_thread.quit)
        self._worker.error.connect(self._gen_thread.quit)

        self._worker.finished.connect(self._worker.deleteLater)
        self._gen_thread.finished.connect(self._gen_thread.deleteLater)

        # IMPORTANT: cleanup
        self._gen_thread.finished.connect(self._on_thread_finished)

        self._gen_thread.start()

        note = f" + {second['generator']} [{second['blend']}]" if second else ""
        self.statusBar().showMessage(f"Generating {gen_name}{note}…")
    
    
    def _on_thread_finished(self):
        self._gen_thread = None
        self._worker = None

    def _on_generate_done(self, img):
        self._current_pattern = img
        self._preview.set_pattern(img)
        bg = self._bg_manager.get_active()
        self._preview.set_background(bg)
        if bg is not None:
            from core.fitness import composite_fitness
            scores = composite_fitness(img, bg)
            self._preview.set_fitness(scores)
            self.statusBar().showMessage(
                f"Done — colour={scores['color']:.3f}  texture={scores['texture']:.3f}  "
                f"disruption={scores['disruption']:.3f}  total={scores['total']:.3f}")
        else:
            self.statusBar().showMessage("Done. (Add a background to see fitness.)")

    def _on_generate_error(self, msg):
        self.statusBar().showMessage("Generation failed — see console.")
        print("=== Generate error ===\n", msg)

    # ── candidate chosen (from evolution) ────────────────────────────────────

    def _on_candidate_chosen(self, img: np.ndarray, scores: dict, gen_name: str, params: dict):
        self._current_pattern = img
        self._preview.set_pattern(img)
        bg = self._bg_manager.get_active()
        self._preview.set_background(bg)
        self._preview.set_fitness(scores)

        # Always load into generator panel so user can see/tweak params
        self._gen_panel.load_pattern(gen_name, params)
        # Switch to Generator tab so they can see it immediately
        gen_idx = self._tabs.indexOf(self._gen_panel)
        # Don't auto-switch tab: let user do it. Just update in background.
        # (Auto-switching would close the evolution view unexpectedly.)

        self.statusBar().showMessage(
            f"Candidate [{gen_name}] — total={scores.get('total',0):.3f}  "
            f"(Generator tab updated with its params)")

    # ── backgrounds ──────────────────────────────────────────────────────────

    def _add_bg_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select background folder")
        if folder:
            self._bg_manager.add_folder(folder)
            self.statusBar().showMessage(f"{len(self._bg_manager)} background(s) loaded.")
            self._preview.set_background(self._bg_manager.get_active())

    def _add_bg_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select background image","",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if path:
            self._bg_manager.add_file(path)
            self._bg_manager.set_active(len(self._bg_manager)-1)
            self._preview.set_background(self._bg_manager.get_active())

    # ── export ────────────────────────────────────────────────────────────────

    def _export_pattern(self):
        if self._current_pattern is None:
            QMessageBox.information(self,"Nothing to export","Generate a pattern first.")
            return
        path, _ = QFileDialog.getSaveFileName(self,"Export pattern","pattern.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tiff)")
        if not path: return
        from utils.image_ops import export_png
        export_png(self._current_pattern, path, tuple(APP["export_size"]))
        self.statusBar().showMessage(f"Exported → {path}")

    def _show_about(self):
        QMessageBox.about(self, "Camouflage Development Tool",
            "<b>Camouflage Dev Tool</b> v0.4<br><br>"
            "Generators: Noise · Blur-Sharp · Gray-Scott RD · L-System · "
            "Recursive Fractal · Urban Geometric · Collage<br>"
            "All outputs seamlessly tileable (toroidal). "
            "Dual palette (Layer 1 / Layer 2).<br>"
            "Evolution: moth-kill interactive or automatic fitness. "
            "Inspect mode populates generator panel with candidate params.<br>"
            "Built with PyQt6 · NumPy · OpenCV · scikit-learn")
