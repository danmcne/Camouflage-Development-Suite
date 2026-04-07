"""
MainWindow – top-level QMainWindow.

Key changes from v2:
  • Connects evolution panel's `wants_fullwidth` signal to expand/collapse
    the splitter so the moth canvas fills the window when Evolution tab is open.
  • `_blend_layers` now handles 4-channel (BGRA) overlays from generators that
    have `transparent_bg=True`.
  • Evolution worker errors are shown in the status bar.
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

def _blend_layers(
    base: np.ndarray,
    overlay: np.ndarray,
    mode: str,
    opacity: float,
) -> np.ndarray:
    """
    Blend two images.  Overlay may be BGR (3ch) or BGRA (4ch).
    If BGRA, the alpha channel is used as the blend mask; otherwise fully opaque.
    Output is always BGR uint8.
    """
    if overlay.ndim == 3 and overlay.shape[2] == 4:
        alpha_mask = overlay[:, :, 3:4].astype(np.float32) / 255.0
        overlay_bgr = overlay[:, :, :3]
    else:
        alpha_mask  = np.ones((*overlay.shape[:2], 1), dtype=np.float32)
        overlay_bgr = overlay

    b = base.astype(np.float32) / 255.0
    o = overlay_bgr.astype(np.float32) / 255.0

    if base.shape[:2] != overlay_bgr.shape[:2]:
        o = cv2.resize(o, (base.shape[1], base.shape[0]))
        alpha_mask = cv2.resize(alpha_mask,
                                (base.shape[1], base.shape[0]))[:, :, np.newaxis]

    if mode == "multiply":
        blended = b * o
    elif mode == "screen":
        blended = 1.0 - (1.0 - b) * (1.0 - o)
    elif mode == "overlay":
        blended = np.where(b < 0.5,
                           2.0 * b * o,
                           1.0 - 2.0 * (1.0 - b) * (1.0 - o))
    elif mode == "soft_light":
        blended = (1.0 - 2.0 * o) * b * b + 2.0 * o * b
    else:
        blended = o

    effective = alpha_mask * opacity
    result = b * (1.0 - effective) + blended * effective
    return (np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)


# ── generate worker ───────────────────────────────────────────────────────────

class _GenerateWorker(QObject):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, gen_name, params, colors_rgb, size, second_layer=None):
        super().__init__()
        self._gen_name   = gen_name
        self._params     = params
        self._colors_rgb = colors_rgb
        self._size       = size
        self._second     = second_layer

    def run(self):
        try:
            self._do_run()
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())

    def _do_run(self):
        from generators import get_generator
        gen = get_generator(self._gen_name)
        img = gen.generate(self._size[0], self._size[1],
                           self._colors_rgb, self._params)

        if self._second:
            gen2  = get_generator(self._second["generator"])
            img2  = gen2.generate(self._size[0], self._size[1],
                                  self._colors_rgb, self._second["params"])
            img   = _blend_layers(img, img2,
                                  self._second["blend"],
                                  self._second["opacity"])

        # Ensure output is BGR
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
        self._gen_worker: _GenerateWorker | None = None
        self._normal_sizes = [400, 880]   # splitter sizes when not in evolution

        self._build_ui()
        self._build_menu()
        self._wire_signals()

        self.statusBar().showMessage("Ready.  Choose a generator and click Generate.")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        self._tabs = QTabWidget()
        self._tabs.setMinimumWidth(320)
        self._tabs.setMaximumWidth(500)

        self._color_panel = ColorPanel()
        self._gen_panel   = GeneratorPanel()
        self._evo_panel   = EvolutionPanel(self._bg_manager)

        self._tabs.addTab(self._color_panel, "🎨 Palette")
        self._tabs.addTab(self._gen_panel,   "⚙ Generator")
        self._tabs.addTab(self._evo_panel,   "🧬 Evolution")

        self._splitter.addWidget(self._tabs)

        self._preview = PreviewCanvas()
        self._splitter.addWidget(self._preview)

        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes(self._normal_sizes)

        root.addWidget(self._splitter)

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        act_export = QAction("&Export pattern…", self)
        act_export.setShortcut(QKeySequence("Ctrl+S"))
        act_export.triggered.connect(self._export_pattern)
        file_menu.addAction(act_export)
        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(QApplication.quit)
        file_menu.addAction(act_quit)

        bg_menu = mb.addMenu("&Backgrounds")
        act_folder = QAction("Add background &folder…", self)
        act_folder.triggered.connect(self._add_bg_folder)
        bg_menu.addAction(act_folder)
        act_file = QAction("Add background &image…", self)
        act_file.triggered.connect(self._add_bg_file)
        bg_menu.addAction(act_file)

        help_menu = mb.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _wire_signals(self):
        self._color_panel.palette_changed.connect(self._on_palette_changed)
        self._gen_panel.generate_requested.connect(self._on_generate_requested)
        self._preview.request_generate.connect(
            lambda: self._on_generate_requested(
                self._gen_panel.get_generator_name(),
                self._gen_panel.get_params(),
            )
        )
        self._evo_panel.candidate_chosen.connect(self._on_candidate_chosen)
        self._evo_panel.wants_fullwidth.connect(self._on_evo_fullwidth)
        self._tabs.currentChanged.connect(self._on_tab_changed)

    # ── tab change: expand/collapse for evolution ─────────────────────────────

    def _on_tab_changed(self, index: int):
        evo_index = self._tabs.indexOf(self._evo_panel)
        if index == evo_index:
            self._evo_panel.on_tab_activated()
        else:
            self._evo_panel.on_tab_deactivated()

    def _on_evo_fullwidth(self, expand: bool):
        if expand:
            # Remember current sizes then go full-width
            self._normal_sizes = self._splitter.sizes()
            total = sum(self._splitter.sizes())
            self._splitter.setSizes([total, 0])
            self._splitter.setMaximumWidth(16777215)
            self._tabs.setMaximumWidth(16777215)
            self._preview.setVisible(False)
        else:
            self._preview.setVisible(True)
            self._tabs.setMaximumWidth(500)
            self._splitter.setSizes(self._normal_sizes)

    # ── signal handlers ───────────────────────────────────────────────────────

    def _on_palette_changed(self, palette: ColorPalette):
        self._evo_panel.set_palette(palette)

    def _on_generate_requested(self, gen_name: str, params: dict):
        if self._gen_thread and self._gen_thread.isRunning():
            return

        colors_rgb = self._color_panel.get_palette().as_rgb()
        second     = self._gen_panel.get_second_layer_config()
        size       = APP["preview_size"]

        self._gen_thread = QThread()
        self._gen_worker = _GenerateWorker(
            gen_name, params, colors_rgb, size, second_layer=second
        )
        self._gen_worker.moveToThread(self._gen_thread)
        self._gen_thread.started.connect(self._gen_worker.run)
        self._gen_worker.finished.connect(self._on_generate_done)
        self._gen_worker.error.connect(self._on_generate_error)
        self._gen_worker.finished.connect(self._gen_thread.quit)
        self._gen_worker.error.connect(self._gen_thread.quit)
        self._gen_thread.start()

        note = f" + {second['generator']} [{second['blend']}]" if second else ""
        self.statusBar().showMessage(f"Generating {gen_name}{note}…")

    def _on_generate_done(self, img: np.ndarray):
        self._current_pattern = img
        self._preview.set_pattern(img)
        bg = self._bg_manager.get_active()
        self._preview.set_background(bg)
        if bg is not None:
            from core.fitness import composite_fitness
            scores = composite_fitness(img, bg)
            self._preview.set_fitness(scores)
            self.statusBar().showMessage(
                f"Done — colour={scores['color']:.3f}  "
                f"texture={scores['texture']:.3f}  "
                f"disruption={scores['disruption']:.3f}  "
                f"total={scores['total']:.3f}"
            )
        else:
            self.statusBar().showMessage("Done. (Add a background to see fitness.)")

    def _on_generate_error(self, msg: str):
        self.statusBar().showMessage("Generation failed — see console.")
        print("=== Generate error ===")
        print(msg)

    def _on_candidate_chosen(self, img: np.ndarray, scores: dict):
        self._current_pattern = img
        self._preview.set_pattern(img)
        self._preview.set_fitness(scores)
        bg = self._bg_manager.get_active()
        self._preview.set_background(bg)
        self.statusBar().showMessage(
            f"Candidate selected — total={scores.get('total',0):.3f}"
        )

    # ── background shortcuts ──────────────────────────────────────────────────

    def _add_bg_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select background folder")
        if folder:
            self._bg_manager.add_folder(folder)
            self.statusBar().showMessage(f"{len(self._bg_manager)} background(s) loaded.")
            self._preview.set_background(self._bg_manager.get_active())

    def _add_bg_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select background image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if path:
            self._bg_manager.add_file(path)
            self._bg_manager.set_active(len(self._bg_manager) - 1)
            self._preview.set_background(self._bg_manager.get_active())

    # ── export ────────────────────────────────────────────────────────────────

    def _export_pattern(self):
        if self._current_pattern is None:
            QMessageBox.information(self, "Nothing to export", "Generate a pattern first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export pattern", "pattern.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tiff)"
        )
        if not path:
            return
        from utils.image_ops import export_png
        export_png(self._current_pattern, path, tuple(APP["export_size"]))
        self.statusBar().showMessage(f"Exported → {path}")

    def _show_about(self):
        QMessageBox.about(
            self, "Camouflage Development Tool",
            "<b>Camouflage Development Tool</b> v0.3<br><br>"
            "All outputs are seamlessly tileable.<br>"
            "Generators: Noise · Gray-Scott RD · L-System · "
            "Recursive Fractal · Urban Geometric · Collage<br>"
            "Second generator layer with alpha-aware blending.<br><br>"
            "Evolution: moth-kill interactive or automatic fitness.<br>"
            "Built with PyQt6 · NumPy · OpenCV · scikit-learn"
        )
