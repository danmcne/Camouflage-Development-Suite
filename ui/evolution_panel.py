"""
EvolutionPanel – Tab 3.

Layout fix: the evolution panel now signals the main window to expand the
splitter to full width when the evolution tab is selected.  This is done via
the `wants_fullwidth` signal.

MothCanvas fills the centre; left=backgrounds (170 px fixed),
right=controls (200 px fixed).  The background image is drawn large and moths
scatter over it.

Worker is now wrapped in try/except and emits an `error` signal on failure
instead of crashing the whole thread.
"""
from __future__ import annotations
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QSpinBox, QRadioButton,
    QGroupBox, QFileDialog, QScrollArea,
    QProgressBar, QSizePolicy, QComboBox, QGridLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, QRect, QPoint
from PyQt6.QtGui import QPainter, QPixmap, QColor, QPen, QFont

from utils.rendering import bgr_to_qpixmap, make_thumbnail
from evolution.background_manager import BackgroundManager
from config.defaults import EVOLUTION, APP


# ── moth ──────────────────────────────────────────────────────────────────────

class Moth:
    __slots__ = ("index","pixmap","x","y","size","killed","fitness")
    def __init__(self, index, pixmap, x, y, size):
        self.index   = index
        self.pixmap  = pixmap
        self.x = x; self.y = y; self.size = size
        self.killed  = False
        self.fitness = 0.0
    def rect(self):  return QRect(self.x, self.y, self.size, self.size)
    def contains(self, pt): return self.rect().contains(pt)


# ── moth canvas ───────────────────────────────────────────────────────────────

class MothCanvas(QWidget):
    moth_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bg_pixmap: QPixmap | None = None
        self._moths: list[Moth] = []
        self._hover = -1
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)

    def set_background(self, pix: QPixmap | None):
        self._bg_pixmap = pix
        self.update()

    def set_moths(self, moths: list[Moth]):
        self._moths = moths
        self.update()

    def clear_moths(self):
        self._moths = []
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        W, H = self.width(), self.height()

        if self._bg_pixmap:
            scaled = self._bg_pixmap.scaled(
                W, H,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            ox = (scaled.width()  - W) // 2
            oy = (scaled.height() - H) // 2
            p.drawPixmap(0, 0, scaled, ox, oy, W, H)
        else:
            p.fillRect(0, 0, W, H, QColor(30, 30, 30))
            p.setPen(QColor(110, 110, 110))
            p.drawText(
                QRect(0, 0, W, H), Qt.AlignmentFlag.AlignCenter,
                "Add a background image\n(use the Backgrounds panel or menu)"
            )

        font = QFont()
        font.setPointSize(8)
        p.setFont(font)

        for moth in self._moths:
            r = moth.rect()
            if moth.killed:
                p.setOpacity(0.22)
                p.drawPixmap(r, moth.pixmap)
                p.setOpacity(1.0)
                pen = QPen(QColor(210, 30, 30), 3)
                p.setPen(pen)
                p.drawLine(r.topLeft(), r.bottomRight())
                p.drawLine(r.topRight(), r.bottomLeft())
                p.setPen(QPen(QColor(210, 30, 30), 2))
                p.drawRect(r)
            else:
                p.setOpacity(1.0)
                p.drawPixmap(r, moth.pixmap)
                col = QColor(255,220,0) if moth.index == self._hover else QColor(255,255,255)
                p.setPen(QPen(col, 2))
                p.drawRect(r)
                p.setPen(QColor(255, 255, 255))
                p.drawText(r.x()+2, r.y() + r.height() + 12, f"{moth.fitness:.3f}")

        p.end()

    def mouseMoveEvent(self, ev):
        pos = ev.position().toPoint()
        for m in reversed(self._moths):
            if m.contains(pos):
                if self._hover != m.index:
                    self._hover = m.index
                    self.update()
                return
        if self._hover != -1:
            self._hover = -1
            self.update()

    def mousePressEvent(self, ev):
        pos = ev.position().toPoint()
        for m in reversed(self._moths):
            if m.contains(pos):
                m.killed = not m.killed
                self.moth_clicked.emit(m.index)
                self.update()
                return


# ── worker ────────────────────────────────────────────────────────────────────

class _EvoWorker(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(self, population, bg_manager, weights, size):
        super().__init__()
        self._pop     = population
        self._bg      = bg_manager
        self._weights = weights
        self._size    = size
        self._abort   = False

    def abort(self): self._abort = True

    def run(self):
        try:
            self._do_run()
        except Exception as exc:
            import traceback
            self.error.emit(traceback.format_exc())

    def _do_run(self):
        from generators import get_generator
        from core.fitness import composite_fitness

        # Signal fractal generator to clear any abort state
        try:
            from generators.recursive_fractal import set_abort
            set_abort(False)
        except Exception:
            pass

        gen     = get_generator(self._pop.generator_type)
        results = []

        for i, ind in enumerate(self._pop.individuals):
            if self._abort:
                break

            colors_rgb = []
            for hx in ind.colors:
                h = hx.lstrip("#")
                colors_rgb.append((int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)))

            try:
                img = gen.generate(self._size[0], self._size[1], colors_rgb, ind.params)
            except Exception:
                img = np.zeros((self._size[1], self._size[0], 3), dtype=np.uint8)

            # Strip alpha if 4-channel
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            ind.image = img

            bg = self._bg.get_active(self._size) if len(self._bg) else None
            if bg is not None:
                try:
                    scores = composite_fitness(img, bg, self._weights)
                except Exception:
                    scores = {"color":0.0,"texture":0.0,"disruption":0.0,"total":0.0}
            else:
                scores = {"color":0.0,"texture":0.0,"disruption":0.0,"total":0.0}

            ind.fitness = scores["total"]
            results.append((img.copy(), scores))
            self.progress.emit(i + 1, len(self._pop.individuals))

        self.finished.emit(results)


# ── main panel ────────────────────────────────────────────────────────────────

class EvolutionPanel(QWidget):
    candidate_chosen = pyqtSignal(object, dict)
    wants_fullwidth  = pyqtSignal(bool)   # True=expand, False=restore

    def __init__(self, bg_manager: BackgroundManager, parent=None):
        super().__init__(parent)
        self._bg_manager  = bg_manager
        self._population  = None
        self._palette: list[str] = []
        self._results: list[tuple] = []
        self._thread: QThread | None = None
        self._worker: _EvoWorker | None = None
        self._moth_size   = 120
        self._moths: list[Moth] = []
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Left: backgrounds ─────────────────────────────────────────────────
        bg_group = QGroupBox("Backgrounds")
        bg_group.setFixedWidth(165)
        bl = QVBoxLayout(bg_group)
        add_folder_btn = QPushButton("📁 Add folder")
        add_folder_btn.clicked.connect(self._add_bg_folder)
        add_file_btn   = QPushButton("🖼 Add image")
        add_file_btn.clicked.connect(self._add_bg_file)
        bl.addWidget(add_folder_btn)
        bl.addWidget(add_file_btn)

        self._bg_list_widget = QWidget()
        self._bg_list_layout = QVBoxLayout(self._bg_list_widget)
        self._bg_list_layout.setSpacing(2)
        self._bg_list_layout.addStretch()
        bg_scroll = QScrollArea()
        bg_scroll.setWidgetResizable(True)
        bg_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        bg_scroll.setWidget(self._bg_list_widget)
        bl.addWidget(bg_scroll)
        root.addWidget(bg_group)

        # ── Centre: moth canvas ────────────────────────────────────────────────
        centre = QVBoxLayout()

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(10)
        centre.addWidget(self._progress)

        self._moth_canvas = MothCanvas()
        self._moth_canvas.moth_clicked.connect(self._on_moth_clicked)
        centre.addWidget(self._moth_canvas, 1)

        self._status_label = QLabel("Seed a population to begin.")
        self._status_label.setStyleSheet("color:#aaa;font-size:11px;")
        centre.addWidget(self._status_label)

        centre_w = QWidget()
        centre_w.setLayout(centre)
        root.addWidget(centre_w, 1)

        # ── Right: controls ───────────────────────────────────────────────────
        ctrl = QVBoxLayout()
        ctrl.setSpacing(5)

        mode_group = QGroupBox("Mode")
        ml = QVBoxLayout(mode_group)
        self._mode_interactive = QRadioButton("Interactive (click to kill)")
        self._mode_automatic   = QRadioButton("Automatic (fitness-driven)")
        self._mode_interactive.setChecked(True)
        ml.addWidget(self._mode_interactive)
        ml.addWidget(self._mode_automatic)
        ctrl.addWidget(mode_group)

        gen_group = QGroupBox("Generator")
        gl = QVBoxLayout(gen_group)
        from generators import REGISTRY
        self._gen_combo = QComboBox()
        for name in REGISTRY:
            self._gen_combo.addItem(name)
        gl.addWidget(self._gen_combo)
        ctrl.addWidget(gen_group)

        pop_group = QGroupBox("Population")
        pl = QGridLayout(pop_group)
        pl.addWidget(QLabel("Size:"), 0, 0)
        self._pop_spin = QSpinBox()
        self._pop_spin.setRange(4, 64)
        self._pop_spin.setValue(EVOLUTION["population_size"])
        pl.addWidget(self._pop_spin, 0, 1)
        ctrl.addWidget(pop_group)

        moth_group = QGroupBox("Moth size")
        moth_l = QVBoxLayout(moth_group)
        self._moth_slider = QSlider(Qt.Orientation.Horizontal)
        self._moth_slider.setRange(50, 250)
        self._moth_slider.setValue(self._moth_size)
        self._moth_size_lbl = QLabel(f"{self._moth_size} px")
        self._moth_slider.valueChanged.connect(self._on_moth_size)
        moth_l.addWidget(self._moth_slider)
        moth_l.addWidget(self._moth_size_lbl)
        ctrl.addWidget(moth_group)

        w_group = QGroupBox("Fitness weights")
        wl = QVBoxLayout(w_group)
        self._weight_sliders: dict[str, QSlider] = {}
        for key, label in [("color","Colour"),("texture","Texture"),("disruption","Disruption")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label+":"))
            sl  = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, 100)
            sl.setValue(int(EVOLUTION["fitness_weights"][key]*100))
            lbl = QLabel(f"{EVOLUTION['fitness_weights'][key]:.2f}")
            lbl.setFixedWidth(30)
            sl.valueChanged.connect(lambda v, l=lbl: l.setText(f"{v/100:.2f}"))
            row.addWidget(sl); row.addWidget(lbl)
            self._weight_sliders[key] = sl
            wl.addLayout(row)
        ctrl.addWidget(w_group)

        self._run_btn = QPushButton("▶  Seed & Run")
        self._run_btn.setFixedHeight(34)
        self._run_btn.setStyleSheet(
            "QPushButton{background:#2e6b2e;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#3d8a3d;}"
        )
        self._run_btn.clicked.connect(self._on_run)
        ctrl.addWidget(self._run_btn)

        self._next_btn = QPushButton("⟳  Next generation")
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._on_next_gen)
        ctrl.addWidget(self._next_btn)

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        ctrl.addWidget(self._stop_btn)

        ctrl.addStretch()
        ctrl_w = QWidget()
        ctrl_w.setFixedWidth(195)
        ctrl_w.setLayout(ctrl)
        root.addWidget(ctrl_w)

    # ── background management ─────────────────────────────────────────────────

    def _add_bg_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select background folder")
        if folder:
            self._bg_manager.add_folder(folder)
            self._refresh_bg_list()
            self._update_canvas_bg()

    def _add_bg_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select background image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if path:
            self._bg_manager.add_file(path)
            self._refresh_bg_list()
            self._update_canvas_bg()

    def _refresh_bg_list(self):
        while self._bg_list_layout.count() > 1:
            item = self._bg_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, path in enumerate(self._bg_manager.paths):
            th = self._bg_manager.get_thumbnail(i, 140)
            if th is None:
                continue
            pix = bgr_to_qpixmap(th)
            lbl = QLabel()
            lbl.setPixmap(pix.scaled(150, 90,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            lbl.setToolTip(path)
            lbl.setCursor(Qt.CursorShape.PointingHandCursor)
            idx = i
            lbl.mousePressEvent = lambda _, j=idx: self._select_bg(j)
            self._bg_list_layout.insertWidget(self._bg_list_layout.count()-1, lbl)

    def _select_bg(self, index: int):
        self._bg_manager.set_active(index)
        self._update_canvas_bg()

    def _update_canvas_bg(self):
        bg = self._bg_manager.get_active((1024, 1024))
        pix = bgr_to_qpixmap(bg) if bg is not None else None
        self._moth_canvas.set_background(pix)

    # ── moth size ─────────────────────────────────────────────────────────────

    def _on_moth_size(self, val: int):
        self._moth_size = val
        self._moth_size_lbl.setText(f"{val} px")
        if self._results:
            self._place_moths(self._results)

    # ── moth placement ────────────────────────────────────────────────────────

    def _place_moths(self, results: list[tuple]):
        import random
        random.seed(getattr(self._population, "generation", 0))
        sz = self._moth_size
        W  = max(self._moth_canvas.width(),  300)
        H  = max(self._moth_canvas.height(), 300)
        moths    = []
        positions= []
        for i, (img, scores) in enumerate(results):
            th  = make_thumbnail(img, sz)
            pix = bgr_to_qpixmap(th)
            # Try non-overlapping placement
            for _ in range(40):
                x = random.randint(0, max(1, W-sz))
                y = random.randint(0, max(1, H-sz))
                if all(abs(x-px)>=sz*0.6 or abs(y-py)>=sz*0.6 for px,py in positions):
                    break
            positions.append((x, y))
            m = Moth(i, pix, x, y, sz)
            m.fitness = scores.get("total", 0.0)
            moths.append(m)
        self._moths = moths
        self._moth_canvas.set_moths(moths)

    # ── moth click ────────────────────────────────────────────────────────────

    def _on_moth_clicked(self, index: int):
        alive  = sum(1 for m in self._moths if not m.killed)
        killed = len(self._moths) - alive
        self._status_label.setText(
            f"Gen {getattr(self._population,'generation',0)}: "
            f"{alive} alive, {killed} killed — "
            f"click 'Next generation' to breed survivors."
        )
        if index < len(self._results):
            img, scores = self._results[index]
            self.candidate_chosen.emit(img, scores)

    # ── evolution controls ────────────────────────────────────────────────────

    def _get_weights(self):
        return {k: sl.value()/100.0 for k, sl in self._weight_sliders.items()}

    def _on_run(self):
        from evolution.population import Population
        gen_name = self._gen_combo.currentText()
        self._population = Population(
            size=self._pop_spin.value(),
            generator_type=gen_name,
            colors=self._palette or [],
        )
        self._population.seed()
        self._moths = []
        self._moth_canvas.clear_moths()
        self._run_worker()

    def _on_next_gen(self):
        if self._population is None:
            return
        if self._mode_interactive.isChecked() and self._moths:
            kept = [m.index for m in self._moths if not m.killed]
            if kept:
                self._population.apply_user_selection(kept)
            else:
                self._population.evolve_step()
        else:
            self._population.evolve_step()
        for m in self._moths:
            m.killed = False
        self._moth_canvas.clear_moths()
        self._run_worker()

    def _on_stop(self):
        if self._worker:
            self._worker.abort()
            # Also signal fractal abort
            try:
                from generators.recursive_fractal import set_abort
                set_abort(True)
            except Exception:
                pass
        self._stop_btn.setEnabled(False)
        self._status_label.setText("Stopped.")

    def _run_worker(self):
        if self._thread and self._thread.isRunning():
            return
        self._run_btn.setEnabled(False)
        self._next_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, self._population.size)
        self._progress.setValue(0)
        self._status_label.setText(f"Generating generation {self._population.generation}…")

        self._thread = QThread()
        self._worker = _EvoWorker(
            self._population, self._bg_manager,
            self._get_weights(), APP["preview_size"],
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_worker_done(self, results: list):
        self._results = results
        self._update_canvas_bg()
        self._place_moths(results)
        self._run_btn.setEnabled(True)
        self._next_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setVisible(False)
        best = max((s.get("total",0) for _,s in results), default=0)
        self._status_label.setText(
            f"Gen {self._population.generation}: {len(results)} moths.  "
            f"Best fitness {best:.3f}.  Click moths to kill; then Next generation."
        )

    def _on_worker_error(self, msg: str):
        self._run_btn.setEnabled(True)
        self._next_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._status_label.setText(f"Error in worker — see console.")
        print("=== Evolution worker error ===")
        print(msg)

    # ── public ────────────────────────────────────────────────────────────────

    def set_palette(self, palette):
        self._palette = list(palette)
        if self._population:
            self._population.colors = self._palette

    def on_tab_activated(self):
        """Called by MainWindow when this tab is shown."""
        self.wants_fullwidth.emit(True)
        self._update_canvas_bg()

    def on_tab_deactivated(self):
        """Called by MainWindow when another tab is shown."""
        self.wants_fullwidth.emit(False)
