"""
Evolution Panel – Tab 3.

Changes:
  • Kill / Inspect mode toggle. In Inspect mode clicks send the candidate to
    the generator panel (with its params) but do NOT mark it killed.
  • In Kill mode the moth border and fitness score are hidden unless hovered,
    so the camo is not betrayed by a bright white rectangle.
  • Evolution worker caps Recursive Fractal depth at EVOLUTION["max_fractal_depth"]
    and respects EVOLUTION["individual_timeout"].
  • `set_seed_params(name, params)` – called by main window so the population
    is seeded near the user's current generator settings.
  • `candidate_chosen` signal carries (img, scores, gen_name, params) for Inspect.
"""
from __future__ import annotations
import numpy as np
import cv2
import copy

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QSpinBox, QRadioButton,
    QGroupBox, QFileDialog, QScrollArea,
    QProgressBar, QSizePolicy, QComboBox, QGridLayout,
    QButtonGroup,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, QRect, QPoint
from PyQt6.QtGui import QPainter, QPixmap, QColor, QPen, QFont

from utils.rendering import bgr_to_qpixmap, make_thumbnail
from evolution.background_manager import BackgroundManager
from config.defaults import EVOLUTION, APP


# ── moth ──────────────────────────────────────────────────────────────────────

class Moth:
    __slots__ = ("index","pixmap","x","y","size","killed","fitness","gen_name","params")
    def __init__(self, index, pixmap, x, y, size):
        self.index    = index
        self.pixmap   = pixmap
        self.x = x; self.y = y; self.size = size
        self.killed   = False
        self.fitness  = 0.0
        self.gen_name = ""
        self.params   = {}
    def rect(self): return QRect(self.x, self.y, self.size, self.size)
    def contains(self, pt): return self.rect().contains(pt)


# ── moth canvas ───────────────────────────────────────────────────────────────

class MothCanvas(QWidget):
    moth_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bg_pixmap: QPixmap | None = None
        self._moths: list[Moth] = []
        self._hover = -1
        self._kill_mode = True   # False = inspect mode
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(300, 300)
        self.setMouseTracking(True)

    def set_background(self, pix): self._bg_pixmap = pix; self.update()
    def set_moths(self, moths): self._moths = moths; self.update()
    def clear_moths(self): self._moths = []; self.update()
    def set_kill_mode(self, v): self._kill_mode = v; self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        W, H = self.width(), self.height()

        if self._bg_pixmap:
            sc = self._bg_pixmap.scaled(W, H,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation)
            ox = (sc.width()-W)//2; oy = (sc.height()-H)//2
            p.drawPixmap(0, 0, sc, ox, oy, W, H)
        else:
            p.fillRect(0,0,W,H,QColor(30,30,30))
            p.setPen(QColor(110,110,110))
            p.drawText(QRect(0,0,W,H), Qt.AlignmentFlag.AlignCenter,
                       "Add a background image\n(Backgrounds panel or menu)")

        font = QFont(); font.setPointSize(8); p.setFont(font)

        for moth in self._moths:
            r = moth.rect()
            hovered  = moth.index == self._hover

            if moth.killed:
                # Killed: dim + red X regardless of mode
                p.setOpacity(0.22)
                p.drawPixmap(r, moth.pixmap)
                p.setOpacity(1.0)
                pen = QPen(QColor(210,30,30), 3)
                p.setPen(pen)
                p.drawLine(r.topLeft(), r.bottomRight())
                p.drawLine(r.topRight(), r.bottomLeft())
                p.setPen(QPen(QColor(210,30,30),2))
                p.drawRect(r)
            else:
                p.setOpacity(1.0)
                p.drawPixmap(r, moth.pixmap)

                if self._kill_mode:
                    # In kill mode: only show border + score on hover
                    if hovered:
                        p.setPen(QPen(QColor(255,220,0),2))
                        p.drawRect(r)
                        p.setPen(QColor(255,220,0))
                        p.drawText(r.x()+2, r.y()+r.height()+12,
                                   f"fit:{moth.fitness:.3f}")
                else:
                    # Inspect mode: subtle border always, bright on hover
                    col = QColor(100,180,255) if hovered else QColor(180,180,180)
                    p.setPen(QPen(col, 1 if not hovered else 2))
                    p.drawRect(r)
                    if hovered:
                        p.setPen(QColor(100,180,255))
                        p.drawText(r.x()+2, r.y()+r.height()+12,
                                   f"inspect: {moth.fitness:.3f}")

        p.end()

    def mouseMoveEvent(self, ev):
        pos = ev.position().toPoint()
        for m in reversed(self._moths):
            if m.contains(pos):
                if self._hover != m.index:
                    self._hover = m.index; self.update()
                return
        if self._hover != -1:
            self._hover = -1; self.update()

    def mousePressEvent(self, ev):
        pos = ev.position().toPoint()
        for m in reversed(self._moths):
            if m.contains(pos):
                if self._kill_mode:
                    m.killed = not m.killed
                # Always emit so main window can display the candidate
                self.moth_clicked.emit(m.index)
                self.update()
                return


# ── worker ────────────────────────────────────────────────────────────────────

class _EvoWorker(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list)   # list of (img, scores, gen_name, params)
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
        try:    self._do_run()
        except Exception:
            import traceback; self.error.emit(traceback.format_exc())

    def _do_run(self):
        import time
        from generators import get_generator
        from core.fitness import composite_fitness

        try:
            from generators.recursive_fractal import set_abort
            set_abort(False)
        except Exception:
            pass

        gen      = get_generator(self._pop.generator_type)
        results  = []
        max_depth= EVOLUTION["max_fractal_depth"]
        timeout  = EVOLUTION["individual_timeout"]

        for i, ind in enumerate(self._pop.individuals):
            if self._abort: break

            colors_rgb = []
            for hx in ind.colors:
                h = hx.lstrip("#")
                colors_rgb.append((int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)))

            # Cap fractal depth in evolution context
            params = copy.deepcopy(ind.params)
            if ind.generator_type == "Recursive Fractal":
                params["max_depth"] = max_depth

            t0 = time.time()
            try:
                img = gen.generate(self._size[0], self._size[1], colors_rgb, params)
            except Exception:
                img = np.zeros((self._size[1], self._size[0], 3), dtype=np.uint8)
            if time.time() - t0 > timeout:
                pass  # proceed with whatever was generated

            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            ind.image = img

            bg = self._bg.get_active(self._size) if len(self._bg) else None
            if bg is not None:
                try:    scores = composite_fitness(img, bg, self._weights)
                except Exception: scores = {"color":0.0,"texture":0.0,"disruption":0.0,"total":0.0}
            else:
                scores = {"color":0.0,"texture":0.0,"disruption":0.0,"total":0.0}

            ind.fitness = scores["total"]
            results.append((img.copy(), scores, ind.generator_type, params))
            self.progress.emit(i+1, len(self._pop.individuals))

        self.finished.emit(results)


# ── main panel ────────────────────────────────────────────────────────────────

class EvolutionPanel(QWidget):
    # Carries image, scores, gen_name, params so Inspect can populate generator panel
    candidate_chosen = pyqtSignal(object, dict, str, dict)
    wants_fullwidth  = pyqtSignal(bool)

    def __init__(self, bg_manager: BackgroundManager, parent=None):
        super().__init__(parent)
        self._bg_manager = bg_manager
        self._population = None
        self._palette: list[str] = []
        self._results: list[tuple] = []
        self._thread: QThread | None = None
        self._worker: _EvoWorker | None = None
        self._moth_size   = 120
        self._moths: list[Moth] = []
        self._seed_params_hint: dict | None = None   # from generator panel
        self._seed_gen_hint: str | None     = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4,4,4,4); root.setSpacing(4)

        # Left: backgrounds
        bg_g = QGroupBox("Backgrounds"); bg_g.setFixedWidth(165)
        bl   = QVBoxLayout(bg_g)
        afb  = QPushButton("📁 Add folder"); afb.clicked.connect(self._add_bg_folder)
        afi  = QPushButton("🖼 Add image");  afi.clicked.connect(self._add_bg_file)
        bl.addWidget(afb); bl.addWidget(afi)
        self._bg_list_widget = QWidget()
        self._bg_list_layout = QVBoxLayout(self._bg_list_widget)
        self._bg_list_layout.setSpacing(2)
        self._bg_list_layout.addStretch()
        bgs = QScrollArea(); bgs.setWidgetResizable(True)
        bgs.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        bgs.setWidget(self._bg_list_widget)
        bl.addWidget(bgs)
        root.addWidget(bg_g)

        # Centre: moth canvas
        centre = QVBoxLayout()
        self._progress = QProgressBar(); self._progress.setVisible(False)
        self._progress.setFixedHeight(10); centre.addWidget(self._progress)
        self._moth_canvas = MothCanvas()
        self._moth_canvas.moth_clicked.connect(self._on_moth_clicked)
        centre.addWidget(self._moth_canvas, 1)
        self._status_label = QLabel("Seed a population to begin.")
        self._status_label.setStyleSheet("color:#aaa;font-size:11px;")
        centre.addWidget(self._status_label)
        cw = QWidget(); cw.setLayout(centre)
        root.addWidget(cw, 1)

        # Right: controls
        ctrl = QVBoxLayout(); ctrl.setSpacing(5)

        # Kill / Inspect mode
        click_g = QGroupBox("Click action")
        click_l = QVBoxLayout(click_g)
        self._kill_radio    = QRadioButton("Kill mode"); self._kill_radio.setChecked(True)
        self._inspect_radio = QRadioButton("Inspect mode")
        self._kill_radio.toggled.connect(self._on_click_mode_changed)
        click_l.addWidget(self._kill_radio)
        click_l.addWidget(self._inspect_radio)
        ctrl.addWidget(click_g)

        # Auto / Interactive
        mode_g = QGroupBox("Evolution mode")
        mode_l = QVBoxLayout(mode_g)
        self._mode_interactive = QRadioButton("Interactive (kill to select)")
        self._mode_automatic   = QRadioButton("Automatic (fitness)")
        self._mode_interactive.setChecked(True)
        mode_l.addWidget(self._mode_interactive)
        mode_l.addWidget(self._mode_automatic)
        ctrl.addWidget(mode_g)

        # Generator
        gen_g = QGroupBox("Generator")
        gen_l = QVBoxLayout(gen_g)
        from generators import REGISTRY
        self._gen_combo = QComboBox()
        for n in REGISTRY: self._gen_combo.addItem(n)
        gen_l.addWidget(self._gen_combo)
        ctrl.addWidget(gen_g)

        # Population
        pop_g = QGroupBox("Population")
        pop_l = QGridLayout(pop_g)
        pop_l.addWidget(QLabel("Size:"), 0, 0)
        self._pop_spin = QSpinBox()
        self._pop_spin.setRange(4, 64)
        self._pop_spin.setValue(EVOLUTION["population_size"])
        pop_l.addWidget(self._pop_spin, 0, 1)
        ctrl.addWidget(pop_g)

        # Moth size
        moth_g = QGroupBox("Moth size")
        moth_l = QVBoxLayout(moth_g)
        self._moth_slider = QSlider(Qt.Orientation.Horizontal)
        self._moth_slider.setRange(50, 250); self._moth_slider.setValue(self._moth_size)
        self._moth_size_lbl = QLabel(f"{self._moth_size} px")
        self._moth_slider.valueChanged.connect(self._on_moth_size)
        moth_l.addWidget(self._moth_slider); moth_l.addWidget(self._moth_size_lbl)
        ctrl.addWidget(moth_g)

        # Fitness weights
        w_g = QGroupBox("Fitness weights")
        wl  = QVBoxLayout(w_g)
        self._weight_sliders: dict[str, QSlider] = {}
        for key, label in [("color","Colour"),("texture","Texture"),("disruption","Disruption")]:
            rw = QHBoxLayout(); rw.addWidget(QLabel(label+":"))
            sl = QSlider(Qt.Orientation.Horizontal); sl.setRange(0,100)
            sl.setValue(int(EVOLUTION["fitness_weights"][key]*100))
            lb = QLabel(f"{EVOLUTION['fitness_weights'][key]:.2f}"); lb.setFixedWidth(30)
            sl.valueChanged.connect(lambda v,l=lb: l.setText(f"{v/100:.2f}"))
            rw.addWidget(sl); rw.addWidget(lb)
            self._weight_sliders[key] = sl; wl.addLayout(rw)
        ctrl.addWidget(w_g)

        self._run_btn = QPushButton("▶  Seed & Run"); self._run_btn.setFixedHeight(34)
        self._run_btn.setStyleSheet(
            "QPushButton{background:#2e6b2e;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#3d8a3d;}")
        self._run_btn.clicked.connect(self._on_run)
        ctrl.addWidget(self._run_btn)

        self._next_btn = QPushButton("⟳  Next generation"); self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._on_next_gen); ctrl.addWidget(self._next_btn)

        self._stop_btn = QPushButton("⏹  Stop"); self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop); ctrl.addWidget(self._stop_btn)

        ctrl.addStretch()
        cw2 = QWidget(); cw2.setFixedWidth(195); cw2.setLayout(ctrl)
        root.addWidget(cw2)

    # ── click mode ────────────────────────────────────────────────────────────

    def _on_click_mode_changed(self, kill_checked):
        self._moth_canvas.set_kill_mode(kill_checked)

    # ── backgrounds ───────────────────────────────────────────────────────────

    def _add_bg_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select background folder")
        if folder:
            self._bg_manager.add_folder(folder)
            self._refresh_bg_list(); self._update_canvas_bg()

    def _add_bg_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select background image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if path:
            self._bg_manager.add_file(path)
            self._refresh_bg_list(); self._update_canvas_bg()

    def _refresh_bg_list(self):
        while self._bg_list_layout.count() > 1:
            item = self._bg_list_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for i, path in enumerate(self._bg_manager.paths):
            th = self._bg_manager.get_thumbnail(i, 140)
            if th is None: continue
            pix = bgr_to_qpixmap(th)
            lbl = QLabel()
            lbl.setPixmap(pix.scaled(150,90,Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))
            lbl.setToolTip(path); lbl.setCursor(Qt.CursorShape.PointingHandCursor)
            idx = i
            lbl.mousePressEvent = lambda _,j=idx: self._select_bg(j)
            self._bg_list_layout.insertWidget(self._bg_list_layout.count()-1, lbl)

    def _select_bg(self, i):
        self._bg_manager.set_active(i); self._update_canvas_bg()

    def _update_canvas_bg(self):
        bg = self._bg_manager.get_active((1024,1024))
        self._moth_canvas.set_background(bgr_to_qpixmap(bg) if bg is not None else None)

    # ── moth size ─────────────────────────────────────────────────────────────

    def _on_moth_size(self, val):
        self._moth_size = val; self._moth_size_lbl.setText(f"{val} px")
        if self._results: self._place_moths(self._results)

    # ── moth placement ────────────────────────────────────────────────────────

    def _place_moths(self, results):
        import random
        random.seed(getattr(self._population,"generation",0))
        sz = self._moth_size
        W  = max(self._moth_canvas.width(), 300)
        H  = max(self._moth_canvas.height(), 300)
        moths = []; positions = []
        for i, (img, scores, gen_name, params) in enumerate(results):
            th  = make_thumbnail(img, sz)
            pix = bgr_to_qpixmap(th)
            for _ in range(40):
                x = random.randint(0, max(1, W-sz))
                y = random.randint(0, max(1, H-sz))
                if all(abs(x-px)>=sz*0.6 or abs(y-py)>=sz*0.6 for px,py in positions):
                    break
            positions.append((x, y))
            m = Moth(i, pix, x, y, sz)
            m.fitness   = scores.get("total", 0.0)
            m.gen_name  = gen_name
            m.params    = params
            moths.append(m)
        self._moths = moths
        self._moth_canvas.set_moths(moths)

    # ── moth click ────────────────────────────────────────────────────────────

    def _on_moth_clicked(self, index):
        alive  = sum(1 for m in self._moths if not m.killed)
        killed = len(self._moths) - alive
        mode   = "Kill" if self._kill_radio.isChecked() else "Inspect"
        self._status_label.setText(
            f"Gen {getattr(self._population,'generation',0)} [{mode} mode]: "
            f"{alive} alive, {killed} killed.")
        if index < len(self._results):
            img, scores, gen_name, params = self._results[index]
            self.candidate_chosen.emit(img, scores, gen_name, params)

    # ── evolution controls ────────────────────────────────────────────────────

    def _get_weights(self):
        return {k: sl.value()/100.0 for k,sl in self._weight_sliders.items()}

    def _on_run(self):
        from evolution.population import Population
        gen_name = self._gen_combo.currentText()

        pop = Population(
            size=self._pop_spin.value(),
            generator_type=gen_name,
            colors=self._palette or [],
        )
        # If the user was editing this generator, seed near their params
        if (self._seed_gen_hint == gen_name and self._seed_params_hint):
            from generators import get_generator
            gen = get_generator(gen_name)
            import copy
            from core.pattern import CamoPattern
            pop.individuals = []
            for _ in range(pop.size):
                mutated = gen.mutate(copy.deepcopy(self._seed_params_hint), strength=0.2)
                pop.individuals.append(CamoPattern(
                    generator_type=gen_name,
                    params=mutated,
                    colors=list(self._palette or []),
                ))
            pop.generation = 0
        else:
            pop.seed()

        self._population = pop
        self._moths = []; self._moth_canvas.clear_moths()
        self._run_worker()

    def _on_next_gen(self):
        if self._population is None: return
        if self._mode_interactive.isChecked() and self._moths:
            kept = [m.index for m in self._moths if not m.killed]
            self._population.apply_user_selection(kept) if kept else self._population.evolve_step()
        else:
            self._population.evolve_step()
        for m in self._moths: m.killed = False
        self._moth_canvas.clear_moths()
        self._run_worker()

    def _on_stop(self):
        if self._worker: self._worker.abort()
        try:
            from generators.recursive_fractal import set_abort; set_abort(True)
        except Exception: pass
        self._stop_btn.setEnabled(False)
        self._status_label.setText("Stopped.")

    def _run_worker(self):
        if self._thread and self._thread.isRunning(): return
        self._run_btn.setEnabled(False); self._next_btn.setEnabled(False)
        self._stop_btn.setEnabled(True); self._progress.setVisible(True)
        self._progress.setRange(0, self._population.size); self._progress.setValue(0)
        self._status_label.setText(f"Generation {self._population.generation}…")

        self._thread = QThread()
        self._worker = _EvoWorker(
            self._population, self._bg_manager,
            self._get_weights(), APP["preview_size"])
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_worker_done(self, results):
        self._results = results
        self._update_canvas_bg(); self._place_moths(results)
        self._run_btn.setEnabled(True); self._next_btn.setEnabled(True)
        self._stop_btn.setEnabled(False); self._progress.setVisible(False)
        best = max((s.get("total",0) for _,s,*_ in results), default=0)
        self._status_label.setText(
            f"Gen {self._population.generation}: {len(results)} moths, best={best:.3f}. "
            f"{'Click to kill, then Next generation.' if self._kill_radio.isChecked() else 'Click to inspect.'}")

    def _on_worker_error(self, msg):
        self._run_btn.setEnabled(True); self._next_btn.setEnabled(False)
        self._stop_btn.setEnabled(False); self._progress.setVisible(False)
        self._status_label.setText("Worker error — see console.")
        print("=== Evolution worker error ===\n", msg)

    # ── public ────────────────────────────────────────────────────────────────

    def set_palette(self, palette):
        self._palette = list(palette)
        if self._population: self._population.colors = self._palette

    def set_seed_params(self, gen_name: str, params: dict):
        """Called by main window when the generator panel changes."""
        self._seed_gen_hint   = gen_name
        self._seed_params_hint = params
        # Sync the combo if it matches
        if self._gen_combo.findText(gen_name) >= 0:
            self._gen_combo.setCurrentText(gen_name)

    def on_tab_activated(self):
        self.wants_fullwidth.emit(True); self._update_canvas_bg()

    def on_tab_deactivated(self):
        self.wants_fullwidth.emit(False)
