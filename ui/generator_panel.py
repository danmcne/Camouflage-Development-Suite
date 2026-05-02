"""
Generator Panel – Tab 2.

Key changes vs original:
  • Every parameter (evolvable OR not) gets a 🔒 lock checkbox.
    Locking prevents the parameter from being mutated during evolution.
  • Lock-state changes immediately emit `lock_changed` so the evolution
    panel always has the current locked set — even if no param VALUE changed.
  • `get_locked_params()` returns the set of locked param keys.
  • `load_pattern` / `load_pattern_layer2` for Inspect mode.
  • `get_current_gen_params_hint()` for seeding evolution populations.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QPushButton, QDoubleSpinBox,
    QSpinBox, QCheckBox, QScrollArea, QGroupBox,
    QFileDialog, QLineEdit, QSlider, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal

from generators import REGISTRY, get_generator
from config.defaults import BLEND_MODES
from core.palette import ColorPalette


class GeneratorPanel(QWidget):
    params_changed     = pyqtSignal(str, dict)
    lock_changed       = pyqtSignal(set)          # emitted when any L1 lock toggle changes
    lock_changed2      = pyqtSignal(set)          # emitted when any L2 lock toggle changes
    generate_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._generator_name = "Procedural Noise"
        self._param_widgets:  dict[str, QWidget]   = {}
        self._param_widgets2: dict[str, QWidget]   = {}
        self._lock_widgets:   dict[str, QCheckBox] = {}
        self._lock_widgets2:  dict[str, QCheckBox] = {}  # layer 2 lock
        self._palette1: ColorPalette | None = None
        self._palette2: ColorPalette | None = None
        self._build_ui()
        self._populate_params(self._gen_combo, self._params_form,
                              self._param_widgets, self._desc_label,
                              self._lock_widgets)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        sr = QHBoxLayout()
        sr.addWidget(QLabel("Generator:"))
        self._gen_combo = QComboBox()
        for nm in REGISTRY:
            self._gen_combo.addItem(nm)
        self._gen_combo.currentTextChanged.connect(self._on_gen_changed)
        sr.addWidget(self._gen_combo, 1)
        root.addLayout(sr)

        self._desc_label = QLabel()
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color:#888;font-style:italic;")
        root.addWidget(self._desc_label)

        legend = QLabel("🔒 column = lock parameter (held fixed during evolution)")
        legend.setStyleSheet("color:#aaa;font-size:9px;font-style:italic;")
        root.addWidget(legend)

        sc1 = QScrollArea()
        sc1.setWidgetResizable(True)
        sc1.setFixedHeight(260)
        sc1.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        pw1 = QWidget()
        self._params_form = QFormLayout(pw1)
        self._params_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        sc1.setWidget(pw1)
        root.addWidget(sc1)

        gen_btn = QPushButton("▶  Generate")
        gen_btn.setFixedHeight(34)
        gen_btn.setStyleSheet(
            "QPushButton{background:#2e6b2e;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#3d8a3d;}")
        gen_btn.clicked.connect(self._on_generate)
        root.addWidget(gen_btn)

        # ── Second layer ──────────────────────────────────────────────────────
        l2g = QGroupBox("Second generator layer")
        l2r = QVBoxLayout(l2g)

        self._layer2_check = QCheckBox("Enable layer 2")
        self._layer2_check.stateChanged.connect(self._on_layer2_toggled)
        l2r.addWidget(self._layer2_check)

        self._layer2_controls = QWidget()
        l2c = QVBoxLayout(self._layer2_controls)
        l2c.setContentsMargins(0, 0, 0, 0)

        g2r = QHBoxLayout()
        g2r.addWidget(QLabel("Generator:"))
        self._gen2_combo = QComboBox()
        for nm in REGISTRY:
            self._gen2_combo.addItem(nm)
        self._gen2_combo.setCurrentText("Urban Geometric")
        self._gen2_combo.currentTextChanged.connect(self._on_gen2_changed)
        g2r.addWidget(self._gen2_combo, 1)
        l2c.addLayout(g2r)

        br = QHBoxLayout()
        br.addWidget(QLabel("Blend:"))
        self._blend_combo = QComboBox()
        for bm in BLEND_MODES:
            self._blend_combo.addItem(bm)
        self._blend_combo.setCurrentText("overlay")
        br.addWidget(self._blend_combo, 1)
        l2c.addLayout(br)

        opr = QHBoxLayout()
        opr.addWidget(QLabel("Opacity:"))
        self._opacity2_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity2_slider.setRange(0, 100)
        self._opacity2_slider.setValue(50)
        self._opacity2_label = QLabel("50%")
        self._opacity2_label.setFixedWidth(32)
        self._opacity2_slider.valueChanged.connect(
            lambda v: self._opacity2_label.setText(f"{v}%"))
        opr.addWidget(self._opacity2_slider)
        opr.addWidget(self._opacity2_label)
        l2c.addLayout(opr)

        self._layer2_palette_check = QCheckBox("Use Layer 2 palette")
        l2c.addWidget(self._layer2_palette_check)

        sc2 = QScrollArea()
        sc2.setWidgetResizable(True)
        sc2.setFixedHeight(180)
        sc2.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        pw2 = QWidget()
        self._params_form2 = QFormLayout(pw2)
        self._params_form2.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        sc2.setWidget(pw2)
        l2c.addWidget(sc2)

        self._layer2_controls.setEnabled(False)
        self._layer2_controls.setVisible(False)
        l2r.addWidget(self._layer2_controls)
        root.addWidget(l2g)
        root.addStretch()

        self._populate_params(self._gen2_combo, self._params_form2,
                              self._param_widgets2, None, self._lock_widgets2)

    # ── param population ──────────────────────────────────────────────────────

    def _populate_params(self, combo, form, wd, desc_lbl,
                         lock_wd: dict | None):
        wd.clear()
        if lock_wd is not None:
            lock_wd.clear()
        while form.rowCount():
            form.removeRow(0)

        name   = combo.currentText()
        gen    = get_generator(name)
        if desc_lbl:
            desc_lbl.setText(gen.description)
        schema = gen.get_param_schema()

        for key, spec in schema.items():
            label   = spec.get("label",   key)
            tip     = spec.get("tip",     "")
            default = spec.get("default")
            options = spec.get("options")
            ptype   = spec.get("type",    "")

            if ptype == "folder":
                widget = self._make_folder_widget(default or "")
            elif options:
                widget = QComboBox()
                for o in options:
                    widget.addItem(o)
                if isinstance(default, str) and default in options:
                    widget.setCurrentText(default)
                widget.currentTextChanged.connect(self._on_param_changed)
            elif ptype == "bool" or isinstance(default, bool):
                widget = QCheckBox()
                widget.setChecked(bool(default))
                widget.stateChanged.connect(self._on_param_changed)
            elif isinstance(default, float):
                widget = QDoubleSpinBox()
                widget.setDecimals(3)
                widget.setRange(float(spec.get("min", 0.0)),
                                float(spec.get("max", 1.0)))
                widget.setSingleStep(float(spec.get("step", 0.01)))
                widget.setValue(float(default))
                widget.valueChanged.connect(self._on_param_changed)
            elif isinstance(default, int):
                widget = QSpinBox()
                widget.setRange(int(spec.get("min", 0)),
                                int(spec.get("max", 100000)))
                widget.setSingleStep(int(spec.get("step", 1)))
                widget.setValue(int(default))
                widget.valueChanged.connect(self._on_param_changed)
            elif isinstance(default, str) and ptype == "str":
                widget = QLineEdit(default)
                widget.textChanged.connect(self._on_param_changed)
            else:
                continue

            widget.setToolTip(tip)

            # Lock checkbox for EVERY param in the main generator panel
            if lock_wd is not None:
                row_w  = QWidget()
                row_l  = QHBoxLayout(row_w)
                row_l.setContentsMargins(0, 0, 0, 0)
                row_l.setSpacing(4)
                row_l.addWidget(widget, 1)

                lock_cb = QCheckBox()
                lock_cb.setFixedWidth(20)
                lock_cb.setToolTip(
                    "🔒 Lock — this value will NOT change during evolution.")
                lock_cb.setStyleSheet(
                    "QCheckBox::indicator { width: 14px; height: 14px; }")
                # Connect lock change → immediate notification to evo panel
                lock_cb.stateChanged.connect(self._on_lock_changed)
                row_l.addWidget(lock_cb)
                lock_wd[key] = lock_cb

                form.addRow(label + ":", row_w)
            else:
                form.addRow(label + ":", widget)

            wd[key] = widget

    def _make_folder_widget(self, default_path: str) -> QWidget:
        c = QWidget()
        r = QHBoxLayout(c)
        r.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(default_path)
        edit.setPlaceholderText("(empty = built-in shapes)")
        btn  = QPushButton("…")
        btn.setFixedWidth(28)
        btn.clicked.connect(
            lambda: edit.setText(
                QFileDialog.getExistingDirectory(self, "Select shape folder")
                or edit.text()))
        edit.textChanged.connect(self._on_param_changed)
        r.addWidget(edit)
        r.addWidget(btn)
        c._edit = edit   # type: ignore[attr-defined]
        return c

    # ── signal handlers ───────────────────────────────────────────────────────

    def _on_gen_changed(self, name):
        self._generator_name = name
        self._populate_params(self._gen_combo, self._params_form,
                              self._param_widgets, self._desc_label,
                              self._lock_widgets)

    def _on_gen2_changed(self, _):
        self._populate_params(self._gen2_combo, self._params_form2,
                              self._param_widgets2, None, self._lock_widgets2)

    def _on_layer2_toggled(self, state):
        en = bool(state)
        self._layer2_controls.setEnabled(en)
        self._layer2_controls.setVisible(en)

    def _on_param_changed(self, *_):
        self.params_changed.emit(self._generator_name, self.get_params())

    def _on_lock_changed(self, *_):
        """A lock (layer 1 or layer 2) checkbox changed — notify evo panel."""
        self.lock_changed.emit(self.get_locked_params())
        self.lock_changed2.emit(self.get_locked_params2())
        self.params_changed.emit(self._generator_name, self.get_params())

    def _on_generate(self):
        self.generate_requested.emit(self._generator_name, self.get_params())

    # ── param read / write ────────────────────────────────────────────────────

    def _read_params(self, wd: dict) -> dict:
        result = {}
        for key, widget in wd.items():
            if hasattr(widget, "_edit"):
                result[key] = widget._edit.text()
            elif isinstance(widget, QComboBox):
                result[key] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
            elif isinstance(widget, QDoubleSpinBox):
                result[key] = widget.value()
            elif isinstance(widget, QSpinBox):
                result[key] = widget.value()
            elif isinstance(widget, QLineEdit):
                result[key] = widget.text()
        return result

    def _write_params(self, wd: dict, params: dict):
        for key, widget in wd.items():
            if key not in params:
                continue
            val = params[key]
            widget.blockSignals(True)
            if hasattr(widget, "_edit"):
                widget._edit.setText(str(val))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(val))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(val))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(val))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(val))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(val))
            widget.blockSignals(False)

    # ── public API ────────────────────────────────────────────────────────────

    def get_params(self)  -> dict: return self._read_params(self._param_widgets)
    def get_params2(self) -> dict: return self._read_params(self._param_widgets2)
    def get_generator_name(self)  -> str: return self._generator_name
    def get_generator2_name(self) -> str: return self._gen2_combo.currentText()

    def get_locked_params(self) -> set:
        """Return set of L1 param keys currently locked."""
        return {key for key, cb in self._lock_widgets.items() if cb.isChecked()}

    def get_locked_params2(self) -> set:
        """Return set of L2 param keys currently locked."""
        return {key for key, cb in self._lock_widgets2.items() if cb.isChecked()}

    def set_palette(self, palette: ColorPalette, layer: int = 0):
        if layer == 0:
            self._palette1 = palette
        else:
            self._palette2 = palette

    def get_second_layer_config(self) -> dict | None:
        if not self._layer2_check.isChecked():
            return None
        use_p2 = self._layer2_palette_check.isChecked()
        pal = (self._palette2 if use_p2 and self._palette2
               else self._palette1) or ColorPalette()
        return {
            "generator": self._gen2_combo.currentText(),
            "params":    self.get_params2(),
            "blend":     self._blend_combo.currentText(),
            "opacity":   self._opacity2_slider.value() / 100.0,
            "palette":   pal,
        }

    def get_current_gen_params_hint(self) -> tuple[str, dict]:
        return self._generator_name, self.get_params()

    def load_pattern(self, gen_name: str, params: dict):
        if gen_name in REGISTRY:
            self._gen_combo.blockSignals(True)
            self._gen_combo.setCurrentText(gen_name)
            self._gen_combo.blockSignals(False)
            self._generator_name = gen_name
            self._populate_params(self._gen_combo, self._params_form,
                                  self._param_widgets, self._desc_label,
                                  self._lock_widgets)
        self._write_params(self._param_widgets, params)

    def load_pattern_layer2(self, gen_name: str, params: dict):
        if not self._layer2_check.isChecked():
            self._layer2_check.setChecked(True)
        if gen_name in REGISTRY:
            self._gen2_combo.blockSignals(True)
            self._gen2_combo.setCurrentText(gen_name)
            self._gen2_combo.blockSignals(False)
            self._populate_params(self._gen2_combo, self._params_form2,
                                  self._param_widgets2, None, self._lock_widgets2)
        self._write_params(self._param_widgets2, params)
