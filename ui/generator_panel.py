"""
GeneratorPanel – Tab 2.

Changes from v1:
  • Special "folder" param type → shows a "Browse…" button (used by Collage).
  • Second generator layer section is fully wired:
      - Enable checkbox
      - Generator combo
      - Blend mode combo
      - Opacity slider
  • get_second_layer_config() returns None or a dict for main_window to blend.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QPushButton, QDoubleSpinBox,
    QSpinBox, QCheckBox, QScrollArea, QGroupBox,
    QFileDialog, QLineEdit, QSlider,
)
from PyQt6.QtCore import Qt, pyqtSignal

from generators import REGISTRY, get_generator
from config.defaults import BLEND_MODES


class GeneratorPanel(QWidget):
    params_changed     = pyqtSignal(str, dict)
    generate_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._generator_name: str = "Procedural Noise"
        self._param_widgets: dict[str, QWidget] = {}

        # Second layer state
        self._gen2_name: str = "Urban Geometric"
        self._param_widgets2: dict[str, QWidget] = {}

        self._build_ui()
        self._populate_params(self._gen_combo, self._params_form,
                              self._param_widgets, self._desc_label)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Primary generator ─────────────────────────────────────────────────
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Generator:"))
        self._gen_combo = QComboBox()
        for name in REGISTRY:
            self._gen_combo.addItem(name)
        self._gen_combo.currentTextChanged.connect(self._on_generator_changed)
        sel_row.addWidget(self._gen_combo, 1)
        root.addLayout(sel_row)

        self._desc_label = QLabel()
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color:#888;font-style:italic;")
        root.addWidget(self._desc_label)

        scroll1 = QScrollArea()
        scroll1.setWidgetResizable(True)
        scroll1.setFixedHeight(260)
        scroll1.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        params_widget1 = QWidget()
        self._params_form = QFormLayout(params_widget1)
        self._params_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        scroll1.setWidget(params_widget1)
        root.addWidget(scroll1)

        # Generate button
        gen_btn = QPushButton("▶  Generate")
        gen_btn.setFixedHeight(34)
        gen_btn.setStyleSheet(
            "QPushButton{background:#2e6b2e;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#3d8a3d;}"
        )
        gen_btn.clicked.connect(self._on_generate)
        root.addWidget(gen_btn)

        # ── Second generator layer ────────────────────────────────────────────
        layer2_group = QGroupBox("Second generator layer")
        layer2_group.setCheckable(False)
        l2_root = QVBoxLayout(layer2_group)

        enable_row = QHBoxLayout()
        self._layer2_check = QCheckBox("Enable layer 2")
        self._layer2_check.stateChanged.connect(self._on_layer2_toggled)
        enable_row.addWidget(self._layer2_check)
        l2_root.addLayout(enable_row)

        self._layer2_controls = QWidget()
        l2c = QVBoxLayout(self._layer2_controls)
        l2c.setContentsMargins(0, 0, 0, 0)

        gen2_row = QHBoxLayout()
        gen2_row.addWidget(QLabel("Generator:"))
        self._gen2_combo = QComboBox()
        for name in REGISTRY:
            self._gen2_combo.addItem(name)
        self._gen2_combo.setCurrentText("Urban Geometric")
        self._gen2_combo.currentTextChanged.connect(self._on_gen2_changed)
        gen2_row.addWidget(self._gen2_combo, 1)
        l2c.addLayout(gen2_row)

        blend_row = QHBoxLayout()
        blend_row.addWidget(QLabel("Blend:"))
        self._blend_combo = QComboBox()
        for bm in BLEND_MODES:
            self._blend_combo.addItem(bm)
        self._blend_combo.setCurrentText("overlay")
        blend_row.addWidget(self._blend_combo, 1)
        l2c.addLayout(blend_row)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity:"))
        self._opacity2_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity2_slider.setRange(0, 100)
        self._opacity2_slider.setValue(50)
        self._opacity2_label = QLabel("50%")
        self._opacity2_label.setFixedWidth(32)
        self._opacity2_slider.valueChanged.connect(
            lambda v: self._opacity2_label.setText(f"{v}%")
        )
        opacity_row.addWidget(self._opacity2_slider)
        opacity_row.addWidget(self._opacity2_label)
        l2c.addLayout(opacity_row)

        scroll2 = QScrollArea()
        scroll2.setWidgetResizable(True)
        scroll2.setFixedHeight(180)
        scroll2.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        params_widget2 = QWidget()
        self._params_form2 = QFormLayout(params_widget2)
        self._params_form2.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        scroll2.setWidget(params_widget2)
        l2c.addWidget(scroll2)

        self._layer2_controls.setEnabled(False)
        self._layer2_controls.setVisible(False)
        l2_root.addWidget(self._layer2_controls)

        root.addWidget(layer2_group)
        root.addStretch()

        # Populate layer 2 params
        self._populate_params(self._gen2_combo, self._params_form2,
                              self._param_widgets2, None)

    # ── param population ──────────────────────────────────────────────────────

    def _populate_params(self, combo: QComboBox, form: QFormLayout,
                         widget_dict: dict, desc_label):
        widget_dict.clear()
        while form.rowCount():
            form.removeRow(0)

        name = combo.currentText()
        gen  = get_generator(name)
        if desc_label:
            desc_label.setText(gen.description)
        schema = gen.get_param_schema()

        for key, spec in schema.items():
            label   = spec.get("label", key)
            tip     = spec.get("tip", "")
            default = spec.get("default")
            options = spec.get("options")
            ptype   = spec.get("type", "")

            if ptype == "folder":
                widget = self._make_folder_widget(key, default or "")
            elif options:
                widget = QComboBox()
                for opt in options:
                    widget.addItem(opt)
                if default in options:
                    widget.setCurrentText(default)
                widget.currentTextChanged.connect(self._on_param_changed)
            elif ptype == "bool" or isinstance(default, bool):
                widget = QCheckBox()
                widget.setChecked(bool(default))
                widget.stateChanged.connect(self._on_param_changed)
            elif isinstance(default, float):
                widget = QDoubleSpinBox()
                widget.setDecimals(3)
                widget.setRange(spec.get("min", 0.0), spec.get("max", 1.0))
                widget.setSingleStep(spec.get("step", 0.01))
                widget.setValue(default)
                widget.valueChanged.connect(self._on_param_changed)
            elif isinstance(default, int):
                widget = QSpinBox()
                widget.setRange(int(spec.get("min", 0)), int(spec.get("max", 100000)))
                widget.setSingleStep(int(spec.get("step", 1)))
                widget.setValue(default)
                widget.valueChanged.connect(self._on_param_changed)
            elif isinstance(default, str) and ptype == "str":
                widget = QLineEdit(default)
                widget.textChanged.connect(self._on_param_changed)
            else:
                continue

            widget.setToolTip(tip)
            form.addRow(label + ":", widget)
            widget_dict[key] = widget

    def _make_folder_widget(self, key: str, default_path: str) -> QWidget:
        """A QLineEdit + Browse button for folder-type params."""
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(default_path)
        edit.setPlaceholderText("(empty = built-in shapes)")
        btn  = QPushButton("…")
        btn.setFixedWidth(28)
        btn.setToolTip("Browse for folder of PNG shapes")

        def _browse():
            folder = QFileDialog.getExistingDirectory(self, "Select shape folder")
            if folder:
                edit.setText(folder)

        btn.clicked.connect(_browse)
        edit.textChanged.connect(self._on_param_changed)
        row.addWidget(edit)
        row.addWidget(btn)
        # Attach edit reference so get_params can read it
        container._edit = edit   # type: ignore[attr-defined]
        container.setToolTip(edit.placeholderText())
        return container

    # ── signal handlers ───────────────────────────────────────────────────────

    def _on_generator_changed(self, name: str):
        self._generator_name = name
        self._populate_params(self._gen_combo, self._params_form,
                              self._param_widgets, self._desc_label)

    def _on_gen2_changed(self, name: str):
        self._gen2_name = name
        self._populate_params(self._gen2_combo, self._params_form2,
                              self._param_widgets2, None)

    def _on_layer2_toggled(self, state: int):
        enabled = bool(state)
        self._layer2_controls.setEnabled(enabled)
        self._layer2_controls.setVisible(enabled)

    def _on_param_changed(self, *_):
        self.params_changed.emit(self._generator_name, self.get_params())

    def _on_generate(self):
        self.generate_requested.emit(self._generator_name, self.get_params())

    # ── public API ────────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        return self._read_params(self._param_widgets)

    def get_params2(self) -> dict:
        return self._read_params(self._param_widgets2)

    def _read_params(self, widget_dict: dict) -> dict:
        result = {}
        for key, widget in widget_dict.items():
            # Folder container
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

    def set_params(self, params: dict):
        self._write_params(self._param_widgets, params)

    def _write_params(self, widget_dict: dict, params: dict):
        for key, widget in widget_dict.items():
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

    def get_generator_name(self) -> str:
        return self._generator_name

    def get_second_layer_config(self) -> dict | None:
        """
        Returns None if layer 2 is disabled, otherwise:
        {
            "generator": str,
            "params":    dict,
            "blend":     str,
            "opacity":   float  (0–1),
        }
        """
        if not self._layer2_check.isChecked():
            return None
        return {
            "generator": self._gen2_combo.currentText(),
            "params":    self.get_params2(),
            "blend":     self._blend_combo.currentText(),
            "opacity":   self._opacity2_slider.value() / 100.0,
        }
