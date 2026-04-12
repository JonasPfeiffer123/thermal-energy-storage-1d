"""
Configuration panel (left sidebar).

Contains tabs for geometry, fluid, losses, and numerics.
Emits ``config_changed`` on every parameter change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from thermal_energy_storage_model import (
    ConstantAmbientLoss,
    ConstantFluidProperties,
    CylinderGeometry,
    GroundTemperatureLoss,
    PointDiffusor,
    SplitAmbientLoss,
    StorageConfig,
    TransientGroundLoss,
    TruncatedConeGeometry,
    UniformDiffusor,
    WaterProperties,
)

try:
    from thermal_energy_storage_model import TruncatedPyramidGeometry
    _HAS_PYRAMID = True
except ImportError:
    _HAS_PYRAMID = False


def _make_dspin(
    value: float,
    min_val: float,
    max_val: float,
    step: float,
    decimals: int = 2,
    suffix: str = "",
    tooltip: str = "",
) -> QDoubleSpinBox:
    """Helper: create a configured QDoubleSpinBox."""
    sb = QDoubleSpinBox()
    sb.setRange(min_val, max_val)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setValue(value)
    if suffix:
        sb.setSuffix(f"  {suffix}")
    if tooltip:
        sb.setToolTip(tooltip)
    return sb


def _scrolled(widget: QWidget) -> QScrollArea:
    """Wrap a widget in a QScrollArea."""
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    return area


class ConfigPanel(QWidget):
    """
    Configuration panel with tabs for all model parameters.

    Signals
    -------
    config_changed(StorageConfig)
        Emitted on every parameter change.
    """

    config_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False  # guard against recursion when loading presets
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Preset bar at top
        preset_box = QHBoxLayout()
        preset_box.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "— manual —",
            "Steel tank (above-ground, 500 m³)",
            "Steel tank (buried, 1000 m³)",
            "PTES (pit storage, ~10000 m³)",
        ])
        self.preset_combo.setToolTip("Load a predefined storage configuration")
        preset_box.addWidget(self.preset_combo, 1)
        btn_load = QPushButton("Load")
        btn_load.setToolTip("Apply selected preset to all fields")
        btn_load.clicked.connect(self._load_preset)
        preset_box.addWidget(btn_load)
        layout.addLayout(preset_box)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(_scrolled(self._build_geometry_tab()), "Geometry")
        self.tabs.addTab(_scrolled(self._build_fluid_tab()), "Fluid")
        self.tabs.addTab(_scrolled(self._build_loss_tab()), "Losses")
        self.tabs.addTab(_scrolled(self._build_numerics_tab()), "Numerics")
        layout.addWidget(self.tabs, 1)

        self.setMinimumWidth(270)
        self.setMaximumWidth(340)

    def _build_geometry_tab(self) -> QWidget:
        """Tab: geometry model."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        # Model selection
        grp_type = QGroupBox("Geometry model")
        f = QFormLayout(grp_type)
        self.geo_type = QComboBox()
        items = ["Cylinder", "Truncated cone (PTES)"]
        if _HAS_PYRAMID:
            items.append("Truncated pyramid")
        self.geo_type.addItems(items)
        self.geo_type.setToolTip(
            "Cylinder for steel tanks; truncated cone for pit thermal energy storage"
        )
        f.addRow("Type:", self.geo_type)
        lay.addWidget(grp_type)

        # Stacked widget: different parameters per type
        self.geo_stack = QStackedWidget()

        # --- Page 0: Cylinder ---
        page_cyl = QWidget()
        fc = QFormLayout(page_cyl)
        self.geo_volume = _make_dspin(500.0, 0.1, 1e7, 10.0, 1, "m³",
                                      "Total volume of the storage")
        self.geo_height = _make_dspin(15.0, 0.1, 500.0, 0.5, 1, "m",
                                      "Height of the storage")
        fc.addRow("Volume:", self.geo_volume)
        fc.addRow("Height:", self.geo_height)
        # Computed radius (read-only)
        self.geo_radius_label = QLabel("—")
        self.geo_radius_label.setStyleSheet("color: gray;")
        fc.addRow("Radius (calc.):", self.geo_radius_label)
        self.geo_stack.addWidget(page_cyl)

        # --- Page 1: Truncated cone ---
        page_cone = QWidget()
        fco = QFormLayout(page_cone)
        self.cone_height = _make_dspin(15.0, 0.1, 500.0, 0.5, 1, "m",
                                       "Height of the truncated cone")
        self.cone_r_bottom = _make_dspin(40.0, 0.1, 500.0, 1.0, 1, "m",
                                         "Inner radius at the bottom")
        self.cone_r_top = _make_dspin(55.0, 0.1, 500.0, 1.0, 1, "m",
                                      "Inner radius at the top edge (larger = sloped sides)")
        fco.addRow("Height:", self.cone_height)
        fco.addRow("Radius bottom:", self.cone_r_bottom)
        fco.addRow("Radius top:", self.cone_r_top)
        self.cone_vol_label = QLabel("—")
        self.cone_vol_label.setStyleSheet("color: gray;")
        fco.addRow("Volume (calc.):", self.cone_vol_label)
        self.geo_stack.addWidget(page_cone)

        # --- Page 2: Truncated pyramid (optional) ---
        if _HAS_PYRAMID:
            page_pyr = QWidget()
            fp = QFormLayout(page_pyr)
            self.pyr_height = _make_dspin(15.0, 0.1, 500.0, 0.5, 1, "m", "Height")
            self.pyr_r_bottom = _make_dspin(40.0, 0.1, 500.0, 1.0, 1, "m",
                                            "Half side length at bottom")
            self.pyr_r_top = _make_dspin(55.0, 0.1, 500.0, 1.0, 1, "m",
                                         "Half side length at top")
            fp.addRow("Height:", self.pyr_height)
            fp.addRow("Side bottom:", self.pyr_r_bottom)
            fp.addRow("Side top:", self.pyr_r_top)
            self.geo_stack.addWidget(page_pyr)

        lay.addWidget(self.geo_stack)

        self.geo_type.currentIndexChanged.connect(self.geo_stack.setCurrentIndex)
        self.geo_type.currentIndexChanged.connect(self._update_geo_labels)
        self.geo_volume.valueChanged.connect(self._update_geo_labels)
        self.geo_height.valueChanged.connect(self._update_geo_labels)
        self.cone_r_bottom.valueChanged.connect(self._update_geo_labels)
        self.cone_r_top.valueChanged.connect(self._update_geo_labels)
        self.cone_height.valueChanged.connect(self._update_geo_labels)

        self._update_geo_labels()
        lay.addStretch()
        return w

    def _build_fluid_tab(self) -> QWidget:
        """Tab: fluid properties."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        grp_type = QGroupBox("Fluid model")
        f = QFormLayout(grp_type)
        self.fluid_type = QComboBox()
        self.fluid_type.addItems([
            "Water (constant, 70 °C)",
            "Water (T-dependent, FreeTTES polynomial)",
        ])
        self.fluid_type.setToolTip(
            "T-dependent properties recommended for operation over a wide temperature range"
        )
        f.addRow("Model:", self.fluid_type)
        lay.addWidget(grp_type)

        # Constant properties (active only for model 0)
        self.grp_const_fluid = QGroupBox("Constant fluid properties")
        fc = QFormLayout(self.grp_const_fluid)
        self.fluid_rho = _make_dspin(977.8, 500.0, 1200.0, 1.0, 1, "kg/m³",
                                     "Fluid density (water at 70 °C: 977.8 kg/m³)")
        self.fluid_cp = _make_dspin(4187.0, 1000.0, 10000.0, 10.0, 0, "J/(kg·K)",
                                    "Specific heat capacity (water: ~4187 J/(kg·K))")
        self.fluid_lambda = _make_dspin(0.663, 0.1, 2.0, 0.01, 3, "W/(m·K)",
                                        "Thermal conductivity of the fluid")
        fc.addRow("Density ρ:", self.fluid_rho)
        fc.addRow("cp:", self.fluid_cp)
        fc.addRow("λ:", self.fluid_lambda)
        lay.addWidget(self.grp_const_fluid)

        # Effective thermal conductivity
        grp_cond = QGroupBox("Effective thermal conductivity")
        fk = QFormLayout(grp_cond)
        self.fluid_lambda_factor = _make_dspin(
            5.0, 1.0, 200.0, 1.0, 1, "—",
            "λ_eff = factor × λ_fluid\n"
            "Accounts for turbulence at the thermocline. Typical: 1–20."
        )
        fk.addRow("λ factor:", self.fluid_lambda_factor)
        lay.addWidget(grp_cond)

        # T-dependent info
        self.lbl_water_info = QLabel(
            "Polynomial-based properties from FreeTTES\n"
            "Valid for T ∈ [0 °C … 130 °C].\n"
            "Density, cp and λ are computed as functions of temperature."
        )
        self.lbl_water_info.setWordWrap(True)
        self.lbl_water_info.setStyleSheet("color: #555; font-style: italic;")
        lay.addWidget(self.lbl_water_info)

        self.fluid_type.currentIndexChanged.connect(self._update_fluid_visibility)
        self._update_fluid_visibility()
        lay.addStretch()
        return w

    def _build_loss_tab(self) -> QWidget:
        """Tab: heat loss model."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        grp_type = QGroupBox("Loss model")
        f = QFormLayout(grp_type)
        self.loss_type = QComboBox()
        self.loss_type.addItems([
            "Constant ambient",
            "Split lid / wall",
            "Ground temperature (depth-dependent)",
            "Transient ground (1D-RC network)",
        ])
        self.loss_type.setToolTip(
            "Constant for above-ground tanks;\n"
            "Ground for buried tanks / pit storage;\n"
            "Transient for long-term simulations with seasonally varying ground."
        )
        f.addRow("Model:", self.loss_type)
        lay.addWidget(grp_type)

        self.loss_stack = QStackedWidget()

        # --- Page 0: Constant ambient ---
        p0 = QWidget()
        f0 = QFormLayout(p0)
        self.loss_u = _make_dspin(0.3, 0.0, 10.0, 0.05, 3, "W/(m²·K)",
                                  "Overall heat transfer coefficient of the tank wall\n"
                                  "Well insulated: 0.1–0.5 W/(m²·K)")
        self.loss_t_amb = _make_dspin(10.0, -30.0, 50.0, 1.0, 1, "°C",
                                      "Constant ambient temperature")
        f0.addRow("U-value:", self.loss_u)
        f0.addRow("T_ambient:", self.loss_t_amb)
        self.loss_stack.addWidget(p0)

        # --- Page 1: Split ---
        p1 = QWidget()
        f1 = QFormLayout(p1)
        self.loss_u_lid = _make_dspin(0.15, 0.0, 10.0, 0.05, 3, "W/(m²·K)",
                                      "U-value lid (free surface / liner)")
        self.loss_u_wall = _make_dspin(0.25, 0.0, 10.0, 0.05, 3, "W/(m²·K)",
                                       "U-value wall + floor (embedded in ground)")
        self.loss_t_amb_split = _make_dspin(10.0, -30.0, 50.0, 1.0, 1, "°C",
                                            "Ambient temperature")
        f1.addRow("U_lid:", self.loss_u_lid)
        f1.addRow("U_wall:", self.loss_u_wall)
        f1.addRow("T_ambient:", self.loss_t_amb_split)
        self.loss_stack.addWidget(p1)

        # --- Page 2: Ground temperature ---
        p2 = QWidget()
        f2 = QFormLayout(p2)
        self.loss_u_ground = _make_dspin(0.3, 0.0, 10.0, 0.05, 3, "W/(m²·K)",
                                         "Effective U-value (wall + ground)")
        self.loss_t_surface = _make_dspin(8.0, -10.0, 30.0, 0.5, 1, "°C",
                                          "Annual mean surface temperature")
        self.loss_t_deep = _make_dspin(11.0, -5.0, 30.0, 0.5, 1, "°C",
                                       "Asymptotic deep-ground temperature")
        self.loss_depth_decay = _make_dspin(2.0, 0.1, 20.0, 0.5, 1, "m",
                                            "Characteristic decay depth")
        self.loss_burial_depth = _make_dspin(0.5, 0.0, 20.0, 0.5, 1, "m",
                                             "Depth of tank top below grade")
        f2.addRow("U-value:", self.loss_u_ground)
        f2.addRow("T_surface:", self.loss_t_surface)
        f2.addRow("T_deep:", self.loss_t_deep)
        f2.addRow("Decay depth:", self.loss_depth_decay)
        f2.addRow("Burial depth:", self.loss_burial_depth)
        self.loss_stack.addWidget(p2)

        # --- Page 3: Transient ground (1D-RC) ---
        p3 = QWidget()
        f3 = QFormLayout(p3)
        self.loss_tgl_u_lid = _make_dspin(0.15, 0.0, 10.0, 0.05, 3, "W/(m²·K)",
                                          "U-value lid (steady-state, air)")
        self.loss_tgl_t_amb_lid = _make_dspin(10.0, -30.0, 50.0, 1.0, 1, "°C",
                                              "Air temperature at the lid")
        self.loss_tgl_lambda = _make_dspin(2.23, 0.1, 10.0, 0.1, 2, "W/(m·K)",
                                           "Thermal conductivity of soil\n"
                                           "Clay silt (PTES): 2.23 W/(m·K)")
        self.loss_tgl_rho = _make_dspin(2000.0, 500.0, 5000.0, 100.0, 0, "kg/m³",
                                        "Soil density (typical 1600–2200)")
        self.loss_tgl_cp = _make_dspin(800.0, 200.0, 2000.0, 50.0, 0, "J/(kg·K)",
                                       "Specific heat capacity of soil (typical 700–1000)")
        self.loss_tgl_d_total = _make_dspin(7.5, 0.5, 50.0, 0.5, 1, "m",
                                            "Total thickness of modelled soil\n"
                                            "U_ss = lambda/d; for PTES: 2.23/7.5 = 0.30 W/(m²K)")
        self.loss_tgl_n_layers = QSpinBox()
        self.loss_tgl_n_layers.setRange(1, 20)
        self.loss_tgl_n_layers.setValue(3)
        self.loss_tgl_n_layers.setToolTip("Number of RC layers (3–5 recommended)")
        self.loss_tgl_t_far = _make_dspin(8.0, -5.0, 30.0, 0.5, 1, "°C",
                                          "Far-field ground temperature (deep, constant)")
        f3.addRow("U_lid:", self.loss_tgl_u_lid)
        f3.addRow("T_air lid:", self.loss_tgl_t_amb_lid)
        f3.addRow("λ_soil:", self.loss_tgl_lambda)
        f3.addRow("ρ_soil:", self.loss_tgl_rho)
        f3.addRow("cp_soil:", self.loss_tgl_cp)
        f3.addRow("d_total:", self.loss_tgl_d_total)
        f3.addRow("Layers:", self.loss_tgl_n_layers)
        f3.addRow("T_far-field:", self.loss_tgl_t_far)
        self.loss_stack.addWidget(p3)

        lay.addWidget(self.loss_stack)
        self.loss_type.currentIndexChanged.connect(self.loss_stack.setCurrentIndex)
        lay.addStretch()
        return w

    def _build_numerics_tab(self) -> QWidget:
        """Tab: numerical settings."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        # Spatial discretisation
        grp_disc = QGroupBox("Spatial discretisation")
        fd = QFormLayout(grp_disc)
        self.num_nodes = QSpinBox()
        self.num_nodes.setRange(5, 500)
        self.num_nodes.setValue(20)
        self.num_nodes.setSuffix("  nodes")
        self.num_nodes.setToolTip(
            "Number of vertical layers.\n"
            "Recommended: 10–50 for co-simulation.\n"
            "Higher values → better resolution, slower computation."
        )
        fd.addRow("Node count N:", self.num_nodes)
        lay.addWidget(grp_disc)

        # Time integration
        grp_time = QGroupBox("Time integration")
        ft = QFormLayout(grp_time)
        self.num_solver = QComboBox()
        self.num_solver.addItems([
            "Explicit (Euler, CFL ≤ 1)",
            "Implicit (TDMA, unconditionally stable)",
        ])
        self.num_solver.setToolTip(
            "Explicit: fast, CFL condition must be satisfied.\n"
            "Implicit: stable for arbitrarily large timesteps."
        )
        ft.addRow("Solver:", self.num_solver)
        lay.addWidget(grp_time)

        # Advection scheme
        grp_adv = QGroupBox("Advection scheme")
        fa = QFormLayout(grp_adv)
        self.num_scheme = QComboBox()
        self.num_scheme.addItems([
            "TVD van Leer (2nd order, recommended)",
            "Upwind (1st order, diffusive)",
        ])
        self.num_scheme.setToolTip(
            "TVD: sharp thermocline, low numerical diffusion.\n"
            "Upwind: robust, but thermocline smears out."
        )
        fa.addRow("Scheme:", self.num_scheme)
        lay.addWidget(grp_adv)

        # Physical models
        grp_phys = QGroupBox("Physical corrections")
        fp = QFormLayout(grp_phys)
        self.num_buoyancy = QCheckBox("Buoyancy correction (convective adjustment)")
        self.num_buoyancy.setChecked(True)
        self.num_buoyancy.setToolTip(
            "Removes unstable density inversions (colder fluid above warmer).\n"
            "Recommended: enabled. Matches FreeTTES behaviour."
        )
        fp.addRow(self.num_buoyancy)
        lay.addWidget(grp_phys)

        # Diffusor
        grp_diff = QGroupBox("Diffusor model")
        fdi = QFormLayout(grp_diff)
        self.diff_type = QComboBox()
        self.diff_type.addItems([
            "Point diffusor (default)",
            "Uniform distribution (UniformDiffusor)",
        ])
        self.diff_type.setToolTip(
            "Point: entire mass flow injected at nearest node.\n"
            "Uniform: mass flow distributed over a zone\n"
            "(better FreeTTES agreement for N ≥ 50)."
        )
        fdi.addRow("Model:", self.diff_type)
        self.diff_h_zone = _make_dspin(1.0, 0.1, 20.0, 0.5, 1, "m",
                                       "Width of the distribution zone")
        self.diff_h_zone.setEnabled(False)
        fdi.addRow("Zone width:", self.diff_h_zone)
        self.diff_type.currentIndexChanged.connect(
            lambda i: self.diff_h_zone.setEnabled(i == 1)
        )
        lay.addWidget(grp_diff)

        lay.addStretch()
        return w

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        """Connect all widgets to _on_change."""
        widgets_dspin = [
            self.geo_volume, self.geo_height,
            self.cone_height, self.cone_r_bottom, self.cone_r_top,
            self.fluid_rho, self.fluid_cp, self.fluid_lambda, self.fluid_lambda_factor,
            self.loss_u, self.loss_t_amb,
            self.loss_u_lid, self.loss_u_wall, self.loss_t_amb_split,
            self.loss_u_ground, self.loss_t_surface, self.loss_t_deep,
            self.loss_depth_decay, self.loss_burial_depth,
            self.diff_h_zone,
        ]
        for w in widgets_dspin:
            w.valueChanged.connect(self._on_change)

        combos = [
            self.geo_type, self.fluid_type, self.loss_type,
            self.num_solver, self.num_scheme, self.diff_type,
        ]
        for c in combos:
            c.currentIndexChanged.connect(self._on_change)

        self.num_nodes.valueChanged.connect(self._on_change)
        self.num_buoyancy.stateChanged.connect(self._on_change)

        if _HAS_PYRAMID:
            self.pyr_height.valueChanged.connect(self._on_change)
            self.pyr_r_bottom.valueChanged.connect(self._on_change)
            self.pyr_r_top.valueChanged.connect(self._on_change)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_change(self, *_):
        """Emit config_changed with a new StorageConfig."""
        if self._updating:
            return
        try:
            config = self.build_config()
            self._update_geo_labels()
            self.config_changed.emit(config)
        except Exception:
            pass  # ignore invalid intermediate values

    def _update_geo_labels(self):
        """Update computed labels (radius, volume)."""
        import math
        # Cylinder: radius from V and H
        try:
            v = self.geo_volume.value()
            h = self.geo_height.value()
            if h > 0 and v > 0:
                r = math.sqrt(v / (math.pi * h))
                self.geo_radius_label.setText(f"{r:.2f} m")
            else:
                self.geo_radius_label.setText("—")
        except Exception:
            self.geo_radius_label.setText("—")

        # Truncated cone: volume
        try:
            h = self.cone_height.value()
            rb = self.cone_r_bottom.value()
            rt = self.cone_r_top.value()
            import math
            v = math.pi / 3.0 * h * (rb**2 + rb * rt + rt**2)
            self.cone_vol_label.setText(f"{v:.0f} m³")
        except Exception:
            self.cone_vol_label.setText("—")

    def _update_fluid_visibility(self):
        """Show/hide constant fluid parameters."""
        is_const = self.fluid_type.currentIndex() == 0
        self.grp_const_fluid.setVisible(is_const)
        self.lbl_water_info.setVisible(not is_const)

    def _load_preset(self):
        """Load a predefined preset and fill the widgets."""
        from thermal_energy_storage_model import StoragePresets

        idx = self.preset_combo.currentIndex()
        if idx == 0:
            return  # no selection

        try:
            if idx == 1:
                config = StoragePresets.steel_tank_aboveground(
                    volume=500.0, height=15.0, n_nodes=20,
                    U_loss=0.3, T_ambient=10.0,
                )
            elif idx == 2:
                config = StoragePresets.steel_tank_buried(
                    volume=1000.0, height=15.0, n_nodes=20,
                    U_loss=0.25, burial_depth=1.0,
                )
            else:
                config = StoragePresets.ptes(
                    r_bottom=40.0, r_top=55.0, height=15.0, n_nodes=30,
                )
            self.set_from_config(config)
        except Exception as exc:
            QMessageBox.warning(self, "Preset error", str(exc))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_config(self) -> StorageConfig:
        """Build a StorageConfig from the current widget values."""
        # --- Geometry ---
        geo_idx = self.geo_type.currentIndex()
        if geo_idx == 0:
            v = self.geo_volume.value()
            h = self.geo_height.value()
            geometry = CylinderGeometry.from_volume(v, h)
        elif geo_idx == 1:
            h = self.cone_height.value()
            rb = self.cone_r_bottom.value()
            rt = self.cone_r_top.value()
            geometry = TruncatedConeGeometry(r_bottom=rb, r_top=rt, height=h)
        else:
            h = self.pyr_height.value()
            rb = self.pyr_r_bottom.value()
            rt = self.pyr_r_top.value()
            # Square footprint: side length = 2 × half-side parameter
            geometry = TruncatedPyramidGeometry(
                a_bottom=2 * rb, b_bottom=2 * rb,
                a_top=2 * rt, b_top=2 * rt,
                height=h,
            )

        # --- Fluid ---
        if self.fluid_type.currentIndex() == 0:
            fluid = ConstantFluidProperties(
                rho=self.fluid_rho.value(),
                cp=self.fluid_cp.value(),
                lambda_fluid=self.fluid_lambda.value(),
            )
        else:
            fluid = WaterProperties()

        # --- Loss model ---
        loss_idx = self.loss_type.currentIndex()
        if loss_idx == 0:
            loss_model = ConstantAmbientLoss(
                U_loss=self.loss_u.value(),
                T_ambient=self.loss_t_amb.value(),
            )
        elif loss_idx == 1:
            loss_model = SplitAmbientLoss(
                U_lid=self.loss_u_lid.value(),
                U_wall=self.loss_u_wall.value(),
                T_ambient=self.loss_t_amb_split.value(),
            )
        elif loss_idx == 2:
            loss_model = GroundTemperatureLoss(
                U_loss=self.loss_u_ground.value(),
                T_surface=self.loss_t_surface.value(),
                T_deep=self.loss_t_deep.value(),
                depth_decay=self.loss_depth_decay.value(),
                burial_depth=self.loss_burial_depth.value(),
            )
        else:
            loss_model = TransientGroundLoss(
                U_lid=self.loss_tgl_u_lid.value(),
                T_ambient_lid=self.loss_tgl_t_amb_lid.value(),
                lambda_soil=self.loss_tgl_lambda.value(),
                rho_soil=self.loss_tgl_rho.value(),
                cp_soil=self.loss_tgl_cp.value(),
                d_total=self.loss_tgl_d_total.value(),
                n_layers=self.loss_tgl_n_layers.value(),
                T_far=self.loss_tgl_t_far.value(),
            )

        # --- Diffusor ---
        if self.diff_type.currentIndex() == 0:
            diffusor = PointDiffusor()
        else:
            diffusor = UniformDiffusor(H_zone=self.diff_h_zone.value())

        # --- Numerics ---
        scheme = "tvd" if self.num_scheme.currentIndex() == 0 else "upwind"
        solver = "explicit" if self.num_solver.currentIndex() == 0 else "implicit"

        return StorageConfig(
            volume=geometry.volume,
            height=geometry.height,
            n_nodes=self.num_nodes.value(),
            geometry=geometry,
            fluid=fluid,
            loss_model=loss_model,
            diffusor_model=diffusor,
            advection_scheme=scheme,
            solver=solver,
            buoyancy=self.num_buoyancy.isChecked(),
            lambda_eff_factor=self.fluid_lambda_factor.value(),
        )

    def set_from_config(self, config: StorageConfig):
        """
        Populate all widgets from an existing StorageConfig.
        Useful for presets and file loading.
        """
        self._updating = True
        try:
            # Geometry
            geom = config.geometry
            if geom is None:
                geom = CylinderGeometry.from_volume(config.volume, config.height)

            if isinstance(geom, CylinderGeometry):
                self.geo_type.setCurrentIndex(0)
                self.geo_volume.setValue(geom.volume)
                self.geo_height.setValue(geom.height)
            elif isinstance(geom, TruncatedConeGeometry):
                self.geo_type.setCurrentIndex(1)
                self.cone_height.setValue(geom.height)
                self.cone_r_bottom.setValue(geom.r_bottom)
                self.cone_r_top.setValue(geom.r_top)
            elif _HAS_PYRAMID and isinstance(geom, TruncatedPyramidGeometry):
                self.geo_type.setCurrentIndex(2)
                self.pyr_height.setValue(geom.height)
                self.pyr_r_bottom.setValue(geom.a_bottom / 2.0)
                self.pyr_r_top.setValue(geom.a_top / 2.0)

            # Fluid
            if isinstance(config.fluid, WaterProperties):
                self.fluid_type.setCurrentIndex(1)
            else:
                self.fluid_type.setCurrentIndex(0)
                if isinstance(config.fluid, ConstantFluidProperties):
                    self.fluid_rho.setValue(config.fluid.rho)
                    self.fluid_cp.setValue(config.fluid.cp)
                    self.fluid_lambda.setValue(config.fluid.lambda_fluid)

            self.fluid_lambda_factor.setValue(config.lambda_eff_factor)

            # Loss model
            loss = config.loss_model
            if isinstance(loss, ConstantAmbientLoss):
                self.loss_type.setCurrentIndex(0)
                self.loss_u.setValue(loss.U_loss)
                self.loss_t_amb.setValue(loss.T_ambient)
            elif isinstance(loss, SplitAmbientLoss):
                self.loss_type.setCurrentIndex(1)
                self.loss_u_lid.setValue(loss.U_lid)
                self.loss_u_wall.setValue(loss.U_wall)
                self.loss_t_amb_split.setValue(loss.T_ambient)
            elif isinstance(loss, GroundTemperatureLoss):
                self.loss_type.setCurrentIndex(2)
                self.loss_u_ground.setValue(loss.U_loss)
                self.loss_t_surface.setValue(loss.T_surface)
                self.loss_t_deep.setValue(loss.T_deep)
                self.loss_depth_decay.setValue(loss.depth_decay)
                self.loss_burial_depth.setValue(loss.burial_depth)
            elif isinstance(loss, TransientGroundLoss):
                self.loss_type.setCurrentIndex(3)
                self.loss_tgl_u_lid.setValue(loss.U_lid)
                self.loss_tgl_t_amb_lid.setValue(loss.T_ambient_lid)
                self.loss_tgl_lambda.setValue(loss.lambda_soil)
                self.loss_tgl_rho.setValue(loss.rho_soil)
                self.loss_tgl_cp.setValue(loss.cp_soil)
                self.loss_tgl_d_total.setValue(loss.d_total)
                self.loss_tgl_n_layers.setValue(loss.n_layers)
                self.loss_tgl_t_far.setValue(loss.T_far)

            # Numerics
            self.num_nodes.setValue(config.n_nodes)
            scheme_idx = 0 if config.advection_scheme == "tvd" else 1
            self.num_scheme.setCurrentIndex(scheme_idx)
            solver_idx = 0 if config.solver == "explicit" else 1
            self.num_solver.setCurrentIndex(solver_idx)
            self.num_buoyancy.setChecked(config.buoyancy)

        finally:
            self._updating = False

        self._update_fluid_visibility()
        self._update_geo_labels()
        self._on_change()

    def config_to_dict(self, config: StorageConfig) -> dict:
        """Serialise a StorageConfig to a JSON-compatible dict."""
        geom = config.geometry
        if isinstance(geom, CylinderGeometry):
            geom_d = {"type": "cylinder", "volume": geom.volume, "height": geom.height}
        elif isinstance(geom, TruncatedConeGeometry):
            geom_d = {"type": "cone", "r_bottom": geom.r_bottom,
                      "r_top": geom.r_top, "height": geom.height}
        else:
            geom_d = {"type": "unknown"}

        loss = config.loss_model
        if isinstance(loss, ConstantAmbientLoss):
            loss_d = {"type": "constant", "U_loss": loss.U_loss, "T_ambient": loss.T_ambient}
        elif isinstance(loss, SplitAmbientLoss):
            loss_d = {"type": "split", "U_lid": loss.U_lid,
                      "U_wall": loss.U_wall, "T_ambient": loss.T_ambient}
        elif isinstance(loss, GroundTemperatureLoss):
            loss_d = {"type": "ground", "U_loss": loss.U_loss,
                      "T_surface": loss.T_surface, "T_deep": loss.T_deep,
                      "depth_decay": loss.depth_decay, "burial_depth": loss.burial_depth}
        elif isinstance(loss, TransientGroundLoss):
            loss_d = {"type": "transient_ground", "U_lid": loss.U_lid,
                      "T_ambient_lid": loss.T_ambient_lid, "lambda_soil": loss.lambda_soil,
                      "rho_soil": loss.rho_soil, "cp_soil": loss.cp_soil,
                      "d_total": loss.d_total, "n_layers": loss.n_layers, "T_far": loss.T_far}
        else:
            loss_d = {}

        fluid = config.fluid
        if isinstance(fluid, WaterProperties):
            fluid_d = {"type": "water_properties"}
        elif isinstance(fluid, ConstantFluidProperties):
            fluid_d = {"type": "constant", "rho": fluid.rho, "cp": fluid.cp,
                       "lambda_fluid": fluid.lambda_fluid}
        else:
            fluid_d = {}

        return {
            "geometry": geom_d,
            "loss_model": loss_d,
            "fluid": fluid_d,
            "n_nodes": config.n_nodes,
            "advection_scheme": config.advection_scheme,
            "solver": config.solver,
            "buoyancy": config.buoyancy,
            "lambda_eff_factor": config.lambda_eff_factor,
        }
