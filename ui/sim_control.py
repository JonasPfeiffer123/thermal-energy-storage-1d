"""
Simulation control widget (bottom dock).

Contains:
  - Initial conditions (T_init, timestep)
  - Phase table (operating mode, duration, mass flows, temperatures)
  - Start / Pause / Stop buttons
  - Progress bar and status display
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from ui.simulation_worker import SimPhase


# Column index constants for the phase table
_COL_MODE = 0
_COL_DUR = 1
_COL_MC = 2
_COL_TC = 3
_COL_MD = 4
_COL_TD = 5
_COL_ZCI = 6
_COL_ZCO = 7
_COL_ZDI = 8
_COL_ZDO = 9


class SimControlWidget(QWidget):
    """
    Simulation control with phase table and run controls.

    Signals
    -------
    run_requested(float, float, list)
        Start simulation: T_init [°C], dt [s], phase list.
    pause_requested()
        Pause / resume simulation.
    stop_requested()
        Stop simulation.
    export_requested()
        Export results.
    """

    run_requested = pyqtSignal(float, float, list)
    pause_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tank_height: float = 15.0  # updated by main window
        self._paused = False
        self._running = False
        self._setup_ui()
        self._populate_default_phases()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 4, 6, 4)
        main_layout.setSpacing(8)

        # --- Initial conditions ---
        grp_init = QGroupBox("Initial Conditions")
        grp_init.setFixedWidth(220)
        f = QFormLayout(grp_init)
        f.setSpacing(4)

        self.spin_T_init = QDoubleSpinBox()
        self.spin_T_init.setRange(0.0, 130.0)
        self.spin_T_init.setValue(60.0)
        self.spin_T_init.setSuffix("  °C")
        self.spin_T_init.setDecimals(1)
        self.spin_T_init.setToolTip("Uniform initial temperature of the storage")
        f.addRow("T_initial:", self.spin_T_init)

        self.spin_dt = QDoubleSpinBox()
        self.spin_dt.setRange(1.0, 86400.0)
        self.spin_dt.setValue(300.0)
        self.spin_dt.setSuffix("  s")
        self.spin_dt.setDecimals(0)
        self.spin_dt.setToolTip(
            "Nominal timestep.\n"
            "CFL violations are automatically corrected by sub-stepping."
        )
        f.addRow("Timestep dt:", self.spin_dt)

        self.spin_update_n = QSpinBox()
        self.spin_update_n.setRange(1, 100)
        self.spin_update_n.setValue(1)
        self.spin_update_n.setSuffix("  steps")
        self.spin_update_n.setToolTip(
            "UI update after every n-th step.\n"
            "Higher value → faster simulation, fewer visualisation updates."
        )
        f.addRow("UI update every:", self.spin_update_n)

        self.lbl_soc_ref = QLabel("T_min: 40 °C / T_max: 90 °C")
        self.lbl_soc_ref.setStyleSheet("color: gray; font-size: 9px;")
        f.addRow("SOC reference:", self.lbl_soc_ref)

        self.spin_soc_tmin = QDoubleSpinBox()
        self.spin_soc_tmin.setRange(0.0, 80.0)
        self.spin_soc_tmin.setValue(40.0)
        self.spin_soc_tmin.setSuffix("  °C")
        self.spin_soc_tmin.setDecimals(0)
        self.spin_soc_tmin.setToolTip("Cold reference temperature for SOC calculation")
        f.addRow("T_min (SOC):", self.spin_soc_tmin)

        self.spin_soc_tmax = QDoubleSpinBox()
        self.spin_soc_tmax.setRange(20.0, 130.0)
        self.spin_soc_tmax.setValue(90.0)
        self.spin_soc_tmax.setSuffix("  °C")
        self.spin_soc_tmax.setDecimals(0)
        self.spin_soc_tmax.setToolTip("Hot reference temperature for SOC calculation")
        f.addRow("T_max (SOC):", self.spin_soc_tmax)

        main_layout.addWidget(grp_init)

        # --- Phase table ---
        grp_phases = QGroupBox("Operating Phases")
        phases_layout = QVBoxLayout(grp_phases)
        phases_layout.setSpacing(4)

        # Table
        self.phase_table = QTableWidget(0, 10)
        self.phase_table.setHorizontalHeaderLabels([
            "Mode", "Duration [h]",
            "ṁ_charge [kg/s]", "T_charge_in [°C]",
            "ṁ_discharge [kg/s]", "T_discharge_in [°C]",
            "z_charge_in [m]", "z_charge_out [m]",
            "z_discharge_in [m]", "z_discharge_out [m]",
        ])
        hdr = self.phase_table.horizontalHeader()
        for i in range(10):
            hdr.setSectionResizeMode(i, hdr.ResizeMode.ResizeToContents)
        self.phase_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.phase_table.setMinimumHeight(120)
        self.phase_table.setMaximumHeight(160)
        phases_layout.addWidget(self.phase_table)

        # Buttons below the table
        btn_row = QHBoxLayout()
        self.btn_add_phase = QPushButton("+ Phase")
        self.btn_add_phase.setToolTip("Add new phase")
        self.btn_add_phase.clicked.connect(self._add_phase_row)

        self.btn_dup_phase = QPushButton("Duplicate")
        self.btn_dup_phase.setToolTip("Duplicate selected phase")
        self.btn_dup_phase.clicked.connect(self._duplicate_phase)

        self.btn_del_phase = QPushButton("− Phase")
        self.btn_del_phase.setToolTip("Remove selected phase")
        self.btn_del_phase.clicked.connect(self._delete_phase)

        self.btn_clear_phases = QPushButton("Clear all")
        self.btn_clear_phases.clicked.connect(lambda: self.phase_table.setRowCount(0))

        for b in [self.btn_add_phase, self.btn_dup_phase,
                  self.btn_del_phase, self.btn_clear_phases]:
            btn_row.addWidget(b)
        btn_row.addStretch()

        phases_layout.addLayout(btn_row)
        main_layout.addWidget(grp_phases, 1)

        # --- Run controls (right) ---
        grp_ctrl = QGroupBox("Simulation")
        grp_ctrl.setFixedWidth(160)
        ctrl_layout = QVBoxLayout(grp_ctrl)
        ctrl_layout.setSpacing(6)

        self.btn_run = QPushButton("▶  Start")
        self.btn_run.setStyleSheet(
            "QPushButton { background: #a6e3a1; color: #1e1e2e; font-weight: bold; "
            "border-radius: 4px; padding: 6px; }"
            "QPushButton:hover { background: #94d48f; }"
        )
        self.btn_run.clicked.connect(self._on_run)

        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setStyleSheet(
            "QPushButton { background: #f9e2af; color: #1e1e2e; font-weight: bold; "
            "border-radius: 4px; padding: 6px; }"
            "QPushButton:disabled { background: #45475a; color: #585b70; }"
        )
        self.btn_pause.clicked.connect(self._on_pause)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "QPushButton { background: #f38ba8; color: #1e1e2e; font-weight: bold; "
            "border-radius: 4px; padding: 6px; }"
            "QPushButton:disabled { background: #45475a; color: #585b70; }"
        )
        self.btn_stop.clicked.connect(self._on_stop)

        self.btn_export = QPushButton("💾  Export")
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet(
            "QPushButton { background: #89b4fa; color: #1e1e2e; "
            "border-radius: 4px; padding: 6px; }"
            "QPushButton:disabled { background: #45475a; color: #585b70; }"
        )
        self.btn_export.clicked.connect(self.export_requested.emit)

        ctrl_layout.addWidget(self.btn_run)
        ctrl_layout.addWidget(self.btn_pause)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addSpacing(6)
        ctrl_layout.addWidget(self.btn_export)
        ctrl_layout.addStretch()

        # Progress bar + status label
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #585b70; border-radius: 3px; "
            "background: #313244; color: white; }"
            "QProgressBar::chunk { background: #a6e3a1; border-radius: 2px; }"
        )
        ctrl_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: #a6e3a1; font-size: 9px;")
        ctrl_layout.addWidget(self.lbl_status)

        main_layout.addWidget(grp_ctrl)

        self.setFixedHeight(210)

    # ------------------------------------------------------------------
    # Default phases
    # ------------------------------------------------------------------

    def _populate_default_phases(self):
        """Populate the table with a default demo scenario."""
        defaults = [
            ("charge",    8.0, 10.0, 85.0, 0.0, 45.0),
            ("idle",      4.0,  0.0, 85.0, 0.0, 45.0),
            ("discharge", 8.0,  0.0, 85.0, 8.0, 45.0),
        ]
        for mode, dur, mc, tc, md, td in defaults:
            self._add_phase_row(mode=mode, duration_h=dur,
                                m_c=mc, T_c=tc, m_d=md, T_d=td)

    # ------------------------------------------------------------------
    # Table operations
    # ------------------------------------------------------------------

    def _add_phase_row(
        self,
        mode: str = "idle",
        duration_h: float = 4.0,
        m_c: float = 0.0,
        T_c: float = 85.0,
        m_d: float = 0.0,
        T_d: float = 45.0,
        z_ci: str = "auto",
        z_co: str = "auto",
        z_di: str = "auto",
        z_do: str = "auto",
    ):
        """Add a new row to the phase table."""
        row = self.phase_table.rowCount()
        self.phase_table.insertRow(row)

        # Mode ComboBox
        combo = QComboBox()
        combo.addItems(["Idle", "Charging", "Discharging", "Simultaneous"])
        mode_map = {"idle": 0, "charge": 1, "discharge": 2, "both": 3}
        combo.setCurrentIndex(mode_map.get(mode, 0))
        combo.currentIndexChanged.connect(lambda i, r=row: self._on_mode_changed(r))
        self.phase_table.setCellWidget(row, _COL_MODE, combo)

        # Numeric fields
        def item(val, fmt=".1f"):
            it = QTableWidgetItem(f"{val:{fmt}}")
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            return it

        self.phase_table.setItem(row, _COL_DUR, item(duration_h))
        self.phase_table.setItem(row, _COL_MC, item(m_c))
        self.phase_table.setItem(row, _COL_TC, item(T_c))
        self.phase_table.setItem(row, _COL_MD, item(m_d))
        self.phase_table.setItem(row, _COL_TD, item(T_d))
        self.phase_table.setItem(row, _COL_ZCI, QTableWidgetItem(z_ci))
        self.phase_table.setItem(row, _COL_ZCO, QTableWidgetItem(z_co))
        self.phase_table.setItem(row, _COL_ZDI, QTableWidgetItem(z_di))
        self.phase_table.setItem(row, _COL_ZDO, QTableWidgetItem(z_do))

        self._update_row_colors(row)

    def _duplicate_phase(self):
        """Duplicate the selected phase."""
        rows = self.phase_table.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[-1].row()
        combo = self.phase_table.cellWidget(r, _COL_MODE)
        mode_keys = ["idle", "charge", "discharge", "both"]
        mode = mode_keys[combo.currentIndex()]

        def get(col, default="auto"):
            item = self.phase_table.item(r, col)
            return item.text() if item else default

        try:
            dur = float(get(_COL_DUR, "4.0"))
            mc = float(get(_COL_MC, "0.0"))
            tc = float(get(_COL_TC, "85.0"))
            md = float(get(_COL_MD, "0.0"))
            td = float(get(_COL_TD, "45.0"))
        except ValueError:
            dur, mc, tc, md, td = 4.0, 0.0, 85.0, 0.0, 45.0

        self._add_phase_row(mode, dur, mc, tc, md, td,
                            get(_COL_ZCI), get(_COL_ZCO),
                            get(_COL_ZDI), get(_COL_ZDO))

    def _delete_phase(self):
        """Delete the selected phase."""
        rows = self.phase_table.selectionModel().selectedRows()
        for r in sorted([idx.row() for idx in rows], reverse=True):
            self.phase_table.removeRow(r)

    def _on_mode_changed(self, row: int):
        """Colour the row according to the mode."""
        self._update_row_colors(row)

    def _update_row_colors(self, row: int):
        """Set row colour matching the operating mode."""
        combo = self.phase_table.cellWidget(row, _COL_MODE)
        if combo is None:
            return
        idx = combo.currentIndex()
        colors = {
            0: ("#45475a", "#cdd6f4"),   # Idle: grey
            1: ("#3b2d30", "#f38ba8"),   # Charging: reddish
            2: ("#2d3147", "#89b4fa"),   # Discharging: bluish
            3: ("#2d3b33", "#a6e3a1"),   # Simultaneous: greenish
        }
        bg, fg = colors.get(idx, ("#1e1e2e", "#cdd6f4"))
        for col in range(1, 10):
            item = self.phase_table.item(row, col)
            if item:
                item.setBackground(QColor(bg))
                item.setForeground(QColor(fg))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_run(self):
        """Start or resume the simulation."""
        phases = self.get_phases()
        if not phases:
            self.lbl_status.setText("No phases defined!")
            return

        self._running = True
        self._paused = False
        self.btn_run.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Running …")

        self.run_requested.emit(
            self.spin_T_init.value(),
            self.spin_dt.value(),
            phases,
        )

    def _on_pause(self):
        """Pause or resume the simulation."""
        self._paused = not self._paused
        if self._paused:
            self.btn_pause.setText("▶  Resume")
            self.lbl_status.setText("Paused")
        else:
            self.btn_pause.setText("⏸  Pause")
            self.lbl_status.setText("Running …")
        self.pause_requested.emit()

    def _on_stop(self):
        """Stop the simulation."""
        self.stop_requested.emit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_tank_height(self, height: float):
        """Update the tank height (for port height validation)."""
        self._tank_height = height

    def get_phases(self) -> list[SimPhase]:
        """Read all phases from the table and return a list."""
        phases = []
        mode_map = {0: "idle", 1: "charge", 2: "discharge", 3: "both"}
        H = self._tank_height

        for row in range(self.phase_table.rowCount()):
            combo = self.phase_table.cellWidget(row, _COL_MODE)
            if combo is None:
                continue
            mode = mode_map[combo.currentIndex()]

            def get_float(col: int, default: float) -> float:
                item = self.phase_table.item(row, col)
                if item is None:
                    return default
                try:
                    return float(item.text())
                except ValueError:
                    return default

            def get_z(col: int, default: float) -> float | None:
                item = self.phase_table.item(row, col)
                if item is None or item.text().strip().lower() == "auto":
                    return None
                try:
                    return float(item.text())
                except ValueError:
                    return None

            phases.append(SimPhase(
                mode=mode,
                duration=get_float(_COL_DUR, 4.0) * 3600.0,  # h → s
                m_dot_charge=get_float(_COL_MC, 0.0),
                T_charge_in=get_float(_COL_TC, 85.0),
                m_dot_discharge=get_float(_COL_MD, 0.0),
                T_discharge_in=get_float(_COL_TD, 45.0),
                z_charge_in=get_z(_COL_ZCI, H),
                z_charge_out=get_z(_COL_ZCO, 0.0),
                z_discharge_in=get_z(_COL_ZDI, 0.0),
                z_discharge_out=get_z(_COL_ZDO, H),
            ))
        return phases

    def get_soc_limits(self) -> tuple[float, float]:
        """Return T_min and T_max for SOC calculation [°C]."""
        return self.spin_soc_tmin.value(), self.spin_soc_tmax.value()

    def get_update_every_n(self) -> int:
        """UI update every n steps."""
        return self.spin_update_n.value()

    def on_simulation_step(self, progress: int, status: str = ""):
        """Called by the main window to update progress."""
        self.progress_bar.setValue(progress)
        if status:
            self.lbl_status.setText(status)

    def on_simulation_finished(self):
        """Reset controls after simulation ends."""
        self._running = False
        self._paused = False
        self.btn_run.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  Pause")
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(True)
        self.progress_bar.setValue(100)
        self.lbl_status.setText("Finished")

    def on_simulation_stopped(self):
        """Reset controls after manual stop."""
        self._running = False
        self._paused = False
        self.btn_run.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  Pause")
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(True)
        self.lbl_status.setText("Stopped")
