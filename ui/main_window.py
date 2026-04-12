"""
Main window of the PyQt6 user interface.

Layout:
    Left:   ConfigPanel (geometry, fluid, losses, numerics)
    Centre: Tank3DWidget (3D visualisation)
    Right:  PlotsWidget (profile, temperatures, power, SOC)
    Bottom: SimControlWidget (phase table, Start/Pause/Stop)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from thermal_energy_storage_model import StorageConfig, ThermalStorage1D

from ui.config_panel import ConfigPanel
from ui.plots_widget import PlotsWidget
from ui.sim_control import SimControlWidget
from ui.simulation_worker import SimPhase, SimulationWorker
from ui.viz3d_widget import Tank3DWidget


class MainWindow(QMainWindow):
    """
    Main window of the 1D Thermal Storage Simulator.

    Connects all UI components and manages the simulation workflow.
    """

    # Port definitions for 3D visualisation (default: two-circuit)
    _DEFAULT_PORTS = [
        {"type": "charge_in",    "label": "Charging in",   "z_frac": 1.0},
        {"type": "charge_out",   "label": "Charging out",  "z_frac": 0.0},
        {"type": "discharge_in", "label": "Discharging in","z_frac": 0.0},
        {"type": "discharge_out","label": "Discharging out","z_frac": 1.0},
    ]

    def __init__(self):
        super().__init__()
        self._storage: Optional[ThermalStorage1D] = None
        self._worker: Optional[SimulationWorker] = None
        self._current_config: Optional[StorageConfig] = None

        # History data for export
        self._history_times: list[float] = []
        self._history_states: list = []
        self._history_outputs: list = []

        # Buffered 3D data for timer-based rendering
        self._pending_3d: Optional[dict] = None   # {config, T, T_lo, T_hi, ports}
        self._viz3d_timer = QTimer(self)
        self._viz3d_timer.setInterval(300)        # max. ~3 Hz frame rate
        self._viz3d_timer.timeout.connect(self._flush_3d_update)

        # Order matters: dock (sim_ctrl) before toolbar
        self._setup_window()
        self._setup_menubar()
        self._setup_central()
        self._setup_dock()       # creates self._sim_ctrl
        self._setup_toolbar()    # references self._sim_ctrl
        self._setup_statusbar()
        self._connect_signals()

        # Initial render after short delay (layout must be complete)
        QTimer.singleShot(200, self._initial_render)

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _setup_window(self):
        self.setWindowTitle("1D Thermal Storage Simulator")
        self.resize(1400, 820)
        self.setMinimumSize(900, 600)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 4px;
                margin-top: 6px;
                padding-top: 6px;
                font-weight: bold;
                color: #cba6f7;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #313244;
                color: #cdd6f4;
                padding: 5px 10px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }
            QTabBar::tab:selected {
                background: #45475a;
                color: #cba6f7;
            }
            QDoubleSpinBox, QSpinBox, QComboBox {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 3px;
                color: #cdd6f4;
                padding: 2px 4px;
            }
            QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
                border: 1px solid #cba6f7;
            }
            QCheckBox { color: #cdd6f4; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #585b70;
                border-radius: 2px;
                background: #313244;
            }
            QCheckBox::indicator:checked {
                background: #cba6f7;
                border-color: #cba6f7;
            }
            QScrollBar:vertical {
                background: #313244;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #585b70;
                border-radius: 4px;
            }
            QTableWidget {
                background: #313244;
                gridline-color: #45475a;
                color: #cdd6f4;
                selection-background-color: #45475a;
            }
            QHeaderView::section {
                background: #45475a;
                color: #cba6f7;
                padding: 3px;
                border: none;
                font-size: 9px;
            }
            QLabel { color: #cdd6f4; }
            QSplitter::handle { background: #45475a; }
        """)

    def _setup_menubar(self):
        mb = self.menuBar()
        mb.setStyleSheet(
            "QMenuBar { background: #181825; color: #cdd6f4; }"
            "QMenuBar::item:selected { background: #313244; }"
            "QMenu { background: #181825; color: #cdd6f4; border: 1px solid #45475a; }"
            "QMenu::item:selected { background: #313244; }"
        )

        # File
        m_file = mb.addMenu("&File")

        act_new = QAction("&New", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.setToolTip("Reset configuration")
        act_new.triggered.connect(self._on_new)
        m_file.addAction(act_new)

        act_open = QAction("&Open …", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._on_open_config)
        m_file.addAction(act_open)

        act_save = QAction("&Save …", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._on_save_config)
        m_file.addAction(act_save)

        m_file.addSeparator()

        m_export = m_file.addMenu("Export")
        act_exp_csv = QAction("Results as CSV …", self)
        act_exp_csv.triggered.connect(self._export_csv)
        m_export.addAction(act_exp_csv)

        act_exp_json = QAction("Results as JSON …", self)
        act_exp_json.triggered.connect(self._export_json)
        m_export.addAction(act_exp_json)

        m_file.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        # View
        m_view = mb.addMenu("&View")
        act_reset_view = QAction("Reset 3D view", self)
        act_reset_view.triggered.connect(
            lambda: self._viz3d.set_view_elevation(25.0, -60.0)
        )
        m_view.addAction(act_reset_view)

        act_refresh = QAction("Refresh all plots", self)
        act_refresh.triggered.connect(lambda: self._plots.refresh_all_plots())
        m_view.addAction(act_refresh)

        # Help
        m_help = mb.addMenu("&Help")
        act_about = QAction("&About …", self)
        act_about.triggered.connect(self._show_about)
        m_help.addAction(act_about)

    def _setup_toolbar(self):
        tb = QToolBar("Tools")
        tb.setMovable(False)
        tb.setStyleSheet(
            "QToolBar { background: #181825; border: none; spacing: 4px; padding: 2px; }"
            "QToolButton { background: #313244; color: #cdd6f4; border-radius: 3px; "
            "padding: 4px 8px; }"
            "QToolButton:hover { background: #45475a; }"
        )
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        act_run = QAction("▶ Start", self)
        act_run.setToolTip("Start simulation (F5)")
        act_run.setShortcut("F5")
        act_run.triggered.connect(self._sim_ctrl.btn_run.click)
        tb.addAction(act_run)

        act_pause = QAction("⏸ Pause", self)
        act_pause.setToolTip("Pause simulation (F6)")
        act_pause.setShortcut("F6")
        act_pause.triggered.connect(self._sim_ctrl.btn_pause.click)
        tb.addAction(act_pause)

        act_stop = QAction("■ Stop", self)
        act_stop.setToolTip("Stop simulation (F7)")
        act_stop.setShortcut("F7")
        act_stop.triggered.connect(self._sim_ctrl.btn_stop.click)
        tb.addAction(act_stop)

        tb.addSeparator()

        act_export = QAction("💾 Export", self)
        act_export.triggered.connect(self._export_csv)
        tb.addAction(act_export)

    def _setup_central(self):
        """QSplitter with ConfigPanel | Tank3DWidget | PlotsWidget."""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left: configuration
        self._config_panel = ConfigPanel()
        splitter.addWidget(self._config_panel)

        # Centre: 3D visualisation
        self._viz3d = Tank3DWidget()
        splitter.addWidget(self._viz3d)

        # Right: plots
        self._plots = PlotsWidget()
        splitter.addWidget(self._plots)

        # Ratio approx. 2:4:4
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 4)

        self.setCentralWidget(splitter)

    def _setup_dock(self):
        """Simulation control as bottom dock."""
        self._sim_ctrl = SimControlWidget()
        dock = QDockWidget("Simulation Control", self)
        dock.setWidget(self._sim_ctrl)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setStyleSheet(
            "QDockWidget::title { background: #181825; color: #cba6f7; "
            "padding: 4px; font-weight: bold; }"
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _setup_statusbar(self):
        sb = self.statusBar()
        sb.setStyleSheet("QStatusBar { background: #181825; color: #a6adc8; }")
        self._status_lbl = QLabel("Ready")
        sb.addWidget(self._status_lbl)

        self._lbl_sim_time = QLabel("")
        self._lbl_sim_time.setAlignment(Qt.AlignmentFlag.AlignRight)
        sb.addPermanentWidget(self._lbl_sim_time)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        # Configuration changes → 3D update
        self._config_panel.config_changed.connect(self._on_config_changed)

        # Simulation control
        self._sim_ctrl.run_requested.connect(self._start_simulation)
        self._sim_ctrl.pause_requested.connect(self._pause_simulation)
        self._sim_ctrl.stop_requested.connect(self._stop_simulation)
        self._sim_ctrl.export_requested.connect(self._export_csv)

        # Tab switch in PlotsWidget → redraw all plots
        self._plots.currentChanged.connect(
            lambda _: self._plots.refresh_all_plots()
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initial_render(self):
        """First render after the window is built."""
        try:
            config = self._config_panel.build_config()
            self._on_config_changed(config)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Config handling
    # ------------------------------------------------------------------

    def _on_config_changed(self, config: StorageConfig):
        """Called when the configuration changes."""
        self._current_config = config
        self._sim_ctrl.set_tank_height(config.height)

        # Rebuild storage (for max_stable_dt etc.)
        try:
            self._storage = ThermalStorage1D(config)
        except Exception as exc:
            self._set_status(f"Configuration error: {exc}", error=True)
            return

        # Update 3D view with uniform temperature
        T_init = self._sim_ctrl.spin_T_init.value()
        temperatures = np.full(config.n_nodes, T_init)
        ports = self._build_port_list(config)
        self._viz3d.update_tank(
            config, temperatures,
            T_min=T_init - 30.0, T_max=T_init + 30.0,
            ports=ports,
        )

        # CFL hint in status bar
        dt = self._sim_ctrl.spin_dt.value()
        try:
            dt_max = self._storage.max_stable_dt(10.0)
            if dt > dt_max:
                self._set_status(
                    f"CFL: dt={dt:.0f}s > dt_max≈{dt_max:.0f}s → "
                    f"auto sub-stepping active"
                )
            else:
                self._set_status(
                    f"Configuration OK – dt_max≈{dt_max:.0f}s at ṁ=10 kg/s"
                )
        except Exception:
            self._set_status("Configuration loaded")

    def _build_port_list(self, config: StorageConfig) -> list[dict]:
        """Build port list for 3D visualisation."""
        H = config.height
        return [
            {**p, "z": p["z_frac"] * H}
            for p in self._DEFAULT_PORTS
        ]

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _start_simulation(self, T_init: float, dt: float, phases: list[SimPhase]):
        """Start a new simulation run."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)

        if self._storage is None or self._current_config is None:
            QMessageBox.warning(self, "Error", "Please build a configuration first.")
            self._sim_ctrl.on_simulation_stopped()
            return

        # Reset history and buffered 3D data
        self._history_times.clear()
        self._history_states.clear()
        self._history_outputs.clear()
        self._plots.clear()
        self._pending_3d = None
        self._viz3d_timer.stop()

        # Initial state
        initial_state = self._storage.initialize(T_init)

        # Rebuild storage with fresh config (safe)
        try:
            self._storage = ThermalStorage1D(self._current_config)
        except Exception as exc:
            QMessageBox.critical(self, "Model error", str(exc))
            self._sim_ctrl.on_simulation_stopped()
            return

        # Start worker
        update_n = self._sim_ctrl.get_update_every_n()
        self._worker = SimulationWorker(
            self._storage, initial_state, phases, dt,
            update_every_n=update_n,
        )
        self._worker.step_complete.connect(self._on_step)
        self._worker.phase_started.connect(self._on_phase_started)
        self._worker.progress_updated.connect(self._sim_ctrl.progress_bar.setValue)
        self._worker.finished.connect(self._on_simulation_finished)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.start()

        self._set_status(f"Simulation started: {len(phases)} phases, T₀={T_init:.1f}°C")

    def _pause_simulation(self):
        """Pause / resume the simulation."""
        if self._worker is None:
            return
        if self._worker._pause_flag:
            self._worker.resume()
            self._set_status("Simulation resumed")
        else:
            self._worker.pause()
            self._set_status("Simulation paused")

    def _stop_simulation(self):
        """Stop the simulation."""
        if self._worker:
            self._worker.stop()
        self._viz3d_timer.stop()
        self._flush_3d_update()
        self._sim_ctrl.on_simulation_stopped()
        self._set_status("Simulation stopped")

    # ------------------------------------------------------------------
    # Worker signals
    # ------------------------------------------------------------------

    def _on_step(self, time_s: float, state, outputs):
        """Called after each displayed simulation step."""
        self._history_times.append(time_s)
        self._history_states.append(state)
        self._history_outputs.append(outputs)

        config = self._current_config
        if config is None:
            return

        # Temperatures
        T = state.temperatures  # index 0 = top
        n = len(T)
        T_top = T[0]
        T_mid = T[n // 2]
        T_bot = T[-1]

        # Outlet temperature (discharge out = top)
        T_out = T[0]
        if outputs.port_temperatures:
            T_out = outputs.port_temperatures[-1]

        # Power estimate (from Q_loss + port temperatures)
        Q_loss = abs(outputs.Q_loss)

        # Reconstruct charge/discharge power from port temperatures
        # Simplified: use current phase flows
        Q_charge = 0.0
        Q_discharge = 0.0
        phases = self._sim_ctrl.get_phases()
        # Determine current phase
        for ph in phases:
            if ph.mode in ("charge", "both"):
                from thermal_energy_storage_model import WaterProperties
                cp = 4187.0
                T_c_out = T[-1]
                Q_charge = ph.m_dot_charge * cp * abs(ph.T_charge_in - T_c_out)
            if ph.mode in ("discharge", "both"):
                cp = 4187.0
                T_d_out = T[0]
                Q_discharge = ph.m_dot_discharge * cp * abs(T_d_out - ph.T_discharge_in)
            break  # Use only first phase for estimate

        # SOC
        T_min_soc, T_max_soc = self._sim_ctrl.get_soc_limits()
        try:
            soc = self._storage.get_soc(state, T_min=T_min_soc, T_max=T_max_soc)
        except Exception:
            soc = 0.5

        # Buffer 3D data – QTimer renders at max. ~3 Hz
        T_lo = min(T_min_soc, float(T.min())) - 2.0
        T_hi = max(T_max_soc, float(T.max())) + 2.0
        self._pending_3d = {
            "config": config,
            "T": T.copy(),
            "T_lo": T_lo,
            "T_hi": T_hi,
            "ports": self._build_port_list(config),
        }
        if not self._viz3d_timer.isActive():
            self._viz3d_timer.start()

        # Profile plot
        z_nodes = np.linspace(0, config.height, n + 1)
        z_centers = (z_nodes[:-1] + z_nodes[1:]) / 2.0
        self._plots.update_profile(z_centers, T)

        # Time-series plots
        self._plots.append_step(
            time_s,
            T_top, T_mid, T_bot, T_out,
            Q_charge, Q_discharge, Q_loss,
            soc,
        )

        # Status bar
        self._lbl_sim_time.setText(
            f"t = {time_s/3600:.2f} h  |  T_top = {T_top:.1f} °C  |  "
            f"SOC = {soc*100:.1f} %"
        )

    def _flush_3d_update(self):
        """Called by QTimer: render the last buffered 3D frame."""
        if self._pending_3d is None:
            return
        d = self._pending_3d
        self._pending_3d = None
        self._viz3d.update_tank(
            d["config"], d["T"],
            T_min=d["T_lo"], T_max=d["T_hi"],
            ports=d["ports"],
        )

    def _on_phase_started(self, idx: int, label: str):
        """Show phase-start message in status bar."""
        self._set_status(f"Phase {idx+1}: {label}")
        self._sim_ctrl.on_simulation_step(
            self._sim_ctrl.progress_bar.value(), f"Phase {idx+1}: {label}"
        )

    def _on_simulation_finished(self, states: list, outputs: list):
        """Simulation completed."""
        self._viz3d_timer.stop()
        self._flush_3d_update()   # render last frame
        self._sim_ctrl.on_simulation_finished()
        n = len(states)
        total_h = (states[-1].time / 3600.0) if states else 0.0
        self._set_status(
            f"Simulation complete: {n} steps, {total_h:.1f} h simulated"
        )
        self._plots.refresh_all_plots()

    def _on_worker_error(self, msg: str):
        """Error in worker thread."""
        self._sim_ctrl.on_simulation_stopped()
        QMessageBox.critical(self, "Simulation error", msg)
        self._set_status(f"Error: {msg}", error=True)

    # ------------------------------------------------------------------
    # File actions
    # ------------------------------------------------------------------

    def _on_new(self):
        """Reset configuration and results."""
        reply = QMessageBox.question(
            self, "New",
            "Discard current results and reset configuration?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._plots.clear()
            self._history_times.clear()
            self._history_states.clear()
            self._history_outputs.clear()
            self._lbl_sim_time.setText("")
            self._set_status("Reset")

    def _on_open_config(self):
        """Load a configuration file (JSON)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open configuration", "", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            config = self._dict_to_config(data)
            self._config_panel.set_from_config(config)
            self._set_status(f"Configuration loaded: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    def _on_save_config(self):
        """Save the current configuration as JSON."""
        if self._current_config is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save configuration", "storage_config.json",
            "JSON files (*.json)"
        )
        if not path:
            return
        try:
            d = self._config_panel.config_to_dict(self._current_config)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
            self._set_status(f"Configuration saved: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    def _export_csv(self):
        """Export simulation results as CSV."""
        if not self._history_states:
            QMessageBox.information(self, "Export", "No simulation data available.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results", "storage_results.csv",
            "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            self._write_csv(path)
            self._set_status(f"Exported: {Path(path).name}")
            QMessageBox.information(
                self, "Export successful",
                f"Results saved:\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def _write_csv(self, path: str):
        """Write the result history as CSV."""
        config = self._current_config
        if config is None:
            return
        n = config.n_nodes

        T_min_soc, T_max_soc = self._sim_ctrl.get_soc_limits()

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")

            # Header
            node_headers = [f"T_node_{i}" for i in range(n)]
            writer.writerow(
                ["time_s", "time_h", "T_top_C", "T_mid_C", "T_bot_C",
                 "Q_loss_W", "SOC_frac"]
                + node_headers
            )

            for state, outputs in zip(self._history_states, self._history_outputs):
                T = state.temperatures
                try:
                    soc = self._storage.get_soc(state, T_min=T_min_soc, T_max=T_max_soc)
                except Exception:
                    soc = 0.0
                row = [
                    f"{state.time:.1f}",
                    f"{state.time/3600:.4f}",
                    f"{T[0]:.4f}",
                    f"{T[n//2]:.4f}",
                    f"{T[-1]:.4f}",
                    f"{outputs.Q_loss:.2f}",
                    f"{soc:.6f}",
                ] + [f"{t:.4f}" for t in T]
                writer.writerow(row)

    def _export_json(self):
        """Export results as JSON."""
        if not self._history_states:
            QMessageBox.information(self, "Export", "No simulation data available.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results (JSON)", "storage_results.json",
            "JSON files (*.json)"
        )
        if not path:
            return
        try:
            config = self._current_config
            T_min_soc, T_max_soc = self._sim_ctrl.get_soc_limits()
            data = {
                "config": self._config_panel.config_to_dict(config) if config else {},
                "steps": []
            }
            for state, outputs in zip(self._history_states, self._history_outputs):
                T = state.temperatures
                try:
                    soc = self._storage.get_soc(state, T_min=T_min_soc, T_max=T_max_soc)
                except Exception:
                    soc = 0.0
                data["steps"].append({
                    "time_s": state.time,
                    "temperatures": T.tolist(),
                    "Q_loss_W": outputs.Q_loss,
                    "soc": soc,
                })
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._set_status(f"JSON exported: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False):
        """Set the status bar message."""
        color = "#f38ba8" if error else "#a6adc8"
        self._status_lbl.setStyleSheet(f"color: {color};")
        self._status_lbl.setText(msg)

    def _show_about(self):
        QMessageBox.about(
            self, "About 1D Thermal Storage Simulator",
            "<b>1D Thermal Storage Simulator</b><br><br>"
            "Physics-based 1D model of a stratified hot-water storage tank<br>"
            "for co-simulation with district heating network software.<br><br>"
            "<b>Model:</b> Explicit Euler + TVD van Leer + buoyancy correction<br>"
            "<b>UI:</b> PyQt6 + matplotlib<br><br>"
            "Developed with Claude Code (Anthropic)"
        )

    def _dict_to_config(self, d: dict) -> StorageConfig:
        """Reconstruct a StorageConfig from a dict (simple)."""
        from thermal_energy_storage_model import (
            ConstantAmbientLoss, ConstantFluidProperties,
            CylinderGeometry, GroundTemperatureLoss,
            SplitAmbientLoss, TruncatedConeGeometry, WaterProperties,
        )
        gd = d.get("geometry", {})
        if gd.get("type") == "cone":
            geom = TruncatedConeGeometry(
                r_bottom=gd["r_bottom"], r_top=gd["r_top"], height=gd["height"]
            )
        else:
            geom = CylinderGeometry.from_volume(gd["volume"], gd["height"])

        ld = d.get("loss_model", {})
        if ld.get("type") == "split":
            loss = SplitAmbientLoss(
                U_lid=ld["U_lid"], U_wall_body=ld["U_wall_body"],
                T_ambient=ld["T_ambient"]
            )
        elif ld.get("type") == "ground":
            loss = GroundTemperatureLoss(
                U_loss=ld["U_loss"], T_surface=ld["T_surface"],
                T_deep=ld["T_deep"], depth_decay=ld["depth_decay"],
                burial_depth=ld["burial_depth"]
            )
        else:
            loss = ConstantAmbientLoss(
                U_loss=ld.get("U_loss", 0.3),
                T_ambient=ld.get("T_ambient", 10.0)
            )

        fd = d.get("fluid", {})
        if fd.get("type") == "water_properties":
            fluid = WaterProperties()
        else:
            fluid = ConstantFluidProperties(
                rho=fd.get("rho", 977.8),
                cp=fd.get("cp", 4187.0),
                lambda_fluid=fd.get("lambda_fluid", 0.663),
            )

        return StorageConfig(
            volume=geom.volume,
            height=geom.height,
            n_nodes=d.get("n_nodes", 20),
            geometry=geom,
            loss_model=loss,
            fluid=fluid,
            advection_scheme=d.get("advection_scheme", "tvd"),
            solver=d.get("solver", "explicit"),
            buoyancy=d.get("buoyancy", True),
            lambda_eff_factor=d.get("lambda_eff_factor", 5.0),
        )

    def closeEvent(self, event):
        """Stop simulation on close."""
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()
