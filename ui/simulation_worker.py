"""
SimulationWorker: Runs the thermal storage simulation in a background thread.

The worker communicates with the UI thread via Qt signals so that the
user interface remains responsive during computation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thermal_energy_storage_model import (
    StorageConfig,
    StorageInputs,
    StorageOutputs,
    StorageState,
    ThermalStorage1D,
)


@dataclass
class SimPhase:
    """
    One operating phase with a constant operating mode.

    Attributes
    ----------
    mode : str
        Operating mode: ``"idle"``, ``"charge"``, ``"discharge"``, or ``"both"``.
    duration : float
        Duration of the phase [s].
    m_dot_charge : float
        Charging mass flow rate [kg/s]. Relevant only for ``"charge"`` and ``"both"``.
    T_charge_in : float
        Charging inlet temperature [°C].
    m_dot_discharge : float
        Discharging mass flow rate [kg/s]. Relevant only for ``"discharge"`` and ``"both"``.
    T_discharge_in : float
        Discharging inlet temperature [°C].
    z_charge_in : float or None
        Charging inlet height [m]. None → default position (top).
    z_charge_out : float or None
        Charging outlet height [m]. None → default position (bottom).
    z_discharge_in : float or None
        Discharging inlet height [m]. None → default position (bottom).
    z_discharge_out : float or None
        Discharging outlet height [m]. None → default position (top).
    """

    mode: str = "idle"
    duration: float = 3600.0
    m_dot_charge: float = 10.0
    T_charge_in: float = 85.0
    m_dot_discharge: float = 0.0
    T_discharge_in: float = 45.0
    z_charge_in: Optional[float] = None
    z_charge_out: Optional[float] = None
    z_discharge_in: Optional[float] = None
    z_discharge_out: Optional[float] = None

    def label(self) -> str:
        """Returns a human-readable label for the phase."""
        labels = {
            "idle": "Idle",
            "charge": "Charging",
            "discharge": "Discharging",
            "both": "Simultaneous",
        }
        h = self.duration / 3600.0
        return f"{labels.get(self.mode, self.mode)} ({h:.1f} h)"


class SimulationWorker(QThread):
    """
    Runs the thermal storage simulation in a separate QThread.

    Signals
    -------
    step_complete(float, StorageState, StorageOutputs)
        Emitted after each simulated timestep.
        Contains: simulation time [s], new state, outputs.
    phase_started(int, str)
        Emitted at the start of each phase.
        Contains: phase index, operating mode label.
    progress_updated(int)
        Progress 0–100 %.
    finished(list, list)
        Completion with complete state and output lists.
    error_occurred(str)
        Error message on unexpected exceptions.
    """

    step_complete = pyqtSignal(float, object, object)
    phase_started = pyqtSignal(int, str)
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(list, list)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        storage: ThermalStorage1D,
        initial_state: StorageState,
        phases: list[SimPhase],
        dt: float,
        update_every_n: int = 1,
        parent=None,
    ):
        """
        Parameters
        ----------
        storage : ThermalStorage1D
            Initialised storage model.
        initial_state : StorageState
            Initial state of the storage.
        phases : list[SimPhase]
            List of operating phases.
        dt : float
            Nominal timestep [s]. Sub-stepped as needed for CFL.
        update_every_n : int
            UI update only after every n-th step (performance).
        """
        super().__init__(parent)
        self.storage = storage
        self.initial_state = initial_state
        self.phases = phases
        self.dt = dt
        self.update_every_n = update_every_n

        self._pause_flag = False
        self._stop_flag = False
        self._all_states: list[StorageState] = []
        self._all_outputs: list[StorageOutputs] = []

    def pause(self):
        """Pause the simulation."""
        self._pause_flag = True

    def resume(self):
        """Resume a paused simulation."""
        self._pause_flag = False

    def stop(self):
        """Stop the simulation."""
        self._stop_flag = True
        self._pause_flag = False

    def run(self):
        """Main loop of the simulation thread."""
        try:
            state = self.initial_state
            total_duration = sum(p.duration for p in self.phases)
            elapsed = 0.0
            step_counter = 0

            for phase_idx, phase in enumerate(self.phases):
                if self._stop_flag:
                    break

                self.phase_started.emit(phase_idx, phase.label())
                inputs = self._build_inputs(phase, self.storage.config.height)

                # Number of nominal timesteps in this phase
                n_steps = max(1, int(round(phase.duration / self.dt)))
                actual_dt = phase.duration / n_steps

                # Check CFL and compute sub-steps
                m_max = max(phase.m_dot_charge, phase.m_dot_discharge, 1e-6)
                try:
                    dt_cfl = self.storage.max_stable_dt(m_max)
                    n_sub = max(1, int(np.ceil(actual_dt / (dt_cfl * 0.9))))
                except Exception:
                    n_sub = 1
                sub_dt = actual_dt / n_sub

                for step in range(n_steps):
                    if self._stop_flag:
                        break

                    # Pause loop
                    while self._pause_flag:
                        self.msleep(100)
                        if self._stop_flag:
                            break

                    # Sub-steps for CFL compliance
                    for _ in range(n_sub):
                        outputs = self.storage.step(state, sub_dt, inputs)
                        state = outputs.state

                    elapsed += actual_dt
                    self._all_states.append(state)
                    self._all_outputs.append(outputs)
                    step_counter += 1

                    # UI update only every update_every_n steps
                    if step_counter % self.update_every_n == 0:
                        self.step_complete.emit(state.time, state, outputs)

                    # Progress bar
                    progress = int(100 * elapsed / total_duration)
                    self.progress_updated.emit(min(progress, 99))

            if not self._stop_flag:
                # Send last state if not already sent
                if self._all_states and step_counter % self.update_every_n != 0:
                    self.step_complete.emit(
                        self._all_states[-1].time,
                        self._all_states[-1],
                        self._all_outputs[-1],
                    )
                self.progress_updated.emit(100)

            self.finished.emit(self._all_states, self._all_outputs)

        except Exception as exc:
            self.error_occurred.emit(f"{type(exc).__name__}: {exc}")

    def _build_inputs(self, phase: SimPhase, height: float) -> StorageInputs:
        """Build StorageInputs from a SimPhase."""
        if phase.mode == "idle":
            return StorageInputs(ports=[], hx_ports=[])

        m_c = phase.m_dot_charge if phase.mode in ("charge", "both") else 0.0
        m_d = phase.m_dot_discharge if phase.mode in ("discharge", "both") else 0.0

        return StorageInputs.two_port(
            m_dot_charge=m_c,
            T_charge_in=phase.T_charge_in,
            m_dot_discharge=m_d,
            T_discharge_in=phase.T_discharge_in,
            height=height,
            z_charge_in=phase.z_charge_in,
            z_charge_out=phase.z_charge_out,
            z_discharge_in=phase.z_discharge_in,
            z_discharge_out=phase.z_discharge_out,
        )
