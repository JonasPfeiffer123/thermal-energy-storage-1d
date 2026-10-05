#!/usr/bin/env python
"""
Wall-clock cost of a single ThermalStorage1D.step() call
==========================================================

Reproduces the co-simulation use case of DistrictHeatingSim: one step()
per hour (dt = 3600 s) for a full year (8 760 steps), with the solver
configuration used there (implicit, TVD, buoyancy, WaterProperties,
StorageInputs.two_port).

Scenario
--------
    Cylinder, 100 m³, 10 m high, ConstantAmbientLoss.
    Daily cycle: 0-7 h charging (85 °C), 8-11 h idle,
    12-19 h discharging (45 °C return), 20-23 h idle.
    Charge/discharge flow exchanges the tank volume in 8 h.

Measured (no profiler, time.perf_counter, inputs built beforehand):
    step       -- step() only
    step+reads -- step() plus the per-hour reads of DistrictHeatingSim
                  (outputs.Q_loss, state.temperatures/T_top/T_bottom/T_mean,
                  model.get_soc(state, T_min, T_max))

Each configuration runs ``--repeats`` times on a fresh storage; the minimum
and the median per-step time are reported.

Usage
-----
    python benchmark/benchmark_step_performance.py
    python benchmark/benchmark_step_performance.py --nodes 5 20 50 --repeats 7
    python benchmark/benchmark_step_performance.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_energy_storage_model import (
    ConstantAmbientLoss,
    CylinderGeometry,
    StorageConfig,
    StorageInputs,
    ThermalStorage1D,
    WaterProperties,
    __version__,
)

N_STEPS = 8760
DT = 3600.0
VOLUME = 100.0
HEIGHT = 10.0


def build_storage(n_nodes: int) -> ThermalStorage1D:
    geom = CylinderGeometry.from_volume(VOLUME, HEIGHT)
    config = StorageConfig(
        volume=VOLUME,
        height=HEIGHT,
        n_nodes=n_nodes,
        geometry=geom,
        fluid=WaterProperties(),
        loss_model=ConstantAmbientLoss(U_loss=0.3, T_ambient=10.0),
        solver="implicit",
        advection_scheme="tvd",
        buoyancy=True,
    )
    return ThermalStorage1D(config)


def build_inputs() -> list[StorageInputs]:
    m_dot = 977.8 * VOLUME / (8.0 * 3600.0)
    inputs = []
    for h in range(N_STEPS):
        hod = h % 24
        if hod < 8:
            inp = StorageInputs.two_port(m_dot_charge=m_dot, T_charge_in=85.0,
                                         height=HEIGHT)
        elif 12 <= hod < 20:
            inp = StorageInputs.two_port(m_dot_discharge=m_dot,
                                         T_discharge_in=45.0, height=HEIGHT)
        else:
            inp = StorageInputs(ports=[])
        inputs.append(inp)
    return inputs


def time_year(n_nodes: int, inputs: list[StorageInputs], reads: bool) -> float:
    """Seconds per step for one simulated year on a fresh storage."""
    storage = build_storage(n_nodes)
    state = storage.initialize(T_init=np.linspace(70.0, 45.0, n_nodes))
    step = storage.step
    if reads:
        get_soc = storage.get_soc
        t0 = time.perf_counter()
        for inp in inputs:
            out = step(state, DT, inp)
            state = out.state
            _ = (out.Q_loss, state.temperatures, state.T_top,
                 state.T_bottom, state.T_mean, get_soc(state, 45.0, 85.0))
        elapsed = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        for inp in inputs:
            state = step(state, DT, inp).state
        elapsed = time.perf_counter() - t0
    return elapsed / len(inputs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="step() wall-clock benchmark")
    p.add_argument("--nodes", type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--json", default=None, help="write results to this file")
    a = p.parse_args(argv)

    inputs = build_inputs()
    # Warm-up (imports, caches, CPU frequency ramp-up).
    time_year(a.nodes[0], inputs[:500], reads=False)

    print(f"thermal-energy-storage-1d {__version__} | Python "
          f"{platform.python_version()} | numpy {np.__version__} | "
          f"{platform.system()} {platform.machine()}")
    print(f"{'nodes':>5} | {'step min':>9} {'median':>9} | "
          f"{'step+reads min':>14} {'median':>9}   [us per step]")
    results = []
    for n in a.nodes:
        row: dict[str, float | int] = {"n_nodes": n}
        for label, reads in (("step", False), ("step_reads", True)):
            ts = [time_year(n, inputs, reads) * 1e6 for _ in range(a.repeats)]
            row[f"{label}_min_us"] = min(ts)
            row[f"{label}_median_us"] = statistics.median(ts)
        results.append(row)
        print(f"{n:5d} | {row['step_min_us']:9.1f} {row['step_median_us']:9.1f} | "
              f"{row['step_reads_min_us']:14.1f} {row['step_reads_median_us']:9.1f}")

    if a.json:
        Path(a.json).write_text(json.dumps({
            "version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": f"{platform.system()} {platform.machine()}",
            "repeats": a.repeats,
            "results": results,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
