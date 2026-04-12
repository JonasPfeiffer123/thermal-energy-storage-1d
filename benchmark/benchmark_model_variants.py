"""
Model Variant Comparison: 0D to 1D-TVD vs. FreeTTES
=====================================================

Compares all relevant model configurations – from the fully mixed 0D model
(N=1) to the fine 1D-TVD grid – in terms of accuracy (MAE T_outlet vs.
FreeTTES) and computational speed.

Goal: Well-founded recommendation for coupling with district heating network
simulation models (e.g. pandapipes), which typically use implicit Euler with
time steps of 3600 s.

Scenario
--------
    Identical to the FreeTTES benchmark (benchmark_comparison.py):
    Phase 1 – Charging (t =  0 .. 23 h): 300 m³/h @ 90 °C
    Phase 2 – Idle     (t = 24 .. 35 h): no flow
    Phase 3 – Discharging (t = 36 .. 59 h): 300 m³/h @ 60 °C

Tested model variants
---------------------
    0D              : N=1,  Upwind, explicit  → fully mixed storage
    1D-N10-Upw-Expl : N=10, Upwind, explicit
    1D-N50-Upw-Expl : N=50, Upwind, explicit
    1D-N10-TVD-Expl : N=10, TVD,   explicit
    1D-N50-TVD-Expl : N=50, TVD,   explicit
    1D-N10-Upw-Impl : N=10, Upwind, implicit
    1D-N50-Upw-Impl : N=50, Upwind, implicit
    1D-N200-Upw-Impl: N=200,Upwind, implicit  (reference for convergence)

Output
------
    benchmark/results/model_variants/
        variants_scatter.svg       – Accuracy vs. speed
        variants_timeseries.svg    – T_outlet of selected variants
        variants_table.csv         – Complete results table
        variants_summary.json      – Machine-readable summary
        variants_recommendation.txt – Recommendation for co-simulation

Usage
-----
    python benchmark/benchmark_model_variants.py
    python benchmark/benchmark_model_variants.py --no-freetttes
"""

from __future__ import annotations

import sys
import csv
import json
import time
import warnings
import argparse

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_DIR   = BENCHMARK_DIR.parent
# Path to your FreeTTES installation (https://github.com/gewv-tu-dresden/FreeTTES).
# Set to None or use --no-freetttes to skip the FreeTTES reference comparison.
FREETTTES_SRC = Path(r"C:\path\to\FreeTTES\src")  # override with --freetttes-src
RESULTS_DIR   = BENCHMARK_DIR / "results" / "model_variants"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(FREETTTES_SRC))
sys.path.insert(0, str(BENCHMARK_DIR))

from thermal_energy_storage_model import (
    StorageConfig, StorageInputs, ThermalStorage1D, UniformDiffusor,
)
from config_benchmark import (
    R_INNER, H_WS, A_CROSS, V_TANK, U_WALL, T_AMB, T_REF_ENUTZ,
    T_CHARGE_IN, T_DISCH_IN, FLOW_M3H,
    H_B_UK_DIF, H_RS_DIF, H_WS_OK_DIF, Z_LOWER_DIFF, Z_UPPER_DIFF,
    DT_S, N_HOURS, T_END_CHARGE, T_END_IDLE, T_END_DISCH,
    T_BODEN, T_DR, START_PROFILE_FREETTTES,
    m_charge, m_discharge,
)
from shared_utils import (
    rho_water, m3h_to_kgs, compute_useful_energy_MWh,
    interpolate_start_profile, print_separator, phase_of, ZeroDStorage,
)


try:
    import FreeTTES_model as freetttes
    FREETTTES_AVAILABLE = True
except ImportError:
    FREETTTES_AVAILABLE = False
    warnings.warn(
        "FreeTTES not found – only the own model will be executed.",
        UserWarning, stacklevel=1,
    )


def _validate_freetttes_config() -> tuple[bool, str]:
    """
    Validate that FreeTTES geometry settings match this benchmark scenario.

    Returns
    -------
    tuple[bool, str]
        (is_valid, message). If invalid, message contains a readable mismatch
        summary and remediation hint.
    """
    try:
        from FreeTTES_config import ensure_initialized  # type: ignore
        speicher_param = ensure_initialized()
    except Exception as exc:  # pragma: no cover - defensive for external repo
        return (
            False,
            "  Could not load FreeTTES configuration "
            f"(FreeTTES_config.ensure_initialized): {exc}",
        )

    def _get_float(key: str) -> float | None:
        val = speicher_param.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    checks = [
        ("R_innen", R_INNER),
        ("H_WS_max", H_WS),
        ("H_B_UK_Dif", H_B_UK_DIF),
        ("H_RS_Dif", H_RS_DIF),
        ("H_WS_OK_Dif", H_WS_OK_DIF),
    ]

    mismatches = []
    atol = 1e-6
    rtol = 1e-3

    for key, bench_val in checks:
        free_val = _get_float(key)
        if free_val is None:
            mismatches.append(
                f"  - {key}: missing/invalid in FreeTTES config"
            )
            continue
        if not np.isclose(free_val, bench_val, rtol=rtol, atol=atol):
            mismatches.append(
                f"  - {key}: FreeTTES={free_val:g}, benchmark={bench_val:g}"
            )

    # Optional cross-check via A_Quer if present
    a_quer = _get_float("A_Quer")
    if a_quer is not None:
        a_bench = float(np.pi * R_INNER ** 2)
        if not np.isclose(a_quer, a_bench, rtol=rtol, atol=atol):
            mismatches.append(
                f"  - A_Quer: FreeTTES={a_quer:g}, benchmark={a_bench:g}"
            )

    if mismatches:
        message = "\n".join([
            "  FreeTTES configuration differs from benchmark scenario:",
            *mismatches,
            "  Hint: align FreeTTES src/config.json with benchmark/config_benchmark.py",
        ])
        return False, message

    return True, ""



# Model variants: (label, n_nodes, solver, scheme, headspace)
# headspace=True activates the optional headspace model (atmospheric storage).
# 0D is handled separately via run_zerod() (own class, N=1 not allowed).
MODEL_VARIANTS = [
    ("1D N=2  Upw-Expl",     2,   "explicit", "upwind", False),
    ("1D N=5  Upw-Expl",     5,   "explicit", "upwind", False),
    ("1D N=10 Upw-Expl",     10,  "explicit", "upwind", False),
    ("1D N=50 Upw-Expl",     50,  "explicit", "upwind", False),
    ("1D N=10 TVD-Expl",     10,  "explicit", "tvd",    False),
    ("1D N=50 TVD-Expl",     50,  "explicit", "tvd",    False),
    ("1D N=2  Upw-Impl",     2,   "implicit", "upwind", False),
    ("1D N=10 Upw-Impl",     10,  "implicit", "upwind", False),
    ("1D N=20 Upw-Impl",     20,  "implicit", "upwind", False),
    ("1D N=50 Upw-Impl",     50,  "implicit", "upwind", False),
    ("1D N=200 Upw-Impl",    200, "implicit", "upwind", False),
    ("1D N=10 TVD-Impl",     10,  "implicit", "tvd",    False),
    ("1D N=50 TVD-Impl",     50,  "implicit", "tvd",    False),
    ("1D N=200 TVD-Impl",    200, "implicit", "tvd",    False),
    # Headspace variants (headspace model, T_hs_init=99 °C)
    ("1D N=50 Upw-Impl+HS",  50,  "implicit", "upwind", True),
    ("1D N=50 TVD-Impl+HS",  50,  "implicit", "tvd",    True),
]

# Color mapping by solver/scheme
_COLORS = {
    ("explicit", "upwind"): "#4878cf",  # blue
    ("explicit", "tvd"):    "#6acc65",  # green
    ("implicit", "upwind"): "#d65f5f",  # red
    ("implicit", "tvd"):    "#e07b39",  # orange
}
_MARKERS = {
    1:   "s",    # square for 0D
    2:   "D",
    5:   "p",
    10:  "o",
    20:  "^",
    50:  "v",
    200: "P",
}


# ── 0D model (fully mixed storage) ────────────────────────────────────────────

def run_zerod() -> dict:
    """Runs the 0D model for the standard scenario."""
    print_separator("0D (fully mixed)")
    storage = ZeroDStorage()

    # Initial temperature from the FreeTTES profile: mean of all support points
    T_vals = list(START_PROFILE_FREETTTES.values())
    T = storage.initialize(float(np.mean(T_vals)))

    dz = H_WS / 20  # for E_nutz calculation: 20-node equivalent grid
    node_h = np.array([(20 - 0.5 - i) * dz for i in range(20)])

    records    = []
    step_times = []
    total_start = time.perf_counter()

    for t_h in range(N_HOURS):
        ph = phase_of(t_h)
        step_start = time.perf_counter()

        if t_h < T_END_CHARGE:
            T, _, T_out = storage.step(
                T, DT_S,
                m_dot_charge=m_charge, T_charge_in=T_CHARGE_IN,
            )
        elif t_h < T_END_IDLE:
            T, _, T_out = storage.step(T, DT_S)
            T_out = float("nan")
        else:
            T, T_out, _ = storage.step(
                T, DT_S,
                m_dot_discharge=m_discharge, T_discharge_in=T_DISCH_IN,
            )

        step_dt = time.perf_counter() - step_start
        step_times.append(step_dt)

        # E_nutz: homogeneous profile with temperature T
        T_uniform = np.full(20, T)
        E_nutz = compute_useful_energy_MWh(T_uniform, dz, node_h, T_REF_ENUTZ)

        # 0D: fully mixed → all heights have the same temperature
        records.append({"t_h": t_h, "phase": ph, "T_outlet": T_out,
                         "E_nutz": E_nutz, "step_time": step_dt,
                         "T_top": T, "T_mid": T, "T_bot": T})
        print(f"  t={t_h:3d}h [{ph:9s}]  T_out={T_out:6.2f} °C  "
              f"E_useful={E_nutz:8.2f} MWh  dt={step_dt*1000:.3f} ms")

    total_time = time.perf_counter() - total_start
    print_separator()
    print(f"  Total time: {total_time:.4f} s  |  "
          f"avg {total_time/N_HOURS*1000:.4f} ms/step")

    return {
        "label":      "0D (fully mixed)",
        "n_nodes":    1,
        "solver":     "explicit",
        "scheme":     "—",
        "headspace":  False,
        "times":      [r["t_h"] for r in records],
        "T_outlet":   [r["T_outlet"] for r in records],
        "E_nutz":     [r["E_nutz"] for r in records],
        "step_times": step_times,
        "total_time": total_time,
        "ms_per_step": total_time / N_HOURS * 1000,
        "dt_max_s":   float("inf"),
        "n_sub":      1,
        "T_top_series": [r["T_top"] for r in records],
        "T_mid_series": [r["T_mid"] for r in records],
        "T_bot_series": [r["T_bot"] for r in records],
        "T_headspace_series": [None] * len(records),
    }




# ── Run FreeTTES ───────────────────────────────────────────────────────────────

def run_freetttes() -> dict | None:
    """Runs FreeTTES benchmark; returns None if not available."""
    if not FREETTTES_AVAILABLE:
        print("  [FreeTTES] not available – skipped.")
        return None

    print_separator("FreeTTES")
    records    = []
    step_times = []
    _state     = None
    # FreeTTES benchmark expects support points inside (0, H_WS) only.
    # Boundary points are added internally by FreeTTES.
    start_profile_freetttes = {
        z: T for z, T in START_PROFILE_FREETTTES.items() if 0.0 < z < H_WS
    }
    total_start = time.perf_counter()

    for t in range(N_HOURS):
        step_start = time.perf_counter()
        ph = phase_of(t)

        if t < T_END_CHARGE:
            result = freetttes.main(
                t=t, dt=DT_S,
                m_VL=m_charge, m_RL=-m_charge,
                T_Zustrom=T_CHARGE_IN, T_amb=T_AMB,
                eingabe_volumen=False,
                zustand_uebernehmen=(t == 0),
                zustand=start_profile_freetttes.copy() if t == 0 else {},
                _state=_state,
            )
        elif t < T_END_IDLE:
            result = freetttes.main(
                t=t, dt=DT_S,
                m_VL=0, m_RL=0,
                T_Zustrom=T_CHARGE_IN, T_amb=T_AMB,
                eingabe_volumen=False,
                zustand_uebernehmen=False, zustand={},
                _state=_state,
            )
        else:
            result = freetttes.main(
                t=t, dt=DT_S,
                m_VL=-m_discharge, m_RL=m_discharge,
                T_Zustrom=T_DISCH_IN, T_amb=T_AMB,
                eingabe_volumen=False,
                zustand_uebernehmen=False, zustand={},
                _state=_state,
            )

        _state     = result["_state"]
        step_dt    = time.perf_counter() - step_start
        step_times.append(step_dt)
        T_out      = result["T_Austritt"]
        E_nutz     = result["E_nutz"] / 3.6

        # Interpolate temperatures at 3 heights from the storage state
        sz      = result["speicherzustand"]
        _h_pts  = np.array(sorted(sz.keys()), dtype=float)
        _T_pts  = np.array([sz[h][0] for h in _h_pts], dtype=float)
        T_top_r = float(np.interp(Z_UPPER_DIFF, _h_pts, _T_pts))
        T_mid_r = float(np.interp(H_WS / 2,    _h_pts, _T_pts))
        T_bot_r = float(np.interp(Z_LOWER_DIFF, _h_pts, _T_pts))

        records.append({"t_h": t, "phase": ph, "T_outlet": T_out,
                         "E_nutz": E_nutz, "step_time": step_dt,
                         "T_top": T_top_r, "T_mid": T_mid_r, "T_bot": T_bot_r})
        print(f"  t={t:3d}h [{ph:9s}]  T_out={T_out:6.2f} °C  "
              f"E_useful={E_nutz:8.2f} MWh  dt={step_dt:.3f} s")

    total_time = time.perf_counter() - total_start
    print_separator()
    print(f"  Total time: {total_time:.2f} s  |  "
          f"avg {total_time/N_HOURS*1000:.1f} ms/step")

    return {
        "label":      "FreeTTES",
        "n_nodes":    None,
        "solver":     "lagrange-explicit",
        "scheme":     "—",
        "times":      [r["t_h"] for r in records],
        "T_outlet":   [r["T_outlet"] for r in records],
        "E_nutz":     [r["E_nutz"] for r in records],
        "step_times": step_times,
        "total_time": total_time,
        "ms_per_step": total_time / N_HOURS * 1000,
        "T_top_series": [r["T_top"] for r in records],
        "T_mid_series": [r["T_mid"] for r in records],
        "T_bot_series": [r["T_bot"] for r in records],
    }


# ── Run own model ──────────────────────────────────────────────────────────────

def run_variant(
    label: str,
    n_nodes: int,
    solver: str,
    scheme: str,
    headspace: bool = False,
) -> dict:
    """Runs a model variant for the standard 60 h scenario."""
    print_separator(label)

    diffusor = UniformDiffusor(H_zone=H_RS_DIF)

    config = StorageConfig(
        volume=V_TANK,
        height=H_WS,
        n_nodes=n_nodes,
        rho=977.8,
        cp=4187.0,
        lambda_fluid=0.663,
        lambda_eff_factor=5.0,
        U_loss=U_WALL,
        T_ambient=T_AMB,
        advection_scheme=scheme,
        buoyancy=True,
        solver=solver,
        diffusor_model=diffusor,
        headspace=headspace,
        T_headspace_init=T_DR,
    )
    storage = ThermalStorage1D(config)

    dz = H_WS / n_nodes
    node_heights_from_bottom = np.array(
        [(n_nodes - 0.5 - i) * dz for i in range(n_nodes)]
    )
    T_start = interpolate_start_profile(
        START_PROFILE_FREETTTES, node_heights_from_bottom
    )
    state = storage.initialize(T_init=T_start)

    # Node indices for internal temperature tracking (3 reference heights)
    idx_top = int(np.argmin(np.abs(node_heights_from_bottom - Z_UPPER_DIFF)))
    idx_mid = int(np.argmin(np.abs(node_heights_from_bottom - H_WS / 2)))
    idx_bot = int(np.argmin(np.abs(node_heights_from_bottom - Z_LOWER_DIFF)))

    # Sub-stepping only required for explicit solver
    m_max   = max(m_charge, m_discharge)
    dt_max  = storage.max_stable_dt(m_max)

    if solver == "implicit":
        n_sub  = 1
        dt_sub = float(DT_S)
        print(f"  N={n_nodes:3d}  dz={dz:.2f} m  IMPLICIT  dt={DT_S} s  "
              f"(CFL limit would be {dt_max:.0f} s)")
    else:
        if dt_max < DT_S:
            n_sub  = int(np.ceil(DT_S / dt_max)) + 1
            dt_sub = DT_S / n_sub
        else:
            n_sub  = 1
            dt_sub = float(DT_S)
        print(f"  N={n_nodes:3d}  dz={dz:.2f} m  EXPLICIT  "
              f"dt_max={dt_max:.0f} s  Substeps={n_sub}×{dt_sub:.0f} s")

    records    = []
    step_times = []
    total_start = time.perf_counter()

    for t_h in range(N_HOURS):
        ph = phase_of(t_h)
        step_start = time.perf_counter()

        if t_h < T_END_CHARGE:
            inputs = StorageInputs.two_port(
                m_dot_charge=m_charge,
                T_charge_in=T_CHARGE_IN,
                height=H_WS,
                z_charge_in=Z_UPPER_DIFF,
                z_charge_out=Z_LOWER_DIFF,
            )
        elif t_h < T_END_IDLE:
            inputs = StorageInputs()
        else:
            inputs = StorageInputs.two_port(
                m_dot_discharge=m_discharge,
                T_discharge_in=T_DISCH_IN,
                height=H_WS,
                z_discharge_in=Z_LOWER_DIFF,
                z_discharge_out=Z_UPPER_DIFF,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(n_sub):
                outputs = storage.step(state, dt=dt_sub, inputs=inputs)
                state   = outputs.state

        step_dt = time.perf_counter() - step_start
        step_times.append(step_dt)

        if t_h < T_END_CHARGE:
            T_out = outputs.port_temperatures[1]
        elif t_h < T_END_IDLE:
            T_out = float("nan")
        else:
            T_out = outputs.port_temperatures[1]

        E_nutz = compute_useful_energy_MWh(
            state.temperatures, dz, node_heights_from_bottom, T_REF_ENUTZ
        )
        T_hs_val = outputs.T_headspace
        records.append({"t_h": t_h, "phase": ph, "T_outlet": T_out,
                         "E_nutz": E_nutz, "step_time": step_dt,
                         "T_top": float(state.temperatures[idx_top]),
                         "T_mid": float(state.temperatures[idx_mid]),
                         "T_bot": float(state.temperatures[idx_bot]),
                         "T_headspace": T_hs_val})

        hs_str = f"  T_hs={T_hs_val:6.2f} °C" if T_hs_val is not None else ""
        print(f"  t={t_h:3d}h [{ph:9s}]  T_out={T_out:6.2f} °C  "
              f"E_useful={E_nutz:8.2f} MWh  dt={step_dt*1000:.2f} ms{hs_str}")

    total_time = time.perf_counter() - total_start
    print_separator()
    print(f"  Total time: {total_time:.3f} s  |  "
          f"avg {total_time/N_HOURS*1000:.3f} ms/step")

    return {
        "label":      label,
        "n_nodes":    n_nodes,
        "solver":     solver,
        "scheme":     scheme,
        "headspace":  headspace,
        "times":      [r["t_h"] for r in records],
        "T_outlet":   [r["T_outlet"] for r in records],
        "E_nutz":     [r["E_nutz"] for r in records],
        "step_times": step_times,
        "total_time": total_time,
        "ms_per_step": total_time / N_HOURS * 1000,
        "dt_max_s":   dt_max,
        "n_sub":      n_sub,
        "T_top_series": [r["T_top"] for r in records],
        "T_mid_series": [r["T_mid"] for r in records],
        "T_bot_series": [r["T_bot"] for r in records],
        "T_headspace_series": [r["T_headspace"] for r in records],
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(result: dict, reference: dict) -> dict:
    """
    Computes accuracy metrics relative to the FreeTTES reference model.

    Only hours with valid (non-NaN) T_outlet in both time series are evaluated
    (charging and discharging phases).
    """
    T_our = np.array(result["T_outlet"], dtype=float)
    T_ref = np.array(reference["T_outlet"], dtype=float)

    # Only hours with flow (no NaN)
    mask = ~(np.isnan(T_our) | np.isnan(T_ref))

    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"),
                "max_err": float("nan"), "bias": float("nan")}

    err  = T_our[mask] - T_ref[mask]
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    max_err = float(np.max(np.abs(err)))
    bias    = float(np.mean(err))

    return {"mae": mae, "rmse": rmse, "max_err": max_err, "bias": bias}


# ── Results table ──────────────────────────────────────────────────────────────

def print_results_table(all_results: list[dict], reference: dict | None) -> None:
    """Prints the complete results table to the console."""
    print_separator("Results table")

    hdr = (f"{'Model':<24} {'N':>5}  {'Solver':<8}  {'Scheme':<7}  "
           f"{'ms/step':>11}  {'Factor vs. FT':>13}  "
           f"{'MAE [K]':>8}  {'RMSE [K]':>9}  {'Max [K]':>8}  "
           f"{'CFL-stable':>11}")
    print(hdr)
    print("-" * len(hdr))

    ref_ms = reference["ms_per_step"] if reference else None

    for r in all_results:
        metrics  = compute_metrics(r, reference) if reference else {}
        ms       = r["ms_per_step"]
        speedup  = f"{ref_ms/ms:.0f}×" if ref_ms else "—"
        mae_s    = f"{metrics.get('mae', float('nan')):8.3f}" if metrics else f"{'—':>8}"
        rmse_s   = f"{metrics.get('rmse', float('nan')):9.3f}" if metrics else f"{'—':>9}"
        max_s    = f"{metrics.get('max_err', float('nan')):8.3f}" if metrics else f"{'—':>8}"
        stable   = "yes" if r["solver"] == "implicit" or r.get("n_sub", 1) == 1 else f"no (×{r['n_sub']})"
        print(f"  {r['label']:<22}  {r['n_nodes'] or '—':>5}  "
              f"{r['solver']:<8}  {r['scheme']:<7}  "
              f"{ms:>11.3f}  {speedup:>13}  "
              f"{mae_s}  {rmse_s}  {max_s}  {stable:>11}")

    if reference:
        ms = reference["ms_per_step"]
        print(f"  {'FreeTTES (reference)':<22}  {'—':>5}  "
              f"{'lagrange':<8}  {'—':<7}  "
              f"{ms:>11.1f}  {'1×':>13}  "
              f"{'0.000':>8}  {'0.000':>9}  {'0.000':>8}  {'—':>11}")
    print_separator()


# ── Plots ─────────────────────────────────────────────────────────────────────

def _phase_shading(ax) -> None:
    ax.axvspan(0,               T_END_CHARGE - 0.5, alpha=0.07, color="steelblue",
               label="_nolegend_")
    ax.axvspan(T_END_CHARGE - 0.5, T_END_IDLE - 0.5, alpha=0.07, color="gold",
               label="_nolegend_")
    ax.axvspan(T_END_IDLE - 0.5,   T_END_DISCH,       alpha=0.07, color="tomato",
               label="_nolegend_")
    ax.axvline(T_END_CHARGE, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(T_END_IDLE,   color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)


def plot_scatter(all_results: list[dict], reference: dict | None,
                 out_dir: Path) -> None:
    """
    Accuracy vs. speed (scatter plot).

    x-axis: speedup factor relative to FreeTTES (log)
    y-axis: MAE T_outlet [K] vs. FreeTTES
    """
    if not reference:
        print("  [scatter] No reference run – plot skipped.")
        return

    ref_ms = reference["ms_per_step"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in all_results:
        metrics = compute_metrics(r, reference)
        mae     = metrics["mae"]
        if np.isnan(mae):
            continue

        speedup = ref_ms / r["ms_per_step"]
        color   = _COLORS.get((r["solver"], r["scheme"]), "gray")
        marker  = _MARKERS.get(r["n_nodes"], "o")
        ax.scatter(speedup, mae, color=color, marker=marker,
                   s=120, zorder=5, edgecolors="white", linewidths=0.5)
        # Label slightly offset
        ax.annotate(
            r["label"].replace("1D ", "").replace(" Upw", "\nUpw").replace(" TVD", "\nTVD"),
            (speedup, mae),
            fontsize=7.5, ha="left", va="bottom",
            xytext=(6, 4), textcoords="offset points",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Speedup relative to FreeTTES [×]  (log)", fontsize=11)
    ax.set_ylabel("MAE T_outlet vs. FreeTTES [K]", fontsize=11)
    ax.set_title(
        "Model variants: Accuracy vs. computational speed\n"
        "60 h scenario: Charging / Idle / Discharging  |  "
        f"V={V_TANK/1000:.0f} m³  R={R_INNER} m  H={H_WS} m",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.3)

    # Legend for colors (solver/scheme) and markers (N)
    legend_handles = [
        Patch(color=_COLORS[("explicit", "upwind")], label="Explicit – Upwind"),
        Patch(color=_COLORS[("explicit", "tvd")],    label="Explicit – TVD"),
        Patch(color=_COLORS[("implicit", "upwind")], label="Implicit – Upwind"),
        Patch(color=_COLORS[("implicit", "tvd")],    label="Implicit – TVD"),
    ]
    # Marker legend for N values
    for n_nodes, m in _MARKERS.items():
        lbl = "0D (N=1)" if n_nodes == 1 else f"N={n_nodes}"
        legend_handles.append(
            Line2D([0], [0], marker=m, color="gray", linestyle="None",
                   markersize=8, label=lbl)
        )
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left",
              framealpha=0.9, ncol=2)

    # Annotate target range for co-simulation
    ax.axhspan(0, 2.0, alpha=0.07, color="green")
    ax.text(1.5, 0.15, "Target precision\nco-simulation", fontsize=8,
            color="darkgreen", alpha=0.9, va="bottom")

    fig.tight_layout()
    out_path = out_dir / "variants_scatter.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_timeseries_selection(all_results: list[dict], reference: dict | None,
                              out_dir: Path) -> None:
    """
    T_outlet time series for selected variants + FreeTTES.

    Shows: 0D, N=10 Impl, N=50 Impl, N=200 Impl + FreeTTES (if available).
    """
    SELECT_LABELS = {
        "0D (fully mixed)",
        "1D N=10 Upw-Impl",
        "1D N=50 Upw-Impl",
        "1D N=200 Upw-Impl",
        "1D N=50 TVD-Expl",
        "1D N=50 Upw-Impl+HS",
    }
    subset = [r for r in all_results if r["label"] in SELECT_LABELS]

    cmap    = plt.get_cmap("tab10")
    colors  = {r["label"]: cmap(i) for i, r in enumerate(subset)}

    # Line styles for the 3 storage heights
    _H_STYLES = [
        ("T_top_series", "--",  "top",    Z_UPPER_DIFF),
        ("T_mid_series", "-.",  "middle", H_WS / 2),
        ("T_bot_series", ":",   "bottom", Z_LOWER_DIFF),
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    _phase_shading(ax)

    if reference:
        times = reference["times"]
        T_ref = [t if not np.isnan(t) else np.nan for t in reference["T_outlet"]]
        ax.plot(times, T_ref, color="black", linewidth=2.2,
                label="FreeTTES T_outlet", zorder=6)
        # Storage temperatures FreeTTES dashed
        for key, ls, lbl, _ in _H_STYLES:
            if key in reference:
                ax.plot(reference["times"], reference[key],
                        color="black", linewidth=0.9, linestyle=ls,
                        alpha=0.6, label=f"FreeTTES T_{lbl}")

    for r in subset:
        col = colors[r["label"]]
        # Outlet temperature: solid line
        T = [t if not np.isnan(t) else np.nan for t in r["T_outlet"]]
        ax.plot(r["times"], T, linewidth=1.6, label=r["label"],
                color=col, linestyle="-")
        # Storage temperatures: dashed, same color
        for key, ls, _, _ in _H_STYLES:
            if key in r:
                ax.plot(r["times"], r[key],
                        color=col, linewidth=0.8, linestyle=ls,
                        alpha=0.55, label="_nolegend_")
        # Headspace temperature (only when headspace=True)
        hs_series = r.get("T_headspace_series", [])
        if hs_series and hs_series[0] is not None:
            ax.plot(r["times"], hs_series,
                    color=col, linewidth=1.0, linestyle=(0, (3, 1, 1, 1, 1, 1)),
                    alpha=0.7, label="_nolegend_")

    ax.set_xlabel("Time [h]", fontsize=11)
    ax.set_ylabel("Temperature [°C]", fontsize=11)
    ax.set_xlim(0, N_HOURS - 1)
    ax.set_title(
        "Outlet and storage temperatures – selected variants vs. FreeTTES\n"
        f"dashed: T_storage (top ≈ {Z_UPPER_DIFF:.1f} m  |  "
        f"middle ≈ {H_WS/2:.1f} m  |  bottom ≈ {Z_LOWER_DIFF:.1f} m)",
        fontsize=9,
    )
    # Phase labels
    y0 = ax.get_ylim()[0]
    ax.text(12, y0 + 1.5, "Charging",     ha="center", fontsize=9,
            color="steelblue", alpha=0.9)
    ax.text(30, y0 + 1.5, "Idle",         ha="center", fontsize=9,
            color="goldenrod", alpha=0.9)
    ax.text(48, y0 + 1.5, "Discharging",  ha="center", fontsize=9,
            color="tomato", alpha=0.9)

    # Legend: add line style explanation
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="gray", linewidth=1.4, linestyle="-",
               label="── T_outlet (model)"),
        Line2D([0], [0], color="gray", linewidth=0.9, linestyle="--",
               label="-- T_storage top"),
        Line2D([0], [0], color="gray", linewidth=0.9, linestyle="-.",
               label="-·- T_storage middle"),
        Line2D([0], [0], color="gray", linewidth=0.9, linestyle=":",
               label="··· T_storage bottom"),
        Line2D([0], [0], color="gray", linewidth=1.0,
               linestyle=(0, (3, 1, 1, 1, 1, 1)),
               label="─·─·─ T_headspace (HS)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right",
              framealpha=0.92, ncol=2)
    fig.tight_layout()

    out_path = out_dir / "variants_timeseries.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_mae_bar(all_results: list[dict], reference: dict | None,
                 out_dir: Path) -> None:
    """MAE bar chart of all variants, sorted by MAE."""
    if not reference:
        return

    data = []
    for r in all_results:
        m   = compute_metrics(r, reference)
        mae = m["mae"]
        if not np.isnan(mae):
            data.append((r["label"], mae, r["solver"], r["scheme"]))

    data.sort(key=lambda x: x[1])

    labels  = [d[0] for d in data]
    maes    = [d[1] for d in data]
    colors  = [_COLORS.get((d[2], d[3]), "gray") for d in data]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(labels, maes, color=colors, edgecolor="white", height=0.65)
    ax.bar_label(bars, fmt="%.2f K", fontsize=8, padding=4)
    ax.set_xlabel("MAE T_outlet vs. FreeTTES [K]", fontsize=11)
    ax.set_title("Accuracy comparison: MAE T_outlet relative to FreeTTES reference",
                 fontsize=10)
    ax.axvline(2.0, color="darkgreen", linestyle="--", linewidth=1.2,
               label="Target precision 2 K")
    ax.grid(True, axis="x", alpha=0.3)
    legend_handles = [
        Patch(color=_COLORS[("explicit", "upwind")], label="Explicit – Upwind"),
        Patch(color=_COLORS[("explicit", "tvd")],    label="Explicit – TVD"),
        Patch(color=_COLORS[("implicit", "upwind")], label="Implicit – Upwind"),
        Patch(color=_COLORS[("implicit", "tvd")],    label="Implicit – TVD"),
        Line2D([0], [0], color="darkgreen", linestyle="--", label="Target precision 2 K"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")
    fig.tight_layout()
    out_path = out_dir / "variants_mae_bar.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── CSV and JSON ───────────────────────────────────────────────────────────────

def save_table_csv(all_results: list[dict], reference: dict | None,
                   out_dir: Path) -> None:
    """Saves results table as CSV."""
    rows = []
    for r in all_results:
        metrics = compute_metrics(r, reference) if reference else {}
        ref_ms  = reference["ms_per_step"] if reference else None
        speedup = (ref_ms / r["ms_per_step"]) if ref_ms else None
        rows.append({
            "label":        r["label"],
            "n_nodes":      r["n_nodes"] or 1,
            "solver":       r["solver"],
            "scheme":       r["scheme"],
            "ms_per_step":  round(r["ms_per_step"], 4),
            "speedup_vs_ft": round(speedup, 0) if speedup else "",
            "mae_K":        round(metrics.get("mae", float("nan")), 4)
                            if metrics else "",
            "rmse_K":       round(metrics.get("rmse", float("nan")), 4)
                            if metrics else "",
            "max_err_K":    round(metrics.get("max_err", float("nan")), 4)
                            if metrics else "",
            "bias_K":       round(metrics.get("bias", float("nan")), 4)
                            if metrics else "",
            "cfl_stable":   "yes" if r["solver"] == "implicit"
                            or r.get("n_sub", 1) == 1 else "no",
            "n_substeps":   r.get("n_sub", 1),
        })

    if reference:
        rows.append({
            "label": "FreeTTES",
            "n_nodes": "—",
            "solver": "lagrange-explicit",
            "scheme": "—",
            "ms_per_step": round(reference["ms_per_step"], 1),
            "speedup_vs_ft": 1,
            "mae_K": 0.0, "rmse_K": 0.0, "max_err_K": 0.0, "bias_K": 0.0,
            "cfl_stable": "—", "n_substeps": 1,
        })

    out_path = out_dir / "variants_table.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out_path}")


def save_named_run(
    all_results: list[dict],
    reference: dict | None,
    name: str,
    include_freetttes: bool,
) -> None:
    """
    Save run results to results/runs/{name}/ in the format expected by compare_runs.py.

    Writes three files:
      - comparison_summary.json  — per-model scalar metrics
      - comparison_results.csv   — hourly T_outlet and E_nutz time series
      - run_metadata.json        — git branch/commit, timestamp, FreeTTES flag
    """
    import subprocess
    from datetime import datetime

    runs_dir = RESULTS_DIR.parent / "runs" / name
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ── run_metadata.json ─────────────────────────────────────────────────────
    def _git(cmd: list[str]) -> str:
        try:
            out = subprocess.check_output(
                ["git", "-C", str(PROJECT_DIR)] + cmd,
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            return out if out else "?"
        except Exception:
            return "?"

    # symbolic-ref works even on repos with no commits yet
    branch = _git(["symbolic-ref", "--short", "HEAD"])
    commit = _git(["rev-parse", "--short", "HEAD"])

    metadata = {
        "name":              name,
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "git_branch":        branch,
        "git_commit":        commit,
        "include_freetttes":  include_freetttes,
        "n_variants":         len(all_results),
        "scenario":           f"{N_HOURS}h: Charging 0-{T_END_CHARGE}h / "
                              f"Idle {T_END_CHARGE}-{T_END_IDLE}h / "
                              f"Discharging {T_END_IDLE}-{N_HOURS}h",
        "T_end_charge_h":     T_END_CHARGE,
        "T_end_idle_h":       T_END_IDLE,
        "T_total_h":          N_HOURS,
    }
    meta_path = runs_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ── comparison_summary.json ───────────────────────────────────────────────
    def _scalar(result: dict) -> dict:
        E   = result["E_nutz"]
        T_o = result["T_outlet"]
        mae = None
        if reference is not None:
            m = compute_metrics(result, reference)
            mae = round(m["mae"], 3) if not np.isnan(m["mae"]) else None
        return {
            "avg_ms_per_step":        round(result["ms_per_step"], 4),
            "E_nutz_start_MWh":       round(float(E[0]), 3),
            "E_nutz_after_charge_MWh":round(float(E[T_END_CHARGE - 1]), 3),
            "E_nutz_end_MWh":         round(float(E[-1]), 3),
            "E_loss_idle_MWh":        round(
                float(E[T_END_CHARGE - 1]) - float(E[T_END_IDLE - 1]), 3
            ),
            "T_outlet_charge_end_C":  (
                round(float(T_o[T_END_CHARGE - 1]), 2)
                if not np.isnan(T_o[T_END_CHARGE - 1]) else None
            ),
            "T_outlet_disch_end_C":   (
                round(float(T_o[-1]), 2)
                if not np.isnan(T_o[-1]) else None
            ),
            "MAE_vs_FreeTTES_K":      mae,
        }

    summary: dict = {}
    for r in all_results:
        summary[r["label"]] = _scalar(r)
    if reference is not None:
        summary["FreeTTES (reference)"] = _scalar(reference)
    summary["tank"] = {
        "V_m3": round(V_TANK, 0), "R_m": R_INNER,
        "H_m": H_WS, "U_loss_W_m2K": U_WALL,
    }

    summary_path = runs_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── comparison_results.csv ────────────────────────────────────────────────
    def _col(label: str) -> str:
        """Sanitise label for use as a CSV column suffix."""
        return label.replace(" ", "_").replace("=", "").replace("+", "p")

    all_series = list(all_results)
    if reference is not None:
        all_series.append(reference)

    times = all_series[0]["times"]
    rows = []
    for i, t in enumerate(times):
        row: dict = {"t_h": t}
        for r in all_series:
            col = _col(r["label"])
            row[f"T_outlet_{col}"] = (
                "nan" if np.isnan(r["T_outlet"][i]) else round(float(r["T_outlet"][i]), 4)
            )
            row[f"E_nutz_MWh_{col}"] = round(float(r["E_nutz"][i]), 4)
        rows.append(row)

    csv_path = runs_dir / "comparison_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Named run '{name}' saved to: {runs_dir}")
    print(f"    {meta_path.name}, {summary_path.name}, {csv_path.name}")


def save_summary_json(all_results: list[dict], reference: dict | None,
                      out_dir: Path) -> None:
    """Saves compact summary as JSON."""
    entries = []
    for r in all_results:
        metrics = compute_metrics(r, reference) if reference else {}
        ref_ms  = reference["ms_per_step"] if reference else None
        entries.append({
            "label":       r["label"],
            "n_nodes":     r["n_nodes"],
            "solver":      r["solver"],
            "scheme":      r["scheme"],
            "ms_per_step": round(r["ms_per_step"], 4),
            "speedup":     round(ref_ms / r["ms_per_step"], 0) if ref_ms else None,
            "mae_K":       round(metrics.get("mae", float("nan")), 3) if metrics else None,
            "rmse_K":      round(metrics.get("rmse", float("nan")), 3) if metrics else None,
        })

    summary = {
        "scenario":     "60h: Charging 0-23h / Idle 24-35h / Discharging 36-59h",
        "tank":         {"V_m3": round(V_TANK, 0), "R_m": R_INNER, "H_m": H_WS,
                          "U_loss": U_WALL},
        "flow_m3h":     FLOW_M3H,
        "dt_s":         DT_S,
        "variants":     entries,
        "reference":    {
            "label":      "FreeTTES",
            "ms_per_step": round(reference["ms_per_step"], 1),
        } if reference else None,
    }

    out_path = out_dir / "variants_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")


# ── Recommendation for co-simulation ──────────────────────────────────────────

def print_and_save_recommendation(all_results: list[dict],
                                   reference: dict | None,
                                   out_dir: Path) -> None:
    """
    Prints a reasoned recommendation for co-simulation with a district heating
    network model (implicit Euler, dt=3600 s).
    """
    ref_ms = reference["ms_per_step"] if reference else None

    # Best implicit variant by MAE
    impl_results = [r for r in all_results if r["solver"] == "implicit"]
    if reference and impl_results:
        best_impl = min(
            impl_results,
            key=lambda r: compute_metrics(r, reference).get("mae", float("inf")),
        )
        best_metrics = compute_metrics(best_impl, reference)
    else:
        best_impl = impl_results[0] if impl_results else None
        best_metrics = {}

    # 0D reference
    zerod = next((r for r in all_results if r["n_nodes"] == 1), None)
    zerod_metrics = compute_metrics(zerod, reference) if (zerod and reference) else {}

    lines = [
        "=" * 70,
        "RECOMMENDATION: Model choice for co-simulation with district heating network",
        "=" * 70,
        "",
        "Context",
        "-------",
        "  The district heating network simulation model (e.g. pandapipes) uses implicit",
        "  Euler with time steps of typically dt = 3600 s. The storage must be solved",
        "  within a few milliseconds at each coupling step so that the overall",
        "  simulation is not dominated by the storage.",
        "",
        "Stability requirement",
        "---------------------",
        "  Explicit solvers require sub-stepping for fine grids (CFL condition).",
        "  For dt = 3600 s and N >= 50, multiple substeps are needed → overhead.",
        "  Implicit solvers are unconditionally stable → no sub-stepping, no",
        "  risk with variable network time steps.",
        "  RECOMMENDATION: Implicit Euler for coupling.",
    ]

    lines += [
        "",
        "0D vs. 1D stratification model",
        "-------------------------------",
    ]

    if zerod_metrics:
        mae_0d = zerod_metrics.get("mae", float("nan"))
        lines.append(
            f"  The fully mixed 0D model (N=1) achieves a MAE of"
        )
        lines.append(
            f"  {mae_0d:.2f} K relative to FreeTTES. It systematically underestimates"
        )
        lines.append(
            f"  the outlet temperature during discharging (thermocline is not"
        )
        lines.append(
            f"  preserved) and overestimates it during charging. For network simulations"
        )
        lines.append(
            f"  where supply and return temperatures directly affect heat pump load"
        )
        lines.append(
            f"  and network hydraulics, this deviation is significant."
        )
    else:
        lines.append(
            "  (No FreeTTES run – quantitative comparison not possible.)"
        )

    lines += [
        "",
        "Recommended configuration",
        "-------------------------",
    ]

    if best_impl and best_metrics:
        mae_best = best_metrics.get("mae", float("nan"))
        ms_best  = best_impl["ms_per_step"]
        su_best  = f"{ref_ms/ms_best:.0f}×" if ref_ms else "n/a"
        lines += [
            f"  Model:     {best_impl['label']}",
            f"  MAE:       {mae_best:.2f} K vs. FreeTTES",
            f"  Speed:     {ms_best:.3f} ms/step  ({su_best} faster than FreeTTES)",
            f"  Stability: unconditionally stable, no sub-stepping",
            "",
            "  This model offers the best trade-off between accuracy,",
            "  stability and computational cost for co-simulation.",
        ]
    elif best_impl:
        lines += [f"  {best_impl['label']} – implicit, unconditionally stable."]

    lines += [
        "",
        "Notes",
        "------------------------------",
        "  - The difference between 0D and 1D (N>=10 implicit) is quantifiable",
        "    and documented in the table variants_table.csv.",
        "  - TVD scheme provides little additional benefit over implicit Euler, as the",
        "    time integration error dominates at dt=3600 s.",
        "  - For pure planning studies (annual simulation, no peak load resolution)",
        "    0D may suffice; for dynamic network calculations,",
        "    1D N=10-50 implicit is recommended.",
        "=" * 70,
    ]

    text = "\n".join(lines)
    print("\n" + text)

    out_path = out_dir / "variants_recommendation.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  Saved: {out_path}")


# ── Main program ───────────────────────────────────────────────────────────────

def load_freetttes_from_csv(csv_path: Path, ms_per_step: float = 424.4) -> dict:
    """Loads FreeTTES reference data from an existing comparison_results.csv."""
    times, T_outlet, E_nutz = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(int(row["t_h"]))
            T_outlet.append(float(row["T_outlet_FreeTTES"]))
            E_nutz.append(float(row["E_nutz_MWh_FreeTTES"]))
    n = len(times)
    total_time = ms_per_step * n / 1000.0
    print(f"  [FreeTTES] Loading {n} time steps from {csv_path.name} "
          f"(ms/step={ms_per_step:.1f} from previous run)")
    return {
        "label":      "FreeTTES",
        "n_nodes":    None,
        "solver":     "lagrange-explicit",
        "scheme":     "—",
        "times":      times,
        "T_outlet":   T_outlet,
        "E_nutz":     E_nutz,
        "step_times": [ms_per_step / 1000.0] * n,
        "total_time": total_time,
        "ms_per_step": ms_per_step,
        "T_top_series": [float("nan")] * n,
        "T_mid_series": [float("nan")] * n,
        "T_bot_series": [float("nan")] * n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model variant comparison 0D–1D vs. FreeTTES"
    )
    parser.add_argument(
        "--no-freetttes", action="store_true",
        help="Skip FreeTTES run (own model only)"
    )
    parser.add_argument(
        "--freetttes-csv", type=Path, default=None,
        metavar="CSV",
        help="Use existing comparison_results.csv with T_outlet_FreeTTES column "
             "instead of re-running FreeTTES"
    )
    parser.add_argument(
        "--freetttes-ms", type=float, default=424.4,
        metavar="MS",
        help="ms/step value for FreeTTES from previous run (default: 424.4)"
    )
    parser.add_argument(
        "--freetttes-src", type=Path, default=None, metavar="PATH",
        help="Path to FreeTTES/src directory (overrides the FREETTTES_SRC constant)"
    )
    parser.add_argument(
        "--name", type=str, default=None, metavar="NAME",
        help="Save run to results/runs/NAME/ for use with compare_runs.py"
    )
    parser.add_argument(
        "--solver", choices=["explicit", "implicit", "all"], default="all",
        help="Run only explicit, only implicit, or all variants (default: all)"
    )
    args = parser.parse_args()

    # Apply --freetttes-src before importing FreeTTES
    if args.freetttes_src is not None:
        import importlib
        freetttes_path = args.freetttes_src.resolve()
        if str(freetttes_path) not in sys.path:
            sys.path.insert(0, str(freetttes_path))
        global FREETTTES_AVAILABLE
        try:
            globals()["freetttes"] = importlib.import_module("FreeTTES_model")
            FREETTTES_AVAILABLE = True
        except ImportError as exc:
            print(f"  Warning: FreeTTES not found at {freetttes_path}: {exc}")

    use_freetttes = not args.no_freetttes

    # Filter variants by solver if requested
    if args.solver == "all":
        active_variants = MODEL_VARIANTS
    else:
        active_variants = [
            v for v in MODEL_VARIANTS if v[2] == args.solver
        ]
        if not active_variants:
            sys.exit(f"No variants match --solver {args.solver}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print_separator("Model variant comparison")
    print(f"  Scenario: {N_HOURS} h  |  V={V_TANK/1000:.0f} m³  "
          f"R={R_INNER} m  H={H_WS} m  U={U_WALL} W/(m²·K)")
    print(f"  Flow rate: {FLOW_M3H} m³/h  "
          f"(m_c={m_charge:.2f} kg/s, m_d={m_discharge:.2f} kg/s)")
    print(f"  Solver filter: {args.solver}  |  Variants: {len(active_variants)}")
    if args.name:
        print(f"  Named run: '{args.name}' → results/runs/{args.name}/")
    print_separator()

    # ── FreeTTES ──────────────────────────────────────────────────────────────
    reference = None
    if args.freetttes_csv is not None:
        print_separator("FreeTTES (from CSV)")
        reference = load_freetttes_from_csv(args.freetttes_csv, args.freetttes_ms)
    elif use_freetttes and FREETTTES_AVAILABLE:
        cfg_ok, cfg_msg = _validate_freetttes_config()
        if not cfg_ok:
            print_separator("FreeTTES configuration mismatch")
            print(cfg_msg)
            print("  FreeTTES run skipped – continuing without reference.")
        else:
            try:
                reference = run_freetttes()
            except Exception as exc:
                print_separator("FreeTTES run failed")
                print(f"  {type(exc).__name__}: {exc}")
                print("  Continuing without reference.")
    elif use_freetttes:
        print("  FreeTTES not available – continuing without reference.")

    # ── 0D model (fully mixed) ────────────────────────────────────────────────
    all_results = [run_zerod()]

    # ── 1D model variants ─────────────────────────────────────────────────────
    for label, n_nodes, solver, scheme, hs in active_variants:
        result = run_variant(label, n_nodes, solver, scheme, headspace=hs)
        all_results.append(result)

    # ── Output ────────────────────────────────────────────────────────────────
    print_separator("Evaluation")
    print_results_table(all_results, reference)

    print_separator("Plots and files")
    plot_scatter(all_results, reference, RESULTS_DIR)
    plot_timeseries_selection(all_results, reference, RESULTS_DIR)
    plot_mae_bar(all_results, reference, RESULTS_DIR)
    save_table_csv(all_results, reference, RESULTS_DIR)
    save_summary_json(all_results, reference, RESULTS_DIR)

    print_and_save_recommendation(all_results, reference, RESULTS_DIR)

    if args.name:
        print_separator("Named run")
        save_named_run(all_results, reference, args.name, use_freetttes)

    print_separator("Done")
    print(f"  All results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
