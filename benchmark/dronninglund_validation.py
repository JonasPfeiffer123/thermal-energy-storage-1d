#!/usr/bin/env python3
"""
Validation of the ThermalStorage1D model against measurement data from the
Dronninglund Pit Thermal Energy Storage (PTES), Denmark – year 2014.

Dataset:   Sifnaios et al. 2023, Solar Energy 251, 68–76
           https://github.com/PitStorages/DronninglundData
Year:      2014; simulation from May (first available FR-direction data)

Storage geometry (truncated pyramid):
  Lid:     90.4 x 90.4 m  (slope 1:2, each 0.5-m layer: –2 m side length)
  Bottom:  26.4 x 26.4 m  (90.4 – 32 × 2 = 26.4 m)
  Depth:   H = 16 m, volume ~ 60 030 m³

Three diffusors at correct heights (Port API):
  Port "top":    z = 15.3 m  (FR_top_c × F_top / 3600)
  Port "mid":    z = 10.9 m  (FR_mid_c × F_mid / 3600)
  Port "bottom": z =  0.4 m  (mass balance: –(f_top + f_mid))
  Convention: m_dot > 0 = inflow INTO storage, m_dot < 0 = outflow

Loss parameters:
  U_LID  = 0.167 W/(m²K)  [lid, measured 2014 (Sifnaios notebook)]
  U_WALL = 0.35  W/(m²K)  [walls/bottom: lambda_soil=0.4 W/(mK), eff. thickness ~1 m]
  T_AMB  = 8.0 °C         [Danish annual mean]

Note Jan–Apr:
  FR_top_c/FR_mid_c and all energy-balance columns are NaN in Jan–Apr
  (no solar production → energy balance cannot be determined). The simulation
  therefore starts on the first day with available FR data (approx. 1 May),
  initialised from the measured temperature profile at that point in time.

Data acquisition
----------------
The measurement data is NOT included in this repository. Download it from:
    https://github.com/PitStorages/DronninglundData

Expected file path (relative to repo root):
    data/DronninglundData/data/Dronninglund_treated_data_and_flow_rates_2014.csv

Clone the data repository into the data/ folder:
    git clone https://github.com/PitStorages/DronninglundData data/DronninglundData
"""

import os
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = (
    ROOT
    / "data/DronninglundData/data"
    / "Dronninglund_treated_data_and_flow_rates_2014.csv"
)
OUT_DIR = Path(os.environ.get("TES_VALIDATION_OUT_DIR", ROOT / "benchmark/results/dronninglund"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from thermal_energy_storage_model import (
    StorageConfig,
    StorageInputs,
    Port,
    ThermalStorage1D,
    TruncatedPyramidGeometry,
    SplitAmbientLoss,
    WaterProperties,
)

# ---------------------------------------------------------------------------
# Storage parameters
# ---------------------------------------------------------------------------
H = 16.0        # m   total depth
A_TOP = 90.4    # m   lid side length    (source: Sifnaios 2023 notebook)
A_BOT = 26.4    # m   bottom side length (90.4 – 32 layers × 2 m/layer)
# Volume of truncated pyramid: V = H/3 × (a_t² + a_t×a_b + a_b²)
V_PTES = H / 3.0 * (A_TOP**2 + A_TOP * A_BOT + A_BOT**2)   # ≈ 60 030 m³

N = 50          # number of nodes
DT = 3600.0     # s  timestep (1 h)

# Diffusor heights above bottom [m]
H_DIFF_TOP = 15.3
H_DIFF_MID = 10.9
H_DIFF_BOT = 0.4

# Heat loss
U_LID  = 0.167  # W/(m²K)  lid U-value measured 2014 (Sifnaios notebook)
U_WALL = 0.35   # W/(m²K)  walls/bottom (lambda_soil=0.4, eff. thickness ~1 m)
T_AMB_WALL = 8.0  # °C  ground temperature (walls/bottom), nearly constant
# Danish air temperature: annual mean ~8 °C, amplitude ~8.5 K
# T_air(doy) = 8.0 – 8.5 · cos(2π · (doy – 15) / 365)
T_AMB_MEAN  = 8.0   # °C  annual mean air temperature
T_AMB_AMP   = 8.5   # K   amplitude (min ~–0.5 °C Jan, max ~16.5 °C Jul)
T_AMB_DMIN  = 15    # day of year with minimum temperature (mid-January)


def t_air_seasonal(doy: int) -> float:
    """Sinusoidal annual outdoor air temperature curve for Denmark [°C]."""
    return T_AMB_MEAN - T_AMB_AMP * np.cos(2 * np.pi * (doy - T_AMB_DMIN) / 365)


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_storage() -> tuple[ThermalStorage1D, SplitAmbientLoss]:
    geometry = TruncatedPyramidGeometry(A_BOT, A_BOT, A_TOP, A_TOP, H)
    loss = SplitAmbientLoss(U_lid=U_LID, U_wall=U_WALL, T_ambient=T_AMB_WALL)
    fluid = WaterProperties()
    config = StorageConfig(
        volume=V_PTES,
        height=H,
        n_nodes=N,
        geometry=geometry,
        loss_model=loss,
        fluid=fluid,
        solver="implicit",
        advection_scheme="upwind",
        buoyancy=True,
    )
    return ThermalStorage1D(config), loss


# ---------------------------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------------------------
def load_and_resample(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.tz_localize(None)   # remove timezone info
    return df.resample("1h").mean()


# ---------------------------------------------------------------------------
# Interpolate initialisation profile from measurement
# ---------------------------------------------------------------------------
def interpolate_T_profile(row: pd.Series, n_nodes: int) -> np.ndarray | None:
    """
    Interpolate measured temperature profile (both sensor strings) onto
    the model grid.  Node k=0 = top (z = H – dz/2), k=N–1 = bottom.
    """
    all_h, all_T = [], []
    for h in np.arange(0.5, 16.0, 1.0):   # String A
        v = row.get(f"T_{h:04.1f}", np.nan)
        if np.isfinite(v):
            all_h.append(h); all_T.append(v)
    for h in np.arange(1.0, 17.0, 1.0):   # String B
        v = row.get(f"T_{h:04.1f}", np.nan)
        if np.isfinite(v):
            all_h.append(h); all_T.append(v)
    if len(all_h) < 2:
        return None
    all_h = np.array(all_h); all_T = np.array(all_T)
    idx = np.argsort(all_h)
    all_h, all_T = all_h[idx], all_T[idx]

    dz = H / n_nodes
    node_heights = np.array([H - (k + 0.5) * dz for k in range(n_nodes)])
    return np.interp(node_heights, all_h, all_T)


# ---------------------------------------------------------------------------
# Flow mapping: 3 diffusors as real ports at their correct heights
# ---------------------------------------------------------------------------
def make_ports(row: pd.Series) -> list:
    """
    Create a port list for all three diffusors of the Dronninglund PTES.

    Convention (Sifnaios 2023 notebook):
      FR_top_c, FR_mid_c: +1 = inflow INTO storage, –1 = outflow
      F_top, F_mid: magnitude in kg/h  (flow meter measurement)
      F_bot is forced from mass balance: f_bot = –(f_top_s + f_mid_s)

    Each diffusor receives a port at its correct height:
      z_top = 15.3 m, z_mid = 10.9 m, z_bot = 0.4 m

    Returns an empty list if FR data are missing (standby).
    """
    def safe(col, default=np.nan):
        v = row.get(col, np.nan)
        return float(v) if np.isfinite(v) else default

    FR_top = safe("FR_top_c")
    FR_mid = safe("FR_mid_c")
    # FR data only available from May; NaN = operation cannot be derived
    if not (np.isfinite(FR_top) or np.isfinite(FR_mid)):
        return []

    F_top = safe("F_top", 0.0) / 3600.0   # kg/s
    F_mid = safe("F_mid", 0.0) / 3600.0
    T_top = safe("T_top", 60.0)
    T_mid = safe("T_mid", 60.0)
    T_bot = safe("T_bot", 10.0)

    f_top_s = (FR_top if np.isfinite(FR_top) else 0.0) * F_top   # positive = inflow
    f_mid_s = (FR_mid if np.isfinite(FR_mid) else 0.0) * F_mid
    f_bot_s = -(f_top_s + f_mid_s)   # mass balance

    ports = []
    if abs(f_top_s) > 0.01:
        ports.append(Port(
            z=H_DIFF_TOP,
            m_dot=f_top_s,
            T_in=T_top if f_top_s > 0 else 0.0,
            label="top",
        ))
    if abs(f_mid_s) > 0.01:
        ports.append(Port(
            z=H_DIFF_MID,
            m_dot=f_mid_s,
            T_in=T_mid if f_mid_s > 0 else 0.0,
            label="mid",
        ))
    if abs(f_bot_s) > 0.01:
        ports.append(Port(
            z=H_DIFF_BOT,
            m_dot=f_bot_s,
            T_in=T_bot if f_bot_s > 0 else 0.0,
            label="bottom",
        ))
    return ports


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run():
    print("Loading Dronninglund dataset 2014 ...")
    df = load_and_resample(DATA_FILE)
    print(f"  {len(df)} hourly values  ({df.index[0].date()} to {df.index[-1].date()})")

    storage, loss = build_storage()
    print(f"  Storage volume: {V_PTES:.0f} m³  (geometry: {A_BOT}×{A_BOT} → {A_TOP}×{A_TOP} m, H={H} m)")

    # Initialisation: first row WITH FR data AND complete temperature profile.
    # Jan–Apr: FR_top_c/FR_mid_c are NaN → boundary conditions cannot be derived.
    # Simulation starts on the first day with FR data (approx. 1 May).
    state = None
    t0_idx = 0
    for i, (ts, row) in enumerate(df.iterrows()):
        # FR data available?
        fr_top = row.get("FR_top_c", np.nan)
        fr_mid = row.get("FR_mid_c", np.nan)
        if not (np.isfinite(fr_top) or np.isfinite(fr_mid)):
            continue
        # Complete temperature profile available?
        T_init = interpolate_T_profile(row, N)
        if T_init is None:
            continue
        state = storage.initialize(T_init)
        t0_idx = i
        print(f"  Initialisation from: {ts}  (T_top={T_init[0]:.1f}°C, T_bottom={T_init[-1]:.1f}°C)")
        break
    if state is None:
        raise RuntimeError("No valid initial temperature measurement with FR data found.")

    n = len(df)
    dz = H / N
    node_heights = np.array([H - (k + 0.5) * dz for k in range(N)])  # height above bottom [m]

    # Node indices for diffusor positions (node 0 = top)
    k_top_diff = max(0, min(N - 1, int((H - H_DIFF_TOP) / dz)))
    k_mid_diff = max(0, min(N - 1, int((H - H_DIFF_MID) / dz)))
    k_bot_diff = max(0, min(N - 1, int((H - H_DIFF_BOT) / dz)))

    T_sim_top = np.full(n, np.nan)
    T_sim_mid = np.full(n, np.nan)
    T_sim_bot = np.full(n, np.nan)
    T_profiles_snap = {}   # {month-str: (idx, T_array)}

    print("Starting simulation ...")
    for i, (ts, row) in enumerate(df.iterrows()):
        if i < t0_idx:
            continue

        T_sim_top[i] = state.temperatures[k_top_diff]
        T_sim_mid[i] = state.temperatures[k_mid_diff]
        T_sim_bot[i] = state.temperatures[k_bot_diff]

        if ts.day == 1 and ts.hour == 0:
            T_profiles_snap[ts.strftime("%Y-%m")] = (i, state.temperatures.copy())

        # Update seasonal air temperature for lid node
        loss.T_ambient_lid = t_air_seasonal(ts.timetuple().tm_yday)

        ports = make_ports(row)
        inputs = StorageInputs(ports=ports)
        outputs = storage.step(state, dt=DT, inputs=inputs)
        state = outputs.state

        if i % 1000 == 0:
            f_net = sum(p.m_dot for p in ports if p.m_dot > 0)
            print(f"  {ts.date()}  ports={len(ports)}  m_in={f_net:.1f} kg/s"
                  f"  T_top={state.temperatures[0]:.1f}°C")

    print("Simulation complete.")

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    T_meas_top = df["T_top"].values
    T_meas_mid = df["T_mid"].values
    T_meas_bot = df["T_bot"].values

    def mae(meas, sim):
        valid = np.isfinite(meas) & np.isfinite(sim)
        return np.mean(np.abs(meas[valid] - sim[valid])) if valid.any() else np.nan

    mae_top = mae(T_meas_top, T_sim_top)
    mae_mid = mae(T_meas_mid, T_sim_mid)
    mae_bot = mae(T_meas_bot, T_sim_bot)
    mae_all = mae(
        np.concatenate([T_meas_top, T_meas_mid, T_meas_bot]),
        np.concatenate([T_sim_top,  T_sim_mid,  T_sim_bot]),
    )
    print(f"\nMAE diffusor temperatures (measurement vs. model N={N}, implicit):")
    print(f"  Top    ({H_DIFF_TOP} m): {mae_top:.2f} K")
    print(f"  Middle ({H_DIFF_MID} m): {mae_mid:.2f} K")
    print(f"  Bottom  ({H_DIFF_BOT} m):  {mae_bot:.2f} K")
    print(f"  Total:           {mae_all:.2f} K")

    # -----------------------------------------------------------------------
    # Plot 1: Diffusor temperature time series
    # -----------------------------------------------------------------------
    times = df.index
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    plot_rows = [
        (f"Top ({H_DIFF_TOP} m)",    T_meas_top, T_sim_top,  mae_top),
        (f"Middle ({H_DIFF_MID} m)", T_meas_mid, T_sim_mid, mae_mid),
        (f"Bottom ({H_DIFF_BOT} m)", T_meas_bot, T_sim_bot,  mae_bot),
    ]
    for ax, (label, meas, sim, e) in zip(axes, plot_rows):
        ax.plot(times, meas, color="black", lw=0.8, label="Measurement (Sifnaios et al., 2023)")
        ax.plot(times, sim, color="tab:red", lw=0.9, alpha=0.85,
                label=f"Model N={N} implicit  |  MAE = {e:.2f} K")
        ax.set_ylabel("T [°C]")
        ax.set_title(f"Diffusor {label}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Dronninglund PTES 2014 – Model comparison diffusor temperatures (3-port)"
        f"\n{A_BOT}×{A_BOT}→{A_TOP}×{A_TOP} m, H={H:.0f} m, N={N}, U_lid={U_LID} U_wall={U_WALL} W/(m²K), T_lid seasonal",
        fontsize=10,
    )
    plt.tight_layout()
    for suffix in (".svg", ".pdf"):
        fig.savefig((OUT_DIR / "dronninglund_timeseries").with_suffix(suffix))
    print(f"Saved: {OUT_DIR}/dronninglund_timeseries.svg/.pdf")

    # -----------------------------------------------------------------------
    # Plot 1b: Abstract version – top + bottom only, compact height
    # -----------------------------------------------------------------------
    MONTHS_EN = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

    def _eng_month(x, pos):
        return MONTHS_EN.get(mdates.num2date(x).month, "")

    from matplotlib.ticker import FuncFormatter

    FS_TITLE  = 11   # subplot title
    FS_LABEL  = 10   # axis label
    FS_LEGEND =  9   # legend
    FS_TICK   =  9   # tick labels

    fig_abs, axes_abs = plt.subplots(2, 1, figsize=(12, 5.0), sharex=True)
    plot_rows_abs = [
        (f"Top diffusor ({H_DIFF_TOP}\u2009m)",    T_meas_top, T_sim_top,  mae_top),
        (f"Bottom diffusor ({H_DIFF_BOT}\u2009m)", T_meas_bot, T_sim_bot,  mae_bot),
    ]
    for ax, (label, meas, sim, e) in zip(axes_abs, plot_rows_abs):
        ax.plot(times, meas, color="black", lw=0.8,
                label="Measurement (Sifnaios et al., 2023)")
        ax.plot(times, sim, color="tab:red", lw=1.0, alpha=0.88,
                label=f"Model N\u202f=\u202f{N} implicit  |  MAE\u202f=\u202f{e:.2f}\u2009K")
        ax.set_ylabel("T [°C]", fontsize=FS_LABEL)
        ax.set_title(label, fontsize=FS_TITLE)
        ax.legend(loc="upper left", fontsize=FS_LEGEND)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(_eng_month))
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for suffix in (".svg", ".pdf"):
        fig_abs.savefig((OUT_DIR / "dronninglund_timeseries_abstract").with_suffix(suffix),
                        bbox_inches="tight")
    print(f"Saved: {OUT_DIR}/dronninglund_timeseries_abstract.svg/.pdf")

    # -----------------------------------------------------------------------
    # Plot 2: Monthly temperature profiles
    # -----------------------------------------------------------------------
    months = sorted(T_profiles_snap.keys())
    n_months = min(len(months), 12)
    if n_months > 0:
        fig2, axes2 = plt.subplots(3, 4, figsize=(14, 10), sharey=True)
        axes2_flat = axes2.flatten()
        heights_A = np.arange(0.5, 16.0, 1.0)
        heights_B = np.arange(1.0, 17.0, 1.0)

        for ax_idx, month in enumerate(months[:12]):
            ax = axes2_flat[ax_idx]
            snap_i, T_snap = T_profiles_snap[month]
            row_snap = df.iloc[snap_i]

            vals_A = [row_snap.get(f"T_{h:04.1f}", np.nan) for h in heights_A]
            vals_B = [row_snap.get(f"T_{h:04.1f}", np.nan) for h in heights_B]
            ax.plot(vals_A, heights_A, "o", ms=3, color="black", label="Meas. A")
            ax.plot(vals_B, heights_B, "s", ms=3, color="gray",  label="Meas. B")
            ax.plot(T_snap, node_heights, "-", color="tab:red", lw=1.5, label="Model")

            for hd, ls in [(H_DIFF_TOP, "--"), (H_DIFF_MID, ":"), (H_DIFF_BOT, "-.")]:
                ax.axhline(hd, color="tab:blue", lw=0.5, ls=ls, alpha=0.5)

            ax.set_title(month, fontsize=9)
            ax.set_xlabel("T [°C]", fontsize=8)
            if ax_idx % 4 == 0:
                ax.set_ylabel("Height above bottom [m]", fontsize=8)
            ax.set_ylim(0, H)
            ax.set_xlim(0, 95)
            ax.grid(True, alpha=0.3)
            if ax_idx == 0:
                ax.legend(fontsize=6)

        for ax_idx in range(n_months, 12):
            axes2_flat[ax_idx].set_visible(False)

        fig2.suptitle(
            "Dronninglund PTES 2014 – Monthly temperature profiles\n"
            f"Black: Measurement  |  Red: Model N={N}  |  Blue: Diffusor positions",
            fontsize=10,
        )
        plt.tight_layout()
        for suffix in (".svg", ".pdf"):
            fig2.savefig((OUT_DIR / "dronninglund_profiles").with_suffix(suffix))
        print(f"Saved: {OUT_DIR}/dronninglund_profiles.svg/.pdf")

    # -----------------------------------------------------------------------
    # CSV export
    # -----------------------------------------------------------------------
    pd.DataFrame(
        {
            "T_sim_top_C":  T_sim_top,
            "T_meas_top_C": T_meas_top,
            "T_sim_mid_C":  T_sim_mid,
            "T_meas_mid_C": T_meas_mid,
            "T_sim_bot_C":  T_sim_bot,
            "T_meas_bot_C": T_meas_bot,
        },
        index=times,
    ).to_csv(OUT_DIR / "dronninglund_comparison.csv")
    print(f"Saved: {OUT_DIR}/dronninglund_comparison.csv")

    plt.show()


if __name__ == "__main__":
    run()
