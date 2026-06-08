#!/usr/bin/env python3
"""
Validation of the ThermalStorage1D model against measurement data from the
Høje Taastrup Pit Thermal Energy Storage (PTES), Denmark – year 2024.

Dataset:   Sifnaios et al. 2025, Data in Brief
           https://github.com/PitStorages/HojeTaastrupData
Year:      2024 (complete, 10-minute resolution)

Storage geometry (Sifnaios et al. 2025, Section 4.2):
  Irregular shape (not a pure trapezoid), equivalent square side lengths:
  Lid:    sqrt(11108 m²) = 105.4 m
  Bottom: sqrt(864 m²)   =  29.4 m
  Depth:  H = 14.05 m (deepest point East), 12.0 m (shallowest point West)
  Volume: 70 880 m³ (paper), model: ~70 578 m³
  Slope 1:2 to 1:2.1

Three diffusors (Sifnaios et al. 2025, Section 4.2):
  Port "top":    z = 13.80 m (F_top)
  Port "mid":    z = 10.50 m (F_mid; at half storage volume)
  Port "bottom": z =  0.45 m (F_bot)
  Convention: m_dot > 0 = inflow INTO storage, m_dot < 0 = outflow
  F_mid is structurally outlet-only (discharge modes 2+3), hence always <= 0.

Inlet temperature approximation:
  If F_top > 0: T_in = T_top (diffusor pipe sensor)
  If F_bot > 0: T_in = T_bot (diffusor pipe sensor)
  F_mid is always <= 0 (no inflow via middle)

Note on T_top/T_mid/T_bot:
  All three are diffusor pipe sensors (Sifnaios 2025 Table 1: "temperature in
  the [top/middle/bottom] diffuser"). During standby they adopt pipe temperature.
  Lance sensors (A_xx) are used as the validation reference.

Loss parameters (Sifnaios et al. 2025):
  U_LID  = 0.151 W/(m²K)  [calculated from lid construction XPS + NOMATEC, Paper Table 2]
  U_WALL = 0.30  W/(m²K)  [estimate: soil lambda=2.23 W/(mK), eff. depth ~7 m]
  soil lambda = 2.23 W/(m·K)  [measured, Table 2]
  T_AMB = T_amb(t)            [measured outdoor temperature, time-varying]

Data acquisition
----------------
The measurement data is NOT included in this repository. Download it from:
    https://github.com/PitStorages/HojeTaastrupData

Expected file path (relative to repo root):
    data/HojeTaastrupData/ptes_operation_data_hoje_taastrup_2024.csv

Clone the data repository into the data/ folder:
    git clone https://github.com/PitStorages/HojeTaastrupData data/HojeTaastrupData
"""

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
    / "data/HojeTaastrupData"
    / "ptes_operation_data_hoje_taastrup_2024.csv"
)
OUT_DIR = ROOT / "benchmark/results/hoje_taastrup"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from thermal_energy_storage_model import (
    StorageConfig,
    StorageInputs,
    Port,
    ThermalStorage1D,
    TruncatedPyramidGeometry,
    SplitAmbientLoss,
    TransientGroundLoss,
    WaterProperties,
)

# ---------------------------------------------------------------------------
# Storage parameters
# ---------------------------------------------------------------------------
# Geometry from Sifnaios et al. 2025, Section 4.2
# Lid area 11108 m², bottom 864 m² → equivalent square side lengths
H = 14.05         # m   deepest point (East); paper: 14.05 m
A_TOP = 105.4     # m   sqrt(11108 m²)
A_BOT = 29.4      # m   sqrt(864 m²)
V_PTES = H / 3.0 * (A_TOP**2 + A_TOP * A_BOT + A_BOT**2)   # ~ 70 578 m³

N = 50            # number of nodes
DT = 600.0        # s  timestep (10 min = data step)

# Diffusor heights above bottom [m]  (Sifnaios et al. 2025, Section 4.2)
H_DIFF_TOP = 13.80
H_DIFF_MID = 10.50   # at half storage volume (volumetric centre, not geometric)
H_DIFF_BOT =  0.45

# Heat loss (Sifnaios 2025: U_lid = 0.151 W/(m²K), calculated from lid construction)
U_LID  = 0.151    # W/(m²K)  lid U-value (XPS + NOMATEC, Paper Table 2)
U_WALL = 0.30     # W/(m²K)  walls/bottom: estimate (lambda=2.23, eff. ~7 m)

# TransientGroundLoss parameters (d_total tuned to U_ss ≈ 0.30 W/(m²K))
LAMBDA_SOIL = 2.23    # W/(m·K)  measured, Sifnaios 2025 Table 2
RHO_SOIL    = 2000.0  # kg/m³    clay till, typical
CP_SOIL     = 800.0   # J/(kg·K) clay till
D_TOTAL     = 7.5     # m  → U_ss = lambda/d = 2.23/7.5 = 0.297 ≈ 0.30 W/(m²K)
N_GROUND    = 3       # number of ground layers
T_FAR       = 8.0     # °C  far-field ground temperature (Denmark, deep ground)

# Threshold for "F_mid active" [m³/h]
F_MID_THRESH = 5.0

# Lance sensors for profile comparison (internal sensors, not pipe sensors)
# Depths from water surface level; height above bottom = H – depth
A_COL_TOP = "A_00.25m"   # 0.25 m from top = 13.80 m above bottom ≈ H_DIFF_TOP
A_COL_MID = "A_03.50m"   # 3.50 m from top = 10.55 m above bottom ≈ H_DIFF_MID (10.5 m)
A_COL_BOT = "A_13.80m"   # 13.80 m from top =  0.25 m above bottom ≈ H_DIFF_BOT

# Lance A: sensor depths from surface level [m]  (top to bottom)
A_DEPTHS_FROM_TOP = [
    0.25, 0.50, 0.75, 1.00, 1.25, 1.50,
    2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
    5.00, 5.50, 6.00, 6.50,
    7.50, 8.50, 9.00, 9.50, 10.00,
    11.50, 12.00, 12.50, 12.85, 13.80,
]
A_COLS = [f"A_{d:05.2f}m" for d in A_DEPTHS_FROM_TOP]
A_HEIGHTS = [H - d for d in A_DEPTHS_FROM_TOP]   # height above bottom [m]


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_storage(t_amb: float) -> ThermalStorage1D:
    """Create storage object with geometry, loss model, and solver."""
    geometry = TruncatedPyramidGeometry(A_BOT, A_BOT, A_TOP, A_TOP, H)
    loss = SplitAmbientLoss(U_lid=U_LID, U_wall=U_WALL, T_ambient=t_amb)
    config = StorageConfig(
        volume=V_PTES,
        height=H,
        n_nodes=N,
        geometry=geometry,
        loss_model=loss,
        fluid=WaterProperties(),
        solver="implicit",
        advection_scheme="upwind",
        buoyancy=True,
    )
    return ThermalStorage1D(config)


def build_storage_transient(t_amb: float) -> ThermalStorage1D:
    """Create storage object with TransientGroundLoss (1D RC ground network)."""
    geometry = TruncatedPyramidGeometry(A_BOT, A_BOT, A_TOP, A_TOP, H)
    loss = TransientGroundLoss(
        U_lid=U_LID,
        T_ambient_lid=t_amb,
        lambda_soil=LAMBDA_SOIL,
        rho_soil=RHO_SOIL,
        cp_soil=CP_SOIL,
        d_total=D_TOTAL,
        n_layers=N_GROUND,
        T_far=T_FAR,
    )
    config = StorageConfig(
        volume=V_PTES,
        height=H,
        n_nodes=N,
        geometry=geometry,
        loss_model=loss,
        fluid=WaterProperties(),
        solver="implicit",
        advection_scheme="upwind",
        buoyancy=True,
    )
    return ThermalStorage1D(config)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load dataset and interpolate short gaps."""
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    df.index = df.index.tz_localize(None)   # remove UTC timezone info

    # Close short gaps (max. 6 steps = 1 h)
    for col in ["F_top", "F_mid", "F_bot", "T_top", "T_mid", "T_bot", "T_amb"]:
        df[col] = df[col].interpolate(method="time", limit=6)
    for col in A_COLS:
        df[col] = df[col].interpolate(method="time", limit=6)

    return df


# ---------------------------------------------------------------------------
# Initialisation profile
# ---------------------------------------------------------------------------
def init_profile_from_lanze(row: pd.Series) -> np.ndarray:
    """
    Interpolate measured temperature profile (Lance A) onto the model grid.
    Node k=0 = top (z = H – dz/2), k=N–1 = bottom.
    """
    T_meas = row[A_COLS].values.astype(float)
    h_meas = np.array(A_HEIGHTS)   # height above bottom [m]

    dz = H / N
    node_heights = np.array([H - (k + 0.5) * dz for k in range(N)])
    # np.interp requires ascending x → reverse (bottom to top)
    idx = np.argsort(h_meas)
    return np.interp(node_heights, h_meas[idx], T_meas[idx])


# ---------------------------------------------------------------------------
# Create port list
# ---------------------------------------------------------------------------
def make_ports(row: pd.Series) -> list:
    """
    Three ports at the diffusor positions.

    Convention (dataset): positive = inflow INTO storage.
    Inlet temperature approximation: T_in = measured temperature at diffusor.
    F_mid is always <= 0, so no inflow via the middle port.
    """
    def safe(col, default=0.0):
        v = row.get(col, np.nan)
        return float(v) if np.isfinite(v) else default

    f_top_m3h = safe("F_top")
    f_mid_m3h = safe("F_mid")
    f_bot_m3h = safe("F_bot")

    T_top = safe("T_top", 60.0)
    T_mid = safe("T_mid", 40.0)
    T_bot = safe("T_bot", 40.0)

    def m3h_to_kgs(f_m3h, T_C):
        """Volumetric flow rate m³/h → mass flow rate kg/s (temperature-dependent density)."""
        rho = 999.85 + 5.332e-2 * T_C - 7.564e-3 * T_C**2
        return f_m3h * rho / 3600.0

    m_top = m3h_to_kgs(f_top_m3h, T_top)
    m_mid = m3h_to_kgs(f_mid_m3h, T_mid)
    m_bot = m3h_to_kgs(f_bot_m3h, T_bot)

    ports = []
    if abs(m_top) > 0.01:
        ports.append(Port(
            z=H_DIFF_TOP,
            m_dot=m_top,
            T_in=T_top if m_top > 0 else 0.0,
            label="top",
        ))
    if abs(m_mid) > 0.01:
        ports.append(Port(
            z=H_DIFF_MID,
            m_dot=m_mid,
            T_in=T_mid if m_mid > 0 else 0.0,
            label="mid",
        ))
    if abs(m_bot) > 0.01:
        ports.append(Port(
            z=H_DIFF_BOT,
            m_dot=m_bot,
            T_in=T_bot if m_bot > 0 else 0.0,
            label="bottom",
        ))
    return ports


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _update_loss_temp(storage: ThermalStorage1D, t_amb: float) -> None:
    """Update the time-varying outdoor temperature in the loss model."""
    lm = storage.config.loss_model
    if hasattr(lm, "T_ambient"):
        lm.T_ambient = t_amb
    if hasattr(lm, "T_ambient_lid"):
        lm.T_ambient_lid = t_amb


def run(storage_factory=None) -> tuple:
    """
    Annual simulation 2024.

    Parameters
    ----------
    storage_factory : callable, optional
        Function ``f(t_amb_mean: float) -> ThermalStorage1D``.
        Default: ``build_storage`` (SplitAmbientLoss).

    Returns (df, T_sim_top, T_sim_mid, T_sim_bot, T_profiles_snap, node_heights).
    """
    if storage_factory is None:
        storage_factory = build_storage

    print("Loading dataset ...")
    df = load_data()
    print(f"  {len(df)} timesteps  ({df.index[0].date()} to {df.index[-1].date()})")

    t_amb_mean = float(df["T_amb"].mean())
    storage = storage_factory(t_amb_mean)
    print(
        f"  Storage volume: {V_PTES:.0f} m³  "
        f"(geometry: {A_BOT}×{A_BOT} → {A_TOP}×{A_TOP} m, H={H} m)"
    )

    # Initialisation from Lance A at first timestamp
    t0 = df.index[0]
    T_init = init_profile_from_lanze(df.loc[t0])
    state = storage.initialize(T_init)
    print(f"  Initialisation: {t0}  T_top={T_init[0]:.1f}°C, T_bottom={T_init[-1]:.1f}°C")

    n = len(df)
    dz = H / N
    node_heights = np.array([H - (k + 0.5) * dz for k in range(N)])

    # Node indices for diffusor positions
    k_top = max(0, min(N - 1, int((H - H_DIFF_TOP) / dz)))
    k_mid = max(0, min(N - 1, int((H - H_DIFF_MID) / dz)))
    k_bot = max(0, min(N - 1, int((H - H_DIFF_BOT) / dz)))

    T_sim_top = np.full(n, np.nan)
    T_sim_mid = np.full(n, np.nan)
    T_sim_bot = np.full(n, np.nan)
    T_profiles_snap = {}   # {month-str: (i, T_array)}

    print("Starting simulation ...")
    for i, (ts, row) in enumerate(df.iterrows()):
        T_sim_top[i] = state.temperatures[k_top]
        T_sim_mid[i] = state.temperatures[k_mid]
        T_sim_bot[i] = state.temperatures[k_bot]

        # Monthly profile snapshots
        if ts.day == 1 and ts.hour == 0 and ts.minute == 0:
            T_profiles_snap[ts.strftime("%Y-%m")] = (i, state.temperatures.copy())

        # Time-varying outdoor temperature
        t_amb = float(row.get("T_amb", t_amb_mean))
        if not np.isfinite(t_amb):
            t_amb = t_amb_mean
        _update_loss_temp(storage, t_amb)

        ports = make_ports(row)
        inputs = StorageInputs(ports=ports)
        outputs = storage.step(state, dt=DT, inputs=inputs)
        state = outputs.state

        if i % 10000 == 0:
            m_in = sum(p.m_dot for p in ports if p.m_dot > 0)
            print(
                f"  {ts.date()}  ports={len(ports)}  m_in={m_in:.1f} kg/s"
                f"  T_top={state.temperatures[0]:.1f}°C"
            )

    print("Simulation complete.")
    return df, T_sim_top, T_sim_mid, T_sim_bot, T_profiles_snap, node_heights


# ---------------------------------------------------------------------------
# Evaluation: MAE
# ---------------------------------------------------------------------------
def compute_mae(df, T_sim_top, T_sim_mid, T_sim_bot, label: str = ""):
    """
    MAE evaluation based on lance sensors (internal sensors).

    T_top, T_mid are pipe sensors and are NOT used:
      - T_top cools during standby to 35–62°C (pipe), lance shows 52–89°C
      - T_mid cools at F_mid=0 to ~7°C (ambient temperature)
    T_bot is nearly identical to A_13.80m (0.22 K offset) – lance used for consistency.
    """
    def mae(meas, sim):
        valid = np.isfinite(meas) & np.isfinite(sim)
        return float(np.mean(np.abs(meas[valid] - sim[valid]))) if valid.any() else np.nan

    T_meas_top = df[A_COL_TOP].values   # Lance A_00.25m
    T_meas_mid = df[A_COL_MID].values   # Lance A_03.50m
    T_meas_bot = df[A_COL_BOT].values   # Lance A_13.80m

    mae_top = mae(T_meas_top, T_sim_top)
    mae_mid = mae(T_meas_mid, T_sim_mid)
    mae_bot = mae(T_meas_bot, T_sim_bot)
    mae_all = mae(
        np.concatenate([T_meas_top, T_meas_mid, T_meas_bot]),
        np.concatenate([T_sim_top,  T_sim_mid,  T_sim_bot]),
    )

    prefix = f" [{label}]" if label else ""
    print(f"\nMAE lance sensors vs. model{prefix} (N={N}, full year 2024):")
    print(f"  Top    ({H_DIFF_TOP} m, A_00.25m): {mae_top:.2f} K")
    print(f"  Middle ({H_DIFF_MID} m, {A_COL_MID}): {mae_mid:.2f} K")
    print(f"  Bottom ({H_DIFF_BOT} m, A_13.80m): {mae_bot:.2f} K")
    print(f"  Total:                      {mae_all:.2f} K")
    return dict(top=mae_top, middle=mae_mid, bottom=mae_bot, total=mae_all)


# ---------------------------------------------------------------------------
# Plot 1: Diffusor temperature time series
# ---------------------------------------------------------------------------
def plot_timeseries(df, T_sim_top, T_sim_mid, T_sim_bot, tag: str = "") -> None:
    idx = df.index

    # Hourly averages for cleaner visualisation
    df_sim = pd.DataFrame(
        {"T_top_sim": T_sim_top, "T_mid_sim": T_sim_mid, "T_bot_sim": T_sim_bot},
        index=idx,
    ).resample("1h").mean()
    # Lance sensors (internal sensors) as reference – not hydraulic pipe sensors
    df_h = df[[A_COL_TOP, A_COL_MID, A_COL_BOT]].resample("1h").mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    specs = [
        ("Top    (A_00.25m, 13.75 m)", "T_top_sim", A_COL_TOP, "tab:red"),
        ("Middle (A_06.50m,  7.50 m)", "T_mid_sim", A_COL_MID, "tab:orange"),
        ("Bottom (A_13.80m,  0.20 m)", "T_bot_sim", A_COL_BOT, "tab:blue"),
    ]
    for ax, (label, col_sim, col_meas, color) in zip(axes, specs):
        ax.plot(df_h.index, df_h[col_meas], color="black", lw=0.8,
                label="Measurement (Lance A)", alpha=0.8)
        ax.plot(df_sim.index, df_sim[col_sim], color=color, lw=1.0,
                label="Simulation", alpha=0.9)
        ax.set_ylabel("Temperature [°C]")
        ax.set_title(f"Lance A – {label}")
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Høje Taastrup PTES 2024 – ThermalStorage1D Validation\n"
        f"N={N}, U_lid={U_LID} U_wall={U_WALL} W/(m²K), implicit + upwind + buoyancy",
        fontsize=11,
    )
    fig.tight_layout()
    fname = OUT_DIR / f"hoje_taastrup_timeseries{tag}"
    fig.savefig(fname.with_suffix(".svg"))
    fig.savefig(fname.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Time series: {fname.with_suffix('.svg')}")


# ---------------------------------------------------------------------------
# Plot 1b: Abstract version – top + bottom only, compact height
# ---------------------------------------------------------------------------
def plot_timeseries_abstract(df, T_sim_top, T_sim_bot, mae_vals: dict) -> None:
    """Two-panel abstract figure (top + bottom)."""
    from matplotlib.ticker import FuncFormatter
    MONTHS_EN = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    def _eng_month(x, pos):
        return MONTHS_EN.get(mdates.num2date(x).month, "")

    idx = df.index
    df_sim = pd.DataFrame(
        {"T_top_sim": T_sim_top, "T_bot_sim": T_sim_bot}, index=idx,
    ).resample("1h").mean()
    df_h = df[[A_COL_TOP, A_COL_BOT]].resample("1h").mean()

    FS_TITLE = 11; FS_LABEL = 10; FS_LEGEND = 9; FS_TICK = 9

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    specs = [
        (f"Top diffusor ({H_DIFF_TOP}\u2009m)",    "T_top_sim", A_COL_TOP,  mae_vals["top"],    "lower center"),
        (f"Bottom diffusor ({H_DIFF_BOT}\u2009m)", "T_bot_sim", A_COL_BOT,  mae_vals["bottom"], "upper center"),
    ]
    for ax, (label, col_sim, col_meas, mae_k, leg_loc) in zip(axes, specs):
        ax.plot(df_h.index, df_h[col_meas], color="black", lw=0.8,
                label="Measurement (Sifnaios et al., 2025)")
        ax.plot(df_sim.index, df_sim[col_sim], color="tab:red", lw=1.0, alpha=0.88,
                label=f"Model N\u202f=\u202f{N} implicit  |  MAE\u202f=\u202f{mae_k:.2f}\u2009K")
        ax.set_ylabel("T [°C]", fontsize=FS_LABEL)
        ax.set_title(label, fontsize=FS_TITLE)
        ax.legend(loc=leg_loc, fontsize=FS_LEGEND)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(_eng_month))
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = OUT_DIR / "hoje_taastrup_timeseries_abstract"
    for suffix in (".svg", ".pdf"):
        fig.savefig(fname.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)
    print(f"  Abstract plot: {fname.with_suffix('.svg')}")


# ---------------------------------------------------------------------------
# Plot 2: Temperature profiles (seasonal)
# ---------------------------------------------------------------------------
def plot_profiles(df, T_profiles_snap, node_heights, tag: str = "") -> None:
    months = ["2024-02", "2024-05", "2024-08", "2024-11"]
    titles = ["Feb",     "May",     "Aug",     "Nov"]
    colors = ["tab:blue", "tab:green", "tab:red", "tab:purple"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)

    for ax, month, title, color in zip(axes, months, titles, colors):
        # Measured profile (Lance A)
        ts_str = month + "-01"
        try:
            ts = pd.Timestamp(ts_str)
            ts_near = df.index[df.index.searchsorted(ts)]
            T_meas = df.loc[ts_near, A_COLS].values.astype(float)
            ax.plot(T_meas, A_HEIGHTS, "ko", ms=3, label="Measurement (Lance A)")
        except Exception:
            pass

        # Simulated profile from snapshot
        if month in T_profiles_snap:
            _, T_sim = T_profiles_snap[month]
            ax.plot(T_sim, node_heights, color=color, lw=1.5, label="Simulation")

        ax.set_title(title)
        ax.set_xlabel("Temperature [°C]")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, H)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Height above bottom [m]")
    fig.suptitle(
        "Høje Taastrup PTES 2024 – Temperature profiles (seasonal)",
        fontsize=11,
    )
    fig.tight_layout()
    fname = OUT_DIR / f"hoje_taastrup_profiles{tag}"
    fig.savefig(fname.with_suffix(".svg"))
    fig.savefig(fname.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Profiles:    {fname.with_suffix('.svg')}")


# ---------------------------------------------------------------------------
# Plot 3: Loss model comparison – SplitAmbientLoss vs. TransientGroundLoss
# ---------------------------------------------------------------------------
def plot_loss_model_comparison(
    df,
    T_split_top, T_split_bot,
    T_tgl_top,   T_tgl_bot,
    mae_split: dict,
    mae_tgl: dict,
) -> None:
    """Two-panel comparison of loss models (top + bottom)."""
    from matplotlib.ticker import FuncFormatter
    MONTHS_EN = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    def _eng_month(x, pos):
        return MONTHS_EN.get(mdates.num2date(x).month, "")

    idx = df.index
    df_split = pd.DataFrame(
        {"T_top": T_split_top, "T_bot": T_split_bot}, index=idx,
    ).resample("1h").mean()
    df_tgl = pd.DataFrame(
        {"T_top": T_tgl_top, "T_bot": T_tgl_bot}, index=idx,
    ).resample("1h").mean()
    df_h = df[[A_COL_TOP, A_COL_BOT]].resample("1h").mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    specs = [
        (f"Top ({H_DIFF_TOP} m)",    "T_top", A_COL_TOP,  "top"),
        (f"Bottom ({H_DIFF_BOT} m)", "T_bot", A_COL_BOT,  "bottom"),
    ]
    for ax, (label, col, col_meas, key) in zip(axes, specs):
        ax.plot(df_h.index, df_h[col_meas], color="black", lw=0.8, label="Measurement")
        ax.plot(df_split.index, df_split[col], color="tab:blue", lw=1.0, alpha=0.85,
                label=f"SplitAmbientLoss  MAE={mae_split[key]:.2f} K")
        ax.plot(df_tgl.index, df_tgl[col], color="tab:red", lw=1.0, alpha=0.85,
                label=f"TransientGroundLoss  MAE={mae_tgl[key]:.2f} K")
        ax.set_ylabel("T [°C]")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(_eng_month))
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Høje Taastrup PTES 2024 – Loss model comparison\n"
        f"N={N}, implicit + upwind + buoyancy",
        fontsize=11,
    )
    fig.tight_layout()
    fname = OUT_DIR / "hoje_taastrup_loss_model_comparison"
    for suffix in (".svg", ".pdf"):
        fig.savefig(fname.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss model comparison: {fname.with_suffix('.svg')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Høje Taastrup PTES 2024 – Validation ===")
    print(f"  File:   {DATA_FILE}")
    print(f"  Output: {OUT_DIR}")
    print()

    # --- Simulation 1: SplitAmbientLoss (steady-state, constant U_wall) ---
    print("\n--- SplitAmbientLoss ---")
    df, T_top_s, T_mid_s, T_bot_s, snaps_s, node_heights = run(build_storage)
    mae_s = compute_mae(df, T_top_s, T_mid_s, T_bot_s, label="SplitAmbientLoss")

    # --- Simulation 2: TransientGroundLoss (1D RC ground network) ---
    print("\n--- TransientGroundLoss ---")
    _, T_top_t, T_mid_t, T_bot_t, snaps_t, _ = run(build_storage_transient)
    mae_t = compute_mae(df, T_top_t, T_mid_t, T_bot_t, label="TransientGroundLoss")

    # --- Summary ---
    print("\n=== MAE Comparison ===")
    print(f"{'Model':<25} {'Top':>8} {'Middle':>8} {'Bottom':>8} {'Total':>8}")
    print(f"{'SplitAmbientLoss':<25} {mae_s['top']:>8.2f} {mae_s['middle']:>8.2f} {mae_s['bottom']:>8.2f} {mae_s['total']:>8.2f}")
    print(f"{'TransientGroundLoss':<25} {mae_t['top']:>8.2f} {mae_t['middle']:>8.2f} {mae_t['bottom']:>8.2f} {mae_t['total']:>8.2f}")
    diff = mae_t["total"] - mae_s["total"]
    print(f"  Difference (TGL - SAL): {diff:+.2f} K")

    # --- Result CSV (SplitAmbientLoss as primary result) ---
    result = pd.DataFrame(
        {
            "T_top_sim":   T_top_s,
            "T_mid_sim":   T_mid_s,
            "T_bot_sim":   T_bot_s,
            "T_top_tgl":   T_top_t,
            "T_mid_tgl":   T_mid_t,
            "T_bot_tgl":   T_bot_t,
            "T_lanze_top": df[A_COL_TOP].values,
            "T_lanze_mid": df[A_COL_MID].values,
            "T_lanze_bot": df[A_COL_BOT].values,
            "T_top_rohr":  df["T_top"].values,
            "T_mid_rohr":  df["T_mid"].values,
            "T_bot_rohr":  df["T_bot"].values,
        },
        index=df.index,
    )
    csv_path = OUT_DIR / "hoje_taastrup_comparison.csv"
    result.to_csv(csv_path)
    print(f"\nResults saved: {csv_path}")

    # --- Plots ---
    print("\nGenerating plots ...")
    plot_timeseries(df, T_top_s, T_mid_s, T_bot_s)
    plot_timeseries_abstract(df, T_top_s, T_bot_s, mae_s)
    plot_profiles(df, snaps_s, node_heights)
    plot_loss_model_comparison(df, T_top_s, T_bot_s, T_top_t, T_bot_t, mae_s, mae_t)

    print("\nDone.")


if __name__ == "__main__":
    main()
