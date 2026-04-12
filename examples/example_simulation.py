"""
Example simulations for the 1D Thermal Storage model
=====================================================

This script demonstrates the use of the ThermalStorage1D model
through three realistic scenarios:

    Scenario 1 – Pure charging:
        A thermal storage tank is fully charged with hot fluid from a heat
        source (e.g. CHP plant or solar collectors).

    Scenario 2 – Pure discharging:
        A fully charged storage supplies the district heating network.

    Scenario 3 – Simultaneous charging and discharging:
        Both circuits operate at the same time, as occurs in real district
        heating systems during peak shaving.

Each scenario shows how states are passed step by step and how outputs
are integrated into a network simulation loop.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from thermal_energy_storage_model import (
    StorageConfig,
    StorageInputs,
    StorageState,
    ThermalStorage1D,
)

# Optional: Matplotlib for visualisation
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not installed – no plots will be produced.\n")


# ===========================================================================
# Helper functions
# ===========================================================================

def print_separator(title: str = "") -> None:
    """Print a formatted separator line."""
    width = 70
    if title:
        padding = (width - len(title) - 2) // 2
        print("=" * padding + f" {title} " + "=" * (width - padding - len(title) - 2))
    else:
        print("=" * width)


def get_port_temps(inputs: StorageInputs, outputs) -> tuple:
    """
    Extract outlet temperatures and thermal powers for two_port inputs.

    Returns (T_charge_out, T_discharge_out, Q_charge, Q_discharge).
    Missing circuits return None or 0.0.

    Note: port label strings match what StorageInputs.two_port() generates.
    """
    cp = 4187.0  # J/(kg·K), simplified
    T_charge_out = None
    T_discharge_out = None
    m_c = 0.0
    T_c_in = 0.0
    m_d = 0.0

    for i, p in enumerate(inputs.ports):
        T_node = outputs.port_temperatures[i]
        if p.label == "charge_out":
            T_charge_out = T_node
            m_c = abs(p.m_dot)
        elif p.label == "charge_in":
            T_c_in = p.T_in
        elif p.label == "discharge_out":
            T_discharge_out = T_node
            m_d = abs(p.m_dot)

    T_d_in = 0.0
    for i, p in enumerate(inputs.ports):
        if p.label == "discharge_in":
            T_d_in = p.T_in

    Q_charge    = m_c * cp * (T_c_in - (T_charge_out or 0.0))
    Q_discharge = m_d * cp * ((T_discharge_out or 0.0) - T_d_in)

    return T_charge_out, T_discharge_out, Q_charge, Q_discharge


def print_step_info(
    t: float,
    inputs: StorageInputs,
    T_charge_out,
    T_discharge_out,
    Q_charge: float,
    Q_discharge: float,
    Q_loss: float,
    state: StorageState,
) -> None:
    """Print the key quantities of a timestep in formatted form."""
    m_c = sum(p.m_dot for p in inputs.ports if p.m_dot > 0 and "charge" in p.label)
    m_d = sum(p.m_dot for p in inputs.ports if p.m_dot > 0 and "discharge" in p.label)
    T_c_in = next((p.T_in for p in inputs.ports if p.label == "charge_in"), 0.0)
    T_d_in = next((p.T_in for p in inputs.ports if p.label == "discharge_in"), 0.0)

    T_c_str = f"{T_charge_out:.1f}" if T_charge_out is not None else "---"
    T_d_str = f"{T_discharge_out:.1f}" if T_discharge_out is not None else "---"

    print(
        f"  t={t/3600:.2f} h | "
        f"m_chg={m_c:.1f} kg/s T_in={T_c_in:.1f} C -> T_out={T_c_str} C "
        f"Q={Q_charge/1e3:.1f} kW | "
        f"m_dis={m_d:.1f} kg/s T_in={T_d_in:.1f} C -> T_out={T_d_str} C "
        f"Q={Q_discharge/1e3:.1f} kW | "
        f"Q_loss={Q_loss/1e3:.2f} kW | SOC={state.T_mean:.1f}°C"
    )


# ===========================================================================
# Scenario 1: Pure charging
# ===========================================================================

def scenario_charging() -> dict:
    """
    Scenario 1: Full charging of a storage tank.

    A 100 m³ buffer storage with 20 layers is heated from 50 °C (cold,
    discharged) to approximately 85 °C (hot, charged). The charging circuit
    supplies hot fluid at 90 °C from the top; the discharging circuit is idle.

    Returns
    -------
    dict
        Time series of simulation results.
    """
    print_separator("Scenario 1: Pure charging")

    # --- Storage configuration ---
    config = StorageConfig(
        volume=100.0,       # m³
        height=10.0,        # m
        n_nodes=20,
        U_loss=0.3,         # W/(m²·K)
        T_ambient=10.0,     # °C
    )
    storage = ThermalStorage1D(config)

    # --- Time parameters ---
    dt = 60.0               # timestep [s]
    t_end = 6 * 3600        # 6 hours of charging

    # --- Initial state: storage cold (discharged) ---
    state = storage.initialize(T_init=50.0)

    # --- Check recommended maximum timestep ---
    m_dot_charge = 8.0  # kg/s
    dt_max = storage.max_stable_dt(m_dot_charge)
    print(f"  Maximum stable timestep: {dt_max:.1f} s (using: {dt} s)")

    # --- Boundary conditions (constant) ---
    inputs = StorageInputs.two_port(
        m_dot_charge=m_dot_charge,
        T_charge_in=90.0,       # °C  (hot fluid from heat source)
        height=config.height,
    )

    # --- Time loop (typical co-simulation structure) ---
    times = []
    T_top_list = []
    T_bottom_list = []
    T_mean_list = []
    T_charge_out_list = []
    Q_charge_list = []
    Q_loss_list = []

    n_steps = int(t_end / dt)
    print_interval = max(1, n_steps // 6)

    for step_idx in range(n_steps):
        outputs = storage.step(state, dt=dt, inputs=inputs)
        state = outputs.state

        T_c_out, T_d_out, Q_c, Q_d = get_port_temps(inputs, outputs)

        t = state.time
        times.append(t)
        T_top_list.append(state.T_top)
        T_bottom_list.append(state.T_bottom)
        T_mean_list.append(state.T_mean)
        T_charge_out_list.append(T_c_out)
        Q_charge_list.append(Q_c)
        Q_loss_list.append(outputs.Q_loss)

        if step_idx % print_interval == 0 or step_idx == n_steps - 1:
            print_step_info(t, inputs, T_c_out, T_d_out, Q_c, Q_d, outputs.Q_loss, state)

    print(f"\n  Final temperature top:    {state.T_top:.2f} °C")
    print(f"  Final temperature bottom: {state.T_bottom:.2f} °C")
    soc = storage.get_soc(state, T_min=50.0, T_max=90.0)
    print(f"  State of charge (SOC):    {soc*100:.1f} %")

    return {
        "times": np.array(times),
        "T_top": np.array(T_top_list),
        "T_bottom": np.array(T_bottom_list),
        "T_mean": np.array(T_mean_list),
        "T_charge_out": np.array(T_charge_out_list),
        "Q_charge": np.array(Q_charge_list),
        "Q_loss": np.array(Q_loss_list),
        "config": config,
        "final_state": state,
    }


# ===========================================================================
# Scenario 2: Pure discharging
# ===========================================================================

def scenario_discharging(initial_state: StorageState, config: StorageConfig) -> dict:
    """
    Scenario 2: Full discharging of a hot storage tank.

    The storage charged at the end of Scenario 1 supplies the district heating
    network. The discharging circuit extracts hot fluid from the top and returns
    cold water (50 °C) at the bottom.

    Parameters
    ----------
    initial_state : StorageState
        Starting state (charged storage from Scenario 1).
    config : StorageConfig
        Storage configuration.

    Returns
    -------
    dict
        Time series of simulation results.
    """
    print_separator("Scenario 2: Pure discharging")

    storage = ThermalStorage1D(config)

    dt = 60.0               # timestep [s]
    t_end = 6 * 3600        # 6 hours of discharging

    state = initial_state.copy()
    state = StorageState(temperatures=state.temperatures, time=0.0)  # reset clock

    m_dot_discharge = 5.0  # kg/s
    inputs = StorageInputs.two_port(
        m_dot_discharge=m_dot_discharge,
        T_discharge_in=50.0,    # °C  (cold network return)
        height=config.height,
    )

    dt_max = storage.max_stable_dt(m_dot_discharge)
    print(f"  Maximum stable timestep: {dt_max:.1f} s (using: {dt} s)")

    times = []
    T_top_list = []
    T_bottom_list = []
    T_mean_list = []
    T_discharge_out_list = []
    Q_discharge_list = []
    Q_loss_list = []

    n_steps = int(t_end / dt)
    print_interval = max(1, n_steps // 6)

    for step_idx in range(n_steps):
        outputs = storage.step(state, dt=dt, inputs=inputs)
        state = outputs.state

        T_c_out, T_d_out, Q_c, Q_d = get_port_temps(inputs, outputs)

        t = state.time
        times.append(t)
        T_top_list.append(state.T_top)
        T_bottom_list.append(state.T_bottom)
        T_mean_list.append(state.T_mean)
        T_discharge_out_list.append(T_d_out)
        Q_discharge_list.append(Q_d)
        Q_loss_list.append(outputs.Q_loss)

        if step_idx % print_interval == 0 or step_idx == n_steps - 1:
            print_step_info(t, inputs, T_c_out, T_d_out, Q_c, Q_d, outputs.Q_loss, state)

    print(f"\n  Final temperature top:    {state.T_top:.2f} °C")
    print(f"  Final temperature bottom: {state.T_bottom:.2f} °C")
    soc = storage.get_soc(state, T_min=50.0, T_max=90.0)
    print(f"  State of charge (SOC):    {soc*100:.1f} %")

    return {
        "times": np.array(times),
        "T_top": np.array(T_top_list),
        "T_bottom": np.array(T_bottom_list),
        "T_mean": np.array(T_mean_list),
        "T_discharge_out": np.array(T_discharge_out_list),
        "Q_discharge": np.array(Q_discharge_list),
        "Q_loss": np.array(Q_loss_list),
    }


# ===========================================================================
# Scenario 3: Simultaneous charging and discharging
# ===========================================================================

def scenario_simultaneous() -> dict:
    """
    Scenario 3: Simultaneous charging and discharging.

    Both circuits are active at the same time. The charging circuit
    (e.g. CHP plant) delivers more power than the network requires,
    so excess heat is stored.

    A 24-hour operation with three phases is simulated:
        - Phase 1 (0–8 h):   Charging only (night, low demand)
        - Phase 2 (8–16 h):  Simultaneous (m_chg > m_dis)
        - Phase 3 (16–24 h): Peak load (m_chg < m_dis, storage discharging)

    Returns
    -------
    dict
        Time series of simulation results.
    """
    print_separator("Scenario 3: Simultaneous charging and discharging (24 h)")

    config = StorageConfig(
        volume=200.0,
        height=12.0,
        n_nodes=20,
        U_loss=0.25,
        T_ambient=12.0,
    )
    storage = ThermalStorage1D(config)

    dt = 300.0              # 5-minute timesteps [s]
    t_end = 24 * 3600       # 24 hours

    # Initial state: half charged
    T_profile = np.linspace(75.0, 55.0, config.n_nodes)
    state = storage.initialize(T_init=T_profile)

    # --- Operating profiles ---
    def get_inputs(t: float) -> StorageInputs:
        """Return time-dependent boundary conditions (simplified daily profile)."""
        t_h = t / 3600.0  # time in hours

        if t_h < 8.0:
            # Night: charging only (CHP running, network demand low)
            return StorageInputs.two_port(
                m_dot_charge=6.0, T_charge_in=88.0,
                height=config.height,
            )
        elif t_h < 16.0:
            # Day: simultaneous (CHP + network)
            return StorageInputs.two_port(
                m_dot_charge=8.0, T_charge_in=90.0,
                m_dot_discharge=5.0, T_discharge_in=52.0,
                height=config.height,
            )
        else:
            # Evening: peak load (discharging dominates)
            return StorageInputs.two_port(
                m_dot_charge=3.0, T_charge_in=85.0,
                m_dot_discharge=10.0, T_discharge_in=50.0,
                height=config.height,
            )

    times = []
    T_top_list = []
    T_bottom_list = []
    T_mean_list = []
    T_charge_out_list = []
    T_discharge_out_list = []
    Q_charge_list = []
    Q_discharge_list = []
    Q_loss_list = []
    m_dot_charge_list = []
    m_dot_discharge_list = []

    n_steps = int(t_end / dt)
    print_interval = max(1, n_steps // 8)

    for step_idx in range(n_steps):
        t = state.time
        inputs = get_inputs(t)

        outputs = storage.step(state, dt=dt, inputs=inputs)
        state = outputs.state

        T_c_out, T_d_out, Q_c, Q_d = get_port_temps(inputs, outputs)

        # Read mass flows from ports
        m_c = sum(p.m_dot for p in inputs.ports if p.label == "charge_in")
        m_d = sum(p.m_dot for p in inputs.ports if p.label == "discharge_in")

        times.append(t)
        T_top_list.append(state.T_top)
        T_bottom_list.append(state.T_bottom)
        T_mean_list.append(state.T_mean)
        T_charge_out_list.append(T_c_out if T_c_out is not None else float("nan"))
        T_discharge_out_list.append(T_d_out if T_d_out is not None else float("nan"))
        Q_charge_list.append(Q_c)
        Q_discharge_list.append(Q_d)
        Q_loss_list.append(outputs.Q_loss)
        m_dot_charge_list.append(m_c)
        m_dot_discharge_list.append(m_d)

        if step_idx % print_interval == 0 or step_idx == n_steps - 1:
            print_step_info(t, inputs, T_c_out, T_d_out, Q_c, Q_d, outputs.Q_loss, state)

    print(f"\n  Final temperature top:    {state.T_top:.2f} °C")
    print(f"  Final temperature bottom: {state.T_bottom:.2f} °C")
    soc = storage.get_soc(state, T_min=50.0, T_max=90.0)
    print(f"  State of charge (SOC):    {soc*100:.1f} %")

    return {
        "times": np.array(times),
        "T_top": np.array(T_top_list),
        "T_bottom": np.array(T_bottom_list),
        "T_mean": np.array(T_mean_list),
        "T_charge_out": np.array(T_charge_out_list),
        "T_discharge_out": np.array(T_discharge_out_list),
        "Q_charge": np.array(Q_charge_list),
        "Q_discharge": np.array(Q_discharge_list),
        "Q_loss": np.array(Q_loss_list),
        "m_dot_charge": np.array(m_dot_charge_list),
        "m_dot_discharge": np.array(m_dot_discharge_list),
    }


# ===========================================================================
# Visualisation
# ===========================================================================

def plot_all_scenarios(
    res1: dict,
    res2: dict,
    res3: dict,
) -> None:
    """
    Create an overview figure for all three scenarios.

    Parameters
    ----------
    res1 : dict
        Results from Scenario 1 (charging).
    res2 : dict
        Results from Scenario 2 (discharging).
    res3 : dict
        Results from Scenario 3 (simultaneous).
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available – no plots produced.")
        return

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "1D Thermal Storage Model – Simulation Results",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ---- Scenario 1: Charging ----
    ax1a = fig.add_subplot(gs[0, 0])
    t1 = res1["times"] / 3600.0
    ax1a.plot(t1, res1["T_top"], label="T top (hot)", color="tab:red")
    ax1a.plot(t1, res1["T_mean"], label="T mean", color="tab:orange", linestyle="--")
    ax1a.plot(t1, res1["T_bottom"], label="T bottom (cold)", color="tab:blue")
    ax1a.set_title("Scenario 1: Pure charging")
    ax1a.set_xlabel("Time [h]")
    ax1a.set_ylabel("Temperature [°C]")
    ax1a.legend(fontsize=8)
    ax1a.grid(True, alpha=0.3)

    ax1b = fig.add_subplot(gs[0, 1])
    ax1b.plot(t1, res1["Q_charge"] / 1e3, label="Charging power", color="tab:red")
    ax1b.plot(t1, res1["Q_loss"] / 1e3, label="Losses", color="gray", linestyle="--")
    ax1b.set_title("Scenario 1: Thermal power")
    ax1b.set_xlabel("Time [h]")
    ax1b.set_ylabel("Power [kW]")
    ax1b.legend(fontsize=8)
    ax1b.grid(True, alpha=0.3)

    # ---- Scenario 2: Discharging ----
    ax2a = fig.add_subplot(gs[1, 0])
    t2 = res2["times"] / 3600.0
    ax2a.plot(t2, res2["T_top"], label="T top (supply)", color="tab:red")
    ax2a.plot(t2, res2["T_mean"], label="T mean", color="tab:orange", linestyle="--")
    ax2a.plot(t2, res2["T_bottom"], label="T bottom (return)", color="tab:blue")
    ax2a.plot(t2, res2["T_discharge_out"], label="Network supply", color="tab:green", linestyle=":")
    ax2a.set_title("Scenario 2: Pure discharging")
    ax2a.set_xlabel("Time [h]")
    ax2a.set_ylabel("Temperature [°C]")
    ax2a.legend(fontsize=8)
    ax2a.grid(True, alpha=0.3)

    ax2b = fig.add_subplot(gs[1, 1])
    ax2b.plot(t2, res2["Q_discharge"] / 1e3, label="Discharging power", color="tab:blue")
    ax2b.plot(t2, res2["Q_loss"] / 1e3, label="Losses", color="gray", linestyle="--")
    ax2b.set_title("Scenario 2: Thermal power")
    ax2b.set_xlabel("Time [h]")
    ax2b.set_ylabel("Power [kW]")
    ax2b.legend(fontsize=8)
    ax2b.grid(True, alpha=0.3)

    # ---- Scenario 3: Simultaneous ----
    ax3a = fig.add_subplot(gs[2, 0])
    t3 = res3["times"] / 3600.0
    ax3a.plot(t3, res3["T_top"], label="T top", color="tab:red")
    ax3a.plot(t3, res3["T_mean"], label="T mean", color="tab:orange", linestyle="--")
    ax3a.plot(t3, res3["T_bottom"], label="T bottom", color="tab:blue")
    ax3a.plot(t3, res3["T_discharge_out"], label="Network supply", color="tab:green", linestyle=":")
    ax3a.plot(t3, res3["T_charge_out"], label="Source return", color="tab:purple", linestyle=":")
    # Phase boundaries
    for xv in [8.0, 16.0]:
        ax3a.axvline(x=xv, color="black", linestyle=":", alpha=0.5)
    ax3a.text(4, 54, "Charging", ha="center", fontsize=8, color="gray")
    ax3a.text(12, 54, "Simultaneous", ha="center", fontsize=8, color="gray")
    ax3a.text(20, 54, "Peak load", ha="center", fontsize=8, color="gray")
    ax3a.set_title("Scenario 3: Simultaneous charging and discharging")
    ax3a.set_xlabel("Time [h]")
    ax3a.set_ylabel("Temperature [°C]")
    ax3a.legend(fontsize=7)
    ax3a.grid(True, alpha=0.3)

    ax3b = fig.add_subplot(gs[2, 1])
    ax3b.plot(t3, res3["Q_charge"] / 1e3, label="Charging power", color="tab:red")
    ax3b.plot(t3, res3["Q_discharge"] / 1e3, label="Discharging power", color="tab:blue")
    ax3b.plot(t3, res3["Q_loss"] / 1e3, label="Losses", color="gray", linestyle="--")
    for xv in [8.0, 16.0]:
        ax3b.axvline(x=xv, color="black", linestyle=":", alpha=0.5)
    ax3b.set_title("Scenario 3: Thermal power")
    ax3b.set_xlabel("Time [h]")
    ax3b.set_ylabel("Power [kW]")
    ax3b.legend(fontsize=8)
    ax3b.grid(True, alpha=0.3)

    out_path = Path(__file__).parent / "simulation_results.svg"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.show()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """Run all three scenarios sequentially and produce plots."""

    print_separator()
    print("  1D Thermal Storage Model – Example Simulations")
    print_separator()
    print()

    # Scenario 1: charging
    res1 = scenario_charging()
    print()

    # Scenario 2: discharging (starts from the end state of Scenario 1)
    res2 = scenario_discharging(
        initial_state=res1["final_state"],
        config=res1["config"],
    )
    print()

    # Scenario 3: simultaneous (independent)
    res3 = scenario_simultaneous()
    print()

    # Visualise results
    print_separator("Plot")
    plot_all_scenarios(res1, res2, res3)

    print_separator()
    print("  Simulation complete.")
    print_separator()


if __name__ == "__main__":
    main()
