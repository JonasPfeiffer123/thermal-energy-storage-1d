"""
Shared utility functions for all benchmark scripts.

Contains physical calculations (water density, enthalpy, usable energy),
profile interpolation, and output helpers.
"""

from __future__ import annotations

import numpy as np

from config_benchmark import (
    A_CROSS, H_WS, T_END_CHARGE, T_END_IDLE,
    R_INNER, V_TANK, U_WALL, T_AMB,
)


# ── Physical helper functions ─────────────────────────────────────────────────

def rho_water(T: float) -> float:
    """
    Water density using the FreeTTES polynomial fit [kg/m³].

    Identical to FreeTTES_model._sw_rho() so that mass flow conversions
    match exactly.

    Parameters
    ----------
    T : float
        Temperature [°C].

    Returns
    -------
    float
        Density [kg/m³].
    """
    return -2.525726e-3 * T**2 - 2.123038e-1 * T + 1.005011e3


def m3h_to_kgs(m3h: float, T_ref: float) -> float:
    """Volumetric flow rate [m³/h] → mass flow rate [kg/s] at temperature T_ref [°C]."""
    return rho_water(T_ref) * m3h / 3600.0


def _sw_h(T: float | np.ndarray) -> float | np.ndarray:
    """
    Specific enthalpy of water [J/kg] using the FreeTTES polynomial.

    Identical to FreeTTES_model._sw_h() so that usable-energy calculations
    match exactly.
    """
    return 4.394221e-01 * T**2 + 4.129877e+03 * T + 1.987100e+03


def compute_useful_energy_MWh(
    temperatures: np.ndarray,
    dz: float,
    node_heights_from_bottom: np.ndarray,
    T_ref: float = 10.0,
    h_exclude_top: float = 0.5,
) -> float:
    """
    Compute the usable energy stored in the tank (analogous to FreeTTES E_nutz).

    Replicates FreeTTES ``__energie_nutz``:
    - Enthalpy-based calculation with temperature-dependent cp (FreeTTES polynomial)
    - Only nodes with T_i > T_ref are counted
    - Nodes above ``H_WS − h_exclude_top`` are excluded
      (FreeTTES excludes the upper diffusor/headspace zone)

    Parameters
    ----------
    temperatures : numpy.ndarray
        Temperature profile [°C], index 0 = top.
    dz : float
        Node height [m] (uniform grid).
    node_heights_from_bottom : numpy.ndarray
        Height of each node above the tank bottom [m], index 0 = top.
    T_ref : float
        Reference temperature [°C]. Default: 10 °C (like FreeTTES T_grenz).
    h_exclude_top : float
        Height of the excluded headspace zone [m]. Default: 0.5 m.

    Returns
    -------
    float
        Usable energy [MWh].
    """
    h_limit = H_WS - h_exclude_top
    h_ref   = _sw_h(T_ref)

    E_J = 0.0
    for T_i, h_i in zip(temperatures, node_heights_from_bottom):
        if T_i <= T_ref:
            continue
        h_top  = h_i + dz / 2
        h_bot  = h_i - dz / 2
        if h_bot >= h_limit:
            continue
        dz_eff = min(h_top, h_limit) - h_bot
        rho_i  = rho_water(T_i)
        E_J   += dz_eff * rho_i * (float(_sw_h(T_i)) - h_ref)

    return E_J * A_CROSS / 3.6e9  # [J] → [MWh]


def interpolate_start_profile(
    freetttes_profile: dict,
    node_heights_from_bottom: np.ndarray,
) -> np.ndarray:
    """
    Interpolate the FreeTTES initial profile onto the model grid.

    Parameters
    ----------
    freetttes_profile : dict
        FreeTTES profile: {height_from_bottom [m]: temperature [°C]}.
    node_heights_from_bottom : np.ndarray
        Heights of grid nodes above the tank bottom [m]. Index 0 = top.

    Returns
    -------
    numpy.ndarray
        Temperature profile [°C], index 0 = top.
    """
    h_pts = np.array(sorted(freetttes_profile.keys()), dtype=float)
    T_pts = np.array([freetttes_profile[h] for h in h_pts], dtype=float)
    return np.interp(node_heights_from_bottom, h_pts, T_pts)


# ── Output helpers ────────────────────────────────────────────────────────────

def print_separator(title: str = "") -> None:
    """Print a formatted separator line."""
    width = 70
    if title:
        print(f"\n{'=' * width}")
        print(f"  {title}")
        print(f"{'=' * width}")
    else:
        print("=" * width)


def phase_of(t_h: int) -> str:
    """Return the operating phase label for hour t_h (standard 60-h scenario)."""
    if t_h < T_END_CHARGE:
        return "Charging"
    if t_h < T_END_IDLE:
        return "Idle"
    return "Discharging"


# ── 0D reference model ────────────────────────────────────────────────────────

class ZeroDStorage:
    """
    Fully mixed thermal storage (0D reference model).

    Energy balance:
        m_total * cp * dT/dt = ṁ_in * cp * (T_in - T) - U*A*(T - T_amb)

    For simultaneous charging and discharging both mass flows are modelled
    as a net advective heat input:
        Q_adv = ṁ_charge    * cp * (T_charge_in    - T)
              + ṁ_discharge * cp * (T_discharge_in - T)

    Note: ThermalStorage1D does not support n_nodes=1; this class is therefore
    the correct implementation of the 0D case.
    """

    def __init__(self) -> None:
        self.rho    = 977.8     # kg/m³
        self.cp     = 4187.0    # J/(kg·K)
        self.U_loss = U_WALL    # W/(m²·K)
        self.V      = V_TANK    # m³
        self.T_amb  = T_AMB     # °C
        # Heat transfer area: lateral wall + lid + bottom
        self.A_wall = (2 * np.pi * R_INNER * H_WS
                       + 2 * np.pi * R_INNER**2)

    def initialize(self, T_init: float) -> float:
        """Initialise with a uniform temperature; returns the state."""
        return T_init

    def step(
        self,
        T: float,
        dt: float,
        m_dot_charge: float = 0.0,
        T_charge_in: float = 0.0,
        m_dot_discharge: float = 0.0,
        T_discharge_in: float = 0.0,
    ) -> tuple[float, float, float]:
        """
        Explicit Euler step.

        Returns
        -------
        T_new : float
            New storage temperature [°C].
        T_charge_out : float
            Charging circuit return temperature (= T, fully mixed) [°C].
        T_discharge_out : float
            Discharging circuit supply temperature (= T, fully mixed) [°C].
        """
        m_total = self.rho * self.V
        Q_adv  = (m_dot_charge    * self.cp * (T_charge_in    - T)
                + m_dot_discharge * self.cp * (T_discharge_in - T))
        Q_loss = self.U_loss * self.A_wall * (self.T_amb - T)
        dT     = dt * (Q_adv + Q_loss) / (m_total * self.cp)
        T_new  = T + dT
        return T_new, T, T   # outlet = storage temperature (fully mixed)
