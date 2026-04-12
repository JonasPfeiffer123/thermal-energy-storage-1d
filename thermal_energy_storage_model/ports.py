
"""Hydraulic and heat-exchanger port dataclasses used by the model."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Port definitions
# ---------------------------------------------------------------------------

@dataclass
class Port:
    """
    Describes an inlet or outlet connection at the storage.

    Ports allow flexible placement of hydraulic connections
    at arbitrary storage heights and modeling of multiple
    independent loops (for example producer + network + solar thermal).

    Parameters
    ----------
    z : float
        Height of the connection above tank bottom [m].
        0 = bottom, height = lid.
    m_dot : float
        Mass flow rate [kg/s].
        Positive = inlet (fluid enters storage).
        Negative = outlet (fluid leaves storage).
    T_in : float, optional
        Inlet temperature [°C]. Only evaluated when m_dot > 0.
        Default: 0.0.
    label : str, optional
        Optional port label (for example ``"producer_in"``).
        For documentation only. Default: ``""``.

    Examples
    --------
    Two-loop operation: producer charges at the top, network also extracts at the top:

        >>> H = 10.0
        >>> ports = [
        ...     Port(z=H,   m_dot=+8.0, T_in=90.0, label="producer_in"),
        ...     Port(z=0.0, m_dot=-8.0,             label="producer_out"),
        ...     Port(z=H,   m_dot=-5.0,             label="network_out"),
        ...     Port(z=0.0, m_dot=+5.0, T_in=50.0, label="network_in"),
        ... ]
    """

    z: float
    """Height above tank bottom [m]."""

    m_dot: float
    """Mass flow rate [kg/s]: positive = inlet, negative = outlet."""

    T_in: float = 0.0
    """Inlet temperature [°C], relevant only for inlet flow (m_dot > 0)."""

    label: str = ""
    """Optional label of the port."""


# ---------------------------------------------------------------------------
# Heat-exchanger port (hydraulically decoupled)
# ---------------------------------------------------------------------------

@dataclass
class HeatExchangerPort:
    """
    Hydraulically decoupled heat exchanger in the storage (epsilon-NTU method).

    A ``HeatExchangerPort`` transfers heat between an external loop and the
    storage fluid without mass exchange. This is the typical case for
    indirect systems with heat exchangers (for example solar collector loop,
    external heat pump, district heating interface).

    Heat transfer is computed with the **epsilon-NTU method**. Since the
    thermal capacity of storage contents is very large compared to the external
    mass flow (C_storage -> infinity), the effectiveness expression simplifies:

        C_ext = ṁ_ext · cp_ext          [W/K]
        NTU   = UA / C_ext              [-]
        ε     = 1 − exp(−NTU)           [-]
        Q     = ε · C_ext · (T_ext_in − T̄_tank)   [W]

    Here T_bar_tank is the area-weighted mean storage temperature
    in the active HX zone [z - H_hx/2 ... z + H_hx/2].

    Parameters
    ----------
    z : float
        Height of the heat exchanger center above tank bottom [m].
    H_hx : float
        Active length of the heat exchanger [m]. Heat is distributed
        uniformly across all nodes in [z - H_hx/2, z + H_hx/2].
    UA : float
        Heat transfer coefficient of the heat exchanger [W/K].
    m_dot_ext : float
        Mass flow rate in the external loop [kg/s]. Must be >= 0.
    T_ext_in : float
        Inlet temperature of the external fluid entering the heat exchanger [°C].
    cp_ext : float, optional
        Specific heat capacity of external fluid [J/(kg*K)].
        Default: 4187 (water).
    label : str, optional
        Optional label (for example ``"solar"``). Default: ``""``.

    Examples
    --------
    Solar collector charges at the top of the storage:

        >>> hx = HeatExchangerPort(
        ...     z=9.0, H_hx=2.0, UA=5000.0,
        ...     m_dot_ext=0.8, T_ext_in=85.0,
        ...     label="Solar"
        ... )
        >>> inputs = StorageInputs(hx_ports=[hx])

    Combined with hydraulic ports:

        >>> inputs = StorageInputs(
        ...     ports=[Port(z=0.0, m_dot=-3.0, label="network_out"),
        ...            Port(z=0.0, m_dot=+3.0, T_in=50.0, label="network_in")],
        ...     hx_ports=[hx],
        ... )
    """

    z: float
    """Height of HX center above tank bottom [m]."""

    H_hx: float
    """Active length of the heat exchanger [m]."""

    UA: float
    """Heat transfer coefficient [W/K]."""

    m_dot_ext: float
    """Mass flow rate in the external loop [kg/s]."""

    T_ext_in: float
    """Inlet temperature of external fluid [°C]."""

    cp_ext: float = 4187.0
    """Specific heat capacity of external fluid [J/(kg*K)]."""

    label: str = ""
    """Optional label of the heat exchanger."""

    segmented: bool = False
    """
    Enable segmented model.

    If ``True``, the HX zone is split into individual nodes and external-fluid
    temperature is propagated node by node. This provides more accurate
    results when the heat exchanger spans the thermocline or when
    ``T_ext_out`` is needed for control.

    If ``False`` (default), the existing lumped epsilon-NTU method is used
    with the mean storage temperature of the HX zone.
    """

    flow_direction: str = "downward"
    """
    Flow direction of the external fluid through the heat exchanger.

    ``"downward"`` : External fluid enters at the top of the HX zone and
                     exits at the bottom (for example hanging coil,
                     top-fed solar charging loop).
    ``"upward"``   : External fluid enters at the bottom of the HX zone and
                     exits at the top (for example bottom-fed discharge HX).

    Only relevant when ``segmented=True``.
    """