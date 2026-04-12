
"""State and I/O container types for the 1D thermal storage model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .ports import Port

# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------

@dataclass
class StorageState:
    """
    Thermal state of the storage at a given simulation time.

    This state is exchanged between two co-simulation partners (storage model
    and network simulation). It contains all information required to continue
    the simulation.

    Parameters
    ----------
    temperatures : numpy.ndarray
        Temperature profile of the N layers [°C], shape (n_nodes,).
        Index 0 = top layer (hot), index N-1 = bottom layer (cold).
    time : float, optional
        Current simulation time [s]. Default: 0.0.

    Notes
    -----
    The temperature profile is internally stored as a numpy array. The
    `copy()` method returns a deep copy so state updates do not affect
    previous states.

    Examples
    --------
        >>> import numpy as np
        >>> state = StorageState(temperatures=np.linspace(80, 40, 20), time=3600.0)
        >>> state.temperatures[0]   # Top temperature
        80.0
        >>> state.temperatures[-1]  # Bottom temperature
        40.0
    """

    temperatures: np.ndarray
    """Temperature profile of all nodes [°C], index 0 = top."""

    time: float = 0.0
    """Current simulation time [s]."""

    T_headspace: Optional[float] = None
    """
    Current headspace temperature [°C].

    ``None`` when the headspace model is not active (``headspace=False``
    in ``StorageConfig``). Updated each time step together with the
    temperature profile.
    """

    def copy(self) -> StorageState:
        """
        Return a deep copy of the state.

        Returns
        -------
        StorageState
            Independent copy of the current state.
        """
        return StorageState(
            temperatures=self.temperatures.copy(),
            time=self.time,
            T_headspace=self.T_headspace,
        )

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the temperature profile."""
        return len(self.temperatures)

    @property
    def T_top(self) -> float:
        """Temperature of the top layer (hot side) [°C]."""
        return float(self.temperatures[0])

    @property
    def T_bottom(self) -> float:
        """Temperature of the bottom layer (cold side) [°C]."""
        return float(self.temperatures[-1])

    @property
    def T_mean(self) -> float:
        """Mean temperature across all layers [°C]."""
        return float(np.mean(self.temperatures))


@dataclass
class StorageInputs:
    """
    Boundary conditions for one timestep: list of hydraulic ports.

    Each port describes one hydraulic connection to the storage with
    a height position, a mass flow rate and (for inlets) an inlet
    temperature. Multiple ports can be active simultaneously, enabling
    operation such as producer + district heating network + solar thermal.

    Parameters
    ----------
    ports : list[Port]
        List of active ports for this timestep. An empty list
        represents idle operation (no flow).

    Notes
    -----
    **Mass balance:** The sum of all port ``m_dot`` values should
    be zero (incompressibility). The model is robust against small numerical
    mismatches, but physically incorrect balances cause temperature drift.

    **Convenience constructor:** For the classic two-loop operation
    (charge top/bottom + discharge bottom/top), use the class method
    :meth:`two_port`.

    Examples
    --------
    Charge only with default positions (top/bottom):

        >>> inputs = StorageInputs.two_port(
        ...     m_dot_charge=8.0, T_charge_in=90.0, height=10.0
        ... )

    Discharge only:

        >>> inputs = StorageInputs.two_port(
        ...     m_dot_discharge=5.0, T_discharge_in=50.0, height=10.0
        ... )

    Multiple loops at the same time:

        >>> H = 10.0
        >>> inputs = StorageInputs(ports=[
        ...     Port(z=H,   m_dot=+8.0, T_in=90.0, label="producer_in"),
        ...     Port(z=0.0, m_dot=-8.0,             label="producer_out"),
        ...     Port(z=H,   m_dot=-5.0,             label="network_out"),
        ...     Port(z=0.0, m_dot=+5.0, T_in=50.0, label="network_in"),
        ... ])
    """

    ports: list = field(default_factory=list)
    """List of active ports [Port]. Empty = no flow (idle state)."""

    hx_ports: list = field(default_factory=list)
    """List of heat exchanger ports [HeatExchangerPort]. Empty = no active HX."""

    @classmethod
    def two_port(
        cls,
        m_dot_charge: float = 0.0,
        T_charge_in: float = 0.0,
        m_dot_discharge: float = 0.0,
        T_discharge_in: float = 0.0,
        height: float = 0.0,
        z_charge_in: Optional[float] = None,
        z_charge_out: Optional[float] = None,
        z_discharge_in: Optional[float] = None,
        z_discharge_out: Optional[float] = None,
    ) -> "StorageInputs":
        """
        Convenience constructor for classic two-loop operation.

        Creates ``StorageInputs`` with up to four ports for the standard
        operating mode: charging loop (inlet top, outlet bottom) and
        discharging loop (inlet bottom, outlet top).
        Inactive loops (m_dot = 0) do not create ports.

        Parameters
        ----------
        m_dot_charge : float, optional
            Charging mass flow rate [kg/s]. Default: 0.0.
        T_charge_in : float, optional
            Charging inlet temperature [°C]. Default: 0.0.
        m_dot_discharge : float, optional
            Discharging mass flow rate [kg/s]. Default: 0.0.
        T_discharge_in : float, optional
            Discharging inlet temperature [°C]. Default: 0.0.
        height : float
            Total storage height [m]. Used for default positions
            (very top / very bottom).
        z_charge_in : float, optional
            Charging inlet height [m]. Default: ``height`` (lid).
        z_charge_out : float, optional
            Charging outlet height [m]. Default: 0.0 (bottom).
        z_discharge_in : float, optional
            Discharging inlet height [m]. Default: 0.0 (bottom).
        z_discharge_out : float, optional
            Discharging outlet height [m]. Default: ``height`` (lid).

        Returns
        -------
        StorageInputs

        Examples
        --------
            >>> inputs = StorageInputs.two_port(
            ...     m_dot_charge=8.0, T_charge_in=90.0,
            ...     m_dot_discharge=5.0, T_discharge_in=50.0,
            ...     height=10.0,
            ... )
        """
        z_ci  = z_charge_in    if z_charge_in    is not None else height
        z_co  = z_charge_out   if z_charge_out   is not None else 0.0
        z_di  = z_discharge_in if z_discharge_in is not None else 0.0
        z_do  = z_discharge_out if z_discharge_out is not None else height

        ports: list = []
        if m_dot_charge > 0.0:
            ports.append(Port(z=z_ci, m_dot=+m_dot_charge, T_in=T_charge_in,
                              label="charge_in"))
            ports.append(Port(z=z_co, m_dot=-m_dot_charge,
                              label="charge_out"))
        if m_dot_discharge > 0.0:
            ports.append(Port(z=z_di, m_dot=+m_dot_discharge, T_in=T_discharge_in,
                              label="discharge_in"))
            ports.append(Port(z=z_do, m_dot=-m_dot_discharge,
                              label="discharge_out"))
        return cls(ports=ports)


@dataclass
class StorageOutputs:
    """
    Simulation results produced after one timestep.

    Contains temperatures at each port (= outlet temperatures for
    outlet ports), wall heat loss, and the new storage state.
    These values are returned to the co-simulation environment.

    Parameters
    ----------
    port_temperatures : list[float]
                Storage node temperature at each port height [°C].
                Order matches the port list in ``StorageInputs.ports``.

                - For **outlet ports** (m_dot < 0): this is the actual
                    fluid outlet temperature (upwind from storage).
                - For **inlet ports** (m_dot > 0): this is the local
                    storage temperature at the inlet point (useful to compute
                    heat flow Q = ṁ · cp · (T_in − T_node)).

    Q_loss : float
                Total heat loss from storage to ambient [W].
                Positive = outward loss (storage cools down).
    state : StorageState
                New storage state at the end of the timestep.

    Notes
    -----
    Heat flow of an individual port can be computed from
    ``port_temperatures``:

        Inlet (m_dot > 0):  Q = m_dot · cp · (T_in   − T_node)
        Outlet (m_dot < 0): Q = m_dot · cp · (T_node − T_sink)

    For the classic two-loop case, ``port_temperatures`` follows the
    port order created by :meth:`StorageInputs.two_port`:
    ``[T@charge_in, T@charge_out, T@discharge_in, T@discharge_out]``.
    """

    port_temperatures: list
    """Node temperatures at port positions [°C], one value per port."""

    Q_loss: float
    """Heat loss to ambient [W]."""

    state: StorageState
    """Storage state at the end of the timestep."""

    hx_outlet_temperatures: list = field(default_factory=list)
    """Outlet temperatures of external fluids for each HX port [°C].
    Order matches ``StorageInputs.hx_ports``."""

    T_headspace: Optional[float] = None
    """Headspace temperature at the end of the timestep [°C].
    ``None`` when the headspace model is not active."""