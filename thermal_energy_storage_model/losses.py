
"""Heat-loss model implementations for the 1D thermal storage solver."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Heat-loss models
# ---------------------------------------------------------------------------

class LossModel:
    """
    Abstract base class for heat-loss models.

    A loss model computes the net heat flow through the tank wall to ambient
    for each node. It abstracts boundary conditions (temperature, U-value)
    from the physical core solver.

    Sign convention
    ---------------
    Positive return value -> heat flows FROM outside INTO storage
                             (occurs when T_ambient > T_node)
    Negative return value -> heat loss from storage to outside
                             (normal case: T_storage > T_ambient)
    """

    def Q_loss_nodes(
        self,
        T_nodes: np.ndarray,
        A_wall_nodes: np.ndarray,
        z_node_centers: np.ndarray,
    ) -> np.ndarray:
        """
        Compute wall heat flow for each node [W].

        Parameters
        ----------
        T_nodes : numpy.ndarray
            Current node temperatures [°C], shape (n,).
            Index 0 = top (hot), index N-1 = bottom (cold).
        A_wall_nodes : numpy.ndarray
            Wall area per node [m²], shape (n). Provided by the geometry
            model and includes lid/bottom for boundary nodes.
        z_node_centers : numpy.ndarray
            z-coordinate of each node center measured from storage bottom [m],
            shape (n). Index 0 has the largest z-value (top).

        Returns
        -------
        numpy.ndarray
            Heat flow from outside into storage per node [W], shape (n).
        """
        raise NotImplementedError

    def advance(
        self,
        T_nodes: np.ndarray,
        A_wall_nodes: np.ndarray,
        z_node_centers: np.ndarray,
        dt: float,
    ) -> None:
        """
        Update internal loss-model state for the next timestep.

        Default implementation: no operation (steady-state models have
        no internal state). Override in transient models such as
        ``TransientGroundLoss``.

        Parameters
        ----------
        T_nodes : numpy.ndarray
            Node temperatures at start of timestep [°C].
        A_wall_nodes : numpy.ndarray
            Wall area per node [m²].
        z_node_centers : numpy.ndarray
            z-coordinate of node centers from bottom [m].
        dt : float
            Timestep size [s].
        """


class ConstantAmbientLoss(LossModel):
    """
    Simple loss model with constant U-value and ambient temperature.

    This corresponds to the classic approach for above-ground, well-mixed
    storages or as an approximation for buried storages in homogeneous soil.

    Parameters
    ----------
    U_loss : float
        Heat transfer coefficient [W/(m²·K)]. Must be >= 0.
    T_ambient : float
        Ambient temperature for loss calculation [°C].

    Examples
    --------
        >>> loss = ConstantAmbientLoss(U_loss=0.3, T_ambient=10.0)
    """

    def __init__(self, U_loss: float, T_ambient: float) -> None:
        if U_loss < 0:
            raise ValueError(f"U_loss must be >= 0, but is {U_loss}.")
        self.U_loss = float(U_loss)
        self.T_ambient = float(T_ambient)

    def Q_loss_nodes(
        self,
        T_nodes: np.ndarray,
        A_wall_nodes: np.ndarray,
        _z_node_centers: np.ndarray,
    ) -> np.ndarray:
        """Q = U · A_wall · (T_ambient - T_node) for each node."""
        del _z_node_centers  # Depth-independent: constant ambient temperature
        return self.U_loss * A_wall_nodes * (self.T_ambient - T_nodes)

    def __repr__(self) -> str:
        return (
            f"ConstantAmbientLoss(U_loss={self.U_loss} W/(m²·K), "
            f"T_ambient={self.T_ambient} °C)"
        )


class SplitAmbientLoss(LossModel):
    """
    Loss model with separate U-values for lid and wall/bottom.

    In real PTES systems, U-values for lid insulation and for buried
    wall/bottom regions can differ significantly. This model separates both
    without changing the existing LossModel interface.

    Parameters
    ----------
    U_lid : float
        U-value for lid (top node) [W/(m²·K)]. Must be >= 0.
    U_wall : float
        U-value for walls and bottom (all remaining nodes) [W/(m²·K)]. Must be >= 0.
    T_ambient : float
        Ambient temperature for wall/bottom [°C].
    T_ambient_lid : float or None
        Ambient temperature specifically for lid [°C]. If None,
        ``T_ambient`` is also used for the lid.

    Examples
    --------
        >>> loss = SplitAmbientLoss(U_lid=0.151, U_wall=0.35, T_ambient=8.0)
        >>> loss = SplitAmbientLoss(U_lid=0.167, U_wall=0.35, T_ambient=8.0,
        ...                         T_ambient_lid=12.0)  # seasonal lid value
    """

    def __init__(
        self,
        U_lid: float,
        U_wall: float,
        T_ambient: float,
        T_ambient_lid: float | None = None,
    ) -> None:
        if U_lid < 0:
            raise ValueError(f"U_lid must be >= 0, but is {U_lid}.")
        if U_wall < 0:
            raise ValueError(f"U_wall must be >= 0, but is {U_wall}.")
        self.U_lid = float(U_lid)
        self.U_wall = float(U_wall)
        self.T_ambient = float(T_ambient)
        self.T_ambient_lid: float = (
            float(T_ambient_lid) if T_ambient_lid is not None else float(T_ambient)
        )

    def Q_loss_nodes(
        self,
        T_nodes: np.ndarray,
        A_wall_nodes: np.ndarray,
        _z_node_centers: np.ndarray,
    ) -> np.ndarray:
        """
        Q = U_lid · A[0] · (T_ambient_lid - T[0]) for top lid node (index 0),
        Q = U_wall · A[k] · (T_ambient     - T[k]) for all remaining nodes.
        """
        del _z_node_centers
        dT = self.T_ambient - T_nodes
        U_nodes = np.full(len(T_nodes), self.U_wall)
        U_nodes[0] = self.U_lid  # Index 0 = top = lid
        Q = U_nodes * A_wall_nodes * dT
        # Optionally override lid node with separate air temperature
        Q[0] = self.U_lid * A_wall_nodes[0] * (self.T_ambient_lid - T_nodes[0])
        return Q

    def __repr__(self) -> str:
        lid_str = (
            f"T_ambient_lid={self.T_ambient_lid} °C, "
            if self.T_ambient_lid != self.T_ambient
            else ""
        )
        return (
            f"SplitAmbientLoss(U_lid={self.U_lid} W/(m²·K), "
            f"U_wall={self.U_wall} W/(m²·K), "
            f"T_ambient={self.T_ambient} °C, {lid_str})"
        )


class GroundTemperatureLoss(LossModel):
    """
    Loss model for buried storages with depth-dependent ground temperature.

    Ground temperature profile is modeled by an exponential function
    describing attenuation of surface temperature with depth:

        T_ground(d) = T_deep + (T_surface - T_deep) · exp(-d / depth_decay)

    where d is depth below ground surface [m].

    Use cases
    ---------
    - Buried steel tanks (at grade or deeper)
    - PTES systems
    - Underground seasonal thermal storages

    Parameters
    ----------
    U_loss : float
        Effective wall/liner heat transfer coefficient [W/(m²·K)].
        Must be >= 0.
    burial_depth : float
        Depth of storage top below ground surface [m]. >= 0.
        burial_depth = 0 means storage roof at surface.
    T_surface : float
        Annual mean temperature at ground surface [°C].
    T_deep : float
        Asymptotic deep-ground temperature [°C].
    depth_decay : float, optional
        Characteristic decay depth [m]. Controls convergence speed
        toward T_deep. Default: 2.0 m.

    Examples
    --------
    PTES with top edge 0.5 m below grade:

        >>> loss = GroundTemperatureLoss(
        ...     U_loss=0.25,
        ...     burial_depth=0.5,
        ...     T_surface=8.0,
        ...     T_deep=11.0,
        ...     depth_decay=2.0,
        ... )
    """

    def __init__(
        self,
        U_loss: float,
        burial_depth: float,
        T_surface: float,
        T_deep: float,
        depth_decay: float = 2.0,
    ) -> None:
        if U_loss < 0:
            raise ValueError(f"U_loss must be >= 0, but is {U_loss}.")
        if burial_depth < 0:
            raise ValueError(
                f"burial_depth must be >= 0, but is {burial_depth}."
            )
        if depth_decay <= 0:
            raise ValueError(
                f"depth_decay must be > 0, but is {depth_decay}."
            )
        self.U_loss = float(U_loss)
        self.burial_depth = float(burial_depth)
        self.T_surface = float(T_surface)
        self.T_deep = float(T_deep)
        self.depth_decay = float(depth_decay)

    def T_ground_at_depth(self, d: float | np.ndarray) -> float | np.ndarray:
        """
        Ground temperature at depth d [m] below surface [°C].

        Parameters
        ----------
        d : float or numpy.ndarray
            Depth below ground surface [m].

        Returns
        -------
        float or numpy.ndarray
            Ground temperature [°C].
        """
        return self.T_deep + (self.T_surface - self.T_deep) * np.exp(
            -d / self.depth_decay
        )

    def Q_loss_nodes(
        self,
        T_nodes: np.ndarray,
        A_wall_nodes: np.ndarray,
        z_node_centers: np.ndarray,
    ) -> np.ndarray:
        """
        Q = U · A_wall · (T_ground(d_k) - T_node) for each node.

        d_k = burial_depth + (H_tank - z_k), where H_tank is total height
        and z_k is node position from storage bottom.
        """
        H_tank = float(z_node_centers[0] + z_node_centers[-1])  # approx. height
        depths = self.burial_depth + (H_tank - z_node_centers)
        T_ground = self.T_ground_at_depth(depths)
        return self.U_loss * A_wall_nodes * (T_ground - T_nodes)

    def __repr__(self) -> str:
        return (
            f"GroundTemperatureLoss("
            f"U_loss={self.U_loss} W/(m²·K), "
            f"burial_depth={self.burial_depth} m, "
            f"T_surface={self.T_surface} °C, "
            f"T_deep={self.T_deep} °C, "
            f"depth_decay={self.depth_decay} m)"
        )


class TransientGroundLoss(LossModel):
    """
    Loss model with transient 1D RC ground network.

    For each storage node, a chain of ``n_layers`` ground layers is simulated
    between tank wall and far-field boundary condition. Each layer has thermal
    capacitance and resistance; internal state (ground temperatures) is
    advanced using explicit Euler in each ``advance()`` call.

    The lid node (index 0) is treated as steady-state because it is exposed
    to air and has no soil above it.

    Physical model (per storage node i, ground layers j=1…n_layers)
    ----------------------------------------------------------------
    Thermal resistance per area:  R = d_layer / lambda_soil  [K·m²/W]
    Heat capacity per area:       C = rho_soil · cp_soil · d_layer  [J/(K·m²)]

    Energy balance of layer j (1 ≤ j ≤ n_layers − 1):
        C · dT_g[i,j]/dt = (T_prev − T_g[i,j]) / R − (T_g[i,j] − T_next) / R
        with T_prev = T_storage[i]   for j = 1
             T_prev = T_g[i,j−1]    for j > 1
             T_next = T_g[i,j+1]    for j < n_layers
             T_next = T_far         for j = n_layers

    Heat flow from ground into storage (node i, j ≥ 1):
        q_wall[i] = (T_g[i,1] − T_storage[i]) / R    [W/m²]
        Q_loss[i] = A_wall[i] · q_wall[i]             [W]

    Parameters
    ----------
    U_lid : float
        Steady-state U-value for lid [W/(m²·K)].
    T_ambient_lid : float
        Air temperature at lid [°C]. Not modeled transiently.
    lambda_soil : float
        Thermal conductivity of ground [W/(m·K)].
    rho_soil : float
        Ground density [kg/m³].
    cp_soil : float
        Specific heat capacity [J/(kg·K)].
    d_total : float
        Total thickness of modeled ground [m].
    n_layers : int
        Number of equally thick ground layers. Default: 4.
    T_far : float
        Far-field temperature (deep ground, fixed) [°C].
    T_init : float or None
        Initial temperature of all ground layers [°C]. If None,
        ``T_far`` is used.

    Examples
    --------
    Høje Taastrup PTES (clay till, λ = 2.23 W/(m·K)):

        >>> loss = TransientGroundLoss(
        ...     U_lid=0.151,
        ...     T_ambient_lid=10.0,
        ...     lambda_soil=2.23,
        ...     rho_soil=2000.0,
        ...     cp_soil=800.0,
        ...     d_total=10.0,
        ...     n_layers=4,
        ...     T_far=8.0,
        ... )
    """

    def __init__(
        self,
        U_lid: float,
        T_ambient_lid: float,
        lambda_soil: float,
        rho_soil: float,
        cp_soil: float,
        d_total: float,
        n_layers: int = 4,
        T_far: float = 8.0,
        T_init: float | None = None,
    ) -> None:
        if U_lid < 0:
            raise ValueError(f"U_lid must be >= 0, but is {U_lid}.")
        if lambda_soil <= 0:
            raise ValueError(f"lambda_soil must be > 0, but is {lambda_soil}.")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, but is {n_layers}.")
        if d_total <= 0:
            raise ValueError(f"d_total must be > 0, but is {d_total}.")

        self.U_lid = float(U_lid)
        self.T_ambient_lid = float(T_ambient_lid)
        self.lambda_soil = float(lambda_soil)
        self.rho_soil = float(rho_soil)
        self.cp_soil = float(cp_soil)
        self.d_total = float(d_total)
        self.n_layers = int(n_layers)
        self.T_far = float(T_far)
        self._T_init = float(T_init) if T_init is not None else float(T_far)

        # Layer thickness, resistance and capacity (per unit area)
        self._d_layer = self.d_total / self.n_layers
        self._R = self._d_layer / self.lambda_soil          # K·m²/W
        self._C = self.rho_soil * self.cp_soil * self._d_layer  # J/(K·m²)

        # Internal state: T_ground[n_nodes, n_layers]
        # Initialized on first advance() or Q_loss_nodes() call.
        self._T_ground: np.ndarray | None = None

    def _ensure_init(self, n_nodes: int) -> None:
        """Initialize T_ground with T_init if not initialized yet."""
        if self._T_ground is None or self._T_ground.shape[0] != n_nodes:
            self._T_ground = np.full(
                (n_nodes, self.n_layers), self._T_init, dtype=float
            )

    def Q_loss_nodes(
        self,
        T_nodes: np.ndarray,
        A_wall_nodes: np.ndarray,
        _z_node_centers: np.ndarray,
    ) -> np.ndarray:
        """
        Heat flow from ground into storage based on current
        ground-temperature field.

        Lid node (index 0): steady-state,
        Q = U_lid · A[0] · (T_ambient_lid − T[0]).
        Remaining nodes: Q = A[i] · (T_g[i,1] − T[i]) / R_1.
        """
        del _z_node_centers
        n = len(T_nodes)
        self._ensure_init(n)

        Q = np.empty(n)
        # Lid: steady-state
        Q[0] = self.U_lid * A_wall_nodes[0] * (self.T_ambient_lid - T_nodes[0])
        # Wall + bottom: transient (heat flow from first ground layer)
        Q[1:] = (
            A_wall_nodes[1:]
            * (self._T_ground[1:, 0] - T_nodes[1:])
            / self._R
        )
        return Q

    def advance(
        self,
        T_nodes: np.ndarray,
        A_wall_nodes: np.ndarray,
        _z_node_centers: np.ndarray,
        dt: float,
    ) -> None:
        """
        Explicit Euler step for all ground layers.

        Node 0 (lid) is not treated transiently.
        """
        del A_wall_nodes, _z_node_centers
        n = len(T_nodes)
        self._ensure_init(n)

        Tg = self._T_ground   # (n_nodes, n_layers), alias for readability
        R = self._R
        C = self._C
        dT = np.empty_like(Tg)

        for j in range(self.n_layers):
            # Left temperature (closer to storage)
            T_left = T_nodes if j == 0 else Tg[:, j - 1]
            # Right temperature (toward far field)
            T_right = np.full(n, self.T_far) if j == self.n_layers - 1 else Tg[:, j + 1]

            dT[:, j] = (dt / C) * ((T_left - Tg[:, j]) / R - (Tg[:, j] - T_right) / R)

        # Do not update lid node (steady-state loss model)
        dT[0, :] = 0.0

        self._T_ground += dT

    @property
    def T_ground(self) -> np.ndarray | None:
        """Current ground-temperature field [n_nodes × n_layers], °C."""
        return self._T_ground

    def __repr__(self) -> str:
        return (
            f"TransientGroundLoss("
            f"U_lid={self.U_lid} W/(m²·K), "
            f"T_ambient_lid={self.T_ambient_lid} °C, "
            f"lambda_soil={self.lambda_soil} W/(m·K), "
            f"rho={self.rho_soil} kg/m³, cp={self.cp_soil} J/(kg·K), "
            f"d_total={self.d_total} m, n_layers={self.n_layers}, "
            f"T_far={self.T_far} °C)"
        )