"""Core solver implementation for the 1D stratified thermal storage model."""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np

from .config import StorageConfig
from .diffusors import DiffusorModel, PointDiffusor
from .fluids import ConstantFluidProperties, FluidProperties
from .geometry import CylinderGeometry, GeometryModel
from .losses import ConstantAmbientLoss, LossModel
from .presets import StoragePresets
from .state import StorageInputs, StorageOutputs, StorageState
from .ports import HeatExchangerPort, Port

# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

# Numerical epsilon to avoid division by zero in the TVD gradient ratio
# (only active for |ΔT| < 1e-12 K, i.e. practically never)
_TVD_EPS: float = 1e-12

class ThermalStorage1D:
    """
    1D model of a stratified hot-water thermal energy storage.

    This model represents a cylindrical hot-water storage as a
    one-dimensional layer stack. It is designed for use in
    co-simulation environments with district-heating network solvers:
    each timestep is computed individually, state is exchanged, and
    network simulation outputs (mass flow rates, inlet temperatures)
    are applied directly.

    The model supports:

        - **Simultaneous charging and discharging** via two separate
            hydraulic loops (for example CHP feed-in + network supply)
        - **Thermal stratification** through upwind discretization of the
            advection term, preserving thermocline structure
        - **Heat losses** through tank wall (shell + bottom + lid)
        - **Heat conduction / diffusion** between neighboring layers with
            effective thermal conductivity (including turbulent mixing)
        - **CFL check** for numerical stability monitoring

    Physical model
    --------------
    Energy balance for node i (i = 0 top, i = N-1 bottom):

        m_k · c_p · dT_i/dt = Q_adv,i + Q_cond,i + Q_loss,i

    Advection term (upwind, net flow direction):

        m_net = m_charge - m_discharge

        Node 0 (top), m_net ≥ 0:
            Q_adv,0 = m_charge · c_p · (T_charge_in - T_0)

        Node 0 (top), m_net < 0:
            Q_adv,0 = m_charge · c_p · T_charge_in
                    + |m_net| · c_p · T_1
                    - m_discharge · c_p · T_0

        Interior node i, m_net > 0 (downward):
            Q_adv,i = m_net · c_p · (T_{i-1} - T_i)

        Interior node i, m_net < 0 (upward):
            Q_adv,i = |m_net| · c_p · (T_{i+1} - T_i)

        Node N-1 (bottom), m_net ≥ 0:
            Q_adv,N-1 = m_net · c_p · T_{N-2}
                      + m_discharge · c_p · T_discharge_in
                      - m_charge · c_p · T_{N-1}

        Node N-1 (bottom), m_net < 0:
            Q_adv,N-1 = m_discharge · c_p · (T_discharge_in - T_{N-1})

    Conduction term:

        Q_cond,i = K_cond · (T_{i-1} - T_i) + K_cond · (T_{i+1} - T_i)
        K_cond = λ_eff · A / Δz       [W/K]
        λ_eff = lambda_eff_factor · lambda_fluid

    Loss term:

        Q_loss,i = U_loss · A_wall,i · (T_ambient - T_i)

    Time integration (explicit Euler):

        T_i(t + Δt) = T_i(t) + Δt · dT_i/dt

    Parameters
    ----------
    config : StorageConfig
        Storage configuration parameters.

    Raises
    ------
    ValueError
        If configuration parameters are physically invalid.

    Examples
    --------
    Simple initialization and one timestep:

        >>> config = StorageConfig(volume=100.0, height=10.0, n_nodes=20)
        >>> storage = ThermalStorage1D(config)
        >>> state = storage.initialize(T_init=70.0)
        >>> inputs = StorageInputs(
        ...     m_dot_charge=6.0, T_charge_in=90.0,
        ...     m_dot_discharge=4.0, T_discharge_in=50.0
        ... )
        >>> outputs = storage.step(state, dt=60.0, inputs=inputs)
        >>> print(f"Vorlauf: {outputs.T_discharge_out:.1f} °C")
        >>> print(f"Charge return: {outputs.T_charge_out:.1f} °C")

    Querying the optimal timestep size:

        >>> dt_max = storage.max_stable_dt(m_dot_max=10.0)
        >>> print(f"Recommended max timestep size: {dt_max:.0f} s")
    """

    def __init__(self, config: StorageConfig) -> None:
        self.config = config
        self._validate_config()
        self._precompute_geometry()

    @classmethod
    def from_preset(cls, preset: str, **params) -> "ThermalStorage1D":
        """
        Create a storage instance from a predefined type.

        Predefined types encapsulate typical geometry and loss-model
        combinations for common storage types. The returned instance
        can be used directly without manually filling `StorageConfig`.

        Parameters
        ----------
        preset : str
            Storage type name. Available presets:

            ``"steel_tank_aboveground"``
                Above-ground steel tank (cylindrical, constant ambient temperature).
                Required parameters: *volume*, *height*.

            ``"steel_tank_buried"``
                Buried steel tank (cylindrical, ground-temperature profile).
                Required parameters: *volume*, *height*.

            ``"ptes"``
                Pit thermal energy storage
                (truncated cone, ground-temperature profile).
                Required parameters: *r_bottom*, *r_top*, *height*.

        **params
            Type-specific parameters. Forwarded directly to the
            matching `StoragePresets` method. Optional parameters
            have physically reasonable defaults.

        Returns
        -------
        ThermalStorage1D
            Fully configured storage instance.

        Raises
        ------
        ValueError
            If *preset* is unknown.

        Examples
        --------
        Freistehender Stahltank:

            >>> tes = ThermalStorage1D.from_preset(
            ...     "steel_tank_aboveground",
            ...     volume=500.0, height=15.0,
            ... )

        Erdbeckenspeicher:

            >>> tes = ThermalStorage1D.from_preset(
            ...     "ptes",
            ...     r_bottom=40.0, r_top=55.0, height=15.0,
            ...     burial_depth=0.5,
            ... )
        """
        preset_map = {
            "steel_tank_aboveground": StoragePresets.steel_tank_aboveground,
            "steel_tank_buried":      StoragePresets.steel_tank_buried,
            "ptes":                   StoragePresets.ptes,
        }
        if preset not in preset_map:
            raise ValueError(
                f"Unknown preset: '{preset}'. "
                f"Available presets: {sorted(preset_map)}."
            )
        config = preset_map[preset](**params)
        return cls(config)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validate configuration parameters for physical plausibility."""
        cfg = self.config
        if cfg.volume <= 0:
            raise ValueError(f"volume must be positive, but is {cfg.volume}.")
        if cfg.height <= 0:
            raise ValueError(f"height must be positive, but is {cfg.height}.")
        if cfg.n_nodes < 2:
            raise ValueError(
                f"n_nodes must be at least 2, but is {cfg.n_nodes}."
            )
        if cfg.rho <= 0:
            raise ValueError(f"rho must be positive, but is {cfg.rho}.")
        if cfg.cp <= 0:
            raise ValueError(f"cp must be positive, but is {cfg.cp}.")
        if cfg.lambda_fluid <= 0:
            raise ValueError(
                f"lambda_fluid must be positive, but is {cfg.lambda_fluid}."
            )
        if cfg.lambda_eff_factor < 1.0:
            raise ValueError(
                f"lambda_eff_factor must be >= 1, but is {cfg.lambda_eff_factor}."
            )
        if cfg.U_loss < 0:
            raise ValueError(
                f"U_loss must be >= 0, but is {cfg.U_loss}."
            )
        if cfg.geometry is not None:
            if not isinstance(cfg.geometry, GeometryModel):
                raise ValueError(
                    "geometry must be an instance of GeometryModel."
                )
        if cfg.loss_model is not None:
            if not isinstance(cfg.loss_model, LossModel):
                raise ValueError(
                    "loss_model must be an instance of LossModel."
                )
        if cfg.fluid is not None:
            if not isinstance(cfg.fluid, FluidProperties):
                raise ValueError(
                    "fluid must be an instance of FluidProperties."
                )
        if cfg.advection_scheme not in ("upwind", "tvd"):
            raise ValueError(
                f"advection_scheme must be 'upwind' or 'tvd', "
                f"but is '{cfg.advection_scheme}'."
            )
        if cfg.solver not in ("explicit", "implicit"):
            raise ValueError(
                f"solver must be 'explicit' or 'implicit', "
                f"but is '{cfg.solver}'."
            )
        if cfg.diffusor_model is not None:
            if not isinstance(cfg.diffusor_model, DiffusorModel):
                raise ValueError(
                    "diffusor_model must be an instance of DiffusorModel."
                )

    def _precompute_geometry(self) -> None:
        """
        Precompute geometric and thermal quantities.

        All derived quantities are computed once during initialization.
        If config.geometry is set, that geometry model is used;
        otherwise a CylinderGeometry is created automatically from
        volume and height (backward compatibility).
        """
        cfg = self.config
        n = cfg.n_nodes

        # Determine geometry model
        if cfg.geometry is not None:
            geom = cfg.geometry
        else:
            geom = CylinderGeometry.from_volume(cfg.volume, cfg.height)
        self._geom: GeometryModel = geom

        self.n: int = n
        self.dz: float = geom.height / n    # uniform node height [m]

        # Per-node geometry
        self._A_cross_nodes: np.ndarray = geom.A_cross_nodes(n)
        self._V_nodes: np.ndarray = geom.V_nodes(n)
        self._m_nodes: np.ndarray = cfg.rho * self._V_nodes
        self._C_nodes: np.ndarray = self._m_nodes * cfg.cp

        # Representative cross-section for CFL calculation: smallest area
        # (conservative: highest flow velocity)
        self.A_cross: float = float(np.min(self._A_cross_nodes))

        # Wall area per node [m²]
        self.A_wall: np.ndarray = geom.A_wall_nodes(n)

        # Effective thermal conductivity
        self._lambda_eff: float = cfg.lambda_fluid * cfg.lambda_eff_factor

        # Conduction coefficients at n-1 interfaces between nodes [W/K]
        #   K_iface[k] = λ_eff · A_iface[k] / dz
        #   A_iface[k] = mean cross-section between node k and k+1
        A_iface = 0.5 * (self._A_cross_nodes[:-1] + self._A_cross_nodes[1:])
        self._K_cond_iface: np.ndarray = self._lambda_eff * A_iface / self.dz

        # Backward-compatible scalar quantities (representative of mid node)
        self.m_node: float = float(np.mean(self._m_nodes))
        self._C_node: float = float(np.mean(self._C_nodes))
        self.K_cond: float = float(np.mean(self._K_cond_iface))
        self._radius: float = float(np.sqrt(geom.volume / (np.pi * geom.height)))

        # z-coordinates of node centers measured from bottom [m]
        # Index 0 = top (z near geom.height), index N-1 = bottom (z near 0)
        self._z_nodes: np.ndarray = np.linspace(
            geom.height - self.dz / 2.0,
            self.dz / 2.0,
            n,
        )

        # Determine loss model
        if cfg.loss_model is not None:
            self._loss_model: LossModel = cfg.loss_model
        else:
            self._loss_model = ConstantAmbientLoss(cfg.U_loss, cfg.T_ambient)

        # Determine fluid-property model
        if cfg.fluid is not None:
            self._fluid: FluidProperties = cfg.fluid
        else:
            self._fluid = ConstantFluidProperties(cfg.rho, cfg.cp, cfg.lambda_fluid)

        # Determine diffusor mixing model
        if cfg.diffusor_model is not None:
            self._diffusor: DiffusorModel = cfg.diffusor_model
        else:
            self._diffusor = PointDiffusor()

    def _compute_wall_areas(self) -> np.ndarray:
        """
        Compute wall area for each node.

        Delegates to the geometry model. No longer called internally by
        _precompute_geometry (kept for backward compatibility).

        Returns
        -------
        numpy.ndarray
            Wall areas [m²], shape (n_nodes,).
        """
        return self._geom.A_wall_nodes(self.n)

    # ------------------------------------------------------------------
    # Public initialization method
    # ------------------------------------------------------------------

    def initialize(
        self,
        T_init: Union[float, np.ndarray],
        time: float = 0.0,
    ) -> StorageState:
        """
        Create the initial storage state.

        Parameters
        ----------
        T_init : float or array-like
                        Initial temperature [°C].

                        - Scalar: uniform initial temperature for all nodes.
                        - Array of length n_nodes: custom temperature profile.
                            Index 0 = top (hot), index N-1 = bottom (cold).
        time : float, optional
                        Start time [s]. Default: 0.0.

        Returns
        -------
        StorageState
            Initial storage state.

        Raises
        ------
        ValueError
            If `T_init` has wrong array length.

        Examples
        --------
        Uniform initial profile:

            >>> state = storage.initialize(T_init=70.0)

        Linear stratified profile (80 °C top, 40 °C bottom):

            >>> import numpy as np
            >>> T_profile = np.linspace(80.0, 40.0, config.n_nodes)
            >>> state = storage.initialize(T_init=T_profile)
        """
        if np.isscalar(T_init):
            temps = np.full(self.n, float(T_init))
        else:
            temps = np.asarray(T_init, dtype=float).copy()
            if temps.shape != (self.n,):
                raise ValueError(
                    f"T_init array must have length n_nodes={self.n}, "
                    f"but has shape {temps.shape}."
                )

        # Optional bottom-zone correction: linear gradient from T_ground (z=0)
        # to regular profile value at z=z_ground_layer.
        # Models heat exchange with ground in diffusor zone.
        if self.config.T_ground is not None:
            T_g = float(self.config.T_ground)
            z_layer = float(self.config.z_ground_layer)
            for i in range(self.n):
                z_i = float(self._z_nodes[i])
                if z_i < z_layer:
                    frac = z_i / z_layer  # 0 = bottom, 1 = upper boundary
                    temps[i] = T_g * (1.0 - frac) + temps[i] * frac

        T_hs = self.config.T_headspace_init if self.config.headspace else None
        return StorageState(temperatures=temps, time=float(time), T_headspace=T_hs)

    # ------------------------------------------------------------------
    # Information methods
    # ------------------------------------------------------------------

    def get_stored_energy(
        self,
        state: StorageState,
        T_ref: float = 0.0,
    ) -> float:
        """
        Compute thermal energy stored in the tank.

        Parameters
        ----------
        state : StorageState
            Current storage state.
        T_ref : float, optional
            Reference temperature for energy calculation [°C]. Default: 0.0.

        Returns
        -------
        float
            Stored energy [J] = m_total · c_p · (T_mean - T_ref).

        Notes
        -----
        For relative usable heat, T_ref is typically network return
        temperature (for example 50 °C).

        For temperature-dependent fluid properties:
        E = Σ_k ρ(T_k) · V_k · cp(T_k) · (T_k − T_ref) is computed.
        """
        T = state.temperatures
        rho_T = np.asarray(self._fluid.rho(T), dtype=float)
        cp_T  = np.asarray(self._fluid.cp(T),  dtype=float)
        if rho_T.ndim == 0:
            rho_T = np.full(self.n, float(rho_T))
        if cp_T.ndim == 0:
            cp_T = np.full(self.n, float(cp_T))
        C_nodes = rho_T * self._V_nodes * cp_T
        return float(np.sum(C_nodes * (T - T_ref)))

    def get_soc(
        self,
        state: StorageState,
        T_min: float,
        T_max: float,
    ) -> float:
        """
        Compute state of charge (SOC) of the storage.

        Parameters
        ----------
        state : StorageState
            Current storage state.
        T_min : float
            Temperature at full discharge [°C].
        T_max : float
            Temperature at full charge [°C].

        Returns
        -------
        float
            State of charge between 0 (empty) and 1 (full).

        Raises
        ------
        ValueError
            If T_max <= T_min.
        """
        if T_max <= T_min:
            raise ValueError(
                f"T_max ({T_max}) must be greater than T_min ({T_min})."
            )
        E_current = self.get_stored_energy(state, T_ref=T_min)
        E_max = float(np.sum(self._C_nodes)) * (T_max - T_min)
        return float(np.clip(E_current / E_max, 0.0, 1.0))

    def max_stable_dt(self, m_dot_max: float) -> float:
        """
        Compute the maximum stable timestep (CFL condition).

        Parameters
        ----------
        m_dot_max : float
            Maximum expected mass flow rate (from one loop or
            their net difference) [kg/s].

        Returns
        -------
        float
            Maximum stable timestep [s].
            Infinity if m_dot_max = 0.

        Notes
        -----
        CFL condition:

            CFL = v · Δt / Δz ≤ 1
            v = ṁ / (ρ · A)

        A safety factor of 0.9 is applied.
        """
        if m_dot_max <= 0.0:
            return float("inf")
        v = m_dot_max / (self.config.rho * self.A_cross)
        return 0.9 * self.dz / v

    def check_cfl(self, dt: float, m_dot_max: float) -> bool:
        """
        Check CFL stability condition.

        Parameters
        ----------
        dt : float
            Used timestep [s].
        m_dot_max : float
            Maximum mass flow rate [kg/s].

        Returns
        -------
        bool
            True if CFL <= 1 (numerically stable).
        """
        if m_dot_max <= 0.0:
            return True
        v = m_dot_max / (self.config.rho * self.A_cross)
        return (v * dt / self.dz) <= 1.0

    # ------------------------------------------------------------------
    # Core: timestep computation
    # ------------------------------------------------------------------

    def step(
        self,
        state: StorageState,
        dt: float,
        inputs: StorageInputs,
    ) -> StorageOutputs:
        """
        Perform one storage simulation timestep.

        This is the model core method. It implements an
        explicit Euler step with upwind discretization and is intended
        for direct coupling with a network simulation:

          1. Outlet temperatures are evaluated from current state
              (returned to network simulation).
          2. Energy balances of all nodes are computed.
          3. New state at timestep end is determined.

        Parameters
        ----------
        state : StorageState
            Current storage state (temperature profile + timestamp).
        dt : float
            Timestep size [s]. Must satisfy CFL condition.
            Recommended: <= max_stable_dt(m_dot_max).
        inputs : StorageInputs
            Boundary conditions for this timestep:
            mass flow rates and inlet temperatures of all loops.

        Returns
        -------
        StorageOutputs
            Contains:
            - ``port_temperatures``: node temperatures at each port [°C]
            - ``Q_loss``: wall heat loss [W]
            - ``state``: new storage state

        RuntimeWarning
            If the CFL condition is violated (numerically unstable).

        Notes
        -----
        **Outlet temperatures:** Upwind principle - outlet temperature
        of an outlet port equals the node temperature at the port height
        at the beginning of the timestep.

        **Heat flow** at port i (for external calculation):

            Inlet (m_dot > 0): Q = m_dot · cp · (T_in − port_temperatures[i])
            Outlet (m_dot < 0): Q = |m_dot| · cp · port_temperatures[i]

        **Mass balance:** Sum of all port ``m_dot`` values must be zero.
        The model does not check this explicitly; incorrect balance leads to
        physically inconsistent results.

        Examples
        --------
        Single timestep in simultaneous operation:

            >>> state = storage.initialize(80.0)
            >>> inputs = StorageInputs.two_port(
            ...     m_dot_charge=5.0, T_charge_in=90.0,
            ...     m_dot_discharge=3.0, T_discharge_in=50.0,
            ...     height=config.height,
            ... )
            >>> out = storage.step(state, dt=60.0, inputs=inputs)
            >>> new_state = out.state  # pass to next step

        Time loop for co-simulation:

            >>> state = storage.initialize(70.0)
            >>> for t in range(0, 3600, 60):
            ...     inputs = get_network_inputs(t)        # from network solver
            ...     outputs = storage.step(state, dt=60.0, inputs=inputs)
            ...     state = outputs.state
            ...     send_to_network(outputs)              # to network solver
        """
        cfg = self.config
        T = state.temperatures.copy()
        n = self.n

        # --- Port-based mass flows and source terms ---
        F    = self._compute_inter_node_fluxes(inputs.ports)
        S, T_src = self._compute_source_terms(inputs.ports)

        # --- Temperature-dependent fluid properties for this timestep ---
        rho_T = np.asarray(self._fluid.rho(T), dtype=float)
        cp_T  = np.asarray(self._fluid.cp(T),  dtype=float)
        if rho_T.ndim == 0:
            rho_T = np.full(n, float(rho_T))
        if cp_T.ndim == 0:
            cp_T = np.full(n, float(cp_T))

        # Local heat capacity per node [J/K]
        C_nodes = rho_T * self._V_nodes * cp_T

        # Thermal conductivity at profile mean; effective coefficients.
        # Note: lambda_fluid is evaluated at mean profile temperature (approx.),
        # not node-wise. For water (0-100 °C) the error is usually < 10 %; for
        # strongly temperature-dependent fluids consider node-wise evaluation.
        T_mean = float(T.mean())
        lambda_eff_T = float(self._fluid.lambda_fluid(T_mean)) * cfg.lambda_eff_factor
        A_iface = 0.5 * (self._A_cross_nodes[:-1] + self._A_cross_nodes[1:])
        K_cond_iface_T = lambda_eff_T * A_iface / self.dz

        # Representative cp for advection term (profile mean)
        cp_mean = float(np.mean(cp_T))

        # --- CFL check: maximum absolute interface flow ---
        rho_min = float(rho_T.min())
        m_for_cfl = float(np.max(np.abs(F)))
        if cfg.solver == "explicit" and m_for_cfl > 0.0:
            cfl_val = (m_for_cfl / (rho_min * self.A_cross)) * dt / self.dz
            if cfl_val > 1.0:
                warnings.warn(
                    f"CFL condition violated (CFL = {cfl_val:.2f} > 1). "
                    f"dt={dt} s, ṁ_max={m_for_cfl:.2f} kg/s. "
                    "Solution may be numerically unstable.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # --- Port outlet temperatures (from current state) ---
        # For outlet ports this is the mass-flow-weighted node temperature
        # in the diffusor zone (for PointDiffusor: exactly one node).
        port_temperatures = [
            float(sum(w * T[k]
                      for k, w in self._diffusor.node_weights(p, self._z_nodes)))
            for p in inputs.ports
        ]

        # --- Heat-exchanger source terms (epsilon-NTU, explicitly linearized) ---
        Q_hx_nodes, hx_outlet_temperatures = self._compute_hx_source_terms(
            T, inputs.hx_ports
        )

        # --- Headspace heat exchange (if enabled) ---
        T_hs_new = state.T_headspace
        if cfg.headspace and state.T_headspace is not None:
            Q_hs, T_hs_new = self._compute_headspace_exchange(
                state.T_headspace, T[0], dt
            )
            Q_hx_nodes[0] += Q_hs   # top node receives headspace heat

        if cfg.solver == "implicit":
            # --- Fully implicit Euler step (TDMA), optional deferred TVD ---
            Q_tvd_explicit = None
            if cfg.advection_scheme == "tvd" and n >= 3:
                F_int = F[1:-1]
                Q_tvd_explicit = self._compute_tvd_correction_ports(
                    T, F_int, cp_mean, dt
                )
            T_new = self._step_implicit(
                T, dt, F, S, T_src, C_nodes, K_cond_iface_T, cp_mean, Q_hx_nodes,
                Q_tvd_explicit=Q_tvd_explicit,
            )
        else:
            # --- Compute temperature-rate terms ---
            dT_dt = np.zeros(n)

            # 1) Advective heat transport with variable port flow
            dT_dt += self._compute_advection_ports(
                T, F, S, T_src, cp=cp_mean, dt=dt
            ) / C_nodes

            # 2) Conductive heat transport
            dT_dt += self._compute_conduction(T, K_cond_iface=K_cond_iface_T) / C_nodes

            # 3) Heat losses to ambient
            dT_dt += self._compute_losses(T) / C_nodes

            # 4) Heat exchangers (epsilon-NTU, explicit)
            dT_dt += Q_hx_nodes / C_nodes

            # --- Explicit Euler integration ---
            T_new = T + dt * dT_dt

        # --- Buoyancy correction: convective adjustment ---
        if self.config.buoyancy:
            T_new = self._convective_adjustment(T_new, self._m_nodes)

        # --- Update transient state of loss model ---
        self._loss_model.advance(T, self.A_wall, self._z_nodes, dt)

        Q_loss = float(-np.sum(self._compute_losses(T)))
        new_state = StorageState(
            temperatures=T_new,
            time=state.time + dt,
            T_headspace=T_hs_new,
        )

        return StorageOutputs(
            port_temperatures=port_temperatures,
            Q_loss=Q_loss,
            state=new_state,
            hx_outlet_temperatures=hx_outlet_temperatures,
            T_headspace=T_hs_new,
        )

    # ------------------------------------------------------------------
    # Implicit solver (TDMA)
    # ------------------------------------------------------------------

    def _step_implicit(
        self,
        T: np.ndarray,
        dt: float,
        F: np.ndarray,
        S: np.ndarray,
        T_src: np.ndarray,
        C_nodes: np.ndarray,
        K_cond_iface: np.ndarray,
        cp: float,
        Q_hx_nodes: Optional[np.ndarray] = None,
        Q_tvd_explicit: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fully implicit Euler step with upwind advection (TDMA), optional
        deferred TVD anti-diffusion correction.

        Advection and conduction are evaluated at new time t+dt
        (unconditionally stable, no CFL restriction). Heat losses and
        heat-exchanger source terms are explicitly linearized (semi-implicit):
        they use T_old as operating point and contribute only to the right-hand
        side. This yields the tridiagonal linear system

            A · T_new = b

        with tridiagonal structure, solved efficiently by the
        Thomas algorithm (TDMA) in O(N).

        TVD correction is applied as a **deferred explicit source term**
        (`Q_tvd_explicit`) added to the right-hand side. Computed by the caller
        from T_old via :meth:`_compute_tvd_correction_ports`, it carries the
        usual `(1 − CFL)` weighting — so when the user picks a timestep so
        large that CFL ≥ 1 at all faces, the correction vanishes and the
        scheme degenerates to pure implicit upwind (unconditionally stable).
        At moderate CFL the correction reduces the upwind smearing of
        thermocline fronts.

        Parameters
        ----------
        T : np.ndarray
            Temperature profile at current time [°C], shape (n,).
        dt : float
            Timestep size [s].
        F : np.ndarray
            Inter-node flow vector [kg/s], shape (n+1). F[0]=F[n]=0.
        S : np.ndarray
            Net source term per node [kg/s], shape (n,).
        T_src : np.ndarray
            Mass-flow-weighted inlet temperature per node [°C], shape (n,).
        C_nodes : np.ndarray
            Thermal node capacity [J/K], shape (n,).
        K_cond_iface : np.ndarray
            Conduction coefficient at each internal interface [W/K],
            shape (n-1,). K_cond_iface[k] lies between nodes k and k+1.
        cp : float
            Representative cp value for advection term [J/(kg·K)].
        Q_hx_nodes : np.ndarray, optional
            Heat-exchanger source terms per node [W], shape (n). Added to
            right-hand side (explicitly linearized). Default: None (= 0).
        Q_tvd_explicit : np.ndarray, optional
            Deferred TVD anti-diffusion correction per node [W], shape (n).
            Caller computes it from T_old (matching the explicit-path
            convention) and passes it in when ``advection_scheme == "tvd"``.
            Default: None (= 0, pure implicit upwind).
        """
        n = self.n
        K = K_cond_iface        # (n-1,)
        C_dt = C_nodes / dt     # thermal capacity / dt [W/K]

        # Diagonals: d[i]*T_new[i] + a[i]*T_new[i-1] + c[i]*T_new[i+1] = b[i]
        a = np.zeros(n)         # lower off-diagonal (coeff. of T_new[i-1])
        d = np.zeros(n)         # main diagonal
        c = np.zeros(n)         # upper off-diagonal (coeff. of T_new[i+1])
        b = np.zeros(n)

        # --- Base term: time derivative ---
        d += C_dt
        b += C_dt * T

        # --- Heat losses (explicitly linearized -> RHS) ---
        b += self._compute_losses(T)

        # --- Heat exchangers (explicitly linearized -> RHS) ---
        if Q_hx_nodes is not None:
            b += Q_hx_nodes

        # --- Deferred TVD anti-diffusion (explicit, -> RHS) ---
        # Conservatively distributed in _compute_tvd_correction_ports so
        # sum(Q_tvd) = 0 by construction → energy-conserving correction.
        if Q_tvd_explicit is not None:
            b += Q_tvd_explicit

        # --- Conduction (implicit) ---
        # Interface k (between nodes k and k+1), k = 0..n-2:
        #   Node k:     d[k]   += K[k],  c[k]   -= K[k]
        #   Node k+1:   d[k+1] += K[k],  a[k+1] -= K[k]
        d[:-1]  += K
        c[:-1]  -= K
        d[1:]   += K
        a[1:]   -= K

        # --- Advection (implicit, upwind) ---
        # Interface j (between nodes j-1 and j), j = 1..n-1,
        # corresponds to index k = j-1 in F[k+1]:
        #   F_val > 0 (downward): upwind = T_new[k]
        #     -> d[k]   += F_val*cp   (node k: heat leaves downward)
        #        a[k+1] -= F_val*cp   (node k+1: receives heat from T_new[k])
        #   F_val < 0 (upward): upwind = T_new[k+1]
        #     -> c[k]   += F_val*cp   (node k: receives heat from T_new[k+1]; F_val<0 -> c[k] decreases)
        #        d[k+1] -= F_val*cp   (node k+1: heat leaves upward; F_val<0 -> d[k+1] increases)
        for k in range(n - 1):
            F_val = float(F[k + 1]) * cp
            if F_val >= 0.0:
                d[k]     += F_val
                a[k + 1] -= F_val
            else:
                c[k]     += F_val
                d[k + 1] -= F_val

        # --- Port source-term energy (semi-implicit) ---
        # Inlet (m_dot > 0): known inlet temperature -> RHS
        # Outlet (m_dot < 0): outlet with current node temperature -> implicit
        d -= cp * np.minimum(S, 0.0)             # outlet: implicit
        b += cp * np.maximum(S, 0.0) * T_src     # inlet: RHS

        return self._solve_tdma(a, d, c, b)

    @staticmethod
    def _solve_tdma(
        a: np.ndarray,
        d: np.ndarray,
        c: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Solve a tridiagonal linear system A·x = b (Thomas algorithm).

        Runtime O(N), low memory overhead, numerically stable for
        diagonally dominant systems.

        Parameters
        ----------
        a : np.ndarray
            Lower diagonal (coefficient of x[i-1] in equation i).
            a[0] is not used.
        d : np.ndarray
            Main diagonal (coefficient of x[i] in equation i).
        c : np.ndarray
            Upper diagonal (coefficient of x[i+1] in equation i).
            c[-1] is not used.
        b : np.ndarray
            Right-hand side.

        Returns
        -------
        np.ndarray
            Solution vector x.
        """
        n = len(d)
        d_ = d.copy()
        b_ = b.copy()

        # Forward elimination
        for i in range(1, n):
            w = a[i] / d_[i - 1]
            d_[i] -= w * c[i - 1]
            b_[i] -= w * b_[i - 1]

        # Back substitution
        x = np.empty(n)
        x[-1] = b_[-1] / d_[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (b_[i] - c[i] * x[i + 1]) / d_[i]

        return x

    # ------------------------------------------------------------------
    # Private helper methods for heat-transport terms
    # ------------------------------------------------------------------

    @staticmethod
    def _van_leer(r: np.ndarray) -> np.ndarray:
        """
        van Leer flux limiter φ(r) = (r + |r|) / (1 + |r|).

        Properties:
        - φ(r) = 0 for r <= 0 (non-monotone regions, no anti-diffusion)
        - φ(1) = 1             (1st/2nd order transition at r = 1)
        - φ(r) → 2 for r → ∞  (second order in smooth regions)
        - Continuous and differentiable -> smoother than minmod or superbee
        """
        abs_r = np.abs(r)
        return (r + abs_r) / (1.0 + abs_r)

    @staticmethod
    def _convective_adjustment(
        T: np.ndarray,
        m_nodes: np.ndarray,
    ) -> np.ndarray:
        """
        Convective adjustment: remove unstable temperature inversions.

        In a thermal storage, a colder (denser) layer cannot persist above
        a warmer (lighter) layer. Buoyancy forces enforce immediate mixing.
        This matches buoyancy correction in FreeTTES and in ocean models
        (convective adjustment).

        Algorithm (stack-based, O(N)):
            Process nodes from top (index 0) to bottom (index N-1).
            Build a stack of "mixing zones":
                - New node i first becomes its own zone.
                - While last stack zone is colder than zone i:
                  -> merge (unstable stratification -> immediate mixing)
            Write back averaged temperatures for each zone.

        The method is exactly energy-conserving and O(N), because each node
        is pushed to the stack at most once and popped at most once.

        Parameters
        ----------
        T : np.ndarray
            Temperature profile before correction, shape (n,).
            Index 0 = top (hot), index n-1 = bottom (cold).
        m_nodes : np.ndarray
            Mass of each node [kg], shape (n). Equal for cylinder geometry;
            node-specific for truncated cone (PTES).

        Returns
        -------
        np.ndarray
            Corrected temperature profile, stable everywhere (T[i] ≥ T[i+1]).
        """
        n = len(T)
        # Stack: each item = [m*T_sum, mass_sum, start_index]
        # Process from index 0 (top) to N-1 (bottom).
        blocks: list = []   # [T_wm, m_sum, start_idx]
        for i in range(n):
            T_wm = float(T[i] * m_nodes[i])
            m_sum = float(m_nodes[i])
            start = i
            # While last zone is colder than current one: merge
            while blocks and blocks[-1][0] / blocks[-1][1] < T_wm / m_sum:
                prev = blocks.pop()
                T_wm  += prev[0]
                m_sum += prev[1]
                start  = prev[2]
            blocks.append([T_wm, m_sum, start])

        # Write back results
        result = T.copy()
        for k in range(len(blocks)):
            T_mix = blocks[k][0] / blocks[k][1]
            end   = blocks[k + 1][2] if k + 1 < len(blocks) else n
            result[blocks[k][2]:end] = T_mix
        return result

    # ------------------------------------------------------------------
    # Port helper methods
    # ------------------------------------------------------------------

    def _port_to_node(self, port: "Port") -> int:
        """
        Return nearest node index for a given port height.

        Parameters
        ----------
        port : Port
            Port with height coordinate z [m] above tank bottom.

        Returns
        -------
        int
            Index of nearest node (0 = top, N-1 = bottom).
        """
        return int(np.argmin(np.abs(self._z_nodes - port.z)))

    def _compute_inter_node_fluxes(self, ports: list) -> np.ndarray:
        """
        Compute inter-node mass-flow vector F [kg/s].

        F[j] (j = 0..N) is net mass flow between nodes j-1 and j,
        positive downward. At tank boundaries F[0] = F[N] = 0.

        Flow is computed as cumulative sum of source terms from top:

            F[0] = 0
            F[j] = F[j-1] + S[j-1] with S[i] = net inlet at node i

        Parameters
        ----------
        ports : list[Port]
            List of active ports.

        Returns
        -------
        np.ndarray
            Flow vector, shape (n_nodes + 1,).
        """
        n = self.n
        S = np.zeros(n)
        for port in ports:
            for k, w in self._diffusor.node_weights(port, self._z_nodes):
                S[k] += port.m_dot * w
        F = np.zeros(n + 1)
        F[1:] = np.cumsum(S)
        return F

    def _compute_source_terms(self, ports: list) -> tuple:
        """
        Compute source term S and mass-flow-weighted
        inlet temperature T_src per node.

        Parameters
        ----------
        ports : list[Port]
            List of active ports.

        Returns
        -------
        S : np.ndarray
            Net source term per node [kg/s], shape (n_nodes,).
            Positive = inlet, negative = outlet.
        T_src : np.ndarray
            Mass-flow-weighted inlet temperature per node [°C],
            shape (n_nodes,). Relevant only for nodes with active inlet.
        """
        n = self.n
        S = np.zeros(n)
        S_pos = np.zeros(n)        # Sum of positive inlets per node
        T_src_weighted = np.zeros(n)
        for port in ports:
            for k, w in self._diffusor.node_weights(port, self._z_nodes):
                m_part = port.m_dot * w
                S[k] += m_part
                if m_part > 0.0:
                    S_pos[k] += m_part
                    T_src_weighted[k] += m_part * port.T_in
        # Mass-flow-weighted mean; 0 where no inlet exists
        T_src = np.where(
            S_pos > 0.0,
            T_src_weighted / np.maximum(S_pos, 1e-30),
            0.0,
        )
        return S, T_src

    def _compute_advection_ports(
        self,
        T: np.ndarray,
        F: np.ndarray,
        S: np.ndarray,
        T_src: np.ndarray,
        cp: float,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute advective heat flow with variable port flow field.

        Instead of a uniform net mass-flow scalar, the spatially varying
        inter-node flow F[j] is used. This allows arbitrarily positioned
        ports at different heights.

        Physical model
        --------------
        For node i:

            Q_adv[i] = cp · (Q_face[i] − Q_face[i+1]) + Q_port[i]

        with:
            Q_face[j] = F[j] · T_upwind[j]

            T_upwind[j] = T[j-1]  if F[j] ≥ 0 (downward, upwind from above)
                        = T[j]   if F[j] < 0 (upward, upwind from below)

            Q_port[i] = max(S[i], 0) · T_src[i]   (inlet energy)
                      + min(S[i], 0) · T[i]         (outlet energy with local T)

        Boundary conditions: F[0] = F[N] = 0 -> no interface flows
        at tank boundaries.

        Parameters
        ----------
        T : np.ndarray
            Current temperature profile [°C], shape (n_nodes,).
        F : np.ndarray
            Inter-node flow vector [kg/s], shape (n_nodes+1).
            F[0] = F[N] = 0.
        S : np.ndarray
            Source term per node [kg/s], shape (n_nodes,).
        T_src : np.ndarray
            Inlet temperature per node [°C], shape (n_nodes,).
        cp : float
            Specific heat capacity [J/(kg·K)].
        dt : float, optional
            Timestep size [s]. Required for TVD correction.

        Returns
        -------
        np.ndarray
            Advective heat flow per node [W], shape (n_nodes,).
        """
        n = self.n

        # --- Upwind interface fluxes at internal faces j=1..N-1 ---
        F_int = F[1:-1]   # flow at interfaces k=0..N-2 (between nodes k and k+1)
        Q_face = np.zeros(n + 1)
        Q_face[1:-1] = cp * np.where(
            F_int >= 0.0,
            F_int * T[:-1],    # downward: upwind temperature from above (node k)
            F_int * T[1:],     # upward: upwind temperature from below (node k+1)
        )

        # --- Net advective heat flow from interface balance ---
        Q_adv = Q_face[:-1] - Q_face[1:]

        # --- Port source-term energy ---
        # Inlet: fluid enters with T_src, adds energy S_pos·cp·T_src
        # Outlet: fluid leaves with local node temperature T[i]
        Q_port = cp * (np.maximum(S, 0.0) * T_src + np.minimum(S, 0.0) * T)

        Q_total = Q_adv + Q_port

        # --- TVD anti-diffusion correction ---
        if self.config.advection_scheme == "tvd" and dt is not None and n >= 3:
            Q_total += self._compute_tvd_correction_ports(T, F_int, cp, dt)

        return Q_total

    def _compute_tvd_correction_ports(
        self,
        T: np.ndarray,
        F_int: np.ndarray,
        cp: float,
        dt: float,
    ) -> np.ndarray:
        """
        TVD anti-diffusion correction (van Leer limiter) for variable flow.

        The correction is computed per interface. At each interface k
        (between nodes k and k+1), the anti-diffusive flux is determined
        in flow direction and distributed conservatively to neighboring nodes.

        Parameters
        ----------
        T : np.ndarray
            Temperature profile [°C].
        F_int : np.ndarray
            Flows at internal interfaces [kg/s], shape (n_nodes-1,).
        cp : float
            Specific heat capacity [J/(kg·K)].
        dt : float
            Timestep size [s].

        Returns
        -------
        np.ndarray
            Additive TVD heat-flow correction [W], shape (n_nodes,).
        """
        n = self.n
        cfg = self.config
        dT = np.diff(T)   # T[k+1] - T[k], shape (N-1,)
        Q_tvd = np.zeros(n)
        A_iface = 0.5 * (self._A_cross_nodes[:-1] + self._A_cross_nodes[1:])

        for k in range(n - 1):
            F_k = float(F_int[k])
            if abs(F_k) < 1e-30:
                continue
            dT_k = float(dT[k])
            cfl_k = min(abs(F_k) * dt / (cfg.rho * float(A_iface[k]) * self.dz), 1.0)

            if F_k >= 0.0:
                # Downward flow: upwind is node k (top)
                dT_up = float(dT[k - 1]) if k > 0 else 0.0
                dT_safe = dT_k + (_TVD_EPS if dT_k >= 0 else -_TVD_EPS)
                r = dT_up / dT_safe
                phi = float(self._van_leer(np.array([r]))[0])
                corr = 0.5 * F_k * cp * (1.0 - cfl_k) * phi * dT_k
                # Conservative distribution: node k loses, node k+1 gains
                Q_tvd[k]     -= corr
                Q_tvd[k + 1] += corr
            else:
                # Upward flow: upwind is node k+1 (bottom)
                dT_flow = -dT_k   # T[k] - T[k+1] = gradient in flow direction
                dT_up = float(-dT[k + 1]) if k < n - 2 else 0.0
                dT_flow_safe = dT_flow + (_TVD_EPS if dT_flow >= 0 else -_TVD_EPS)
                r = dT_up / dT_flow_safe
                phi = float(self._van_leer(np.array([r]))[0])
                corr = 0.5 * (-F_k) * cp * (1.0 - cfl_k) * phi * dT_flow
                # Conservative distribution: node k+1 loses, node k gains
                Q_tvd[k]     += corr
                Q_tvd[k + 1] -= corr

        return Q_tvd

    def _compute_conduction(
        self,
        T: np.ndarray,
        K_cond_iface: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute conductive heat flow between neighboring nodes [W].

        Uses a second-order central-difference scheme with
        node-specific conduction coefficients (supports variable
        cross-sections). Boundary conditions are adiabatic.

        Parameters
        ----------
        T : numpy.ndarray
            Current temperature profile [°C].

        Returns
        -------
        numpy.ndarray
            Conductive heat flow per node [W], shape (n_nodes,).

        Notes
        -----
        For node i:

            Q_cond,i = K_iface[i-1] · (T_{i-1} - T_i)   (flow from above)
                     + K_iface[i]   · (T_{i+1} - T_i)   (flow from below)

        K_iface[k] = λ_eff · A_iface[k] / Δz is the thermal conductance at the
        interface between nodes k and k+1.
        Boundary nodes have only one neighbor (adiabatic boundaries).
        """
        Q = np.zeros(self.n)
        K = K_cond_iface if K_cond_iface is not None else self._K_cond_iface

        if self.n > 1:
            # Flow across upper interface (from node i-1 to i): i = 1..N-1
            Q[1:]  += K * (T[:-1] - T[1:])
            # Flow across lower interface (from node i+1 to i): i = 0..N-2
            Q[:-1] += K * (T[1:]  - T[:-1])

        return Q

    def _compute_losses(self, T: np.ndarray) -> np.ndarray:
        """
        Compute ambient heat loss for each node [W].

        Parameters
        ----------
        T : numpy.ndarray
            Current temperature profile [°C].

        Returns
        -------
        numpy.ndarray
            Heat loss per node [W], shape (n_nodes,).
            Positive value = heat gain from ambient (unlikely but
            physically correct for T_i < T_ambient).

        Notes
        -----
        For node i:

            Q_loss,i = U_loss · A_wall,i · (T_ambient - T_i)

        A negative value means heat loss from storage to ambient.
        For boundary nodes, A_wall,i includes top/bottom surfaces.
        """
        return self._loss_model.Q_loss_nodes(T, self.A_wall, self._z_nodes)

    def _get_hx_weights(self, hx: "HeatExchangerPort") -> list:
        """
        Return equal-weight ``(node_index, weight)`` pairs for an HX zone.

        All nodes in ``[z - H_hx/2, z + H_hx/2]`` are weighted equally.
        If no node lies in this range, the nearest node is returned with
        weight 1.0.

        Parameters
        ----------
        hx : HeatExchangerPort
            Heat-exchanger port.

        Returns
        -------
        list[tuple[int, float]]
            List of (node_index, weight) pairs. Weights sum to 1.
        """
        z_lo = hx.z - 0.5 * hx.H_hx
        z_hi = hx.z + 0.5 * hx.H_hx
        indices = [k for k, z in enumerate(self._z_nodes) if z_lo <= z <= z_hi]
        if not indices:
            # Use nearest node
            k = int(np.argmin(np.abs(self._z_nodes - hx.z)))
            return [(k, 1.0)]
        w = 1.0 / len(indices)
        return [(k, w) for k in indices]

    def _compute_hx_source_terms(
        self,
        T: np.ndarray,
        hx_ports: list,
    ) -> tuple:
        """
        Compute heat-source terms for all heat-exchanger ports (epsilon-NTU method).

        Supports two calculation modes per port:

        **Lumped** (``hx.segmented=False``, default):
            Transferred heat is computed once using the mean storage
            temperature in the HX zone and distributed uniformly to all
            nodes in that zone. Suitable when the HX is located in a
            homogeneous temperature zone.

        **Segmented** (``hx.segmented=True``):
            The HX zone is traversed node by node. The external fluid
            temperature is updated after each node:

                NTU_k = (UA / n_zone) / C_ext
                ε_k   = 1 − exp(−NTU_k)
                Q_k   = ε_k · C_ext · (T_ext_current − T_storage[k])
                T_ext_current -= Q_k / C_ext

            Node order follows ``hx.flow_direction``:
            ``"downward"`` -> from high z to low z (inlet at top),
            ``"upward"``   -> from low z to high z (inlet at bottom).
            The segmented model yields accurate ``T_ext_out`` values
            and is correct when the HX bridges the thermocline.

        Parameters
        ----------
        T : numpy.ndarray
            Current temperature profile [°C], shape (n,).
        hx_ports : list[HeatExchangerPort]
            List of active heat-exchanger ports.

        Returns
        -------
        Q_hx_nodes : numpy.ndarray
            Heat-source terms per node [W], shape (n,). Positive = heat input.
        T_ext_out_list : list[float]
            Outlet temperatures of the external fluid for each HX port [°C].
        """
        Q_hx_nodes = np.zeros(self.n)
        T_ext_out_list = []

        for hx in hx_ports:
            C_ext = hx.m_dot_ext * hx.cp_ext  # [W/K]
            if C_ext <= 0.0:
                T_ext_out_list.append(float(hx.T_ext_in))
                continue

            weights = self._get_hx_weights(hx)

            if hx.segmented:
                # --- Segmented model ---
                # Sort nodes in external-fluid flow direction.
                # _z_nodes[0] = top (z=H), _z_nodes[N-1] = bottom (z=0).
                reverse = (hx.flow_direction == "upward")
                ordered_nodes = sorted(
                    [k for k, _ in weights],
                    key=lambda k: self._z_nodes[k],
                    reverse=reverse,
                )
                n_seg = len(ordered_nodes)
                UA_seg = hx.UA / n_seg
                NTU_seg = UA_seg / C_ext
                eps_seg = 1.0 - float(np.exp(-NTU_seg))

                T_ext_current = hx.T_ext_in
                for k in ordered_nodes:
                    Q_k = eps_seg * C_ext * (T_ext_current - T[k])
                    Q_hx_nodes[k] += Q_k
                    T_ext_current -= Q_k / C_ext

                T_ext_out_list.append(float(T_ext_current))

            else:
                # --- Lumped model (existing implementation) ---
                T_tank_mean = sum(T[k] * w for k, w in weights)
                NTU = hx.UA / C_ext
                eps = 1.0 - float(np.exp(-NTU))
                Q_total = eps * C_ext * (hx.T_ext_in - T_tank_mean)
                T_ext_out = hx.T_ext_in - Q_total / C_ext
                for k, w in weights:
                    Q_hx_nodes[k] += Q_total * w
                T_ext_out_list.append(float(T_ext_out))

        return Q_hx_nodes, T_ext_out_list

    def _compute_headspace_exchange(
        self,
        T_headspace: float,
        T_top: float,
        dt: float,
    ) -> tuple:
        """
        Compute heat exchange between headspace and water surface.

        Headspace energy balance (explicit Euler):

            C_hs · dT_hs/dt = -Q_Dach - Q_hs→Wasser

        with:
            Q_Dach      = U_roof · A_cross · (T_hs - T_ambient)
            Q_hs→Wasser = h_hs  · A_cross · (T_hs - T_top)

        Both terms cool the headspace if T_hs > T_ambient and T_hs > T_top.
        Q_hs->water is added as a source term to the top water node
        (positive = heat into storage).

        Parameters
        ----------
        T_headspace : float
            Current headspace temperature [°C].
        T_top : float
            Temperature of the topmost water node [°C].
        dt : float
            Timestep size [s].

        Returns
        -------
        Q_hs_to_water : float
            Heat flow from headspace to the top water node [W].
            Positive = heat flows into storage.
        T_hs_new : float
            New headspace temperature after timestep [°C].
        """
        cfg  = self.config
        A    = float(self._A_cross_nodes[0])   # roof area approx. tank cross-section

        # Thermal mass: rho/cp from config (default = concrete-equivalent roof)
        C_hs = max(cfg.rho_headspace * A * cfg.H_headspace * cfg.cp_headspace, 1.0)

        # Heat flows
        Q_hs_to_water = cfg.h_headspace_water * A * (T_headspace - T_top)
        Q_roof        = cfg.U_roof * A * (T_headspace - cfg.T_ambient)

        # Explicit Euler for headspace temperature
        dT_hs  = dt * (-Q_roof - Q_hs_to_water) / C_hs
        return float(Q_hs_to_water), float(T_headspace + dT_hs)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"ThermalStorage1D("
            f"V={cfg.volume} m³, H={cfg.height} m, "
            f"N={cfg.n_nodes}, "
            f"U_loss={cfg.U_loss} W/(m²·K))"
        )