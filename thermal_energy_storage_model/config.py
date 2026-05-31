"""Configuration dataclass for the 1D thermal storage model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .diffusors import DiffusorModel
    from .fluids import FluidProperties
    from .geometry import GeometryModel
    from .losses import LossModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StorageConfig:
    """
    Configuration parameters for the thermal energy storage model.

    Contains all geometric and thermophysical parameters that define the
    storage model. Parameters are specified in SI units.

    Parameters
    ----------
    volume : float
        Total storage volume [m³]. Must be positive.
    height : float
        Storage height [m]. Must be positive.
    n_nodes : int, optional
        Number of vertical discretization nodes. More nodes increase
        spatial resolution and computation cost. Typical range for
        co-simulation: 10-50. Default: 20.
    rho : float, optional
        Fluid density [kg/m³]. Default water value near 70 °C: 977.8.
    cp : float, optional
        Fluid specific heat capacity [J/(kg·K)].
        Default water value: 4187.0.
    lambda_fluid : float, optional
        Fluid thermal conductivity [W/(m·K)]. Default: 0.663.
    lambda_eff_factor : float, optional
        Multiplicative factor for effective thermal conductivity to account
        for turbulent mixing near the thermocline. Typical: 1-100.
        Default: 5.0.
    U_loss : float, optional
        Storage wall heat transfer coefficient [W/(m²·K)].
        Well-insulated storages: 0.1-0.5 W/(m²·K). Default: 0.3.
        Only used when ``loss_model=None`` (backward compatibility).
    T_ambient : float, optional
        Ambient temperature for heat-loss calculation [°C]. Default: 10.0.
        Only used when ``loss_model=None`` (backward compatibility).
    loss_model : LossModel, optional
        Loss model for wall heat-loss calculations.
        Supported types include ``ConstantAmbientLoss`` and
        ``GroundTemperatureLoss``. If ``None`` (default),
        ``ConstantAmbientLoss`` is created from ``U_loss`` and
        ``T_ambient`` (backward compatibility).

    Examples
    --------
    Large district-heating buffer storage:

        >>> config = StorageConfig(
        ...     volume=500.0,
        ...     height=15.0,
        ...     n_nodes=30,
        ...     U_loss=0.2,
        ...     T_ambient=15.0
        ... )

    Small domestic buffer storage:

        >>> config = StorageConfig(volume=1.0, height=1.8, n_nodes=10)
    """

    volume: float
    """Total storage volume [m³]."""

    height: float
    """Storage height [m]."""

    n_nodes: int = 20
    """Number of vertical discretization nodes [-]."""

    rho: float = 977.8
    """Fluid density [kg/m³] (water at about 70 °C)."""

    cp: float = 4187.0
    """Fluid specific heat capacity [J/(kg·K)]."""

    lambda_fluid: float = 0.663
    """Fluid thermal conductivity [W/(m·K)] (water at about 70 °C)."""

    lambda_eff_factor: float = 5.0
    """
    Multiplicative factor for effective thermal conductivity.

    Accounts for turbulent mixing near the thermocline.
    lambda_eff = lambda_eff_factor * lambda_fluid.
    """

    U_loss: float = 0.3
    """Storage wall heat transfer coefficient [W/(m²·K)]."""

    T_ambient: float = 10.0
    """Ambient temperature used for heat-loss calculation [°C]."""

    geometry: Optional["GeometryModel"] = field(default=None, repr=False)
    """
    Storage geometry model.

    If provided, it overrides ``volume`` and ``height`` for all geometric
    calculations. Typical types: ``CylinderGeometry`` and
    ``TruncatedConeGeometry``.

    If ``None`` (default), a ``CylinderGeometry`` is created from
    ``volume`` and ``height`` (backward compatibility).
    """

    loss_model: Optional["LossModel"] = field(default=None, repr=False)
    """
    Storage loss model.

    Supported types: ``ConstantAmbientLoss``, ``GroundTemperatureLoss``.

    If ``None`` (default), ``ConstantAmbientLoss`` is created from
    ``U_loss`` and ``T_ambient`` (backward compatibility).
    """

    fluid: Optional["FluidProperties"] = field(default=None, repr=False)
    """
    Fluid-property model for the storage.

    Supported types: ``ConstantFluidProperties``, ``WaterProperties``.

    If ``None`` (default), ``ConstantFluidProperties`` is created from
    ``rho``, ``cp`` and ``lambda_fluid`` (backward compatibility).
    ``WaterProperties()`` uses temperature-dependent polynomial fits
    (FreeTTES, 0-130 °C).
    """

    advection_scheme: str = "tvd"
    """
    Discretization scheme for the advection term.

    ``"upwind"``
        First-order upwind differencing. Stable but numerically diffusive.
        Thermocline sharpness decreases with distance from inlet.

    ``"tvd"``
        Total Variation Diminishing (TVD) with van Leer limiter.
        Second order in smooth regions, first order near discontinuities.
        Preserves monotonicity (no spurious overshoot/undershoot) and keeps
        the thermocline sharper than upwind. Recommended default.
    """

    buoyancy: bool = True
    """
    Buoyancy model: convective adjustment after each timestep.

    In real storage, a denser (cooler) layer cannot persist above a lighter
    (warmer) layer. Buoyancy forces trigger convective mixing.

    ``True`` (Standard)
        After each Euler integration, the profile is checked for unstable
        inversions (T[i] < T[i+1], i.e., cooler above warmer). Neighboring
        inverted nodes are mixed using mass-weighted averaging until the
        profile is stable everywhere. The procedure is energy-conserving.

    ``False``
        No buoyancy model. Backward compatible with older versions.
    """

    T_ground: Optional[float] = None
    """
    Ground temperature for initialization of the bottom zone [°C].

    If set, the initial temperature profile in the bottom
    ``z_ground_layer`` meters above tank bottom is linearly interpolated
    from ``T_ground`` (bottom, z = 0) to the regular profile value.
    This approximates heat exchange with ground in the bottom diffusor zone.
    Default: ``None`` (no correction).
    """

    z_ground_layer: float = 2.0
    """
    Thickness of bottom zone for ground-temperature interpolation [m].

    Only relevant when ``T_ground`` is set. Default: 2.0 m.
    """

    solver: str = "explicit"
    """
    Time-integration solver.

    ``"explicit"`` (Standard)
        Explicit Euler method. Fast and simple, but requires the CFL
        condition (v*dt/dz <= 1). Typical stable timestep range is 30-600 s.

    ``"implicit"``
        Fully implicit Euler method with upwind discretization (TDMA).
        Unconditionally stable for large timesteps. The tridiagonal solve
        (Thomas algorithm) has O(N) complexity. Heat losses are treated
        semi-implicitly (explicit linearization). Recommended for long
        timesteps or very fine grids.

    Notes
    -----
    Combinations:

    * ``solver="explicit"`` + ``advection_scheme="tvd"`` — original default.
      Anti-diffusion TVD correction is integrated explicitly in the Euler
      step; subject to CFL stability for the advection AND for explicit
      conduction.
    * ``solver="implicit"`` + ``advection_scheme="upwind"`` — unconditionally
      stable; pure upwind advection (more thermocline smearing at coarse
      timesteps).
    * ``solver="implicit"`` + ``advection_scheme="tvd"`` — implicit upwind
      base + deferred-correction TVD anti-diffusion added to the right-hand
      side. Unconditionally stable for the bulk dynamics; the TVD correction
      itself self-deactivates as CFL approaches 1 (via the ``(1 − CFL)``
      weight in the van-Leer flux) so very large timesteps degenerate to
      pure implicit upwind. Recommended for production runs that need both
      sharp-front accuracy and arbitrary timestep choice.

    Buoyancy correction (``buoyancy``) is always applied as a post-step
    convective adjustment, independent of solver and advection_scheme.
    """

    diffusor_model: Optional["DiffusorModel"] = field(default=None, repr=False)
    """
    Diffusor mixing model for all ports.

    Controls how a port mass flow is spatially distributed over storage nodes.
    Affects source-term distribution, inter-node flows and outlet-temperature
    evaluation.

    Supported types
    ---------------
    ``None`` (Standard)
        Automatically maps to :class:`PointDiffusor` - full mass flow to the
        nearest node (previous behavior).

    :class:`PointDiffusor`
        Explicit point diffusor. Equivalent to ``None``.

    :class:`UniformDiffusor` (H_zone)
        Uniform distribution over zone width H_zone [m].
        Recommended: ``H_zone = H_RS_Dif`` from FreeTTES config
        (benchmark default: 1.0 m).

    Note
    ----
    The diffusor model is solver-agnostic and works for both
    ``solver='explicit'`` and ``solver='implicit'``.
    """

    headspace: bool = False
    """
    Headspace model for atmospheric outdoor storages.

    Models a hot gas/vapor space above the water surface. The headspace has
    its own temperature ``T_headspace_init``, loses heat through the roof
    (``U_roof``), and exchanges heat with the top water node
    (``h_headspace_water``).

    ``False`` (Standard)
        No headspace model.

    ``True``
        Headspace enabled.
    """

    T_headspace_init: float = 99.0
    """
    Initial headspace temperature [°C].

    Typically higher than charging temperature due to roof heating.
    Only relevant when ``headspace=True``. Default: 99 °C.
    """

    H_headspace: float = 0.5
    """
    Headspace height (gas volume above water surface) [m].

    Determines thermal mass of the headspace:
    ``C_hs = ρ_Luft · A_cross · H_headspace · cp_Luft``.
    Only relevant when ``headspace=True``. Default: 0.5 m.
    """

    U_roof: float = 0.2
    """
    Heat transfer coefficient of the storage roof [W/(m²·K)].

    Determines heat loss from headspace to ambient:
    ``Q_Dach = U_roof · A_cross · (T_hs - T_ambient)``.
    Only relevant when ``headspace=True``. Default: 0.2 W/(m²·K).
    """

    h_headspace_water: float = 5.0
    """
    Heat transfer coefficient between headspace and water surface [W/(m²·K)].

    Determines heat exchange between hot headspace gas and top water node:
    ``Q_hs→Wasser = h_hs · A_cross · (T_hs - T_top)``.
    Typical range: 1-20 W/(m²·K) for natural gas-water convection.
    Only relevant when ``headspace=True``. Default: 5 W/(m²·K).
    """

    rho_headspace: float = 2400.0
    """
    Effective density of the headspace thermal mass [kg/m³].

    For large concrete storages, roof thermal mass can dominate over gas
    thermal mass. Default 2400 kg/m³ represents concrete and provides high
    thermal inertia. Pure gas-space modeling would use values near
    1.2 kg/m³ (air) or 0.9 kg/m³ (steam around 90 °C).
    """

    cp_headspace: float = 880.0
    """
    Effective specific heat capacity of headspace thermal mass [J/(kg·K)].

    Default 880 J/(kg·K) represents concrete (consistent with
    ``rho_headspace=2400``). Air-model values are around 1005 J/(kg·K),
    steam around 2000 J/(kg·K). Only relevant when ``headspace=True``.
    """