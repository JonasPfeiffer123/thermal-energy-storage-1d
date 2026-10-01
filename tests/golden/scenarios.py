"""
Golden-master scenarios: one simulated year per configuration.

Each scenario builds a fresh :class:`ThermalStorage1D` (fresh loss-model state
for ``TransientGroundLoss``), drives it for 8 760 hourly steps with a
deterministic charge / discharge / simultaneous / idle profile and records
every quantity a co-simulation partner reads per step (full temperature
profile, ``Q_loss``, SOC, stored energy, ``T_mean``/``T_top``/``T_bottom``,
port and heat-exchanger outlet temperatures, headspace temperature).

The recorded trajectories are reduced to

* a SHA-256 digest per quantity over the *complete* trajectory (bitwise
  comparison of every node and every hour), and
* the values at a few checkpoints (tolerance comparison and diagnostics when
  the bitwise check does not apply, e.g. on another platform whose ``exp`` /
  ``pow`` differ in the last ulp).

See ``tests/golden/generate.py`` for (re)generating the reference file and
``tests/test_golden_master.py`` for the comparison.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from thermal_energy_storage_model import (
    ConstantAmbientLoss,
    ConstantFluidProperties,
    CylinderGeometry,
    GroundTemperatureLoss,
    HeatExchangerPort,
    Port,
    SplitAmbientLoss,
    StorageConfig,
    StorageInputs,
    ThermalStorage1D,
    TransientGroundLoss,
    TruncatedConeGeometry,
    TruncatedPyramidGeometry,
    UniformDiffusor,
    WaterProperties,
)

N_STEPS = 8760
DT = 3600.0
# Checkpoint hours (0-based step index after which values are stored).
CHECKPOINTS = np.arange(1459, N_STEPS, 1460)   # 6 checkpoints, last = 8759
T_SOC_MIN = 40.0
T_SOC_MAX = 85.0
MAX_PORTS = 5
MAX_HX = 2

# Quantities that are hashed / checkpointed per scenario.
QUANTITIES = (
    "T", "Q_loss", "soc", "energy", "T_mean", "T_top", "T_bottom",
    "time", "ports", "hx", "T_headspace",
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _geometry(name: str):
    if name == "cylinder":
        return CylinderGeometry.from_volume(100.0, 8.0)
    if name == "cone":
        return TruncatedConeGeometry(r_bottom=2.5, r_top=3.5, height=5.0)
    if name == "pyramid":
        return TruncatedPyramidGeometry.from_slope(
            a_bottom=4.0, b_bottom=3.0, height=4.0, slope=0.5,
        )
    raise ValueError(name)


def _loss(name: str):
    if name == "constant":
        return ConstantAmbientLoss(U_loss=0.4, T_ambient=10.0)
    if name == "split":
        return SplitAmbientLoss(U_lid=0.25, U_wall=0.4, T_ambient=8.0,
                                T_ambient_lid=12.0)
    if name == "ground":
        return GroundTemperatureLoss(U_loss=0.3, burial_depth=0.5,
                                     T_surface=8.0, T_deep=11.0,
                                     depth_decay=2.0)
    if name == "transient":
        return TransientGroundLoss(U_lid=0.2, T_ambient_lid=10.0,
                                   lambda_soil=1.5, rho_soil=1800.0,
                                   cp_soil=900.0, d_total=5.0, n_layers=4,
                                   T_far=9.0, T_init=12.0)
    raise ValueError(name)


def _fluid(name: str):
    if name == "water":
        return WaterProperties()
    if name == "const":
        return ConstantFluidProperties(977.8, 4187.0, 0.663)
    raise ValueError(name)


@dataclass(frozen=True)
class Scenario:
    """One golden-master configuration."""

    sid: str
    build: Callable[[], ThermalStorage1D] = field(repr=False, compare=False)
    # Optional extras: "mid_ports" (3-level ports), "hx" (heat exchangers).
    extras: tuple[str, ...] = ()


def _config_scenario(
    *,
    n: int,
    geometry: str = "cylinder",
    loss: str = "constant",
    fluid: str = "water",
    buoyancy: bool = True,
    solver: str = "implicit",
    advection: str = "tvd",
    extras: tuple[str, ...] = (),
    **cfg_extra,
) -> Scenario:
    parts = [solver[:4], advection, f"n{n}", geometry, loss, fluid,
             "buoy" if buoyancy else "nobuoy", *extras]
    sid = "-".join(parts)

    def build() -> ThermalStorage1D:
        geom = _geometry(geometry)
        cfg_kwargs = dict(cfg_extra)
        if "uniform_diffusor" in extras:
            cfg_kwargs["diffusor_model"] = UniformDiffusor(H_zone=1.0)
        config = StorageConfig(
            volume=geom.volume,
            height=geom.height,
            n_nodes=n,
            geometry=geom,
            loss_model=_loss(loss),
            fluid=_fluid(fluid),
            buoyancy=buoyancy,
            solver=solver,
            advection_scheme=advection,
            **cfg_kwargs,
        )
        return ThermalStorage1D(config)

    return Scenario(sid=sid, build=build, extras=extras)


def _preset_scenario(preset: str, n: int, **params) -> Scenario:
    def build() -> ThermalStorage1D:
        return ThermalStorage1D.from_preset(preset, n_nodes=n, **params)

    return Scenario(sid=f"preset-{preset}-n{n}", build=build)


def build_scenarios() -> list[Scenario]:
    """Return the full golden-master scenario matrix."""
    sc: list[Scenario] = []
    nodes = (5, 20, 50)
    geometries = ("cylinder", "cone", "pyramid")
    losses = ("constant", "split", "ground", "transient")
    fluids = ("water", "const")

    # 1) DistrictHeatingSim configuration (implicit + TVD + buoyancy):
    #    every node count x geometry x loss model x fluid.
    for n in nodes:
        for g in geometries:
            for loss in losses:
                for f in fluids:
                    sc.append(_config_scenario(n=n, geometry=g, loss=loss, fluid=f))

    # 2) Buoyancy off for the same solver, every geometry and fluid.
    for n in nodes:
        for g in geometries:
            for f in fluids:
                sc.append(_config_scenario(n=n, geometry=g, fluid=f,
                                           buoyancy=False))

    # 3) Other solver / advection variants: every node count with both
    #    fluids and both buoyancy settings (paired to bound the runtime of
    #    the sub-stepped explicit solver), plus the non-cylindrical shapes.
    variants = (("implicit", "upwind"), ("explicit", "tvd"),
                ("explicit", "upwind"))
    for solver, adv in variants:
        for n in nodes:
            for f, buoy in (("water", True), ("const", False)):
                sc.append(_config_scenario(n=n, fluid=f, buoyancy=buoy,
                                           solver=solver, advection=adv))
        for g in ("cone", "pyramid"):
            sc.append(_config_scenario(n=20, geometry=g, solver=solver,
                                       advection=adv))

    # 4) Presets (default solver of StorageConfig = explicit, TVD).
    for n in nodes:
        sc.append(_preset_scenario("steel_tank_aboveground", n,
                                   volume=100.0, height=8.0))
        sc.append(_preset_scenario("steel_tank_buried", n,
                                   volume=100.0, height=8.0, burial_depth=0.5))
        sc.append(_preset_scenario("ptes", n,
                                   r_bottom=4.0, r_top=6.0, height=4.0))

    # 5) Feature paths touched by the port / source-term handling.
    for solver in ("implicit", "explicit"):
        sc.append(_config_scenario(n=20, solver=solver,
                                   extras=("uniform_diffusor",)))
        sc.append(_config_scenario(n=20, solver=solver, extras=("mid_ports",)))
        sc.append(_config_scenario(n=20, solver=solver, extras=("hx",)))
    sc.append(_config_scenario(n=20, extras=("headspace",), headspace=True,
                               T_headspace_init=60.0))
    sc.append(_config_scenario(n=20, extras=("ground_init",), T_ground=12.0,
                               z_ground_layer=2.0))
    sc.append(_config_scenario(n=10, extras=("uniform_diffusor", "mid_ports"),
                               loss="split"))
    # Explicit solver without sub-stepping at a CFL-safe node count.
    sc.append(_config_scenario(n=5, solver="explicit", extras=("no_substep",),
                               auto_substep=False))

    ids = [s.sid for s in sc]
    assert len(ids) == len(set(ids)), "duplicate scenario ids"
    return sc


# ---------------------------------------------------------------------------
# Operating profile
# ---------------------------------------------------------------------------

@dataclass
class HourInputs:
    """Hourly boundary conditions (independent of the storage object)."""

    m_charge: float
    T_charge: float
    m_discharge: float
    T_discharge: float


def build_profile(seed: int = 20260930) -> list[HourInputs]:
    """
    Deterministic one-year hourly profile in units of a reference flow.

    Flows are expressed as fractions of the flow that exchanges the full tank
    volume in 8 h; :func:`run_scenario` scales them per storage. The daily
    pattern covers charge-only (downward flow), discharge-only (upward flow),
    simultaneous operation with either loop dominating, exactly balanced
    loops (zero net flow) and idle hours; every seventh day is fully idle.
    Charge temperatures vary between 55 and 92 °C and return temperatures
    between 35 and 60 °C, so inversions at the top and bottom (buoyancy) and
    all TVD flow directions occur.
    """
    rng = np.random.default_rng(seed)
    u = rng.random((N_STEPS, 4))
    profile: list[HourInputs] = []
    for h in range(N_STEPS):
        day, hod = divmod(h, 24)
        winter = 0.5 + 0.5 * math.cos(2.0 * math.pi * day / 365.0)
        a, b, c, d = (float(x) for x in u[h])
        mc = md = 0.0
        Tc = 55.0 + 37.0 * a
        Td = 35.0 + 25.0 * b
        if day % 7 == 6:
            pass                                    # idle day
        elif hod < 6:                               # night charging
            mc = 0.5 + 0.5 * c
        elif hod < 10:                              # simultaneous
            mc = 0.3 + 0.4 * c
            md = (0.2 + 0.6 * winter) * (0.5 + d)
        elif hod < 12:                              # idle
            pass
        elif hod < 16:                              # solar-like charging
            mc = (1.0 - winter) * (0.3 + 0.7 * c)
        elif hod < 22:                              # evening discharge
            md = (0.3 + 0.7 * winter) * (0.4 + 0.6 * d)
        elif hod == 22:                             # exactly balanced loops
            mc = md = 0.4 + 0.2 * c
        profile.append(HourInputs(mc, Tc, md, Td))
    return profile


def _hx_ports(H: float, h: int, p: HourInputs) -> list:
    ports = [
        HeatExchangerPort(z=0.75 * H, H_hx=0.3 * H, UA=4000.0,
                          m_dot_ext=0.6 * (h % 3 != 0), T_ext_in=p.T_charge,
                          label="solar"),
        HeatExchangerPort(z=0.25 * H, H_hx=0.3 * H, UA=3000.0,
                          m_dot_ext=0.4, T_ext_in=p.T_discharge,
                          segmented=True,
                          flow_direction="upward" if h % 2 else "downward",
                          label="hp"),
    ]
    return ports


def _inputs(storage: ThermalStorage1D, scenario: Scenario, h: int,
            p: HourInputs) -> StorageInputs:
    H = storage._geom.height
    rho_ref = 977.8
    m_ref = rho_ref * storage._geom.volume / (8.0 * 3600.0)
    mc = p.m_charge * m_ref
    md = p.m_discharge * m_ref
    if "mid_ports" in scenario.extras:
        # Charge at the top, discharge return at mid height, outlets split
        # between bottom and one third height.
        ports = []
        if mc > 0.0:
            ports.append(Port(z=H, m_dot=+mc, T_in=p.T_charge))
            ports.append(Port(z=0.0, m_dot=-0.6 * mc))
            ports.append(Port(z=H / 3.0, m_dot=-0.4 * mc))
        if md > 0.0:
            ports.append(Port(z=0.5 * H, m_dot=+md, T_in=p.T_discharge))
            ports.append(Port(z=H, m_dot=-md))
        inputs = StorageInputs(ports=ports)
    else:
        inputs = StorageInputs.two_port(
            m_dot_charge=mc, T_charge_in=p.T_charge,
            m_dot_discharge=md, T_discharge_in=p.T_discharge,
            height=H,
        )
    if "hx" in scenario.extras:
        inputs.hx_ports = _hx_ports(H, h, p)
    return inputs


# ---------------------------------------------------------------------------
# Running and reducing
# ---------------------------------------------------------------------------

def run_scenario(scenario: Scenario,
                 profile: list[HourInputs]) -> dict[str, np.ndarray]:
    """Run one year and return the full trajectories of all quantities."""
    storage = scenario.build()
    n = storage.n
    state = storage.initialize(T_init=np.linspace(70.0, 45.0, n))

    T = np.empty((N_STEPS, n))
    ports = np.full((N_STEPS, MAX_PORTS), np.nan)
    hx = np.full((N_STEPS, MAX_HX), np.nan)
    scalars = {k: np.empty(N_STEPS) for k in (
        "Q_loss", "soc", "energy", "T_mean", "T_top", "T_bottom", "time",
        "T_headspace",
    )}

    for h in range(N_STEPS):
        inputs = _inputs(storage, scenario, h, profile[h])
        out = storage.step(state, dt=DT, inputs=inputs)
        state = out.state
        # Exactly the per-step reads of DistrictHeatingSim ...
        T[h] = state.temperatures
        scalars["Q_loss"][h] = out.Q_loss
        scalars["T_top"][h] = state.T_top
        scalars["T_bottom"][h] = state.T_bottom
        scalars["T_mean"][h] = state.T_mean
        scalars["soc"][h] = storage.get_soc(state, T_SOC_MIN, T_SOC_MAX)
        # ... plus the remaining outputs.
        scalars["energy"][h] = storage.get_stored_energy(state)
        scalars["time"][h] = state.time
        scalars["T_headspace"][h] = (
            np.nan if out.T_headspace is None else out.T_headspace
        )
        pt = out.port_temperatures
        ports[h, :len(pt)] = pt
        ht = out.hx_outlet_temperatures
        hx[h, :len(ht)] = ht

    return {"T": T, "ports": ports, "hx": hx, **scalars}


def digest(arr: np.ndarray) -> str:
    """SHA-256 over the exact bytes of a float64 C-contiguous array."""
    a = np.ascontiguousarray(arr, dtype="<f8")
    return hashlib.sha256(a.tobytes()).hexdigest()


def reduce_run(traj: dict[str, np.ndarray]) -> dict[str, np.ndarray | str]:
    """Reduce full trajectories to digests + checkpoint values."""
    out: dict[str, np.ndarray | str] = {}
    for q in QUANTITIES:
        out[f"{q}.sha256"] = digest(traj[q])
        out[f"{q}.checkpoints"] = np.asarray(traj[q][CHECKPOINTS], dtype=float)
    return out
