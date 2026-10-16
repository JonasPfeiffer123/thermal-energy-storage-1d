# Implementation Architecture

This document describes the structure of the model at the code level.

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Class Hierarchy](#2-class-hierarchy)
3. [Data Flow](#3-data-flow)
4. [Core Classes in Detail](#4-core-classes-in-detail)
5. [Extension Points](#5-extension-points)

---

## 1. Module Overview

```
thermal_energy_storage_model/            Modular Python package (core model)
    __init__.py                          Public API re-exports
    config.py                            StorageConfig
    state.py                             StorageState, StorageInputs, StorageOutputs
    geometry.py                          CylinderGeometry, TruncatedConeGeometry, …
    fluids.py                            WaterProperties, ConstantFluidProperties
    losses.py                            ConstantAmbientLoss, GroundTemperatureLoss, …
    ports.py                             Port, HeatExchangerPort
    diffusors.py                         PointDiffusor, UniformDiffusor
    solver.py                            ThermalStorage1D (explicit/implicit Euler, TDMA)
    _kernels.py                          Per-step loops on Python floats (TVD, TDMA, buoyancy)
    model.py                             Facade re-export of ThermalStorage1D
    presets.py                           StoragePresets factory
examples/example_simulation.py           Three example scenarios
benchmark/
    config_benchmark.py                  Shared tank parameters
    shared_utils.py                      Physical helpers, 0D reference model
    benchmark_model_variants.py          0D-1D accuracy vs. speed comparison (optional FreeTTES reference)
    dronninglund_validation.py           Validation vs. Dronninglund PTES 2014
    hoje_taastrup_validation.py          Validation vs. Høje Taastrup PTES 2024
    compare_runs.py                      Compare named benchmark runs
    benchmark_step_performance.py        Wall-clock cost of one step() (co-simulation use)
    results/                             Pre-computed plots and CSVs
```

The core model lives in the `thermal_energy_storage_model/` package, split by
concern (geometry, fluid, loss, port, diffusor, solver) rather than kept as a
single file. Each abstract interface (`GeometryModel`, `FluidProperties`,
`LossModel`, `DiffusorModel`) has its own module so custom implementations can
be added — or the relevant module copied into another project — without
pulling in the whole package.

---

## 2. Class Hierarchy

```
GeometryModel (abstract)
├── CylinderGeometry
├── TruncatedConeGeometry
└── TruncatedPyramidGeometry

FluidProperties (abstract)
├── ConstantFluidProperties
└── WaterProperties

LossModel (abstract)
├── ConstantAmbientLoss
├── SplitAmbientLoss
├── GroundTemperatureLoss
└── TransientGroundLoss

DiffusorModel (abstract)
├── PointDiffusor              Nearest node (default)
└── UniformDiffusor            Uniform distribution over H_zone

Port                           (Dataclass: one hydraulic connection)
HeatExchangerPort              (Dataclass: hydraulically decoupled heat exchanger)
StorageConfig                  (Dataclass: all configuration parameters)
StorageState                   (Dataclass: complete simulation state)
StorageInputs                  (Dataclass: boundary conditions per timestep)
StorageOutputs                 (Dataclass: results per timestep)

ThermalStorage1D               (Main class)
StoragePresets                 (Factory class for preconfigured setups)
```

The four abstract base classes (`GeometryModel`, `FluidProperties`, `LossModel`, `DiffusorModel`) each define one interface. The core model `ThermalStorage1D` works only against these interfaces — custom implementations can be plugged in without modifying the core model.

---

## 3. Data Flow

```
StorageConfig
    ├── geometry   : GeometryModel
    ├── fluid      : FluidProperties
    ├── loss_model : LossModel
    ├── n_nodes, advection_scheme, buoyancy, ...
    │
    ▼
ThermalStorage1D.__init__()
    ├── _validate_config()         Consistency checks
    └── _precompute_geometry()     A_cross, V_nodes, m_nodes, A_wall_nodes

        ┌──────────────────────────────────────────────────┐
        │  Simulation loop (externally controlled)         │
        │                                                  │
        │  state = storage.initialize(T_init)              │
        │                                                  │
        │  for each timestep:                              │
        │      inputs = StorageInputs(ports=[...])         │
        │      outputs = storage.step(state, dt, inputs)   │
        │      state = outputs.state                       │
        └──────────────────────────────────────────────────┘

storage.step(state, dt, inputs)
    ├── _validate_inputs()               dt > 0, finite port / HX values
    ├── Explicit + auto_substep: _required_substeps()
    │       └── _stability_numbers()     CFL <= 0.9 and CFL + conduction/HX <= 1
    └── _step_single(state, dt_sub, inputs)   (once, or n_sub times)
        ├── _port_flows()                Port sources S, inter-node flows F, inlet T
        │       └── _port_weights()      diffusor.node_weights(), cached per port height
        ├── _fluid_state()               rho, cp → node heat capacities, mean cp
        │                                (shared with get_soc() for the same profile)
        ├── loss model of the step       inputs.T_ambient → with_air_temperature()
        ├── _compute_headspace_exchange() exact integration of the headspace ODE
        ├── _compute_losses()            loss_model.Q_loss_nodes(..., state.loss_state)
        ├── Explicit: _compute_hx_source_terms() (ε-NTU at T_old),
        │   _compute_advection_ports() (+ _compute_tvd_correction_ports()),
        │   _compute_conduction(), Euler integration → T_new
        │   or Implicit: _kernels.tvd_fluxes() (only when advection_scheme="tvd")
        │                _step_implicit() → _kernels.implicit_solve() → T_new
        │                (assembly inside the TDMA forward sweep;
        │                 b += Q_loss + Q_headspace + Q_tvd, explicitly linearized)
        │                with active HX ports: _step_implicit_hx()
        │                (_kernels.implicit_coefficients() + HX couplings,
        │                 dense solve; T_ext_out from T_new)
        ├── _port_temperatures()         outlets: temperature the outflow leaves at
        ├── _kernels.convective_adjustment() (only when buoyancy=True)
        └── loss_model.advance()         new StorageState.loss_state
```

### Performance of `step()`

In co-simulation `step()` runs once per coupling interval on small grids
(5–50 nodes), where the cost of a NumPy call (~0.4 µs, independent of
the array length) dominates the arithmetic. The per-step loops (TVD
correction, matrix assembly and Thomas algorithm, convective adjustment,
`WaterProperties` for up to 32 nodes) therefore run on Python floats in
`_kernels.py`; configuration constants (interface areas, TVD denominators,
port → node mapping) are computed once, and NumPy is used where its pairwise
summation defines the result (`T_mean`, mean cp, `Q_loss`, stored energy).
Every kernel performs the floating-point operations of the former array code
in the same order, so results are bit-identical to v1.0.0:
`tests/test_golden_master.py` compares complete one-year trajectories of 133
configurations, `tests/test_kernels.py` the kernels against the former array
implementations. The loss models stay on NumPy with as few calls per step as
possible (`TransientGroundLoss.advance()` updates all ground layers in one
array expression); `tests/test_losses.py` checks them bit for bit against
their former implementations. `benchmark/benchmark_step_performance.py`
measures the per-step cost (`--loss` selects the loss model).

### State Objects

`StorageState` is the **complete** simulation state: temperature profile,
time, headspace temperature and the internal state of the loss model
(`loss_state`, e.g. the ground temperatures of `TransientGroundLoss`).
`step()` depends only on `(state, dt, inputs)` and the configuration and
never modifies the state passed in; it returns a new one. This allows:

- Reuse of the same state for predictions
- Easy rollback (keep the old state)
- Iterative coupling: repeating a coupling step from the same state gives
  the same result

The model objects (`ThermalStorage1D`, loss models, ...) hold parameters
and caches only. Parameters such as a loss model's ambient temperature may
be changed between steps; for the air temperature, prefer the per-step
input `StorageInputs.T_ambient`.

---

## 4. Core Classes in Detail

### StorageConfig

All parameters are validated at construction time (`_validate_config`). Geometry, fluid, and loss model are passed as objects; if missing, default objects are constructed from scalar parameters (backwards compatibility).

Relevant fields:

| Field | Type | Description |
|---|---|---|
| `volume`, `height` | float | Storage geometry [m³, m] |
| `n_nodes` | int | Discretization |
| `rho`, `cp`, `lambda_fluid` | float | Fallback fluid properties (when `fluid=None`) |
| `lambda_eff_factor` | float | Effective thermal conductivity multiplier, default 5 |
| `U_loss`, `T_ambient` | float | Fallback loss parameters (when `loss_model=None`) |
| `geometry` | GeometryModel | Geometry object |
| `fluid` | FluidProperties | Fluid properties object |
| `loss_model` | LossModel | Loss model object |
| `advection_scheme` | str | `"upwind"` or `"tvd"` |
| `buoyancy` | bool | Convective adjustment on/off |
| `solver` | str | `"explicit"` (default) or `"implicit"` (TDMA) |
| `diffusor_model` | DiffusorModel | Inflow distribution; default: `PointDiffusor()` |
| `T_ground`, `z_ground_layer` | float | Initial profile with ground temperature gradient |
| `headspace` | bool | Headspace model (atmospheric tanks), default: `False` |
| `T_headspace_init` | float | Initial headspace temperature [°C], default: 99.0 |
| `H_headspace` | float | Effective headspace height [m], default: 0.5 |
| `U_roof` | float | Roof heat transfer coefficient [W/(m²·K)], default: 0.2 |
| `h_headspace_water` | float | Convective HTC headspace ↔ water [W/(m²·K)], default: 5.0 |
| `rho_headspace` | float | Headspace body density [kg/m³], default: 2400 (concrete) |
| `cp_headspace` | float | Headspace body heat capacity [J/(kg·K)], default: 880 (concrete) |

### StorageState

Complete snapshot of the simulation state (see *State Objects*):

```python
@dataclass
class StorageState:
    temperatures: np.ndarray            # [°C], index 0 = top
    time: float                         # [s]
    T_headspace: float | None           # [°C] or None (when headspace=False)
    loss_state: np.ndarray | None       # loss-model state, e.g. ground
                                        # temperatures (n_nodes, n_layers)

    # Properties (read-only):
    # .n_nodes, .T_top, .T_bottom, .T_mean, .copy()
```

### StorageInputs

Describes all boundary conditions for one timestep — hydraulic ports and heat exchangers:

```python
@dataclass
class StorageInputs:
    ports:     list[Port]               = field(default_factory=list)
    hx_ports:  list[HeatExchangerPort]  = field(default_factory=list)
    T_ambient: float | None             = None   # air temperature of this step [°C]

    @classmethod
    def two_port(cls, m_dot_charge, T_charge_in, ..., height, z_charge_in=None, ...,
                 T_ambient=None) -> StorageInputs:
        ...  # Convenience constructor for classic two-circuit operation
```

An empty `StorageInputs()` (no ports, no HX ports) corresponds to standby mode.
`T_ambient` replaces the air-side temperature of the loss model
(`LossModel.air_temperature_attr`) and the headspace roof temperature for
one step.

### StorageOutputs

```python
@dataclass
class StorageOutputs:
    port_temperatures:      list[float]  # Outlet temperature per outlet port [°C]
    Q_loss:                 float        # Heat loss applied in the step [W]
    state:                  StorageState # New state after the timestep
    hx_outlet_temperatures: list[float]  # External outlet temperature per HX port [°C]
    T_headspace:            float | None # Headspace temperature [°C]
```

The outputs close the energy balance of the step: the change of
`get_stored_energy()` equals the inlet enthalpy minus the outlet enthalpy at
the reported outlet temperatures, plus the heat-exchanger heat, minus
`Q_loss` (exact for constant fluid properties). An outlet port reports the
temperature at which its fluid leaves: the start-of-step node temperature
(explicit solver) or the end-of-step one (implicit solver); where an inflow
enters the same node, the outflow takes the inflowing fluid first, as in a
hydraulic header.

### ThermalStorage1D

The main class stores no state variables — all state flows through `StorageState`. Geometry arrays are computed once in the constructor and held as attributes:

```python
class ThermalStorage1D:
    config: StorageConfig
    n: int                 # Number of nodes
    dz: float              # Node height [m]
    m_node: float          # Mean mass per node [kg]
    A_cross: float         # Smallest cross-section [m²] (CFL reference)
    A_wall: np.ndarray     # Loss area per node [m²] (lid excluded with headspace)
```

**Key public methods:**

| Method | Description |
|---|---|
| `initialize(T_init, time)` | Creates `StorageState`; `T_init` can be scalar or array |
| `step(state, dt, inputs)` | Core method: computes one timestep |
| `max_stable_dt(m_dot_max)` | Largest timestep of the advective CFL condition [s] |
| `check_cfl(dt, m_dot_max)` | CFL check, returns bool |
| `get_stored_energy(state, T_ref)` | Stored energy [J] |
| `get_soc(state, T_min, T_max)` | State of charge 0–1 |
| `from_preset(preset, **params)` | Class method: construct from preset |

### StoragePresets

Factory class for typical storage configurations:

| Preset | Geometry | Loss model | Typical use case |
|---|---|---|---|
| `"steel_tank_aboveground"` | Cylinder | ConstantAmbientLoss | Above-ground buffer tank |
| `"steel_tank_buried"` | Cylinder | GroundTemperatureLoss | Buried steel tank |
| `"ptes"` | TruncatedCone | GroundTemperatureLoss | Pit thermal energy storage |

```python
# Via class method:
tes = ThermalStorage1D.from_preset("steel_tank_aboveground", volume=500.0, height=15.0)

# Or directly:
config = StoragePresets.ptes(r_bottom=40.0, r_top=55.0, height=15.0)
tes = ThermalStorage1D(config)
```

---

## 5. Extension Points

The model can be extended by implementing the abstract interfaces without modifying the package:

### Custom Geometry Model

```python
class MyGeometry(GeometryModel):
    def volume(self) -> float: ...
    def height(self) -> float: ...
    def A_cross(self, z: float) -> float: ...
    def A_cross_nodes(self, n_nodes: int) -> np.ndarray: ...
    def V_nodes(self, n_nodes: int) -> np.ndarray: ...
    def A_wall_nodes(self, n_nodes: int) -> np.ndarray: ...

config = StorageConfig(..., geometry=MyGeometry())
```

### Custom Loss Model

A steady-state loss model implements `Q_loss_nodes()`; set
`air_temperature_attr` to the attribute that `StorageInputs.T_ambient`
should replace (if the model has an air-exposed surface):

```python
class AirLoss(LossModel):
    air_temperature_attr = "T_air"

    def __init__(self, U, T_air):
        self.U, self.T_air = U, T_air

    def Q_loss_nodes(self, T_nodes, A_wall_nodes, z_node_centers) -> np.ndarray:
        return self.U * A_wall_nodes * (self.T_air - T_nodes)

config = StorageConfig(..., loss_model=AirLoss(U=0.3, T_air=10.0))
```

A model with internal state (history) keeps it in `StorageState.loss_state`,
so that `step()` stays a function of the state:

```python
class MyTransientLoss(LossModel):
    def initial_state(self, n_nodes) -> np.ndarray:
        return np.full(n_nodes, 10.0)                 # e.g. wall temperatures

    def Q_loss_nodes(self, T_nodes, A_wall_nodes, z, state) -> np.ndarray:
        return self.G * A_wall_nodes * (state - T_nodes)

    def advance(self, T_nodes, A_wall_nodes, z, dt, state) -> np.ndarray:
        return state + dt * ...                       # new array, do not modify state
```

`TransientGroundLoss` is the built-in example. (Models that override only
`advance(T, A, z, dt)` and keep their state inside the object still work,
but such a state is not part of `StorageState`: stepping the same state twice
then gives different results.)

### Custom Fluid Properties

```python
class BrineProperties(FluidProperties):
    def rho(self, T): return ...
    def cp(self, T): return ...
    def lambda_fluid(self, T): return ...

config = StorageConfig(..., fluid=BrineProperties())
```

### Custom Diffusor Model

```python
class GaussianDiffusor(DiffusorModel):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def node_weights(self, port, node_heights) -> list[tuple[int, float]]:
        dists = np.abs(node_heights - port.z)
        w = np.exp(-0.5 * (dists / self.sigma) ** 2)
        w /= w.sum()
        return [(int(i), float(w[i])) for i in range(len(node_heights)) if w[i] > 1e-6]

config = StorageConfig(..., diffusor_model=GaussianDiffusor(sigma=1.0))
```

The interface is minimal: `node_weights()` returns a list of `(node_index, weight)` pairs. Weights must sum to 1.0. The model uses this distribution consistently in mass balance, source term computations, and outlet temperatures.

### Hydraulically Decoupled Heat Exchanger

`HeatExchangerPort` is not an abstract class but a ready-made dataclass. It is passed directly in `StorageInputs.hx_ports` and requires no subclassing:

```python
from thermal_energy_storage_model import HeatExchangerPort, StorageInputs

# Solar collector charges the upper part of the storage
hx = HeatExchangerPort(
    z=9.0,          # Center 1 m below lid
    H_hx=2.0,       # 2 m active zone
    UA=5000.0,       # [W/K]
    m_dot_ext=1.2,   # [kg/s]
    T_ext_in=85.0,   # [°C]
    label="Solar",
)
inputs = StorageInputs(hx_ports=[hx])
out = tes.step(state, dt=60.0, inputs=inputs)
print(out.hx_outlet_temperatures[0])   # Solar return temperature [°C]
```

Multiple HX ports can be active simultaneously and can be combined with hydraulic ports.
