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
thermal_energy_storage_model.py          Complete model (single file)
examples/example_simulation.py Three example scenarios
benchmark/
    benchmark_model_variants.py Model variant comparison (optional FreeTTES reference)
    benchmark_scenarios.py      Multi-scenario benchmark (optional FreeTTES reference)
    compare_runs.py             Compare named benchmark runs
    results/                    Outputs (PNG, CSV, JSON)
    results/runs/{name}/        Named persistent runs
```

`thermal_energy_storage_model.py` contains all classes — intentionally kept as a single file so that integration into other simulation environments is possible by simply copying the file.

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
StorageState                   (Dataclass: thermal state)
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
    ├── _compute_inter_node_fluxes()     Mass balance → F[i→i+1]
    │       └── diffusor.node_weights()  Port → weighted node distribution
    ├── _compute_source_terms()          Port heat sources per node
    │       └── diffusor.node_weights()  (same distribution)
    ├── _compute_hx_source_terms()       Heat exchanger source terms (ε-NTU)
    │       └── _get_hx_weights()        HX zone → equally weighted nodes
    ├── _compute_advection_ports()       Upwind advection term
    ├── _compute_tvd_correction_ports()  (only when advection_scheme="tvd")
    ├── _compute_conduction()            Conduction term
    ├── _compute_losses()                loss_model.Q_loss_nodes()
    ├── Explicit: Euler integration → T_new  (incl. Q_hx / C_nodes)
    │   or Implicit: _step_implicit() → _solve_tdma() → T_new
    │                (b += Q_hx_nodes, explicitly linearized)
    ├── _convective_adjustment()         (only when buoyancy=True)
    └── port_temperatures computed       diffusor.node_weights() → StorageOutputs
        hx_outlet_temperatures           ε-NTU → T_ext_out per HX port
```

### State Objects

The model uses **immutable state objects** (`StorageState`). `step()` always returns a new state; the old one remains untouched. This allows:

- Reuse of the same state for predictions
- Easy rollback (keep the old state)
- Thread safety without locking

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

Immutable snapshot of the thermal state:

```python
@dataclass
class StorageState:
    temperatures: np.ndarray        # [°C], index 0 = top
    time: float                     # [s]
    T_headspace: Optional[float]    # [°C] or None (when headspace=False)

    # Properties (read-only):
    # .n_nodes, .T_top, .T_bottom, .T_mean, .copy()
```

### StorageInputs

Describes all boundary conditions for one timestep — hydraulic ports and heat exchangers:

```python
@dataclass
class StorageInputs:
    ports:    list[Port]               = field(default_factory=list)
    hx_ports: list[HeatExchangerPort]  = field(default_factory=list)

    @classmethod
    def two_port(cls, m_dot_charge, T_charge_in, ..., height, z_charge_in=None, ...) -> StorageInputs:
        ...  # Convenience constructor for classic two-circuit operation
```

An empty `StorageInputs()` (no ports, no HX ports) corresponds to standby mode.

### StorageOutputs

```python
@dataclass
class StorageOutputs:
    port_temperatures:      list[float]  # Node temperature per port [°C]
    Q_loss:                 float        # Total heat loss [W]
    state:                  StorageState # New state after the timestep
    hx_outlet_temperatures: list[float]  # External outlet temperature per HX port [°C]
```

### ThermalStorage1D

The main class stores no state variables — all state flows through `StorageState`. Geometry arrays are computed once in the constructor and held as attributes:

```python
class ThermalStorage1D:
    config: StorageConfig
    m_node: float          # Mass per node [kg]
    A_cross: np.ndarray    # Cross-sectional area per node [m²]
    A_wall: np.ndarray     # Wall area per node [m²]
    dz: float              # Node height [m]
    node_heights: np.ndarray   # z-coordinate of each node center [m]
```

**Key public methods:**

| Method | Description |
|---|---|
| `initialize(T_init, time)` | Creates `StorageState`; `T_init` can be scalar or array |
| `step(state, dt, inputs)` | Core method: computes one timestep |
| `max_stable_dt(m_dot_max)` | Maximum stable timestep [s] |
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

The model can be extended by implementing the abstract interfaces without modifying `thermal_energy_storage_model.py`:

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

```python
class SeasonalLoss(LossModel):
    def Q_loss_nodes(self, T_nodes, A_wall_nodes, z_node_centers) -> np.ndarray:
        T_ground = self._seasonal_ground_temp(self.current_day)
        return self.U_loss * A_wall_nodes * (T_ground - T_nodes)

config = StorageConfig(..., loss_model=SeasonalLoss(...))
```

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
