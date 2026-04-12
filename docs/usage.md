# Usage & Benchmarks

This document describes installation, quick start, all configuration options, and benchmark operation.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [StorageConfig — All Parameters](#3-storageconfig--all-parameters)
4. [StorageInputs — Operating Modes](#4-storageinputs--operating-modes)
5. [LossModel — Loss Models](#5-lossmodel--loss-models)
6. [Initial Profile](#6-initial-profile)
7. [Presets](#7-presets)
8. [DiffusorModel — Inflow Distribution](#8-diffusormodel--inflow-distribution)
9. [HeatExchangerPort — Hydraulically Decoupled Heat Transfer](#9-heatexchangerport--hydraulically-decoupled-heat-transfer)
10. [Headspace Model](#10-headspace-model)
11. [Graphical User Interface (PyQt6)](#11-graphical-user-interface-pyqt6)
12. [Example Simulation](#12-example-simulation)
13. [Benchmark: Comparison Against FreeTTES](#13-benchmark-comparison-against-freetttes)
14. [Benchmark: Model Matrix (Ablation Study)](#14-benchmark-model-matrix-ablation-study)
15. [Benchmark: Named Runs & Comparisons](#15-benchmark-named-runs--comparisons)

---

## 1. Installation

```bash
pip install numpy matplotlib
# or:
pip install -r requirements.txt
```

Requires Python ≥ 3.10.

---

## 2. Quick Start

```python
from thermal_energy_storage_model import (
    StorageConfig, ThermalStorage1D, StorageInputs, WaterProperties
)

# Configure storage
config = StorageConfig(
    volume=500.0,       # m³
    height=15.0,        # m
    n_nodes=20,
    U_loss=0.3,         # W/(m²·K)
    T_ambient=10.0,     # °C
)
storage = ThermalStorage1D(config)

# Initial state: uniform 60 °C
state = storage.initialize(T_init=60.0)

# Time loop: 6 hours of charging
dt = 300.0  # s
for t in range(0, 6 * 3600, int(dt)):
    inputs = StorageInputs.two_port(
        m_dot_charge=5.0, T_charge_in=90.0,
        height=config.height,
    )
    outputs = storage.step(state, dt=dt, inputs=inputs)
    state = outputs.state

print(f"T top: {state.T_top:.1f} °C | T bottom: {state.T_bottom:.1f} °C")
print(f"Losses: {outputs.Q_loss/1000:.2f} kW")
```

---

## 3. StorageConfig — All Parameters

```python
StorageConfig(
    # ── Required fields ────────────────────────────────────────────────────────
    volume          = 500.0,    # m³     Total volume
    height          = 15.0,     # m      Storage height

    # ── Discretization ─────────────────────────────────────────────────────────
    n_nodes         = 20,       # –      Number of nodes (more = more accurate, slower)

    # ── Fluid properties (fallback if fluid=None) ───────────────────────────────
    rho             = 977.8,    # kg/m³  Fluid density (water ~70 °C)
    cp              = 4187.0,   # J/(kg·K)
    lambda_fluid    = 0.663,    # W/(m·K)
    lambda_eff_factor = 5.0,    # –      Effective thermal conductivity multiplier

    # ── Heat losses (fallback if loss_model=None) ───────────────────────────────
    U_loss          = 0.3,      # W/(m²·K) Overall heat transfer coefficient
    T_ambient       = 10.0,     # °C     Ambient temperature

    # ── Physics modules (optional, override fallback parameters) ────────────────
    geometry    = CylinderGeometry.from_volume(500.0, 15.0),
    fluid       = WaterProperties(),              # or ConstantFluidProperties(...)
    loss_model  = ConstantAmbientLoss(0.3, 10.0), # or GroundTemperatureLoss(...)

    # ── Numerics ───────────────────────────────────────────────────────────────
    advection_scheme = "tvd",      # "tvd" (default) or "upwind"
    buoyancy         = True,       # Convective adjustment (buoyancy)
    solver           = "explicit", # "explicit" (default) or "implicit" (TDMA)
    diffusor_model   = PointDiffusor(),  # or UniformDiffusor(H_zone=1.0)

    # ── Initial profile (optional) ─────────────────────────────────────────────
    T_ground        = None,     # °C  Ground temperature for initialization gradient
    z_ground_layer  = None,     # m   Gradient layer height from bottom
)
```

### Recommended Values for Grid Resolution

| Application | n_nodes | Δz | Δt_max (typical) |
|---|---|---|---|
| Rough estimate | 10–20 | 0.5–2 m | ≥ 300 s |
| Standard co-simulation | 20–50 | 0.3–0.8 m | 60–300 s |
| High accuracy | 50–200 | 0.08–0.3 m | 10–120 s |

**Rule of thumb:** N=50 offers a good trade-off between accuracy (~1.2 K MAE against FreeTTES) and computation time (~0.25 ms/step).

---

## 4. StorageInputs — Operating Modes

### Charging Only

```python
inputs = StorageInputs.two_port(
    m_dot_charge=8.0, T_charge_in=90.0,
    height=config.height,
)
```

### Discharging Only

```python
inputs = StorageInputs.two_port(
    m_dot_discharge=5.0, T_discharge_in=55.0,
    height=config.height,
)
```

### Simultaneous Charging and Discharging

```python
inputs = StorageInputs.two_port(
    m_dot_charge=8.0,    T_charge_in=90.0,
    m_dot_discharge=5.0, T_discharge_in=55.0,
    height=config.height,
)
```

### Standby (No Flow)

```python
inputs = StorageInputs()   # empty port list
```

### Explicit Diffusor Positions (e.g. for FreeTTES validation)

```python
inputs = StorageInputs.two_port(
    m_dot_charge=80.0, T_charge_in=90.0,
    height=39.9,
    z_charge_in=39.65,   # upper diffusor center [m]
    z_charge_out=1.00,   # lower diffusor center [m]
)
```

### More Than Two Circuits (direct port list)

```python
from thermal_energy_storage_model import Port

inputs = StorageInputs(ports=[
    Port(z=9.8,  m_dot=+5.0, T_in=85.0, label="solar_in"),
    Port(z=5.0,  m_dot=-5.0,            label="solar_out"),
    Port(z=0.2,  m_dot=+3.0, T_in=45.0, label="net_return"),
    Port(z=9.0,  m_dot=-3.0,            label="net_supply"),
])
```

### Reading Outlet Temperatures

```python
outputs = storage.step(state, dt, inputs)

# port_temperatures returns the node temperature at each port position.
# Index corresponds to the order in inputs.ports.
# For two_port() with charging only:
#   ports[0] = charging inlet (inlet → T_in is given, port_temperatures shows node temp.)
#   ports[1] = charging outlet (outlet → port_temperatures[1] is the outlet temperature)
T_charge_out = outputs.port_temperatures[1]
```

---

## 5. LossModel — Loss Models

All loss models implement the `LossModel` interface and are passed via `StorageConfig(loss_model=...)`.

### ConstantAmbientLoss

```python
from thermal_energy_storage_model import ConstantAmbientLoss

loss = ConstantAmbientLoss(U_loss=0.3, T_ambient=10.0)
```

Constant U-value for all nodes, constant ambient temperature. Suitable for free-standing tanks.

### SplitAmbientLoss

```python
from thermal_energy_storage_model import SplitAmbientLoss

loss = SplitAmbientLoss(U_lid=0.151, U_wall=0.30, T_ambient=10.0)
# optional: separate lid ambient temperature
loss = SplitAmbientLoss(U_lid=0.151, U_wall=0.30, T_ambient=10.0, T_ambient_lid=8.0)
```

Separate U-value for lid (air, membrane liner) and wall/bottom (ground). Recommended for PTES (pit thermal energy storage). `T_ambient` can be updated dynamically via `loss.T_ambient = t_amb`.

### GroundTemperatureLoss

```python
from thermal_energy_storage_model import GroundTemperatureLoss

loss = GroundTemperatureLoss(
    U_loss=0.30, T_surface=8.0, T_deep=12.0,
    depth_decay=2.0, burial_depth=0.5,
)
```

Depth-dependent ground temperature with exponential profile. Suitable for buried tanks with known temperature profile.

### TransientGroundLoss

```python
from thermal_energy_storage_model import TransientGroundLoss

loss = TransientGroundLoss(
    U_lid=0.151,           # W/(m²·K) lid (stationary, air)
    T_ambient_lid=10.0,    # °C air temperature at lid
    lambda_soil=2.23,      # W/(m·K) soil thermal conductivity
    rho_soil=2000.0,       # kg/m³ soil density
    cp_soil=800.0,         # J/(kg·K) soil specific heat capacity
    d_total=7.5,           # m total thickness (→ U_ss = lambda/d ≈ 0.30 W/(m²K))
    n_layers=3,            # number of RC layers
    T_far=8.0,             # °C far field (deep ground)
)
```

Transient 1D RC ground network: per storage node, a chain of `n_layers` ground layers with thermal capacity. The ground state is updated each timestep using explicit Euler.

**Physical consistency:** Choose `d_total = lambda_soil / U_ss`. For Høje Taastrup: λ = 2.23 W/(m·K), U_wall = 0.30 W/(m²K) → d_total = 7.5 m, U_ss = 0.297 ≈ 0.30 ✓

**When to use:** For long-term simulations where the ground heats up significantly (e.g. PTES with multi-year operation). For annual short simulations, the difference from `SplitAmbientLoss` is small (HT 2024: ΔMAE = 0.07 K).

**Update time-varying lid temperature** during simulation:
```python
storage.config.loss_model.T_ambient_lid = current_t_amb
```

---

## 6. Initial Profile

### Uniform Temperature

```python
state = storage.initialize(T_init=70.0)   # all nodes at 70 °C
```

### Custom Profile

```python
import numpy as np
T_profile = np.linspace(90.0, 60.0, config.n_nodes)  # linear from top to bottom
state = storage.initialize(T_init=T_profile)
```

### With Ground Temperature Gradient (analogous to FreeTTES)

```python
config = StorageConfig(
    ...,
    T_ground=22.35,      # °C  Ground temperature at tank bottom
    z_ground_layer=2.0,  # m   Gradient from z=0 to z=2 m
)
```

### Profile from Measurements

```python
measured = {0.5: 22.0, 2.0: 60.0, 10.0: 60.0, 20.0: 80.0, 39.0: 90.0}
node_heights_from_bottom = np.array([...])   # node center heights
T_init = np.interp(node_heights_from_bottom,
                   sorted(measured.keys()),
                   [measured[h] for h in sorted(measured)])
state = storage.initialize(T_init=T_init)
```

---

## 7. Presets

Predefined configurations for typical storage types:

```python
# Above-ground steel tank (buffer tank, district heating network)
tes = ThermalStorage1D.from_preset(
    "steel_tank_aboveground",
    volume=500.0, height=15.0,
    U_loss=0.3, T_ambient=10.0,
)

# Buried steel tank
tes = ThermalStorage1D.from_preset(
    "steel_tank_buried",
    volume=2000.0, height=20.0,
    T_surface=8.0, T_deep=12.0, depth_deep=10.0,
)

# Pit thermal energy storage (PTES, truncated cone)
tes = ThermalStorage1D.from_preset(
    "ptes",
    r_bottom=40.0, r_top=55.0, height=15.0,
    T_surface=8.0, T_deep=12.0,
)
```

---

## 8. DiffusorModel — Inflow Distribution

Controls how a port distributes its mass and heat flux across surrounding grid nodes.

### PointDiffusor (Default)

```python
from thermal_energy_storage_model import StorageConfig, PointDiffusor

config = StorageConfig(..., diffusor_model=PointDiffusor())
```

Assigns all flow to the nearest node. Corresponds to the default behavior; computationally minimal.

### UniformDiffusor

```python
from thermal_energy_storage_model import StorageConfig, UniformDiffusor

config = StorageConfig(
    ...,
    diffusor_model=UniformDiffusor(H_zone=1.0),  # distribution over ±0.5 m around port z
)
```

Distributes flow uniformly across all nodes within a zone `H_zone` around the port position. If no node falls in the zone, falls back to the nearest node.

**Recommendation:** `UniformDiffusor(H_zone=H_diffusor)` with the physical diffusor height (e.g. 1.0 m for the FreeTTES tank). Reduces MAE at N=200 from 0.93 K to 0.67 K against FreeTTES.

### Validation Results: Diffusor Model Comparison

| Model | Diffusor | MAE T_outlet [K] | Avg time [ms/step] |
|---|---|---|---|
| N=200 Implicit | PointDiffusor | 0.93 | 0.40 |
| N=200 Implicit | UniformDiffusor | **0.67** | 0.50 |
| N=200 Explicit | PointDiffusor | 1.42 | 2.22 |
| N=200 Explicit | UniformDiffusor | 0.86 | 2.66 |
| N=50 Implicit | PointDiffusor | 0.89 | 0.15 |
| N=50 Explicit | PointDiffusor | 1.22 | 0.21 |

**Best configuration:** `solver="implicit"`, `n_nodes=200`, `UniformDiffusor(H_zone=1.0)` — 0.67 K MAE at 0.50 ms/step (~870× faster than FreeTTES).

### Custom Diffusor Model

```python
from thermal_energy_storage_model import DiffusorModel, StorageConfig
import numpy as np

class GaussianDiffusor(DiffusorModel):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def node_weights(self, port, node_heights) -> list[tuple[int, float]]:
        dists = np.abs(node_heights - port.z)
        w = np.exp(-0.5 * (dists / self.sigma) ** 2)
        w /= w.sum()
        return [(int(i), float(w[i])) for i in range(len(node_heights)) if w[i] > 1e-6]

config = StorageConfig(..., diffusor_model=GaussianDiffusor(sigma=0.5))
```

---

## 9. HeatExchangerPort — Hydraulically Decoupled Heat Transfer

A `HeatExchangerPort` transfers heat between an external circuit and the storage **without mass exchange**. Typical applications: solar collector, external heat pump, district heating substation.

### Basic Usage

```python
from thermal_energy_storage_model import HeatExchangerPort, StorageInputs

# Solar collector charges the upper part of the storage
hx_solar = HeatExchangerPort(
    z=9.0,           # HX center height [m]
    H_hx=2.0,        # Active zone: 8–10 m
    UA=5000.0,        # Overall heat transfer coefficient [W/K]
    m_dot_ext=1.2,   # Solar circuit mass flow [kg/s]
    T_ext_in=85.0,   # Collector flow temperature [°C]
    label="Solar",
)
inputs = StorageInputs(hx_ports=[hx_solar])
out = tes.step(state, dt=60.0, inputs=inputs)

# Solar circuit return temperature
T_solar_return = out.hx_outlet_temperatures[0]
```

### Combined with Hydraulic Ports

HX ports and hydraulic ports can be active simultaneously — e.g. solar charges via HX, network withdraws directly:

```python
from thermal_energy_storage_model import HeatExchangerPort, StorageInputs, Port

H = 10.0
inputs = StorageInputs(
    # Hydraulic network (direct coupling)
    ports=[
        Port(z=H,   m_dot=-5.0,            label="net_out"),
        Port(z=0.0, m_dot=+5.0, T_in=50.0, label="net_in"),
    ],
    # Solar collector (indirect heat exchanger)
    hx_ports=[
        HeatExchangerPort(z=9.0, H_hx=2.0, UA=5000.0,
                          m_dot_ext=1.2, T_ext_in=85.0, label="Solar"),
    ],
)
out = tes.step(state, dt=60.0, inputs=inputs)
T_net_out    = out.port_temperatures[0]       # Network supply temperature [°C]
T_solar_ret  = out.hx_outlet_temperatures[0]  # Solar return temperature [°C]
```

### Multiple Heat Exchangers

```python
inputs = StorageInputs(
    hx_ports=[
        HeatExchangerPort(z=9.0, H_hx=2.0, UA=5000.0,
                          m_dot_ext=1.2, T_ext_in=85.0, label="Solar"),
        HeatExchangerPort(z=1.0, H_hx=2.0, UA=3000.0,
                          m_dot_ext=2.0, T_ext_in=40.0, label="HP_return"),
    ]
)
# out.hx_outlet_temperatures = [T_Solar_ret, T_HP_supply]
```

### Physical Model: ε-NTU

```
C_ext     = ṁ_ext · cp_ext           [W/K]
NTU       = UA / C_ext               [-]
ε         = 1 − exp(−NTU)            [-]
T̄_tank   = mean T[k] in zone         [°C]
Q         = ε · C_ext · (T_ext_in − T̄_tank)    [W]
T_ext_out = T_ext_in − Q / C_ext     [°C]
```

The transferred heat flux `Q` is distributed uniformly across all nodes in the range `[z − H_hx/2, z + H_hx/2]`.

### Configuration Notes

| Parameter | Typical range | Note |
|---|---|---|
| `UA` | 500–20 000 W/K | Increases with heat exchanger area and k-value |
| `H_hx` | 0.5–5.0 m | Corresponds to physical length of the heat exchanger |
| `m_dot_ext` | 0.3–5.0 kg/s | External circuit; affects NTU and thus ε |
| `cp_ext` | 4187 (water) | Adjust for glycol mixtures |

---

## 10. Headspace Model

The optional headspace model represents the gas space above the water level in atmospheric large-scale heat storage tanks. It is relevant when:

- The storage is operated at atmospheric pressure (e.g. seasonal storage, pit storage)
- The headspace has an initial temperature different from the water temperature (e.g. due to solar irradiation on the roof)
- Closest possible agreement with the FreeTTES reference model is desired

### Simple Example

```python
config = StorageConfig(
    volume=500.0,
    height=15.0,
    n_nodes=20,
    headspace=True,           # activate headspace
    T_headspace_init=95.0,    # initial headspace temperature [°C]
    U_roof=0.2,               # roof heat transfer coefficient [W/(m²·K)]
    h_headspace_water=5.0,    # convective HTC headspace ↔ water [W/(m²·K)]
    rho_headspace=2400.0,     # concrete equivalent density [kg/m³]
    cp_headspace=880.0,       # concrete heat capacity [J/(kg·K)]
    H_headspace=0.5,          # effective headspace height [m]
)

storage = ThermalStorage1D(config)
state = storage.initialize(T_init=60.0)

outputs = storage.step(state, dt=3600.0, inputs=StorageInputs())
print(f"T_headspace: {outputs.T_headspace:.2f} °C")
```

### Behaviour

- `T_headspace` starts at `T_headspace_init` (default: 99 °C)
- The headspace temperature decreases through heat losses via the roof (`U_roof`)
- Heat transfer from headspace → water raises the temperature of node 0
- For large tanks (A_cross >> 1 m²), the thermal mass of the roof is very large → slow temperature change

### Output

```python
outputs.T_headspace        # Current headspace temperature [°C] or None
outputs.state.T_headspace  # Stored in state for the next step
```

---

## 11. Graphical User Interface (PyQt6)

![PyQt6 Interface](img/PyQt6_UI_1D_Storage.png)

The graphical interface enables interactive parameterization, real-time visualization, and export without Python knowledge.

### Installation

```bash
pip install -r requirements_ui.txt
# In addition to requirements.txt: PyQt6>=6.4.0, matplotlib>=3.7.0
```

### Launch

```bash
python run_ui.py
```

### Interface Layout

The window is divided into four areas:

| Area | Contents |
|---|---|
| **Left sidebar** | Configuration panel with tabs for Geometry, Fluid, Losses, Numerics |
| **Center top** | 3D tank visualization with temperature colour coding |
| **Right** | Plots: temperature profile T(z), time series, power, state of charge |
| **Bottom** | Simulation control: phase definition, Run/Pause/Stop, Export |

### Geometry Models

The **Geometry** tab offers three types:

- **Cylinder** – above-ground steel tanks; input: volume + height, radius is calculated
- **Truncated cone** – circular pit thermal energy storage (PTES); input: r_bottom, r_top, height
- **Truncated pyramid** – rectangular pit storage; input: half side lengths bottom/top, height (square footprint assumed)

The 3D view updates with every parameter change and scales automatically with geometry.

### Loading Presets

The preset dropdown in the upper left loads preconfigured storages:

| Preset | Geometry | Loss model |
|---|---|---|
| Steel tank, free-standing, 500 m³ | Cylinder, V=500 m³, H=15 m | Constant ambient, U=0.3 W/(m²·K) |
| Steel tank, buried, 1000 m³ | Cylinder, V=1000 m³, H=15 m | Ground temperature |
| PTES pit storage | Truncated cone, r_b=40 m, r_t=55 m, H=15 m | SplitAmbientLoss |

### Scenario Editor

A table in the lower area defines simulation phases:

| Column | Meaning |
|---|---|
| Mode | Charge / Discharge / Both / Idle |
| Duration [h] | Phase duration in hours |
| ṁ_charge / ṁ_discharge [kg/s] | Mass flows |
| T_charge_in / T_discharge_in [°C] | Inlet temperatures |
| z_charge_in … z_discharge_out [m] | Port positions (default: top/bottom) |

Rows can be added or removed via **+ Phase** and **− Phase**. Colour coding: Red = Charging, Blue = Discharging, Green = Simultaneous, Grey = Idle.

### Simulation

| Button | Function |
|---|---|
| **Start** | Starts worker thread, runs all phases sequentially |
| **Pause** | Pauses simulation, resume by clicking again |
| **Stop** | Aborts simulation |

The 3D visualization is updated at ~3 Hz during simulation (throttled to avoid UI blocking). Plots are updated in real time.

### Real-Time Plots

| Tab | Contents |
|---|---|
| **Profile T(z)** | Current temperature profile over storage height; previous phase as dashed ghost line |
| **Temperatures** | T_top, T_mid, T_bottom and T_outlet as time series |
| **Power** | Q_charge, Q_discharge, Q_loss in kW |
| **State of charge** | SOC in % (based on T_min / T_max from simulation settings) |

### Export

**File → Export Results** offers two formats:

**CSV** — one row per timestep, columns:

```
time_s, time_h, T_top, T_mid, T_bot, Q_loss, SOC, T_node_0, T_node_1, …
```

**JSON** — complete configuration + all timesteps:

```json
{
  "config": { "geometry": {...}, "loss_model": {...}, ... },
  "steps": [
    {"time_s": 0, "T": [90.0, 89.5, ...], "Q_loss": 1234.5, "SOC": 0.98},
    ...
  ]
}
```

### Save and Load Configuration

**File → Save / Load Configuration** saves the current storage parameterization as a JSON file and can be restored in a later session.

---

## 12. Example Simulation

```bash
python examples/example_simulation.py
```

Runs three scenarios:
1. **Charging only** (6 h, 8 kg/s @ 90 °C)
2. **Discharging only** (6 h, 5 kg/s @ 55 °C)
3. **Simultaneous operation** (24 h, charging and discharging at the same time)

Output: time series plots as `examples/simulation_results.svg`.

---

## 13. Benchmark: Comparison Against FreeTTES

The benchmark scripts compare the model against [FreeTTES](https://github.com/gewv-tu-dresden/FreeTTES) (Narusavicius et al. 2024), a Lagrangian segment model for thermal energy storage.

### Prerequisites

- FreeTTES installed locally; configure `FREETTTES_SRC` path at the top of the script
- Run from the project root

### Basic Usage

```bash
# Model variants comparison (N=1 to N=200):
python benchmark/benchmark_model_variants.py

# Without FreeTTES reference:
python benchmark/benchmark_model_variants.py --no-freetttes

# Multi-scenario benchmark (weekly, fast-switching):
python benchmark/benchmark_scenarios.py
python benchmark/benchmark_scenarios.py --no-freetttes
```

### Benchmark Scenario

The benchmark scenario matches the FreeTTES reference case:

| Phase | Duration | Condition |
|---|---|---|
| Charging | t = 0–23 h | 300 m³/h @ 90 °C entering at top |
| Standby | t = 24–35 h | no flow |
| Discharging | t = 36–59 h | 300 m³/h @ 60 °C entering at bottom |

Initial profile: uniform temperature or FreeTTES-compatible initialization with ground temperature gradient (T_ground = 22.35 °C at z = 0).

### Validation Results (current state)

Standard configuration (TVD + buoyancy + PointDiffusor):

| Model | MAE T_outlet [K] | Avg time [ms/step] | Speedup vs. FreeTTES |
|---|---|---|---|
| FreeTTES (reference) | — | ~437 | 1× |
| TES-N20 | 1.35 | 0.13 | ~3360× |
| TES-N50 | 1.22 | 0.25 | ~1750× |
| TES-N200 | 1.42 | 2.05 | ~213× |

**Best configuration** (Implicit + buoyancy + UniformDiffusor, N=200):

| Model | MAE T_outlet [K] | Avg time [ms/step] | Speedup vs. FreeTTES |
|---|---|---|---|
| N=200 Impl+Uniform | **0.67** | 0.50 | ~870× |

Remaining deviation (~0.7 K) is due to model differences intentionally not implemented: no Lagrangian plug flow, no foundation model, no wall thermal capacity.

---

## 14. Benchmark: Model Matrix (Ablation Study)

```bash
python benchmark/benchmark_model_variants.py
```

Tests all **model variants** combining three binary model flags (N=50):

| Dimension | Level 0 | Level 1 |
|---|---|---|
| Advection | `upwind` | `tvd` |
| Buoyancy | off (`buoyancy=False`) | on (`buoyancy=True`) |
| Fluid properties | `ConstantFluidProperties` | `WaterProperties` |

### Result (typical, with FreeTTES reference)

| Variant | MAE [K] | Avg time [ms] | Verdict |
|---|---|---|---|
| UP+B+W | 1.195 | 0.11 | Best price-performance |
| UP+B | 1.205 | 0.10 | Recommended (without W overhead) |
| TVD+B+W | 1.213 | 0.19 | Same accuracy as UP+B, twice as slow |
| TVD+B | 1.223 | 0.19 | Standard configuration |
| UP | 1.384 | 0.07 | No buoyancy: +0.18 K |
| TVD | 1.461 | 0.17 | TVD without buoyancy worse than UP+B |

**Key finding:** `buoyancy=True` is the only significant feature contribution (~0.18 K). `WaterProperties` and TVD bring marginal improvement but no relevant gain.

---

## 15. Benchmark: Named Runs & Comparisons

### Name and Save a Run

```bash
# Results persist under results/runs/v1_upwind/
python benchmark/benchmark_model_variants.py --name v1_upwind

# Next configuration for comparison
python benchmark/benchmark_model_variants.py --name v2_tvd_buoyancy
```

### List Runs

```bash
python benchmark/compare_runs.py --list
```

```
Available runs:
  v1_upwind        2026-03-14T08:12  branch: main  commit: abc1234
  v2_tvd_buoyancy  2026-03-14T09:05  branch: main  commit: def5678
```

### Compare Runs

```bash
# Tabular comparison:
python benchmark/compare_runs.py v1_upwind v2_tvd_buoyancy

# With comparison plot:
python benchmark/compare_runs.py v1_upwind v2_tvd_buoyancy --plot
```

Output: table with Δ-values (MAE, E_useful, timing) between runs.
