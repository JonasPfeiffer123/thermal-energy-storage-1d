# Graphical User Interface — User Guide

![PyQt6 Interface](img/PyQt6_UI_1D_Storage.png)

The graphical interface (`run_ui.py`) lets you configure, simulate, and export results from the 1D Thermal Storage model without writing any Python code.

---

## Table of Contents

1. [Installation & Launch](#1-installation--launch)
2. [Window Overview](#2-window-overview)
3. [Configuration Panel (Left Sidebar)](#3-configuration-panel-left-sidebar)
4. [3D Tank Visualisation (Centre)](#4-3d-tank-visualisation-centre)
5. [Plot Tabs (Right)](#5-plot-tabs-right)
6. [Simulation Control (Bottom Dock)](#6-simulation-control-bottom-dock)
7. [Running a Simulation](#7-running-a-simulation)
8. [Export](#8-export)
9. [Save & Load Configuration](#9-save--load-configuration)
10. [Presets](#10-presets)

---

## 1. Installation & Launch

Install the UI dependencies in addition to the core package:

```bash
pip install ".[ui]"
# or manually:
pip install -r requirements_ui.txt   # PyQt6>=6.4.0, matplotlib>=3.7.0
```

Launch the interface from the repository root:

```bash
python run_ui.py
```

---

## 2. Window Overview

The window is divided into four main areas:

```
┌──────────────┬───────────────────┬──────────────────────┐
│              │                   │                      │
│  Config      │   3D Tank View    │   Plot Tabs          │
│  Panel       │   (temperature    │   Profile / Temps /  │
│  (Geometry,  │    colour-coded)  │   Power / SOC        │
│  Fluid,      │                   │                      │
│  Losses,     │                   │                      │
│  Numerics)   │                   │                      │
├──────────────┴───────────────────┴──────────────────────┤
│  Simulation Control: Initial conditions, phase table,   │
│  Start / Pause / Stop, progress bar                     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Configuration Panel (Left Sidebar)

The panel is organised into four tabs. Every parameter change immediately updates the 3D view and recalculates the model configuration.

### Geometry Tab

Choose one of three geometry types from the dropdown:

| Type | Parameters | Typical use |
|---|---|---|
| **Cylinder** | Volume [m³], Height [m] | Above-ground steel tanks |
| **Truncated cone** | r_bottom [m], r_top [m], Height [m] | Circular pit storage (PTES) |
| **Truncated pyramid** | Half-side bottom [m], Half-side top [m], Height [m] | Rectangular pit storage |

The radius (for cylinder) or geometry dimensions are displayed as read-only calculated fields.

### Fluid Tab

| Parameter | Default | Description |
|---|---|---|
| Fluid model | Water Properties | Polynomial fit ρ(T), cp(T), λ(T) |
| Constant fluid | — | Fixed ρ, cp, λ (faster, less accurate) |
| λ_eff factor | 5.0 | Effective thermal conductivity multiplier (accounts for turbulent mixing) |

### Losses Tab

Select a loss model and configure its parameters:

| Model | Parameters | Use case |
|---|---|---|
| **Constant ambient** | U_loss [W/(m²·K)], T_ambient [°C] | Free-standing insulated tanks |
| **Split ambient** | U_lid, U_wall [W/(m²·K)], T_ambient [°C] | PTES (different lid vs. wall insulation) |
| **Ground temperature** | U_loss, T_surface, T_deep [°C], depth_decay [m] | Buried tanks with depth-varying ground temperature |
| **Transient ground** | U_lid, T_ambient_lid, λ_soil, ρ_soil, cp_soil, d_total, n_layers, T_far | Long-term seasonal storage with ground heat-up |

### Numerics Tab

| Parameter | Options | Description |
|---|---|---|
| **N nodes** | 5–500 | Vertical discretisation (more = more accurate, slower) |
| **Advection scheme** | TVD (default), Upwind | TVD van Leer gives 2nd-order accuracy |
| **Solver** | Explicit (default), Implicit | Implicit (TDMA) is unconditionally stable, recommended for dt > 300 s |
| **Buoyancy** | On (default) | Convective adjustment prevents temperature inversions |
| **Diffusor model** | Point (default), Uniform | Uniform distributes inflow over a configurable zone height |
| **Uniform zone H** | [m] | Active only when Uniform diffusor is selected |

---

## 4. 3D Tank Visualisation (Centre)

The 3D view renders the storage geometry as a colour-stratified body:

- **Colour scale**: RdYlBu_r — red = hot, blue = cold; scale adapts to current temperature range
- **Ports**: shown as coloured arrows
  - Red arrows: charging circuit (charge_in at top, charge_out at bottom)
  - Blue arrows: discharging circuit (discharge_out at top, discharge_in at bottom)
- **Update rate**: throttled to ~3 Hz during simulation to keep the UI responsive
- The view rotates automatically with geometry changes

---

## 5. Plot Tabs (Right)

Four tabs update in real time during simulation (Catppuccin Mocha dark theme):

| Tab | X axis | Lines plotted |
|---|---|---|
| **Profile T(z)** | Temperature [°C] | Current profile (solid), previous phase end (dashed ghost) |
| **Temperatures** | Time [h] | T_top, T_mid, T_bottom, T_outlet |
| **Power** | Time [h] | Q_charge, Q_discharge, Q_loss [kW] |
| **State of Charge** | Time [h] | SOC [%] (relative to T_min / T_max from scenario settings) |

Each tab contains a matplotlib NavigationToolbar for zoom, pan, and saving individual figures.

---

## 6. Simulation Control (Bottom Dock)

### Initial Conditions

| Field | Description |
|---|---|
| **T_init [°C]** | Uniform initial temperature of the entire storage |
| **Timestep [s]** | Simulation timestep; auto sub-stepping applies for explicit solver |

### Phase Table

Each row defines one operating phase. Phases run sequentially from top to bottom.

| Column | Description |
|---|---|
| **Mode** | Charge / Discharge / Both / Idle |
| **Duration [h]** | Phase duration |
| **ṁ_charge [kg/s]** | Charging mass flow (active when mode = Charge or Both) |
| **T_charge_in [°C]** | Charging inlet temperature |
| **ṁ_discharge [kg/s]** | Discharging mass flow (active when mode = Discharge or Both) |
| **T_discharge_in [°C]** | Discharging return temperature |
| **z_charge_in [m]** | Charging inlet port height (default: top of tank) |
| **z_charge_out [m]** | Charging outlet port height (default: bottom of tank) |
| **z_discharge_in [m]** | Discharging inlet port height (default: bottom of tank) |
| **z_discharge_out [m]** | Discharging outlet port height (default: top of tank) |

Row colour coding:
- Red = Charge only
- Blue = Discharge only
- Green = Both (simultaneous)
- Grey = Idle

**+ Phase / − Phase** buttons add or remove rows. Rows can be reordered by drag-and-drop.

### Control Buttons

| Button | Function |
|---|---|
| **Start** | Starts simulation worker thread; runs all phases sequentially |
| **Pause** | Pauses simulation; click again to resume |
| **Stop** | Aborts simulation immediately |
| **Export** | Opens the export dialog (available after simulation completes) |

A progress bar shows the fraction of total simulation time completed. The status bar shows the current simulation time, current phase, and last computed outlet temperatures.

---

## 7. Running a Simulation

1. **Configure** the storage geometry, fluid, losses, and numerics in the left panel.
2. **Set** initial temperature and timestep in the simulation control.
3. **Define phases** in the phase table (add rows with **+ Phase**).
4. Click **Start**.
5. Observe the 3D view and plots updating in real time.
6. Click **Stop** to abort or wait for all phases to finish.

**Tip:** Use the Implicit solver with a larger timestep (e.g. 3600 s) for multi-day or seasonal scenarios. Use the Explicit solver with a smaller timestep (60–300 s) for accurate short-term dynamics.

**Tip:** The status bar shows the recommended maximum stable timestep for the explicit solver based on the current mass flow and grid resolution.

---

## 8. Export

After a simulation has run, **File → Export Results** (or the Export button) opens a dialog with two format options:

### CSV

One row per timestep:

```
time_s, time_h, T_top, T_mid, T_bot, Q_loss, SOC, T_node_0, T_node_1, …
```

- `T_node_0` … `T_node_N-1` are the full nodal temperature profile (index 0 = top)
- Suitable for post-processing in Excel, pandas, etc.

### JSON

Complete configuration plus all timesteps:

```json
{
  "config": {
    "geometry": { "type": "cylinder", "volume": 100.0, "height": 5.0 },
    "loss_model": { "type": "constant", "U_loss": 0.3, "T_ambient": 10.0 },
    "n_nodes": 20,
    "solver": "explicit",
    "advection_scheme": "tvd"
  },
  "steps": [
    { "time_s": 0.0, "T": [60.0, 60.0, ...], "Q_loss": 0.0, "SOC": 0.0 },
    { "time_s": 60.0, "T": [60.1, 60.0, ...], "Q_loss": 152.3, "SOC": 0.01 },
    ...
  ]
}
```

- Useful for reproducibility and comparing different configurations
- The `config` block can be loaded back via **File → Load Configuration**

---

## 9. Save & Load Configuration

**File → Save Configuration** saves the current storage parameterization (geometry, fluid, losses, numerics) as a JSON file — **not** the simulation results.

**File → Load Configuration** restores a previously saved configuration. All panel widgets update to reflect the loaded values, and the 3D view refreshes.

---

## 10. Presets

The **Presets** dropdown in the upper-left loads preconfigured storages:

| Preset | Geometry | Loss model |
|---|---|---|
| Steel tank, free-standing, 500 m³ | Cylinder, V=500 m³, H=15 m | Constant ambient, U=0.3 W/(m²·K) |
| Steel tank, buried, 1000 m³ | Cylinder, V=1000 m³, H=15 m | Ground temperature profile |
| PTES pit storage | Truncated cone, r_b=40 m, r_t=55 m, H=15 m | SplitAmbientLoss |

Loading a preset populates all configuration tabs and updates the 3D view. You can then fine-tune individual parameters before running a simulation.
