# Physical Model: Description and Derivation

This document explains the physical effects modelled in the 1D thermal energy storage model and justifies each modelling decision with literature references. It is intended for readers who want to describe or reproduce the model in a scientific context.

---

## Table of Contents

1. [Physical Overview](#1-physical-overview)
2. [Geometry and Discretization](#2-geometry-and-discretization)
3. [Energy Balance: the Governing Equation](#3-energy-balance-the-governing-equation)
4. [Convective Heat Transport (Advection)](#4-convective-heat-transport-advection)
5. [Axial Heat Conduction and Effective Thermal Conductivity](#5-axial-heat-conduction-and-effective-thermal-conductivity)
6. [Thermal Stratification and Buoyancy](#6-thermal-stratification-and-buoyancy)
7. [Wall Heat Losses](#7-wall-heat-losses)
8. [Hydraulic Port Model](#8-hydraulic-port-model)
9. [Fluid Properties](#9-fluid-properties)
10. [Numerical Time Integration and Stability](#10-numerical-time-integration-and-stability)
11. [Geometry Variants](#11-geometry-variants)
12. [Advanced Loss Models](#12-advanced-loss-models)
13. [Diffusor Models](#13-diffusor-models)
14. [Heat Exchanger Port (ε-NTU)](#14-heat-exchanger-port-ε-ntu)
15. [Headspace Model](#15-headspace-model)
16. [References](#16-references)

---

## 1. Physical Overview

### What do we model?

A thermal energy storage (TES) for district heating networks is essentially a large water-filled vessel. It serves as a buffer between heat sources (e.g. CHP plants, solar thermal collectors) and the distribution network: excess heat is stored and demand peaks are covered.

The key physical property that characterises a good storage is **thermal stratification**: hot water is lighter than cold water and therefore floats on top. A well-stratified storage has a hot zone at the top and a cold zone at the bottom, separated by a thin transition layer called the **thermocline**. The sharper the thermocline, the more usable energy is available during discharge [Dahash et al. 2020, Lemelle et al. 2025].

### Why a 1D model?

For practical simulations — especially when coupled with district heating network software — 3D CFD simulations are too computationally expensive. In cylindrical or conical tanks, the vertical temperature stratification dominates; radial gradients are negligible by comparison. Therefore, a **one-dimensional finite-volume model** along the vertical axis is the most common approach [Falkner 2014, Xiang et al. 2022, Untrau et al. 2023].

The fundamental assumption is **plug flow**: the fluid moves with uniform horizontal velocity across the full cross-section, without lateral mixing. This assumption holds as long as thermal buoyancy forces clearly dominate inertia forces, i.e. for large **Richardson numbers** Ri = Gr/Re² ≫ 1. For large-scale district heating storage tanks this condition is practically always satisfied [Weiss et al. 2021, Xiang et al. 2022].

### Overview of Modelled Effects

The 1D model accounts for four physical effects:

| Effect | Physical cause | Model term |
|---|---|---|
| **Advection** | Heat transport by mass flow (charging/discharging) | $\dot{Q}_\text{adv}$ |
| **Axial conduction** | Molecular diffusion + turbulent mixing at the thermocline | $\dot{Q}_\text{cond}$ |
| **Buoyancy** | Density-driven convection under inverse stratification | Convective adjustment |
| **Wall losses** | Heat transfer through the tank wall to the surroundings | $\dot{Q}_\text{loss}$ |

---

## 2. Geometry and Discretization

### Node Layout

The storage is divided along the vertical axis into **N equal-volume nodes**. Each node represents a horizontal water layer of height Δz = H/N:

```
      ┌─────────────┐
  0   │  T[0]       │  ← top (hot), z = H
      ├─────────────┤
  1   │  T[1]       │
      ├─────────────┤
  ... │   ...       │
      ├─────────────┤
 N-1  │  T[N-1]     │  ← bottom (cold), z = 0
      └─────────────┘
```

**Index convention:** Index 0 is at the top (hot), index N−1 is at the bottom (cold). The coordinate z increases upward (z = 0 at the bottom, z = H at the lid).

Geometric quantities per node:

| Quantity | Formula | Unit |
|---|---|---|
| Node height | $\Delta z = H / N$ | m |
| Node volume | $V_k = V / N$ | m³ |
| Node mass | $m_k = \rho \cdot V_k$ | kg |
| Cross-sectional area | $A = V / H$ | m² |

### Choice of Number of Nodes N

Jäger et al. [2024] and Untrau et al. [2023] show that at least N = 40–100 layers are needed to reliably represent the thermocline: below N ≈ 10 the stored energy can be underestimated by up to 5 MWh (for 500 m³). For large-scale network simulations, N = 20–50 is a practical compromise between accuracy and computation time [Jäger et al. 2024, Lemelle et al. 2025].

---

## 3. Energy Balance: the Governing Equation

### Continuous Transport Equation

The physical basis is the 1D heat transport equation derived from Reynolds' transport theorem for a flowing fluid [Falkner 2014, Untrau et al. 2023]:

$$\rho \, c_p \, A \, \frac{\partial T}{\partial t} = -\dot{m} \, c_p \, \frac{\partial T}{\partial z} + \frac{\partial}{\partial z}\!\left(\lambda_\text{eff} \cdot A \cdot \frac{\partial T}{\partial z}\right) - U \cdot P \cdot (T - T_\text{amb})$$

The three terms on the right-hand side describe:
1. **Advection**: heat transport by the mass flow $\dot{m}$
2. **Axial diffusion**: conduction and turbulent mixing
3. **Wall losses**: heat dissipation through the tank wall with perimeter $P$

**Symbols:**

| Symbol | Meaning | Unit |
|---|---|---|
| $\rho$ | Fluid density | kg/m³ |
| $c_p$ | Specific heat capacity of the fluid | J/(kg·K) |
| $A$ | Cross-sectional area of the storage ($= V/H$) | m² |
| $T$ | Temperature | °C |
| $t$ | Time | s |
| $z$ | Vertical coordinate (0 at bottom, H at lid) | m |
| $\dot{m}$ | Mass flow in the storage (positive = downward) | kg/s |
| $\lambda_\text{eff}$ | Effective thermal conductivity (Section 5) | W/(m·K) |
| $U$ | Overall heat transfer coefficient of the tank wall | W/(m²·K) |
| $P$ | Perimeter of the storage cross-section ($= 2\pi r$) | m |
| $T_\text{amb}$ | Ambient temperature | °C |

### Discretized Energy Balance per Node

After spatial discretization into N control volumes, the energy balance for each node i is:

$$m_k \cdot c_p \cdot \frac{dT_i}{dt} = \dot{Q}_{\text{adv},i} + \dot{Q}_{\text{cond},i} + \dot{Q}_{\text{loss},i}$$

The three heat flows are derived in the following sections.

**Symbols:**

| Symbol | Meaning | Unit |
|---|---|---|
| $m_k = \rho \cdot V/N$ | Mass of one node | kg |
| $c_p$ | Specific heat capacity of the fluid | J/(kg·K) |
| $T_i$ | Temperature at node i | °C |
| $\dot{Q}_{\text{adv},i}$ | Advective heat flow (Section 4) | W |
| $\dot{Q}_{\text{cond},i}$ | Conductive heat flow (Section 5) | W |
| $\dot{Q}_{\text{loss},i}$ | Wall heat loss (Section 7) | W |

---

## 4. Convective Heat Transport (Advection)

### Physical Background

During charging, hot water enters at the top of the storage and cool water exits at the bottom. During discharging, the reverse occurs. This mass transport carries enthalpy — this is **advection**. It is by far the dominant heat transport mechanism during charging and discharging [Rendall et al. 2021].

Numerical methods are required to discretize the advection term. The simplest stable scheme is the **upwind scheme** (1st order); a more accurate alternative is the **TVD scheme** (Total Variation Diminishing) with van Leer limiter (2nd order in smooth regions).

### Port-Based Formulation

The model uses a general **port system**: each hydraulic connection injects or withdraws mass at a defined height position. The implementation works in three steps:

**Step 1 — Mass flux vector:** From the source terms $S[i]$ (kg/s per node, positive = inlet), the cumulative inter-node mass flux $F[j]$ is computed:

$$F[j] = \sum_{i=0}^{j-1} S[i], \quad j = 0, \ldots, N$$

with boundary conditions $F[0] = F[N] = 0$ (no flux through lid/bottom).

**Step 2 — Upwind interface flux:** At each node boundary j:
- If $F[j] > 0$ (flow downward): upwind temperature = $T[j-1]$ (upper node)
- If $F[j] < 0$ (flow upward): upwind temperature = $T[j]$ (lower node)

$$\dot{Q}_{\text{face},j} = F[j] \cdot c_p \cdot T_{\text{upwind}}[j]$$

**Step 3 — Node energy balance:**

$$\dot{Q}_{\text{adv},i} = \dot{Q}_{\text{face},i} - \dot{Q}_{\text{face},i+1} + \dot{Q}_{\text{source},i}$$

The source term $\dot{Q}_{\text{source},i}$ contributes the enthalpy of inflowing fluid:

$$\dot{Q}_{\text{source},i} = c_p \cdot \bigl(\max(S_i, 0) \cdot T_{\text{src},i} + \min(S_i, 0) \cdot T_i\bigr)$$

Inlet ports contribute at the inlet temperature $T_\text{src}$; outlet ports withdraw enthalpy at the local node temperature $T_i$. This formulation ensures exact mass balance at every node.

### Upwind Scheme (1st Order)

The upwind scheme is the simplest stable discretization for convection-dominated problems. It is energy-conserving and maintains thermal stratification, but causes gradual thermocline smearing over time due to **numerical diffusion**. The numerical diffusion acts similarly to increased axial thermal conductivity [Untrau et al. 2023].

| Property | Value |
|---|---|
| Order | 1st order (spatial) |
| Stability condition | CFL ≤ 1 |
| Thermocline | Moderate numerical diffusion |
| Recommendation | Simple applications, coarse grids |

### TVD Scheme with van Leer Limiter (2nd Order)

The **TVD scheme** (Total Variation Diminishing) with van Leer limiter extends the upwind scheme to 2nd order. It significantly reduces numerical diffusion without producing physically inadmissible over- or undershoots, maintaining a much sharper thermocline.

The TVD scheme adds a correction to the upwind flux:

$$F_{\text{TVD}} = F_{\text{upwind}} + \tfrac{1}{2} \, \phi(r) \cdot (1 - \text{CFL}) \cdot (T_{\text{down}} - T_{\text{upwind}})$$

The **van Leer limiter** $\phi(r)$ constrains the correction based on the local gradient ratio $r$:

$$\phi(r) = \frac{r + |r|}{1 + |r|}, \qquad r = \frac{T_{\text{upwind}} - T_{\text{far-upwind}}}{T_{\text{down}} - T_{\text{upwind}}}$$

- In smooth regions ($r \approx 1$): $\phi \approx 1$ → full 2nd order
- At discontinuities ($r \ll 1$ or $r < 0$): $\phi \to 0$ → falls back to upwind (stable)

**Symbols:**

| Symbol | Meaning | Unit |
|---|---|---|
| $\phi(r)$ | van Leer limiter, constrains correction to [0, 1] | — |
| $r$ | Ratio of upwind-to-local temperature gradients | — |
| $T_{\text{upwind}}$ | Temperature in the upwind node | °C |
| $T_{\text{down}}$ | Temperature in the downwind node | °C |
| $T_{\text{far-upwind}}$ | Temperature in the second-upwind node | °C |
| CFL | Courant number $= v \cdot \Delta t / \Delta z$ | — |

| Property | Value |
|---|---|
| Order | 2nd order (smooth regions), 1st order (discontinuities) |
| Stability condition | CFL ≤ 1 |
| Thermocline | Sharp, no overshoots |
| Recommendation | Default for all applications |

```python
config = StorageConfig(..., advection_scheme="tvd")    # default
config = StorageConfig(..., advection_scheme="upwind")
```

### Model Limitation: Inlet Mixing and Diffusor Effects

The port model assumes that inflowing fluid **immediately distributes horizontally** across the full cross-section — the plug-flow assumption. In practice this is only satisfied with an ideal diffusor and high Richardson number.

With a poor diffusor or high inlet momentum, the fluid enters as a **free jet** and seeks its **neutral buoyancy level** — the height at which its (mixing-modified) temperature matches the local storage temperature:

```
Inlet (top, hot)
      │
      ▼  momentum jet
 ─────────────────    ← geometric inlet position
 ░░░░░░░░░░░░░░░░░    ← mixing with ambient fluid
      │
      ▼  (buoyancy decelerates jet)
 ─────────────────    ← neutral buoyancy level: fluid spreads out
```

The effective inflow height is therefore **lower** than the geometric inlet position. The thermocline thickens from the start more than the 1D model predicts.

Weiss et al. [2021] quantify this effect by CFD: an ideal plug-flow model underestimates thermocline thickness by **more than a factor of 2** compared to experimental measurements. Rendall et al. [2021] show that this inlet mixing effect far exceeds the influence of axial thermal conductivity ($\lambda_\text{eff}$) at large flow rates.

**Consequence for the model:** The inlet mixing effect cannot be directly represented in the 1D framework. It is implicitly contained in the calibration parameter $f_\lambda$: an increased value of $f_\lambda$ compensates for the additional thermocline smearing caused by the inlet jet. This is a **calibration parameter**, not physics. FreeTTES [Narusavicius et al. 2024] circumvents this limitation by explicitly modelling an inversion plume — a Lagrangian cell that exchanges mass with its surroundings until it reaches its equilibrium height. This is an inherent accuracy limitation of all 1D Euler models.

---

## 5. Axial Heat Conduction and Effective Thermal Conductivity

### Physical Background

In addition to advection, there is always diffusive heat exchange between adjacent water layers. This has two causes:

1. **Molecular thermal conductivity** of the fluid (water: λ ≈ 0.63 W/(m·K) at 60 °C)
2. **Turbulent mixing** at the thermocline: even under plug flow, inlet jets, wall friction, and density gradients produce some turbulence that increases effective heat transport

These effects cannot be separated exactly and are therefore combined into an **effective thermal conductivity** [Falkner 2014]:

$$\lambda_\text{eff} = f_\lambda \cdot \lambda_\text{fluid}$$

The empirical factor $f_\lambda > 1$ (default: 5) accounts for turbulence at the thermocline. Falkner [2014] derives an equivalent formulation from the wall conductivity contribution:

$$\lambda_\text{eff} = \lambda_\text{fluid} + \lambda_\text{wall} \cdot \frac{A_\text{wall}}{A_\text{axial}}$$

For large tanks (small $A_\text{wall}/A_\text{axial}$), fluid conductivity dominates; for small tanks $\lambda_\text{eff}$ can reach 2–5 times $\lambda_\text{fluid}$. Identical formulations appear in Jäger et al. [2024], Untrau et al. [2023], and Lemelle et al. [2025], confirming the generality of this approach.

> **Important note on thermocline:** Untrau et al. [2023] point out that in 1D upwind models with few nodes, **numerical diffusion** far exceeds physical axial conduction. This means: the choice of $f_\lambda$ has practically no influence on thermocline sharpness for coarse grids (N < 50) — numerical diffusion dominates. Only with fine grids (N ≥ 100, TVD scheme) does $\lambda_\text{eff}$ become a relevant calibration parameter.

### Discretized Conduction Term

The conductive heat flux between adjacent nodes is approximated as a central difference:

$$\dot{Q}_{\text{cond},i} = K_{\text{cond}} \cdot (T_{i-1} - T_i) + K_{\text{cond}} \cdot (T_{i+1} - T_i)$$

with the thermal conductance:

$$K_{\text{cond}} = \frac{\lambda_{\text{eff}} \cdot A}{\Delta z} \quad [\text{W/K}]$$

**Boundary condition:** No conductive flux beyond the lid and bottom (adiabatic end nodes).

**Symbols:**

| Symbol | Meaning | Unit |
|---|---|---|
| $K_{\text{cond}}$ | Thermal conductance between two nodes | W/K |
| $\lambda_{\text{eff}} = f_\lambda \cdot \lambda_\text{fluid}$ | Effective thermal conductivity | W/(m·K) |
| $f_\lambda$ | Empirical amplification factor (default: 5) | — |
| $A = V/H$ | Cross-sectional area | m² |
| $\Delta z = H/N$ | Node height | m |

---

## 6. Thermal Stratification and Buoyancy

### Physical Background

The density of water decreases with increasing temperature. A well-stratified storage is therefore naturally stable: the lighter hot water lies on top, the heavier cold water at the bottom. However, if — for example due to numerical errors or certain operating conditions — **inverse stratification** occurs (cold above warm), this is hydrodynamically unstable.

The Rayleigh criterion states that free convection sets in as soon as Ra > Ra$_\text{crit}$ ≈ 1708. Falkner [2014] shows that for water at 70 °C, even the smallest temperature differences (proportional to 1/D³) suffice to exceed this threshold — in any real tank, convection therefore occurs immediately whenever inverse stratification is present. Xiang et al. [2022] confirm this as a standard result in PTES literature.

Inverse stratification must therefore be **actively corrected** after each timestep.

### Convective Adjustment (Caloric Mixing)

After each timestep, the temperature profile is checked for stability (monotonically decreasing from top to bottom). Unstable node pairs are immediately set to the energetically correct mixing temperature:

1. Check each adjacent node pair $(i, i+1)$ from top to bottom
2. If $T[i] < T[i+1]$ (cold above warm → unstable): mix both nodes:

$$T_{\text{mix}} = \frac{m_i \cdot T_i + m_{i+1} \cdot T_{i+1}}{m_i + m_{i+1}}$$

3. Repeat until the entire profile is monotone

The algorithm is **energy-conserving** and has **O(N) complexity**. This method is known in the literature as *caloric mixing* or *caloric complete-mixing* [Falkner 2014, Jäger et al. 2024] and is the standard approach for 1D PTES models [Xiang et al. 2022].

**Symbols:**

| Symbol | Meaning | Unit |
|---|---|---|
| $T_{\text{mix}}$ | Mixing temperature after convective adjustment | °C |
| $m_i = \rho \cdot V/N$ | Mass of node i | kg |

```python
config = StorageConfig(..., buoyancy=True)    # default
config = StorageConfig(..., buoyancy=False)   # disable
```

> **Benchmark result:** The buoyancy correction reduces MAE against FreeTTES from ~1.38 K to ~1.20 K (N=50, TVD+buoyancy+WaterProperties).

---

## 7. Wall Heat Losses

### Physical Background

No real storage is perfectly insulated. Heat is continuously dissipated through the tank wall (lateral surface, lid, bottom) to the surroundings. For district heating storage tanks in central Europe, wall losses are a significant efficiency factor: Dahash et al. [2020] show for the Dronninglund PTES (60 000 m³) that annual wall losses amount to approximately 1 275 MWh, with the lid area dominating at 788 MWh.

Heat flux through the tank wall is modelled as linear heat transfer [Sterner & Stadler 2017]:

$$\dot{Q}_{\text{loss},i} = U \cdot A_{\text{wall},i} \cdot (T_{\text{amb}} - T_i)$$

### Loss Areas per Node

The wall area per node consists of:
- **Lateral surface** (all nodes): $A_{\text{lat},i} = 2\pi r \cdot \Delta z$
- **Lid** (node 0, top): additionally $+\pi r^2$
- **Bottom** (node N−1, bottom): additionally $+\pi r^2$

### Model 1: Constant Ambient Temperature (`ConstantAmbientLoss`)

The simplest model uses a uniform ambient temperature:

$$\dot{Q}_{\text{loss},i} = U \cdot A_{\text{wall},i} \cdot (T_{\text{amb}} - T_i)$$

**Symbols:**

| Symbol | Meaning | Unit |
|---|---|---|
| $U$ | Overall heat transfer coefficient of the tank wall | W/(m²·K) |
| $A_{\text{wall},i}$ | Wall area of node i | m² |
| $T_{\text{amb}}$ | Ambient temperature | °C |

Typical values: $U = 0.163 \ldots 3.3\ \mathrm{W/(m^2 K)}$ for various wall constructions [Sterner & Stadler 2017]; $U = 0.2\ \mathrm{W/(m^2 K)}$ for large atmospheric tanks [Dahash et al. 2020].

```python
loss_model = ConstantAmbientLoss(U_loss=0.2, T_ambient=10.0)
```

### Model 2: Depth-Dependent Ground Temperature (`GroundTemperatureLoss`)

For buried storage tanks (PTES), the ambient temperature varies with depth. A linear profile is assumed:

$$T_{\text{ground}}(d) = T_{\text{surf}} + (T_{\text{deep}} - T_{\text{surf}}) \cdot \frac{d}{d_{\text{deep}}}$$

**Symbols:**

| Symbol | Meaning | Unit |
|---|---|---|
| $d$ | Depth below ground surface | m |
| $T_{\text{surf}}$ | Temperature at the ground surface | °C |
| $T_{\text{deep}}$ | Temperature in the deep ground (below $d_{\text{deep}}$) | °C |
| $d_{\text{deep}}$ | Depth at which $T_{\text{deep}}$ applies | m |

---

## 8. Hydraulic Port Model

### Concept

Each hydraulic connection is described as a `Port` object:

```python
@dataclass
class Port:
    z: float        # Height above tank bottom [m]
    m_dot: float    # Mass flow [kg/s]: positive = inlet, negative = outlet
    T_in: float     # Inlet temperature [°C], relevant only for m_dot > 0
    label: str      # Label
```

The model accepts **any number of ports** at arbitrary height positions — this allows modelling not only the classic two-circuit operation but also plants with solar thermal feed-in at mid-height or multiple consumer circuits.

| Operating mode | Ports |
|---|---|
| Charging only | Inlet top (+), outlet bottom (−) |
| Discharging only | Inlet bottom (+), outlet top (−) |
| Simultaneous | All four ports active |
| Standby | No ports |
| Solar + network | 3–4 ports at various heights |

### Convenience Constructor

```python
# Charging only
inputs = StorageInputs.two_port(
    m_dot_charge=8.0, T_charge_in=90.0, height=10.0,
)

# Simultaneous with explicit port positions
inputs = StorageInputs.two_port(
    m_dot_charge=8.0,    T_charge_in=90.0,
    m_dot_discharge=5.0, T_discharge_in=55.0,
    height=10.0,
    z_charge_in=9.5,    z_charge_out=1.0,
    z_discharge_in=1.0, z_discharge_out=9.5,
)
```

### Port Model Limitations

The port model assumes ideal flow distribution (plug flow) at the inlet. Weiss et al. [2021] show by CFD that real inlet diffusors lead to a significantly thicker thermocline than the idealised 1D model: the difference is a factor > 2. This systematic underestimation of thermocline thickness is an inherent limitation of all 1D models and must be considered when making accuracy statements.

---

## 9. Fluid Properties

### Constant Fluid Properties (`ConstantFluidProperties`)

The classic modelling approach uses constant, temperature-independent fluid properties:

```python
fluid = ConstantFluidProperties(rho=977.8, cp=4187.0, lambda_fluid=0.663)
```

For water in the range 20–90 °C, density varies by approximately ±2 %, specific heat capacity by less than ±1 % — the approximation is sufficient for most applications.

### Temperature-Dependent Fluid Properties (`WaterProperties`)

For higher accuracy, especially when comparing with reference models, temperature-dependent polynomial correlations are used (matching FreeTTES coefficients, valid approximately 0–100 °C):

| Quantity | Polynomial | Unit |
|---|---|---|
| $\rho(T)$ | $-2.526 \cdot 10^{-3} T^2 - 0.2123 \, T + 1005.0$ | kg/m³ |
| $c_p(T)$ | $9.777 \cdot 10^{-3} T^2 - 0.7677 \, T + 4194.8$ | J/(kg·K) |
| $\lambda(T)$ | 3rd order polynomial | W/(m·K) |

```python
fluid = WaterProperties()
```

> The influence on simulation accuracy is small: in the 60-h benchmark scenario, the MAE difference between `ConstantFluidProperties` and `WaterProperties` is less than 2 %.

---

## 10. Numerical Time Integration and Stability

### Explicit Euler (Default)

Time integration uses the **explicit Euler method**:

$$T_i(t + \Delta t) = T_i(t) + \frac{\Delta t}{m_k \cdot c_p} \cdot \left(\dot{Q}_{\text{adv},i} + \dot{Q}_{\text{cond},i} + \dot{Q}_{\text{loss},i}\right)$$

The explicit Euler method is simple and efficient but requires the CFL stability condition to be satisfied.

### CFL Stability Condition

The explicit Euler method with upwind discretization is only stable when the **Courant-Friedrichs-Lewy (CFL)** condition is satisfied [Falkner 2014, Jäger et al. 2024]:

$$\text{CFL} = \frac{v \cdot \Delta t}{\Delta z} \leq 1, \qquad v = \frac{\dot{m}_{\text{net}}}{\rho \cdot A}$$

This means physically: the fluid may travel at most one node height $\Delta z$ per timestep. The maximum stable timestep is:

$$\Delta t_{\max} = 0.9 \cdot \frac{m_k}{\dot{m}_{\max}}$$

(Safety factor 0.9 against the theoretical limit CFL = 1.)

**Typical values (N = 50):**

| Storage | Volume | $\dot{m}_{\max}$ | $\Delta t_{\max}$ |
|---|---|---|---|
| Buffer tank | 50 m³ | 5 kg/s | ~130 s |
| Network heat storage | 500 m³ | 20 kg/s | ~320 s |
| Large tank (FreeTTES benchmark) | 50 000 m³ | 82 kg/s | ~3 100 s |

If the external timestep $\Delta t_\text{ext}$ exceeds the CFL limit, **sub-stepping** is applied automatically *inside* `step()` (config flag `auto_substep`, default `True`). The model computes the smallest number of equal sub-steps that keeps each sub-step at $\text{CFL} \leq 0.9$ and integrates them sequentially:

$$n_\text{sub} = \left\lceil \frac{\text{CFL}(\Delta t_\text{ext})}{0.9} \right\rceil, \qquad \Delta t_\text{sub} = \frac{\Delta t_\text{ext}}{n_\text{sub}}$$

The publicly returned state advances by the full external $\Delta t_\text{ext}$; outlet quantities (`port_temperatures`, `hx_outlet_temperatures`) and `Q_loss` are returned as the mean over the sub-steps, which represents the external coupling interval more faithfully than a single start-of-step evaluation. The caller therefore does not need to respect the CFL limit:

```python
config = StorageConfig(..., solver="explicit", auto_substep=True)   # default
outputs = storage.step(state, dt=3600.0, inputs=inputs)             # any dt is stable
```

Setting `auto_substep=False` disables this: a CFL violation then only emits a `RuntimeWarning` and the single (potentially unstable) explicit step is taken — useful for reproducing the raw single-step behaviour. The implicit solver is unconditionally stable and ignores this flag.

### Implicit Solver (TDMA)

For very large timesteps (e.g. hourly values without sub-stepping), a **fully implicit solver** is available (`solver="implicit"`). Advection and conduction are evaluated at the new time level $t + \Delta t$ — the method is therefore **unconditionally stable** (no CFL constraint).

The fully implicit energy balance for node i is:

$$\frac{C_i}{\Delta t}(T_i^{\text{new}} - T_i) = Q_{\text{adv},i}^{\text{impl}} + Q_{\text{cond},i}^{\text{impl}} + Q_{\text{loss},i}^{\text{expl}}$$

The resulting **tridiagonal system** (structure: $a_i T_{i-1}^{\text{new}} + d_i T_i^{\text{new}} + c_i T_{i+1}^{\text{new}} = b_i$) is solved with the **Thomas algorithm (TDMA)** in O(N).

```python
config = StorageConfig(..., solver="explicit")   # default
config = StorageConfig(..., solver="implicit")
```

---

## 11. Geometry Variants

### Cylindrical Tank (`CylinderGeometry`)

Standard for above-ground steel and concrete tanks. Constant cross-section at all heights.

```python
geom = CylinderGeometry.from_volume(volume=500.0, height=15.0)
```

### Truncated Cone (`TruncatedConeGeometry`)

For **pit thermal energy storage (PTES)**, a truncated cone is the typical geometry: the slope angle is typically 30° for V > 10 000 m³ [Xiang et al. 2022]. Cross-sectional and wall areas vary with height; the model integrates these numerically for each node.

```python
geom = TruncatedConeGeometry(r_bottom=40.0, r_top=55.0, height=15.0)
```

### Truncated Pyramid (`TruncatedPyramidGeometry`)

For PTES with **rectangular footprint** — the most common PTES geometry in practice (e.g. Dronninglund, Danish pit thermal energy storages). Cross-sectional area varies quadratically with height:

$$A(z) = a(z) \cdot b(z), \qquad a(z) = a_\text{bot} + (a_\text{top} - a_\text{bot}) \cdot \frac{z}{H}$$

Volume is computed analytically:

$$V = \frac{H}{6} \cdot (2\,a_t b_t + 2\,a_b b_b + a_t b_b + a_b b_t)$$

The wall area of each segment accounts for the inclination of the embankment walls (slant height rather than vertical height). Typical slope: 1:2 (1 m vertical, 2 m horizontal) [Xiang et al. 2022].

```python
geom = TruncatedPyramidGeometry(
    a_bottom=26.0, b_bottom=26.0,   # Bottom side lengths [m]
    a_top=91.0,    b_top=91.0,      # Lid side lengths [m]
    height=16.0,                    # Dronninglund dimensions
)
# Shorthand via slope:
geom = TruncatedPyramidGeometry.from_slope(
    a_bottom=100.0, b_bottom=100.0, height=10.0, slope=2.0
)
```

---

## 12. Advanced Loss Models

### Model 3: Separate U-Values for Lid and Wall (`SplitAmbientLoss`)

For PTES, the heat transfer coefficients of the **lid** (exposed, air) and **buried walls/bottom** (ground) differ significantly. Dahash et al. [2020] calibrate for Dronninglund $U_\text{lid} = 0.186\ \mathrm{W/(m^2 K)}$ and a separate ground conductivity for the sidewalls. This model separates both:

$$\dot{Q}_{\text{loss},i} = \begin{cases} U_\text{lid} \cdot A_{\text{wall},0} \cdot (T_\text{amb,lid} - T_0) & i = 0\ \text{(lid node)} \\ U_\text{wall} \cdot A_{\text{wall},i} \cdot (T_\text{amb} - T_i) & i > 0 \end{cases}$$

Optionally, a separate ambient temperature $T_\text{amb,lid}$ (air temperature) can be specified for the lid, while $T_\text{amb}$ describes the ground temperature for walls and bottom.

```python
loss_model = SplitAmbientLoss(
    U_lid=0.186, U_wall=0.35,
    T_ambient=8.0, T_ambient_lid=12.0,
)
```

### Model 4: Transient Ground Network (`TransientGroundLoss`)

The three preceding models treat the ground as a **stationary boundary condition** — the ground temperature is given and does not change. In reality, the ground around the storage warms up over years of operation: a newly commissioned storage loses more heat to the ground than after years of operation when the surrounding ground has already warmed.

The transient ground model simulates this effect with a **1D RC chain**: between the tank wall and the far, undisturbed ground, $n$ ground layers with thermal mass and thermal resistance are arranged. Their temperatures develop dynamically:

$$C \cdot \frac{dT_{g,i,j}}{dt} = \frac{T_{j-1} - T_{g,i,j}}{R} - \frac{T_{g,i,j} - T_{j+1}}{R}$$

with boundary conditions:

$$T_{j=0} = T_{\text{storage},i} \quad \text{(tank wall)}, \qquad T_{j=n+1} = T_\text{far} \quad \text{(unchanged far field)}$$

The heat flux into the storage follows from the first ground layer:

$$\dot{Q}_{\text{loss},i} = A_{\text{wall},i} \cdot \frac{T_{g,i,1} - T_{\text{storage},i}}{R}$$

**Symbols:**

| Symbol | Meaning | Unit |
|---|---|---|
| $C = \rho_\text{soil} \cdot c_{p,\text{soil}} \cdot d_\text{layer}$ | Thermal capacity per area of one ground layer | J/(K·m²) |
| $R = d_\text{layer} / \lambda_\text{soil}$ | Thermal resistance per area | K·m²/W |
| $T_{g,i,j}$ | Temperature of ground layer j adjacent to storage node i | °C |
| $T_\text{far}$ | Deep ground temperature (far field, unchanged) | °C |
| $\lambda_\text{soil}$ | Thermal conductivity of the ground | W/(m·K) |

The lid node continues to be treated as stationary (no ground above the lid).

```python
loss_model = TransientGroundLoss(
    U_lid=0.151, T_ambient_lid=10.0,
    lambda_soil=2.23, rho_soil=2000.0, cp_soil=800.0,
    d_total=10.0, n_layers=4, T_far=8.0,
)
```

---

## 13. Diffusor Models

The diffusor model determines how a hydraulic port distributes its mass flow spatially across the grid nodes. It is separate from the advection scheme (Upwind/TVD) and can be configured independently.

### Point Diffusor (`PointDiffusor`, Default)

The entire mass flow is concentrated at the **nearest node**. This corresponds to the plug-flow assumption: the fluid distributes immediately horizontally across the full cross-section. Suitable for coarse grids (N ≤ 20) and reference runs.

### Uniform Diffusor (`UniformDiffusor`)

The mass flow is distributed **uniformly** across all nodes within a diffusor zone of height $H_\text{zone}$:

$$S_k = \frac{\dot{m}}{n_\text{zone}}, \quad \text{for all } k \text{ with } |z_k - z_\text{port}| \leq \frac{H_\text{zone}}{2}$$

This approximates the horizontal spreading of the inlet jet over the diffusor height and produces — in contrast to the point diffusor — a shallower temperature gradient in the inflow zone. The approach comes closer to the Lagrangian behaviour of FreeTTES in the inlet zone without explicitly modelling the inversion plume physics (cf. Section 4, model limitation).

```python
port = Port(z=9.5, m_dot=+80.0, T_in=90.0,
            diffusor=UniformDiffusor(H_zone=1.0))
```

---

## 14. Heat Exchanger Port (ε-NTU)

### Physical Background

For **indirect charging or discharging** — e.g. via a solar collector circuit or a heat pump — there is no mass exchange between the external circuit and the storage. Instead, heat is transferred via a heat exchanger. This is modelled using the **ε-NTU method** (effectiveness-NTU method).

Since the thermal capacity of the storage contents is very large compared to the external mass flow ($C_\text{storage} \to \infty$), the effectiveness definition simplifies to the case of a heat exchanger with an infinitely large thermal reservoir:

$$\text{NTU} = \frac{UA}{\dot{m}_\text{ext} \cdot c_{p,\text{ext}}}, \qquad \varepsilon = 1 - e^{-\text{NTU}}$$

$$\dot{Q} = \varepsilon \cdot \dot{m}_\text{ext} \cdot c_{p,\text{ext}} \cdot (T_\text{ext,in} - \bar{T}_\text{tank})$$

Here $\bar{T}_\text{tank}$ is the area-weighted mean of storage temperatures in the active heat exchanger zone $[z - H_{HX}/2,\; z + H_{HX}/2]$.

**Symbols:**

| Symbol | Meaning | Unit |
|---|---|---|
| $UA$ | Overall heat transfer coefficient of the heat exchanger | W/K |
| $\dot{m}_\text{ext}$ | Mass flow in the external circuit | kg/s |
| $c_{p,\text{ext}}$ | Specific heat capacity of the external fluid | J/(kg·K) |
| $\varepsilon$ | Heat exchanger effectiveness | — |
| $\text{NTU}$ | Number of Transfer Units | — |
| $T_\text{ext,in}$ | Inlet temperature of the external fluid | °C |
| $\bar{T}_\text{tank}$ | Mean storage temperature in the HX zone | °C |
| $H_{HX}$ | Active length of the heat exchanger | m |

```python
hx = HeatExchangerPort(
    z=9.0, H_hx=2.0, UA=5000.0,
    m_dot_ext=0.8, T_ext_in=85.0,
    label="Solar",
)
inputs = StorageInputs(hx_ports=[hx])
```

### Segmented Model

The lumped model uses a single mean storage temperature $\bar{T}_\text{tank}$ for the entire HX zone. This is sufficient as long as the heat exchanger lies in a **homogeneous temperature zone**. However, if it spans the **thermocline**, $T_\text{storage}(z)$ varies significantly along the heat exchanger — and the averaging distorts the local heat transfer.

The segmented model discretizes the heat exchanger into the actual storage nodes of the HX zone and tracks the external fluid temperature $T_\text{ext}$ node by node in the flow direction:

$$\text{NTU}_k = \frac{UA / n_\text{zone}}{\dot{m}_\text{ext} \cdot c_{p,\text{ext}}}, \qquad \varepsilon_k = 1 - e^{-\text{NTU}_k}$$

$$\dot{Q}_k = \varepsilon_k \cdot \dot{m}_\text{ext} \cdot c_{p,\text{ext}} \cdot (T_\text{ext,current} - T_\text{storage}[k])$$

$$T_\text{ext,current} \mathrel{-}= \frac{\dot{Q}_k}{\dot{m}_\text{ext} \cdot c_{p,\text{ext}}}$$

At the end of the last node, $T_\text{ext,current} = T_\text{ext,out}$.

The **flow direction** of the external fluid determines the order of nodes:

| `flow_direction` | External fluid | Node order |
|---|---|---|
| `"downward"` (default) | Entry at top (high z), exit at bottom | Decreasing z (increasing node indices) |
| `"upward"` | Entry at bottom (low z), exit at top | Increasing z (decreasing node indices) |

**Model comparison:**

| Property | Lumped | Segmented |
|---|---|---|
| Storage temperature | Mean $\bar{T}_\text{tank}$ | Local $T_\text{storage}[k]$ per node |
| $T_\text{ext,out}$ | From energy balance (correct) | Directly propagated (correct) |
| Thermocline spanned | Inaccurate | Correct |
| Flow direction | Not relevant | Configurable |
| Computational cost | O(1) | O($n_\text{zone}$) |

```python
# Segmented model: solar circuit flows top-to-bottom through the HX
hx = HeatExchangerPort(
    z=9.0, H_hx=2.0, UA=5000.0,
    m_dot_ext=0.8, T_ext_in=85.0,
    segmented=True, flow_direction="downward",
    label="Solar",
)
```

---

## 15. Headspace Model

### Physical Background

In pressureless tanks or partially filled vessels, a **gas cushion** (headspace) sits above the water surface. This has its own thermal mass and exchanges heat both with the roof above (to the outside) and with the water surface (into the storage).

### Headspace Energy Balance

The headspace is integrated as a single homogeneous control volume with explicit Euler:

$$C_\text{hs} \cdot \frac{dT_\text{hs}}{dt} = -\underbrace{U_\text{roof} \cdot A \cdot (T_\text{hs} - T_\text{amb})}_{\dot{Q}_\text{roof}\ \text{(loss to outside)}} - \underbrace{h_\text{hs} \cdot A \cdot (T_\text{hs} - T[0])}_{\dot{Q}_\text{hs,water}\ \text{(to top node)}}$$

The term $\dot{Q}_\text{hs,water}$ is added as a source term to the topmost water node (index 0).

**Symbols:**

| Symbol | Meaning | Unit |
|---|---|---|
| $C_\text{hs} = \rho_\text{hs} \cdot A \cdot H_\text{hs} \cdot c_{p,\text{hs}}$ | Thermal mass of the headspace | J/K |
| $U_\text{roof}$ | Overall heat transfer coefficient of the roof | W/(m²·K) |
| $h_\text{hs}$ | Convective HTC headspace to water surface | W/(m²·K) |
| $H_\text{hs}$ | Height of the headspace | m |
| $T_\text{hs}$ | Temperature of the headspace | °C |
| $T[0]$ | Temperature of the topmost water node | °C |

---

## 16. References

| Key | Full reference |
|---|---|
| [Falkner 2014] | Falkner S. (2014). *Modellierung und Simulation von thermischen Speichern*. Diploma thesis, Institute of Energy Technology and Thermodynamics E302, TU Vienna. |
| [Dahash et al. 2020] | Dahash A., Ochs F., Tosatto A., Streicher W. (2020). Toward efficient numerical modeling and analysis of large-scale thermal energy storage for renewable district heating. *Applied Energy 279*, 115840. https://doi.org/10.1016/j.apenergy.2020.115840 |
| [Jäger et al. 2024] | Jäger S., Pabst V., Renze P. (2024). Multizone Modeling for Hybrid Thermal Energy Storage. *Energies 17*, 2854. https://doi.org/10.3390/en17122854 |
| [Untrau et al. 2023] | Untrau A., Sochard S., Marias F., Reneaume J.-M., Le Roux G.A.C., Serra S. (2023). A fast and accurate 1-dimensional model for dynamic simulation and optimization of a stratified thermal energy storage. *Applied Energy 333*, 120614. https://doi.org/10.1016/j.apenergy.2022.120614 |
| [Lemelle et al. 2025] | Lemelle A.-G., Lamaison N., Vasset N., Reneaume J.-M., Serra S. (2025). Optimisation of thermal energy storage in district heating networks: Review and comparison of models. *ECOS 2025*, Paris. HAL: hal-05231608. |
| [Xiang et al. 2022] | Xiang Y., Xie Z., Furbo S., Wang D., Gao M., Fan J. (2022). A comprehensive review on pit thermal energy storage. *Journal of Energy Storage 55*, 105716. https://doi.org/10.1016/j.est.2022.105716 |
| [Sterner & Stadler 2017] | Sterner M., Stadler I. (eds.) (2017). *Energiespeicher – Bedarf, Technologien, Integration*. 2nd ed. Springer Vieweg. DOI 10.1007/978-3-662-48893-5 |
| [Weiss et al. 2021] | Weiss J., Ortega-Fernández I., Müller R., Bielsa D., Fluri T. (2021). Improved thermocline initialization through optimized inlet design for single-tank thermal energy storage systems. *Journal of Energy Storage 42*, 103088. https://doi.org/10.1016/j.est.2021.103088 |
| [Boß et al. 2024] | Boß V., Felsmann C., Narusavicius B., Rühling K. (2024). Planning Tools for Decentralized Heat Supply. *ISEC 2024*. https://doi.org/10.52825/isec.v1i.1110 |
| [Rendall et al. 2021] | Rendall J., Karg Bulnes F., Gluesenkamp K., Abu-Heiba A., Worek W., Nawaz K. (2021). A Flow Rate Dependent 1D Model for Thermally Stratified Hot-Water Energy Storage. *Energies 14*, 2611. https://doi.org/10.3390/en14092611 |
| [Narusavicius et al. 2024] | Narusavicius L., Koch K., Boß V., Felsmann C., Rühling K. (2024). Prediction of the Temperature Field inside a Large-Scale Thermal Energy Storage. *ISEC 2024*. https://doi.org/10.52825/isec.v1i.1109 |
