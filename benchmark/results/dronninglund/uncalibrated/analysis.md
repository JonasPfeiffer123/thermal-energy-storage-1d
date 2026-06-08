# Dronninglund PTES 2014 – Uncalibrated Validation

**Date:** 2026-06-08
**Model:** ThermalStorage1D, N=50, implicit, Upwind, Buoyancy
**Geometry:** TruncatedPyramidGeometry, 26.4×26.4 → 90.4×90.4 m, H=16 m, V≈60 030 m³
**Loss:** SplitAmbientLoss, U_lid=0.167, U_wall=0.35 W/(m²K); T_amb(walls)=8.0 °C, seasonal lid air temperature (sinusoidal, mean 8.0 °C, amplitude 8.5 K)
**Fluid:** WaterProperties (temperature-dependent)
**Simulation period:** 1 May 2014 (first FR data) to 31 Dec 2014
**Dataset:** Sifnaios et al. 2023, Solar Energy 251, 68–76

> These results correspond to the configuration in `benchmark/dronninglund_validation.py`,
> which is the authoritative source. Parameters are physically motivated, not
> calibrated (no parameter fitting). An earlier configuration used a single global
> `ConstantAmbientLoss` U = 0.25 W/(m²K) and gave a total MAE of 4.77 K; this was
> superseded by the split lid/wall loss model documented here.

---

## Parameter choice (uncalibrated)

| Parameter | Value | Source |
|-----------|-------|--------|
| U_lid (lid) | 0.167 W/(m²K) | Measured 2014 (Sifnaios notebook) |
| U_wall (walls/floor) | 0.35 W/(m²K) | Estimate: λ_soil ≈ 0.4 W/(m·K), eff. thickness 1 m |
| T_amb (walls/floor) | 8.0 °C | Danish annual mean (DMI Tylstrup) |
| T_amb (lid) | seasonal, mean 8.0 °C ± 8.5 K | Danish air temperature, sinusoidal |
| Fluid | WaterProperties (T-dependent) | FreeTTES polynomial fits |
| N | 50 | Number of nodes |
| Solver | implicit (TDMA) | Upwind discretisation |

---

## MAE results (May–Dec 2014)

| Diffusor | Height | MAE |
|----------|--------|-----|
| Top    | 15.3 m | **3.94 K** |
| Middle | 10.9 m | **7.64 K** |
| Bottom |  0.4 m | **2.37 K** |
| **Total** | – | **4.65 K** |

Sensor measurement accuracy: ±0.15 K (PT100, Sifnaios et al. 2023).

---

## Qualitative analysis of deviations

### Bottom diffusor (good)
MAE 2.37 K — the model matches the bottom zone well. The lower diffusor
receives cold return water from the district heating network in summer, passed
directly as `T_discharge_in`. As the bottom zone shows little stratification
effects, the 1D approximation is most reliable here.

### Top diffusor (slight overcooling in autumn)
MAE 3.94 K — from October onward the model still cools slightly more strongly at
the upper diffusor (15.3 m) than the measurements show, though the split loss
model (using the measured lid U-value of 0.167 W/(m²K) instead of the earlier
global 0.25 W/(m²K)) reduced this error. The insulated Nomalén three-layer
insulation (λ=0.047 W/(m·K)) protects the water surface; the residual deviation
is consistent with the seasonal lid air temperature and the lumped lid
representation.

### Middle diffusor (slower cooling Oct–Nov)
MAE 7.64 K — the largest error. In October the thermocline in the model sits
~2–3 m too deep (the model retains more heat than measured). Two causes:

1. **Ground heat losses underestimated**: The storage floor lies only 1–1.5 m
   above the aquifer (Tg ≈ 10–12 °C, flow ~15 m/year). In reality the
   groundwater extracts more heat from the lowest storage zone than a constant
   T_amb = 8 °C assumes. The `GroundTemperatureLoss` model with measured soil
   temperature sensors (Tg_10–Tg_25) would be more appropriate here.

2. **Diffusor mixing zone**: The middle diffusor (10.9 m) operates
   bidirectionally in real operation — sometimes as inlet (summer charging),
   sometimes as outlet (winter discharging). The turbulent mixing zone at the
   diffusor produces a wider transition zone in reality than the 1D model
   captures.

### Jan–Apr: no comparison possible
The FR direction data (FR_top_c, FR_mid_c) and all energy balance columns
(Q_ch_net, Q_dis_net) are NaN in Jan–Apr because no solar production is
available for the energy balance calculation. The simulation therefore starts
on 1 May with the measured temperature profile as initial condition. During
this period the storage was presumably in standby or heat pump operation
(no balance possible).

---

## Context in the literature

Ochs et al. 2021 report MAE values of 2–10 K for various 1D and layered
models at Dronninglund without calibration. Narusavicius et al. 2024 achieve
closer agreement with the FreeTTES Lagrangian model and explicit diffusor
plume modelling, but require significantly more computation time. The 4.65 K
(total) achieved here without any calibration and using purely physically
motivated parameters is within the expected range for uncalibrated 1D-FV
models.

---

## Possible future calibration

The results above are **uncalibrated**: every parameter is taken from a
measurement or a physical estimate, with no fitting. The split lid/wall loss
model (`SplitAmbientLoss`) is already in use; a formal calibration of its
parameters remains open future work and is **not** part of this repository:

- `U_lid`: loss at the top node (lid) — physical value: 0.167 W/(m²K)
- `U_wall`: loss at side walls and floor — physical estimate: 0.35 W/(m²K)
- `T_amb`: effective ambient temperature — physical value: 8.0 °C

A formal calibration would minimise MAE(T_top, T_mid, T_bot) for May–Dec 2014
(e.g. grid search or `scipy.optimize`), most plausibly improving the middle
diffusor by combining the split loss with `GroundTemperatureLoss` for the
floor zone.
