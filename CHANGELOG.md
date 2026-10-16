# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-10-16

### Upgrading from 1.x
- Ground temperatures of `TransientGroundLoss`: read `state.loss_state`
  instead of `loss.T_ground`. States from `initialize()` carry them; a
  state built by hand starts from the initial ground field. Custom loss
  models with a history implement `initial_state()` and take / return the
  state in `Q_loss_nodes()` / `advance()` (see `docs/architecture.md`).
- Outlet port temperatures are energy-consistent (see *Changed*): results
  differ where inflow and outflow share a node (simultaneous operation) and
  with the implicit solver (end-of-step temperature).
- Time-varying air temperature: pass `StorageInputs.T_ambient` per step
  (setting loss-model attributes between steps still works).
- matplotlib is no longer installed with the package: install the extras
  `examples`, `benchmark` or `ui` for the scripts and the UI.
- `step()` raises `ValueError` for inputs it used to accept silently
  (non-positive `dt`, NaN, negative heat-exchanger flows, ...).

### Changed
- **Breaking:** `StorageState` is the complete simulation state. The ground
  temperatures of `TransientGroundLoss` moved from the loss-model object into
  the new field `StorageState.loss_state` (shape `(n_nodes, n_layers)`, set by
  `initialize()`). `step()` now depends only on `(state, dt, inputs)` and the
  configuration: stepping the same state twice gives the same result
  (rollback, iterative coupling), which it did not before.
  `TransientGroundLoss.T_ground` is removed (use `state.loss_state`);
  `Q_loss_nodes()` and `advance()` take the ground field as an extra
  argument, and `advance()` returns the new field instead of modifying it.
  New `LossModel.initial_state()` for loss models with state; custom models
  that keep a state inside the object (overriding only `advance()`) still
  run, with that state outside `StorageState`. This API change makes
  this release a major version.
- **Breaking (results):** outlet port temperatures are the temperatures at
  which the model removes the outflowing energy, so the step outputs close
  the energy balance at the coupling interface. Where an inflow enters the
  same node (e.g. simultaneous charging and discharging through the top node
  with `two_port()`), the outflow takes the inflowing fluid first, as in a
  hydraulic header; before, the outlet reported the node temperature while
  the energy balance had removed the inlet temperature (golden master: up to
  48 K in hours of simultaneous operation). With the implicit solver the
  outlet reports the end-of-step temperature, at which the outflow leaves
  (`tests/test_regression.py`: 78.00 → 73.05 °C). Inlet ports report the
  start-of-step node temperature as before.
- `StorageOutputs.Q_loss` is the loss applied in the step.
  `TransientGroundLoss` reported the loss of the ground field after the
  step (golden master: up to 16 W in 18 scenarios).
- `get_soc()` takes the full-charge energy with the fluid properties at
  `T_max` (was the nominal `rho`/`cp`): a tank uniformly at `T_max` has
  SOC = 1 also with `WaterProperties` (was 0.9916 at 90 °C).
- Headspace model: the lid is no longer part of the top node's loss area
  (the water surface lost heat through the lid twice: directly and through
  the headspace and roof), the roof sees the air-side temperature of the loss
  model (was always `StorageConfig.T_ambient`), and the exchange area is the
  lid area of the geometry.
- Explicit solver: sub-steps keep the Courant number at or below 0.9 (as
  before) and, new, the combined stability number CFL + conduction /
  heat-exchanger number at or below 1; where advection dominates the
  sub-step count is unchanged. The end time after sub-steps is exactly
  `t + dt` (was the sum of the sub-steps, off by up to 5e-6 s per year).
  With `auto_substep=False` the warning names the combined number.
- Golden master re-recorded (`tests/golden/reference.npz`). Temperatures are
  bit-identical in 120 of 133 scenarios; the other 13 are the explicit
  scenarios whose old steps exceeded the stability limit (50 nodes; 20 nodes
  on the 4 m pyramid and PTES), the heat-exchanger scenarios and the
  headspace scenario. SOC changed only with `WaterProperties`. The bitwise
  check of the implicit heat-exchanger scenario (LAPACK) also requires the
  NumPy version of the reference; `GOLDEN_BITWISE=0` disables the bitwise
  check (CI).
- Validation results unchanged: `benchmark/dronninglund_validation.py` and
  `benchmark/hoje_taastrup_validation.py` produce byte-identical comparison
  CSVs (MAE 4.65 K; 2.08 / 2.15 K).
- `ThermalStorage1D.step()` costs the same within the measurement noise
  (±5 %, A/B in one process), including the new input checks.
- NumPy is the only required dependency (`numpy>=1.21.6`, tested with
  Python 3.10); matplotlib moved to the extras `examples`, `benchmark`
  (with pandas, needed by the validation scripts and not declared before)
  and `ui`. `requirements*.txt` list the same packages.
- `pytest` also runs the docstring examples (`--doctest-modules`); all
  examples run, the class example of `ThermalStorage1D` used a removed API.
- `benchmark/results/model_variants/` regenerated with the current code
  (`--no-freetttes`); without the FreeTTES reference no configuration is
  recommended (the committed text recommended N=2, the coarsest grid).
- Docstrings: German leftovers translated, `geometry.py` without BOM.
- The `RuntimeWarning` for unbalanced port mass flows is emitted once per
  storage instead of on every step; it now also names the simulation time of
  the first unbalanced step. Its text contains the flow values, so Python's
  warning filter could not merge repeats: a persistent imbalance printed one
  warning per step (about 41 000 per year at 10-minute steps), which
  flooded the output and slowed the run down.
- `TransientGroundLoss` costs about half as much per step, with
  **bit-identical results** and no API change: `advance()` updates all
  ground layers in one array expression instead of a loop over the layers,
  and `Q_loss_nodes()` needs fewer NumPy calls. `step()` with this loss model
  (scenario of `benchmark/benchmark_step_performance.py --loss transient`;
  Windows, Python 3.11, NumPy 2.3; minimum of 3 interleaved runs):

  | nodes | before [µs/step] | after [µs/step] | speed-up |
  |------:|-----------------:|----------------:|---------:|
  | 5     | 58.5             | 41.3            | 1.42x    |
  | 20    | 80.2             | 63.2            | 1.27x    |
  | 50    | 113.5            | 98.9            | 1.15x    |

  Høje Taastrup 2024 (50 nodes, 10-minute steps): 102 → 90 µs/step.
- `SplitAmbientLoss.Q_loss_nodes()` needs fewer NumPy calls (about 0.7 µs
  per step, bit-identical).
- `benchmark/hoje_taastrup_validation.py` loads the dataset once and
  iterates the inputs without `DataFrame.iterrows()`; the full validation
  run takes 15 s instead of 24 s.

### Added
- `StorageInputs.T_ambient` (also `two_port(T_ambient=...)`): ambient air
  temperature for one step, e.g. from a weather file. It replaces the
  air-side temperature of the loss model (`ConstantAmbientLoss.T_ambient`,
  the lid temperature of `SplitAmbientLoss` and `TransientGroundLoss`;
  `LossModel.air_temperature_attr`, `with_air_temperature()`) and the
  headspace roof temperature, without modifying the configured loss model.
- Input validation: `step()` rejects a non-positive or non-finite `dt`, a
  state of another storage, non-finite port values, negative heat-exchanger
  flows or UA and unknown `flow_direction`; the configuration rejects NaN,
  non-integer `n_nodes` and invalid headspace / `T_ground` parameters
  (`ValueError`, `TypeError` for `n_nodes`).
- Tests: energy balance of the step outputs for simultaneous operation,
  diffusors, heat exchangers and `TransientGroundLoss`
  (`tests/test_conservation.py`); the co-simulation contract
  (`tests/test_cosim_contract.py`); stability of heat exchangers, headspace,
  conduction-limited sub-steps and thin ground layers.
- CI: Windows next to Ubuntu, Python 3.14, a job with the oldest supported
  NumPy on Python 3.10, and a smoke test of the example and benchmark
  scripts.
- `benchmark/benchmark_step_performance.py --loss {constant,split,transient}`
  selects the loss model of the benchmark scenario.
- Bit-identity tests of `SplitAmbientLoss` and `TransientGroundLoss` against
  their former implementations (`tests/test_losses.py`).

### Fixed
- `mypy` 2.4 reports the `cast` in `ThermalStorage1D.initialize()` as
  redundant (newer NumPy stubs narrow `np.isscalar()`), which failed the CI
  lint job. `initialize()` now decides scalar vs. profile on the converted
  array and needs no cast; it also accepts a 0-d array as uniform
  temperature (was rejected).
- Heat exchangers could heat the storage far beyond their inlet temperature
  with long timesteps, also with the "unconditionally stable" implicit
  solver, which evaluated them explicitly (300 L tank, `UA` = 2000 W/K,
  70 °C inlet, one 1-h step: 153 °C). The implicit solver now includes them
  in the implicit system (dense solve when heat exchangers are active), and
  the explicit sub-step count includes their conductance; the external
  outlet temperature is evaluated with the same temperatures as the heat.
- The headspace temperature diverged with explicit Euler for a light
  gas-only headspace (air, 10-min steps: 1e5 °C after one hour). It is now
  integrated exactly per step; headspace, water and roof exchange exactly
  the same energy.
- The explicit solver ignored the conduction (Fourier) limit: on fine grids
  with a large `lambda_eff_factor` the result was wrong by up to 23 K
  without a warning (the buoyancy correction hid the instability).
- `TransientGroundLoss` diverged for thin ground layers (40 layers over 2 m,
  hourly steps: 1e39 °C); `advance()` sub-steps beyond the explicit limit
  `R·C/2`.
- The UI sub-stepped the explicit solver a second time, and the implicit
  solver for no reason; it now leaves stability to the model.
- `benchmark/hoje_taastrup_validation.py` reads the 2024 dataset from
  `data/HojeTaastrupData/Data/`, where the HojeTaastrupData repository now
  stores its CSV files.
- `benchmark/hoje_taastrup_validation.py` feeds balanced port mass flows.
  It converted every measured volume flow with a density polynomial
  truncated after the quadratic term (2 % too low at 85 °C) at its own
  diffusor temperature, which left the mass flows of 41 099 of 52 704 steps
  unbalanced by a median 3.3 % (one `RuntimeWarning` per step). Volume flows
  are now converted with `WaterProperties`, and the outflows are scaled to
  the total inflow, as the fixed-volume model requires. MAE against the
  lance sensors (top / middle / bottom / total), result plots regenerated:

  | loss model          | before                    | after                     |
  |---------------------|---------------------------|---------------------------|
  | SplitAmbientLoss    | 2.13 / 3.22 / 1.59 / 2.31 | 2.39 / 2.85 / 1.00 / 2.08 |
  | TransientGroundLoss | 2.09 / 3.48 / 1.57 / 2.38 | 2.35 / 3.15 / 0.95 / 2.15 |

## [1.1.0] - 2026-09-30

### Changed
- `ThermalStorage1D.step()` is about 4-5x faster for the grid sizes used in
  co-simulation, with **bit-identical results** (same floating-point
  operations in the same order) and no API change. Measured with
  `benchmark/benchmark_step_performance.py` (implicit, TVD, buoyancy,
  `WaterProperties`, `StorageInputs.two_port`, dt = 3600 s, 100 m³
  cylinder, one year; Windows, Python 3.12, NumPy 2.5; minimum of 5
  interleaved runs):

  | nodes | before [µs/step] | after [µs/step] | speed-up |
  |------:|-----------------:|----------------:|---------:|
  | 5     | 86.6             | 17.7            | 4.9x     |
  | 20    | 153.3            | 38.4            | 4.0x     |
  | 50    | 291.7            | 75.2            | 3.9x     |

  With the per-step reads of a co-simulation (`get_soc`, `T_mean`, ...)
  the time per coupling step drops from 105 / 178 / 316 µs to
  23 / 43 / 78 µs.
- The per-step loops (TVD correction, matrix assembly + Thomas algorithm,
  convective adjustment, and `WaterProperties` on grids of up to 32 nodes)
  run on Python floats in the new private module `_kernels.py`, since NumPy
  call overhead dominates on 5-50 element arrays; the implicit matrix is
  assembled inside the forward sweep of the Thomas algorithm.
- Per-configuration constants (interface areas, TVD denominators, total heat
  capacity) are precomputed; the port-to-node mapping of `PointDiffusor` and
  `UniformDiffusor` is cached per port height; fluid properties are evaluated
  once per temperature profile (shared between `get_soc()` and the next
  `step()`); steady-state loss models (`advance()` not overridden) are
  evaluated once per step; `np.mean`/`np.sum`/`np.clip` wrappers are
  replaced by `np.add.reduce` / `min(max(...))` with identical results.
- `WaterProperties` polynomial coefficients are class constants shared by
  `rho()`/`cp()` and the list evaluation (values unchanged).

### Added
- Golden-master regression tests (`tests/test_golden_master.py`): complete
  one-year trajectories of 133 configurations (node counts, geometries, loss
  models, fluids, buoyancy, solver/advection variants, presets, diffusor,
  heat-exchanger and headspace paths) are compared bitwise against the
  v1.0.0 results on the recording platform and within 1e-12 relative
  deviation everywhere. `python -m tests.golden.generate --check` runs the
  comparison in parallel; `pytest -m "not golden"` skips it.
- Kernel tests (`tests/test_kernels.py`) comparing the Python-float kernels
  bitwise with the former array implementations.
- `benchmark/benchmark_step_performance.py`: wall-clock cost per `step()`.
- Publications section in the README and a poster reference in
  `CITATION.cff`, linking the EuroSun 2026 poster archived on ResearchGate
  (DOI: 10.13140/RG.2.2.14082.44481).

## [1.0.0] - 2026-06-08

Archived on Zenodo: [10.5281/zenodo.20595165](https://doi.org/10.5281/zenodo.20595165)

### Added
- Initial public release of the 1D finite-volume stratified thermal energy
  storage model.
- Explicit (CFL-limited, with automatic sub-stepping via `StorageConfig.auto_substep`)
  and implicit (TDMA, unconditionally stable) Euler solvers.
- Upwind and TVD (van Leer) advection schemes; deferred-correction TVD for the
  implicit solver.
- Convective adjustment (buoyancy), flexible hydraulic ports with diffusor models,
  and ε-NTU heat-exchanger ports.
- Geometry models (cylinder, truncated cone, truncated pyramid), loss models
  (constant ambient, split lid/wall, ground, transient ground), and
  temperature-dependent water properties.
- Validation against the Dronninglund (2014) and Høje Taastrup (2024) PTES sites,
  benchmark comparison against FreeTTES, and an interactive PyQt6 UI.
- `pytest` test suite (`tests/`): energy/mass conservation, analytical
  verification (conduction, exponential cooling, plug-flow advection front),
  TDMA solver correctness, convective adjustment, auto sub-stepping, and a
  golden-scenario regression. Optional Dronninglund validation regression that
  skips when measurement data is absent.
- `CITATION.cff` and machine-readable version metadata (`__version__`).
  References the EuroSun 2026 extended abstract describing and validating
  this model (accepted; full paper forthcoming).
- CI workflow running the test suite on push and pull request.

[Unreleased]: https://github.com/JonasPfeiffer123/thermal-energy-storage-1d/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/JonasPfeiffer123/thermal-energy-storage-1d/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/JonasPfeiffer123/thermal-energy-storage-1d/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/JonasPfeiffer123/thermal-energy-storage-1d/releases/tag/v1.0.0
