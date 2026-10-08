# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
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
- `benchmark/benchmark_step_performance.py --loss {constant,split,transient}`
  selects the loss model of the benchmark scenario.
- Bit-identity tests of `SplitAmbientLoss` and `TransientGroundLoss` against
  their former implementations (`tests/test_losses.py`).

### Fixed
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

[Unreleased]: https://github.com/JonasPfeiffer123/thermal-energy-storage-1d/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/JonasPfeiffer123/thermal-energy-storage-1d/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/JonasPfeiffer123/thermal-energy-storage-1d/releases/tag/v1.0.0
