# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
