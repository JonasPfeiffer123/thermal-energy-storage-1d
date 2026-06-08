# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-08

### Added
- Automatic CFL sub-stepping for the explicit solver (`StorageConfig.auto_substep`,
  default `True`): an over-large external timestep is split internally into the
  smallest number of CFL-stable sub-steps, so callers no longer need to respect
  the CFL limit. Outlet temperatures and `Q_loss` are averaged over the sub-steps.
- `pytest` test suite (`tests/`): energy/mass conservation, analytical
  verification (conduction, exponential cooling, plug-flow advection front),
  TDMA solver correctness, convective adjustment, auto sub-stepping, and a
  golden-scenario regression. Optional Dronninglund validation regression that
  skips when measurement data is absent.
- `CITATION.cff` and machine-readable version metadata (`__version__`).
  References the EuroSun 2026 extended abstract describing and validating
  this model (accepted; full paper forthcoming).

### Changed
- `pyproject.toml` now derives the package version dynamically from
  `thermal_energy_storage_model.__version__` (single source of truth).
- `docs/physics.md` §10 documents the in-model auto sub-stepping behaviour
  instead of caller-side pseudo-code.

## [0.1.0] - 2026-06-08

### Added
- Initial release of the 1D finite-volume stratified thermal energy storage model.
- Explicit (CFL-limited) and implicit (TDMA, unconditionally stable) Euler solvers.
- Upwind and TVD (van Leer) advection schemes; deferred-correction TVD for the
  implicit solver.
- Convective adjustment (buoyancy), flexible hydraulic ports with diffusor models,
  and ε-NTU heat-exchanger ports.
- Geometry models (cylinder, truncated cone, truncated pyramid), loss models
  (constant ambient, split lid/wall, ground, transient ground), and
  temperature-dependent water properties.
- Validation against the Dronninglund (2014) and Høje Taastrup (2024) PTES sites,
  benchmark comparison against FreeTTES, and an interactive PyQt6 UI.

[Unreleased]: https://github.com/JonasPfeiffer123/thermal-energy-storage-1d/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/JonasPfeiffer123/thermal-energy-storage-1d/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/JonasPfeiffer123/thermal-energy-storage-1d/releases/tag/v0.1.0
