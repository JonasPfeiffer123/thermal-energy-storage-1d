"""Automatic CFL sub-stepping of the explicit solver (``auto_substep``)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from thermal_energy_storage_model import StorageInputs


def _cfl_violating_inputs():
    # n_nodes=50, dt=3600 s, m_dot=20 kg/s -> CFL >> 1 for this tank.
    return StorageInputs.two_port(m_dot_charge=20.0, T_charge_in=85.0, height=5.0)


def test_autosubstep_stable_under_cfl_violation(make_storage):
    """With auto_substep on (default) a CFL-violating step stays bounded and warning-free."""
    # U_loss=0 so the only temperature drivers are advection/conduction, giving
    # the clean physical bounds [T_init, T_charge] without wall cooling.
    storage = make_storage(n_nodes=50, solver="explicit", auto_substep=True, U_loss=0.0)
    state = storage.initialize(T_init=60.0)
    inputs = _cfl_violating_inputs()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = storage.step(state, dt=3600.0, inputs=inputs)

    T = out.state.temperatures
    assert np.all(np.isfinite(T))
    assert T.max() <= 85.0 + 1e-6      # no overshoot above charge temperature
    assert T.min() >= 60.0 - 1e-6      # no undershoot below initial temperature


def test_autosubstep_disabled_warns(make_storage):
    """With auto_substep off, a CFL violation must raise a RuntimeWarning."""
    storage = make_storage(n_nodes=50, solver="explicit", auto_substep=False)
    state = storage.initialize(T_init=60.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        storage.step(state, dt=3600.0, inputs=_cfl_violating_inputs())
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_required_substeps_counts(make_storage):
    """A CFL-safe step needs 1 sub-step; a violating step needs more than one."""
    storage = make_storage(n_nodes=50, solver="explicit")
    state = storage.initialize(T_init=60.0)
    inputs = _cfl_violating_inputs()

    assert storage._required_substeps(state, 3600.0, inputs.ports) > 1
    # No flow -> no CFL constraint -> single step.
    assert storage._required_substeps(state, 3600.0, []) == 1
    # Tiny dt -> CFL satisfied -> single step.
    assert storage._required_substeps(state, 1.0, inputs.ports) == 1


def test_autosubstep_matches_manual_substepping(make_storage):
    """step() with auto_substep must equal an explicit manual sub-step loop."""
    storage = make_storage(n_nodes=50, solver="explicit", auto_substep=True)
    state = storage.initialize(T_init=60.0)
    inputs = _cfl_violating_inputs()

    n = storage._required_substeps(state, 3600.0, inputs.ports)
    auto = storage.step(state, dt=3600.0, inputs=inputs)

    cur = state
    for _ in range(n):
        cur = storage._step_single(cur, 3600.0 / n, inputs).state

    assert np.array_equal(auto.state.temperatures, cur.temperatures)


def test_autosubstep_advances_full_dt(make_storage):
    """The returned state time advances by the full external dt, not a sub-step."""
    storage = make_storage(n_nodes=50, solver="explicit", auto_substep=True)
    state = storage.initialize(T_init=60.0, time=0.0)
    out = storage.step(state, dt=3600.0, inputs=_cfl_violating_inputs())
    assert out.state.time == pytest.approx(3600.0)


# ---------------------------------------------------------------------------
# Conduction (Fourier) limit
# ---------------------------------------------------------------------------
# Explicit Euler with upwind advection and central conduction keeps the old
# node temperature with weight 1 - CFL - Fo (Fo = dt·(K_above+K_below)/C);
# for Fo > 1 it overshoots. Before, only the CFL number was sub-stepped, so
# an idle tank on a fine grid with high lambda_eff_factor was integrated
# with Fo ≈ 48: 23 K off after 100 min, hidden by the buoyancy correction.


def _fine_grid(make_storage, solver, **kw):
    return make_storage(volume=1.0, height=1.0, n_nodes=50, lambda_eff_factor=100.0,
                        U_loss=0.0, solver=solver, **kw)


def test_conduction_limit_is_sub_stepped(make_storage):
    T0 = np.where(np.arange(50) < 25, 80.0, 20.0)
    storage = _fine_grid(make_storage, "explicit")
    state = storage.initialize(T_init=T0)
    assert storage._required_substeps(state, 600.0, []) > 1   # no flow at all

    reference = _fine_grid(make_storage, "implicit")
    ref_state = reference.initialize(T_init=T0)
    for _ in range(10):
        state = storage.step(state, dt=600.0, inputs=StorageInputs()).state
    for _ in range(1000):
        ref_state = reference.step(ref_state, dt=6.0, inputs=StorageInputs()).state
    assert np.max(np.abs(state.temperatures - ref_state.temperatures)) < 0.1


def test_conduction_limit_violation_warns_without_substeps(make_storage):
    storage = _fine_grid(make_storage, "explicit", auto_substep=False)
    state = storage.initialize(T_init=np.where(np.arange(50) < 25, 80.0, 20.0))
    with pytest.warns(RuntimeWarning, match="stability limit"):
        storage.step(state, dt=600.0, inputs=StorageInputs())


def test_advection_dominated_substep_count_is_unchanged(make_storage):
    """Coarse grid: the conduction term stays far below the limit, so the
    sub-step count is that of the CFL condition alone, as before."""
    storage = make_storage(n_nodes=50, solver="explicit")
    state = storage.initialize(T_init=60.0)
    inputs = _cfl_violating_inputs()
    F = storage._compute_inter_node_fluxes(inputs.ports)
    cfl = np.max(np.abs(F)) / (977.8 * storage.A_cross) * 3600.0 / storage.dz
    assert storage._required_substeps(state, 3600.0, inputs.ports) == int(np.ceil(cfl / 0.9))


def test_substepped_end_time_is_exact(make_storage):
    storage = make_storage(n_nodes=50, solver="explicit")
    state = storage.initialize(T_init=60.0, time=0.1)
    assert storage._required_substeps(state, 3600.0, _cfl_violating_inputs().ports) > 2
    out = storage.step(state, dt=3600.0, inputs=_cfl_violating_inputs())
    assert out.state.time == 0.1 + 3600.0
