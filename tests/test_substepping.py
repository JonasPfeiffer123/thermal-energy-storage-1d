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
