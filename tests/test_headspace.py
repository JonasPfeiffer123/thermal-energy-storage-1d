"""Headspace model correctness.

The headspace model (atmospheric/outdoor pit storages) had no coverage
outside the ``--no-freetttes`` benchmark comparison script -- no unit test
ever asserted its energy balance, only a visual/aggregate comparison plot
that most runs skip. These tests check the per-step exchange formula
(exact solution of the headspace ODE) directly, its stability for a light
gas-only headspace, and a combined water+headspace energy-conservation
identity end to end through ``step()``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from thermal_energy_storage_model import (
    SplitAmbientLoss,
    StorageConfig,
    StorageInputs,
    ThermalStorage1D,
)


def _make_headspace_storage(**overrides):
    params = dict(
        volume=100.0, height=5.0, n_nodes=20, U_loss=0.0,
        headspace=True, T_headspace_init=99.0, H_headspace=0.5,
        U_roof=0.2, h_headspace_water=5.0,
        rho_headspace=2400.0, cp_headspace=880.0, T_ambient=10.0,
    )
    params.update(overrides)
    from tests._helpers import const_fluid
    params.setdefault("fluid", const_fluid())
    return ThermalStorage1D(StorageConfig(**params))


# ---------------------------------------------------------------------------
# Direct formula checks
# ---------------------------------------------------------------------------


def _exact_headspace(T_hs, T_top, T_roof, dt, A=20.0, h=5.0, U=0.2,
                     rho=2400.0, H=0.5, cp=880.0):
    """Closed-form T_hs(dt) and energies of the headspace ODE (fixed T_top)."""
    C_hs = rho * A * H * cp
    G_w, G_r = h * A, U * A
    T_eq = (G_r * T_roof + G_w * T_top) / (G_w + G_r)
    tau = C_hs / (G_w + G_r)
    T_new = T_eq + (T_hs - T_eq) * math.exp(-dt / tau)
    integral = (T_hs - T_eq) * tau * (1.0 - math.exp(-dt / tau))  # ∫(T_hs - T_eq)
    E_water = G_w * ((T_eq - T_top) * dt + integral)
    E_roof = G_r * ((T_eq - T_roof) * dt + integral)
    return T_new, E_water, E_roof, C_hs


def test_headspace_exchange_matches_exact_solution():
    storage = _make_headspace_storage()
    T_hs, T_top, dt = 99.0, 60.0, 30.0

    Q_hs_to_water, T_hs_new = storage._compute_headspace_exchange(T_hs, T_top, dt)

    T_expected, E_water, _, _ = _exact_headspace(T_hs, T_top, 10.0, dt)
    assert T_hs_new == pytest.approx(T_expected, rel=1e-12)
    assert Q_hs_to_water == pytest.approx(E_water / dt, rel=1e-12)
    # Short step: close to the start-of-step rate h·A·(T_hs − T_top)
    assert Q_hs_to_water == pytest.approx(5.0 * 20.0 * (T_hs - T_top), rel=1e-3)


def test_headspace_exchange_conserves_energy_over_long_steps():
    """Headspace energy change = −(heat to water + roof loss), for any dt."""
    storage = _make_headspace_storage()
    for dt in (1.0, 600.0, 3600.0, 1e6):
        Q_w, T_new = storage._compute_headspace_exchange(99.0, 60.0, dt)
        _, E_water, E_roof, C_hs = _exact_headspace(99.0, 60.0, 10.0, dt)
        assert Q_w * dt == pytest.approx(E_water, rel=1e-10)
        assert C_hs * (T_new - 99.0) == pytest.approx(-(E_water + E_roof), rel=1e-10)


def test_light_headspace_is_stable_for_long_steps():
    """A gas-only headspace (air: 1.2 kg/m³) relaxes in about two minutes.

    With explicit Euler, 600 s steps made T_headspace oscillate with growing
    amplitude (1e5 °C after six steps). The exact update relaxes to the
    equilibrium between water surface and roof and stays there.
    """
    storage = _make_headspace_storage(
        volume=1000.0, height=10.0, rho_headspace=1.2, cp_headspace=1005.0,
        solver="implicit",
    )
    state = storage.initialize(T_init=80.0)
    for _ in range(20):
        state = storage.step(state, dt=600.0, inputs=StorageInputs()).state
        assert 10.0 <= state.T_headspace <= 99.0
    T_eq = (0.2 * 10.0 + 5.0 * state.temperatures[0]) / 5.2
    assert state.T_headspace == pytest.approx(T_eq, abs=0.05)


def test_headspace_removes_lid_from_top_node_loss_area():
    """With a headspace the lid path runs through the headspace (U_roof),
    not directly from the water: the lid is not counted twice."""
    without = _make_headspace_storage(headspace=False, U_loss=0.3)
    with_hs = _make_headspace_storage(U_loss=0.3)
    A_lid = 100.0 / 5.0
    assert with_hs.A_wall[0] == pytest.approx(without.A_wall[0] - A_lid)
    assert np.array_equal(with_hs.A_wall[1:], without.A_wall[1:])


def test_headspace_roof_uses_loss_model_air_temperature():
    """The roof sees the air-side temperature of the loss model ..."""
    loss = SplitAmbientLoss(U_lid=0.2, U_wall=0.3, T_ambient=8.0, T_ambient_lid=-5.0)
    storage = _make_headspace_storage(loss_model=loss)
    state = storage.initialize(T_init=60.0)
    out = storage.step(state, dt=3600.0, inputs=StorageInputs())
    T_expected, _, _, _ = _exact_headspace(99.0, 60.0, -5.0, 3600.0)
    assert out.T_headspace == pytest.approx(T_expected, rel=1e-12)

    # ... or StorageInputs.T_ambient when given for the step.
    out = storage.step(state, dt=3600.0, inputs=StorageInputs(T_ambient=20.0))
    T_expected, _, _, _ = _exact_headspace(99.0, 60.0, 20.0, 3600.0)
    assert out.T_headspace == pytest.approx(T_expected, rel=1e-12)


def test_headspace_exchange_is_a_fixed_point_at_thermal_equilibrium():
    """If headspace, top water and ambient are all equal, no heat flows and
    the headspace temperature must not drift."""
    storage = _make_headspace_storage()
    Q_hs_to_water, T_hs_new = storage._compute_headspace_exchange(
        T_headspace=10.0, T_top=10.0, dt=100.0
    )
    assert Q_hs_to_water == pytest.approx(0.0)
    assert T_hs_new == pytest.approx(10.0)


def test_headspace_hotter_than_water_and_ambient_cools_down():
    storage = _make_headspace_storage()
    Q_hs_to_water, T_hs_new = storage._compute_headspace_exchange(
        T_headspace=99.0, T_top=60.0, dt=30.0
    )
    assert Q_hs_to_water > 0.0   # heat flows into storage, per the sign convention
    assert T_hs_new < 99.0       # headspace loses energy via both paths


# ---------------------------------------------------------------------------
# Integration through step()
# ---------------------------------------------------------------------------


def test_headspace_initializes_from_config():
    storage = _make_headspace_storage(T_headspace_init=77.0)
    state = storage.initialize(T_init=40.0)
    assert state.T_headspace == pytest.approx(77.0)


def test_headspace_disabled_leaves_t_headspace_none():
    storage = _make_headspace_storage(headspace=False)
    state = storage.initialize(T_init=40.0)
    assert state.T_headspace is None
    out = storage.step(state, dt=60.0, inputs=StorageInputs())
    assert out.T_headspace is None


def test_headspace_warms_only_top_node_in_one_step():
    """With no ports and no wall loss, a single explicit step can only
    spread heat by one interface via conduction -- the bottom node, far
    from the top, must be completely untouched."""
    storage = _make_headspace_storage(n_nodes=20)
    state = storage.initialize(T_init=40.0)
    out = storage.step(state, dt=10.0, inputs=StorageInputs())
    T = out.state.temperatures
    assert T[0] > 40.0 + 1e-9
    assert T[-1] == pytest.approx(40.0, abs=1e-12)


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_headspace_water_energy_balance_accounts_for_roof_loss_only(solver):
    """The only path for energy to leave the combined water+headspace
    system (U_loss=0, no ports) is the roof loss to ambient; the internal
    headspace<->top-node exchange must net out to zero across the combined
    system: dE_water + dE_headspace == -sum(E_roof), with the roof energy of
    each step from the exact headspace solution for that step's start-of-step
    top temperature.
    """
    volume, height = 100.0, 5.0
    storage = _make_headspace_storage(
        volume=volume, height=height, n_nodes=20, solver=solver,
    )

    state = storage.initialize(T_init=40.0)
    E0 = storage.get_stored_energy(state)
    T_hs0 = state.T_headspace

    dt = 600.0
    roof_loss_integrated = 0.0
    C_hs = None
    for _ in range(50):
        _, _, E_roof, C_hs = _exact_headspace(
            state.T_headspace, state.temperatures[0], 10.0, dt,
        )
        roof_loss_integrated += E_roof
        out = storage.step(state, dt=dt, inputs=StorageInputs())
        state = out.state

    dE_water = storage.get_stored_energy(state) - E0
    dE_headspace = C_hs * (state.T_headspace - T_hs0)

    assert (dE_water + dE_headspace) == pytest.approx(-roof_loss_integrated, rel=1e-9)
