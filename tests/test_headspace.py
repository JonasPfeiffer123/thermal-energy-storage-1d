"""Headspace model correctness.

The headspace model (atmospheric/outdoor pit storages) had no coverage
outside the ``--no-freetttes`` benchmark comparison script -- no unit test
ever asserted its energy balance, only a visual/aggregate comparison plot
that most runs skip. These tests check the per-step exchange formula
directly and a combined water+headspace energy-conservation identity end to
end through ``step()``.
"""

from __future__ import annotations

import pytest

from thermal_energy_storage_model import StorageConfig, StorageInputs, ThermalStorage1D


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


def test_headspace_exchange_matches_energy_balance_formula():
    storage = _make_headspace_storage()
    A = 100.0 / 5.0  # cylinder cross-section, volume/height
    T_hs, T_top, dt = 99.0, 60.0, 30.0

    Q_hs_to_water, T_hs_new = storage._compute_headspace_exchange(T_hs, T_top, dt)

    Q_hs_expected = 5.0 * A * (T_hs - T_top)         # h_headspace_water * A * (T_hs - T_top)
    Q_roof_expected = 0.2 * A * (T_hs - 10.0)          # U_roof * A * (T_hs - T_ambient)
    C_hs = 2400.0 * A * 0.5 * 880.0
    T_hs_new_expected = T_hs + dt * (-Q_roof_expected - Q_hs_expected) / C_hs

    assert Q_hs_to_water == pytest.approx(Q_hs_expected)
    assert T_hs_new == pytest.approx(T_hs_new_expected)


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
    system. Verified derivation: dE_water + dE_headspace == -sum(Q_roof*dt),
    with Q_roof evaluated at each step's start-of-step T_headspace (matching
    the explicit-Euler evaluation point used internally).
    """
    volume, height = 100.0, 5.0
    A = volume / height
    storage = _make_headspace_storage(
        volume=volume, height=height, n_nodes=20, solver=solver,
    )
    cfg = storage.config
    C_hs = cfg.rho_headspace * A * cfg.H_headspace * cfg.cp_headspace

    state = storage.initialize(T_init=40.0)
    E0 = storage.get_stored_energy(state)
    T_hs0 = state.T_headspace

    dt = 30.0
    roof_loss_integrated = 0.0
    for _ in range(50):
        Q_roof = cfg.U_roof * A * (state.T_headspace - cfg.T_ambient)
        roof_loss_integrated += Q_roof * dt
        out = storage.step(state, dt=dt, inputs=StorageInputs())
        state = out.state

    dE_water = storage.get_stored_energy(state) - E0
    dE_headspace = C_hs * (state.T_headspace - T_hs0)

    assert (dE_water + dE_headspace) == pytest.approx(-roof_loss_integrated, rel=1e-8)
