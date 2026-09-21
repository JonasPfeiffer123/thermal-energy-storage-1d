"""Loss model correctness for the models beyond ``ConstantAmbientLoss``.

``SplitAmbientLoss``, ``GroundTemperatureLoss`` and ``TransientGroundLoss``
are exercised in the optional (data-gated) validation scripts but had no
direct unit coverage: a wiring bug (wrong sign, wrong array, wrong node)
could ship unnoticed since those scripts only compare an aggregate MAE
against measurement data, and are skipped entirely without a cloned data
repo.

These tests instantiate each loss model directly against small, hand-built
node arrays rather than going through ``ThermalStorage1D``, so failures
point at the loss model itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_energy_storage_model import (
    ConstantAmbientLoss,
    GroundTemperatureLoss,
    SplitAmbientLoss,
    TransientGroundLoss,
)

# ---------------------------------------------------------------------------
# SplitAmbientLoss
# ---------------------------------------------------------------------------


def test_split_ambient_loss_uses_lid_value_only_at_top_node():
    loss = SplitAmbientLoss(U_lid=0.15, U_wall=0.35, T_ambient=8.0, T_ambient_lid=12.0)
    T_nodes = np.array([50.0, 40.0, 30.0, 20.0])
    A_wall = np.array([5.0, 6.0, 6.0, 7.0])
    z = np.array([7.5, 5.0, 2.5, 0.5])  # unused by this model, arbitrary

    Q = loss.Q_loss_nodes(T_nodes, A_wall, z)

    assert Q[0] == pytest.approx(0.15 * 5.0 * (12.0 - 50.0))
    for i in range(1, 4):
        assert Q[i] == pytest.approx(0.35 * A_wall[i] * (8.0 - T_nodes[i]))


def test_split_ambient_loss_defaults_lid_temperature_to_ambient():
    loss = SplitAmbientLoss(U_lid=0.15, U_wall=0.35, T_ambient=8.0)
    assert loss.T_ambient_lid == pytest.approx(8.0)
    T_nodes = np.array([50.0, 40.0])
    A_wall = np.array([5.0, 6.0])
    Q = loss.Q_loss_nodes(T_nodes, A_wall, np.zeros(2))
    assert Q[0] == pytest.approx(0.15 * 5.0 * (8.0 - 50.0))


@pytest.mark.parametrize("kwargs", [
    dict(U_lid=-0.1, U_wall=0.3, T_ambient=8.0),
    dict(U_lid=0.1, U_wall=-0.3, T_ambient=8.0),
])
def test_split_ambient_loss_rejects_negative_u_values(kwargs):
    with pytest.raises(ValueError):
        SplitAmbientLoss(**kwargs)


def test_constant_ambient_loss_rejects_negative_u_value():
    with pytest.raises(ValueError):
        ConstantAmbientLoss(U_loss=-0.1, T_ambient=8.0)


# ---------------------------------------------------------------------------
# GroundTemperatureLoss
# ---------------------------------------------------------------------------


def test_ground_temperature_at_depth_matches_exponential_formula():
    loss = GroundTemperatureLoss(
        U_loss=0.25, burial_depth=0.5, T_surface=8.0, T_deep=11.0, depth_decay=2.0,
    )
    assert loss.T_ground_at_depth(0.0) == pytest.approx(8.0)  # surface value
    d = 3.0
    expected = 11.0 + (8.0 - 11.0) * np.exp(-d / 2.0)
    assert loss.T_ground_at_depth(d) == pytest.approx(expected)
    # Deep asymptote.
    assert loss.T_ground_at_depth(1000.0) == pytest.approx(11.0, abs=1e-6)


def test_ground_temperature_loss_q_matches_depth_dependent_formula():
    loss = GroundTemperatureLoss(
        U_loss=0.25, burial_depth=0.5, T_surface=8.0, T_deep=11.0, depth_decay=2.0,
    )
    n, H = 4, 10.0
    dz = H / n
    # Node centers as used by the solver: index 0 = top, index n-1 = bottom.
    z = np.linspace(H - dz / 2.0, dz / 2.0, n)
    T_nodes = np.array([50.0, 40.0, 30.0, 20.0])
    A_wall = np.array([5.0, 6.0, 6.0, 7.0])

    Q = loss.Q_loss_nodes(T_nodes, A_wall, z)

    depths = 0.5 + (H - z)  # burial_depth + distance below the tank lid
    T_ground = 11.0 + (8.0 - 11.0) * np.exp(-depths / 2.0)
    expected = 0.25 * A_wall * (T_ground - T_nodes)
    assert np.allclose(Q, expected)
    # Bottom node is deepest -> warmer ground (closer to T_deep) than the top.
    assert loss.T_ground_at_depth(depths[-1]) > loss.T_ground_at_depth(depths[0])


@pytest.mark.parametrize("kwargs", [
    dict(U_loss=-0.1, burial_depth=0.5, T_surface=8.0, T_deep=11.0),
    dict(U_loss=0.25, burial_depth=-1.0, T_surface=8.0, T_deep=11.0),
    dict(U_loss=0.25, burial_depth=0.5, T_surface=8.0, T_deep=11.0, depth_decay=0.0),
])
def test_ground_temperature_loss_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        GroundTemperatureLoss(**kwargs)


# ---------------------------------------------------------------------------
# TransientGroundLoss
# ---------------------------------------------------------------------------


def _make_transient_loss(n_layers=4, T_far=8.0, T_init=None):
    return TransientGroundLoss(
        U_lid=0.15, T_ambient_lid=10.0,
        lambda_soil=2.0, rho_soil=1800.0, cp_soil=900.0,
        d_total=8.0, n_layers=n_layers, T_far=T_far, T_init=T_init,
    )


def test_transient_ground_loss_lid_uses_steady_state_formula():
    """The lid node (index 0) bypasses the RC chain entirely."""
    loss = _make_transient_loss()
    T_nodes = np.full(5, 60.0)
    A_wall = np.full(5, 10.0)
    z = np.linspace(4.5, 0.5, 5)

    Q = loss.Q_loss_nodes(T_nodes, A_wall, z)
    assert Q[0] == pytest.approx(0.15 * 10.0 * (10.0 - 60.0))


def test_transient_ground_loss_initial_wall_loss_uses_t_init():
    """Before any advance(), ground layers start at T_init (default T_far)."""
    loss = _make_transient_loss(T_far=8.0)
    T_nodes = np.full(5, 60.0)
    A_wall = np.full(5, 10.0)
    z = np.linspace(4.5, 0.5, 5)
    R = (8.0 / 4) / 2.0  # d_layer / lambda_soil

    Q = loss.Q_loss_nodes(T_nodes, A_wall, z)
    expected_wall = A_wall[1:] * (8.0 - T_nodes[1:]) / R
    assert np.allclose(Q[1:], expected_wall)


def test_transient_ground_loss_explicit_t_init_overrides_t_far():
    loss = _make_transient_loss(T_far=8.0, T_init=15.0)
    T_nodes = np.full(3, 60.0)
    A_wall = np.full(3, 10.0)
    z = np.linspace(2.5, 0.5, 3)
    loss.Q_loss_nodes(T_nodes, A_wall, z)  # triggers _ensure_init
    assert np.all(loss.T_ground == pytest.approx(15.0))


def test_transient_ground_loss_steady_state_is_a_fixed_point():
    """The analytical steady-state profile of the resistor chain (uniform
    heat flux q = (T_storage - T_far) / ((n_layers+1)*R), linear temperature
    drop of q*R per layer) must be left unchanged by advance() -- this
    verifies the discretized RC update against a closed-form reference
    without relying on run-to-convergence timing/stability.
    """
    n_nodes, n_layers = 5, 4
    lambda_soil, rho_soil, cp_soil = 2.0, 1800.0, 900.0
    d_total, T_far = 8.0, 8.0
    loss = TransientGroundLoss(
        U_lid=0.15, T_ambient_lid=10.0,
        lambda_soil=lambda_soil, rho_soil=rho_soil, cp_soil=cp_soil,
        d_total=d_total, n_layers=n_layers, T_far=T_far,
    )
    d_layer = d_total / n_layers
    R = d_layer / lambda_soil

    T_storage = np.full(n_nodes, 60.0)
    q = (T_storage[1] - T_far) / ((n_layers + 1) * R)
    steady = np.array([T_storage[0] - (j + 1) * q * R for j in range(n_layers)])
    loss._T_ground = np.tile(steady, (n_nodes, 1))
    before = loss._T_ground.copy()

    A_wall = np.full(n_nodes, 10.0)
    z = np.linspace(4.5, 0.5, n_nodes)
    loss.advance(T_storage, A_wall, z, dt=500.0)

    # Row 0 (lid) is intentionally excluded from the transient update.
    assert np.allclose(loss._T_ground[1:], before[1:], atol=1e-9)


def test_transient_ground_loss_lid_row_never_updated():
    loss = _make_transient_loss()
    T_nodes = np.full(5, 90.0)  # far from any equilibrium
    A_wall = np.full(5, 10.0)
    z = np.linspace(4.5, 0.5, 5)
    loss.Q_loss_nodes(T_nodes, A_wall, z)  # initialize
    loss._T_ground[0, :] = 42.0  # sentinel value on the (unused) lid row

    for _ in range(50):
        loss.advance(T_nodes, A_wall, z, dt=1000.0)

    assert np.all(loss.T_ground[0] == 42.0)


def test_transient_ground_loss_reinitializes_on_node_count_change():
    loss = _make_transient_loss(T_far=8.0)
    T_nodes = np.full(3, 60.0)
    A_wall = np.full(3, 10.0)
    z = np.linspace(2.5, 0.5, 3)
    loss.advance(T_nodes, A_wall, z, dt=10.0)
    assert loss.T_ground.shape == (3, 4)

    T_nodes2 = np.full(6, 60.0)
    A_wall2 = np.full(6, 10.0)
    z2 = np.linspace(5.5, 0.5, 6)
    # Q_loss_nodes() also triggers _ensure_init() but performs no Euler step,
    # so it isolates the reset from the state change advance() would add.
    loss.Q_loss_nodes(T_nodes2, A_wall2, z2)
    assert loss.T_ground.shape == (6, 4)
    assert np.all(loss.T_ground == pytest.approx(8.0))  # freshly re-initialized to T_far


@pytest.mark.parametrize("kwargs", [
    dict(U_lid=-0.1, T_ambient_lid=10.0, lambda_soil=2.0, rho_soil=1800.0,
         cp_soil=900.0, d_total=8.0),
    dict(U_lid=0.15, T_ambient_lid=10.0, lambda_soil=0.0, rho_soil=1800.0,
         cp_soil=900.0, d_total=8.0),
    dict(U_lid=0.15, T_ambient_lid=10.0, lambda_soil=2.0, rho_soil=1800.0,
         cp_soil=900.0, d_total=8.0, n_layers=0),
    dict(U_lid=0.15, T_ambient_lid=10.0, lambda_soil=2.0, rho_soil=1800.0,
         cp_soil=900.0, d_total=0.0),
])
def test_transient_ground_loss_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        TransientGroundLoss(**kwargs)


# ---------------------------------------------------------------------------
# __repr__ smoke test (catches attribute typos in the formatting string)
# ---------------------------------------------------------------------------


def test_loss_model_reprs_do_not_raise():
    models = [
        ConstantAmbientLoss(U_loss=0.3, T_ambient=10.0),
        SplitAmbientLoss(U_lid=0.15, U_wall=0.35, T_ambient=8.0, T_ambient_lid=12.0),
        GroundTemperatureLoss(U_loss=0.25, burial_depth=0.5, T_surface=8.0, T_deep=11.0),
        _make_transient_loss(),
    ]
    for model in models:
        text = repr(model)
        assert type(model).__name__ in text
