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
    state = loss.initial_state(3)
    assert state.shape == (3, 4)
    assert np.all(state == pytest.approx(15.0))


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
    before = np.tile(steady, (n_nodes, 1))

    A_wall = np.full(n_nodes, 10.0)
    z = np.linspace(4.5, 0.5, n_nodes)
    after = loss.advance(T_storage, A_wall, z, 500.0, before)

    # Row 0 (lid) is intentionally excluded from the transient update.
    assert np.allclose(after[1:], before[1:], atol=1e-9)


def test_transient_ground_loss_lid_row_never_updated():
    loss = _make_transient_loss()
    T_nodes = np.full(5, 90.0)  # far from any equilibrium
    A_wall = np.full(5, 10.0)
    z = np.linspace(4.5, 0.5, 5)
    state = loss.initial_state(5)
    state[0, :] = 42.0  # sentinel value on the (unused) lid row

    for _ in range(50):
        state = loss.advance(T_nodes, A_wall, z, 1000.0, state)

    assert np.all(state[0] == 42.0)


def test_transient_ground_loss_advance_does_not_modify_its_input():
    """The ground field lives in StorageState: advance() returns a new field."""
    loss = _make_transient_loss()
    T_nodes = np.full(5, 90.0)
    A_wall = np.full(5, 10.0)
    z = np.linspace(4.5, 0.5, 5)
    state = loss.initial_state(5)
    before = state.copy()

    after = loss.advance(T_nodes, A_wall, z, 3600.0, state)

    assert np.array_equal(state, before)
    assert not np.array_equal(after, before)


def test_transient_ground_loss_has_no_hidden_state():
    """Same inputs give the same result however often the model is used."""
    loss = _make_transient_loss()
    T_nodes = np.full(5, 70.0)
    A_wall = np.full(5, 10.0)
    z = np.linspace(4.5, 0.5, 5)
    state = loss.initial_state(5)
    first = loss.advance(T_nodes, A_wall, z, 3600.0, state)
    loss.advance(T_nodes, A_wall, z, 3600.0, first)   # unrelated call
    second = loss.advance(T_nodes, A_wall, z, 3600.0, state)
    assert first.tobytes() == second.tobytes()
    assert (loss.Q_loss_nodes(T_nodes, A_wall, z, first).tobytes()
            == loss.Q_loss_nodes(T_nodes, A_wall, z, second).tobytes())


def test_transient_ground_loss_rejects_state_of_wrong_shape():
    loss = _make_transient_loss()
    T_nodes = np.full(5, 60.0)
    A_wall = np.full(5, 10.0)
    z = np.linspace(4.5, 0.5, 5)
    with pytest.raises(ValueError, match="shape"):
        loss.Q_loss_nodes(T_nodes, A_wall, z, loss.initial_state(6))
    with pytest.raises(ValueError, match="shape"):
        loss.advance(T_nodes, A_wall, z, 60.0, loss.initial_state(3))


def test_transient_ground_loss_substeps_beyond_explicit_limit():
    """Thin layers: dt far above R*C/2 is sub-stepped and stays bounded.

    Before sub-stepping, 40 layers over 2 m with dt = 3600 s (limit about
    900 s) diverged to ~1e39 °C within two days.
    """
    loss = TransientGroundLoss(
        U_lid=0.15, T_ambient_lid=10.0, lambda_soil=2.2, rho_soil=2000.0,
        cp_soil=800.0, d_total=2.0, n_layers=40, T_far=8.0,
    )
    dt = 3600.0
    assert loss.substeps(dt) > 1
    limit = loss._R * loss._C / 2.0
    assert loss.substeps(limit) == 1

    T_nodes = np.full(4, 80.0)
    A_wall = np.full(4, 10.0)
    z = np.linspace(3.5, 0.5, 4)
    state = loss.initial_state(4)
    for _ in range(48):
        state = loss.advance(T_nodes, A_wall, z, dt, state)
    assert np.all(np.isfinite(state))
    assert state[1:].min() >= 8.0 - 1e-9
    assert state[1:].max() <= 80.0 + 1e-9


def test_transient_ground_loss_substeps_match_manual_steps():
    """One sub-stepped advance() equals the same number of plain steps."""
    loss = TransientGroundLoss(
        U_lid=0.15, T_ambient_lid=10.0, lambda_soil=2.2, rho_soil=2000.0,
        cp_soil=800.0, d_total=2.0, n_layers=40, T_far=8.0,
    )
    T_nodes = np.linspace(80.0, 40.0, 4)
    A_wall = np.full(4, 10.0)
    z = np.linspace(3.5, 0.5, 4)
    dt = 3600.0
    n = loss.substeps(dt)
    manual = loss.initial_state(4)
    for _ in range(n):
        manual = loss.advance(T_nodes, A_wall, z, dt / n, manual)
    assert np.array_equal(loss.advance(T_nodes, A_wall, z, dt, None), manual)


def test_loss_models_air_temperature():
    """StorageInputs.T_ambient replaces the air-side temperature only."""
    constant = ConstantAmbientLoss(U_loss=0.3, T_ambient=10.0)
    split = SplitAmbientLoss(U_lid=0.15, U_wall=0.35, T_ambient=8.0, T_ambient_lid=12.0)
    ground = GroundTemperatureLoss(U_loss=0.25, burial_depth=0.5, T_surface=8.0,
                                   T_deep=11.0)
    transient = _make_transient_loss()

    assert constant.air_temperature == 10.0
    assert split.air_temperature == 12.0
    assert ground.air_temperature is None
    assert transient.air_temperature == 10.0

    hot = constant.with_air_temperature(30.0)
    assert hot.T_ambient == 30.0 and constant.T_ambient == 10.0   # copy
    hot_split = split.with_air_temperature(30.0)
    assert (hot_split.T_ambient_lid, hot_split.T_ambient) == (30.0, 8.0)
    assert transient.with_air_temperature(30.0).T_ambient_lid == 30.0
    assert ground.with_air_temperature(30.0) is ground   # no air surface


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


# ---------------------------------------------------------------------------
# Bit-identity with the former array code
# ---------------------------------------------------------------------------
# Reference implementations: the loss-model methods of v1.1.0, before the
# per-step cost of SplitAmbientLoss and TransientGroundLoss was reduced.

def ref_split_Q_loss_nodes(loss, T_nodes, A_wall_nodes):
    dT = loss.T_ambient - T_nodes
    U_nodes = np.full(len(T_nodes), loss.U_wall)
    U_nodes[0] = loss.U_lid
    Q = U_nodes * A_wall_nodes * dT
    Q[0] = loss.U_lid * A_wall_nodes[0] * (loss.T_ambient_lid - T_nodes[0])
    return Q


def ref_transient_Q_loss_nodes(loss, Tg, T_nodes, A_wall_nodes):
    n = len(T_nodes)
    Q = np.empty(n)
    Q[0] = loss.U_lid * A_wall_nodes[0] * (loss.T_ambient_lid - T_nodes[0])
    Q[1:] = A_wall_nodes[1:] * (Tg[1:, 0] - T_nodes[1:]) / loss._R
    return Q


def ref_transient_advance(loss, Tg, T_nodes, dt):
    n = len(T_nodes)
    R = loss._R
    C = loss._C
    dT = np.empty_like(Tg)
    for j in range(loss.n_layers):
        T_left = T_nodes if j == 0 else Tg[:, j - 1]
        T_right = np.full(n, loss.T_far) if j == loss.n_layers - 1 else Tg[:, j + 1]
        dT[:, j] = (dt / C) * ((T_left - Tg[:, j]) / R - (Tg[:, j] - T_right) / R)
    dT[0, :] = 0.0
    Tg += dT


RNG_SEED = 20261008


@pytest.mark.parametrize("n_nodes", [1, 2, 5, 20, 50])
def test_split_ambient_loss_matches_reference_bitwise(n_nodes):
    rng = np.random.default_rng(RNG_SEED + n_nodes)
    loss = SplitAmbientLoss(U_lid=0.151, U_wall=0.3, T_ambient=8.0,
                            T_ambient_lid=-3.5)
    z = np.linspace(9.5, 0.5, n_nodes)
    for _ in range(50):
        T_nodes = rng.uniform(5.0, 95.0, n_nodes)
        A_wall = rng.uniform(1.0, 2000.0, n_nodes)
        loss.T_ambient = float(rng.uniform(-10.0, 30.0))
        Q = loss.Q_loss_nodes(T_nodes, A_wall, z)
        assert Q.tobytes() == ref_split_Q_loss_nodes(loss, T_nodes, A_wall).tobytes()


@pytest.mark.parametrize("n_layers", [1, 2, 4, 7])
@pytest.mark.parametrize("n_nodes", [1, 2, 5, 20, 50])
def test_transient_ground_loss_matches_reference_bitwise(n_nodes, n_layers):
    """advance() and Q_loss_nodes() over a long, varying trajectory."""
    rng = np.random.default_rng(RNG_SEED + 100 * n_layers + n_nodes)
    loss = TransientGroundLoss(
        U_lid=0.151, T_ambient_lid=8.0, lambda_soil=2.23, rho_soil=2000.0,
        cp_soil=800.0, d_total=7.5, n_layers=n_layers, T_far=8.0, T_init=11.0,
    )
    Tg_ref = np.full((n_nodes, n_layers), 11.0)
    state = loss.initial_state(n_nodes)
    z = np.linspace(9.5, 0.5, n_nodes)
    A_wall = rng.uniform(1.0, 2000.0, n_nodes)
    for _ in range(300):
        T_nodes = rng.uniform(5.0, 95.0, n_nodes)
        loss.T_ambient_lid = float(rng.uniform(-10.0, 30.0))
        loss.T_far = float(rng.uniform(6.0, 12.0))
        dt = float(rng.choice([60.0, 600.0, 3600.0]))
        Q = loss.Q_loss_nodes(T_nodes, A_wall, z, state)
        assert Q.tobytes() == ref_transient_Q_loss_nodes(
            loss, Tg_ref, T_nodes, A_wall).tobytes()
        state = loss.advance(T_nodes, A_wall, z, dt, state)
        ref_transient_advance(loss, Tg_ref, T_nodes, dt)
        assert state.tobytes() == Tg_ref.tobytes()
