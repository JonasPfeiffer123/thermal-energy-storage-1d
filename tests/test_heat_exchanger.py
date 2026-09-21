"""HeatExchangerPort (epsilon-NTU) correctness -- lumped and segmented modes.

This is the single largest untested piece of the solver core: before this
file, ``HeatExchangerPort`` had zero coverage anywhere in the repository
(no unit test, no benchmark script, no UI, no example). A wiring bug in
either mode could ship silently.

Private solver methods (``_compute_hx_source_terms``, ``_get_hx_weights``)
are called directly to isolate the epsilon-NTU math from time integration,
following the existing precedent in this suite (``test_tdma.py`` calls
``_solve_tdma`` directly, ``test_buoyancy.py`` calls
``_convective_adjustment`` directly).
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_energy_storage_model import HeatExchangerPort, StorageInputs

# ---------------------------------------------------------------------------
# HX zone weighting (_get_hx_weights)
# ---------------------------------------------------------------------------


def test_hx_weights_select_nodes_in_zone_with_equal_weight(make_storage):
    storage = make_storage(n_nodes=20, height=5.0)  # dz = 0.25
    hx = HeatExchangerPort(z=2.5, H_hx=1.0, UA=1000.0, m_dot_ext=0.5, T_ext_in=90.0)
    weights = storage._get_hx_weights(hx)
    z_lo, z_hi = 2.5 - 0.5, 2.5 + 0.5
    expected_indices = [
        k for k, z in enumerate(storage._z_nodes) if z_lo <= z <= z_hi
    ]
    assert [k for k, _ in weights] == expected_indices
    assert len(expected_indices) > 1
    assert all(w == pytest.approx(1.0 / len(expected_indices)) for _, w in weights)
    assert sum(w for _, w in weights) == pytest.approx(1.0)


def test_hx_weights_fall_back_to_nearest_node(make_storage):
    """A zone narrower than the node spacing, centered between two node
    centers, must fall back to a single nearest node."""
    storage = make_storage(n_nodes=10, height=10.0)  # dz = 1.0, centers at *.5
    hx = HeatExchangerPort(z=5.0, H_hx=0.1, UA=1000.0, m_dot_ext=0.5, T_ext_in=90.0)
    weights = storage._get_hx_weights(hx)
    assert len(weights) == 1
    assert weights[0][1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Lumped mode
# ---------------------------------------------------------------------------


def test_lumped_hx_matches_epsilon_ntu_formula(make_storage):
    storage = make_storage(n_nodes=20, height=5.0)
    T = np.linspace(80.0, 40.0, 20)
    hx = HeatExchangerPort(z=2.5, H_hx=1.0, UA=2000.0, m_dot_ext=0.5, T_ext_in=90.0)

    Q_nodes, T_ext_out = storage._compute_hx_source_terms(T, [hx])

    weights = storage._get_hx_weights(hx)
    T_zone_mean = sum(T[k] * w for k, w in weights)
    C_ext = hx.m_dot_ext * hx.cp_ext
    eps = 1.0 - np.exp(-hx.UA / C_ext)
    Q_expected = eps * C_ext * (hx.T_ext_in - T_zone_mean)

    assert np.sum(Q_nodes) == pytest.approx(Q_expected)
    assert T_ext_out[0] == pytest.approx(hx.T_ext_in - Q_expected / C_ext)
    # Lumped mode spreads the same total evenly across the zone.
    in_zone = Q_nodes[Q_nodes != 0.0]
    assert np.allclose(in_zone, in_zone[0])


def test_lumped_hx_zero_external_flow_is_a_no_op(make_storage):
    storage = make_storage(n_nodes=10, height=5.0)
    T = np.full(10, 50.0)
    hx = HeatExchangerPort(z=2.5, H_hx=1.0, UA=1000.0, m_dot_ext=0.0, T_ext_in=90.0)
    Q_nodes, T_ext_out = storage._compute_hx_source_terms(T, [hx])
    assert np.all(Q_nodes == 0.0)
    assert T_ext_out == [pytest.approx(90.0)]


def test_multiple_hx_ports_superpose_independently(make_storage):
    storage = make_storage(n_nodes=20, height=5.0)
    T = np.linspace(80.0, 40.0, 20)
    hx_top = HeatExchangerPort(z=4.5, H_hx=1.0, UA=1500.0, m_dot_ext=0.4, T_ext_in=90.0, label="top")
    hx_bottom = HeatExchangerPort(z=0.5, H_hx=1.0, UA=1200.0, m_dot_ext=0.3, T_ext_in=95.0, label="bottom")

    Q_both, T_out_both = storage._compute_hx_source_terms(T, [hx_top, hx_bottom])
    Q_top, T_out_top = storage._compute_hx_source_terms(T, [hx_top])
    Q_bot, T_out_bot = storage._compute_hx_source_terms(T, [hx_bottom])

    assert np.allclose(Q_both, Q_top + Q_bot)
    assert T_out_both == [pytest.approx(T_out_top[0]), pytest.approx(T_out_bot[0])]


# ---------------------------------------------------------------------------
# Segmented mode
# ---------------------------------------------------------------------------


def _reference_segmented_hx(T, z_nodes, hx):
    """Independent re-implementation of the documented segmented epsilon-NTU
    recursion (docs/physics.md, Section 14), used to cross-check the solver's
    private implementation rather than restating its code.

    Node order is derived from the documented flow semantics directly
    (index 0 = top / high z): "downward" enters at the top, so it must
    process nodes in descending-z order; "upward" enters at the bottom, so
    ascending-z order.
    """
    z_lo, z_hi = hx.z - 0.5 * hx.H_hx, hx.z + 0.5 * hx.H_hx
    indices = [k for k, z in enumerate(z_nodes) if z_lo <= z <= z_hi]
    reverse = hx.flow_direction == "downward"
    ordered = sorted(indices, key=lambda k: z_nodes[k], reverse=reverse)
    n_seg = len(ordered)
    C_ext = hx.m_dot_ext * hx.cp_ext
    eps_seg = 1.0 - np.exp(-(hx.UA / n_seg) / C_ext)
    T_cur = hx.T_ext_in
    Q = np.zeros(len(z_nodes))
    for k in ordered:
        Q_k = eps_seg * C_ext * (T_cur - T[k])
        Q[k] = Q_k
        T_cur -= Q_k / C_ext
    return Q, T_cur


@pytest.mark.parametrize("flow_direction", ["downward", "upward"])
def test_segmented_hx_matches_independent_reference(make_storage, flow_direction):
    storage = make_storage(n_nodes=20, height=5.0)
    T = np.linspace(80.0, 20.0, 20)  # thermocline profile
    hx = HeatExchangerPort(
        z=2.5, H_hx=2.0, UA=4000.0, m_dot_ext=0.4, T_ext_in=85.0,
        segmented=True, flow_direction=flow_direction,
    )
    Q_nodes, T_ext_out = storage._compute_hx_source_terms(T, [hx])
    Q_ref, T_out_ref = _reference_segmented_hx(T, storage._z_nodes, hx)
    assert np.allclose(Q_nodes, Q_ref)
    assert T_ext_out[0] == pytest.approx(T_out_ref)


def test_segmented_hx_flow_direction_determines_entry_node(make_storage):
    """The first node processed (which sees the *full* T_ext_in, undiminished
    by any prior heat transfer) must be the top of the zone for "downward"
    (entry at top, per its docstring) and the bottom of the zone for
    "upward". This is the direct physical meaning of the parameter, and
    the exact property a sign inversion in the node-ordering logic breaks
    silently while leaving totals merely "some other plausible number" --
    which is what made this bug invisible without a test pinned to the
    documented entry point rather than an aggregate.
    """
    storage = make_storage(n_nodes=20, height=5.0)
    T = np.linspace(80.0, 20.0, 20)  # hot top (index 0) -> cold bottom (index 19)
    kwargs = dict(z=2.5, H_hx=2.0, UA=4000.0, m_dot_ext=0.4, T_ext_in=85.0, segmented=True)
    hx_down = HeatExchangerPort(flow_direction="downward", **kwargs)
    hx_up = HeatExchangerPort(flow_direction="upward", **kwargs)

    zone_indices = [k for k, _ in storage._get_hx_weights(hx_down)]
    top_of_zone = min(zone_indices)      # smallest index = highest z = top
    bottom_of_zone = max(zone_indices)   # largest index = lowest z = bottom

    C_ext = hx_down.m_dot_ext * hx_down.cp_ext
    eps_seg = 1.0 - np.exp(-(hx_down.UA / len(zone_indices)) / C_ext)

    Q_down, _ = storage._compute_hx_source_terms(T, [hx_down])
    Q_up, _ = storage._compute_hx_source_terms(T, [hx_up])

    Q_entry_down = eps_seg * C_ext * (hx_down.T_ext_in - T[top_of_zone])
    Q_entry_up = eps_seg * C_ext * (hx_up.T_ext_in - T[bottom_of_zone])
    assert Q_down[top_of_zone] == pytest.approx(Q_entry_down)
    assert Q_up[bottom_of_zone] == pytest.approx(Q_entry_up)

    # And the two directions must actually disagree on the aggregate result
    # across a thermocline (verified: downward totals 60127 W, upward 46350 W
    # for these parameters -- upward's large initial extraction at the cold
    # end depletes T_ext_current enough that it partially reverses, giving
    # heat back to the hot top-of-zone water later in its path).
    assert np.sum(Q_down) != pytest.approx(np.sum(Q_up), rel=1e-3)


def test_segmented_matches_lumped_total_in_homogeneous_zone(make_storage):
    """In a homogeneous-temperature zone, splitting UA into N equal series
    stages against the *same* constant sink temperature is an exact
    analytical identity with the single-stage lumped result:
    1 - (1 - (1-exp(-NTU/N)))^N == 1 - exp(-NTU). Per-node distribution
    still differs (segmented front-loads heat transfer), but the totals
    and T_ext_out must match to floating-point precision."""
    storage = make_storage(n_nodes=20, height=5.0)
    T = np.full(20, 40.0)
    kwargs = dict(z=2.5, H_hx=2.0, UA=4000.0, m_dot_ext=0.4, T_ext_in=85.0)
    hx_lumped = HeatExchangerPort(segmented=False, **kwargs)
    hx_seg = HeatExchangerPort(segmented=True, flow_direction="downward", **kwargs)

    Q_lumped, T_out_lumped = storage._compute_hx_source_terms(T, [hx_lumped])
    Q_seg, T_out_seg = storage._compute_hx_source_terms(T, [hx_seg])

    assert np.sum(Q_seg) == pytest.approx(np.sum(Q_lumped), rel=1e-12)
    assert T_out_seg[0] == pytest.approx(T_out_lumped[0], rel=1e-12)

    # But the per-node split is not the same: segmented front-loads heat
    # transfer onto the first node encountered (here, downward -> top of
    # zone), unlike lumped's uniform split.
    in_zone_seg = Q_seg[Q_seg != 0.0]
    assert not np.allclose(in_zone_seg, in_zone_seg[0])
    assert np.all(np.diff(in_zone_seg) < 0.0)  # strictly decreasing along flow


# ---------------------------------------------------------------------------
# End-to-end energy conservation through step()
# ---------------------------------------------------------------------------


def test_hx_energy_conservation_over_time(make_storage):
    """Lossless tank, HX-only charging (no hydraulic ports): the stored-
    energy change must equal the accumulated enthalpy given up by the
    external HX fluid, m_dot_ext * cp_ext * (T_ext_in - T_ext_out) * dt."""
    storage = make_storage(n_nodes=20, height=5.0, U_loss=0.0)
    state = storage.initialize(T_init=40.0)
    hx = HeatExchangerPort(z=2.5, H_hx=5.0, UA=3000.0, m_dot_ext=0.3, T_ext_in=90.0)

    E0 = storage.get_stored_energy(state)
    enthalpy_in = 0.0
    dt = 60.0
    for _ in range(30):
        out = storage.step(state, dt=dt, inputs=StorageInputs(hx_ports=[hx]))
        T_ext_out = out.hx_outlet_temperatures[0]
        enthalpy_in += hx.m_dot_ext * hx.cp_ext * (hx.T_ext_in - T_ext_out) * dt
        state = out.state

    dE = storage.get_stored_energy(state) - E0
    assert abs(dE - enthalpy_in) / abs(enthalpy_in) < 1e-8


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_hx_warms_zone_nodes_only(make_storage, solver):
    """An HX zone confined to the top of the tank must not directly affect
    nodes clearly outside the zone within a single step (no conduction)."""
    storage = make_storage(
        n_nodes=20, height=5.0, solver=solver, U_loss=0.0, lambda_eff_factor=1.0,
    )
    state = storage.initialize(T_init=40.0)
    hx = HeatExchangerPort(z=4.875, H_hx=0.25, UA=5000.0, m_dot_ext=0.5, T_ext_in=90.0)
    out = storage.step(state, dt=10.0, inputs=StorageInputs(hx_ports=[hx]))
    T = out.state.temperatures
    assert T[0] > 40.0 + 1e-6          # top node (in zone) warmed
    assert T[-1] == pytest.approx(40.0, abs=1e-9)  # bottom node untouched
