"""Diffusor model correctness.

``UniformDiffusor`` had no coverage outside the ``--no-freetttes`` benchmark
script. These tests check ``node_weights()`` directly against hand-computed
node sets, plus a solver-level integration test confirming mass is actually
spread across multiple nodes (not silently collapsing to point-source
behaviour) and that mass/energy conservation still holds end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_energy_storage_model import (
    PointDiffusor,
    Port,
    StorageInputs,
    UniformDiffusor,
)

# 10 nodes, dz = 1 m, height = 10 m, index 0 = top (z=9.5) .. index 9 = bottom (z=0.5).
_NODE_HEIGHTS = np.linspace(9.5, 0.5, 10)


# ---------------------------------------------------------------------------
# PointDiffusor
# ---------------------------------------------------------------------------


def test_point_diffusor_selects_single_nearest_node():
    port = Port(z=7.3, m_dot=5.0, T_in=80.0)
    weights = PointDiffusor().node_weights(port, _NODE_HEIGHTS)
    assert weights == [(2, 1.0)]  # z=7.5 is nearest to 7.3


# ---------------------------------------------------------------------------
# UniformDiffusor
# ---------------------------------------------------------------------------


def test_uniform_diffusor_selects_all_nodes_in_zone_with_equal_weight():
    port = Port(z=7.0, m_dot=5.0, T_in=80.0)
    weights = UniformDiffusor(H_zone=3.0).node_weights(port, _NODE_HEIGHTS)
    # |z_k - 7.0| <= 1.5 -> z in {8.5, 7.5, 6.5, 5.5} -> indices 1..4
    assert [k for k, _ in weights] == [1, 2, 3, 4]
    assert all(w == pytest.approx(0.25) for _, w in weights)


def test_uniform_diffusor_weights_always_sum_to_one():
    port = Port(z=3.2, m_dot=5.0, T_in=80.0)
    weights = UniformDiffusor(H_zone=2.5).node_weights(port, _NODE_HEIGHTS)
    assert sum(w for _, w in weights) == pytest.approx(1.0)


def test_uniform_diffusor_falls_back_to_nearest_node_when_zone_too_narrow():
    """If H_zone is narrower than the node spacing and the port sits between
    two node centers, no node center may fall inside the zone -- must fall
    back to a single nearest node, matching PointDiffusor behaviour."""
    port = Port(z=7.0, m_dot=5.0, T_in=80.0)  # exactly between nodes 7.5 and 6.5
    weights = UniformDiffusor(H_zone=0.5).node_weights(port, _NODE_HEIGHTS)
    assert len(weights) == 1
    assert weights[0][1] == pytest.approx(1.0)
    assert weights[0][0] == PointDiffusor().node_weights(port, _NODE_HEIGHTS)[0][0]


def test_uniform_diffusor_rejects_nonpositive_zone_height():
    with pytest.raises(ValueError):
        UniformDiffusor(H_zone=0.0)
    with pytest.raises(ValueError):
        UniformDiffusor(H_zone=-1.0)


# ---------------------------------------------------------------------------
# Solver integration: mass is actually spread, and conservation still holds.
# ---------------------------------------------------------------------------


def test_uniform_diffusor_spreads_inflow_across_multiple_nodes(make_storage):
    """With a zone spanning several nodes, a single charging port must warm
    more than one node in a single step (unlike PointDiffusor, which would
    warm only the nearest node)."""
    storage = make_storage(
        n_nodes=20, solver="explicit", U_loss=0.0,
        diffusor_model=UniformDiffusor(H_zone=1.5),
    )
    state = storage.initialize(T_init=20.0)
    inputs = StorageInputs(ports=[Port(z=4.9, m_dot=8.0, T_in=80.0, label="in"),
                                   Port(z=0.0, m_dot=-8.0, label="out")])
    out = storage.step(state, dt=10.0, inputs=inputs)
    warmed = np.sum(out.state.temperatures > 20.0 + 1e-6)
    assert warmed > 1


def test_uniform_diffusor_conserves_energy_like_point_diffusor(make_storage):
    """Switching the diffusor model must not break the lossless energy
    balance: enthalpy added by the charging port must still equal the
    change in stored energy, to the same tolerance as the PointDiffusor
    case in test_conservation.py."""
    from tests._helpers import CP

    storage = make_storage(
        n_nodes=30, solver="explicit", U_loss=0.0, lambda_eff_factor=1.0,
        advection_scheme="upwind",
        diffusor_model=UniformDiffusor(H_zone=1.0),
    )
    state = storage.initialize(T_init=20.0)
    m_dot, T_in, dt = 10.0, 80.0, 100.0
    inputs = StorageInputs.two_port(m_dot_charge=m_dot, T_charge_in=T_in, height=5.0)

    E0 = storage.get_stored_energy(state)
    enthalpy_in = 0.0
    for _ in range(20):
        out = storage.step(state, dt=dt, inputs=inputs)
        T_out = out.port_temperatures[1]
        enthalpy_in += m_dot * CP * (T_in - T_out) * dt
        state = out.state

    dE = storage.get_stored_energy(state) - E0
    assert abs(dE - enthalpy_in) / abs(enthalpy_in) < 1e-10
