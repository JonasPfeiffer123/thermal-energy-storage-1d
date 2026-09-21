"""Coverage for small public utility methods with zero prior test usage:
``get_soc``, ``max_stable_dt``, ``check_cfl``. These are the kind of methods
a co-simulation caller queries before choosing a timestep or reporting state
of charge -- silent breakage here would only surface downstream, in an
integrating tool, not in this repository's own test suite.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# get_soc
# ---------------------------------------------------------------------------


def test_soc_full_tank_is_one(make_storage):
    storage = make_storage(n_nodes=10)
    state = storage.initialize(T_init=80.0)
    assert storage.get_soc(state, T_min=40.0, T_max=80.0) == pytest.approx(1.0)


def test_soc_empty_tank_is_zero(make_storage):
    storage = make_storage(n_nodes=10)
    state = storage.initialize(T_init=40.0)
    assert storage.get_soc(state, T_min=40.0, T_max=80.0) == pytest.approx(0.0)


def test_soc_half_charged_tank_is_one_half(make_storage):
    storage = make_storage(n_nodes=10)
    state = storage.initialize(T_init=60.0)  # midpoint of [40, 80]
    assert storage.get_soc(state, T_min=40.0, T_max=80.0) == pytest.approx(0.5)


def test_soc_clips_outside_zero_one_range(make_storage):
    storage = make_storage(n_nodes=10)
    hotter_than_max = storage.initialize(T_init=95.0)
    colder_than_min = storage.initialize(T_init=10.0)
    assert storage.get_soc(hotter_than_max, T_min=40.0, T_max=80.0) == pytest.approx(1.0)
    assert storage.get_soc(colder_than_min, T_min=40.0, T_max=80.0) == pytest.approx(0.0)


def test_soc_rejects_t_max_not_greater_than_t_min(make_storage):
    storage = make_storage(n_nodes=10)
    state = storage.initialize(T_init=60.0)
    with pytest.raises(ValueError):
        storage.get_soc(state, T_min=80.0, T_max=80.0)
    with pytest.raises(ValueError):
        storage.get_soc(state, T_min=80.0, T_max=40.0)


# ---------------------------------------------------------------------------
# max_stable_dt / check_cfl
# ---------------------------------------------------------------------------


def test_max_stable_dt_is_infinite_for_no_flow(make_storage):
    storage = make_storage(n_nodes=10)
    assert storage.max_stable_dt(m_dot_max=0.0) == float("inf")


def test_max_stable_dt_matches_cfl_formula(make_storage):
    storage = make_storage(n_nodes=10, volume=100.0, height=5.0)
    m_dot_max = 8.0
    v = m_dot_max / (storage.config.rho * storage.A_cross)
    expected = 0.9 * storage.dz / v
    assert storage.max_stable_dt(m_dot_max) == pytest.approx(expected)


def test_check_cfl_always_true_for_no_flow(make_storage):
    storage = make_storage(n_nodes=10)
    assert storage.check_cfl(dt=1e9, m_dot_max=0.0) is True


def test_check_cfl_agrees_with_max_stable_dt_boundary(make_storage):
    """max_stable_dt() already bakes in the 0.9 safety margin, so using it
    directly as dt must satisfy check_cfl (CFL = 0.9 <= 1); scaling it up to
    the raw CFL=1 boundary must still pass (<=), and stepping past that must
    fail."""
    storage = make_storage(n_nodes=10, volume=100.0, height=5.0)
    m_dot_max = 8.0
    dt_safe = storage.max_stable_dt(m_dot_max)
    assert storage.check_cfl(dt_safe, m_dot_max) is True

    dt_at_cfl_1 = dt_safe / 0.9
    assert storage.check_cfl(dt_at_cfl_1, m_dot_max) is True
    assert storage.check_cfl(dt_at_cfl_1 * 1.001, m_dot_max) is False
