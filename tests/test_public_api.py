"""Coverage for small public utility methods with zero prior test usage:
``get_soc``, ``max_stable_dt``, ``check_cfl``. These are the kind of methods
a co-simulation caller queries before choosing a timestep or reporting state
of charge -- silent breakage here would only surface downstream, in an
integrating tool, not in this repository's own test suite. Also the input
validation of the configuration and of ``step()``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from thermal_energy_storage_model import (
    HeatExchangerPort,
    Port,
    StorageConfig,
    StorageInputs,
    ThermalStorage1D,
    WaterProperties,
)

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


def test_soc_full_tank_is_one_with_temperature_dependent_water(make_storage):
    """E_full uses the fluid properties at T_max (was the nominal rho/cp,
    giving SOC = 0.9916 for a tank uniformly at T_max = 90 °C)."""
    storage = make_storage(n_nodes=10, fluid=WaterProperties())
    full = storage.initialize(T_init=90.0)
    empty = storage.initialize(T_init=40.0)
    assert storage.get_soc(full, T_min=40.0, T_max=90.0) == 1.0
    assert storage.get_soc(empty, T_min=40.0, T_max=90.0) == 0.0


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


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("overrides, error", [
    (dict(volume=math.nan), ValueError),
    (dict(height=-1.0), ValueError),
    (dict(n_nodes=2.5), TypeError),
    (dict(n_nodes=True), TypeError),
    (dict(n_nodes=1), ValueError),
    (dict(U_loss=math.nan), ValueError),
    (dict(T_ambient=math.inf), ValueError),
    (dict(T_ground=10.0, z_ground_layer=0.0), ValueError),
    (dict(headspace=True, H_headspace=-0.5), ValueError),
    (dict(headspace=True, rho_headspace=0.0), ValueError),
    (dict(headspace=True, U_roof=-0.1), ValueError),
])
def test_invalid_configuration_is_rejected(overrides, error):
    params = dict(volume=10.0, height=2.0, n_nodes=10)
    params.update(overrides)
    with pytest.raises(error):
        ThermalStorage1D(StorageConfig(**params))


def test_numpy_integer_node_count_is_accepted():
    storage = ThermalStorage1D(StorageConfig(volume=10.0, height=2.0, n_nodes=np.int64(8)))
    assert storage.n == 8


def test_non_finite_initial_temperature_is_rejected(make_storage):
    storage = make_storage(n_nodes=5)
    with pytest.raises(ValueError, match="finite"):
        storage.initialize(T_init=[60.0, 60.0, math.nan, 60.0, 60.0])


def _hx(**kw):
    params = dict(z=2.5, H_hx=1.0, UA=1000.0, m_dot_ext=0.5, T_ext_in=80.0)
    params.update(kw)
    return HeatExchangerPort(**params)


@pytest.mark.parametrize("dt, inputs", [
    (0.0, StorageInputs()),
    (-60.0, StorageInputs()),
    (math.nan, StorageInputs()),
    (60.0, StorageInputs.two_port(m_dot_charge=1.0, T_charge_in=math.nan, height=5.0)),
    (60.0, StorageInputs(ports=[Port(z=math.nan, m_dot=0.0)])),
    (60.0, StorageInputs(ports=[Port(z=5.0, m_dot=math.inf, T_in=80.0)])),
    (60.0, StorageInputs(hx_ports=[_hx(m_dot_ext=-0.5)])),
    (60.0, StorageInputs(hx_ports=[_hx(UA=math.nan)])),
    (60.0, StorageInputs(hx_ports=[_hx(cp_ext=0.0)])),
    (60.0, StorageInputs(hx_ports=[_hx(segmented=True, flow_direction="up")])),
    (60.0, StorageInputs(T_ambient=math.nan)),
])
def test_invalid_step_inputs_are_rejected(make_storage, dt, inputs):
    storage = make_storage(n_nodes=10)
    state = storage.initialize(T_init=60.0)
    with pytest.raises(ValueError):
        storage.step(state, dt=dt, inputs=inputs)


def test_outlet_ports_need_no_inlet_temperature(make_storage):
    """T_in is not used for an outlet and may be NaN there."""
    storage = make_storage(n_nodes=10)
    state = storage.initialize(T_init=60.0)
    inputs = StorageInputs(ports=[Port(z=5.0, m_dot=1.0, T_in=80.0),
                                  Port(z=0.0, m_dot=-1.0, T_in=math.nan)])
    out = storage.step(state, dt=60.0, inputs=inputs)
    assert np.all(np.isfinite(out.state.temperatures))
