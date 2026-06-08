"""Golden-scenario regression test.

A fixed three-phase scenario (charge / idle / discharge) is run with the
implicit solver and constant fluid properties, and the resulting temperatures
are compared against hard-coded reference values. Any change in the numerical
core that shifts these values by more than 1 mK will fail this test, flagging
an unintended behavioural change.

The reference values were generated with this same configuration; they are not
an external ground truth but a self-consistent regression baseline.
"""

from __future__ import annotations

import pytest

from thermal_energy_storage_model import StorageInputs

# Reference outputs (constant fluid, implicit + upwind, dt = 3600 s).
REF = {
    "charge_top": 83.810935,
    "charge_bot": 61.051884,
    "idle_top": 83.316037,
    "idle_bot": 60.904028,
    "final_top": 73.049360,
    "final_bot": 46.032594,
    "final_mean": 59.859434,
    "discharge_outlet": 78.003608,
}
ATOL = 1e-3  # K


def test_three_phase_golden_scenario(make_storage):
    storage = make_storage(
        n_nodes=20, solver="implicit", advection_scheme="upwind",
        U_loss=0.3, T_ambient=10.0,
    )
    state = storage.initialize(T_init=60.0)

    # Phase 1: charge 2 h at 85 °C, 5 kg/s.
    charge = StorageInputs.two_port(m_dot_charge=5.0, T_charge_in=85.0, height=5.0)
    for _ in range(2):
        state = storage.step(state, dt=3600.0, inputs=charge).state
    assert state.T_top == pytest.approx(REF["charge_top"], abs=ATOL)
    assert state.T_bottom == pytest.approx(REF["charge_bot"], abs=ATOL)

    # Phase 2: idle 3 h.
    for _ in range(3):
        state = storage.step(state, dt=3600.0, inputs=StorageInputs(ports=[])).state
    assert state.T_top == pytest.approx(REF["idle_top"], abs=ATOL)
    assert state.T_bottom == pytest.approx(REF["idle_bot"], abs=ATOL)

    # Phase 3: discharge 2 h at 45 °C, 4 kg/s.
    discharge = StorageInputs.two_port(m_dot_discharge=4.0, T_discharge_in=45.0, height=5.0)
    out = None
    for _ in range(2):
        out = storage.step(state, dt=3600.0, inputs=discharge)
        state = out.state
    assert state.T_top == pytest.approx(REF["final_top"], abs=ATOL)
    assert state.T_bottom == pytest.approx(REF["final_bot"], abs=ATOL)
    assert state.T_mean == pytest.approx(REF["final_mean"], abs=ATOL)
    # Discharge outlet is the second port (discharge_out at the top).
    assert out.port_temperatures[1] == pytest.approx(REF["discharge_outlet"], abs=ATOL)
