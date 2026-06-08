"""Conservation tests: a closed, lossless tank must conserve thermal energy.

These tests exercise the transport core (advection + conduction + buoyancy)
without any sink or source, so the total stored energy and the mass-weighted
mean temperature are invariants that must hold to machine precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_energy_storage_model import StorageInputs


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_energy_conserved_closed_system(make_storage, solver):
    """No flow + no loss: stored energy is invariant under conduction/buoyancy."""
    storage = make_storage(
        n_nodes=20, solver=solver, U_loss=0.0, lambda_eff_factor=50.0,
    )
    state = storage.initialize(T_init=np.linspace(80.0, 40.0, 20))
    E0 = storage.get_stored_energy(state)

    idle = StorageInputs(ports=[])
    for _ in range(300):
        state = storage.step(state, dt=120.0, inputs=idle).state

    E1 = storage.get_stored_energy(state)
    assert abs(E1 - E0) / abs(E0) < 1e-12


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_mean_temperature_conserved_pure_conduction(make_storage, solver):
    """Pure conduction redistributes heat but preserves the profile mean."""
    storage = make_storage(
        n_nodes=20, solver=solver, U_loss=0.0, lambda_eff_factor=50.0,
        buoyancy=False,
    )
    T0 = np.linspace(80.0, 40.0, 20)
    state = storage.initialize(T_init=T0)

    idle = StorageInputs(ports=[])
    for _ in range(300):
        state = storage.step(state, dt=120.0, inputs=idle).state

    assert abs(state.temperatures.mean() - T0.mean()) < 1e-9


def test_advection_energy_balance(make_storage):
    """Lossless charging: stored-energy change equals net enthalpy flux in.

    With no wall loss, the only way energy enters the tank is the charging
    port. Accumulating m_dot * cp * (T_in - T_out) over the run must equal the
    measured change in stored energy. This is the discrete conservation
    statement for the advection scheme and must hold to machine precision.
    """
    from tests._helpers import CP

    storage = make_storage(
        n_nodes=30, solver="explicit", U_loss=0.0, lambda_eff_factor=1.0,
        advection_scheme="upwind",
    )
    state = storage.initialize(T_init=20.0)
    m_dot, T_in, dt = 10.0, 80.0, 100.0
    inputs = StorageInputs.two_port(
        m_dot_charge=m_dot, T_charge_in=T_in, height=5.0,
    )

    E0 = storage.get_stored_energy(state)
    enthalpy_in = 0.0
    for _ in range(20):
        out = storage.step(state, dt=dt, inputs=inputs)
        # ports order from two_port charge-only: [charge_in (top), charge_out (bottom)]
        T_out = out.port_temperatures[1]
        enthalpy_in += m_dot * CP * (T_in - T_out) * dt
        state = out.state

    dE = storage.get_stored_energy(state) - E0
    assert abs(dE - enthalpy_in) / abs(enthalpy_in) < 1e-10
