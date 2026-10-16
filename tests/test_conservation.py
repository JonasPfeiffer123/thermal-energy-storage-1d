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


# ---------------------------------------------------------------------------
# Energy balance of the step outputs (co-simulation interface)
# ---------------------------------------------------------------------------
# The outputs of step() must close the energy balance: the change of the
# stored energy equals the enthalpy brought in by the inlets minus the
# enthalpy taken out at the reported outlet temperatures, plus the heat-
# exchanger heat, minus the reported loss. This is what a network partner
# books; any mismatch is energy created or lost at the coupling interface.


def _port_enthalpy(inputs, out, dt):
    """Σ ṁ·cp·T·dt over the ports: inlets at T_in, outlets at the reported T."""
    from tests._helpers import CP

    E = 0.0
    for port, T_port in zip(inputs.ports, out.port_temperatures):
        T = port.T_in if port.m_dot > 0.0 else T_port
        E += port.m_dot * CP * T * dt
    return E


def _hx_heat(inputs, out, dt):
    return sum(hx.m_dot_ext * hx.cp_ext * (hx.T_ext_in - T_out) * dt
               for hx, T_out in zip(inputs.hx_ports, out.hx_outlet_temperatures))


def _assert_balance(storage, state, schedule, dt):
    E0 = storage.get_stored_energy(state)
    booked = throughput = 0.0
    for inputs in schedule:
        out = storage.step(state, dt=dt, inputs=inputs)
        booked += (_port_enthalpy(inputs, out, dt) + _hx_heat(inputs, out, dt)
                   - out.Q_loss * dt)
        throughput += sum(abs(p.m_dot) for p in inputs.ports) * 4187.0 * 60.0 * dt
        state = out.state
    dE = storage.get_stored_energy(state) - E0
    assert abs(dE - booked) <= 1e-10 * max(throughput, abs(dE), 1.0)


def _simultaneous_schedule(height):
    """Charge and discharge through the same top and bottom nodes, with
    either loop dominating and exactly balanced loops (pure bypass)."""
    return [
        StorageInputs.two_port(m_dot_charge=mc, T_charge_in=Tc,
                               m_dot_discharge=md, T_discharge_in=Td,
                               height=height)
        for mc, Tc, md, Td in [
            (8.0, 85.0, 3.0, 40.0), (3.0, 80.0, 8.0, 45.0), (5.0, 60.0, 5.0, 50.0),
            (6.0, 90.0, 0.0, 0.0), (0.0, 0.0, 6.0, 35.0),
        ] * 6
    ]


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
@pytest.mark.parametrize("scheme", ["upwind", "tvd"])
def test_energy_balance_simultaneous_operation_shared_nodes(make_storage, solver, scheme):
    """Inflow and outflow at the same node: the outflow takes the inflowing
    fluid first, and the reported outlet temperature says so.

    Before, the outlet ports reported the node temperature while the energy
    balance had removed the inlet temperature (and the implicit solver the
    end-of-step temperature): up to tens of K for simultaneous operation.
    """
    storage = make_storage(n_nodes=20, solver=solver, advection_scheme=scheme,
                           U_loss=0.0)
    state = storage.initialize(T_init=np.linspace(75.0, 45.0, 20))
    _assert_balance(storage, state, _simultaneous_schedule(5.0), dt=600.0)


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_energy_balance_with_wall_loss_and_substeps(make_storage, solver):
    """Long coupling steps: explicit sub-steps average the outputs consistently."""
    storage = make_storage(n_nodes=30, solver=solver, U_loss=0.4)
    state = storage.initialize(T_init=60.0)
    _assert_balance(storage, state, _simultaneous_schedule(5.0), dt=3600.0)


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_energy_balance_uniform_diffusor(make_storage, solver):
    from thermal_energy_storage_model import Port, UniformDiffusor

    storage = make_storage(n_nodes=20, solver=solver, U_loss=0.0,
                           diffusor_model=UniformDiffusor(H_zone=1.0))
    state = storage.initialize(T_init=np.linspace(75.0, 45.0, 20))
    schedule = [StorageInputs(ports=[
        Port(z=4.8, m_dot=+6.0, T_in=88.0), Port(z=4.5, m_dot=-2.0),
        Port(z=2.5, m_dot=+3.0, T_in=50.0), Port(z=0.2, m_dot=-7.0),
    ])] * 20
    _assert_balance(storage, state, schedule, dt=900.0)


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
@pytest.mark.parametrize("segmented", [False, True])
def test_energy_balance_heat_exchangers(make_storage, solver, segmented):
    from thermal_energy_storage_model import HeatExchangerPort

    storage = make_storage(n_nodes=20, solver=solver, U_loss=0.3)
    state = storage.initialize(T_init=np.linspace(70.0, 30.0, 20))
    hx = [HeatExchangerPort(z=1.5, H_hx=2.0, UA=6000.0, m_dot_ext=1.0,
                            T_ext_in=80.0, segmented=segmented,
                            flow_direction="upward")]
    schedule = [StorageInputs.two_port(m_dot_discharge=2.0, T_discharge_in=35.0,
                                       height=5.0)] * 24
    for inputs in schedule:
        inputs.hx_ports = hx
    _assert_balance(storage, state, schedule, dt=3600.0)


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_energy_balance_transient_ground_loss(make_storage, solver):
    """Q_loss is the loss applied in the step (it was evaluated with the
    ground field after the step before)."""
    from thermal_energy_storage_model import TransientGroundLoss

    loss = TransientGroundLoss(U_lid=0.2, T_ambient_lid=10.0, lambda_soil=1.5,
                               rho_soil=1800.0, cp_soil=900.0, d_total=5.0,
                               n_layers=4, T_far=9.0)
    storage = make_storage(n_nodes=20, solver=solver, loss_model=loss)
    state = storage.initialize(T_init=80.0)
    _assert_balance(storage, state, _simultaneous_schedule(5.0), dt=3600.0)
