"""The co-simulation contract of ``ThermalStorage1D.step()``.

A coupling master may repeat a step from the same state (iterative coupling,
rollback, predictions) and changes the boundary conditions every step. That
requires

* a complete state: ``step()`` depends only on ``(state, dt, inputs)`` and
  the configuration, and does not modify ``state`` -- including the ground
  temperatures of ``TransientGroundLoss``, which lived inside the loss-model
  object before (stepping the same state twice gave different results),
* per-step boundary conditions: ``StorageInputs.T_ambient``.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_energy_storage_model import (
    ConstantAmbientLoss,
    HeatExchangerPort,
    SplitAmbientLoss,
    StorageInputs,
    StorageState,
    TransientGroundLoss,
)


def _transient_loss():
    return TransientGroundLoss(U_lid=0.2, T_ambient_lid=10.0, lambda_soil=1.5,
                               rho_soil=1800.0, cp_soil=900.0, d_total=5.0,
                               n_layers=4, T_far=9.0, T_init=12.0)


def _inputs(height):
    inputs = StorageInputs.two_port(m_dot_charge=4.0, T_charge_in=85.0,
                                    m_dot_discharge=2.0, T_discharge_in=40.0,
                                    height=height)
    inputs.hx_ports = [HeatExchangerPort(z=1.0, H_hx=1.0, UA=3000.0,
                                         m_dot_ext=0.5, T_ext_in=70.0)]
    return inputs


def _same_outputs(a, b):
    assert a.state.temperatures.tobytes() == b.state.temperatures.tobytes()
    assert a.state.loss_state.tobytes() == b.state.loss_state.tobytes()
    assert a.state.T_headspace == b.state.T_headspace
    assert a.port_temperatures == b.port_temperatures
    assert a.hx_outlet_temperatures == b.hx_outlet_temperatures
    assert a.Q_loss == b.Q_loss


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_step_is_a_pure_function_of_state_and_inputs(make_storage, solver):
    storage = make_storage(n_nodes=20, solver=solver, loss_model=_transient_loss(),
                           headspace=True)
    state = storage.initialize(T_init=np.linspace(80.0, 40.0, 20))
    for _ in range(5):   # move away from the initial ground field
        state = storage.step(state, dt=3600.0, inputs=_inputs(5.0)).state
    before = state.copy()

    first = storage.step(state, dt=3600.0, inputs=_inputs(5.0))
    storage.step(first.state, dt=3600.0, inputs=_inputs(5.0))   # unrelated step
    second = storage.step(state, dt=3600.0, inputs=_inputs(5.0))

    _same_outputs(first, second)
    # The input state is untouched
    assert state.temperatures.tobytes() == before.temperatures.tobytes()
    assert state.loss_state.tobytes() == before.loss_state.tobytes()
    assert state.T_headspace == before.T_headspace


def test_initialize_and_copy_carry_the_loss_state(make_storage):
    storage = make_storage(n_nodes=10, loss_model=_transient_loss())
    state = storage.initialize(T_init=60.0)
    assert state.loss_state.shape == (10, 4)
    assert np.all(state.loss_state == 12.0)

    copy = state.copy()
    assert copy.loss_state is not state.loss_state
    assert np.array_equal(copy.loss_state, state.loss_state)

    steady = make_storage(n_nodes=10).initialize(T_init=60.0)
    assert steady.loss_state is None


def test_state_without_loss_state_starts_from_initial_ground_field(make_storage):
    """A state built by hand (no loss_state) behaves like initialize()."""
    storage = make_storage(n_nodes=10, loss_model=_transient_loss())
    T0 = np.linspace(70.0, 40.0, 10)
    via_init = storage.step(storage.initialize(T_init=T0), 3600.0, _inputs(5.0))
    by_hand = storage.step(StorageState(temperatures=T0.copy()), 3600.0, _inputs(5.0))
    _same_outputs(via_init, by_hand)


def test_state_of_another_storage_is_rejected(make_storage):
    small = make_storage(n_nodes=10, loss_model=_transient_loss())
    large = make_storage(n_nodes=20, loss_model=_transient_loss())
    with pytest.raises(ValueError, match="shape"):
        large.step(small.initialize(T_init=60.0), 3600.0, StorageInputs())


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_ambient_input_equals_configured_ambient(make_storage, solver):
    """StorageInputs.T_ambient for one step == a loss model configured with it,
    and the configured loss model is not modified."""
    loss = ConstantAmbientLoss(U_loss=0.5, T_ambient=10.0)
    storage = make_storage(n_nodes=20, solver=solver, loss_model=loss)
    reference = make_storage(n_nodes=20, solver=solver,
                             loss_model=ConstantAmbientLoss(U_loss=0.5, T_ambient=-8.0))
    state = storage.initialize(T_init=70.0)

    inputs = StorageInputs(T_ambient=-8.0)
    out = storage.step(state, dt=3600.0, inputs=inputs)
    ref = reference.step(state, dt=3600.0, inputs=StorageInputs())

    assert out.state.temperatures.tobytes() == ref.state.temperatures.tobytes()
    assert out.Q_loss == ref.Q_loss
    assert loss.T_ambient == 10.0


def test_ambient_input_sets_only_the_lid_of_split_loss(make_storage):
    loss = SplitAmbientLoss(U_lid=0.2, U_wall=0.4, T_ambient=8.0, T_ambient_lid=12.0)
    storage = make_storage(n_nodes=10, loss_model=loss)
    reference = make_storage(n_nodes=10, loss_model=SplitAmbientLoss(
        U_lid=0.2, U_wall=0.4, T_ambient=8.0, T_ambient_lid=-3.0))
    state = storage.initialize(T_init=60.0)
    out = storage.step(state, dt=3600.0, inputs=StorageInputs(T_ambient=-3.0))
    ref = reference.step(state, dt=3600.0, inputs=StorageInputs())
    assert out.state.temperatures.tobytes() == ref.state.temperatures.tobytes()


def test_two_port_passes_ambient_temperature():
    inputs = StorageInputs.two_port(m_dot_charge=1.0, T_charge_in=80.0,
                                    height=5.0, T_ambient=3.5)
    assert inputs.T_ambient == 3.5
    assert StorageInputs.two_port(height=5.0).T_ambient is None
