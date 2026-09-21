"""Mass-balance sanity check on StorageInputs.ports.

StorageInputs' docstring requires sum(port.m_dot) == 0 for a fixed-volume
tank, but nothing enforced or even flagged a violation -- a caller who
forgot to balance a port pair got silent, physically inconsistent drift
instead of a warning. See BACKLOG.md P2.
"""

from __future__ import annotations

import warnings

import pytest

from thermal_energy_storage_model import Port, StorageInputs


def test_unbalanced_ports_trigger_runtime_warning(make_storage):
    storage = make_storage(n_nodes=10)
    state = storage.initialize(T_init=50.0)
    # Inlet only, no matching outlet -> gross imbalance.
    inputs = StorageInputs(ports=[Port(z=5.0, m_dot=5.0, T_in=80.0)])

    with pytest.warns(RuntimeWarning, match="not balanced"):
        storage.step(state, dt=10.0, inputs=inputs)


def test_balanced_ports_do_not_warn(make_storage):
    storage = make_storage(n_nodes=10)
    state = storage.initialize(T_init=50.0)
    inputs = StorageInputs.two_port(m_dot_charge=5.0, T_charge_in=80.0, height=5.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        storage.step(state, dt=10.0, inputs=inputs)  # must not raise


def test_negligible_numerical_mismatch_does_not_warn(make_storage):
    """The docstring promises robustness against small numerical mismatches
    -- a floating-point-scale imbalance must not trigger the warning."""
    storage = make_storage(n_nodes=10)
    state = storage.initialize(T_init=50.0)
    inputs = StorageInputs(ports=[
        Port(z=5.0, m_dot=5.0, T_in=80.0),
        Port(z=0.0, m_dot=-5.0 + 1e-10),  # ~2e-11 relative imbalance
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        storage.step(state, dt=10.0, inputs=inputs)  # must not raise


def test_idle_state_does_not_warn(make_storage):
    storage = make_storage(n_nodes=10)
    state = storage.initialize(T_init=50.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        storage.step(state, dt=10.0, inputs=StorageInputs(ports=[]))  # must not raise
