"""Analytical verification against closed-form reference solutions.

Each test isolates one physical mechanism and checks the numerical result
against an exact (or near-exact) analytical solution.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_energy_storage_model import (
    ConstantAmbientLoss,
    StorageInputs,
)
from tests._helpers import RHO


def test_pure_conduction_flattens_profile_monotonically(make_storage):
    """Conduction is dissipative: the profile variance must decrease at every step."""
    storage = make_storage(
        n_nodes=20, solver="explicit", U_loss=0.0, lambda_eff_factor=50.0,
        buoyancy=False,
    )
    state = storage.initialize(T_init=np.linspace(80.0, 40.0, 20))

    idle = StorageInputs(ports=[])
    variances = [float(np.var(state.temperatures))]
    for _ in range(200):
        state = storage.step(state, dt=60.0, inputs=idle).state
        variances.append(float(np.var(state.temperatures)))

    diffs = np.diff(variances)
    assert np.all(diffs <= 1e-9)  # non-increasing (tolerate fp noise)
    assert variances[-1] < variances[0]  # actually flattened


def test_exponential_cooling(make_storage):
    """Uniform tank with wall loss only relaxes exponentially toward ambient.

    A two-node cylinder has symmetric lid/bottom areas, so a uniform profile
    cools uniformly and the lumped energy balance C dT/dt = -UA (T - T_amb)
    applies exactly. The mass-weighted mean must follow T_amb + (T0 - T_amb)
    exp(-k t), with k inferred from the first step.
    """
    T_amb, T0 = 10.0, 70.0
    storage = make_storage(
        n_nodes=2, solver="explicit", lambda_eff_factor=1.0, buoyancy=False,
        loss_model=ConstantAmbientLoss(U_loss=2.0, T_ambient=T_amb),
    )

    # Infer the lumped decay constant k = UA / C from a tiny probe step.
    probe = storage.step(storage.initialize(T_init=T0), dt=1.0,
                         inputs=StorageInputs(ports=[]))
    k = -((probe.state.T_mean - T0) / 1.0) / (T0 - T_amb)
    assert k > 0.0

    state = storage.initialize(T_init=T0)
    dt, t = 30.0, 0.0
    ratios, prev = [], state.T_mean - T_amb
    max_rel_err = 0.0
    for _ in range(120):
        state = storage.step(state, dt=dt, inputs=StorageInputs(ports=[])).state
        t += dt
        T_analytic = T_amb + (T0 - T_amb) * np.exp(-k * t)
        max_rel_err = max(max_rel_err, abs(state.T_mean - T_analytic) / (T_analytic - T_amb))
        cur = state.T_mean - T_amb
        ratios.append(cur / prev)
        prev = cur

    assert float(np.std(state.temperatures)) < 1e-9       # stays uniform
    assert max(ratios) - min(ratios) < 1e-10              # exact geometric decay
    assert max_rel_err < 1e-4                             # matches continuous solution
    assert state.T_mean > T_amb                           # never overshoots ambient


def test_plug_flow_front_speed(make_storage):
    """A charging thermocline advances at the bulk fluid velocity v = m_dot/(rho A)."""
    N = 40
    storage = make_storage(
        n_nodes=N, solver="explicit", U_loss=0.0, lambda_eff_factor=1.0,
        advection_scheme="upwind",
    )
    state = storage.initialize(T_init=20.0)

    height = 5.0
    A = 100.0 / height
    m_dot = 20.0
    v = m_dot / (RHO * A)
    inputs = StorageInputs.two_port(m_dot_charge=m_dot, T_charge_in=80.0, height=height)

    t_total, dt = 800.0, 100.0
    for _ in range(int(t_total / dt)):
        state = storage.step(state, dt=dt, inputs=inputs).state

    # Depth (from top) where the profile crosses the 50 % temperature.
    z = np.asarray(storage._z_nodes, dtype=float)  # node centers, index 0 = top
    T = state.temperatures
    idx = int(np.where(T < 50.0)[0][0])
    depth_model = height - z[idx]
    depth_analytic = v * t_total
    dz = height / N
    assert abs(depth_model - depth_analytic) < 2.0 * dz
