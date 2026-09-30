"""
Bit-identity of the Python-float kernels with the former array code.

The reference functions below are the NumPy implementations of
``ThermalStorage1D`` before the step() performance work (v1.0.0). The kernels
in ``thermal_energy_storage_model._kernels`` must reproduce them bit for bit
on randomized inputs; the one-year golden-master tests cover the integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_energy_storage_model._kernels import (
    convective_adjustment,
    implicit_solve,
    solve_tdma,
    tvd_correction,
    tvd_fluxes,
)

RNG_SEED = 20260930
TVD_EPS = 1e-12


# ---------------------------------------------------------------------------
# Reference implementations (v1.0.0 array code)
# ---------------------------------------------------------------------------

def ref_solve_tdma(a, d, c, b):
    n = len(d)
    d_ = d.copy()
    b_ = b.copy()
    for i in range(1, n):
        w = a[i] / d_[i - 1]
        d_[i] -= w * c[i - 1]
        b_[i] -= w * b_[i - 1]
    x = np.empty(n)
    x[-1] = b_[-1] / d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b_[i] - c[i] * x[i + 1]) / d_[i]
    return x


def ref_van_leer(r):
    abs_r = np.abs(r)
    return (r + abs_r) / (1.0 + abs_r)


def ref_tvd(T, F_int, cp, dt, rho, A_iface, dz):
    n = len(T)
    dT = np.diff(T)
    Q_tvd = np.zeros(n)
    for k in range(n - 1):
        F_k = float(F_int[k])
        if abs(F_k) < 1e-30:
            continue
        dT_k = float(dT[k])
        cfl_k = min(abs(F_k) * dt / (rho * float(A_iface[k]) * dz), 1.0)
        if F_k >= 0.0:
            dT_up = float(dT[k - 1]) if k > 0 else 0.0
            dT_safe = dT_k + (TVD_EPS if dT_k >= 0 else -TVD_EPS)
            phi = float(ref_van_leer(np.array([dT_up / dT_safe]))[0])
            corr = 0.5 * F_k * cp * (1.0 - cfl_k) * phi * dT_k
            Q_tvd[k] -= corr
            Q_tvd[k + 1] += corr
        else:
            dT_flow = -dT_k
            dT_up = float(-dT[k + 1]) if k < n - 2 else 0.0
            dT_flow_safe = dT_flow + (TVD_EPS if dT_flow >= 0 else -TVD_EPS)
            phi = float(ref_van_leer(np.array([dT_up / dT_flow_safe]))[0])
            corr = 0.5 * (-F_k) * cp * (1.0 - cfl_k) * phi * dT_flow
            Q_tvd[k] += corr
            Q_tvd[k + 1] -= corr
    return Q_tvd


def ref_implicit(T, dt, F, S, T_src, C_nodes, K, cp, Q_loss, Q_hx, Q_tvd):
    n = len(T)
    C_dt = C_nodes / dt
    a = np.zeros(n)
    d = np.zeros(n)
    c = np.zeros(n)
    b = np.zeros(n)
    d += C_dt
    b += C_dt * T
    b += Q_loss
    b += Q_hx
    if Q_tvd is not None:
        b += Q_tvd
    d[:-1] += K
    c[:-1] -= K
    d[1:] += K
    a[1:] -= K
    for k in range(n - 1):
        F_val = float(F[k + 1]) * cp
        if F_val >= 0.0:
            d[k] += F_val
            a[k + 1] -= F_val
        else:
            c[k] += F_val
            d[k + 1] -= F_val
    d -= cp * np.minimum(S, 0.0)
    b += cp * np.maximum(S, 0.0) * T_src
    return ref_solve_tdma(a, d, c, b)


def ref_convective_adjustment(T, m_nodes):
    n = len(T)
    blocks: list = []
    for i in range(n):
        T_wm = float(T[i] * m_nodes[i])
        m_sum = float(m_nodes[i])
        start = i
        while blocks and blocks[-1][0] / blocks[-1][1] < T_wm / m_sum:
            prev = blocks.pop()
            T_wm += prev[0]
            m_sum += prev[1]
            start = prev[2]
        blocks.append([T_wm, m_sum, start])
    result = T.copy()
    for k in range(len(blocks)):
        end = blocks[k + 1][2] if k + 1 < len(blocks) else n
        result[blocks[k][2]:end] = blocks[k][0] / blocks[k][1]
    return result


# ---------------------------------------------------------------------------
# Random inputs
# ---------------------------------------------------------------------------

def _profile(rng, n):
    """Stratified profile with occasional inversions / flat regions."""
    T = np.sort(rng.uniform(35.0, 90.0, n))[::-1].copy()
    kind = rng.integers(4)
    if kind == 1:
        for _ in range(rng.integers(1, 4)):
            T[rng.integers(n)] += rng.normal(0.0, 5.0)
    elif kind == 2:
        T[:] = 60.0
    elif kind == 3:
        T[rng.integers(n):] = T[0]
    return T


def _ports(rng, n):
    """Balanced random port flows mapped to nodes: S, F (n+1), T_src."""
    S = np.zeros(n)
    T_src = np.zeros(n)
    for _ in range(rng.integers(0, 3)):
        m = rng.uniform(0.5, 30.0) * (rng.random() < 0.9)
        k_in, k_out = rng.integers(n, size=2)
        T_in = rng.uniform(40.0, 95.0)
        T_src[k_in] = (T_src[k_in] * max(S[k_in], 0.0) + m * T_in) / (max(S[k_in], 0.0) + m) \
            if m > 0.0 else T_src[k_in]
        S[k_in] += m
        S[k_out] -= m
    F = np.zeros(n + 1)
    F[1:] = np.cumsum(S)
    return S, F, T_src


@pytest.fixture
def rng():
    return np.random.default_rng(RNG_SEED)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 5, 8, 20, 50])
def test_solve_tdma_bitwise(rng, n):
    for _ in range(300):
        a, c = rng.normal(size=(2, n))
        d = np.abs(a) + np.abs(c) + rng.uniform(0.1, 2.0, n)
        b = rng.normal(0.0, 100.0, n)
        x = solve_tdma(a.tolist(), d.tolist(), c.tolist(), b.tolist())
        assert np.array(x).tobytes() == ref_solve_tdma(a, d, c, b).tobytes()


@pytest.mark.parametrize("n", [3, 5, 20, 50])
@pytest.mark.parametrize("uniform_area", [True, False])
def test_tvd_bitwise(rng, n, uniform_area):
    rho, dz, cp = 977.8, 0.4, 4187.0
    A_iface = np.full(n - 1, 12.5) if uniform_area else rng.uniform(5.0, 40.0, n - 1)
    denominator = [rho * a * dz for a in A_iface.tolist()]
    for _ in range(300):
        T = _profile(rng, n)
        _, F, _ = _ports(rng, n)
        F_int = F[1:-1]
        dt = float(rng.choice([60.0, 900.0, 3600.0]))
        ref = ref_tvd(T, F_int, cp, dt, rho, A_iface, dz)
        new = tvd_correction(T.tolist(), F_int.tolist(), cp, dt, denominator)
        assert np.array(new).tobytes() == ref.tobytes()


@pytest.mark.parametrize("n", [2, 3, 5, 20, 50])
@pytest.mark.parametrize("with_tvd", [True, False])
def test_implicit_solve_bitwise(rng, n, with_tvd):
    rho, dz, cp, dt = 977.8, 0.4, 4187.0, 3600.0
    A_iface = rng.uniform(5.0, 40.0, n - 1)
    denominator = [rho * a * dz for a in A_iface.tolist()]
    for trial in range(200):
        T = _profile(rng, n)
        S, F, T_src = _ports(rng, n)
        C_nodes = rho * rng.uniform(2.0, 20.0, n) * 4180.0
        K = 3.3 * A_iface / dz
        Q_loss = rng.normal(-50.0, 20.0, n)
        Q_hx = rng.normal(0.0, 500.0, n) if trial % 3 == 0 else np.zeros(n)
        use_tvd = with_tvd and n >= 3
        Q_tvd = ref_tvd(T, F[1:-1], cp, dt, rho, A_iface, dz) if use_tvd else None
        ref = ref_implicit(T, dt, F, S, T_src, C_nodes, K, cp, Q_loss, Q_hx, Q_tvd)

        port_nodes = [int(k) for k in np.flatnonzero(S)]
        T_src_nodes = {k: float(T_src[k]) for k in port_nodes if S[k] > 0.0}
        flux = (tvd_fluxes(T.tolist(), F[1:-1].tolist(), cp, dt, denominator)
                if use_tvd else None)
        new = implicit_solve(
            T.tolist(), dt, F[1:-1].tolist(), S.tolist(), T_src_nodes, port_nodes,
            C_nodes.tolist(), K.tolist(), cp, Q_loss.tolist(),
            Q_hx.tolist() if trial % 3 == 0 else None, flux,
        )
        assert np.array(new).tobytes() == ref.tobytes()


@pytest.mark.parametrize("n", [1, 2, 3, 5, 20, 50])
def test_convective_adjustment_bitwise(rng, n):
    for _ in range(500):
        T = _profile(rng, n) if n > 1 else rng.uniform(20.0, 90.0, 1)
        if rng.random() < 0.3:
            T = rng.uniform(20.0, 90.0, n)
        m = rng.uniform(500.0, 5000.0, n)
        new = convective_adjustment(T.tolist(), m.tolist())
        assert np.array(new).tobytes() == ref_convective_adjustment(T, m).tobytes()
