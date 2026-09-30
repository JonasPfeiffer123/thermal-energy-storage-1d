"""
Per-step numerical kernels on Python floats.

The storage grids used in co-simulation are small (typically 5-50 nodes) and
``ThermalStorage1D.step()`` is called once per coupling interval, so the
per-step cost is dominated by call overhead rather than arithmetic: a NumPy
ufunc call on such short arrays costs ~0.4 µs regardless of length, and
indexing an array element-wise creates a NumPy scalar per access. The
kernels below therefore run their loops on plain Python floats (lists).

Every kernel performs exactly the IEEE-754 double operations of the NumPy
formulation it replaces, in the same order (no reassociation, no fused
multiply-add, no compensated sums), so results are bit-identical; the
golden-master tests (``tests/test_golden_master.py``) check this.
"""

from __future__ import annotations

from itertools import pairwise

# Numerical epsilon to avoid division by zero in the TVD gradient ratio
# (only active for |ΔT| < 1e-12 K, i.e. practically never)
TVD_EPS: float = 1e-12


def tvd_correction(
    T: list[float],
    F_int: list[float],
    cp: float,
    dt: float,
    denominator: list[float],
) -> list[float]:
    """
    TVD anti-diffusion correction (van Leer limiter) per node [W].

    See ``ThermalStorage1D._compute_tvd_correction_ports`` for the scheme.

    Parameters
    ----------
    T : list[float]
        Node temperatures [°C], length n.
    F_int : list[float]
        Mass flow at the n-1 internal interfaces [kg/s], positive downward.
    cp : float
        Specific heat capacity [J/(kg·K)].
    dt : float
        Timestep size [s].
    denominator : list[float]
        ``rho · A_iface[k] · dz`` per interface [kg] (Courant number
        ``cfl_k = |F_k| · dt / denominator[k]``).

    Returns
    -------
    list[float]
        Heat-flow correction per node [W], length n (conservative: each
        interface moves heat between its two adjacent nodes).
    """
    n = len(T)
    Q = [0.0] * n
    dT = [t_next - t for t, t_next in pairwise(T)]   # T[k+1] - T[k]
    # Upwind gradients: dT_ext[k] = dT[k-1], dT_ext[k+2] = dT[k+1],
    # zero beyond the tank boundaries.
    dT_ext = [0.0, *dT, 0.0]
    eps = TVD_EPS

    # corr = 0.5·|F_k|·cp·(1 − cfl_k) · φ · ΔT_flow is evaluated left to right;
    # its flow-dependent prefix is reused while F_k and the denominator
    # repeat (uniform flow through equal cross-sections, the common case).
    F_prev: float | None = None
    den_prev: float | None = None
    skip = downward = True
    prefix = 0.0
    for k, F_k in enumerate(F_int):
        den = denominator[k]
        if F_k != F_prev or den != den_prev:
            F_prev, den_prev = F_k, den
            abs_F = abs(F_k)
            skip = abs_F < 1e-30
            if skip:
                continue
            cfl_k = min(abs_F * dt / den, 1.0)
            downward = F_k >= 0.0
            half_F = 0.5 * F_k if downward else 0.5 * (-F_k)
            prefix = half_F * cp * (1.0 - cfl_k)
        elif skip:
            continue

        dT_k = dT[k]
        if downward:
            # Downward flow: upwind is node k (top)
            r = dT_ext[k] / (dT_k + (eps if dT_k >= 0 else -eps))
            abs_r = abs(r)
            phi = (r + abs_r) / (1.0 + abs_r)            # van Leer limiter
            corr = prefix * phi * dT_k
            # Conservative distribution: node k loses, node k+1 gains
            Q[k] -= corr
            Q[k + 1] += corr
        else:
            # Upward flow: upwind is node k+1 (bottom)
            dT_flow = -dT_k   # T[k] - T[k+1] = gradient in flow direction
            r = -dT_ext[k + 2] / (dT_flow + (eps if dT_flow >= 0 else -eps))
            abs_r = abs(r)
            phi = (r + abs_r) / (1.0 + abs_r)
            corr = prefix * phi * dT_flow
            # Conservative distribution: node k+1 loses, node k gains
            Q[k] += corr
            Q[k + 1] -= corr
    return Q


def solve_tdma(
    a: list[float],
    d: list[float],
    c: list[float],
    b: list[float],
) -> list[float]:
    """
    Solve a tridiagonal system A·x = b with the Thomas algorithm.

    Same conventions as ``ThermalStorage1D._solve_tdma``: equation i reads
    ``a[i]·x[i-1] + d[i]·x[i] + c[i]·x[i+1] = b[i]``; ``a[0]`` and ``c[-1]``
    are not used. No pivoting (the storage matrices are diagonally
    dominant).
    """
    # Forward elimination
    d_prev = d[0]
    b_prev = b[0]
    d_mod = [d_prev]
    b_mod = [b_prev]
    for a_i, d_i, c_prev, b_i in zip(a[1:], d[1:], c, b[1:]):
        w = a_i / d_prev
        d_prev = d_i - w * c_prev
        b_prev = b_i - w * b_prev
        d_mod.append(d_prev)
        b_mod.append(b_prev)

    # Back substitution (built bottom-up, reversed at the end)
    x_next = b_prev / d_prev
    x = [x_next]
    for i in range(len(d) - 2, -1, -1):
        x_next = (b_mod[i] - c[i] * x_next) / d_mod[i]
        x.append(x_next)
    x.reverse()
    return x
