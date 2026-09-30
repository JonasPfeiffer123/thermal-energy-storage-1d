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
golden-master tests (``tests/test_golden_master.py``) and the kernel tests
(``tests/test_kernels.py``, which keep the former array implementations as
references) check this.
"""

from __future__ import annotations

from itertools import pairwise

# Numerical epsilon to avoid division by zero in the TVD gradient ratio
# (only active for |ΔT| < 1e-12 K, i.e. practically never)
TVD_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# TVD anti-diffusion (van Leer)
# ---------------------------------------------------------------------------

def tvd_fluxes(
    T: list[float],
    F_int: list[float],
    cp: float,
    dt: float,
    denominator: list[float],
) -> list[float]:
    """
    Anti-diffusive TVD heat flux across each internal interface [W].

    See ``ThermalStorage1D._compute_tvd_correction_ports`` for the scheme.
    The flux ``s[k]`` across interface k (between nodes k and k+1) is
    counted positive downward: node k loses ``s[k]``, node k+1 gains it
    (:func:`tvd_node_correction`).

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
        Signed interface fluxes, length n-1 (0.0 where ``|F_k| < 1e-30``).
    """
    dT = [t_next - t for t, t_next in pairwise(T)]   # T[k+1] - T[k]
    # Upwind gradients: dT_ext[k] = dT[k-1], dT_ext[k+2] = dT[k+1],
    # zero beyond the tank boundaries.
    dT_ext = [0.0, *dT, 0.0]
    eps = TVD_EPS
    s: list[float] = []
    append = s.append

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
            if not skip:
                cfl_k = min(abs_F * dt / den, 1.0)
                downward = F_k >= 0.0
                half_F = 0.5 * F_k if downward else 0.5 * (-F_k)
                prefix = half_F * cp * (1.0 - cfl_k)
        if skip:
            append(0.0)
        elif downward:
            # Downward flow: upwind is node k (top); node k loses corr
            dT_k = dT[k]
            r = dT_ext[k] / (dT_k + (eps if dT_k >= 0 else -eps))
            abs_r = abs(r)
            phi = (r + abs_r) / (1.0 + abs_r)            # van Leer limiter
            append(prefix * phi * dT_k)
        else:
            # Upward flow: upwind is node k+1 (bottom); node k+1 loses corr
            dT_flow = -dT[k]   # T[k] - T[k+1] = gradient in flow direction
            r = -dT_ext[k + 2] / (dT_flow + (eps if dT_flow >= 0 else -eps))
            abs_r = abs(r)
            phi = (r + abs_r) / (1.0 + abs_r)
            append(-(prefix * phi * dT_flow))
    return s


def tvd_node_correction(s: list[float]) -> list[float]:
    """
    Per-node TVD heat flow [W] from the interface fluxes of :func:`tvd_fluxes`.

    ``Q[i] = (0 + s[i-1]) - s[i]`` (terms beyond the boundaries omitted):
    the accumulation order of the array version, which started from zeros
    and applied the interfaces from top to bottom. Subtracting ``-corr`` is
    exactly adding ``corr``, so the signed fluxes reproduce both flow
    directions bit for bit.
    """
    return [(0.0 + s_above) - s_below
            for s_above, s_below in zip([0.0, *s], [*s, 0.0])]


def tvd_correction(
    T: list[float],
    F_int: list[float],
    cp: float,
    dt: float,
    denominator: list[float],
) -> list[float]:
    """TVD anti-diffusion correction per node [W] (conservative)."""
    return tvd_node_correction(tvd_fluxes(T, F_int, cp, dt, denominator))


# ---------------------------------------------------------------------------
# Implicit step (tridiagonal system)
# ---------------------------------------------------------------------------

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


def implicit_solve(
    T: list[float],
    dt: float,
    F_int: list[float],
    S: list[float],
    T_src: dict[int, float],
    port_nodes: list[int],
    C_nodes: list[float],
    K: list[float],
    cp: float,
    Q_loss: list[float],
    Q_hx: list[float] | None = None,
    tvd_flux: list[float] | None = None,
) -> list[float]:
    """
    Assemble and solve the implicit-Euler system of one storage step.

    See ``ThermalStorage1D._step_implicit`` for the scheme. Row i of the
    tridiagonal system ``a[i]·x[i-1] + d[i]·x[i] + c[i]·x[i+1] = b[i]``:

    - ``d[i] = (((0 + C[i]/dt) + K[i] + K[i-1]) - up[i-1] + down[i])
      - cp·min(S[i], 0)``
    - ``a[i] = (0 - K[i-1]) - down[i-1]``, ``c[i] = (0 - K[i]) + up[i]``
    - ``b[i] = ((((0 + C[i]/dt·T[i]) + Q_loss[i]) + Q_hx[i]) + Q_tvd[i])
      + cp·max(S[i], 0)·T_src[i]``

    with the interface advection ``F_int[k]·cp`` split into its downward
    (``down``, ≥ 0) and upward (``up``, < 0) part, the other part being 0.0;
    terms beyond the boundaries are omitted. These are the operations, in
    order, of the former array assembly (zero arrays, then time derivative,
    conduction, per-interface advection, port terms); the added/subtracted
    zeros leave the nonzero coefficients unchanged. Rows are assembled
    inside the forward sweep of the Thomas algorithm (:func:`solve_tdma`).

    Parameters
    ----------
    T : list[float]
        Temperatures at the old time level [°C], length n (n ≥ 2).
    dt : float
        Timestep size [s].
    F_int : list[float]
        Mass flow across the n-1 internal interfaces [kg/s], positive downward.
    S : list[float]
        Net port source per node [kg/s], length n.
    T_src : dict[int, float]
        Mass-flow-weighted inlet temperature of every node with inflow [°C].
    port_nodes : list[int]
        Nodes connected to a port (all others have ``S == 0``).
    C_nodes : list[float]
        Node heat capacities [J/K], length n.
    K : list[float]
        Interface conductances [W/K], length n-1.
    cp : float
        Specific heat capacity for advection [J/(kg·K)].
    Q_loss : list[float]
        Wall heat flow per node [W] (explicit).
    Q_hx : list[float], optional
        Heat-exchanger/headspace heat flow per node [W] (explicit).
    tvd_flux : list[float], optional
        TVD interface fluxes of :func:`tvd_fluxes` (explicit, deferred).

    Returns
    -------
    list[float]
        Temperatures at the new time level [°C], length n.
    """
    n = len(T)
    C_dt = [C / dt for C in C_nodes]              # thermal capacity / dt [W/K]

    # --- Right-hand side (complete before the sweep) ---
    b = [(0.0 + c_dt * t) + q for c_dt, t, q in zip(C_dt, T, Q_loss)]
    if Q_hx is not None:
        b = [b_i + q for b_i, q in zip(b, Q_hx)]
    if tvd_flux is not None:
        b = [b_i + ((0.0 + s_above) - s_below)
             for b_i, s_above, s_below in zip(b, [0.0, *tvd_flux], [*tvd_flux, 0.0])]
    # Ports: inflow energy -> RHS; outflow at the node temperature -> implicit
    d_out = [0.0] * n
    for k in port_nodes:
        s_k = S[k]
        if s_k < 0.0:
            d_out[k] = cp * s_k
        elif s_k > 0.0:
            b[k] += cp * s_k * T_src[k]

    # --- Forward sweep; row i is assembled from interface i-1 ("above") ---
    # --- and interface i ("below") ---
    # (Not min/max below: a NaN flow must take the upward branch, as before.)
    f = F_int[0] * cp
    down, up = (f, 0.0) if f >= 0.0 else (0.0, f)
    K_above = K[0]
    d_i = ((((0.0 + C_dt[0]) + K_above) + 0.0) - 0.0 + down) - d_out[0]
    b_i = b[0]
    c_i = (0.0 - K_above) + up
    d_mod = [d_i]
    b_mod = [b_i]
    c_up = [c_i]
    down_above, up_above = down, up
    last = n - 1
    for i in range(1, last):
        f = F_int[i] * cp
        down, up = (f, 0.0) if f >= 0.0 else (0.0, f)
        K_below = K[i]
        w = ((0.0 - K_above) - down_above) / d_i                 # a[i] / d'[i-1]
        d_i = ((((((0.0 + C_dt[i]) + K_below) + K_above) - up_above + down)
                - d_out[i]) - w * c_i)
        b_i = b[i] - w * b_i
        c_i = (0.0 - K_below) + up
        d_mod.append(d_i)
        b_mod.append(b_i)
        c_up.append(c_i)
        K_above, down_above, up_above = K_below, down, up
    # Bottom row (no interface below)
    w = ((0.0 - K_above) - down_above) / d_i
    d_i = ((((((0.0 + C_dt[last]) + 0.0) + K_above) - up_above + 0.0)
            - d_out[last]) - w * c_i)
    b_i = b[last] - w * b_i

    # --- Back substitution (built bottom-up, reversed at the end) ---
    x_next = b_i / d_i
    x = [x_next]
    for i in range(last - 1, -1, -1):
        x_next = (b_mod[i] - c_up[i] * x_next) / d_mod[i]
        x.append(x_next)
    x.reverse()
    return x


# ---------------------------------------------------------------------------
# Buoyancy
# ---------------------------------------------------------------------------

def convective_adjustment(T: list[float], m: list[float]) -> list[float]:
    """
    Mix unstable layers (a colder layer above a warmer one), top to bottom.

    Stack-based O(n) algorithm of ``ThermalStorage1D._convective_adjustment``:
    every node starts as its own mixing zone (mean temperature
    ``(T·m)/m``) and merges with the zone above while that zone is colder.

    Parameters
    ----------
    T : list[float]
        Node temperatures [°C], index 0 = top.
    m : list[float]
        Node masses [kg].

    Returns
    -------
    list[float]
        Mass-weighted zone mean temperature of every node.
    """
    # Fast path: nodes as single zones; done if no zone is colder than the
    # one below it (the stack algorithm would not merge anything).
    single = [(T_i * m_i) / m_i for T_i, m_i in zip(T, m)]
    for j, (upper, lower) in enumerate(pairwise(single), 1):
        if upper < lower:
            break
    else:
        return single

    # Nodes 0..j-1 are single zones on the stack; continue from node j.
    energy = [T_i * m_i for T_i, m_i in zip(T[:j], m[:j])]   # Σ T·m per zone
    mass = m[:j]                                             # Σ m per zone
    start = list(range(j))                                   # first node
    mean = single[:j]                                        # zone mean
    for i in range(j, len(T)):
        m_sum = m[i]
        e = T[i] * m_sum
        first = i
        T_mix = e / m_sum
        # While the zone above is colder: merge
        while mean and mean[-1] < T_mix:
            e += energy.pop()
            m_sum += mass.pop()
            first = start.pop()
            mean.pop()
            T_mix = e / m_sum
        energy.append(e)
        mass.append(m_sum)
        start.append(first)
        mean.append(T_mix)

    result: list[float] = []
    for T_mix, first, end in zip(mean, start, [*start[1:], len(T)]):
        result.extend([T_mix] * (end - first))
    return result
