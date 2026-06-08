"""Correctness of the Thomas algorithm (TDMA) tridiagonal solver."""

from __future__ import annotations

import numpy as np
import pytest

from thermal_energy_storage_model import ThermalStorage1D


def _dense_from_diagonals(a, d, c):
    """Assemble a dense matrix from lower (a), main (d) and upper (c) diagonals."""
    n = len(d)
    M = np.diag(d).astype(float)
    for i in range(1, n):
        M[i, i - 1] = a[i]
    for i in range(n - 1):
        M[i, i + 1] = c[i]
    return M


@pytest.mark.parametrize("n", [2, 5, 20, 50])
def test_tdma_matches_dense_solve(n):
    """TDMA result must equal a dense LU solve for diagonally dominant systems."""
    rng = np.random.default_rng(seed=n)
    a = np.zeros(n)
    c = np.zeros(n)
    a[1:] = rng.uniform(-1.0, 1.0, size=n - 1)
    c[:-1] = rng.uniform(-1.0, 1.0, size=n - 1)
    # Force strict diagonal dominance -> well-conditioned, unique solution.
    d = np.abs(a) + np.abs(c) + rng.uniform(1.0, 3.0, size=n)
    b = rng.uniform(-10.0, 10.0, size=n)

    x_tdma = ThermalStorage1D._solve_tdma(a, d, c, b)
    x_dense = np.linalg.solve(_dense_from_diagonals(a, d, c), b)
    assert np.allclose(x_tdma, x_dense, rtol=1e-10, atol=1e-12)


def test_tdma_identity_system():
    """The identity matrix returns the right-hand side unchanged."""
    n = 8
    a = np.zeros(n)
    c = np.zeros(n)
    d = np.ones(n)
    b = np.arange(n, dtype=float)
    x = ThermalStorage1D._solve_tdma(a, d, c, b)
    assert np.allclose(x, b)


def test_tdma_does_not_mutate_inputs():
    """The solver must not modify the caller's diagonal/RHS arrays in place."""
    n = 6
    a = np.concatenate([[0.0], np.full(n - 1, -1.0)])
    c = np.concatenate([np.full(n - 1, -1.0), [0.0]])
    d = np.full(n, 4.0)
    b = np.ones(n)
    d_ref, b_ref = d.copy(), b.copy()
    ThermalStorage1D._solve_tdma(a, d, c, b)
    assert np.array_equal(d, d_ref)
    assert np.array_equal(b, b_ref)
