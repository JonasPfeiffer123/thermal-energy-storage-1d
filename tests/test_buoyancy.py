"""Convective adjustment (buoyancy) correctness.

The convective adjustment removes unstable inversions (cold above warm) by
caloric mixing. It must (a) conserve energy (mass-weighted mean) and
(b) produce a gravitationally stable profile (non-increasing top to bottom).
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_energy_storage_model import ThermalStorage1D

adjust = ThermalStorage1D._convective_adjustment


def test_adjustment_conserves_energy_equal_masses():
    T = np.array([20.0, 60.0, 40.0, 80.0, 50.0])
    m = np.full(5, 2.0)
    before = float(np.sum(m * T))
    result = adjust(T.copy(), m)
    assert abs(float(np.sum(m * result)) - before) < 1e-9


def test_adjustment_conserves_energy_unequal_masses():
    T = np.array([20.0, 60.0, 40.0, 80.0, 50.0, 30.0])
    m = np.array([1.0, 3.0, 2.0, 5.0, 1.5, 2.5])
    before = float(np.sum(m * T))
    result = adjust(T.copy(), m)
    assert abs(float(np.sum(m * result)) - before) < 1e-9


def test_adjustment_produces_stable_profile():
    """After adjustment, no node may be cooler than the node below it."""
    T = np.array([20.0, 60.0, 40.0, 80.0, 50.0])
    m = np.full(5, 2.0)
    result = adjust(T.copy(), m)
    assert np.all(np.diff(result) <= 1e-9)  # T[i] >= T[i+1] (top hot -> bottom cold)


def test_stable_profile_unchanged():
    """An already stable profile must pass through untouched."""
    T = np.array([80.0, 70.0, 60.0, 50.0, 40.0])
    m = np.full(5, 2.0)
    result = adjust(T.copy(), m)
    assert np.allclose(result, T)


def test_fully_inverted_profile_mixes_to_mean():
    """A strictly inverted profile (equal masses) collapses to the mean."""
    T = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # cold on top, warm below
    m = np.full(5, 2.0)
    result = adjust(T.copy(), m)
    assert np.allclose(result, np.full(5, 30.0))
