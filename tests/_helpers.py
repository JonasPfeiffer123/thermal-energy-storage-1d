"""Shared constants and helpers for the test suite (plain importable module)."""

from __future__ import annotations

from thermal_energy_storage_model import ConstantFluidProperties

# Constant fluid properties used throughout the suite. A constant fluid (rather
# than temperature-dependent WaterProperties) keeps the heat capacity
# C = rho * V * cp constant, which makes energy-conservation and analytical
# decay checks exact rather than approximate.
RHO = 977.8
CP = 4187.0
LAMBDA = 0.663


def const_fluid() -> ConstantFluidProperties:
    """Constant-property water fluid model used by the analytical tests."""
    return ConstantFluidProperties(RHO, CP, LAMBDA)
