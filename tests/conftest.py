"""Shared pytest fixtures for the thermal storage test suite."""

from __future__ import annotations

import pytest

from thermal_energy_storage_model import StorageConfig, ThermalStorage1D
from tests._helpers import const_fluid


@pytest.fixture
def make_storage():
    """
    Factory fixture that builds a ``ThermalStorage1D`` with constant fluid.

    Any ``StorageConfig`` keyword can be overridden; ``fluid`` defaults to the
    constant-property water model so derived analytical quantities are exact.
    """

    def _make(**overrides) -> ThermalStorage1D:
        params = dict(
            volume=100.0,
            height=5.0,
            n_nodes=20,
            fluid=const_fluid(),
        )
        params.update(overrides)
        return ThermalStorage1D(StorageConfig(**params))

    return _make
