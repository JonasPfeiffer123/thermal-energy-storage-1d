"""Repository-level pytest configuration: namespace of the docstring examples."""

from __future__ import annotations

import numpy as np
import pytest

import thermal_energy_storage_model as tes


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace):
    """Docstring examples use ``np`` and the public API without importing them."""
    doctest_namespace["np"] = np
    for name in tes.__all__:
        doctest_namespace[name] = getattr(tes, name)
