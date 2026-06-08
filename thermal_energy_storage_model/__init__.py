"""Public API for the thermal_energy_storage_model package."""

# Single source of truth for the package version (pyproject.toml reads this).
__version__ = "1.0.0"

from .config import StorageConfig
from .diffusors import DiffusorModel, PointDiffusor, UniformDiffusor
from .fluids import ConstantFluidProperties, FluidProperties, WaterProperties
from .geometry import (
    CylinderGeometry,
    GeometryModel,
    TruncatedConeGeometry,
    TruncatedPyramidGeometry,
)
from .losses import (
    ConstantAmbientLoss,
    GroundTemperatureLoss,
    LossModel,
    SplitAmbientLoss,
    TransientGroundLoss,
)
from .ports import HeatExchangerPort, Port
from .presets import StoragePresets
from .state import StorageInputs, StorageOutputs, StorageState
from .model import ThermalStorage1D

__all__ = [
    "__version__",
    "GeometryModel",
    "CylinderGeometry",
    "TruncatedConeGeometry",
    "TruncatedPyramidGeometry",
    "FluidProperties",
    "ConstantFluidProperties",
    "WaterProperties",
    "LossModel",
    "ConstantAmbientLoss",
    "SplitAmbientLoss",
    "GroundTemperatureLoss",
    "TransientGroundLoss",
    "Port",
    "HeatExchangerPort",
    "DiffusorModel",
    "PointDiffusor",
    "UniformDiffusor",
    "StorageConfig",
    "StorageState",
    "StorageInputs",
    "StorageOutputs",
    "StoragePresets",
    "ThermalStorage1D",
]
