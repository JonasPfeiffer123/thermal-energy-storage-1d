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
from .model import ThermalStorage1D
from .ports import HeatExchangerPort, Port
from .presets import StoragePresets
from .state import StorageInputs, StorageOutputs, StorageState

__all__ = [
    "ConstantAmbientLoss",
    "ConstantFluidProperties",
    "CylinderGeometry",
    "DiffusorModel",
    "FluidProperties",
    "GeometryModel",
    "GroundTemperatureLoss",
    "HeatExchangerPort",
    "LossModel",
    "PointDiffusor",
    "Port",
    "SplitAmbientLoss",
    "StorageConfig",
    "StorageInputs",
    "StorageOutputs",
    "StoragePresets",
    "StorageState",
    "ThermalStorage1D",
    "TransientGroundLoss",
    "TruncatedConeGeometry",
    "TruncatedPyramidGeometry",
    "UniformDiffusor",
    "WaterProperties",
    "__version__",
]
