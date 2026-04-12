"""Fluid property models for the 1D thermal storage solver."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Fluid properties
# ---------------------------------------------------------------------------

class FluidProperties:
    """
    Abstract base class for temperature-dependent fluid properties.

    Defines the interface for all fluids. All methods accept scalar
    temperatures [°C] or numpy arrays and return corresponding
    scalar or array values.

    Sign convention / units
    -----------------------
    - T in °C
    - rho   in kg/m³
    - cp    in J/(kg·K)
    - lambda_fluid in W/(m·K)
    """

    def rho(self, T: float | np.ndarray) -> float | np.ndarray:
        """Fluid density at temperature T [kg/m³]."""
        raise NotImplementedError

    def cp(self, T: float | np.ndarray) -> float | np.ndarray:
        """Specific heat capacity at temperature T [J/(kg·K)]."""
        raise NotImplementedError

    def lambda_fluid(self, T: float | np.ndarray) -> float | np.ndarray:
        """Fluid thermal conductivity at temperature T [W/(m·K)]."""
        raise NotImplementedError


class ConstantFluidProperties(FluidProperties):
    """
    Fluid properties as temperature-independent constants.

    Matches the previous model behavior (rho, cp, lambda_fluid
    as fixed values in StorageConfig). Used for backward compatibility
    and cases where temperature dependence is negligible.

    Parameters
    ----------
    rho : float
        Fluid density [kg/m³].
    cp : float
        Specific heat capacity [J/(kg·K)].
    lambda_fluid : float
        Thermal conductivity [W/(m·K)].

    Examples
    --------
        >>> fluid = ConstantFluidProperties(rho=977.8, cp=4187.0, lambda_fluid=0.663)
    """

    def __init__(
        self, rho: float, cp: float, lambda_fluid: float
    ) -> None:
        self._rho = float(rho)
        self._cp = float(cp)
        self._lambda = float(lambda_fluid)

    def rho(self, _: float | np.ndarray) -> float | np.ndarray:
        return self._rho

    def cp(self, _: float | np.ndarray) -> float | np.ndarray:
        return self._cp

    def lambda_fluid(self, _: float | np.ndarray) -> float | np.ndarray:
        return self._lambda

    def __repr__(self) -> str:
        return (
            f"ConstantFluidProperties("
            f"rho={self._rho} kg/m³, cp={self._cp} J/(kg·K), "
            f"lambda_fluid={self._lambda} W/(m·K))"
        )


class WaterProperties(FluidProperties):
    """
    Temperature-dependent properties for liquid water (0-130 °C).

    Polynomial fits identical to FreeTTES (FreeTTES_model.py, lines 2617-2625).
    Valid range: 0-130 °C at atmospheric pressure.

    Fits were derived from tabulated references (IAPWS/VDI Heat Atlas)
    for the temperature range relevant to thermal storage.

    Coefficient source
    ------------------
    FreeTTES (_sw_rho, _sw_cp, _sw_lambda), adopted for consistency
    with the benchmark reference model.

    Examples
    --------
        >>> fluid = WaterProperties()
        >>> fluid.rho(70.0)    # → ~977.8 kg/m³
        >>> fluid.cp(70.0)     # → ~4186.9 J/(kg·K)
        >>> import numpy as np
        >>> T = np.array([20.0, 60.0, 90.0])
        >>> fluid.rho(T)       # → array(...)
    """

    def rho(self, T: float | np.ndarray) -> float | np.ndarray:
        """ρ(T) = -2.525726e-3·T² - 2.123038e-1·T + 1.005011e3  [kg/m³]"""
        return -2.525726e-3 * T**2 - 2.123038e-1 * T + 1.005011e3

    def cp(self, T: float | np.ndarray) -> float | np.ndarray:
        """cp(T) = 9.776500e-3·T² - 7.677243e-1·T + 4.194836e3  [J/(kg·K)]"""
        return 9.776500e-3 * T**2 - 7.677243e-1 * T + 4.194836e3

    def lambda_fluid(self, T: float | np.ndarray) -> float | np.ndarray:
        """λ(T) = 3.097195e-8·T³ - 1.565775e-5·T² + 2.517120e-3·T + 5.531103e-1  [W/(m·K)]"""
        return (
            3.097195e-8 * T**3
            - 1.565775e-5 * T**2
            + 2.517120e-3 * T
            + 5.531103e-1
        )

    def __repr__(self) -> str:
        return "WaterProperties() [FreeTTES polynomials, 0-130 °C]"
