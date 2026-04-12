"""Factory presets for common thermal storage configurations."""

from __future__ import annotations

from .config import StorageConfig
from .geometry import CylinderGeometry, TruncatedConeGeometry
from .losses import ConstantAmbientLoss, GroundTemperatureLoss

# ---------------------------------------------------------------------------
# Storage presets
# ---------------------------------------------------------------------------

class StoragePresets:
    """
    Factory class for predefined storage types.

    Each method returns a ready-to-use `StorageConfig` that can be
    passed directly to `ThermalStorage1D` or used via
    `ThermalStorage1D.from_preset()`.

    Available presets
    -----------------
    - :meth:`steel_tank_aboveground` - above-ground steel tank
    - :meth:`steel_tank_buried` - buried steel tank
    - :meth:`ptes` - pit thermal energy storage (PTES)

    Examples
    --------
        >>> config = StoragePresets.steel_tank_aboveground(volume=500.0, height=15.0)
        >>> tes = ThermalStorage1D(config)

        >>> tes = ThermalStorage1D.from_preset("ptes", r_bottom=40.0, r_top=55.0, height=15.0)
    """

    @staticmethod
    def steel_tank_aboveground(
        volume: float,
        height: float,
        *,
        n_nodes: int = 20,
        U_loss: float = 0.3,
        T_ambient: float = 10.0,
        lambda_eff_factor: float = 5.0,
    ) -> StorageConfig:
        """
        Above-ground steel tank.

        Cylindrical geometry with constant U-value to ambient air.
        Typical for buffer storages and district-heating network storages.

        Parameters
        ----------
        volume : float
            Storage volume [m³].
        height : float
            Storage height [m].
        n_nodes : int, optional
            Number of discretization nodes. Default: 20.
        U_loss : float, optional
            Heat transfer coefficient [W/(m²·K)]. Default: 0.3.
        T_ambient : float, optional
            Ambient temperature [°C]. Default: 10.0.
        lambda_eff_factor : float, optional
            Multiplicative factor for effective thermal conductivity. Default: 5.0.

        Returns
        -------
        StorageConfig
            Configuration for an above-ground steel tank.

        Examples
        --------
            >>> config = StoragePresets.steel_tank_aboveground(
            ...     volume=500.0, height=15.0, U_loss=0.2,
            ... )
        """
        return StorageConfig(
            volume=volume,
            height=height,
            n_nodes=n_nodes,
            lambda_eff_factor=lambda_eff_factor,
            U_loss=U_loss,
            T_ambient=T_ambient,
            geometry=CylinderGeometry.from_volume(volume, height),
            loss_model=ConstantAmbientLoss(U_loss=U_loss, T_ambient=T_ambient),
        )

    @staticmethod
    def steel_tank_buried(
        volume: float,
        height: float,
        *,
        n_nodes: int = 20,
        U_loss: float = 0.3,
        burial_depth: float = 0.0,
        T_surface: float = 8.0,
        T_deep: float = 11.0,
        depth_decay: float = 2.0,
        lambda_eff_factor: float = 5.0,
    ) -> StorageConfig:
        """
        Buried steel tank (soil-covered).

        Cylindrical geometry with depth-dependent ground temperature.
        Suitable for underground long-term storages or soil-covered tanks.

        Parameters
        ----------
        volume : float
            Storage volume [m³].
        height : float
            Storage height [m].
        n_nodes : int, optional
            Number of discretization nodes. Default: 20.
        U_loss : float, optional
            Effective heat transfer coefficient through wall + soil [W/(m²·K)].
            Default: 0.3.
        burial_depth : float, optional
            Depth of tank top below ground surface [m]. Default: 0.0
            (tank roof at surface).
        T_surface : float, optional
            Annual mean surface temperature [°C]. Default: 8.0.
        T_deep : float, optional
            Asymptotic deep-ground temperature [°C]. Default: 11.0.
        depth_decay : float, optional
            Characteristic decay depth for the temperature profile [m].
            Default: 2.0.
        lambda_eff_factor : float, optional
            Multiplicative factor for effective thermal conductivity. Default: 5.0.

        Returns
        -------
        StorageConfig
            Configuration for a buried steel tank.

        Examples
        --------
            >>> config = StoragePresets.steel_tank_buried(
            ...     volume=1000.0, height=12.0,
            ...     burial_depth=1.5, T_surface=9.0,
            ... )
        """
        return StorageConfig(
            volume=volume,
            height=height,
            n_nodes=n_nodes,
            lambda_eff_factor=lambda_eff_factor,
            U_loss=U_loss,
            T_ambient=T_deep,  # representative fallback
            geometry=CylinderGeometry.from_volume(volume, height),
            loss_model=GroundTemperatureLoss(
                U_loss=U_loss,
                burial_depth=burial_depth,
                T_surface=T_surface,
                T_deep=T_deep,
                depth_decay=depth_decay,
            ),
        )

    @staticmethod
    def ptes(
        r_bottom: float,
        r_top: float,
        height: float,
        *,
        n_nodes: int = 20,
        U_loss: float = 0.25,
        burial_depth: float = 0.5,
        T_surface: float = 8.0,
        T_deep: float = 11.0,
        depth_decay: float = 2.0,
        lambda_eff_factor: float = 5.0,
    ) -> StorageConfig:
        """
        Pit thermal energy storage (PTES).

        Truncated-cone geometry (trapezoidal cross-section) with
        depth-dependent ground temperature. Typical for seasonal storages
        embedded in soil with waterproof lining and insulation.

        Parameters
        ----------
        r_bottom : float
            Inner radius at pit bottom [m].
        r_top : float
            Inner radius at pit top edge [m].
            Typically larger than r_bottom (sloped walls).
        height : float
            Pit depth [m].
        n_nodes : int, optional
            Number of discretization nodes. Default: 20.
        U_loss : float, optional
            Effective U-value through liner and insulation [W/(m²·K)].
            Default: 0.25.
        burial_depth : float, optional
            Depth of pit top edge below ground surface [m].
            Default: 0.5 m.
        T_surface : float, optional
            Annual mean surface temperature [°C]. Default: 8.0.
        T_deep : float, optional
            Asymptotic deep-ground temperature [°C]. Default: 11.0.
        depth_decay : float, optional
            Characteristic decay depth for temperature profile [m].
            Default: 2.0.
        lambda_eff_factor : float, optional
            Multiplicative factor for effective thermal conductivity. Default: 5.0.

        Returns
        -------
        StorageConfig
            Configuration for a PTES storage.

        Examples
        --------
        Typical district-heating PTES setup:

            >>> config = StoragePresets.ptes(
            ...     r_bottom=40.0, r_top=55.0, height=15.0,
            ...     burial_depth=0.5, T_surface=8.0,
            ... )
            >>> print(f"Volume: {config.geometry.volume:.0f} m³")
        """
        geom = TruncatedConeGeometry(
            r_bottom=r_bottom, r_top=r_top, height=height
        )
        return StorageConfig(
            volume=geom.volume,
            height=height,
            n_nodes=n_nodes,
            lambda_eff_factor=lambda_eff_factor,
            U_loss=U_loss,
            T_ambient=T_deep,  # representative fallback
            geometry=geom,
            loss_model=GroundTemperatureLoss(
                U_loss=U_loss,
                burial_depth=burial_depth,
                T_surface=T_surface,
                T_deep=T_deep,
                depth_decay=depth_decay,
            ),
        )