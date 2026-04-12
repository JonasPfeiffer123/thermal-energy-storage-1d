"""Geometry models used by the 1D thermal storage solver."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Geometry models
# ---------------------------------------------------------------------------

class GeometryModel:
    """
    Abstract base class for storage geometries.

    Defines the interface all concrete geometry classes must implement.
    Each subclass describes the spatial storage shape and provides
    corresponding derived quantities.

    Nodes are distributed uniformly over height (uniform grid).
    Index 0 = top (hot side), index N-1 = bottom (cold side).
    """

    @property
    def volume(self) -> float:
        """Total storage volume [m³]."""
        raise NotImplementedError

    @property
    def height(self) -> float:
        """Total storage height [m]."""
        raise NotImplementedError

    def A_cross(self, z: float) -> float:
        """
        Cross-sectional area at height z above bottom [m²].

        Parameters
        ----------
        z : float
            Height above bottom [m]. 0 = bottom, height = lid.
        """
        raise NotImplementedError

    def A_cross_nodes(self, n_nodes: int) -> np.ndarray:
        """
        Cross-sectional area at each node center [m²].

        Nodes are uniformly distributed. Index 0 = top, N-1 = bottom.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.

        Returns
        -------
        numpy.ndarray
            Cross-sectional areas, shape (n_nodes,).
        """
        raise NotImplementedError

    def V_nodes(self, n_nodes: int) -> np.ndarray:
        """
        Volume for each node [m³].

        Parameters
        ----------
        n_nodes : int
            Number of nodes.

        Returns
        -------
        numpy.ndarray
            Node volumes, shape (n_nodes,).
        """
        raise NotImplementedError

    def A_wall_nodes(self, n_nodes: int) -> np.ndarray:
        """
        Effective wall area for each node [m²].

        Includes lateral area of each segment plus the top lid area
        for the top node and bottom base area for the bottom node.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.

        Returns
        -------
        numpy.ndarray
            Wall areas for loss calculation, shape (n_nodes,).
        """
        raise NotImplementedError


class CylinderGeometry(GeometryModel):
    """
    Cylindrical storage with a constant circular cross-section.

    Typical for above-ground or buried steel tanks.

    Parameters
    ----------
    radius : float
        Inner cylinder radius [m].
    height : float
        Total storage height [m].

    Examples
    --------
    From radius and height:

        >>> geom = CylinderGeometry(radius=5.0, height=15.0)

    From volume and height:

        >>> geom = CylinderGeometry.from_volume(volume=500.0, height=15.0)
    """

    def __init__(self, radius: float, height: float) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be positive, but is {radius}.")
        if height <= 0:
            raise ValueError(f"height must be positive, but is {height}.")
        self._radius = radius
        self._height = height
        self._A_cross_val: float = np.pi * radius ** 2
        self._volume: float = self._A_cross_val * height

    @classmethod
    def from_volume(cls, volume: float, height: float) -> "CylinderGeometry":
        """
        Create a cylinder from volume and height.

        Parameters
        ----------
        volume : float
            Total volume [m³].
        height : float
            Total height [m].

        Returns
        -------
        CylinderGeometry
        """
        if volume <= 0:
            raise ValueError(f"volume must be positive, but is {volume}.")
        if height <= 0:
            raise ValueError(f"height must be positive, but is {height}.")
        radius = np.sqrt(volume / (np.pi * height))
        return cls(radius=radius, height=height)

    @property
    def radius(self) -> float:
        """Inner cylinder radius [m]."""
        return self._radius

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def height(self) -> float:
        return self._height

    def A_cross(self, z: float) -> float:
        return self._A_cross_val

    def A_cross_nodes(self, n_nodes: int) -> np.ndarray:
        return np.full(n_nodes, self._A_cross_val)

    def V_nodes(self, n_nodes: int) -> np.ndarray:
        return np.full(n_nodes, self._volume / n_nodes)

    def A_wall_nodes(self, n_nodes: int) -> np.ndarray:
        dz = self._height / n_nodes
        A_lateral = 2.0 * np.pi * self._radius * dz
        A_cap = self._A_cross_val
        areas = np.full(n_nodes, A_lateral)
        areas[0] += A_cap    # Top lid
        areas[-1] += A_cap   # Bottom base
        return areas


class TruncatedConeGeometry(GeometryModel):
    """
    Truncated-cone geometry for pit thermal energy storage (PTES).

    Models a storage with trapezoidal cross-section profile, typical
    for buried PTES systems: narrower at bottom, wider at top.

    Cross-sectional area varies linearly with height:

        A(z) = π · r(z)²
        r(z) = r_bottom + (r_top - r_bottom) · z / height

    Volume follows the truncated-cone formula:

        V = π · h / 3 · (r_top² + r_top · r_bottom + r_bottom²)

    Parameters
    ----------
    r_bottom : float
        Radius at storage bottom [m].
    r_top : float
        Radius at storage top surface (lid) [m].
    height : float
        Total storage height [m].

    Notes
    -----
    For PTES, typically r_top > r_bottom.
    r_top == r_bottom is equivalent to a cylinder.

    Examples
    --------
    PTES with 45° slope:

        >>> geom = TruncatedConeGeometry(r_bottom=10.0, r_top=20.0, height=10.0)
        >>> print(f"Volume: {geom.volume:.1f} m³")
    """

    def __init__(self, r_bottom: float, r_top: float, height: float) -> None:
        if r_bottom <= 0:
            raise ValueError(f"r_bottom must be positive, but is {r_bottom}.")
        if r_top <= 0:
            raise ValueError(f"r_top must be positive, but is {r_top}.")
        if height <= 0:
            raise ValueError(f"height must be positive, but is {height}.")
        self._r_bottom = r_bottom
        self._r_top = r_top
        self._height = height
        self._volume = (
            np.pi * height / 3.0
            * (r_top ** 2 + r_top * r_bottom + r_bottom ** 2)
        )

    @property
    def r_bottom(self) -> float:
        """Radius at bottom [m]."""
        return self._r_bottom

    @property
    def r_top(self) -> float:
        """Radius at lid [m]."""
        return self._r_top

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def height(self) -> float:
        return self._height

    def _radius_at(self, z: float) -> float:
        """Radius at height z above bottom [m] (linear interpolation)."""
        return self._r_bottom + (self._r_top - self._r_bottom) * z / self._height

    def A_cross(self, z: float) -> float:
        return np.pi * self._radius_at(z) ** 2

    def A_cross_nodes(self, n_nodes: int) -> np.ndarray:
        """Cross-section at node center, index 0 = top."""
        dz = self._height / n_nodes
        # z_center[i]: node-center height above bottom, index 0 = top
        z_centers = np.array([(n_nodes - 0.5 - i) * dz for i in range(n_nodes)])
        radii = (
            self._r_bottom
            + (self._r_top - self._r_bottom) * z_centers / self._height
        )
        return np.pi * radii ** 2

    def V_nodes(self, n_nodes: int) -> np.ndarray:
        """Exact node volume as truncated-cone layer."""
        dz = self._height / n_nodes
        V = np.empty(n_nodes)
        for i in range(n_nodes):
            z_top = (n_nodes - i) * dz       # Upper node boundary from bottom
            z_bot = (n_nodes - i - 1) * dz   # Lower node boundary from bottom
            r_t = self._radius_at(z_top)
            r_b = self._radius_at(z_bot)
            V[i] = (np.pi * dz / 3.0) * (r_t ** 2 + r_t * r_b + r_b ** 2)
        return V

    def A_wall_nodes(self, n_nodes: int) -> np.ndarray:
        """Lateral area of each truncated-cone segment plus lid/bottom."""
        dz = self._height / n_nodes
        areas = np.empty(n_nodes)
        for i in range(n_nodes):
            z_top = (n_nodes - i) * dz
            z_bot = (n_nodes - i - 1) * dz
            r_t = self._radius_at(z_top)
            r_b = self._radius_at(z_bot)
            # Truncated-cone lateral area: pi * (r_t + r_b) * slant
            slant = np.sqrt(dz ** 2 + (r_t - r_b) ** 2)
            areas[i] = np.pi * (r_t + r_b) * slant
        # Top lid, bottom base
        areas[0] += np.pi * self._r_top ** 2
        areas[-1] += np.pi * self._r_bottom ** 2
        return areas


class TruncatedPyramidGeometry(GeometryModel):
    """
    Truncated-pyramid geometry for rectangular pit thermal storages (PTES).

    Models a storage with rectangular footprint and uniformly sloped walls,
    typical for Danish and German PTES systems. In contrast to the truncated
    cone, cross-section changes quadratically with height:

        A(z) = a(z) · b(z)
        a(z) = a_bottom + (a_top − a_bottom) · z / height
        b(z) = b_bottom + (b_top − b_bottom) · z / height

    Volume is obtained by analytical integration:

        V = H/6 · (2·a_t·b_t + 2·a_b·b_b + a_t·b_b + a_b·b_t)

    Segment wall area accounts for wall inclination
    (slant height instead of vertical height).

    Parameters
    ----------
    a_bottom : float
        Bottom rectangle length [m].
    b_bottom : float
        Bottom rectangle width [m].
    a_top : float
        Top rectangle length [m]. Typically a_top > a_bottom.
    b_top : float
        Top rectangle width [m].
    height : float
        Total storage height [m].

    Notes
    -----
    Typical PTES slope: 1:2 (1 m vertical, 2 m horizontal).
    For a 10 m deep storage with 100 m × 100 m bottom area and 1:2 slope:
    a_top = b_top = 100 + 2·2·10 = 140 m.

    Examples
    --------
    From explicit dimensions:

        >>> geom = TruncatedPyramidGeometry(
        ...     a_bottom=100.0, b_bottom=100.0,
        ...     a_top=140.0,    b_top=140.0,
        ...     height=10.0,
        ... )
        >>> print(f"Volume: {geom.volume:.0f} m³")

    From bottom area, height and slope (short form):

        >>> geom = TruncatedPyramidGeometry.from_slope(
        ...     a_bottom=100.0, b_bottom=100.0, height=10.0, slope=2.0
        ... )
    """

    def __init__(
        self,
        a_bottom: float,
        b_bottom: float,
        a_top: float,
        b_top: float,
        height: float,
    ) -> None:
        for name, val in [
            ("a_bottom", a_bottom), ("b_bottom", b_bottom),
            ("a_top",    a_top),    ("b_top",    b_top),
            ("height",   height),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be positive, but is {val}.")
        self._a_bottom = a_bottom
        self._b_bottom = b_bottom
        self._a_top    = a_top
        self._b_top    = b_top
        self._height   = height
        # Analytical volume: H/6·(2·a_t·b_t + 2·a_b·b_b + a_t·b_b + a_b·b_t)
        self._volume = (
            height / 6.0 * (
                2.0 * a_top    * b_top
                + 2.0 * a_bottom * b_bottom
                +       a_top    * b_bottom
                +       a_bottom * b_top
            )
        )
        # Precomputed slope factors (horizontal per vertical, per side)
        self._m_a = (a_top - a_bottom) / (2.0 * height)  # Slope in a-direction
        self._m_b = (b_top - b_bottom) / (2.0 * height)  # Slope in b-direction

    @classmethod
    def from_slope(
        cls,
        a_bottom: float,
        b_bottom: float,
        height: float,
        slope: float = 2.0,
    ) -> "TruncatedPyramidGeometry":
        """
        Create a rectangular PTES from bottom area, height and slope.

        Parameters
        ----------
        a_bottom : float
            Bottom rectangle length [m].
        b_bottom : float
            Bottom rectangle width [m].
        height : float
            Storage depth/height [m].
        slope : float
            Slope as horizontal distance per vertical unit
            (default: 2.0, i.e. 1:2 slope - 2 m horizontal per 1 m vertical).

        Returns
        -------
        TruncatedPyramidGeometry
        """
        a_top = a_bottom + 2.0 * slope * height
        b_top = b_bottom + 2.0 * slope * height
        return cls(a_bottom=a_bottom, b_bottom=b_bottom,
                   a_top=a_top, b_top=b_top, height=height)

    @property
    def a_bottom(self) -> float:
        """Length at bottom [m]."""
        return self._a_bottom

    @property
    def b_bottom(self) -> float:
        """Width at bottom [m]."""
        return self._b_bottom

    @property
    def a_top(self) -> float:
        """Length at lid [m]."""
        return self._a_top

    @property
    def b_top(self) -> float:
        """Width at lid [m]."""
        return self._b_top

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def height(self) -> float:
        return self._height

    def _dims_at(self, z: float) -> tuple[float, float]:
        """Return (a(z), b(z)) at height z above bottom [m]."""
        t = z / self._height
        a = self._a_bottom + (self._a_top - self._a_bottom) * t
        b = self._b_bottom + (self._b_top - self._b_bottom) * t
        return a, b

    def A_cross(self, z: float) -> float:
        a, b = self._dims_at(z)
        return a * b

    def A_cross_nodes(self, n_nodes: int) -> np.ndarray:
        """Cross-section at node center, index 0 = top."""
        dz = self._height / n_nodes
        z_centers = np.array([(n_nodes - 0.5 - i) * dz for i in range(n_nodes)])
        t = z_centers / self._height
        a = self._a_bottom + (self._a_top - self._a_bottom) * t
        b = self._b_bottom + (self._b_top - self._b_bottom) * t
        return a * b

    def V_nodes(self, n_nodes: int) -> np.ndarray:
        """
        Exact node volume from analytical integration of A(z) = a(z)·b(z).

        V = integral_{z_bot}^{z_top} a(z)·b(z) dz
          = a0·b0·Δz + (a0·db + b0·da)·(z_top²−z_bot²)/(2H)
                     + da·db·(z_top³−z_bot³)/(3H²)
        """
        dz = self._height / n_nodes
        H  = self._height
        a0, b0 = self._a_bottom, self._b_bottom
        da = self._a_top - a0
        db = self._b_top - b0

        V = np.empty(n_nodes)
        for i in range(n_nodes):
            z_top = (n_nodes - i)       * dz   # Upper boundary from bottom
            z_bot = (n_nodes - i - 1)   * dz   # Lower boundary from bottom
            V[i] = (
                a0 * b0 * (z_top - z_bot)
                + (a0 * db + b0 * da) * (z_top**2 - z_bot**2) / (2.0 * H)
                + da * db             * (z_top**3 - z_bot**3) / (3.0 * H**2)
            )
        return V

    def A_wall_nodes(self, n_nodes: int) -> np.ndarray:
        """
        Wall area per segment - four inclined trapezoid walls plus lid/bottom.

        The two 'a-walls' (parallel to the longitudinal axis) are inclined in
        b-direction; the two 'b-walls' accordingly in a-direction.

            A_a_walls = (a_bot + a_top) · dz · sqrt(1 + m_b²)
            A_b_walls = (b_bot + b_top) · dz · sqrt(1 + m_a²)
        """
        dz      = self._height / n_nodes
        slant_a = np.sqrt(1.0 + self._m_a**2)   # Slant of b-walls
        slant_b = np.sqrt(1.0 + self._m_b**2)   # Slant of a-walls

        areas = np.empty(n_nodes)
        for i in range(n_nodes):
            z_top = (n_nodes - i)     * dz
            z_bot = (n_nodes - i - 1) * dz
            a_t, b_t = self._dims_at(z_top)
            a_b, b_b = self._dims_at(z_bot)
            # Two a-walls (long sides, slope in b-direction)
            A_a = (a_b + a_t) * dz * slant_b
            # Two b-walls (short sides, slope in a-direction)
            A_b = (b_b + b_t) * dz * slant_a
            areas[i] = A_a + A_b

        # Top lid and bottom base
        areas[0]  += self._a_top    * self._b_top
        areas[-1] += self._a_bottom * self._b_bottom
        return areas