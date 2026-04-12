"""Diffusor distribution models for mapping port flow to storage nodes."""

from __future__ import annotations

import numpy as np

from .ports import Port

# ---------------------------------------------------------------------------
# Diffusor models
# ---------------------------------------------------------------------------

class DiffusorModel:
    """
    Abstract base class for diffusor mixing models.

    A diffusor model describes how a hydraulic port distributes its mass flow
    spatially across storage grid nodes. It abstracts diffusor geometry
    (single-node vs. distributed zone) from the physical core solver.

    Implementation
    --------------
    Subclasses must override :meth:`node_weights`. The method returns, for
    each port, a list of ``(node_index, weight)`` pairs whose weights sum to 1.0.

    Integration
    -----------
    The diffusor model is used by :class:`ThermalStorage1D` in three places:

    - ``_compute_inter_node_fluxes()`` - distributes net mass flow
    - ``_compute_source_terms()`` - distributes source temperature
    - ``port_temperatures`` - mass-flow-weighted outlet temperature

    This makes the model solver-agnostic: it works equally with explicit
    and implicit solvers.

    Note
    ----
    For energy balance, mass balance must remain satisfied. The sum of
    distributed mass flows across all nodes must match the port's
    original mass flow.
    """

    def node_weights(
        self,
        port: "Port",
        node_heights: np.ndarray,
    ) -> list[tuple[int, float]]:
        """
        Return the weighted node distribution for one port.

        Parameters
        ----------
        port : Port
            Port with height coordinate ``z`` [m] above tank bottom.
        node_heights : numpy.ndarray
            Height coordinates of all node centers above tank bottom [m],
            shape (n,). Index 0 = top (hot), index N-1 = bottom (cold).

        Returns
        -------
        list[tuple[int, float]]
            List of ``(node_index, weight)`` pairs.
            Weights must sum to 1.0.
        """
        raise NotImplementedError  # noqa: ARG002


class PointDiffusor(DiffusorModel):
    """
    Point diffusor: full mass flow assigned to the nearest node.

    This is the default model and matches previous behavior.
    No overhead compared to the former ``_port_to_node`` approach.

    Suitable for
    ------------
    - Grid resolutions where $\Delta z \approx H_{diffusor}$ (coarse grids, N <= 20)
    - Reference runs and benchmarks
    """

    def node_weights(
        self,
        port: "Port",
        node_heights: np.ndarray,
    ) -> list[tuple[int, float]]:
        k = int(np.argmin(np.abs(node_heights - port.z)))
        return [(k, 1.0)]


class UniformDiffusor(DiffusorModel):
    """
    Uniform diffusor: even distribution over a zone width.

    Port mass flow and enthalpy are uniformly distributed over all nodes
    whose node center lies within the diffusor zone:

        |z_knoten - z_port| ≤ H_zone / 2

    This approximates homogeneous inflow over the diffusor height and
    creates a flatter temperature gradient in the diffusor zone, which
    is closer to FreeTTES Lagrange behavior in this region.

    Parameters
    ----------
    H_zone : float
        Height of the mixing zone [m]. Typical value: ``H_RS_Dif`` from
        FreeTTES configuration (benchmark default: 1.0 m).

    Note
    ----
    If no node lies inside the zone (very coarse grid), the model
    automatically falls back to the nearest node.
    """

    def __init__(self, H_zone: float) -> None:
        if H_zone <= 0.0:
            raise ValueError(f"H_zone must be positive, but is {H_zone}.")
        self.H_zone = float(H_zone)

    def node_weights(
        self,
        port: "Port",
        node_heights: np.ndarray,
    ) -> list[tuple[int, float]]:
        dists = np.abs(node_heights - port.z)
        in_zone = dists <= self.H_zone / 2.0
        if not np.any(in_zone):
            # Fallback: nearest node
            return [(int(np.argmin(dists)), 1.0)]
        indices = np.where(in_zone)[0]
        w = 1.0 / len(indices)
        return [(int(i), w) for i in indices]
