"""
3D tank visualisation using matplotlib Axes3D.

Renders the storage as a colour-stratified cylinder or truncated cone.
The temperature distribution is shown as a colourmap (RdYlBu_r).
Ports are drawn as coloured arrows.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib.cm as mcm
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

sys.path.insert(0, str(Path(__file__).parent.parent))
from thermal_energy_storage_model import (
    CylinderGeometry,
    StorageConfig,
    TruncatedConeGeometry,
)
try:
    from thermal_energy_storage_model import TruncatedPyramidGeometry as _TruncPyramid
except ImportError:
    _TruncPyramid = None


class Tank3DWidget(QWidget):
    """
    Widget for interactive 3D visualisation of the thermal storage tank.

    Strategy for stable updates without layout drift:
    - Figure layout and colorbar are created only once on the first render.
    - On every subsequent update only the 3D surfaces (collections) and
      port arrows (lines/quiver) are removed and redrawn.
    - The colorbar norm is updated in-place (no repeated fig.colorbar() calls).
    """

    _PORT_COLORS = {
        "charge_in":     "#e74c3c",
        "charge_out":    "#e67e22",
        "discharge_in":  "#3498db",
        "discharge_out": "#1abc9c",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config: Optional[StorageConfig] = None
        self._sm: Optional[ScalarMappable] = None   # for colorbar
        self._norm: Optional[Normalize] = None
        self._cmap_fn = mcm.get_cmap("RdYlBu_r")
        self._colorbar = None
        self._placeholder_active = True
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._fig = Figure(figsize=(5, 6), facecolor="#1e1e2e")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._toolbar = NavToolbar(self._canvas, self)
        self._toolbar.setStyleSheet("background: #2d2d3d; color: white;")

        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

        # Create Axes3D once
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._draw_placeholder()

    # ------------------------------------------------------------------
    # Public update method
    # ------------------------------------------------------------------

    def update_tank(
        self,
        config: StorageConfig,
        temperatures: Optional[np.ndarray] = None,
        T_min: float = 10.0,
        T_max: float = 95.0,
        ports: Optional[list[dict]] = None,
    ):
        """
        Update the 3D visualisation.

        First call: builds layout, colorbar, and axis labels.
        Subsequent calls: removes only surfaces/arrows and redraws them.
        The colorbar is updated in-place, not recreated.
        """
        self._config = config
        ports = ports or []

        if config is None:
            self._clear_artists()
            self._draw_placeholder()
            self._canvas.draw_idle()
            return

        n_nodes = config.n_nodes
        H = config.height

        # Prepare temperatures
        if temperatures is None or len(temperatures) != n_nodes:
            T = np.linspace(T_max, T_min, n_nodes)
        else:
            T = np.asarray(temperatures, dtype=float)

        # Geometry
        import math
        geom = config.geometry
        pyr_dims = None  # (a_bottom, b_bottom, a_top, b_top) for pyramid only
        if isinstance(geom, TruncatedConeGeometry):
            geo_type, r_bottom, r_top = "cone", geom.r_bottom, geom.r_top
        elif _TruncPyramid is not None and isinstance(geom, _TruncPyramid):
            geo_type = "pyramid"
            pyr_dims = (geom.a_bottom, geom.b_bottom, geom.a_top, geom.b_top)
            # r_bottom/r_top for axis scaling: half diagonal
            r_bottom = math.sqrt(geom.a_bottom**2 + geom.b_bottom**2) / 2
            r_top    = math.sqrt(geom.a_top**2    + geom.b_top**2)    / 2
        elif isinstance(geom, CylinderGeometry):
            geo_type = "cylinder"
            r_bottom = r_top = geom.radius
        else:
            geo_type = "cylinder"
            r_bottom = r_top = math.sqrt(config.volume / (math.pi * H))

        # Farbskala
        T_lo = min(T_min, float(T.min()))
        T_hi = max(T_max, float(T.max()))
        if T_hi <= T_lo:
            T_hi = T_lo + 1.0

        # ── First call: build layout + colorbar once ────────────────────
        if self._placeholder_active or self._colorbar is None:
            self._placeholder_active = False
            self._norm = Normalize(vmin=T_lo, vmax=T_hi)
            self._sm = ScalarMappable(norm=self._norm, cmap=self._cmap_fn)
            self._sm.set_array([])

            # Set axis labels once
            self._ax.set_xlabel("x [m]", color="white", labelpad=4)
            self._ax.set_ylabel("y [m]", color="white", labelpad=4)
            self._ax.set_zlabel("Height [m]", color="white", labelpad=4)
            self._ax.tick_params(colors="white", labelsize=7)
            self._ax.xaxis.pane.fill = False
            self._ax.yaxis.pane.fill = False
            self._ax.zaxis.pane.fill = False
            self._ax.xaxis.pane.set_edgecolor("#555")
            self._ax.yaxis.pane.set_edgecolor("#555")
            self._ax.zaxis.pane.set_edgecolor("#555")

            # Create colorbar once (reserves space in the layout)
            self._colorbar = self._fig.colorbar(
                self._sm, ax=self._ax, shrink=0.55, pad=0.05
            )
            self._colorbar.set_label("Temperature [°C]", color="white")
            self._colorbar.ax.tick_params(colors="white")

            # Set axis limits once
            r_max = max(r_bottom, r_top)
            self._ax.set_xlim(-r_max * 1.15, r_max * 1.15)
            self._ax.set_ylim(-r_max * 1.15, r_max * 1.15)
            self._ax.set_zlim(0, H)
            self._ax.view_init(elev=25, azim=-60)

        else:
            # ── Subsequent calls: update norm, keep layout stable ────────
            self._norm.vmin = T_lo
            self._norm.vmax = T_hi
            self._sm.set_clim(T_lo, T_hi)
            self._colorbar.update_normal(self._sm)

            # Update axis limits when geometry changes
            r_max = max(r_bottom, r_top)
            self._ax.set_xlim(-r_max * 1.15, r_max * 1.15)
            self._ax.set_ylim(-r_max * 1.15, r_max * 1.15)
            self._ax.set_zlim(0, H)

        # ── Remove surfaces and redraw ───────────────────────────────────
        self._clear_artists()

        if geo_type == "pyramid" and pyr_dims is not None:
            a_b, b_b, a_t, b_t = pyr_dims
            self._render_pyramid_surface(T, H, a_b, b_b, a_t, b_t)
            self._render_rect_cap(H, a_t, b_t, float(T[0]))
            self._render_rect_cap(0.0, a_b, b_b, float(T[-1]))
        else:
            self._render_tank_surface(T, H, r_bottom, r_top, geo_type)
            self._render_cap(H, r_top, float(T[0]))
            self._render_cap(0.0, r_bottom, float(T[-1]))
        for port in ports:
            self._render_port(port, H, r_bottom, r_top)

        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal render methods
    # ------------------------------------------------------------------

    def _clear_artists(self):
        """Remove only 3D surfaces and quiver objects, not the axes themselves."""
        # Collections = plot_surface objects
        for coll in list(self._ax.collections):
            coll.remove()
        # Lines = quiver / other lines
        for line in list(self._ax.lines):
            line.remove()
        # Text (port labels) — remove transData texts, keep transAxes texts
        for txt in list(self._ax.texts):
            txt.remove()

    def _render_tank_surface(
        self,
        T: np.ndarray,
        H: float,
        r_bottom: float,
        r_top: float,
        geo_type: str,
    ):
        """Draw the side wall as individual layers (one plot_surface each)."""
        n_nodes = len(T)
        n_theta = 36
        theta = np.linspace(0, 2 * np.pi, n_theta + 1)
        z_edges = np.linspace(0, H, n_nodes + 1)

        for i in range(n_nodes):
            z0, z1 = z_edges[i], z_edges[i + 1]
            node_idx = n_nodes - 1 - i          # layer 0 = bottom = last node
            color = self._cmap_fn(self._norm(float(T[node_idx])))

            if geo_type == "cone":
                r0 = r_bottom + (r_top - r_bottom) * z0 / H
                r1 = r_bottom + (r_top - r_bottom) * z1 / H
            else:
                r0 = r1 = float(r_bottom)

            Theta, Z = np.meshgrid(theta, [z0, z1])
            R = np.array([[r0] * (n_theta + 1), [r1] * (n_theta + 1)])
            self._ax.plot_surface(
                R * np.cos(Theta), R * np.sin(Theta), Z,
                color=color, shade=False, linewidth=0,
                antialiased=False, alpha=0.9,
            )

    def _render_cap(self, z: float, radius: float, temp: float):
        """Draw lid or base as a circular disc."""
        theta = np.linspace(0, 2 * np.pi, 37)
        r_arr = np.linspace(0, radius, 3)
        R, Theta = np.meshgrid(r_arr, theta)
        color = self._cmap_fn(self._norm(temp))
        Z = np.full_like(R, z)
        self._ax.plot_surface(
            R * np.cos(Theta), R * np.sin(Theta), Z,
            color=color, shade=False, linewidth=0,
            antialiased=False, alpha=0.95,
        )

    def _render_pyramid_surface(
        self,
        T: np.ndarray,
        H: float,
        a_bottom: float,
        b_bottom: float,
        a_top: float,
        b_top: float,
    ):
        """Draw truncated pyramid: 4 trapezoidal faces per layer."""
        n_nodes = len(T)
        z_edges = np.linspace(0, H, n_nodes + 1)

        for i in range(n_nodes):
            z0, z1 = z_edges[i], z_edges[i + 1]
            node_idx = n_nodes - 1 - i
            color = self._cmap_fn(self._norm(float(T[node_idx])))

            # Rectangle dimensions at z0 and z1 (linearly interpolated)
            a0 = a_bottom + (a_top - a_bottom) * z0 / H
            b0 = b_bottom + (b_top - b_bottom) * z0 / H
            a1 = a_bottom + (a_top - a_bottom) * z1 / H
            b1 = b_bottom + (b_top - b_bottom) * z1 / H
            ha0, hb0 = a0 / 2, b0 / 2
            ha1, hb1 = a1 / 2, b1 / 2

            # 4 side faces, each as a 2×2 plot_surface
            faces = [
                # Front (y = +b/2)
                (np.array([[-ha0, ha0], [-ha1, ha1]]),
                 np.array([[hb0,  hb0], [hb1,  hb1]]),
                 np.array([[z0,   z0 ], [z1,   z1 ]])),
                # Back (y = -b/2)
                (np.array([[-ha0, ha0], [-ha1, ha1]]),
                 np.array([[-hb0, -hb0], [-hb1, -hb1]]),
                 np.array([[z0,   z0 ], [z1,   z1 ]])),
                # Right (x = +a/2)
                (np.array([[ha0,  ha0], [ha1,  ha1]]),
                 np.array([[-hb0, hb0], [-hb1, hb1]]),
                 np.array([[z0,   z0 ], [z1,   z1 ]])),
                # Left (x = -a/2)
                (np.array([[-ha0, -ha0], [-ha1, -ha1]]),
                 np.array([[-hb0,  hb0], [-hb1,  hb1]]),
                 np.array([[z0,    z0 ], [z1,    z1 ]])),
            ]
            for X, Y, Z in faces:
                self._ax.plot_surface(
                    X, Y, Z,
                    color=color, shade=False, linewidth=0,
                    antialiased=False, alpha=0.9,
                )

    def _render_rect_cap(self, z: float, a: float, b: float, temp: float):
        """Draw rectangular lid/base face for a truncated pyramid."""
        ha, hb = a / 2, b / 2
        X = np.array([[-ha, ha], [-ha, ha]])
        Y = np.array([[-hb, -hb], [hb, hb]])
        Z = np.full_like(X, z)
        color = self._cmap_fn(self._norm(temp))
        self._ax.plot_surface(
            X, Y, Z,
            color=color, shade=False, linewidth=0,
            antialiased=False, alpha=0.95,
        )

    def _render_port(self, port: dict, H: float, r_bottom: float, r_top: float):
        """Draw a port as a coloured arrow."""
        z = port.get("z", H / 2)
        port_type = port.get("type", "charge_in")
        label = port.get("label", "")
        color = self._PORT_COLORS.get(port_type, "white")

        r = r_bottom + (r_top - r_bottom) * (z / H) if H > 0 else r_bottom
        is_inlet = port_type in ("charge_in", "discharge_in")
        arrow_len = r * 0.35
        sign = -1.0 if is_inlet else 1.0
        x_start = r * (1.0 + (1.0 if is_inlet else 0.0) * 0.35)

        self._ax.quiver(
            x_start, 0.0, z,
            sign * arrow_len, 0.0, 0.0,
            color=color, arrow_length_ratio=0.4, linewidth=2.0,
        )
        if label:
            x_lbl = r * 1.45 if not is_inlet else r * 1.6
            self._ax.text(
                x_lbl, 0.0, z, label,
                color=color, fontsize=7, ha="left", va="center",
            )

    def _draw_placeholder(self):
        """Placeholder text shown before the first render."""
        self._ax.text2D(
            0.5, 0.5,
            "Load configuration\nand start simulation",
            ha="center", va="center",
            transform=self._ax.transAxes,
            color="#8888aa", fontsize=10,
        )
        self._ax.tick_params(colors="#555")
        self._canvas.draw_idle()

    def set_view_elevation(self, elev: float = 25.0, azim: float = -60.0):
        """Set the camera elevation and azimuth angles."""
        self._ax.view_init(elev=elev, azim=azim)
        self._canvas.draw_idle()
