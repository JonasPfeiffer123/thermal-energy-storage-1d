"""
Plot panel with four tabs.

Tabs:
  1. Temperature profile T(z)  – current thermocline (horizontal)
  2. Temperatures              – time series T_top / T_mid / T_bot / outlet
  3. Power                     – Q_charge / Q_discharge / Q_loss
  4. State of Charge (SOC)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy, QTabWidget, QVBoxLayout, QWidget

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Dark colour scheme ───────────────────────────────────────────────────────
_BG = "#1e1e2e"
_FG = "#cdd6f4"
_GRID = "#313244"
_COLORS = {
    "T_top":      "#f38ba8",   # red
    "T_mid":      "#fab387",   # orange
    "T_bot":      "#89b4fa",   # blue
    "T_outlet":   "#a6e3a1",   # green
    "Q_charge":   "#f38ba8",   # red
    "Q_discharge":"#89b4fa",   # blue
    "Q_loss":     "#f9e2af",   # yellow
    "SOC":        "#cba6f7",   # purple
}


def _dark_figure(nrows: int = 1, ncols: int = 1, **kwargs) -> tuple[Figure, any]:
    """Matplotlib Figure with dark background."""
    fig = Figure(facecolor=_BG, **kwargs)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.91, bottom=0.13)
    if nrows == 1 and ncols == 1:
        ax = fig.add_subplot(111)
    else:
        axes = fig.subplots(nrows, ncols)
        return fig, axes
    _style_ax(ax)
    return fig, ax


def _style_ax(ax):
    """Apply dark style to an axis."""
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.5, linestyle="--", alpha=0.7)


def _canvas_in_widget(fig: Figure, parent=None) -> tuple[QWidget, FigureCanvas]:
    """Wrap a FigureCanvas and NavigationToolbar in a QWidget."""
    w = QWidget(parent)
    layout = QVBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    canvas = FigureCanvas(fig)
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    toolbar = NavToolbar(canvas, w)
    toolbar.setStyleSheet(f"background: #2d2d3d; color: {_FG};")
    layout.addWidget(toolbar)
    layout.addWidget(canvas, 1)
    return w, canvas


class PlotsWidget(QTabWidget):
    """
    Four-tab plot panel for simulation results.

    Methods
    -------
    update_profile(z_nodes, temperatures)
        Update the temperature profile T(z).
    update_history(time_s, T_top, T_mid, T_bot, T_out,
                   Q_charge, Q_discharge, Q_loss, soc)
        Update all time-series plots.
    clear()
        Clear all stored data points.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_tabs()
        self.clear()
        self.setMinimumWidth(350)

    # ------------------------------------------------------------------
    # Tab setup
    # ------------------------------------------------------------------

    def _setup_tabs(self):
        # --- Tab 1: Temperature profile ---
        self._fig_profile, self._ax_profile = _dark_figure(figsize=(4, 5))
        self._ax_profile.set_xlabel("Temperature [°C]")
        self._ax_profile.set_ylabel("Height z [m]")
        self._ax_profile.set_title("Temperature profile T(z)")
        self._line_profile, = self._ax_profile.plot(
            [], [], "-o", color=_COLORS["T_top"], linewidth=2,
            markersize=3, label="current"
        )
        self._line_profile_prev, = self._ax_profile.plot(
            [], [], "-", color="#585b70", linewidth=1, alpha=0.5, label="previous"
        )
        self._ax_profile.legend(fontsize=7, labelcolor=_FG, facecolor=_BG,
                                 edgecolor=_GRID)
        w_profile, self._canvas_profile = _canvas_in_widget(self._fig_profile)
        self.addTab(w_profile, "Profile T(z)")

        # --- Tab 2: Temperatures ---
        self._fig_temp, self._ax_temp = _dark_figure(figsize=(5, 4))
        self._ax_temp.set_xlabel("Time [h]")
        self._ax_temp.set_ylabel("Temperature [°C]")
        self._ax_temp.set_title("Temperatures")
        self._lines_temp = {}
        for key, lbl in [
            ("T_top",    "T_top"),
            ("T_mid",    "T_mid"),
            ("T_bot",    "T_bot"),
            ("T_outlet", "T_outlet"),
        ]:
            line, = self._ax_temp.plot([], [], "-", color=_COLORS[key],
                                        linewidth=1.5, label=lbl)
            self._lines_temp[key] = line
        self._ax_temp.legend(fontsize=7, labelcolor=_FG, facecolor=_BG,
                              edgecolor=_GRID, loc="best")
        w_temp, self._canvas_temp = _canvas_in_widget(self._fig_temp)
        self.addTab(w_temp, "Temperatures")

        # --- Tab 3: Power ---
        self._fig_pow, self._ax_pow = _dark_figure(figsize=(5, 4))
        self._ax_pow.set_xlabel("Time [h]")
        self._ax_pow.set_ylabel("Power [kW]")
        self._ax_pow.set_title("Power")
        self._lines_pow = {}
        for key, lbl in [
            ("Q_charge",    "Charging"),
            ("Q_discharge", "Discharging"),
            ("Q_loss",      "Losses"),
        ]:
            line, = self._ax_pow.plot([], [], "-", color=_COLORS[key],
                                       linewidth=1.5, label=lbl)
            self._lines_pow[key] = line
        self._ax_pow.axhline(0, color=_GRID, linewidth=0.8, linestyle="--")
        self._ax_pow.legend(fontsize=7, labelcolor=_FG, facecolor=_BG,
                             edgecolor=_GRID)
        w_pow, self._canvas_pow = _canvas_in_widget(self._fig_pow)
        self.addTab(w_pow, "Power")

        # --- Tab 4: State of Charge ---
        self._fig_soc, self._ax_soc = _dark_figure(figsize=(5, 4))
        self._ax_soc.set_xlabel("Time [h]")
        self._ax_soc.set_ylabel("State of charge SOC [%]")
        self._ax_soc.set_title("State of Charge (SOC)")
        self._ax_soc.set_ylim(-5, 105)
        self._ax_soc.axhline(100, color=_COLORS["Q_charge"], linewidth=0.8,
                              linestyle=":", alpha=0.5)
        self._ax_soc.axhline(0, color=_COLORS["Q_discharge"], linewidth=0.8,
                              linestyle=":", alpha=0.5)
        self._line_soc, = self._ax_soc.plot([], [], "-", color=_COLORS["SOC"],
                                             linewidth=2.0)
        # Fill area
        self._fill_soc = None
        w_soc, self._canvas_soc = _canvas_in_widget(self._fig_soc)
        self.addTab(w_soc, "State of Charge")

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def clear(self):
        """Clear all stored history data."""
        self._time_h: list[float] = []
        self._T_top: list[float] = []
        self._T_mid: list[float] = []
        self._T_bot: list[float] = []
        self._T_out: list[float] = []
        self._Q_charge: list[float] = []
        self._Q_discharge: list[float] = []
        self._Q_loss: list[float] = []
        self._soc: list[float] = []
        self._prev_profile_T: Optional[np.ndarray] = None
        self._prev_profile_z: Optional[np.ndarray] = None
        self._refresh_all()

    def update_profile(self, z_nodes: np.ndarray, temperatures: np.ndarray):
        """
        Update the T(z) profile.

        Parameters
        ----------
        z_nodes : ndarray
            Node heights [m], ascending (bottom → top).
        temperatures : ndarray
            Temperatures [°C], corresponding to z_nodes.
            (Index 0 = top, so temperatures[::-1] matches z_nodes.)
        """
        # Save previous profile
        if len(self._line_profile.get_xdata()) > 0:
            self._prev_profile_T = np.array(self._line_profile.get_xdata())
            self._prev_profile_z = np.array(self._line_profile.get_ydata())

        # Temperatures: index 0 = top → z_nodes[-1] = top
        T_sorted = temperatures[::-1]  # now index 0 = bottom
        self._line_profile.set_data(T_sorted, z_nodes)

        if self._prev_profile_T is not None:
            self._line_profile_prev.set_data(self._prev_profile_T, self._prev_profile_z)

        # Adjust axis limits
        T_min = min(T_sorted.min(), 5.0)
        T_max = max(T_sorted.max(), 20.0)
        margin = (T_max - T_min) * 0.05
        self._ax_profile.set_xlim(T_min - margin, T_max + margin)
        self._ax_profile.set_ylim(z_nodes[0], z_nodes[-1])

        self._canvas_profile.draw_idle()

    def append_step(
        self,
        time_s: float,
        T_top: float,
        T_mid: float,
        T_bot: float,
        T_out: float,
        Q_charge_W: float,
        Q_discharge_W: float,
        Q_loss_W: float,
        soc_frac: float,
    ):
        """
        Append one simulation step to the history.

        Parameters
        ----------
        time_s : float
            Simulation time [s].
        T_top, T_mid, T_bot : float
            Temperatures at top / middle / bottom [°C].
        T_out : float
            Outlet temperature (discharge) [°C].
        Q_charge_W, Q_discharge_W : float
            Charging and discharging power [W] (positive = energy in/out).
        Q_loss_W : float
            Heat loss to environment [W] (positive = loss).
        soc_frac : float
            State of charge [0–1].
        """
        t_h = time_s / 3600.0
        self._time_h.append(t_h)
        self._T_top.append(T_top)
        self._T_mid.append(T_mid)
        self._T_bot.append(T_bot)
        self._T_out.append(T_out)
        self._Q_charge.append(Q_charge_W / 1000.0)      # W → kW
        self._Q_discharge.append(Q_discharge_W / 1000.0)
        self._Q_loss.append(Q_loss_W / 1000.0)
        self._soc.append(soc_frac * 100.0)               # 0–1 → %

        # Only update the active tab (performance)
        active = self.currentIndex()
        if active == 1:
            self._update_temp_plot()
        elif active == 2:
            self._update_power_plot()
        elif active == 3:
            self._update_soc_plot()

    def refresh_all_plots(self):
        """Redraw all four tabs at once (e.g. after tab switch)."""
        self._refresh_all()

    # ------------------------------------------------------------------
    # Internal drawing routines
    # ------------------------------------------------------------------

    def _refresh_all(self):
        self._update_temp_plot()
        self._update_power_plot()
        self._update_soc_plot()

    def _update_temp_plot(self):
        t = self._time_h
        if not t:
            for line in self._lines_temp.values():
                line.set_data([], [])
            self._canvas_temp.draw_idle()
            return

        self._lines_temp["T_top"].set_data(t, self._T_top)
        self._lines_temp["T_mid"].set_data(t, self._T_mid)
        self._lines_temp["T_bot"].set_data(t, self._T_bot)
        self._lines_temp["T_outlet"].set_data(t, self._T_out)

        all_T = self._T_top + self._T_mid + self._T_bot + self._T_out
        T_min = min(all_T)
        T_max = max(all_T)
        margin = max((T_max - T_min) * 0.05, 1.0)
        self._ax_temp.set_xlim(0, max(t) * 1.02)
        self._ax_temp.set_ylim(T_min - margin, T_max + margin)
        self._canvas_temp.draw_idle()

    def _update_power_plot(self):
        t = self._time_h
        if not t:
            for line in self._lines_pow.values():
                line.set_data([], [])
            self._canvas_pow.draw_idle()
            return

        self._lines_pow["Q_charge"].set_data(t, self._Q_charge)
        self._lines_pow["Q_discharge"].set_data(t, self._Q_discharge)
        self._lines_pow["Q_loss"].set_data(t, self._Q_loss)

        all_Q = self._Q_charge + self._Q_discharge + self._Q_loss
        Q_min = min(all_Q) if all_Q else 0
        Q_max = max(all_Q) if all_Q else 1
        margin = max((Q_max - Q_min) * 0.05, 0.1)
        self._ax_pow.set_xlim(0, max(t) * 1.02)
        self._ax_pow.set_ylim(Q_min - margin, Q_max + margin)
        self._canvas_pow.draw_idle()

    def _update_soc_plot(self):
        t = self._time_h
        if not t:
            self._line_soc.set_data([], [])
            self._canvas_soc.draw_idle()
            return

        soc = self._soc
        self._line_soc.set_data(t, soc)

        # Remove and redraw fill area
        if self._fill_soc is not None:
            self._fill_soc.remove()
        self._fill_soc = self._ax_soc.fill_between(
            t, 0, soc, color=_COLORS["SOC"], alpha=0.2
        )

        self._ax_soc.set_xlim(0, max(t) * 1.02)
        self._ax_soc.set_ylim(-5, 105)
        self._canvas_soc.draw_idle()
