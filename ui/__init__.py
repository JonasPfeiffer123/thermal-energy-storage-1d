"""
UI package for the 1D Thermal Storage Simulator.

Contains all Qt (PySide6) components for the graphical user interface
of the 1D thermal energy storage model.
"""

# Load the Qt binding before any matplotlib Qt backend: matplotlib uses the
# binding that is already imported and would otherwise prefer PyQt6 if that
# is installed as well (two bindings in one process do not work together).
import PySide6.QtCore  # noqa: F401
