#!/usr/bin/env python3
"""
Entry point for the Qt (PySide6) user interface of the 1D Thermal Storage Simulator.

Usage:
    python run_ui.py

Requirements:
    pip install -r requirements_ui.txt
"""

import sys
from pathlib import Path

# Ensure project directory is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from PySide6.QtWidgets import QApplication
except ImportError:
    print(
        "PySide6 not found. Please install:\n"
        "  pip install PySide6-Essentials\n"
        "or:\n"
        "  pip install -r requirements_ui.txt"
    )
    sys.exit(1)

from ui.main_window import MainWindow


def main():
    # High-DPI support is enabled by default in Qt 6
    app = QApplication(sys.argv)
    app.setApplicationName("1D Thermal Storage Simulator")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("TES-Simulation")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
