"""Optional validation regression against the Dronninglund PTES measurement data.

This test is skipped automatically unless both the measurement CSV (cloned into
``data/DronninglundData`` per the README) and ``pandas`` are available, so the
core suite stays self-contained. When the data is present it runs the existing
validation script and asserts the reported total MAE stays below a loose
regression bound -- a guard against gross physics regressions, not a tight
accuracy claim.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = (
    ROOT / "data/DronninglundData/data"
    / "Dronninglund_treated_data_and_flow_rates_2014.csv"
)
SCRIPT = ROOT / "benchmark" / "dronninglund_validation.py"
MAE_REGRESSION_BOUND_K = 8.0  # loose guard; current model is well below this


pandas = pytest.importorskip("pandas", reason="pandas not installed")


@pytest.mark.skipif(not DATA_FILE.exists(), reason="Dronninglund data not cloned")
def test_dronninglund_total_mae_regression():
    env = dict(
        os.environ,
        MPLBACKEND="Agg",          # headless: plt.show() becomes a no-op
        PYTHONIOENCODING="utf-8",  # the script prints non-ASCII (²,→); avoid cp1252 crash
    )
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(ROOT), env=env, capture_output=True,
        encoding="utf-8", errors="replace", timeout=600,
    )
    assert proc.returncode == 0, f"validation script failed:\n{proc.stderr}"

    match = re.search(r"Total:\s+([\d.]+)\s*K", proc.stdout)
    assert match is not None, f"could not parse total MAE from output:\n{proc.stdout}"
    total_mae = float(match.group(1))
    assert total_mae < MAE_REGRESSION_BOUND_K, f"total MAE regressed to {total_mae} K"
