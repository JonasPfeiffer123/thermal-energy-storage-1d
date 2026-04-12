"""
compare_runs.py – Comparison of named benchmark runs
=====================================================

Reads results from benchmark/results/runs/{name}/ and displays them
as a table and optionally as a plot.

Usage
-----
    # List all available runs:
    python benchmark/compare_runs.py --list

    # Compare two runs (table):
    python benchmark/compare_runs.py baseline_explicit v2_implicit

    # With comparison plot:
    python benchmark/compare_runs.py baseline_explicit v2_implicit --plot
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).resolve().parent
RUNS_DIR      = BENCHMARK_DIR / "results" / "runs"


def list_runs() -> None:
    """List all available runs with metadata."""
    if not RUNS_DIR.exists() or not any(
        d for d in RUNS_DIR.iterdir() if d.is_dir() and not d.name.startswith("compare_")
    ):
        print("No runs found. Run first:")
        print("  python benchmark/benchmark_model_variants.py --name <runname>")
        return

    print(f"\n{'Run':<30}  {'Branch':<25}  {'Commit':<8}  {'Timestamp':<20}  {'FreeTTES'}")
    print("  " + "-" * 95)
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("compare_"):
            continue
        meta_path = run_dir / "run_metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            ft = "yes" if meta.get("include_freetttes") else "no"
            print(
                f"  {run_dir.name:<30}  "
                f"{meta.get('git_branch', '?'):<25}  "
                f"{meta.get('git_commit', '?'):<8}  "
                f"{meta.get('timestamp', '?'):<20}  "
                f"{ft}"
            )
        else:
            print(f"  {run_dir.name:<30}  (no metadata)")
    print()


def load_summary(run_name: str) -> dict:
    """Load comparison_summary.json for a run."""
    path = RUNS_DIR / run_name / "comparison_summary.json"
    if not path.exists():
        sys.exit(
            f"Error: {path} not found. "
            f"Run '{run_name}' does not exist or has no results."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_metadata(run_name: str) -> dict:
    """Load run_metadata.json for a run (optional)."""
    path = RUNS_DIR / run_name / "run_metadata.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_csv(run_name: str) -> dict[str, np.ndarray]:
    """
    Load comparison_results.csv for a run as a dict of float arrays.

    Empty strings and 'nan' in T_outlet columns are converted to np.nan
    so that matplotlib renders them as line gaps rather than errors.
    """
    import csv
    path = RUNS_DIR / run_name / "comparison_results.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    result: dict[str, np.ndarray] = {}
    if not rows:
        return result
    for key in rows[0]:
        vals = []
        all_float = True
        for r in rows:
            raw = r[key].strip()
            if raw in ("", "nan", "NaN"):
                vals.append(float("nan"))
            else:
                try:
                    vals.append(float(raw))
                except ValueError:
                    all_float = False
                    break
        if all_float:
            result[key] = np.array(vals, dtype=float)
        else:
            result[key] = np.array([r[key] for r in rows])
    return result


def compare_table(run_names: list[str]) -> None:
    """Print a comparison table for the given runs."""
    summaries  = {name: load_summary(name)  for name in run_names}
    metadatas  = {name: load_metadata(name) for name in run_names}

    # Collect all model labels (excluding the 'tank' key)
    all_labels: list[str] = []
    for name in run_names:
        for key in summaries[name]:
            if key != "tank" and key not in all_labels:
                all_labels.append(key)

    # Header row
    col_w = 28
    run_w = 50
    print()
    print(f"  {'Metric':<{col_w}}", end="")
    for name in run_names:
        meta = metadatas[name]
        branch = meta.get("git_branch", "?")
        commit = meta.get("git_commit", "?")
        header = f"{name} [{branch}@{commit}]"
        print(f"  {header:<{run_w}}", end="")
    print()
    print("  " + "-" * (col_w + len(run_names) * (run_w + 2)))

    # Metric rows
    fields = [
        ("Avg ms/step",              "avg_ms_per_step"),
        ("E_useful start [MWh]",     "E_nutz_start_MWh"),
        ("E_useful after chg [MWh]", "E_nutz_after_charge_MWh"),
        ("E_useful end [MWh]",       "E_nutz_end_MWh"),
        ("E_loss idle [MWh]",        "E_loss_idle_MWh"),
        ("T_outlet chg end [°C]",    "T_outlet_charge_end_C"),
        ("T_outlet dis end [°C]",    "T_outlet_disch_end_C"),
        ("MAE vs. FreeTTES [K]",     "MAE_vs_FreeTTES_K"),
    ]

    for label in all_labels:
        print(f"\n  -- {label} --")
        for field_name, field_key in fields:
            print(f"  {field_name:<{col_w}}", end="")
            for name in run_names:
                val = summaries[name].get(label, {}).get(field_key, "n/a")
                val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
                print(f"  {val_str:<{run_w}}", end="")
            print()
    print()


# ── Colour / style helpers ─────────────────────────────────────────────────────

# Fixed palette: up to 20 distinct model colours (tab20)
_TAB20 = None

def _tab20(i: int):
    global _TAB20
    if _TAB20 is None:
        import matplotlib.pyplot as plt
        _TAB20 = plt.cm.tab20.colors
    return _TAB20[i % 20]


_RUN_STYLES = ["-", "--", "-.", ":"]   # solid for run 0, dashed for run 1, …


def _phase_bands(ax, t_charge: float, t_idle: float, t_total: float) -> None:
    """Draw shaded phase regions and labels on axes ax."""
    ax.axvspan(0,          t_charge, alpha=0.07, color="tab:orange", zorder=0)
    ax.axvspan(t_charge,   t_idle,   alpha=0.07, color="tab:gray",   zorder=0)
    ax.axvspan(t_idle,     t_total,  alpha=0.07, color="tab:blue",   zorder=0)
    ax.axvline(t_charge, color="gray", linewidth=0.8, linestyle=":", zorder=1)
    ax.axvline(t_idle,   color="gray", linewidth=0.8, linestyle=":", zorder=1)
    ymin, ymax = ax.get_ylim()
    mid_y = ymin + 0.03 * (ymax - ymin)
    for x, lbl in [
        (t_charge / 2,              "Charging"),
        ((t_charge + t_idle) / 2,  "Idle"),
        ((t_idle + t_total) / 2,   "Discharging"),
    ]:
        ax.text(x, mid_y, lbl, ha="center", va="bottom", fontsize=7,
                color="gray", style="italic")


def compare_plot(run_names: list[str], out_dir: Path) -> None:
    """
    Generate a graphical comparison of T_outlet and E_useful time series.

    Colour encodes the model variant; line style encodes the run.
    Phase regions are shaded. Output is saved to out_dir/.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # ── Load data ──────────────────────────────────────────────────────────────
    all_data:  list[dict] = []
    all_meta:  list[dict] = []
    for name in run_names:
        d = load_csv(name)
        if not d:
            print(f"Warning: no CSV data for run '{name}'")
        all_data.append(d)
        all_meta.append(load_metadata(name))

    # ── Collect all model names across runs (ordered, unique) ─────────────────
    all_model_names: list[str] = []
    for d in all_data:
        for key in d:
            if key.startswith("T_outlet_"):
                name = key[len("T_outlet_"):]
                if name not in all_model_names:
                    all_model_names.append(name)

    _FREETTTES_COLOR = "black"
    _FREETTTES_LW    = 2.8

    color_map = {
        m: (_FREETTTES_COLOR if "FreeTTES" in m else _tab20(i))
        for i, m in enumerate(all_model_names)
    }

    # ── Phase boundaries (from first run's metadata, fallback to common values) ─
    meta0 = all_meta[0] if all_meta else {}
    t_charge = float(meta0.get("T_end_charge_h", 24))
    t_idle   = float(meta0.get("T_end_idle_h",   36))
    t_total  = float(meta0.get("T_total_h",       60))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_t, ax_e) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    for run_idx, (run_name, d) in enumerate(zip(run_names, all_data)):
        if not d:
            continue
        ls = _RUN_STYLES[run_idx % len(_RUN_STYLES)]
        t  = d.get("t_h", np.arange(60))
        lw = 1.6 if run_idx == 0 else 1.2

        for col_name in all_model_names:
            outlet_key = f"T_outlet_{col_name}"
            enutz_key  = f"E_nutz_MWh_{col_name}"
            is_ft  = "FreeTTES" in col_name
            color  = color_map[col_name]
            lw_col = _FREETTTES_LW if is_ft else lw
            alpha  = 1.0 if is_ft else 0.85

            if outlet_key in d:
                y = d[outlet_key]
                # mask NaN so matplotlib draws a line break during idle phase
                mask = ~np.isnan(y)
                t_m, y_m = t[mask], y[mask]
                ax_t.plot(t_m, y_m, ls, color=color, linewidth=lw_col, alpha=alpha,
                          zorder=3 if is_ft else 2)

            if enutz_key in d:
                ax_e.plot(t, d[enutz_key], ls, color=color, linewidth=lw_col, alpha=alpha,
                          zorder=3 if is_ft else 2)

    # ── Phase bands ───────────────────────────────────────────────────────────
    for ax in (ax_t, ax_e):
        _phase_bands(ax, t_charge, t_idle, t_total)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax_t.set_ylabel("T_outlet [°C]", fontsize=10)
    ax_t.set_title(
        "Outlet temperature  —  run comparison: " + " vs. ".join(run_names),
        fontsize=11,
    )
    ax_t.grid(True, alpha=0.3)

    ax_e.set_xlabel("Time [h]", fontsize=10)
    ax_e.set_ylabel("E_useful [MWh]", fontsize=10)
    ax_e.set_title("Usable energy", fontsize=11)
    ax_e.grid(True, alpha=0.3)

    ax_t.set_xlim(0, t_total)
    ax_e.set_xlim(0, t_total)

    # ── Legend: two parts ─────────────────────────────────────────────────────
    # Part 1 – model colours (top legend, placed outside above)
    model_handles = [
        Line2D([0], [0], color=color_map[m],
               linewidth=_FREETTTES_LW if "FreeTTES" in m else 2,
               label=m.replace("_", " "))
        for m in all_model_names
    ]
    # Part 2 – run line styles (bottom legend)
    run_handles = [
        Line2D([0], [0], color="black", linestyle=_RUN_STYLES[i], linewidth=2,
               label=run_names[i])
        for i in range(len(run_names))
    ]

    # Place model legend above the top plot
    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(model_handles), 5),
        fontsize=7,
        title="Model variant",
        title_fontsize=8,
        framealpha=0.9,
    )
    # Place run legend in the bottom-right of the lower plot
    ax_e.legend(
        handles=run_handles,
        loc="lower right",
        fontsize=9,
        title="Run (line style)",
        title_fontsize=9,
        framealpha=0.9,
    )

    fig.subplots_adjust(top=0.88, hspace=0.08, left=0.08, right=0.97, bottom=0.07)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "comparison_plot.svg"
    pdf_path = out_dir / "comparison_plot.pdf"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Comparison plot saved: {svg_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparison of named benchmark runs")
    parser.add_argument("runs", nargs="*", help="Run names (from benchmark/results/runs/)")
    parser.add_argument("--list", action="store_true", help="List all available runs")
    parser.add_argument("--plot", action="store_true", help="Generate a comparison plot")
    args = parser.parse_args()

    if args.list or not args.runs:
        list_runs()
        return

    if len(args.runs) < 2:
        sys.exit("Provide at least two run names to create a comparison.")

    compare_table(args.runs)

    if args.plot:
        names_str  = "_vs_".join(args.runs)
        out_dir    = RUNS_DIR / f"compare_{names_str}"
        compare_plot(args.runs, out_dir)


if __name__ == "__main__":
    main()
