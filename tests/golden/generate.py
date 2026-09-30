"""
Generate or check the golden-master reference.

Usage (from the repository root)::

    # Record the reference from the *current* code (refuses to overwrite
    # an existing file unless --force is given):
    python -m tests.golden.generate

    # Compare the current code against the reference, in parallel, and
    # print the digest matches and max. relative checkpoint deviations:
    python -m tests.golden.generate --check

    # Optionally keep / compare the complete trajectories (large, use a
    # scratch directory) to quantify deviations over every hour and node:
    python -m tests.golden.generate --check --save-full DIR
    python -m tests.golden.generate --check --compare-full DIR

The reference must be recorded from code whose results are to be preserved,
*before* the change that is to be verified.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .reference import (
    REFERENCE_FILE,
    load_reference,
    max_rel_deviation,
    platform_tag,
    save_reference,
)
from .scenarios import (
    QUANTITIES,
    build_profile,
    build_scenarios,
    reduce_run,
    run_scenario,
)


def _run_one(args: tuple[str, str | None]) -> tuple[str, dict]:
    sid, save_full = args
    scenario = next(s for s in build_scenarios() if s.sid == sid)
    traj = run_scenario(scenario, build_profile())
    if save_full:
        np.savez(Path(save_full) / f"{sid}.npz", **traj)
    return sid, reduce_run(traj)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _compare_full(sid: str, full_dir: Path) -> dict[str, tuple[float, float]]:
    """Max abs / rel deviation per quantity over complete trajectories."""
    scenario = next(s for s in build_scenarios() if s.sid == sid)
    traj = run_scenario(scenario, build_profile())
    with np.load(full_dir / f"{sid}.npz") as ref:
        res = {}
        for q in QUANTITIES:
            new, old = traj[q], ref[q]
            ok = ~np.isnan(old)
            d = np.abs(new[ok] - old[ok]) if ok.any() else np.zeros(1)
            res[q] = (float(np.max(d, initial=0.0)),
                      max_rel_deviation(new, old))
    return res


def _compare_full_job(args: tuple[str, str]) -> tuple[str, dict]:
    sid, full_dir = args
    return sid, _compare_full(sid, Path(full_dir))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true",
                   help="compare against the reference instead of writing it")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing reference file")
    p.add_argument("--jobs", type=int, default=None, help="worker processes")
    p.add_argument("--only", default=None,
                   help="restrict to scenario ids containing this substring")
    p.add_argument("--save-full", default=None, metavar="DIR",
                   help="also store complete trajectories in DIR")
    p.add_argument("--compare-full", default=None, metavar="DIR",
                   help="compare complete trajectories against DIR")
    p.add_argument("--output", default=str(REFERENCE_FILE))
    a = p.parse_args(argv)

    sids = [s.sid for s in build_scenarios()
            if a.only is None or a.only in s.sid]
    out_path = Path(a.output)

    if a.compare_full:
        with ProcessPoolExecutor(a.jobs) as ex:
            results = dict(ex.map(_compare_full_job,
                                  [(sid, a.compare_full) for sid in sids]))
        worst = {q: (0.0, 0.0, "") for q in QUANTITIES}
        for sid, res in results.items():
            for q, (dabs, drel) in res.items():
                if drel > worst[q][1] or dabs > worst[q][0]:
                    worst[q] = (max(dabs, worst[q][0]), max(drel, worst[q][1]), sid)
        n_exact = sum(all(v == (0.0, 0.0) for v in r.values())
                      for r in results.values())
        print(f"{n_exact}/{len(results)} scenarios identical over the full year")
        for q, (dabs, drel, sid) in worst.items():
            print(f"  {q:12s} max abs {dabs:.3e}  max rel {drel:.3e}  ({sid})")
        return 0 if n_exact == len(results) else 1

    if a.save_full:
        Path(a.save_full).mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(a.jobs) as ex:
        reduced = dict(ex.map(_run_one, [(sid, a.save_full) for sid in sids]))

    if not a.check:
        if out_path.exists() and not a.force:
            print(f"{out_path} exists; use --force to overwrite.", file=sys.stderr)
            return 2
        meta = {
            "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "platform": platform_tag(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        }
        save_reference(out_path, meta, reduced)
        print(f"wrote {len(reduced)} scenarios to {out_path}")
        return 0

    meta, data = load_reference(out_path)
    n_bitwise = 0
    worst_rel = 0.0
    for sid in sids:
        red = reduced[sid]
        digests = meta["digests"][sid]
        diff_q = [q for q in QUANTITIES if red[f"{q}.sha256"] != digests[q]]
        if not diff_q:
            n_bitwise += 1
            continue
        devs = {q: max_rel_deviation(red[f"{q}.checkpoints"], data[sid][q])
                for q in diff_q}
        worst_rel = max(worst_rel, *devs.values())
        txt = ", ".join(f"{q}={d:.2e}" for q, d in devs.items())
        print(f"DIFF {sid}: {txt}")
    print(f"{n_bitwise}/{len(sids)} scenarios bit-identical; "
          f"max rel. checkpoint deviation {worst_rel:.3e}")
    return 0 if n_bitwise == len(sids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
