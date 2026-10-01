"""
Golden-master regression: one simulated year per configuration.

The reference (``tests/golden/reference.npz``) was recorded from the solver
*before* the step() performance work (see its ``git_commit`` metadata) and
covers node counts 5/20/50, every geometry, loss model and fluid model,
buoyancy on/off, the implicit/explicit x upwind/TVD variants, the presets and
the port/diffusor/heat-exchanger/headspace paths (``tests/golden/scenarios.py``).

Two checks per scenario:

* ``test_golden_master_bitwise`` -- the complete hourly trajectories of every
  recorded quantity (all node temperatures, ``Q_loss``, SOC, stored energy,
  ``T_mean``/``T_top``/``T_bottom``, port/HX outlet and headspace
  temperatures) hash to the recorded SHA-256 digests, i.e. they are
  bit-identical. Only run on the platform the reference was recorded on:
  ``exp``/``pow`` from the platform math library may differ in the last ulp
  elsewhere.
* ``test_golden_master_tolerance`` -- checkpoint values agree within
  ``RTOL`` relative deviation on every platform.

Deselect with ``pytest -m "not golden"`` for a quick run. Regenerate the
reference only from code whose results are to be preserved
(``python -m tests.golden.generate --force``).
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from tests.golden.reference import load_reference, max_rel_deviation, platform_tag
from tests.golden.scenarios import (
    QUANTITIES,
    build_profile,
    build_scenarios,
    reduce_run,
    run_scenario,
)

pytestmark = pytest.mark.golden

# Maximum relative checkpoint deviation accepted on any platform.
RTOL = 1e-12

META, DATA = load_reference()
SCENARIOS = {s.sid: s for s in build_scenarios()}


@functools.lru_cache(maxsize=1)
def _profile():
    return build_profile()


@functools.cache
def _reduced(sid: str) -> dict:
    return reduce_run(run_scenario(SCENARIOS[sid], _profile()))


def test_reference_covers_all_scenarios():
    assert set(META["digests"]) == set(SCENARIOS)


@pytest.mark.parametrize("sid", list(SCENARIOS))
def test_golden_master_tolerance(sid):
    red = _reduced(sid)
    devs = {q: max_rel_deviation(red[f"{q}.checkpoints"], DATA[sid][q])
            for q in QUANTITIES}
    bad = {q: d for q, d in devs.items() if d > RTOL}
    assert not bad, f"relative deviation above {RTOL:g}: {bad}"


@pytest.mark.parametrize("sid", list(SCENARIOS))
def test_golden_master_bitwise(sid):
    if platform_tag() != META["platform"]:
        pytest.skip(f"reference recorded on {META['platform']}, "
                    f"running on {platform_tag()}")
    red = _reduced(sid)
    ref = META["digests"][sid]
    changed = [q for q in QUANTITIES if red[f"{q}.sha256"] != ref[q]]
    if changed:
        devs = {q: f"{max_rel_deviation(red[f'{q}.checkpoints'], DATA[sid][q]):.2e}"
                for q in changed}
        pytest.fail(f"not bit-identical: {changed} "
                    f"(max rel. checkpoint deviation {devs})")


def test_checkpoint_layout():
    # Guards the reduction itself: last checkpoint is the final hour.
    red = _reduced(next(iter(SCENARIOS)))
    assert red["time.checkpoints"][-1] == pytest.approx(8760 * 3600.0)
    assert np.all(np.isfinite(red["T.checkpoints"]))
