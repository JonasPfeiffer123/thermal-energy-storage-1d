"""Loading and comparing the golden-master reference file."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

from .scenarios import QUANTITIES

REFERENCE_FILE = Path(__file__).with_name("reference.npz")


def platform_tag() -> str:
    """Identifies platforms on which bitwise reproduction is expected."""
    return f"{platform.system()}-{platform.machine()}"


def load_reference(path: Path = REFERENCE_FILE) -> tuple[dict, dict]:
    """
    Return ``(meta, data)``.

    ``meta`` holds the generation metadata and the per-scenario digests,
    ``data[sid][quantity]`` the checkpoint values.
    """
    with np.load(path) as npz:
        meta = json.loads(str(npz["__meta__"]))
        data: dict[str, dict[str, np.ndarray]] = {}
        for sid, shapes in meta["shapes"].items():
            flat = npz[sid]
            pos = 0
            data[sid] = {}
            for q in QUANTITIES:
                shape = tuple(shapes[q])
                size = int(np.prod(shape))
                data[sid][q] = flat[pos:pos + size].reshape(shape)
                pos += size
    return meta, data


def save_reference(path: Path, meta: dict, reduced: dict[str, dict]) -> None:
    """Write digests/shapes (in ``meta``) and checkpoint values to ``path``."""
    arrays: dict[str, np.ndarray] = {}
    digests: dict[str, dict[str, str]] = {}
    shapes: dict[str, dict[str, list[int]]] = {}
    for sid, red in reduced.items():
        digests[sid] = {q: red[f"{q}.sha256"] for q in QUANTITIES}
        cps = [np.asarray(red[f"{q}.checkpoints"], dtype=float) for q in QUANTITIES]
        shapes[sid] = {q: list(a.shape) for q, a in zip(QUANTITIES, cps)}
        arrays[sid] = np.concatenate([a.ravel() for a in cps])
    meta = {**meta, "digests": digests, "shapes": shapes}
    np.savez_compressed(path, __meta__=np.array(json.dumps(meta)), **arrays)


def max_rel_deviation(new: np.ndarray, ref: np.ndarray) -> float:
    """Largest |new - ref| / |ref| (NaN positions must coincide)."""
    new = np.asarray(new, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if new.shape != ref.shape:
        return float("inf")
    nan_new, nan_ref = np.isnan(new), np.isnan(ref)
    if not np.array_equal(nan_new, nan_ref):
        return float("inf")
    ok = ~nan_ref
    diff = np.abs(new[ok] - ref[ok])
    if diff.size == 0 or not np.any(diff):
        return 0.0
    scale = np.abs(ref[ok])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0.0, diff / scale, np.where(diff > 0, np.inf, 0.0))
    return float(np.max(rel))
