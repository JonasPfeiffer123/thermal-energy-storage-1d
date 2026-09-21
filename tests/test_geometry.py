"""Geometry model correctness.

``CylinderGeometry`` is exercised indirectly by nearly every other test (via
the ``make_storage`` fixture default), but its own constructor/validation
paths are not. ``TruncatedConeGeometry`` and ``TruncatedPyramidGeometry`` -
the geometries used for PTES, i.e. exactly what the Dronninglund/Høje
Taastrup validation targets - have no coverage at all outside the optional,
data-gated validation scripts.

Where possible, checks compare against a closed-form or independently
computed reference rather than restating the implementation's own formula,
so a discretization bug (not just a typo) would be caught:

- Node volumes are exact sub-frustums/sub-prismatoids, so ``sum(V_nodes(n))``
  must equal ``.volume`` exactly for *any* n (no discretization error).
- Node lateral wall areas sum (via the trapezoid-rule-is-exact-for-linear-
  profiles identity) to the closed-form frustum/trapezoid lateral-area
  formula, again exactly for any n.
- Pyramid volume is additionally checked against Simpson's rule integration
  of A(z) = a(z)*b(z), which is exact for the quadratic integrand and uses a
  numerical method independent of the closed-form volume formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_energy_storage_model import (
    CylinderGeometry,
    TruncatedConeGeometry,
    TruncatedPyramidGeometry,
)

# ---------------------------------------------------------------------------
# CylinderGeometry
# ---------------------------------------------------------------------------


def test_cylinder_from_volume_matches_explicit_radius():
    volume, height = 500.0, 15.0
    geom = CylinderGeometry.from_volume(volume=volume, height=height)
    assert geom.volume == pytest.approx(volume, rel=1e-12)
    expected_radius = np.sqrt(volume / (np.pi * height))
    assert geom.radius == pytest.approx(expected_radius, rel=1e-12)


def test_cylinder_a_cross_independent_of_height():
    geom = CylinderGeometry(radius=3.0, height=10.0)
    assert geom.A_cross(0.0) == pytest.approx(np.pi * 3.0**2)
    assert geom.A_cross(10.0) == pytest.approx(np.pi * 3.0**2)


def test_cylinder_wall_area_matches_lateral_plus_caps():
    r, H, n = 3.0, 10.0, 10
    geom = CylinderGeometry(radius=r, height=H)
    areas = geom.A_wall_nodes(n)
    dz = H / n
    lateral = 2.0 * np.pi * r * dz
    cap = np.pi * r**2
    assert areas[0] == pytest.approx(lateral + cap)
    assert areas[-1] == pytest.approx(lateral + cap)
    assert np.allclose(areas[1:-1], lateral)


@pytest.mark.parametrize("kwargs", [
    dict(radius=0.0, height=5.0),
    dict(radius=-1.0, height=5.0),
    dict(radius=5.0, height=0.0),
])
def test_cylinder_rejects_nonpositive_dimensions(kwargs):
    with pytest.raises(ValueError):
        CylinderGeometry(**kwargs)


def test_cylinder_from_volume_rejects_nonpositive():
    with pytest.raises(ValueError):
        CylinderGeometry.from_volume(volume=0.0, height=5.0)
    with pytest.raises(ValueError):
        CylinderGeometry.from_volume(volume=5.0, height=0.0)


# ---------------------------------------------------------------------------
# TruncatedConeGeometry
# ---------------------------------------------------------------------------


def test_cone_volume_matches_frustum_formula():
    r_b, r_t, H = 10.0, 20.0, 8.0
    geom = TruncatedConeGeometry(r_bottom=r_b, r_top=r_t, height=H)
    expected = np.pi * H / 3.0 * (r_t**2 + r_t * r_b + r_b**2)
    assert geom.volume == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("n", [1, 2, 5, 17, 50])
def test_cone_v_nodes_sum_to_total_volume(n):
    geom = TruncatedConeGeometry(r_bottom=10.0, r_top=20.0, height=8.0)
    assert np.sum(geom.V_nodes(n)) == pytest.approx(geom.volume, rel=1e-10)


@pytest.mark.parametrize("n", [1, 3, 10, 40])
def test_cone_wall_area_matches_frustum_lateral_formula(n):
    r_b, r_t, H = 10.0, 20.0, 8.0
    geom = TruncatedConeGeometry(r_bottom=r_b, r_top=r_t, height=H)
    areas = geom.A_wall_nodes(n)
    total_lateral = np.sum(areas) - np.pi * r_t**2 - np.pi * r_b**2
    slant = np.sqrt(H**2 + (r_t - r_b) ** 2)
    expected_lateral = np.pi * (r_b + r_t) * slant
    assert total_lateral == pytest.approx(expected_lateral, rel=1e-10)


def test_cone_degenerate_equals_cylinder():
    """r_bottom == r_top must reproduce CylinderGeometry exactly."""
    r, H, n = 6.0, 12.0, 15
    cone = TruncatedConeGeometry(r_bottom=r, r_top=r, height=H)
    cyl = CylinderGeometry(radius=r, height=H)
    assert cone.volume == pytest.approx(cyl.volume, rel=1e-12)
    assert np.allclose(cone.A_cross_nodes(n), cyl.A_cross_nodes(n))
    assert np.allclose(cone.V_nodes(n), cyl.V_nodes(n))
    assert np.allclose(cone.A_wall_nodes(n), cyl.A_wall_nodes(n))


def test_cone_cross_section_shrinks_toward_narrow_end():
    """Index 0 = top. With r_top > r_bottom, cross-section must decrease
    monotonically from top (index 0) to bottom (index N-1)."""
    geom = TruncatedConeGeometry(r_bottom=5.0, r_top=15.0, height=10.0)
    A = geom.A_cross_nodes(20)
    assert np.all(np.diff(A) < 0.0)
    assert geom.A_cross(0.0) == pytest.approx(np.pi * 5.0**2)
    assert geom.A_cross(10.0) == pytest.approx(np.pi * 15.0**2)


@pytest.mark.parametrize("kwargs", [
    dict(r_bottom=0.0, r_top=5.0, height=5.0),
    dict(r_bottom=5.0, r_top=-1.0, height=5.0),
    dict(r_bottom=5.0, r_top=5.0, height=0.0),
])
def test_cone_rejects_nonpositive_dimensions(kwargs):
    with pytest.raises(ValueError):
        TruncatedConeGeometry(**kwargs)


def test_cone_exposes_input_dimensions():
    geom = TruncatedConeGeometry(r_bottom=5.0, r_top=15.0, height=10.0)
    assert geom.r_bottom == pytest.approx(5.0)
    assert geom.r_top == pytest.approx(15.0)
    assert geom.height == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# TruncatedPyramidGeometry
# ---------------------------------------------------------------------------

_PYRAMID_DIMS = dict(a_bottom=10.0, b_bottom=8.0, a_top=20.0, b_top=14.0, height=6.0)


def test_pyramid_volume_matches_closed_form():
    geom = TruncatedPyramidGeometry(**_PYRAMID_DIMS)
    a_b, b_b, a_t, b_t, H = (
        _PYRAMID_DIMS["a_bottom"], _PYRAMID_DIMS["b_bottom"],
        _PYRAMID_DIMS["a_top"], _PYRAMID_DIMS["b_top"], _PYRAMID_DIMS["height"],
    )
    expected = H / 6.0 * (2 * a_t * b_t + 2 * a_b * b_b + a_t * b_b + a_b * b_t)
    assert geom.volume == pytest.approx(expected, rel=1e-12)


def test_pyramid_volume_matches_simpson_integration():
    """Independent numerical check: Simpson's rule is exact for the
    quadratic integrand A(z) = a(z)*b(z), giving a machine-precision cross
    check that does not share the closed-form formula with the
    implementation."""
    a_b, b_b, a_t, b_t, H = (
        _PYRAMID_DIMS["a_bottom"], _PYRAMID_DIMS["b_bottom"],
        _PYRAMID_DIMS["a_top"], _PYRAMID_DIMS["b_top"], _PYRAMID_DIMS["height"],
    )
    geom = TruncatedPyramidGeometry(**_PYRAMID_DIMS)

    n_int = 1000  # even, required by composite Simpson's rule
    z = np.linspace(0.0, H, n_int + 1)
    a = a_b + (a_t - a_b) * z / H
    b = b_b + (b_t - b_b) * z / H
    f = a * b
    weights = np.ones(n_int + 1)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    dz = H / n_int
    V_simpson = dz / 3.0 * np.sum(weights * f)

    assert geom.volume == pytest.approx(V_simpson, rel=1e-10)


@pytest.mark.parametrize("n", [1, 2, 5, 17, 50])
def test_pyramid_v_nodes_sum_to_total_volume(n):
    geom = TruncatedPyramidGeometry(**_PYRAMID_DIMS)
    assert np.sum(geom.V_nodes(n)) == pytest.approx(geom.volume, rel=1e-10)


@pytest.mark.parametrize("n", [1, 3, 10, 40])
def test_pyramid_wall_area_matches_closed_form(n):
    a_b, b_b, a_t, b_t, H = (
        _PYRAMID_DIMS["a_bottom"], _PYRAMID_DIMS["b_bottom"],
        _PYRAMID_DIMS["a_top"], _PYRAMID_DIMS["b_top"], _PYRAMID_DIMS["height"],
    )
    geom = TruncatedPyramidGeometry(**_PYRAMID_DIMS)
    areas = geom.A_wall_nodes(n)
    total_lateral = np.sum(areas) - a_t * b_t - a_b * b_b

    m_a = (a_t - a_b) / (2.0 * H)
    m_b = (b_t - b_b) / (2.0 * H)
    slant_a = np.sqrt(1.0 + m_a**2)
    slant_b = np.sqrt(1.0 + m_b**2)
    expected_lateral = (a_b + a_t) * H * slant_b + (b_b + b_t) * H * slant_a
    assert total_lateral == pytest.approx(expected_lateral, rel=1e-10)


def test_pyramid_degenerate_equals_rectangular_prism():
    """Zero slope (a_top=a_bottom, b_top=b_bottom) reduces to a box: volume
    = a*b*H, walls are vertical with area 2*(a+b)*dz per interior node."""
    a, b, H, n = 6.0, 4.0, 10.0, 12
    geom = TruncatedPyramidGeometry(a_bottom=a, b_bottom=b, a_top=a, b_top=b, height=H)
    assert geom.volume == pytest.approx(a * b * H, rel=1e-12)
    assert np.allclose(geom.V_nodes(n), np.full(n, a * b * H / n))

    dz = H / n
    areas = geom.A_wall_nodes(n)
    lateral_per_node = 2.0 * (a + b) * dz
    assert np.allclose(areas[1:-1], lateral_per_node)
    assert areas[0] == pytest.approx(lateral_per_node + a * b)
    assert areas[-1] == pytest.approx(lateral_per_node + a * b)


def test_pyramid_exposes_input_dimensions_and_cross_section():
    geom = TruncatedPyramidGeometry(**_PYRAMID_DIMS)
    assert geom.a_bottom == pytest.approx(_PYRAMID_DIMS["a_bottom"])
    assert geom.b_bottom == pytest.approx(_PYRAMID_DIMS["b_bottom"])
    assert geom.a_top == pytest.approx(_PYRAMID_DIMS["a_top"])
    assert geom.b_top == pytest.approx(_PYRAMID_DIMS["b_top"])
    assert geom.height == pytest.approx(_PYRAMID_DIMS["height"])

    H = _PYRAMID_DIMS["height"]
    assert geom.A_cross(0.0) == pytest.approx(
        _PYRAMID_DIMS["a_bottom"] * _PYRAMID_DIMS["b_bottom"]
    )
    assert geom.A_cross(H) == pytest.approx(
        _PYRAMID_DIMS["a_top"] * _PYRAMID_DIMS["b_top"]
    )
    # A_cross_nodes at the node centers must agree with the continuous A_cross.
    n = 8
    dz = H / n
    z_centers = np.array([(n - 0.5 - i) * dz for i in range(n)])
    expected = np.array([geom.A_cross(z) for z in z_centers])
    assert np.allclose(geom.A_cross_nodes(n), expected)


def test_pyramid_from_slope_matches_explicit_construction():
    a_b, b_b, H, slope = 100.0, 100.0, 10.0, 2.0
    via_slope = TruncatedPyramidGeometry.from_slope(
        a_bottom=a_b, b_bottom=b_b, height=H, slope=slope
    )
    a_t = a_b + 2.0 * slope * H
    b_t = b_b + 2.0 * slope * H
    explicit = TruncatedPyramidGeometry(
        a_bottom=a_b, b_bottom=b_b, a_top=a_t, b_top=b_t, height=H
    )
    assert via_slope.volume == pytest.approx(explicit.volume, rel=1e-12)
    assert via_slope.a_top == pytest.approx(a_t)
    assert via_slope.b_top == pytest.approx(b_t)


@pytest.mark.parametrize("kwargs", [
    dict(a_bottom=0.0, b_bottom=5.0, a_top=5.0, b_top=5.0, height=5.0),
    dict(a_bottom=5.0, b_bottom=-1.0, a_top=5.0, b_top=5.0, height=5.0),
    dict(a_bottom=5.0, b_bottom=5.0, a_top=0.0, b_top=5.0, height=5.0),
    dict(a_bottom=5.0, b_bottom=5.0, a_top=5.0, b_top=-2.0, height=5.0),
    dict(a_bottom=5.0, b_bottom=5.0, a_top=5.0, b_top=5.0, height=0.0),
])
def test_pyramid_rejects_nonpositive_dimensions(kwargs):
    with pytest.raises(ValueError):
        TruncatedPyramidGeometry(**kwargs)
