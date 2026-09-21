"""StoragePresets factory and ThermalStorage1D.from_preset() correctness.

Only "steel_tank_aboveground" was exercised (indirectly, via CylinderGeometry
defaults) before this file; "steel_tank_buried", "ptes" and the
``from_preset`` classmethod itself had no coverage at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_energy_storage_model import (
    ConstantAmbientLoss,
    CylinderGeometry,
    GroundTemperatureLoss,
    StorageInputs,
    StoragePresets,
    ThermalStorage1D,
    TruncatedConeGeometry,
)

# ---------------------------------------------------------------------------
# StoragePresets: config wiring
# ---------------------------------------------------------------------------


def test_steel_tank_aboveground_wiring():
    config = StoragePresets.steel_tank_aboveground(
        volume=500.0, height=15.0, U_loss=0.2, T_ambient=12.0,
    )
    assert isinstance(config.geometry, CylinderGeometry)
    assert config.geometry.volume == pytest.approx(500.0)
    assert config.geometry.height == pytest.approx(15.0)
    assert isinstance(config.loss_model, ConstantAmbientLoss)
    assert config.loss_model.U_loss == pytest.approx(0.2)
    assert config.loss_model.T_ambient == pytest.approx(12.0)


def test_steel_tank_buried_wiring():
    config = StoragePresets.steel_tank_buried(
        volume=1000.0, height=12.0, U_loss=0.25,
        burial_depth=1.5, T_surface=9.0, T_deep=11.5, depth_decay=2.5,
    )
    assert isinstance(config.geometry, CylinderGeometry)
    assert config.geometry.volume == pytest.approx(1000.0)
    assert isinstance(config.loss_model, GroundTemperatureLoss)
    assert config.loss_model.U_loss == pytest.approx(0.25)
    assert config.loss_model.burial_depth == pytest.approx(1.5)
    assert config.loss_model.T_surface == pytest.approx(9.0)
    assert config.loss_model.T_deep == pytest.approx(11.5)
    assert config.loss_model.depth_decay == pytest.approx(2.5)


def test_ptes_wiring():
    config = StoragePresets.ptes(
        r_bottom=40.0, r_top=55.0, height=15.0,
        U_loss=0.22, burial_depth=0.5, T_surface=8.0, T_deep=11.0,
    )
    assert isinstance(config.geometry, TruncatedConeGeometry)
    assert config.geometry.r_bottom == pytest.approx(40.0)
    assert config.geometry.r_top == pytest.approx(55.0)
    assert config.geometry.height == pytest.approx(15.0)
    assert isinstance(config.loss_model, GroundTemperatureLoss)
    assert config.loss_model.U_loss == pytest.approx(0.22)
    assert config.loss_model.burial_depth == pytest.approx(0.5)
    # config.volume must be derived from the cone geometry, not left at 0.
    expected_volume = np.pi * 15.0 / 3.0 * (55.0**2 + 55.0 * 40.0 + 40.0**2)
    assert config.volume == pytest.approx(expected_volume)


# ---------------------------------------------------------------------------
# Each preset must produce a working, physically sane storage.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_kwargs", [
    ("steel_tank_aboveground", dict(volume=500.0, height=15.0)),
    ("steel_tank_buried", dict(volume=500.0, height=15.0, burial_depth=1.0)),
    ("ptes", dict(r_bottom=20.0, r_top=30.0, height=10.0)),
])
def test_preset_produces_working_storage(preset_kwargs):
    preset_name, kwargs = preset_kwargs
    config_factory = getattr(StoragePresets, preset_name)
    config = config_factory(**kwargs)
    storage = ThermalStorage1D(config)
    state = storage.initialize(T_init=60.0)

    inputs = StorageInputs.two_port(
        m_dot_charge=5.0, T_charge_in=85.0, height=config.geometry.height,
    )
    out = storage.step(state, dt=60.0, inputs=inputs)

    assert np.all(np.isfinite(out.state.temperatures))
    assert np.isfinite(out.Q_loss)
    # Wall/ground loss means unheated nodes may cool slightly, but the node
    # directly receiving the 85 degC inflow must warm, and nothing may
    # overshoot above the charge temperature.
    assert out.state.T_top > 60.0
    assert np.all(out.state.temperatures < 85.0)


# ---------------------------------------------------------------------------
# ThermalStorage1D.from_preset()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name,kwargs", [
    ("steel_tank_aboveground", dict(volume=500.0, height=15.0)),
    ("steel_tank_buried", dict(volume=500.0, height=15.0)),
    ("ptes", dict(r_bottom=20.0, r_top=30.0, height=10.0)),
])
def test_from_preset_matches_manual_construction(preset_name, kwargs):
    via_classmethod = ThermalStorage1D.from_preset(preset_name, **kwargs)
    config = getattr(StoragePresets, preset_name)(**kwargs)
    manual = ThermalStorage1D(config)

    assert via_classmethod.config.geometry.volume == pytest.approx(
        manual.config.geometry.volume
    )
    assert type(via_classmethod.config.loss_model) is type(manual.config.loss_model)

    state_a = via_classmethod.initialize(T_init=50.0)
    state_b = manual.initialize(T_init=50.0)
    out_a = via_classmethod.step(state_a, dt=60.0, inputs=StorageInputs())
    out_b = manual.step(state_b, dt=60.0, inputs=StorageInputs())
    assert np.allclose(out_a.state.temperatures, out_b.state.temperatures)


def test_from_preset_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown preset"):
        ThermalStorage1D.from_preset("not_a_real_preset", volume=1.0, height=1.0)
