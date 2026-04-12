"""
Shared simulation parameters for all benchmark scripts.

All values match the FreeTTES reference configuration (config.json).
Changes here affect benchmark_model_variants.py and benchmark_scenarios.py.
"""

import numpy as np

# ── Storage geometry (FreeTTES config.json) ───────────────────────────────────

R_INNER    = 20.0                  # m
H_WS       = 39.9                  # m   (H_WS_max)
A_CROSS    = np.pi * R_INNER**2    # m²  ≈ 1256.6 m²
V_TANK     = A_CROSS * H_WS        # m³  ≈ 50 137 m³

U_WALL     = 0.2                   # W/(m²·K)  (U_Mantel)
T_AMB      = 10.0                  # °C        (ground temperature)

# ── Operating parameters ──────────────────────────────────────────────────────

T_CHARGE_IN  = 90.0                # °C  (charging supply temperature)
T_DISCH_IN   = 60.0                # °C  (discharging return temperature)
FLOW_M3H     = 300.0               # m³/h

# Derived mass flow rates (computed from FLOW_M3H and density)
# rho_water polynomial inlined so config_benchmark can be imported without shared_utils
def _rho(T: float) -> float:
    return -2.525726e-3 * T**2 - 2.123038e-1 * T + 1.005011e3

m_charge    = _rho(T_CHARGE_IN) * FLOW_M3H / 3600.0   # kg/s
m_discharge = _rho(T_DISCH_IN)  * FLOW_M3H / 3600.0   # kg/s

# ── Diffusor positions (FreeTTES config.json) ─────────────────────────────────

H_B_UK_DIF  = 0.5    # m  (config.json: H_B_UK_Dif – bottom edge of lower diffusor)
H_RS_DIF    = 1.0    # m  (config.json: H_RS_Dif   – diffusor zone height)
H_WS_OK_DIF = 0.5    # m  (config.json: H_WS_OK_Dif – zone below lid)

Z_LOWER_DIFF = H_B_UK_DIF + H_RS_DIF / 2     # = 1.0 m  (centre of lower diffusor)
Z_UPPER_DIFF = H_WS - H_WS_OK_DIF / 2        # = 39.65 m (centre of upper diffusor)

# ── Standard benchmark: 60-h scenario ────────────────────────────────────────

DT_S         = 3600    # timestep [s]
N_HOURS      = 60      # total duration [h]

T_END_CHARGE = 24      # h – end of charging phase
T_END_IDLE   = 36      # h – end of idle phase
T_END_DISCH  = 60      # h – end of discharging phase

# ── Initial temperature profile (identical to FreeTTES benchmark.py) ──────────
#
# FreeTTES internally adds two boundary points during initialisation
# (FreeTTES_model.py lines 991–1003) before linear interpolation:
#   z = 0.0 m  →  T_BODEN = 22.35 °C  (ground temperature, config.json)
#   z = H_WS   →  T_DR    = 99.0  °C  (headspace temperature, config.json)
# This creates a linear gradient in the bottom zone (z = 0–2 m) from
# 22.35 °C to 60 °C. We replicate these boundary points so that the
# initial profile of both models is identical.

T_BODEN = 22.35   # °C  (config.json: T_Boden – ground boundary value)
T_DR    = 99.0    # °C  (config.json: T_DR    – headspace boundary value)
#
# Note: FreeTTES resets T_DR each timestep as a fixed boundary condition
# (constant heat source at the topmost node). Our model can represent T_DR
# either as a fixed boundary or via the dynamic headspace model.

START_PROFILE_FREETTTES: dict[float, float] = {
     0.0: T_BODEN,   # ground boundary value (added internally by FreeTTES)
     2.0: 60.0,
     6.0: 60.0,
    10.0: 60.0,
    14.0: 60.0,
    18.0: 62.0,
    22.0: 85.0,
    26.0: 90.0,
    30.0: 90.0,
    34.0: 90.0,
    38.0: 90.0,
    H_WS: T_DR,      # headspace boundary value (added internally by FreeTTES)
}

# ── Energy reference ──────────────────────────────────────────────────────────

# FreeTTES result["E_nutz"] uses T_grenz = T_AMB = 10 °C as reference
# (not T_RL = 60 °C). For a fair comparison we also use T_AMB.
T_REF_ENUTZ = T_AMB   # °C

T_RL = 60.0   # °C  (minimum usable temperature – only for E_nutz_momentan)
