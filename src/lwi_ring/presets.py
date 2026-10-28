"""Configuration presets that are set-up rules rather than driver code (pure dict transforms; nothing here runs).

* rescale_energy: move a case to another kinetic energy with the solenoid field times the rigidity ratio and the population times
  beta^2 gamma^3 (equal analytic space-charge strength), nothing else changed.
* FIELD_2X, FIELD_1P5X (applied with field_scaled): higher-field lattices re-matched to the working point: solenoid field times the
  factor, the arcs re-solved by the driver for the tune integer + fractional part, natural chromaticity -Q (baseline 0.135 T,
  Q = 4.38, 35 m straights).
* station_positions: one dipole station per column kept inside a sub-column for dyn_resets in {1, 2, 4, 7}.
"""
from __future__ import annotations

import math

MP_MEV = 938.27208816

FIELD_2X = dict(solenoid_bz_t=0.27, tune=8.38, tune_x=8.38, tune_y=8.38, proton_xi_x=-8.38, proton_xi_y=-8.38)
FIELD_1P5X = dict(solenoid_bz_t=0.2025, tune=6.38, tune_x=6.38, tune_y=6.38, proton_xi_x=-6.38, proton_xi_y=-6.38)
STATION_POS = {1: None, 2: [0.75], 4: [0.625], 7: [0.5]}


def gamma(t_mev: float) -> float:
    return 1 + t_mev / MP_MEV


def beta(t_mev: float) -> float:
    return math.sqrt(1 - gamma(t_mev) ** -2)


def beta2gamma3(t_mev: float) -> float:
    return beta(t_mev) ** 2 * gamma(t_mev) ** 3


def brho(t_mev: float) -> float:
    """Magnetic rigidity (T m) of a proton of kinetic energy t_mev."""
    return gamma(t_mev) * beta(t_mev) * MP_MEV * 1e6 / 299792458.0


def rescale_energy(cfg: dict, energy_mev: float) -> dict:
    """Move a configuration (dict) to another kinetic energy (see the module doc)."""
    e0 = cfg.get("energy_mev", 20.0)
    out = dict(cfg, energy_mev=energy_mev, population=cfg.get("population", 1e12) * beta2gamma3(energy_mev) / beta2gamma3(e0))
    if cfg.get("solenoid_bz_t"):
        out["solenoid_bz_t"] = cfg["solenoid_bz_t"] * brho(energy_mev) / brho(e0)
    return out


def field_scaled(cfg: dict, preset: dict) -> dict:
    """A field-scaled lattice re-matched to the working point (FIELD_2X or FIELD_1P5X) applied to a configuration (dict)."""
    return dict(cfg, **preset)


def station_positions(resets: int) -> dict:
    """dyn_resets with one dipole station per column inside a sub-column (rem_station_pos where needed)."""
    d = dict(dyn_resets=resets)
    if STATION_POS.get(resets) is not None:
        d["rem_station_pos"] = STATION_POS[resets]
    return d
