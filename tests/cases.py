"""Bit-identity cases: for each source copy, short cases (1-2 turns, small particle counts) that lwi_ring must reproduce bit for bit.

Every case is a configuration (the study base j1_dynK1_N2e14: 480 m, 12 cells of 35 m solenoid columns, dynamic electrons, lattice
errors, coasting longitudinal motion, or the smooth-ring defaults), an entry point and the source copy whose code made the reference:
  run        driver.run (ring_driver.run in the copy)
  many_turn  many_turn.run with the csrdrift_bitid.py settings (aperture 0.17 m, 2048 tracked, 16 samples per turn, checkpoint every turn)
  study      the study runner (lwi_ring.studyrun.run_case; the copy's scripts/<runner> for the reference) with its flags
Sizes: FAST (4096 per bunch, 32 x 32 x 64, 256 steps per turn, 1 turn) for the default suite; SLOW (8192 per bunch, 2 turns: the
studies' own bit-identity size, csrdrift_bitid.py) for the -m slow suite.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
J1 = json.loads((DATA / "configs/j1_dynK1_N2e14.json").read_text())

FAST = dict(particles_per_bunch=4096, grid=[32, 32, 64], steps_per_turn=256, tune_samples_per_turn=16, turns=1, backend="cpu")
SLOW = dict(particles_per_bunch=8192, grid=[32, 32, 64], steps_per_turn=256, tune_samples_per_turn=16, turns=2, backend="cpu")
SMOOTH_FAST = dict(particles_per_bunch=4096, grid=[32, 32, 64], steps_per_turn=128, tune_samples_per_turn=32, turns=1, backend="cpu")

# Config keys of j1 that a copy does not know (introduced later in the lineage): stripped for that copy's cases.
LATER = {"canonical": ("diag_bins", "diag_bin_centroids", "column_recentre_stations"),
         "dipole": ("diag_bin_centroids", "column_recentre_stations"),
         "stage3d": ("diag_bin_centroids", "column_recentre_stations"),
         "drift": ("column_recentre_stations",)}

# Source copies (relative to the study directory) and the runner scripts of the study cases.
SOURCES = {
    "canonical": "ring_driver",
    "dipole": "results_local/dipole/driver/ring_driver",
    "stage3d": "results_local/stage3d/driver/ring_driver",
    "drift": "results_local/drift/driver/ring_driver",
    "recentre": "results_local/recentre/driver/ring_driver",
    "recentre_dyn": "results_local/recentre_dyn/driver/ring_driver",
    "esize": "results_local/esize/driver/ring_driver",
    "m2": "results_local/m2/driver/ring_driver",
    "dyn400": "results_local/dyn400/driver/ring_driver",
    "remedy2": "results_local/remedy2/driver/ring_driver",
    "remedy": "results_local/remedy/driver/ring_driver",
    "dynprof": "results_local/dynprof/driver/ring_driver",
    "wallfb": "results_local/wallfb/driver/ring_driver",
}

RIGID = dict(column_electrons="rigid")
CENTROID = dict(RIGID, column_centroid_drift=True)
DYN0 = dict(column_recentre_stations=0, dyn_birth_sigma="fixed", dyn_aperture_kill=True)
FIELD_2X = dict(solenoid_bz_t=0.27, tune=8.38, tune_x=8.38, tune_y=8.38, proton_xi_x=-8.38, proton_xi_y=-8.38)
D400 = dict(energy_mev=400.0, solenoid_bz_t=0.3489961637143074, population=1774068726263879.2, dyn_aperture_kill=True)


def j1(source: str, *layers: dict, **overrides) -> dict:
    """The j1 base for a source copy, updated by `layers` in order, then by keyword overrides."""
    base = {k: v for k, v in J1.items() if k not in LATER.get(source, ())}
    for layer in layers:
        base.update(layer)
    base.update(overrides)
    return base


def smooth(**overrides) -> dict:
    return dict(overrides)


# name: (source, entry, config, runner script or None, runner flags)
CASES = {
    # ---- canonical ring_driver: defaults of every lwi_ring option ---------------------------------------------------------------
    "canon_smooth_hminus": ("canonical", "run", smooth(**SMOOTH_FAST), None, []),
    "canon_lens_long_mt": ("canonical", "many_turn", smooth(**SMOOTH_FAST, opposing="absent", lens_count=4, half_xy_m=0.18, gamma_t=5.0,
                                                              proton_delta_rms=1e-3, proton_xi_x=-2.0, proton_a_xx=0.01, error_dk_rms=0.01,
                                                              error_dipole_rms_rad=1e-4), None, []),
    "canon_rigid_mt": ("canonical", "many_turn", j1("canonical", **RIGID, **FAST), None, []),
    "canon_dyn": ("canonical", "run", j1("canonical", **FAST), None, []),
    # ---- dipole: transverse damper, coherent diagnostics, trim quadrupoles -----------------------------------------------------
    "dipole_damper": ("dipole", "run", j1("dipole", **RIGID, **FAST, damper_kappa=[0.02, 0.02], damper_noise_m=1e-4,
                                          error_trim_dk=[1e-3 * (k % 3 - 1) for k in range(24)]), None, []),
    "dipole_damper_mt": ("dipole", "many_turn", j1("dipole", **RIGID, **FAST, damper_kappa=[0.02, -0.01], damper_delay_turns=1,
                                                   damper_planes="x"), None, []),
    # ---- stage3d: painted injection and painted column profile -----------------------------------------------------------------
    "stage3d_painted": ("stage3d", "run", j1("stage3d", **RIGID, **FAST, injection_profile="painted", injection_edge_m=None,
                                             injection_painted_p=-0.5, column_profile="painted"), None, []),
    # ---- drift: drifting-centroid columns with resets ----------------------------------------------------------------------------
    "drift_resets_mt": ("drift", "many_turn", j1("drift", **CENTROID, **FAST, column_drift_resets=7), None, []),
    "drift_flux_run": ("drift", "run", j1("drift", **CENTROID, **FAST, column_drift_resets=2, column_sigma_flux=True,
                                          column_drift_bins_per_step=1, column_drift_smooth_m=1.0), None, []),
    # ---- recentre: re-centring stations, feed-forward, noise, delay ----------------------------------------------------------------
    "recentre_monitor": ("recentre", "run", j1("recentre", **CENTROID, **FAST, column_recentre_stations=1, column_recentre_noise_m=1e-4,
                                               column_recentre_offset_m=2e-4, column_recentre_target="proton"), None, []),
    "recentre_feedforward": ("recentre", "run", j1("recentre", **CENTROID, **FAST, column_recentre_stations=2,
                                                   column_recentre_mode="feedforward", column_recentre_noise_m=1e-4,
                                                   column_recentre_eta=0.8), None, []),
    "recentre_delay_mt": ("recentre", "many_turn", j1("recentre", **CENTROID, **FAST, column_recentre_stations=1,
                                                      column_recentre_delay_steps=2, column_recentre_side_kick=0.5), None, []),
    # ---- recentre_dyn (esize's dynamic lanes): dynamic electrons with a station -------------------------------------------------
    "recentre_dyn_K1": ("recentre_dyn", "study", j1("recentre_dyn", **FAST), "csrrcd_run.py", []),
    "recentre_dyn_flux": ("recentre_dyn", "run", j1("recentre_dyn", **FAST, column_sigma_flux=True, lens_sigma_ratio=0.9), None, []),
    # ---- esize: overlap drift rate with a fixed electron size --------------------------------------------------------------------
    "esize_overlap": ("esize", "run", j1("esize", **CENTROID, **FAST, column_drift_omega_overlap=True, lens_sigma_ratio=0.9,
                                         column_sigma_flux=True, column_recentre_stations=1), None, []),
    # ---- m2 (= dyn400 driver): resets, birth rules, AB2, aperture kill, shaping stations, m = 2 diagnostics ---------------------
    "m2_resets_ab2": ("m2", "run", j1("m2", **FAST, **DYN0, dyn_resets=2, dyn_gc_scheme="ab2", dyn_b_scale=1.5), None, []),
    "m2_shaping_diag": ("m2", "study", j1("m2", **FAST, column_recentre_stations=2, column_recentre_m2=1.0,
                                          column_recentre_m2_target="proton", column_recentre_dipole=0.5, dyn_birth_sigma="proton_cov",
                                          dyn_aperture_kill=True), "csrm2_run.py", []),
    "m2_paired": ("m2", "run", j1("m2", FAST, DYN0, dyn_birth_sigma="paired_fixed"), None, []),
    # ---- dyn400: the 400 MeV self-consistent set-up (d01_rc1_N18 at test size) -------------------------------------------------
    "dyn400_rc1": ("dyn400", "study", j1("dyn400", **FAST, **D400), "csrdyn400_run.py", []),
    # ---- remedy2: field-scaled lattice, sub-column stations, shaping, pickup delay, strain noise, early stop -----------------
    "remedy2_hf2_R2S2": ("remedy2", "study", j1("remedy2", **FAST, **FIELD_2X, dyn_aperture_kill=True, dyn_birth_sigma="fixed",
                                                dyn_resets=2, column_recentre_stations=2, rem_station_pos=[0.25, 0.75],
                                                column_recentre_m2=1.0, rem_delay_steps=1, rem_m2_noise=0.01),
                         "csrrm2_run.py", ["--stop-growth=1.0"]),
    # ---- remedy: gun-size tracking ---------------------------------------------------------------------------------------------
    "remedy_track": ("remedy", "study", j1("remedy", **FAST, dyn_aperture_kill=True, dyn_birth_sigma="fixed", dyn_resets=7,
                                           rem_station_pos=[0.5], rem_birth_track=True, rem_birth_ratio=1.05),
                     "csrrem_run.py", ["--stop-growth=1.0"]),
    # ---- dynprof: painted / flat-top / hollow electrons, painted protons, profile-aware match, shape diagnostic -------------
    "dynprof_painted": ("dynprof", "study", j1("dynprof", **FAST, dyn_aperture_kill=True, dyn_birth_sigma="fixed", dyn_resets=7,
                                               rem_station_pos=[0.5], injection_profile="painted", injection_edge_m=None,
                                               injection_painted_p=-0.5, dyn_profile="painted", dyn_painted_p=-0.5,
                                               dyn_match_profile=True), "csrdp_run.py", ["--stop-growth=1.0"]),
    "dynprof_hollow_ellipse": ("dynprof", "run", j1("dynprof", **FAST, **DYN0, dyn_profile="hollow", dyn_hollow_ratio=0.6,
                                                    dyn_ellipticity=0.1, dyn_ellipse_angle_deg=30.0, dyn_size_ratio=1.05), None, []),
    # ---- wallfb: segmented wall-electrode feedback ---------------------------------------------------------------------------
    "wallfb_noise": ("wallfb", "study", j1("wallfb", **FAST, **DYN0, wfb_enable=True, wfb_segment_m=4.0, wfb_start_turn=0,
                                           wfb_noise1_m=1e-4, wfb_noise2=0.01, wfb_bw_mhz=5.0, wfb_delay_ns=20.0, wfb_vmax_v=500.0),
                     "csrwr_run.py", []),
}

# The studies' own bit-identity size (csrdrift_bitid.py): one case per copy, -m slow.
SLOW_CASES = {}
for _name in ("canon_dyn", "canon_rigid_mt", "dipole_damper_mt", "stage3d_painted", "drift_resets_mt", "recentre_monitor",
              "recentre_dyn_K1", "esize_overlap", "m2_shaping_diag", "dyn400_rc1", "remedy2_hf2_R2S2", "remedy_track",
              "dynprof_painted", "wallfb_noise"):
    _src, _entry, _cfg, _script, _flags = CASES[_name]
    SLOW_CASES[_name + "_slow"] = (_src, _entry, dict(_cfg, **SLOW), _script, _flags)

ALL_CASES = {**CASES, **SLOW_CASES}
