"""Study-only strong–strong ring PIC; SI units, lab time, and diagnostic RR off.

The native deposit, gather, integrated-cell Green, and Boris kernels are used.
Only the longitudinal periodic convolution and external focusing live here.

lwi_ring: the consolidated driver (the study's ring_driver/ring_driver.py plus every feature of the study copies behind a Config
option; all defaults reproduce ring_driver/ring_driver.py bit for bit). See the package README for the option map.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy import constants, fft
from scipy.special import ndtri
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
# Output root (solver locks, default run directories, many_turn's output check): $LWI_RING_OUT if set, else results_local/ring_pic
# below the working directory. The driver copies used <copy>/../results_local/ring_pic; set_output_root() changes it at run time.
OUT = Path(os.environ.get("LWI_RING_OUT", str(Path.cwd() / "results_local/ring_pic"))).resolve()
# Native Numba decorators may create their cache directory during import.
# Keep that cache inside the permitted machine-local output tree as well.
os.environ.setdefault("NUMBA_CACHE_DIR", str(OUT / "numba_cache"))

from .longitudinal import Bucket, Longitudinal, matched_sample, synchrotron_tune
from . import injection
from core.pic.backend import PICBackend, select_backend
from core.pic import kernels as native_kernels
from core.pic.grid import Species
from core.pic.kernels import C, COULOMB, integrated_green

E = constants.elementary_charge
MP = constants.proton_mass
# H-minus mass includes two electrons and hydrogen + electron binding energies.
MH = MP + 2 * constants.electron_mass - 14.3522 * E / C**2


RECYCLE_DEFAULTS = dict(recycle_gun_every=0, recycle_rotate_deg=0.0, recycle_centroid_phase_deg=0.0, recycle_centroid_eta=0.0,
                        recycle_strain_eta=0.0, recycle_current_loss=0.0, recycle_scrape_sigma=0.0, recycle_window_m=4.0,
                        recycle_noise_m=0.0, recycle_seed=0)


@dataclass
class Config:
    energy_mev: float = 20.0
    circumference_m: float = 50.0
    population: float = 1e12
    eta: float | None = None
    harmonic: int = 4
    duty: float = 0.25
    sigma_m: float = 0.005
    tune: float = 2.3
    tune_x: float | None = None               # per-plane bare tunes; None = tune
    tune_y: float | None = None
    particles_per_bunch: int = 32768
    grid: tuple[int, int, int] = (96, 96, 256)
    half_xy_m: float | None = None
    outside_fraction_limit: float = 1e-4
    tune_shift_budget: float = 0.1
    images: int = 8
    shape_order: int = 2
    steps_per_turn: int = 512
    turns: int = 20
    slices: int = 5
    tune_samples_per_turn: int = 64
    seed: int = 710
    opposing: str = "charged"
    collective: bool = True
    backend: str = "cpu"
    wall_limit_s: float | None = None
    aperture_m: float | None = None           # many_turn absorbing aperture; None = --aperture-mm (30 mm)
    # Mitigation tools; every default reproduces the earlier behaviour exactly.
    hminus_refresh_turns: int | None = None   # many_turn only: fresh matched H- every k turns
    hminus_sigma_ratio: float = 1.0           # H- rms size / proton sigma_m (gaussian profile)
    hminus_tune: float | None = None          # H- bare tune; None = proton tune
    hminus_profile: str = "gaussian"          # or "hollow" (uniform annulus)
    hollow_inner_ratio: float = 1.5           # annulus radii in units of sigma_m
    hollow_outer_ratio: float = 3.0
    hollow_momentum: str = "circular"         # hollow H-: "circular" orbits or "thermal"
    lens_count: int | None = None             # weak-strong co-moving electron lenses
    lens_length_m: float = 0.1
    lens_fraction: float = 1.0                # f: fraction of proton self-field compensated
    lens_profile: str = "gaussian"            # or "hollow", same annulus radii
    lens_sigma_ratio: float = 1.0
    # Lattice errors: thin kicks at M = error_count (default 4 x error_superperiods) evenly spaced
    # locations, seeded and fixed per run. Normal gradient error: dx' = -g x, dy' = +g y with
    # g = dk * K * C/M, dk ~ N(0, error_dk_rms) a fraction of the smooth focusing K = (Q/R)^2
    # integrated over one location's share of the ring. Dipole error: dx', dy' ~ N(0, rms) rad.
    error_superperiods: int = 1
    error_count: int | None = None
    error_dk_rms: float = 0.0
    error_dipole_rms_rad: float = 0.0
    error_seed: int = 4242
    # Racetrack layout: two straights, two arcs, periodicity 2, laid out as straight A [0, Ls), arc,
    # straight B, arc. All None/default = smooth ring, bit for bit. The bare tunes stay the
    # whole-ring zero-current tunes; the arc focusing is solved so that they hold for the ratio.
    straight_length_m: float | None = None    # each of the two straights
    arc_length_m: float | None = None         # each of the two arcs; 2 (straight + arc) = circumference
    straight_focus_ratio: float = 1.0         # k_straight / k_arc, k = 2 pi Q_local / C wave number
    solenoid_bz_t: float = 0.0                # hard-edge axial field in the straights (protons only)
    solenoid_polarity: str = "alternating"    # "alternating" (+ in A, - in B) or "same"
    lens_region: str = "ring"                 # "ring" (evenly round the ring) or "straights"
    # Landau-damping knobs (per beam; all zero = earlier behaviour, bit for bit).
    # Distributed octupole-like detuning: dQ_x = a_xx Jx + a_xy Jy, dQ_y = a_xy Jx + a_yy Jy,
    # with J = (x^2 + (beta_bare x')^2)/(2 sigma_m^2), so <J> = 1 per plane for the matched
    # proton beam. A real octupole has a_yy = a_xx and a_xy = -2 a_xx. Signs are free.
    proton_a_xx: float = 0.0
    proton_a_xy: float = 0.0
    proton_a_yy: float = 0.0
    hminus_a_xx: float = 0.0
    hminus_a_xy: float = 0.0
    hminus_a_yy: float = 0.0
    # Frozen per-particle relative momentum offset delta (gaussian, exact rms, zero mean; no
    # synchrotron motion, no path-length slip, no dispersion). It only shifts the bare tune:
    # Q_i = Q0 + xi * delta_i (xi = dQ/d delta, absolute chromaticity, not Q0 (1 + xi delta)).
    proton_delta_rms: float = 0.0
    hminus_delta_rms: float = 0.0
    proton_xi_x: float = 0.0
    proton_xi_y: float = 0.0
    hminus_xi_x: float = 0.0
    hminus_xi_y: float = 0.0
    # Longitudinal dynamics of the proton beam (see longitudinal.py). All defaults = none, bit for
    # bit. Enabled by gamma_t (path-length slip, dynamic delta, chromatic tunes) and/or an RF voltage.
    # In this mode proton_delta_rms is the initial gaussian u_z spread of a coasting beam (RF off);
    # with RF the bunch is loaded matched to the bucket and proton_delta_rms must be 0. proton_xi_*
    # then act on the dynamic delta; the per-particle tune is Q0 + xi delta per ring turn of that
    # particle (transverse frequency scales with its arc speed). The H- beam stays at constant energy.
    gamma_t: float | None = None              # alpha_c = 1/gamma_t^2; eta = alpha_c - 1/gamma^2
    rf_voltage_v: float = 0.0                 # fundamental, volts per turn
    rf_harmonic: int | None = None            # None = harmonic (number of bunches)
    rf_phase_s_deg: float = 0.0               # synchronous-phase guess; solved from V(phi_s) = dE/turn / e
    rf_h2_ratio: int = 2                      # second harmonic number relative to the fundamental
    rf_h2_voltage_v: float = 0.0
    rf_h2_phase_deg: float = 0.0              # V = V1 sin(phi) + V2 sin(n2 phi + this)
    rf_dE_turn_ev: float | None = None        # reference energy gain per turn (ramp); None = stationary
    rf_fill: float = 0.9                      # matched load: truncation at this fraction of the separatrix W
    rf_rms_psi_rad: float | None = None       # target rms RF-phase length; None = 2 pi duty
    rf_mismatch_dp: float = 1.0               # scale the matched delta by this (1 = matched)
    rf_mismatch_psi: float = 1.0              # scale the matched psi by this
    rf_offset_psi_rad: float = 0.0            # displace the bunch in RF phase
    # ---- Stage-2 additions (compensated storage ring); every default reproduces the earlier drivers bit for bit.
    # (b) Periodic solenoid lattice: the racetrack generalised to P identical cells [straight, arc]
    # (straight_length_m, arc_length_m per cell; P (straight + arc) = circumference). P = 2 is the racetrack.
    # With alternating polarity the solenoid sign flips from cell to cell. solenoid_retune adds the Larmor
    # focusing K^2 (K = qB/2p) of the solenoid to the straights' focusing when the arcs are solved and the
    # beam is matched, so the ring-averaged tunes stay the configured ones; straight_focus_ratio = 0 then
    # gives a lattice focused by the solenoids alone (round, Q_x = Q_y).
    lattice_periods: int = 2
    solenoid_retune: bool = False
    # The legacy entry-edge treatment adds the fringe kick after the step's Boris rotation, so the kick is not
    # rotated through the rest of the step: a first-order (one-step) error, ~ K ds of the focusing. True rotates
    # the kick by the Larmor angle of the part of the step inside the field (second order). Default = legacy.
    solenoid_edge_rotation: bool = False
    # The legacy fringe treatment computes the edge kick from the pre-push coordinates and applies it, with a position correction,
    # after the Boris push: that map is not area preserving (det of the one-turn map 0.985 at 512 steps per turn for a 0.147 T
    # six-cell ring, 1 - det ~ steps^-2), so beams shrink by ~0.8% per turn per plane. True applies each edge kick inside the
    # half-drift that crosses the edge (drift to the edge, kick at the crossing position, drift on with the new slope): every
    # operation then preserves phase-space volume. Default = legacy, bit for bit.
    solenoid_edge_in_drift: bool = False
    # (a) Distributed co-moving electron columns (replace lens_count): column_sections is "straights" (every
    # straight of the periodic lattice), "ring" (the whole circumference), or [[z_start, z_end], ...] in
    # metres. lens_fraction keeps its ring-average meaning unless column_local_fraction (the compensation
    # inside the columns, 1 = the proton core self-force cancelled) is set. column_follow_orbit puts the
    # column axis on the zero-current closed orbit of the lattice (dipole errors, solenoid edges); the
    # profile rms follows the local matched beam size if column_sigma_follow. The line density of the
    # electrons always follows the proton macroparticles (rigid columns) or is carried by them (dynamic).
    column_sections: object = None
    column_local_fraction: float | None = None
    column_follow_orbit: bool = False
    column_sigma_follow: bool = False
    # Stage 3d: electron-column transverse shape with column_sigma_follow: "gaussian" (default), "waterbag" (parabolic, hard edge
    # sqrt(6) sigma_rms) or "painted" (the projection of the 4-D density (1 - rho^2/R^2)^p with p = injection_painted_p, R^2 = 2 (p + 3)
    # sigma_rms^2; p = 0 is the waterbag).
    column_profile: str = "gaussian"
    # (e) column_electrons = "dynamic": co-moving electron macroparticles, one per proton macroparticle, born at
    # each column entrance on the (closed-orbit) axis with the column profile and evolving self-consistently
    # in the PIC fields: E x B guiding-centre drift in the confining field column_b_t plus longitudinal
    # acceleration (see edyn_columns.py). Single-turn driver ring_driver.run only.
    column_electrons: str = "rigid"
    column_b_t: float = 0.2
    # (c) Injection distribution of the protons (4-D phase-space, stationary in a linear lattice):
    # "gaussian" (default), "truncated_gaussian" (Gaussian in the 4-D normalised amplitude, cut at
    # injection_cut times the underlying rms, so |x| <= cut sigma0), "waterbag" (uniform in the 4-D
    # ball: transverse density parabolic, hard edge sqrt(6) sigma). sigma_m is always the rms size.
    # injection_edge_m (truncated only) instead fixes the hard transverse edge radius and solves the cut.
    injection_profile: str = "gaussian"
    # Stage 3d: injection_profile "painted" is the 4-D density (1 - rho^2/R^2)^p of the normalised amplitude rho (p > -1; p = 0 the
    # waterbag, p -> -1 the Kapchinsky-Vladimirsky shell). rho^2/R^2 is Beta(2, 1 + p) distributed, R^2 = 2 (p + 3) sigma_m^2.
    injection_painted_p: float = -0.5
    injection_cut: float = 3.0
    injection_edge_m: float | None = None
    # Multi-turn painting model: the particle set is injected over injection_paint_turns turns. Each batch is
    # a small beam (rms injection_paint_sigma_ratio * sigma_m per normalised coordinate) displaced in
    # normalised x and y by an amplitude A_k = A_max sqrt((k + 1/2)/K) (equal area per batch: uniform disc), at
    # angle injection_paint_phase_deg (x) and + injection_paint_phase_deg + 90 (y) = an anti-correlated-style
    # corner; A_max is injection_paint_amax_sigma (in sigma_m). Unborn macroparticles carry zero weight.
    injection_paint_turns: int = 1
    injection_paint_amax_sigma: float = 0.0
    injection_paint_sigma_ratio: float = 0.2
    injection_paint_phase_deg: float = 0.0
    start_on_closed_orbit: bool = False       # displace the injected beam onto the closed orbit (xy and slopes)
    # ---- Stage-2b additions (compensated rapid-cycling synchrotron, see ramp.py and README); all inert while
    # ramp_final_energy_mev is None. The reference energy follows a programme from energy_mev to
    # ramp_final_energy_mev in ramp_time_s ("linear" in kinetic energy, "sine" = half-sine in momentum) and then
    # holds. The RF follows (rf_programme: constant | min_phase | table), the smooth focusing is ramped with the
    # rigidity (constant k), the DC solenoid's Larmor focusing is made up by ramped quadrupoles in the straights
    # (ramp_solenoid_follow = 0) or the solenoid field is ramped as well (1). The columns follow the proton
    # velocity and line density with a compensation programme f(t) (constant | taper | table; taper: f =
    # clamp(1 - N_unc(E)/N, 0, f_max)) and a programmed transverse size (ramp_column_sigma).
    ramp_final_energy_mev: float | None = None
    ramp_time_s: float = 0.0
    ramp_shape: str = "sine"
    rf_programme: str = "constant"
    rf_phase_max_deg: float = 35.0            # min_phase: raise V so that the synchronous phase stays below this
    rf_voltage_table: list | None = None      # [[t_s, scale of rf_voltage_v and rf_h2_voltage_v], ...]
    ramp_comp_programme: str = "constant"
    ramp_comp_table: list | None = None       # [[t_s, f], ...]
    ramp_comp_n_unc0: float | None = None     # N_unc at the injection energy (taper)
    ramp_comp_n_design: float | None = None   # N in the taper formula; None = circulating population
    ramp_comp_scaling: str = "adiabatic"      # N_unc(E): "adiabatic" beta gamma^2 (norm. emittance fixed) | "fixed_size" beta^2 gamma^3
    ramp_comp_f_max: float = 1.0
    ramp_comp_ring_average: bool = False      # the programme f is the ring-average fraction; the columns' local fraction is f C/L_columns
    ramp_column_sigma: str = "fixed"          # column rms: fixed | adiabatic (~ (beta gamma)^-1/2) | measured (proton rms)
    ramp_electron_beta_error: float = 0.0     # electron velocity relative to the proton velocity
    ramp_solenoid_follow: float = 0.0
    ramp_quad_scale: float = 1.0              # scale of the quadrupole make-up (1 = tunes held exactly; test knob)
    ramp_regrid_tol: float = 0.01             # rebuild the solver's Green function when gamma moved by this fraction
    # csr-ramp additions (2026-10-27; inert at their defaults): ramp_shape "dual" adds a second magnet harmonic,
    # b(s) = [(1 - cos pi s) + (alpha/4)(1 - cos 2 pi s)]/2 of the momentum swing (start (1 + alpha) times steeper, same end points);
    # ramp_shape "table": kinetic energy interpolated in ramp_energy_table (e.g. a real cycle mapped onto the twin);
    # dynamic (self-consistent) electrons follow the ramp: birth weight f C/L_columns, confining field x the solenoid scale,
    # fixed gun size x the ramp_column_sigma scale, station side kick at the present rigidity.
    ramp_dual_alpha: float = 0.0
    ramp_energy_table: list | None = None     # ramp_shape "table": [[t_s, kinetic energy MeV], ...] (e.g. a real cycle mapped onto the twin)
    # ---- Dipole-mode mitigation additions (2026-10-22, damper.py); all inert at their defaults.
    # Transverse feedback damper: a slice-by-slice (damper_slots azimuthal slices) proportional damper. The pickup in the middle of
    # arc damper_pickup_arc reads the mean position of each slice (minus the closed orbit, plus damper_noise_m rms noise); the kicker in the
    # middle of arc damper_arc gives the same slice, damper_delay_turns (>= 0 whole turns; 0 needs the kicker downstream of the pickup)
    # later, the slope kick dx' = kappa x_read (kappa = damper_kappa[plane] in rad/m; kappa > 0 damps when the pickup-to-kicker
    # coherent phase advance is between 0 and pi). damper_planes selects the planes.
    damper_kappa: list | None = None
    damper_pickup_arc: int = 0
    damper_arc: int = 1
    damper_delay_turns: int = 0
    damper_slots: int = 64
    damper_noise_m: float = 0.0
    damper_planes: str = "xy"
    # diag_bins > 0 adds a "coherent" entry to every turn record (lab-bin centroid amplitude, decohered emittance).
    diag_bins: int = 0
    # error_trim_dk: extra relative gradient errors (same convention as error_dk_rms) added to the lattice-error locations (trim quadrupoles).
    error_trim_dk: list | None = None
    # ---- Column-drift additions (2026-10-23, report_column_drift.md); all inert at their defaults.
    # column_centroid_drift: each (rigid) column carries a per-slice electron centroid e (complex x + i y, relative to the column axis),
    # born on the axis at the column entrance and E x B-drifting about the local proton slice centroid u as de/ds = i sigma Omega (u - e)
    # (sigma the sign of the solenoid field along the beam). Omega = lambda_p (nbar - 1/(pi b^2)) / (2 eps0 |B| beta gamma^2 c) per lab
    # metre: lambda_p the local proton line density, nbar = int rho^2 dA of the column profile (protons matched to it), b =
    # column_drift_bore_m (the image of the displaced protons), times column_drift_omega_scale (a field-scaling proxy). The column kick is
    # evaluated about e. Slices are lab bins of C/(steps_per_turn column_drift_bins_per_step) advected with the electrons; the proton slice
    # centroid is smoothed with a Gaussian of column_drift_smooth_m. column_drift_resets > 1 gives fresh electrons that many times per
    # column (equal sub-columns, gaps neglected). column_sigma_flux fixes the column profile rms within each (sub-)column at its value at
    # the entrance (the gun and r^2 B conservation) instead of following the local matched size (needs column_sigma_follow).
    column_centroid_drift: bool = False
    column_drift_bore_m: float = 0.15
    column_drift_omega_scale: float = 1.0
    column_drift_resets: int = 1
    column_drift_bins_per_step: int = 2
    column_drift_smooth_m: float = 2.0
    column_sigma_flux: bool = False
    # ---- Electron-size tolerance (2026-10-24, report_electron_size_tolerance.md); inert at the default.
    # column_drift_omega_overlap: the drift rate uses the overlap density int rho_e rho_p dA of the (gaussian) column profile with a
    # gaussian proton profile of the local matched rms (sigma_at without lens_sigma_ratio and flux), instead of the column's own
    # int rho_e^2 dA (which assumes protons matched to the column). The two are equal for lens_sigma_ratio = 1 without column_sigma_flux.
    column_drift_omega_overlap: bool = False
    # ---- Electron re-centring (2026-10-24, report_electron_recentring.md); inert at column_recentre_stations = 0.
    # K = column_recentre_stations stations at L k/(K+1), k = 1..K, inside every (sub-)column. At a station the electron centroid of each
    # slice is moved by d = -eta (m - t): m = e + n (electron monitor, n complex Gaussian noise of rms column_recentre_noise_m per plane and
    # slice) or, for column_recentre_mode = "feedforward", the model value eh, which drifts like e but about the proton slice centroid h
    # measured (with the same noise) at the last gun or station; t = the closed orbit plus a static per-station offset (rms
    # column_recentre_offset_m), or the proton slice centroid u for column_recentre_target = "proton". The protons of the slice receive the
    # side kick of the static field that moves the electrons, x' + i y' += s i sigma (B/Brho) d with s = column_recentre_side_kick
    # (0: a kick-free corrector pair). eta = 1 without noise, offset or side kick equals column_drift_resets x (K+1) resets.
    # column_recentre_delay_steps J > 0: the monitor reading is the same slice's e from J steps earlier (a pickup J step paths upstream,
    # or, with J = steps_per_turn, the same station one turn earlier); monitor mode only.
    column_recentre_stations: int = 0
    column_recentre_eta: float = 1.0
    column_recentre_noise_m: float = 0.0
    column_recentre_offset_m: float = 0.0
    column_recentre_mode: str = "monitor"
    column_recentre_target: str = "orbit"
    column_recentre_side_kick: float = 1.0
    column_recentre_seed: int = 0
    column_recentre_delay_steps: int = 0
    # ---- Electron m = 2 / tune-chaos study (2026-10-24, report_electron_m2_and_tune_chaos.md); all inert at their defaults, and all
    # (except none) for column_electrons = "dynamic" only (edyn_columns.py).
    # dyn_birth_sigma: rms of the electrons at birth: "follow" (default: the local matched size, ElectronLens.sigma_at), "fixed" (the gun's
    # fixed size sigma_m * lens_sigma_ratio at every birth), "proton" (the measured rms of the protons entering with them, round: gun size
    # control) or "proton_cov" (their full transverse covariance: size and ellipse); "paired" puts each electron at its own proton's offset
    # from the centroid of the protons entering with it (no independent electron shot noise: a low-noise numerical control), "paired_fixed"
    # the same pattern scaled to the gun's fixed rms.
    dyn_birth_sigma: str = "follow"
    # dyn_resets > 1: fresh electrons that many times per column (equal sub-columns, gaps neglected; no stations).
    dyn_resets: int = 1
    # dyn_b_scale scales the field in the electrons' E x B drift only (a column-field proxy; the proton focusing is unchanged).
    dyn_b_scale: float = 1.0
    # dyn_gc_scheme: "euler" (default; field at the start-of-step electron positions) or "ab2" (second-order Adams-Bashforth in the drift
    # velocity, same field solves).
    dyn_gc_scheme: str = "euler"
    # dyn_aperture_kill: protons farther than aperture_m from the column axis (closed orbit) are removed (weight zero, parked on the axis)
    # with their electrons, and counted as lost, instead of ending the run when they leave the field grid.
    dyn_aperture_kill: bool = False
    # column_recentre_m2 > 0 (stations, dynamic electrons): the station fit also takes a strain term q conj(b) (the electron ellipse
    # distortion since birth) and removes column_recentre_m2 * (q - q_t) by an electrostatic quadrupole-like shaping field; q_t = 0 for
    # column_recentre_m2_target = "birth", or the strain that makes the electron ellipse equal the protons' for "proton". The paired protons
    # get the field's own kick x' + i y' += s i sigma (B/Brho) dq conj(b_p), the m = 2 counterpart of the side kick.
    # column_recentre_dipole = 0 switches the dipole re-centring off while keeping the station (m = 2 only).
    column_recentre_m2: float = 0.0
    column_recentre_m2_target: str = "birth"
    column_recentre_m2_kick: float = 1.0       # scale of the shaping field's kick on the protons (1 = the field's own kick; 0 = test only)
    column_recentre_dipole: float = 1.0
    # ---- m = 2 remedy study (2026-10-25, report_m2_remedies.md; dynamic electrons with stations only; inert at the defaults).
    # rem_station_pos: station positions as fractions of the column length (K = column_recentre_stations values in (0, 1), increasing),
    # instead of L k/(K+1); with dyn_resets > 1 (allowed with stations in this copy) each station must lie inside a sub-column.
    # rem_delay_steps J > 0: every station acts on the fit of the same electrons taken J step paths upstream (a pickup J * C/steps_per_turn
    # before the station, inside the same (sub-)column), the counterpart of column_recentre_delay_steps of the centroid model.
    # rem_m2_noise: complex Gaussian noise (rms per component) added to the fitted strain q at every station fit (electron-ellipse
    # monitor resolution; the measured ellipticity |Q|/V of a round birth is 2|q|).
    rem_station_pos: list | None = None
    rem_delay_steps: int = 0
    rem_m2_noise: float = 0.0
    # rem_birth_ratio: electron/proton birth size ratio for dyn_birth_sigma = "fixed" (the gun size times this factor); with
    # rem_birth_track the gun size follows the protons slowly: once every steps_per_turn column updates it is set to rem_birth_ratio times
    # the rms (per plane, about the column axis) of all live protons inside the columns (a programmed gun field ratio, not a fast loop).
    rem_birth_ratio: float = 1.0
    rem_birth_track: bool = False
    # ---- Realistic electron profiles (2026-10-25, report_dynamic_profiles.md; dynamic electrons only; inert at the defaults).
    # dyn_profile: transverse birth profile of the electrons: "gaussian" (default, the truncated Gaussian draw), "painted" (the projection
    # of the 4-D density (1 - rho^2/R^2)^p of injection_profile "painted", p = dyn_painted_p: radial density (1 - r^2/R^2)^(p + 1),
    # R^2 = 2 (p + 3) rms^2), "flattop" (uniform disc, R = 2 rms: the KV-like column) or "hollow" (uniform annulus, inner/outer radius
    # dyn_hollow_ratio, outer radius 2 rms / sqrt(1 + ratio^2)). The rms per plane is set by dyn_birth_sigma exactly as for the Gaussian
    # (local matched size, the gun's fixed size, or the measured proton size / covariance at every (sub-)column entrance), times
    # dyn_size_ratio. dyn_ellipticity e in [0, 1): a birth ellipse with rms^2 (1 + e) and rms^2 (1 - e) along axes turned by
    # dyn_ellipse_angle_deg from x, y (not with proton_cov, which takes the protons' ellipse). dyn_match_profile: the initial-beam envelope
    # match takes the column add-back of the electron profile (at the gun's size) instead of the Gaussian column's.
    dyn_profile: str = "gaussian"
    dyn_painted_p: float = -0.5
    dyn_hollow_ratio: float = 0.5
    dyn_size_ratio: float = 1.0
    dyn_ellipticity: float = 0.0
    dyn_ellipse_angle_deg: float = 0.0
    dyn_match_profile: bool = False
    # ---- Recycled electrons (2026-10-27, csr-recycle, report_recycled_electrons.md; dynamic electrons on the straights of a periodic
    # lattice; inert unless recycle_stride > 0). The electrons leaving column k are not collected but carried through a proton-free
    # transport (path length equal to the ring's, so they meet the same proton slice) and re-injected into column k + recycle_stride
    # (2: the leapfrog of two interleaved streams). With dyn_resets = R > 1 the units are the R sub-columns of each column. Fresh (gun) electrons only at a stream start: the first recycle_stride columns of
    # the turn and, if recycle_gun_every = M > 0, every M-th column of each stream; a stream ends at a collector after its last column
    # (the turn's last columns, or before the next gun). The transport map acts on the stored exit offsets from the column axis, per
    # electron slice: centroid m and covariance S of the exiting electrons within +-recycle_window_m/2 of the slice (by exit time),
    # then r = offset - m is reshaped by the strain cleaning S -> (1 - recycle_strain_eta) S + recycle_strain_eta tr(S)/2 I (linear
    # map S_t^1/2 S^-1/2), turned by recycle_rotate_deg about the slice centroid (the ellipse rotation of the transport: the self-field
    # rotation is a positive angle for the same field sign as the column), and the centroid becomes e^(i phi) m - recycle_centroid_eta
    # (e^(i phi) m + n), phi = recycle_centroid_phase_deg (l = 1 wall precession), n complex Gaussian monitor noise of rms
    # recycle_noise_m per plane (seed recycle_seed). Electrons beyond recycle_scrape_sigma slice rms (per plane) from the slice centre are
    # scraped (0 = none); each transport keeps a fraction 1 - recycle_current_loss of the stream's charge.
    recycle_stride: int = 0
    recycle_gun_every: int = 0
    recycle_rotate_deg: float = 0.0
    recycle_centroid_phase_deg: float = 0.0
    recycle_centroid_eta: float = 0.0
    recycle_strain_eta: float = 0.0
    recycle_current_loss: float = 0.0
    recycle_scrape_sigma: float = 0.0
    recycle_window_m: float = 4.0
    recycle_noise_m: float = 0.0
    recycle_seed: int = 0
    # diag_bin_centroids adds the bin centroids (x, y per diag_bins bin, metres from the closed orbit) to the "coherent" record.
    diag_bin_centroids: bool = False
    # ---- Segmented wall-electrode feedback (2026-10-26, csr-wallring, report_wall_feedback_ring.md; dynamic electrons on the straights of a
    # periodic lattice; inert unless wfb_enable). See wallfb.py: nseg = round(L / wfb_segment_m) segments per column, wfb_sectors azimuthal
    # electrodes in a bore of wfb_bore_m, each segment driven from the net (proton + electron) l = 1, 2 wall signal of a pickup
    # wfb_pickup_offset segments upstream (slices of wfb_bin_m in the co-moving frame), hold law with gains wfb_gain1/2, orientation
    # errors wfb_phase1/2_deg and profile factors wfb_kappa1/2 (1 = Gaussian), timing error wfb_delay_ns, one-pole bandwidth wfb_bw_mhz
    # (0 = unlimited; a Gaussian slice filter in quadrature with wfb_smooth_m, the PIC's longitudinal resolution, -1 = 0.707 lab cells),
    # slow baseline subtraction over wfb_baseline_turns (0 = none), loop
    # closed from wfb_start_turn, pickup noise wfb_noise1_m (m) / wfb_noise2 (ellipticity) at the bunch-peak line density, amplifier
    # limit wfb_vmax_v (peak sector volts, 0 = none), wfb_sweep (a segment carries one voltage at a time), and the diagnostics
    # wfb_proton_kick / wfb_electron_drive (electrode field on the protons / electrons).
    wfb_enable: bool = False
    wfb_segment_m: float = 2.0
    wfb_sectors: int = 16
    wfb_bore_m: float = 0.15
    wfb_pickup_offset: int = 1
    wfb_bin_m: float = 0.25
    wfb_gain1: float = 1.0
    wfb_gain2: float = 1.0
    wfb_phase1_deg: float = 0.0
    wfb_phase2_deg: float = 0.0
    wfb_kappa1: float = 1.0
    wfb_kappa2: float = 1.0
    wfb_delay_ns: float = 0.0
    wfb_bw_mhz: float = 0.0
    wfb_smooth_m: float = -1.0
    wfb_baseline_turns: float = 20.0
    wfb_start_turn: int = 1
    wfb_noise1_m: float = 0.0
    wfb_noise2: float = 0.0
    wfb_vmax_v: float = 0.0
    wfb_sweep: bool = True
    wfb_proton_kick: bool = True
    wfb_electron_drive: bool = True
    wfb_seed: int = 0

    def ramp_enabled(self) -> bool:
        return self.ramp_final_energy_mev is not None

    def long_enabled(self) -> bool:
        return self.gamma_t is not None or self.rf_voltage_v > 0

    def columns_enabled(self) -> bool:
        return self.column_sections is not None

    def lens_enabled(self) -> bool:
        return bool(self.lens_count) or self.columns_enabled()

    def electron_resets(self) -> int:
        """Fresh electron beams per column of the dynamic model: dyn_resets, or its alias column_drift_resets."""
        return int(self.dyn_resets if self.dyn_resets != 1 else self.column_drift_resets)

    def station_delay_steps(self) -> int:
        """Pickup delay (step paths upstream) of the dynamic model's stations: rem_delay_steps, or its alias column_recentre_delay_steps."""
        return int(self.rem_delay_steps or self.column_recentre_delay_steps)

    def plane_tunes(self, role: str = "proton") -> np.ndarray:
        """Bare (x, y) tunes of a beam; hminus_tune, if set, applies to both H- planes."""
        if role == "hminus" and self.hminus_tune is not None:
            return np.array([self.hminus_tune, self.hminus_tune])
        return np.array([self.tune if self.tune_x is None else self.tune_x,
                         self.tune if self.tune_y is None else self.tune_y])

    def error_locations(self) -> int:
        return self.error_count or 4 * self.error_superperiods

    def errors_enabled(self) -> bool:
        return bool(self.error_dk_rms or self.error_dipole_rms_rad)

    def racetrack(self) -> bool:
        return self.straight_length_m is not None or self.arc_length_m is not None

    def half_extent(self) -> float:
        """Transverse field half-width: 6 sigma unless the H- shape needs more."""
        if self.half_xy_m is not None:
            return self.half_xy_m
        scale = 1.0
        if self.opposing != "absent":
            if self.hminus_profile == "hollow":
                scale = max(1.0, 1.5 * self.hollow_outer_ratio / 6)
            else:
                scale = max(1.0, self.hminus_sigma_ratio)
        return 6 * self.sigma_m * scale

    def validate_stage2(self) -> None:
        """Checks of the column, injection and painting options (all inert at their defaults)."""
        if self.columns_enabled():
            sec = self.column_sections
            if self.opposing != "absent":
                raise ValueError("columns replace the H- beam: set opposing=absent")
            if isinstance(sec, str):
                if sec not in ("straights", "ring") or (sec == "straights") != self.racetrack():
                    raise ValueError("column_sections: 'straights' on a periodic lattice, 'ring' or a list on a smooth ring")
            else:
                if not sec or any(len(x) != 2 or not 0 <= x[0] < x[1] <= self.circumference_m for x in sec):
                    raise ValueError("column_sections list needs [start, end] with 0 <= start < end <= circumference")
                ordered = sorted(map(tuple, sec))
                if any(a[1] > b[0] + 1e-12 for a, b in zip(ordered, ordered[1:])):
                    raise ValueError("column sections overlap")
                if self.racetrack():
                    raise ValueError("on a periodic lattice use column_sections 'straights' or 'ring'")
            if self.column_local_fraction is not None and not 0 < self.column_local_fraction <= 1.5:
                raise ValueError("column_local_fraction must be in (0, 1.5]")
            if self.column_electrons not in ("rigid", "dynamic"):
                raise ValueError("column_electrons: rigid or dynamic")
            if self.column_profile not in ("gaussian", "waterbag", "painted"):
                raise ValueError("column_profile: gaussian, waterbag or painted")
            if self.column_profile != "gaussian" and not self.column_sigma_follow:
                raise ValueError("column_profile waterbag or painted needs column_sigma_follow")
            if self.column_electrons == "dynamic" and (not math.isfinite(self.column_b_t) or self.column_b_t <= 0):
                raise ValueError("dynamic electrons need column_b_t > 0")
            if self.column_centroid_drift:
                if self.column_electrons != "rigid" or self.ramp_enabled():
                    raise ValueError("column_centroid_drift needs rigid columns and no ramp")
                if not (self.racetrack() and self.solenoid_bz_t and self.column_sections == "straights"):
                    raise ValueError("column_centroid_drift needs solenoid columns on the straights of a periodic lattice")
                if self.column_drift_bins_per_step < 1 or self.column_drift_resets < 1 or self.column_drift_bore_m <= 0 \
                        or self.column_drift_smooth_m < 0 or self.column_drift_omega_scale < 0:
                    raise ValueError("column_drift_*: bins_per_step, resets >= 1; bore > 0; smooth, omega_scale >= 0")
            if self.column_recentre_stations:
                if not (self.column_centroid_drift or self.column_electrons == "dynamic") or self.column_recentre_stations < 0:
                    raise ValueError("column_recentre_stations needs column_centroid_drift or dynamic electrons and must be >= 0")
                if self.column_recentre_mode not in ("monitor", "feedforward") or self.column_recentre_target not in ("orbit", "proton") \
                        or self.column_recentre_noise_m < 0 or self.column_recentre_offset_m < 0 or self.column_recentre_delay_steps < 0 \
                        or (self.column_recentre_delay_steps and self.column_recentre_mode != "monitor"):
                    raise ValueError("column_recentre_*: mode monitor|feedforward, target orbit|proton, noise and offset >= 0")
            if self.dyn_birth_sigma != "follow" or self.dyn_resets != 1 or self.dyn_b_scale != 1.0 or self.dyn_gc_scheme != "euler" \
                    or self.dyn_aperture_kill or self.column_recentre_m2 or self.column_recentre_dipole != 1.0:
                if self.column_electrons != "dynamic":
                    raise ValueError("dyn_* and column_recentre_m2/dipole options need column_electrons = dynamic")
                if self.dyn_birth_sigma not in ("follow", "fixed", "proton", "proton_cov", "paired", "paired_fixed") or self.dyn_resets < 1 or self.dyn_b_scale <= 0 \
                        or self.dyn_gc_scheme not in ("euler", "ab2") or self.column_recentre_m2 < 0 \
                        or self.column_recentre_m2_target not in ("birth", "proton"):
                    raise ValueError("dyn_birth_sigma follow|fixed|proton|proton_cov|paired|paired_fixed, dyn_resets >= 1, dyn_b_scale > 0, dyn_gc_scheme euler|ab2, "
                                     "column_recentre_m2 >= 0, column_recentre_m2_target birth|proton")
                if (self.column_recentre_m2 or self.column_recentre_dipole != 1.0) and not self.column_recentre_stations:
                    raise ValueError("column_recentre_m2 and column_recentre_dipole need column_recentre_stations")
            if self.column_electrons == "dynamic":
                # lwi_ring: the centroid-model names column_drift_resets / column_recentre_delay_steps also act on dynamic electrons
                # (aliases of dyn_resets / rem_delay_steps; the copies ignored them there). Both set and different is an error.
                if self.dyn_resets != 1 and self.column_drift_resets != 1 and self.dyn_resets != self.column_drift_resets:
                    raise ValueError("dyn_resets and column_drift_resets are the same option for dynamic electrons: set one")
                if self.rem_delay_steps and self.column_recentre_delay_steps and self.rem_delay_steps != self.column_recentre_delay_steps:
                    raise ValueError("rem_delay_steps and column_recentre_delay_steps are the same option for dynamic electrons: set one")
                if self.column_drift_resets < 1:
                    raise ValueError("column_drift_resets must be >= 1")
                if self.electron_resets() > 1 and self.column_recentre_stations and self.rem_station_pos is None:
                    raise ValueError("dyn_resets > 1 with re-centring stations needs rem_station_pos (each station inside a sub-column)")
            if (self.rem_birth_ratio != 1.0 or self.rem_birth_track) and (self.column_electrons != "dynamic" or self.dyn_birth_sigma != "fixed"
                                                                        or self.rem_birth_ratio <= 0):
                raise ValueError("rem_birth_ratio / rem_birth_track need dynamic electrons with dyn_birth_sigma = fixed, ratio > 0")
            if self.rem_station_pos is not None or self.rem_delay_steps or self.rem_m2_noise:
                if self.column_electrons != "dynamic" or not self.column_recentre_stations:
                    raise ValueError("rem_* options need dynamic electrons with column_recentre_stations")
                if self.rem_delay_steps < 0 or self.rem_m2_noise < 0:
                    raise ValueError("rem_delay_steps and rem_m2_noise must be >= 0")
                if self.rem_station_pos is not None:
                    f = list(self.rem_station_pos)
                    if len(f) != self.column_recentre_stations or any(not 0 < x < 1 for x in f) or any(b <= a for a, b in zip(f, f[1:])):
                        raise ValueError("rem_station_pos: column_recentre_stations increasing fractions in (0, 1)")
            if self.dyn_profile != "gaussian" or self.dyn_size_ratio != 1.0 or self.dyn_ellipticity or self.dyn_match_profile:
                if self.column_electrons != "dynamic":
                    raise ValueError("dyn_profile, dyn_size_ratio, dyn_ellipticity and dyn_match_profile need column_electrons = dynamic")
                if self.dyn_profile not in ("gaussian", "painted", "flattop", "hollow") \
                        or not (math.isfinite(self.dyn_painted_p) and self.dyn_painted_p > -0.99) \
                        or not 0 <= self.dyn_hollow_ratio < 0.95 or not (math.isfinite(self.dyn_size_ratio) and self.dyn_size_ratio > 0) \
                        or not 0 <= self.dyn_ellipticity < 1:
                    raise ValueError("dyn_profile gaussian|painted|flattop|hollow, dyn_painted_p > -0.99, 0 <= dyn_hollow_ratio < 0.95, "
                                     "dyn_size_ratio > 0, 0 <= dyn_ellipticity < 1")
                if self.dyn_birth_sigma in ("paired", "paired_fixed") and (self.dyn_profile != "gaussian" or self.dyn_ellipticity):
                    raise ValueError("paired electron births copy the protons' pattern: no dyn_profile or dyn_ellipticity")
                if self.dyn_birth_sigma == "proton_cov" and self.dyn_ellipticity:
                    raise ValueError("dyn_ellipticity cannot be combined with dyn_birth_sigma = proton_cov")
            if self.recycle_stride:
                if self.column_electrons != "dynamic" or self.column_sections != "straights" or not self.racetrack() \
                        or self.ramp_enabled():
                    raise ValueError("recycle_stride needs dynamic electrons on the straights of a periodic lattice, without a ramp")
                if not (0 < self.recycle_stride < (self.lattice_periods or 0) * self.electron_resets() and self.recycle_gun_every >= 0
                        and 0 <= self.recycle_centroid_eta <= 1 and 0 <= self.recycle_strain_eta <= 1
                        and 0 <= self.recycle_current_loss < 1 and self.recycle_scrape_sigma >= 0 and self.recycle_window_m > 0
                        and self.recycle_noise_m >= 0
                        and all(math.isfinite(v) for v in (self.recycle_rotate_deg, self.recycle_centroid_phase_deg))):
                    raise ValueError("recycle_*: 0 < stride < lattice_periods x resets, gun_every >= 0, centroid/strain eta in [0, 1], "
                                     "0 <= current_loss < 1, scrape_sigma >= 0, window > 0, noise >= 0, finite angles")
            elif any(getattr(self, k) != v for k, v in RECYCLE_DEFAULTS.items()):
                raise ValueError("recycle_* options need recycle_stride > 0")
            if self.wfb_enable:
                if self.column_electrons != "dynamic" or self.column_sections != "straights" or not self.racetrack() or self.ramp_enabled():
                    raise ValueError("wfb_enable needs dynamic electrons on the straights of a periodic lattice, without a ramp")
                if not (self.wfb_segment_m > 0 and self.wfb_sectors >= 3 and self.wfb_bore_m > 0 and self.wfb_pickup_offset >= 1
                        and self.wfb_bin_m > 0 and self.wfb_bw_mhz >= 0 and self.wfb_baseline_turns >= 0 and self.wfb_start_turn >= 0
                        and self.wfb_noise1_m >= 0 and self.wfb_noise2 >= 0 and self.wfb_vmax_v >= 0
                        and all(math.isfinite(v) for v in (self.wfb_gain1, self.wfb_gain2, self.wfb_phase1_deg, self.wfb_phase2_deg,
                                                           self.wfb_kappa1, self.wfb_kappa2, self.wfb_delay_ns))):
                    raise ValueError("wfb_*: segment, bore, bin > 0; sectors >= 3; pickup offset >= 1; bandwidth, baseline, start, noise, "
                                     "vmax >= 0; finite gains, phases, kappas and delay")
            if self.column_drift_omega_overlap and not (self.column_centroid_drift and self.column_sigma_follow
                                                         and self.column_profile == "gaussian"):
                raise ValueError("column_drift_omega_overlap needs column_centroid_drift, column_sigma_follow and gaussian columns")
            if self.column_sigma_flux and not (self.column_sigma_follow and self.column_sections == "straights"):
                raise ValueError("column_sigma_flux needs column_sigma_follow on the straights")
        elif self.column_follow_orbit or self.column_sigma_follow or self.column_electrons != "rigid" \
                or self.column_local_fraction is not None or self.column_centroid_drift or self.column_sigma_flux or self.wfb_enable \
                or self.column_drift_omega_overlap or self.recycle_stride:
            raise ValueError("column options need column_sections")
        if self.injection_profile not in ("gaussian", "truncated_gaussian", "waterbag", "painted"):
            raise ValueError("injection_profile: gaussian, truncated_gaussian, waterbag or painted")
        if self.injection_profile == "painted" and not (math.isfinite(self.injection_painted_p) and self.injection_painted_p > -0.99):
            raise ValueError("injection_painted_p must exceed -0.99")
        if self.injection_profile != "truncated_gaussian" and self.injection_edge_m is not None:
            raise ValueError("injection_edge_m belongs to the truncated Gaussian")
        if not math.isfinite(self.injection_cut) or self.injection_cut <= 1.5 or (
                self.injection_edge_m is not None and not (math.isfinite(self.injection_edge_m) and self.injection_edge_m > 0)):
            raise ValueError("injection_cut must exceed 1.5; injection_edge_m positive")
        k = self.injection_paint_turns
        if not isinstance(k, int) or k < 1 or (k > 1 and (self.injection_paint_amax_sigma <= 0
                or not 0 < self.injection_paint_sigma_ratio < 2 or k > self.turns)):
            raise ValueError("painting needs integer turns <= turns, positive A_max and a sensible injection size")
        if k > 1 and (self.opposing != "absent" or self.hminus_refresh_turns):
            raise ValueError("painting is a proton-only model")

    def validate(self) -> None:
        for name in ("energy_mev", "circumference_m", "population", "duty",
                     "sigma_m", "tune", "tune_shift_budget"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("harmonic", "particles_per_bunch", "images", "steps_per_turn",
                     "turns", "slices", "tune_samples_per_turn"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.half_xy_m is not None and (not math.isfinite(self.half_xy_m) or self.half_xy_m <= 0):
            raise ValueError("half_xy_m must be positive or null (6 sigma)")
        if not 0 <= self.outside_fraction_limit <= 1e-4:
            raise ValueError("outside_fraction_limit must be between 0 and 1e-4")
        for name in ("tune_x", "tune_y"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive or null")
        if self.tune_shift_budget >= self.plane_tunes().min():
            raise ValueError("tune_shift_budget must be smaller than the bare tune")
        if self.hminus_tune is None and (self.tune_x is not None or self.tune_y is not None) and any(
                getattr(self, f"hminus_{n}") for n in ("a_xx", "a_xy", "a_yy", "xi_x", "xi_y")):
            raise ValueError("H- Landau knobs are only implemented for equal plane tunes")
        n = self.particles_per_bunch
        if n < 32 or n & (n - 1):
            raise ValueError("particle count must be a power of two, at least 32")
        if len(self.grid) != 3 or any(int(n) != n or n < 4 or n % 2 for n in self.grid):
            raise ValueError("grid needs three even integer dimensions >= 4")
        if self.shape_order not in (1, 2) or self.backend not in ("cpu", "mlx"):
            raise ValueError("supported deposition orders: 1/2; backends: cpu/mlx")
        if self.opposing not in ("charged", "neutral", "absent"):
            raise ValueError("opposing must be charged, neutral, or absent")
        if self.steps_per_turn % self.tune_samples_per_turn:
            raise ValueError("steps_per_turn must divide into tune samples")
        if self.eta is not None and (not math.isfinite(self.eta) or self.eta <= 0):
            raise ValueError("eta must be positive; use opposing=absent for no beam")
        if self.aperture_m is not None and (not math.isfinite(self.aperture_m) or self.aperture_m <= 0):
            raise ValueError("aperture_m must be positive or null")
        if self.wall_limit_s is not None and self.wall_limit_s <= 0:
            raise ValueError("wall_limit_s must be positive")
        if self.hminus_refresh_turns is not None and (
                not isinstance(self.hminus_refresh_turns, int) or self.hminus_refresh_turns < 1
                or self.opposing == "absent"):
            raise ValueError("hminus_refresh_turns needs a positive integer and an H- beam")
        for name in ("hminus_sigma_ratio", "lens_length_m", "lens_fraction", "lens_sigma_ratio"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for role in ("proton", "hminus"):
            for name in ("a_xx", "a_xy", "a_yy", "xi_x", "xi_y"):
                if not math.isfinite(getattr(self, f"{role}_{name}")):
                    raise ValueError(f"{role}_{name} must be finite")
            rms = getattr(self, f"{role}_delta_rms")
            if not math.isfinite(rms) or rms < 0:
                raise ValueError(f"{role}_delta_rms must be finite and non-negative")
            if role == "hminus" and self.opposing == "absent" and (
                    rms or any(getattr(self, f"hminus_{n}") for n in ("a_xx", "a_xy", "a_yy", "xi_x", "xi_y"))):
                raise ValueError("H- Landau knobs need an H- beam")
        if self.long_enabled():
            if self.gamma_t is None or not self.gamma_t > 1:
                raise ValueError("gamma_t must be > 1 (set it whenever the RF is on)")
            for name in ("rf_voltage_v", "rf_h2_voltage_v", "rf_mismatch_dp", "rf_mismatch_psi"):
                if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                    raise ValueError(f"{name} must be finite and non-negative")
            if self.rf_h2_voltage_v and not self.rf_voltage_v:
                raise ValueError("a second harmonic needs the fundamental")
            if self.rf_harmonic is not None and (self.rf_harmonic < 1 or self.rf_harmonic % self.harmonic):
                raise ValueError("rf_harmonic must be a positive multiple of harmonic")
            if not isinstance(self.rf_h2_ratio, int) or self.rf_h2_ratio < 2:
                raise ValueError("rf_h2_ratio must be an integer >= 2")
            if not 0 < self.rf_fill <= 1:
                raise ValueError("rf_fill must be in (0, 1]")
            if self.rf_voltage_v and self.proton_delta_rms:
                raise ValueError("with RF the bunch is loaded matched: proton_delta_rms must be 0")
            if self.rf_dE_turn_ev is not None and not self.rf_voltage_v:
                raise ValueError("a ramp needs RF voltage")
        elif self.rf_h2_voltage_v or self.rf_dE_turn_ev is not None:
            raise ValueError("RF settings need gamma_t")
        if self.hminus_tune is not None and (not math.isfinite(self.hminus_tune) or self.hminus_tune <= 0):
            raise ValueError("hminus_tune must be positive or null")
        if self.hminus_profile not in ("gaussian", "hollow") or self.lens_profile not in ("gaussian", "hollow"):
            raise ValueError("profiles must be gaussian or hollow")
        if self.hollow_momentum not in ("circular", "thermal"):
            raise ValueError("hollow_momentum must be circular or thermal")
        if not 0 <= self.hollow_inner_ratio < self.hollow_outer_ratio:
            raise ValueError("need 0 <= hollow_inner_ratio < hollow_outer_ratio")
        if (not isinstance(self.error_superperiods, int) or self.error_superperiods < 1
                or self.error_count is not None and (not isinstance(self.error_count, int) or self.error_count < 1)
                or self.error_dk_rms < 0 or self.error_dipole_rms_rad < 0
                or not math.isfinite(self.error_dk_rms + self.error_dipole_rms_rad)):
            raise ValueError("invalid lattice-error settings")
        if self.errors_enabled() and 2 * self.error_locations() >= self.steps_per_turn:
            raise ValueError("a step must not span two error locations: need steps_per_turn > 2 M")
        if self.lens_region not in ("ring", "straights") or self.solenoid_polarity not in ("alternating", "same"):
            raise ValueError("lens_region: ring/straights; solenoid_polarity: alternating/same")
        if not math.isfinite(self.solenoid_bz_t) or not math.isfinite(self.straight_focus_ratio) \
                or self.straight_focus_ratio < 0 or (self.straight_focus_ratio == 0 and not self.solenoid_retune):
            raise ValueError("solenoid_bz_t must be finite and straight_focus_ratio positive (zero needs solenoid_retune)")
        if not isinstance(self.lattice_periods, int) or self.lattice_periods < 1:
            raise ValueError("lattice_periods must be a positive integer")
        if self.racetrack():
            ls, la = self.straight_length_m, self.arc_length_m
            period = self.circumference_m / self.lattice_periods
            if ls is None:
                ls = period - la
            if la is None:
                la = period - ls
            if ls <= 0 or la <= 0 or abs(self.lattice_periods * (ls + la) - self.circumference_m) > 1e-9 * self.circumference_m:
                raise ValueError("periodic lattice: straight and arc lengths must be positive with P (straight + arc) = circumference")
            if self.opposing != "absent" or self.hminus_profile != "gaussian" or any(
                    getattr(self, f"hminus_{n}") for n in ("a_xx", "a_xy", "a_yy", "xi_x", "xi_y")):
                raise ValueError("the periodic-lattice layout is implemented for a proton-only beam")
            if self.steps_per_turn * min(ls, la) / self.circumference_m < 2:
                raise ValueError("a step must not span two layout boundaries: more steps_per_turn")
            if self.lens_count and self.lens_region != "straights":
                raise ValueError("racetrack lenses belong in the straights: set lens_region=straights")
            if self.solenoid_retune and not self.solenoid_bz_t and self.straight_focus_ratio == 0:
                raise ValueError("a straight without external focusing needs a solenoid")
        elif self.lens_region != "ring" or self.solenoid_bz_t:
            raise ValueError("lens_region=straights and the solenoid need a racetrack layout")
        self.validate_stage2()
        if self.damper_kappa is not None:
            if len(self.damper_kappa) != 2 or self.damper_slots < 1 or self.steps_per_turn % self.damper_slots or self.damper_delay_turns < 0 \
                    or not set(self.damper_planes) <= {"x", "y"} or (self.damper_delay_turns == 0 and self.damper_arc <= self.damper_pickup_arc):
                raise ValueError("damper: two kappas, slots dividing steps_per_turn, delay >= 0 (0: kicker downstream of the pickup), planes within xy")
        if self.error_trim_dk is not None and len(self.error_trim_dk) != self.error_locations():
            raise ValueError("error_trim_dk needs one value per lattice-error location")
        if self.solenoid_edge_in_drift and not (self.racetrack() and self.solenoid_bz_t):
            raise ValueError("solenoid_edge_in_drift needs a periodic lattice with a solenoid")
        from . import ramp
        ramp.validate(self)
        if self.lens_count is not None:
            if not isinstance(self.lens_count, int) or self.lens_count < 1:
                raise ValueError("lens_count must be a positive integer")
            if self.opposing != "absent":
                raise ValueError("the electron-lens mode replaces the H- beam: set opposing=absent")
            if self.columns_enabled():
                raise ValueError("lens_count and column_sections are alternatives")
            if self.lens_count * self.lens_length_m > self.circumference_m:
                raise ValueError("lenses do not fit in the ring")
            if self.lens_region == "straights" and (self.lens_count % self.lattice_periods or self.lens_count // self.lattice_periods * self.lens_length_m
                    > (self.straight_length_m if self.straight_length_m is not None else self.circumference_m / 2 - self.arc_length_m) * (1 + 1e-12)):
                raise ValueError("straight lenses: an even count that fits in each straight")


def plane_frequency(w: np.ndarray) -> float | list[float]:
    """A plain float when both planes agree (the single-tune case), else [x, y]."""
    return float(w[0]) if w[0] == w[1] else w.tolist()


def physics(cfg: Config) -> dict:
    gamma = 1 + cfg.energy_mev * 1e6 * E / (MP * C**2)
    beta = math.sqrt(1 - gamma**-2)
    omega_r = 2 * math.pi * beta * C / cfg.circumference_m
    kappa = (1 + beta**2) / (1 - beta**2)
    rp = COULOMB * E**2 / (MP * C**2)
    strength = rp * cfg.population * cfg.circumference_m / (
        4 * math.pi**2 * beta**2 * gamma**3 * cfg.sigma_m**2)
    return dict(gamma=gamma, beta=beta, omega_r=omega_r,
                omega_beta=plane_frequency(cfg.plane_tunes() * omega_r),
                omega_beta_hminus=plane_frequency(cfg.plane_tunes("hminus") * omega_r),
                period_s=2 * math.pi / omega_r,
                kappa=kappa, D=strength,
                eta=1 / kappa if cfg.eta is None else cfg.eta,
                hminus_energy_mev=cfg.energy_mev * MH / MP)


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def backend_for(selection: str) -> PICBackend:
    """Use the required interpreter; optionally expose an existing MLX package.

    Append, rather than prepend, the compatible external site-packages so the
    integrator environment remains authoritative for NumPy and SciPy. CPU
    checks never import MLX. No environment, package, or cache is installed.
    """
    if selection == "mlx" and importlib.util.find_spec("mlx") is None:
        site = Path(os.environ.get("RING_MLX_SITE", str(Path.home() /
            f".local/share/lw-gpu-spike/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")))
        if not (site / "mlx").is_dir():
            raise RuntimeError("MLX absent: set RING_MLX_SITE to an existing compatible installation")
        sys.path.append(str(site))
        os.environ["RING_MLX_SITE"] = str(site)
    return select_backend(selection)


def core_hashes() -> dict[str, str]:
    root = Path(native_kernels.__file__).resolve().parents[2]
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
        "core/pic/kernels.py", "core/pic/backend.py", "core/pic/grid.py",
        "core/pic/gpu_backend.py", "core/pic/gpu_source.py", "core/pic/mlx_backend.py")}


def set_output_root(path: str | Path) -> Path:
    """Change the output root (solver locks, default run directories, many_turn's output check) at run time."""
    global OUT
    OUT = Path(path).resolve()
    return OUT


def require_acceptance(directory: str | Path) -> dict:
    """The copies' production gate: the setup checks' evidence (checks.json in `directory`) must pass and hash this driver.

    lwi_ring does not ship the setup checks (they stay in ring_driver/); its own gate is the pytest suite. The function is kept for
    launchers that run the setup checks against this package and store the evidence next to their configuration.
    """
    here = Path(directory)
    evidence = json.loads((here/"checks.json").read_text())
    if not evidence["all_required_pass"]:
        raise RuntimeError("setup acceptance failed; long run is disabled")
    hashes = dict(driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  setup_checks_sha256=hashlib.sha256((here/"setup_checks.py").read_bytes()).hexdigest(),
                  core_hashes=core_hashes(),
                  reference_sha256=hashlib.sha256((here/"reference.json").read_bytes()).hexdigest(),
                  per_bunch_reference_sha256=hashlib.sha256((here/"per_bunch_reference.py").read_bytes()).hexdigest())
    if any(evidence.get(key) != value for key,value in hashes.items()):
        raise RuntimeError("source or reference changed since CPU checks; rerun --checks")
    return evidence


@contextmanager
def solver_lock(slots: int = 1):
    """Exclusive serial gate, or a bounded pool behind the same shared gate."""
    if slots not in (1,2,3):
        raise ValueError("solver slots must be 1, 2, or 3")
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"solver.lock").open("a") as gate:
        mode = fcntl.LOCK_EX if slots == 1 else fcntl.LOCK_SH
        fcntl.flock(gate,mode|fcntl.LOCK_NB)
        if slots == 1:
            yield
            return
        token = None
        for i in range(slots):
            candidate = (OUT/f"solver_slot_{i}.lock").open("a")
            try:
                fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                candidate.close()
                continue
            token = candidate
            break
        if token is None:
            raise RuntimeError("all permitted ring solver slots are occupied")
        try:
            yield
        finally:
            token.close()


def quiet_start(count: int, seed: int) -> np.ndarray:
    """All 32 independent reflections in x, y, z, px, py; exact rms moments."""
    base = ndtri(qmc.Sobol(5, scramble=True, seed=seed).random_base2(
        int(math.log2(count // 32))))
    signs = np.asarray(list(itertools.product((-1, 1), repeat=5)))
    result = (base[None, :, :] * signs[:, None, :]).reshape(count, 5)
    result -= result.mean(axis=0)
    result /= np.sqrt(np.mean(result**2, axis=0))
    return result


def beam_shape(cfg: Config, role: str) -> dict:
    """Transverse density model of a beam: gaussian rms or uniform annulus (metres)."""
    if role == "hminus" and cfg.hminus_profile == "hollow":
        return dict(kind="hollow", r_in=cfg.hollow_inner_ratio * cfg.sigma_m,
                    r_out=cfg.hollow_outer_ratio * cfg.sigma_m)
    return dict(kind="gaussian",
                sigma=cfg.sigma_m * (cfg.hminus_sigma_ratio if role == "hminus" else 1.0))


def lens_shape(cfg: Config) -> dict:
    if cfg.lens_profile == "hollow":
        return dict(kind="hollow", r_in=cfg.hollow_inner_ratio * cfg.sigma_m,
                    r_out=cfg.hollow_outer_ratio * cfg.sigma_m)
    return dict(kind="gaussian", sigma=cfg.sigma_m * cfg.lens_sigma_ratio)


def dyn_shape(cfg: Config) -> dict:
    """Transverse shape of a dynamic electron column at the gun's size (dyn_profile; enclosed_fraction kinds; flat-top = annulus r_in = 0)."""
    sigma = cfg.sigma_m * cfg.lens_sigma_ratio * cfg.dyn_size_ratio
    if cfg.dyn_profile == "painted":
        return dict(kind="painted", sigma=sigma, p=cfg.dyn_painted_p)
    if cfg.dyn_profile in ("flattop", "hollow"):
        h = cfg.dyn_hollow_ratio if cfg.dyn_profile == "hollow" else 0.0
        r_out = 2 * sigma / math.sqrt(1 + h * h)
        return dict(kind="hollow", r_in=h * r_out, r_out=r_out)
    return dict(kind="gaussian", sigma=sigma)


def section_matrix(k2: float, length: float) -> np.ndarray:
    """2x2 transfer matrix of x'' = -k2 x over a length."""
    if k2 > 0:
        k = math.sqrt(k2)
        return np.array([[math.cos(k * length), math.sin(k * length) / k],
                         [-k * math.sin(k * length), math.cos(k * length)]])
    if k2 < 0:
        k = math.sqrt(-k2)
        return np.array([[math.cosh(k * length), math.sinh(k * length) / k],
                         [k * math.sinh(k * length), math.cosh(k * length)]])
    return np.array([[1., length], [0., 1.]])


def twiss_through(k2: float, length: float, beta: float, alpha: float) -> tuple[float, float, float]:
    """Twiss (beta, alpha) after a section and the unwrapped phase advance (rad) across it."""
    pieces = max(1, math.ceil(math.sqrt(abs(k2)) * length / 0.4))
    m = section_matrix(k2, length / pieces)
    psi = 0.
    for _ in range(pieces):
        psi += math.atan2(m[0, 1], beta * m[0, 0] - alpha * m[0, 1])
        gamma = (1 + alpha**2) / beta
        beta, alpha = (m[0, 0]**2 * beta - 2 * m[0, 0] * m[0, 1] * alpha + m[0, 1]**2 * gamma,
                       -(m[0, 0] * m[1, 0] * beta - (m[0, 0] * m[1, 1] + m[0, 1] * m[1, 0]) * alpha
                         + m[0, 1] * m[1, 1] * gamma))
    return beta, alpha, psi


def half_ring_twiss(k2s: float, k2a: float, ls: float, la: float) -> dict | None:
    """Periodic Twiss at the start of the straight for one cell [straight, arc] (the half ring of a racetrack); None if unstable."""
    m = section_matrix(k2a, la) @ section_matrix(k2s, ls)
    cos_mu = (m[0, 0] + m[1, 1]) / 2
    if abs(cos_mu) >= 1:
        return None
    sin_mu = math.sqrt(1 - cos_mu**2) * (1 if m[0, 1] > 0 else -1)
    beta, alpha = m[0, 1] / sin_mu, (m[0, 0] - m[1, 1]) / (2 * sin_mu)
    b1, a1, psi_s = twiss_through(k2s, ls, beta, alpha)
    _, _, psi_a = twiss_through(k2a, la, b1, a1)
    return dict(beta=beta, alpha=alpha, arc_beta=b1, arc_alpha=a1, half_phase=psi_s + psi_a)


def solve_arc_wavenumber(tune: float, ratio: float, ls: float, la: float,
                         periods: int = 2, larmor2: float = 0.0) -> float:
    """k_arc (1/m) such that the ring of `periods` cells [straight, arc] has zero-current tune `tune`,
    k_straight = ratio k_arc (plus the solenoid's Larmor focusing larmor2 = K^2 in the straight).

    The cell phase advance (the integral of dz/beta, unwrapped) is continuous and rises with k
    through the stable bands, so the root is bracketed by a scan.
    """
    from scipy.optimize import brentq
    def phase(k: float) -> float:
        twiss = half_ring_twiss((ratio * k)**2 + larmor2, k**2, ls, la)
        return math.nan if twiss is None else twiss["half_phase"] - 2 * math.pi * tune / periods
    k0 = 2 * math.pi * tune / (periods * (ls + la))
    grid = np.linspace(0.2 * k0, 3 * k0, 600)
    values = np.array([phase(k) for k in grid])
    for i in range(len(grid) - 1):
        if np.isfinite(values[i:i + 2]).all() and values[i] * values[i + 1] <= 0:
            return float(brentq(phase, grid[i], grid[i + 1], xtol=1e-14, rtol=1e-14))
    raise ValueError("no stable racetrack focusing reproduces the requested tune")


class Racetrack:
    """P cells [straight, arc] (P = lattice_periods, default 2 = racetrack) with piecewise-constant smooth focusing.

    Lab layout: straight A [0, Ls), arc 1, straight B [Ls + La, 2 Ls + La), arc 2 (P = 2); cell j
    starts at j (Ls + La). Wave numbers
    k = omega_beta / (beta c) are constant in a section; the arc value is solved per plane so the
    whole-ring zero-current tune is the configured one. Each step takes the focusing strength
    that matches its path length inside the straights, so the integrated strength is exact for
    any position of a boundary in the step. An optional hard-edge solenoid in the straights
    acts on the protons (Boris rotation inside, fringe kick at each edge, polarity per straight).
    """

    def __init__(self, cfg: Config, p: dict, dq2: np.ndarray | None = None):
        self.cfg, self.p = cfg, p
        self.periods = cfg.lattice_periods
        self.cell = cfg.circumference_m / self.periods
        self.ls = cfg.straight_length_m if cfg.straight_length_m is not None else self.cell - cfg.arc_length_m
        self.la = self.cell - self.ls
        self.tunes = cfg.plane_tunes()
        # Larmor focusing K^2 of the solenoid, K = q B / (2 p): added to the straights when retuning.
        self.larmor = cfg.solenoid_bz_t * E / (2 * MP * C * p["gamma"] * p["beta"]) if cfg.solenoid_retune else 0.
        self.k_arc = np.array([solve_arc_wavenumber(q, cfg.straight_focus_ratio, self.ls, self.la,
                                                    self.periods, self.larmor**2) for q in self.tunes])
        self.k_straight = cfg.straight_focus_ratio * self.k_arc
        self.omega_arc = p["beta"] * C * self.k_arc
        self.omega_straight = p["beta"] * C * self.k_straight
        self.starts = np.arange(self.periods) * (self.ls + self.la)   # straight origins
        self.sign = np.array([1. if (cfg.solenoid_polarity == "same" or j % 2 == 0) else -1.
                              for j in range(self.periods)])
        self.match_dq2 = None
        self.beta_now, self.b_scale, self.extra_k2 = None, 1.0, 0.0     # ramp state (None = design)
        self.edge_in_drift = bool(cfg.solenoid_edge_in_drift and cfg.solenoid_bz_t)
        if dq2 is not None:
            self.set_match(dq2)

    def set_ramp(self, beta: float, b_scale: float, extra_k2: float) -> None:
        """Ramp state: reference beta, solenoid field scale, extra straight k^2 (ramped quadrupole make-up)."""
        self.beta_now, self.b_scale, self.extra_k2 = beta, b_scale, extra_k2

    def set_match(self, dq2: np.ndarray) -> None:
        """Effective-lattice Twiss for the initial beam: dq2 = [arc, straight] shift of Q^2 (per plane).

        The collective (space-charge) defocusing and the lens add-back enter the matched lattice
        as a uniform shift of k^2 in each section, the same mean-field estimate as the smooth ring.
        """
        scale = (2 * math.pi / self.cfg.circumference_m)**2
        k2 = np.array([[(self.k_straight[i]**2 + self.larmor**2 + scale * dq2[1][i]), (self.k_arc[i]**2 + scale * dq2[0][i])]
                       for i in range(2)])
        if np.any(k2 <= 0):
            raise ValueError("racetrack: the depressed focusing has no positive k^2 in a section")
        self.eff_k2, self.entry = k2, []
        for i in range(2):
            twiss = half_ring_twiss(k2[i, 0], k2[i, 1], self.ls, self.la)
            if twiss is None:
                raise ValueError("racetrack: the space-charge-depressed lattice is unstable")
            self.entry.append((twiss["beta"], twiss["alpha"], twiss["arc_beta"], twiss["arc_alpha"]))
        z = (np.arange(4096) + 0.5) / 4096 * self.cfg.circumference_m
        self.emittance = np.array([self.cfg.sigma_m**2 / np.mean(self.twiss(z, i)[0]) for i in range(2)])
        self.match_dq2 = np.asarray(dq2)

    def match(self, own: np.ndarray, lens: np.ndarray, sigma_lens: float | None) -> None:
        """Self-consistent envelope match of the depressed lattice (per plane, section-averaged).

        The mean-field defocusing of a gaussian beam scales as 1/sigma(z)^2, so with unequal
        focusing the beam size, hence the depression, differs between arcs and straights.
        own: Q^2 depression at sigma_m (arcs); lens: Q^2 add-back at sigma_m in the straights;
        sigma_lens: gaussian lens rms (the lens force does not follow sigma(z)), None = unscaled,
        "follow" = a gaussian column whose rms follows the local beam size (add-back scales like own).
        Fixed-point iteration of: dq2 = -own <sigma_m^2/sigma^2> (+ lens <(sL^2 + sigma_m^2)/(sL^2 + sigma^2)>).
        """
        sm2 = self.cfg.sigma_m**2
        own, lens = np.broadcast_to(own, (2,)), np.broadcast_to(lens, (2,))
        grid = (np.arange(2048) + 0.5) / 2048 * self.cfg.circumference_m
        straight = (grid % (self.cfg.circumference_m / self.periods)) < self.ls
        dq2 = np.array([-own, -own + lens])
        for _ in range(200):
            self.set_match(dq2)
            new = np.empty_like(dq2)
            for plane in range(2):
                s2 = self.emittance[plane] * self.twiss(grid, plane)[0]
                for j, mask in enumerate((~straight, straight)):
                    new[j, plane] = -own[plane] * np.mean(sm2 / s2[mask])
                    if j == 1 and lens[plane] and isinstance(sigma_lens, str):
                        new[j, plane] += lens[plane] * np.mean(sm2 / s2[mask])
                    elif j == 1 and lens[plane] and sigma_lens is not None:
                        new[j, plane] += lens[plane] * np.mean((sigma_lens**2 + sm2) / (sigma_lens**2 + s2[mask]))
                    elif j == 1:
                        new[j, plane] += lens[plane]
            change = np.max(np.abs(new - dq2))
            dq2 = 0.5 * dq2 + 0.5 * new
            if change < 1e-10:
                break
        self.set_match(dq2)
        self.match_iterations = _ + 1

    def twiss(self, z: np.ndarray, plane: int) -> tuple[np.ndarray, np.ndarray]:
        """Periodic Twiss (beta, alpha) of the effective lattice at lab positions z."""
        u = z % (self.cfg.circumference_m / self.periods)
        in_arc = u >= self.ls
        b0, a0, b1, a1 = self.entry[plane]
        beta0 = np.where(in_arc, b1, b0)
        alpha0 = np.where(in_arc, a1, a0)
        k2 = np.where(in_arc, self.eff_k2[plane, 1], self.eff_k2[plane, 0])
        s = np.where(in_arc, u - self.ls, u)
        k = np.sqrt(k2)
        m11, m12, m21 = np.cos(k * s), np.sin(k * s) / k, -k * np.sin(k * s)
        m22 = m11
        gamma0 = (1 + alpha0**2) / beta0
        beta = m11**2 * beta0 - 2 * m11 * m12 * alpha0 + m12**2 * gamma0
        alpha = -(m11 * m21 * beta0 - (m11 * m22 + m12 * m21) * alpha0 + m12 * m22 * gamma0)
        return beta, alpha

    def overlaps(self, z: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
        """Fraction (n, P) of each step's path inside straight A, B, ... (the P straights)."""
        length = self.cfg.circumference_m
        lo, hi = z - v * dt / 2, z + v * dt / 2
        result = np.zeros((len(z), self.periods))
        for j, start in enumerate(self.starts):
            for shift in (-length, 0., length):
                result[:, j] += np.clip(np.minimum(hi, start + shift + self.ls)
                                        - np.maximum(lo, start + shift), 0, None)
        return result / (v * dt)[:, None]

    def omega(self, s: Species, dt: float) -> np.ndarray:
        """Per-particle omega_beta (n, 2) from the path-weighted k^2."""
        phi = self.overlaps(s.position_m[:, 2], s.velocity_m_s[:, 2], dt).sum(axis=1)[:, None]
        if self.beta_now is not None:
            bc2 = (self.beta_now * C)**2
            omega_arc2, omega_straight2 = bc2 * self.k_arc**2, bc2 * (self.k_straight**2 + self.extra_k2)
            return np.sqrt(np.maximum(omega_arc2 + phi * (omega_straight2 - omega_arc2), 0.))
        return np.sqrt(np.maximum(self.omega_arc**2 + phi * (self.omega_straight**2 - self.omega_arc**2), 0.))

    def solenoid(self, s: Species, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """(B_z per particle, edge momentum kick in mc units (n, 2), edge position correction (n, 2))."""
        b = self.cfg.solenoid_bz_t
        if not b:
            return None
        if self.b_scale != 1.0:
            b = b * self.b_scale
        length = self.cfg.circumference_m
        if self.edge_in_drift:               # the whole solenoid acts inside the half-drifts (edge_drift): nothing here
            z, v = s.position_m[:, 2], s.velocity_m_s[:, 2]
            return np.zeros(len(z)), np.zeros((len(z), 2)), np.zeros((len(z), 2))
        z, v, u = s.position_m[:, 2], s.velocity_m_s[:, 2], s.momentum_mc
        bz = (self.overlaps(z, v, dt) * self.sign).sum(axis=1) * b
        lo, hi = z - v * dt / 2, z + v * dt / 2
        slope = u[:, :2] / u[:, 2:3]
        du, dx = np.zeros((len(z), 2)), np.zeros((len(z), 2))
        coupling = s.charge_c / (2 * s.mass_kg * C)       # Delta u = coupling B (y, -x), mc units
        for j, start in enumerate(self.starts):
            for edge, direction in ((start, 1.), (start + self.ls, -1.)):
                for shift in (-length, 0., length):
                    where = edge + shift
                    crossed = (where > lo) & (where <= hi)
                    if not crossed.any():
                        continue
                    xy = s.position_m[:, :2] + slope * (where - z)[:, None]
                    kick = direction * self.sign[j] * b * coupling * np.column_stack((xy[:, 1], -xy[:, 0]))
                    if self.cfg.solenoid_edge_rotation and direction > 0:
                        # the field acts on the new slope for the rest of the step: rotate the kick (Boris sense)
                        gamma_p = np.sqrt(1 + np.sum(u**2, axis=1))
                        angle = -(s.charge_c * self.sign[j] * b / (gamma_p * s.mass_kg)) * (hi - where) / v
                        cos, sin = np.cos(angle), np.sin(angle)
                        kick = np.column_stack((cos * kick[:, 0] - sin * kick[:, 1], sin * kick[:, 0] + cos * kick[:, 1]))
                    kick = np.where(crossed[:, None], kick, 0.)
                    du += kick
                    dx += kick / u[:, 2:3] * (z - where)[:, None]
        return bz, du, dx

    def field_sign(self, z: np.ndarray) -> np.ndarray:
        """Signed unit solenoid field (0 outside, +-1 in the straights by polarity) at lab positions z."""
        zz = z % self.cfg.circumference_m
        out = np.zeros(len(z))
        for j, start in enumerate(self.starts):
            out[(zz >= start) & (zz < start + self.ls)] = self.sign[j]
        return out

    def edge_drift(self, s: Species, h: float, lon=None) -> None:
        """Half-drift by time h through the solenoid: exact helix in the field, hard-edge kicks at the crossings.

        The whole solenoid lives in the drifts (the Boris push then carries only the self fields): a particle moves on the
        exact helix of a uniform B_z (velocity rotates by -Omega t, Omega = q B_z/(gamma m); position follows), and a
        particle whose z-segment crosses an edge drifts to it, receives the kick q B/(2 m c) (y, -x) (+ at entry, - at
        exit, times the cell's polarity) at its position there, and goes on in the new field. Every piece is volume
        preserving and the composition drift-push-drift is time symmetric, so the one-turn map has det = 1 and second-order
        accuracy for any position of an edge in a step. `lon` (protons with longitudinal dynamics) gives the arc-speed
        factor 1/(1 + alpha delta) on the z velocity, as in Longitudinal.half_drift.
        """
        b = self.cfg.solenoid_bz_t * self.b_scale
        length = self.cfg.circumference_m
        v = s.velocity_m_s
        if lon is not None and lon.alpha:
            v[:, 2] = v[:, 2] / (1 + lon.alpha * lon.delta(s))
        z0 = s.position_m[:, 2].copy()
        z1 = z0 + h * v[:, 2]
        n = len(z0)
        gamma = np.sqrt(1 + np.sum(s.momentum_mc**2, axis=1))
        zeta = s.position_m[:, 0] + 1j * s.position_m[:, 1]
        w = v[:, 0] + 1j * v[:, 1]
        field0 = self.field_sign(z0)
        t_first = np.full(n, h)
        crossing = np.zeros(n, bool)
        where_all = np.zeros(n)
        kick_sign = np.zeros(n)
        for j, start in enumerate(self.starts):
            for edge, direction in ((start, 1.), (start + self.ls, -1.)):
                for shift in (-length, 0., length):
                    where = edge + shift
                    hit = (where > z0) & (where <= z1) & ~crossing
                    if hit.any():
                        crossing |= hit
                        where_all[hit] = where
                        kick_sign[hit] = direction * self.sign[j]
        t_first[crossing] = (where_all[crossing] - z0[crossing]) / (z1[crossing] - z0[crossing]) * h

        def helix(zeta, w, field, t, gam):
            omega = s.charge_c * b * field / (gam * s.mass_kg)
            phase = np.exp(-1j * omega * t)
            g = np.where(omega == 0, t, -np.expm1(-1j * omega * t) / (1j * np.where(omega == 0, 1.0, omega)))
            return zeta + w * g, w * phase

        zeta, w = helix(zeta, w, field0, t_first, gamma)
        if crossing.any():
            i = np.flatnonzero(crossing)
            coupling = s.charge_c / (2 * s.mass_kg * C)
            u_xy = w[i] * gamma[i] / C - 1j * kick_sign[i] * b * coupling * zeta[i]       # rotated momentum + edge kick
            gamma[i] = np.sqrt(1 + np.abs(u_xy)**2 + s.momentum_mc[i, 2]**2)
            w[i] = u_xy * C / gamma[i]
            field1 = self.field_sign(z1)
            zeta[i], w[i] = helix(zeta[i], w[i], field1[i], (h - t_first)[i], gamma[i])
        s.position_m[:, 0], s.position_m[:, 1] = zeta.real, zeta.imag
        s.position_m[:, 2] = z1
        # momentum from the final velocity (gamma is conserved by the rotation; kicked particles carry their new gamma)
        s.momentum_mc[:, 0] = (w.real * gamma / C)
        s.momentum_mc[:, 1] = (w.imag * gamma / C)

    def describe(self) -> dict:
        return dict(straight_length_m=self.ls, arc_length_m=self.la, straight_focus_ratio=self.cfg.straight_focus_ratio,
                    k_arc_per_m=self.k_arc.tolist(), k_straight_per_m=self.k_straight.tolist(),
                    solenoid_bz_t=self.cfg.solenoid_bz_t, solenoid_polarity=self.cfg.solenoid_polarity,
                    **(dict(lattice_periods=self.periods, solenoid_larmor_K_per_m=self.larmor) if self.periods != 2 or self.larmor else {}),
                    match_dq2_arc_straight=None if self.match_dq2 is None else self.match_dq2.tolist(),
                    match_iterations=getattr(self, 'match_iterations', None),
                    emittance_m_rad=None if self.match_dq2 is None else self.emittance.tolist(),
                    lens_region=self.cfg.lens_region)


def enclosed_fraction(shape: dict, r2: np.ndarray) -> np.ndarray:
    """Fraction of a beam's charge inside radius r (2-D Gauss law)."""
    if shape["kind"] == "gaussian":
        return -np.expm1(-r2 / (2 * shape["sigma"]**2))
    if shape["kind"] == "waterbag":
        x = np.minimum(r2 / (6 * shape["sigma"]**2), 1.0)       # 4-D waterbag: density ~ 1 - r^2/a^2, a^2 = 6 sigma_rms^2
        return 2 * x - x * x
    if shape["kind"] == "painted":
        x = np.minimum(r2 / (2 * (shape["p"] + 3) * shape["sigma"]**2), 1.0)   # 4-D (1 - rho^2/R^2)^p: radial density (1 - r^2/R^2)^(p+1)
        return 1 - (1 - x)**(shape["p"] + 2)
    if shape["kind"] == "empirical":
        return np.searchsorted(shape["r2"], r2, side="right") / len(shape["r2"])
    return np.clip((r2 - shape["r_in"]**2) / (shape["r_out"]**2 - shape["r_in"]**2), 0, 1)


def transverse_sample(normals: np.ndarray, shape: dict) -> np.ndarray:
    """Map quiet-start standard-normal pairs to the density; keeps the sign symmetry.

    Gaussian: scale. Annulus: the Gaussian radial CDF u = 1-exp(-rho^2/2) is
    uniform on [0,1), so r^2 = r_in^2 + u (r_out^2 - r_in^2) is area-uniform.
    """
    if shape["kind"] == "gaussian":
        return normals * shape["sigma"]
    u = -np.expm1(-np.sum(normals**2, axis=1) / 2)
    r = np.sqrt(shape["r_in"]**2 + u * (shape["r_out"]**2 - shape["r_in"]**2))
    phi = np.arctan2(normals[:, 1], normals[:, 0])
    return np.column_stack((r * np.cos(phi), r * np.sin(phi)))


def virial_factor(source: dict, observer: dict, observer_xy: np.ndarray, sigma_m: float) -> float:
    """sigma^2 <enclosed_source(r)> / <x^2> over the observer; equals 1/2 for equal gaussians."""
    if source["kind"] == "gaussian" and observer["kind"] == "gaussian":
        return sigma_m**2 / (source["sigma"]**2 + observer["sigma"]**2)
    r2 = np.sum(observer_xy**2, axis=1)
    return float(sigma_m**2 * np.mean(enclosed_fraction(source, r2)) / np.mean(observer_xy**2))


def frozen_delta(count: int, rms: float, seed: int, offset: int, bunch: int, role: str) -> np.ndarray:
    """Per-particle relative momentum offsets: gaussian pairs +-d, exact zero mean and rms."""
    rng = np.random.default_rng([seed, offset, bunch, 1 if role == "hminus" else 0, 6151])
    half = rng.standard_normal(count // 2)
    d = rng.permutation(np.concatenate((half, -half)))
    return d * (rms / math.sqrt(np.mean(d**2)))


def initial_beams(cfg: Config, roles: tuple | None = None, seed_offset: int = 0
                  ) -> tuple[list[Species], list[dict], dict]:
    p = physics(cfg)
    # Report Section 5: approximate Gaussian rms virial, averaged over the train.
    beff = 1 + 2 * sum(math.exp(-4 * math.pi**2 * m*m * cfg.duty**2)
                       for m in range(1, 100))
    eta_charged = p["eta"] if cfg.opposing == "charged" else 0
    track = Racetrack(cfg, p) if cfg.racetrack() else None
    beams, metadata = [], []
    # Non-default H- size or profile switches the force balance to the general virial.
    generic = cfg.hminus_sigma_ratio != 1 or cfg.hminus_profile != "gaussian" or cfg.injection_profile != "gaussian"
    shapes = {role: beam_shape(cfg, role) for role in ("proton", "hminus")}
    lon0 = Longitudinal(cfg, p) if cfg.long_enabled() else None
    reference = quiet_start(cfg.particles_per_bunch, cfg.seed)
    shape_xy = {role: transverse_sample(reference[:, :2], shapes[role]) for role in shapes}
    if cfg.injection_profile != "gaussian":
        # Non-gaussian injection: the force balance uses the loaded sample itself (empirical enclosed fraction).
        loaded = injection.shaped(reference, cfg)[:, :2] * cfg.sigma_m
        shapes["proton"] = dict(kind="empirical", r2=np.sort(np.sum(loaded**2, axis=1)))
        shape_xy["proton"] = loaded
    orbit = None
    if cfg.start_on_closed_orbit:
        from .lattice_linear import LinearLattice
        orbit = LinearLattice(cfg, p, track, LatticeErrors(cfg, p) if cfg.errors_enabled() else None).closed_orbit()
    for role, mass, charge, direction, ratio in (
        ("proton", MP, E, 1, 1),
        ("hminus", MH, -E if cfg.opposing == "charged" else 0, -1, p["eta"]),
    ):
        if role == "hminus" and cfg.opposing == "absent":
            continue
        if roles is not None and role not in roles:
            continue
        ds = p["D"] * MP / mass
        own = ratio if charge else 0
        other = eta_charged if role == "proton" else (1 if charge else 0)
        tune = cfg.plane_tunes(role)
        if generic:
            peer = shapes["hminus" if role == "proton" else "proton"]
            own_term = ds * own * beff * virial_factor(shapes[role], shapes[role], shape_xy[role], cfg.sigma_m)
            other_term = ds * other * p["kappa"] * virial_factor(peer, shapes[role], shape_xy[role], cfg.sigma_m)
            variance = tune**2 - own_term + other_term
        else:
            variance = tune**2 - 0.5 * ds * own * beff + 0.5 * ds * other * p["kappa"]
        lens_back = 0.
        if cfg.lens_enabled() and role == "proton":
            # The lens cancels a fraction f of the proton self-force (z-profile and
            # train factor as for the self-force): add back its focusing to the balance.
            lens_back = ds * effective_lens_fraction(cfg) * beff * virial_factor(
                dyn_shape(cfg) if cfg.dyn_match_profile else lens_shape(cfg), shapes[role], shape_xy[role], cfg.sigma_m)
            variance += lens_back
        if not cfg.collective:
            variance = tune**2
            lens_back = 0.
        if track is not None:
            # Racetrack: arcs carry the full mean-field defocusing; the lens add-back (a ring
            # fraction f) is concentrated in the straights, C / (2 Ls) times stronger there.
            track.match(tune**2 - (variance - lens_back), lens_back * cfg.circumference_m / (track.periods * track.ls),
                        ("follow" if cfg.column_sigma_follow else cfg.sigma_m * cfg.lens_sigma_ratio)
                        if cfg.lens_profile == "gaussian" else None)
        if np.any(variance <= 0):
            raise ValueError("approximate rms force balance has no positive momentum variance")
        sigma_mom = cfg.sigma_m
        if role == "hminus" and generic:
            sigma_mom = float(np.sqrt(np.mean(shape_xy[role]**2)))
        for bunch in range(cfg.harmonic):
            a = quiet_start(cfg.particles_per_bunch, cfg.seed + bunch + seed_offset)
            paint_batch = None
            if role == "proton":
                if cfg.injection_paint_turns > 1:
                    a, paint_batch = injection.paint(a, cfg)
                else:
                    a = injection.shaped(a, cfg)
            center = bunch * cfg.circumference_m / cfg.harmonic
            pos = a[:, :3] * [cfg.sigma_m, cfg.sigma_m,
                                  cfg.duty * cfg.circumference_m / cfg.harmonic]
            if role == "hminus" and generic:
                pos[:, :2] = transverse_sample(a[:, :2], shapes[role])
            lon_delta = None
            if lon0 is not None and role == "proton":
                if cfg.rf_voltage_v:
                    load = matched_sample(lon0.bucket(), cfg.rf_fill,
                                          cfg.rf_rms_psi_rad or 2 * math.pi * cfg.duty, len(pos),
                                          cfg.seed + 31 * bunch + seed_offset)
                    psi = load["psi"] * cfg.rf_mismatch_psi + cfg.rf_offset_psi_rad
                    lon_delta = load["delta"] * cfg.rf_mismatch_dp
                    if track is not None:
                        # The twiss map makes x depend on the local beta(z): decorrelate the longitudinal load from the
                        # quiet start's transverse reflection groups (else x and y rms differ by tens of percent).
                        order = np.random.default_rng([cfg.seed, seed_offset, bunch, 4409]).permutation(len(psi))
                        psi, lon_delta = psi[order], lon_delta[order]
                    pos[:, 2] = -psi * cfg.circumference_m / (2 * math.pi * lon0.h)
                elif cfg.proton_delta_rms > 0:
                    lon_delta = frozen_delta(len(pos), cfg.proton_delta_rms, cfg.seed, seed_offset, bunch, role)
            z_local = (pos[:, 2] + cfg.circumference_m / 2) % cfg.circumference_m - cfg.circumference_m / 2
            pos[:, 2] = (pos[:, 2] + center) % cfg.circumference_m
            u = np.zeros_like(pos)
            u[:, :2] = (a[:, 3:] * p["gamma"] * p["omega_r"]
                        * sigma_mom * np.sqrt(variance) / C)
            if track is not None:
                # Matched to the local Twiss of the depressed racetrack lattice at each particle's z.
                for plane in (0, 1):
                    beta_z, alpha_z = track.twiss(pos[:, 2], plane)
                    root = np.sqrt(track.emittance[plane] / beta_z)
                    pos[:, plane] = a[:, plane] * np.sqrt(track.emittance[plane] * beta_z)
                    u[:, plane] = root * (a[:, 3 + plane] - alpha_z * a[:, plane]) * p["gamma"] * p["beta"]
            if track is not None and track.larmor:
                # Round beam with zero canonical angular momentum inside a solenoid: the lab slopes carry the Larmor
                # rotation term K (y, -x) (the same kick the entry fringe gives); the twiss match above is the
                # rotating-frame (uncoupled) one.
                inside = (pos[:, 2] % track.cell) < track.ls
                cell_index = np.minimum((pos[:, 2] // track.cell).astype(int), track.periods - 1)
                swirl = np.where(inside, track.sign[cell_index] * track.larmor, 0.)
                u[:, 0] += swirl * pos[:, 1] * p["gamma"] * p["beta"]
                u[:, 1] -= swirl * pos[:, 0] * p["gamma"] * p["beta"]
            if role == "hminus" and generic and shapes[role]["kind"] == "hollow" \
                    and cfg.hollow_momentum == "circular":
                # Circular orbits at the effective tune, half in each sense (zero net
                # angular momentum): stationary in a linear well, so the hollow persists.
                sense = np.sign(a[:, 3])
                u[:, 0] = -sense * pos[:, 1]
                u[:, 1] = sense * pos[:, 0]
                u[:, :2] *= p["gamma"] * p["omega_r"] * np.sqrt(variance) / C
            u[:, 2] = direction * p["gamma"] * p["beta"]
            if lon_delta is not None:
                u[:, 2] *= 1 + lon_delta
            if orbit is not None and role == "proton":
                co = orbit.at(pos[:, 2])
                pos[:, :2] += co[:, [0, 2]]
                u[:, :2] += co[:, [1, 3]] * u[:, 2:3]
            s = Species(f"{role}_{bunch}", pos, u, charge, mass,
                        cfg.population * ratio / cfg.harmonic / len(pos))
            if paint_batch is not None:
                injection.attach_painting(s, paint_batch)
            delta_rms = getattr(cfg, f"{role}_delta_rms")
            if delta_rms > 0:
                s.delta = frozen_delta(len(pos), delta_rms, cfg.seed, seed_offset, bunch, role)
            beams.append(s)
            # Fixed identities are material slices, not reassigned each turn.
            edges = np.quantile(z_local, np.linspace(0, 1, cfg.slices + 1))
            slice_id = np.clip(np.searchsorted(edges[1:-1], z_local, side="right"), 0, cfg.slices-1)
            metadata.append(dict(role=role, bunch=bunch, design_z_m=center,
                                 direction=direction, slice_id=slice_id,
                                 matched_variance=float(variance[0]) if variance[0] == variance[1] else variance.tolist(),
                                 initial_centroid_offset_m=[float(pos[:, 0].mean()),
                                     float(pos[:, 1].mean()), float(z_local.mean())]))
    if track is not None:
        p["racetrack_match"] = track.describe()
    return beams, metadata, p


def periodic_octant(x: np.ndarray, y: np.ndarray, z: np.ndarray, spacing: np.ndarray,
                    length: float, images: int, axis: int) -> np.ndarray:
    """Native exact central cell integral plus stable distant-image quadrature.

    Distant images use stable 2-point tensor Gauss cell quadrature to avoid
    subtracting eight almost equal logarithmic primitives at large distance.
    This is fourth order in cell size/distance and independently checked.
    The odd/even projection enforces pair symmetry, including the half period.
    """
    value = integrated_green((x, y, z), spacing, axis)
    for image in range(1, images + 1):
        for direction in (-1, 1):
            zz = z + direction * image * length
            part = np.zeros_like(value)
            for signs in itertools.product((-1, 1), repeat=3):
                xxq = x + signs[0] * spacing[0] / math.sqrt(12)
                yyq = y + signs[1] * spacing[1] / math.sqrt(12)
                zzq = zz + signs[2] * spacing[2] / math.sqrt(12)
                r2 = xxq**2 + yyq**2 + zzq**2
                part += COULOMB / 8 * (xxq, yyq, zzq)[axis] / r2**1.5
            value += part
    return value


def periodic_green(shape: tuple, spacing: np.ndarray, images: int, axis: int) -> np.ndarray:
    """Open transverse convolution, with odd/even periodic longitudinal parity."""
    nx, ny, nz = shape
    x = (np.arange(nx) * spacing[0])[:, None, None]
    y = (np.arange(ny) * spacing[1])[None, :, None]
    z = (np.arange(nz // 2 + 1) * spacing[2])[None, None, :]
    value = periodic_octant(x,y,z,spacing,nz*spacing[2],images,axis)
    if axis in (0, 1):
        index = [slice(None)] * 3
        index[axis] = 0
        value[tuple(index)] = 0
    else:
        value[:, :, 0] = 0
        value[:, :, -1] = 0
    ix = np.minimum(np.abs(np.r_[np.arange(nx), np.arange(-nx, 0)]), nx - 1)
    iy = np.minimum(np.abs(np.r_[np.arange(ny), np.arange(-ny, 0)]), ny - 1)
    iz = np.minimum(np.arange(nz), nz - np.arange(nz))
    mesh = value[np.ix_(ix, iy, iz)]
    if axis < 2:
        n = shape[axis]
        sign = np.r_[np.ones(n), -np.ones(n)]
    else:
        sign = np.where(np.arange(nz) <= nz // 2, 1, -1)
    dims = [1, 1, 1]
    dims[axis] = len(sign)
    mesh *= sign.reshape(dims)
    return mesh


class RingFields:
    """One moving ring-period rest grid per beam, with a fixed design boost.

    The population-weighted transverse centroid follows the whole beam. Bunch
    identities stay separate for pushing and diagnostics. Two transverse guard
    cells contain complete particle stencils at the physical box edge.
    """
    _cpu_kernel_cache = None

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.p = physics(cfg)
        self.backend = backend_for(cfg.backend)
        self.half_xy_m = cfg.half_extent()
        self.shape = (cfg.grid[0]+4, cfg.grid[1]+4, cfg.grid[2])
        self.last_boundary = {}
        self.last_core = {}
        self.z_shift = 0.0          # reference arc position minus design-speed position (RF mode)
        self.gamma_now, self.beta_now = self.p["gamma"], self.p["beta"]     # ramp: reference values of the step
        self.kernel_gamma = self.p["gamma"]
        self.lab_dz = None
        self.spacing = np.asarray([2 * self.half_xy_m / cfg.grid[0],
                                   2 * self.half_xy_m / cfg.grid[1],
                                   cfg.circumference_m * self.p["gamma"] / cfg.grid[2]])
        self.spectra = []
        self.fft_shape = (2 * self.shape[0], 2 * self.shape[1], self.shape[2])
        self.lab_dz = self.spacing[2] / self.p["gamma"]        # lab cell length, independent of the Green function's gamma
        self.build_kernel()

    def build_kernel(self) -> None:
        cfg = self.cfg
        key = (self.shape,tuple(self.spacing),cfg.images)
        cached = RingFields._cpu_kernel_cache
        if cfg.backend == "cpu" and cached is not None and cached[0] == key:
            self.spectra = cached[1]
            return
        self.spectra = []
        for axis in range(3):
            mesh = periodic_green(self.shape, self.spacing, cfg.images, axis)
            self.spectra.append(self.rfft(self.backend.array(mesh)))
            self.backend.synchronize(self.spectra[-1])
        if cfg.backend == "cpu":
            RingFields._cpu_kernel_cache = (key,self.spectra)

    def set_gamma(self, gamma: float, tolerance: float) -> None:
        """Ramp: boost of the step; the longitudinal Green function is rebuilt when gamma has moved by `tolerance`."""
        self.gamma_now, self.beta_now = gamma, math.sqrt(1 - gamma**-2)
        if abs(gamma / self.kernel_gamma - 1) > tolerance:
            self.kernel_gamma = gamma
            self.spacing = np.asarray([self.spacing[0], self.spacing[1], self.cfg.circumference_m * gamma / self.cfg.grid[2]])
            self.build_kernel()

    def rfft(self, a: object, shape: tuple | None = None) -> object:
        if self.cfg.backend == "cpu":
            return fft.rfftn(a, s=shape, workers=1)
        return self.backend.xp.fft.rfftn(a, s=shape, axes=(0, 1, 2))

    def irfft(self, a: object) -> object:
        if self.cfg.backend == "cpu":
            return fft.irfftn(a, s=self.fft_shape, workers=1)
        return self.backend.xp.fft.irfftn(a, s=self.fft_shape, axes=(0, 1, 2))

    def coordinates(self, position: np.ndarray, center: np.ndarray) -> np.ndarray:
        # A cell-centred translated grid, as in native ElectrostaticPIC.
        co = np.empty_like(position)
        co[:, :2] = (position[:, :2] - center[:2] + self.half_xy_m) / self.spacing[:2] - 0.5 + 2
        co[:, 2] = ((position[:, 2] - center[2] + self.cfg.circumference_m / 2)
                     % self.cfg.circumference_m) / self.lab_dz - 0.5
        base = np.floor(co[:, :2] + (0.5 if self.cfg.shape_order == 2 else 0))
        if self.cfg.shape_order == 2:
            base -= 1
        if np.any(base < 0) or np.any(base + self.cfg.shape_order >= np.asarray(self.shape[:2])):
            raise ValueError("transverse stencil left free-space grid; enlarge half_xy_m")
        return co

    def deposit(self, co: np.ndarray, charges: np.ndarray) -> object:
        # Two halo cells tolerate float32 rounding exactly onto the seam.
        nx, ny, nz = self.shape
        shifted = co.copy()
        shifted[:, 2] += 2
        raw = self.backend.deposit(shifted, charges, (nx, ny, nz + 4), self.cfg.shape_order)
        xp = self.backend.xp
        return xp.concatenate((raw[:, :, 2:4] + raw[:, :, -2:], raw[:, :, 4:-4],
                               raw[:, :, -4:-2] + raw[:, :, :2]), axis=2)

    def solve(self, charge: object) -> object:
        spectrum = self.rfft(charge, self.fft_shape)
        components = [self.irfft(spectrum * green)[:self.shape[0], :self.shape[1], :]
                      for green in self.spectra]
        field = self.backend.xp.stack(components, axis=-1)
        self.backend.synchronize(field)
        return field

    def gather(self, co: np.ndarray, field: object) -> np.ndarray:
        padded = self.backend.xp.concatenate((field[:, :, -2:, :], field, field[:, :, :2, :]), axis=2)
        shifted = co.copy()
        shifted[:, 2] += 2
        rest = self.backend.to_host(self.backend.gather(shifted, padded, self.cfg.shape_order)).astype(float)
        rest[:, :2] *= self.gamma_now
        return rest

    def groups(self, beams: list[Species], meta: list[dict], lab_time: float) -> list[dict]:
        groups = []
        for role in ("proton", "hminus", "electron"):
            indices = [i for i, info in enumerate(meta) if info["role"] == role]
            if not indices:
                continue
            positions = np.concatenate([beams[i].position_m for i in indices])
            weights = np.concatenate([beams[i].population for i in indices])
            if role == "electron" and not weights.sum():
                continue          # no electrons alive anywhere: no column occupied
            direction = meta[indices[0]]["direction"]
            shift = self.z_shift if direction > 0 else 0.0
            center = np.r_[np.average(positions[:, :2], axis=0, weights=weights),
                           (direction*self.p["beta"]*C*lab_time + shift) % self.cfg.circumference_m]
            groups.append(dict(role=role, indices=indices, positions=positions,
                charges=np.concatenate([beams[i].source_charge_c for i in indices]),
                weights=weights, center=center, direction=direction,
                charge=beams[indices[0]].charge_c, mass=beams[indices[0]].mass_kg))
        return groups

    def boundary(self, groups: list[dict]) -> dict:
        result = {}
        for g in groups:
            outside = np.any(abs(g["positions"][:, :2]-g["center"][:2]) > self.half_xy_m, axis=1)
            fraction = float(np.mean(outside))
            result[g["role"]] = dict(outside_count=int(outside.sum()), particle_count=len(outside),
                outside_fraction=fraction, population_fraction=float(g["weights"][outside].sum()/g["weights"].sum()),
                half_extent_m=self.half_xy_m, centroid_xy_m=g["center"][:2].tolist())
        self.last_boundary = result
        if any(row["outside_fraction"] > self.cfg.outside_fraction_limit for row in result.values()):
            raise ValueError("more than the allowed particle fraction outside transverse box")
        return result

    def core_probes(self, g: dict, meta: list[dict], lab_time: float) -> np.ndarray:
        shift = self.z_shift if g["direction"] > 0 else 0.0
        zs = [(meta[i]["design_z_m"]+g["direction"]*self.p["beta"]*C*lab_time+shift)
              % self.cfg.circumference_m for i in g["indices"]]
        offsets = self.cfg.sigma_m*np.asarray([-0.12, -0.06, 0.06, 0.12])
        return np.asarray([g["center"]+[x,y,z-g["center"][2]] for axis in (0,1)
                           for z in zs for v in offsets
                           for x,y in [(v,0) if axis==0 else (0,v)]])

    def fields(self, beams: list[Species], meta: list[dict], lab_time: float,
               measure_core: bool = False) -> tuple[list, list, dict]:
        electric = [np.zeros_like(s.position_m) for s in beams]
        magnetic = [np.zeros_like(s.position_m) for s in beams]
        timings = dict(deposit_s=0., solve_s=0., gather_s=0., source_solves=0, beam_gathers=0)
        groups = self.groups(beams, meta, lab_time)
        self.boundary(groups)
        probes = [self.core_probes(g, meta, lab_time) if measure_core else np.empty((0,3)) for g in groups]
        pe = [np.zeros_like(x) for x in probes]
        pb = [np.zeros_like(x) for x in probes]
        for source in groups:
            if not source["charge"]:
                continue
            center = source["center"]
            tick = time.monotonic()
            charge = self.deposit(self.coordinates(source["positions"], center), source["charges"])
            timings["deposit_s"] += time.monotonic() - tick
            tick = time.monotonic()
            field = self.solve(charge)
            timings["solve_s"] += time.monotonic() - tick
            timings["source_solves"] += 1
            tick = time.monotonic()
            beta_vec = np.asarray([0, 0, source["direction"] * self.beta_now])
            for j, observer in enumerate(groups):
                # Core probes join the beam gather, retaining four gather calls.
                positions = np.concatenate((observer["positions"], probes[j]))
                ee = self.gather(self.coordinates(positions, center), field)
                bb = np.cross(beta_vec, ee) / C
                first = 0
                for i in observer["indices"]:
                    last = first + len(beams[i].position_m)
                    electric[i] += ee[first:last]
                    magnetic[i] += bb[first:last]
                    first = last
                pe[j] += ee[first:]
                pb[j] += bb[first:]
                timings["beam_gathers"] += 1
            timings["gather_s"] += time.monotonic() - tick
        self.last_core = {}
        if measure_core:
            for j, g in enumerate(groups):
                offsets = probes[j][:,:2] - g["center"][:2]
                force = g["charge"]*(pe[j]+np.cross([0,0,g["direction"]*self.beta_now*C],pb[j]))
                gradients = np.asarray([np.dot(offsets[:,axis],force[:,axis])/np.dot(offsets[:,axis],offsets[:,axis])
                                        for axis in (0,1)])
                ds = gradients/(self.gamma_now*g["mass"]*self.p["omega_r"]**2*(self.beta_now/self.p["beta"])**2)
                self.last_core[g["role"]] = ds
        return electric, magnetic, timings


def omega_beta_of(p: dict, info: dict) -> float:
    return np.asarray(p["omega_beta_hminus"] if info["role"] == "hminus" else p["omega_beta"])


class LatticeErrors:
    """Thin-lens gradient and dipole errors at M fixed locations (external, seeded, per run).

    Location j sits at lab z_j = (j + 1/2) C/M. A particle receives the kick once, in the
    step whose path segment [z - v dt/2, z + v dt/2] contains z_j, so the total is independent
    of the step size (as for the electron lens). The gradient kick uses x at the crossing.
    Gradient: dx' = -g_j x, dy' = +g_j y, g_j = dk_j K_ref C/M, K_ref = (Q_ref/R)^2 with Q_ref
    the mean bare tune of the beam. Dipole: dx', dy' = theta_j. Both are converted to momentum
    increments via u_z = gamma beta_z. The draws (dk, then theta) do not touch the particle
    random stream and are made even when an amplitude is zero, so the pattern scales with it.
    """

    def __init__(self, cfg: Config, p: dict):
        self.cfg, self.p = cfg, p
        self.count = cfg.error_locations()
        rng = np.random.default_rng(cfg.error_seed)
        self.dk = rng.standard_normal(self.count) * cfg.error_dk_rms
        self.theta = rng.standard_normal((self.count, 2)) * cfg.error_dipole_rms_rad
        if cfg.error_trim_dk is not None:
            self.dk = self.dk + np.asarray(cfg.error_trim_dk, float)
        self.spacing = cfg.circumference_m / self.count
        self.location = (np.arange(self.count) + 0.5) * self.spacing

    def describe(self) -> dict:
        return dict(count=self.count, seed=self.cfg.error_seed, dk_rms=self.cfg.error_dk_rms,
                    dipole_rms_rad=self.cfg.error_dipole_rms_rad, location_m=self.location.tolist(),
                    dk=self.dk.tolist(), dipole_rad=self.theta.tolist())

    def kicks(self, beams: list[Species], meta: list[dict], dt: float) -> list:
        """(momentum increment in mc units, position correction in m), x and y, per beam.

        Both are applied at the step midpoint. The position correction angle*(z - z_j) moves the
        kick to its true crossing point, which makes the thin kick second order in the step.
        """
        result = []
        length = self.cfg.circumference_m
        for s, info in zip(beams, meta):
            z, u = s.position_m[:, 2], s.momentum_mc
            v = s.velocity_m_s[:, 2]
            lo, hi = z - v * dt / 2, z + v * dt / 2
            # last location at or below hi (count may be periodic-wrapped), kicked if above lo
            j = np.floor(hi / self.spacing - 0.5).astype(int)
            crossed = self.location[j % self.count] + (j // self.count) * length > lo
            where = self.location[j % self.count] + (j // self.count) * length
            k = j % self.count
            slope = u[:, :2] / u[:, 2:3]
            q_ref = float(np.mean(self.cfg.plane_tunes(info["role"])))
            gain = self.dk[k] * (q_ref * 2 * math.pi / length)**2 * self.spacing
            at_x = s.position_m[:, :2] + slope * (where - z)[:, None]
            angle = self.theta[k] + gain[:, None] * at_x * np.array([-1., 1.])
            angle = np.where(crossed[:, None], angle, 0.)
            result.append((angle * u[:, 2:3], angle * (z - where)[:, None]))
        return result


def column_geometry(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(start, length) arrays in metres of the electron-column sections."""
    sec = cfg.column_sections
    if sec == "ring":
        return np.array([0.]), np.array([cfg.circumference_m])
    if sec == "straights":
        cell = cfg.circumference_m / cfg.lattice_periods
        ls = cfg.straight_length_m if cfg.straight_length_m is not None else cell - cfg.arc_length_m
        return np.arange(cfg.lattice_periods) * cell, np.full(cfg.lattice_periods, ls)
    ordered = sorted(map(tuple, sec))
    return np.array([a for a, _ in ordered], float), np.array([b - a for a, b in ordered], float)


def effective_lens_fraction(cfg: Config) -> float:
    """Ring-averaged compensation fraction f (lens_fraction, or the column's local fraction times its ring share)."""
    if cfg.columns_enabled() and cfg.column_local_fraction is not None:
        return cfg.column_local_fraction * float(column_geometry(cfg)[1].sum()) / cfg.circumference_m
    return cfg.lens_fraction


class ElectronLens:
    """WEAK-STRONG co-moving electron lenses or distributed columns (the electrons do not evolve).

    N_lens lenses of length L sit at fixed lab positions. Strongly magnetised
    electrons ride at the proton velocity, so a proton stays in its own slice of
    electrons throughout the lens. The electron charge is a rigid, refreshed-each-pass
    distribution: at the proton's longitudinal position the electron line density is
    lambda_e(z) = -f lambda_p(z) C / (N_lens L), with lambda_p the (smoothed) lab line
    density of the actual proton macroparticles. Transverse profile: fixed gaussian or
    uniform annulus on the machine axis. Lab-frame force on a proton:
    F = q E_r (1 - beta_p beta_e), beta_e = beta_design; for co-moving beams
    this is the proton's own space-charge factor 1/gamma^2, so f = 1 cancels the
    core self-force exactly for equal profiles. Impulse = F * (path overlap with the
    lens) / v_z, so each proton receives F L / v_z once per turn.

    Columns (Config.column_sections) generalise the lenses to arbitrary lattice sections. The column
    axis can follow the zero-current closed orbit of the lattice (column_follow_orbit; a proton on the
    closed orbit then feels no dipole kick) and the gaussian profile can follow the local matched rms
    size (column_sigma_follow). The electron line density follows the proton macroparticles in every case.
    """
    bins = 1024
    smooth_bins = 2.0

    def __init__(self, cfg: Config, p: dict):
        self.cfg, self.p = cfg, p
        self.shape = lens_shape(cfg)
        self.orbit = None
        self.sigma_track = None
        self.beta_e = p["beta"]            # electron velocity (ramp: follows the proton velocity)
        self.sigma_scale = 1.0             # ramp: column size relative to the injection size
        self.fraction_norm = 1.0           # scale = -f * fraction_norm
        self.drift = None                  # ColumnDrift (column_centroid_drift)
        if cfg.columns_enabled():
            self.start, self.lengths = column_geometry(cfg)
            total = float(self.lengths.sum())
            self.scale = -cfg.column_local_fraction if cfg.column_local_fraction is not None \
                else -cfg.lens_fraction * cfg.circumference_m / total
            self.fraction_norm = 1.0 if cfg.column_local_fraction is not None else cfg.circumference_m / total
            if cfg.column_follow_orbit:
                from .lattice_linear import LinearLattice
                track = Racetrack(cfg, p) if cfg.racetrack() else None
                errors = LatticeErrors(cfg, p) if cfg.errors_enabled() else None
                self.orbit = LinearLattice(cfg, p, track, errors).closed_orbit()
            if cfg.column_sigma_follow and cfg.racetrack() and "racetrack_match" in p:
                match = p["racetrack_match"]
                self.sigma_track = Racetrack(cfg, p, np.asarray(match["match_dq2_arc_straight"]))
            if cfg.column_centroid_drift:
                self.drift = ColumnDrift(self, cfg, p)
            return
        self.start = (np.arange(cfg.lens_count) + 0.5) * cfg.circumference_m / cfg.lens_count \
            - cfg.lens_length_m / 2
        if cfg.lens_region == "straights":
            # n/2 lenses evenly spaced in each straight (n = 2 with L = straight length: continuous).
            track = Racetrack(cfg, p)
            per = cfg.lens_count // track.periods
            self.start = np.concatenate([origin + (np.arange(per) + 0.5) * track.ls / per - cfg.lens_length_m / 2
                                         for origin in track.starts])
        self.lengths = np.full(len(self.start), cfg.lens_length_m)
        self.scale = -self.cfg.lens_fraction * self.cfg.circumference_m / (self.cfg.lens_count * self.cfg.lens_length_m)
        self.fraction_norm = self.cfg.circumference_m / (self.cfg.lens_count * self.cfg.lens_length_m)

    def set_ramp(self, beta_e: float, fraction: float, sigma_scale: float) -> None:
        """Ramp state of the step: electron velocity, compensation fraction f, column size relative to injection."""
        self.beta_e, self.scale, self.sigma_scale = beta_e, -fraction * self.fraction_norm, sigma_scale

    def ramp_diagnostics(self, protons: list[Species], beta: float) -> dict:
        """Compensation as realised on the macroparticles: electron/proton charge inside the columns, peak current."""
        density = self.line_density(protons)
        z = (np.arange(self.bins) + 0.5) / self.bins * self.cfg.circumference_m
        inside = self.inside(z)
        dz = self.cfg.circumference_m / self.bins
        q_p = float(density[inside].sum() * dz)
        q_e = self.scale * q_p
        return dict(electron_to_proton_charge_in_columns=-q_e / q_p if q_p else None,
                    electron_peak_current_a=abs(self.scale) * float(density.max()) * self.beta_e * C,
                    electron_charge_in_columns_c=q_e, proton_charge_in_columns_c=q_p,
                    proton_peak_current_a=float(density.max()) * beta * C)

    def line_density(self, protons: list[Species]) -> np.ndarray:
        """Proton charge per metre of ring (C/m) on a periodic grid."""
        from scipy.ndimage import gaussian_filter1d
        length = self.cfg.circumference_m
        z = np.concatenate([s.position_m[:, 2] for s in protons]) % length
        q = np.concatenate([s.source_charge_c for s in protons])
        hist = np.bincount(np.minimum((z / length * self.bins).astype(int), self.bins - 1),
                           weights=q, minlength=self.bins)
        return gaussian_filter1d(hist, self.smooth_bins, mode="wrap") / (length / self.bins)

    def density_at(self, density: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = (z % self.cfg.circumference_m) / self.cfg.circumference_m * self.bins - 0.5
        i = np.floor(x).astype(int)
        frac = x - i
        return density[i % self.bins] * (1 - frac) + density[(i + 1) % self.bins] * frac

    def path_overlap(self, z: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
        """Length (m) of each particle's step segment inside any lens."""
        length = self.cfg.circumference_m
        lo_seg, hi_seg = z - v * dt / 2, z + v * dt / 2
        total = np.zeros_like(z)
        for start, extent in zip(self.start, self.lengths):
            for shift in (-length, 0., length):
                total += np.clip(np.minimum(hi_seg, start + shift + extent)
                                 - np.maximum(lo_seg, start + shift), 0, None)
        return total

    def inside(self, z: np.ndarray) -> np.ndarray:
        """True where lab position z lies in a column section."""
        zz = np.asarray(z) % self.cfg.circumference_m
        mask = np.zeros(zz.shape, bool)
        for start, extent in zip(self.start, self.lengths):
            mask |= (zz >= start) & (zz < start + extent)
        return mask

    def axis_at(self, z: np.ndarray) -> np.ndarray | None:
        """Column axis (n, 2): the closed orbit if followed, else None (the machine axis)."""
        return None if self.orbit is None else self.orbit.xy(z)

    def sigma_at(self, z: np.ndarray) -> np.ndarray | float:
        """Gaussian rms of the column profile at z (metres), following the local beam size if asked."""
        base = self.shape.get("sigma")
        if self.sigma_track is None or base is None:
            return base
        if self.cfg.column_sigma_flux:
            z = self.entrance_of(z)
        grid = self.sigma_track
        local = np.sqrt(np.sqrt(grid.emittance[0] * grid.twiss(np.asarray(z), 0)[0]
                                * grid.emittance[1] * grid.twiss(np.asarray(z), 1)[0]))
        return local * self.cfg.lens_sigma_ratio

    def matched_sigma_at(self, z: np.ndarray) -> np.ndarray:
        """Local matched proton rms at z (the column_sigma_follow envelope without lens_sigma_ratio or column_sigma_flux)."""
        grid = self.sigma_track
        return np.sqrt(np.sqrt(grid.emittance[0] * grid.twiss(np.asarray(z), 0)[0] * grid.emittance[1] * grid.twiss(np.asarray(z), 1)[0]))

    def entrance_of(self, z: np.ndarray) -> np.ndarray:
        """Lab position of the entrance of the (sub-)column containing z (straights; column_drift_resets sub-columns each)."""
        zz = np.asarray(z) % self.cfg.circumference_m
        cell = self.cfg.circumference_m / self.cfg.lattice_periods
        j = np.floor(zz / cell)
        sub = float(self.lengths[0]) / self.cfg.column_drift_resets
        k = np.clip(np.floor((zz - j * cell) / sub), 0, self.cfg.column_drift_resets - 1)
        return j * cell + k * sub

    def transverse_kick(self, xy: np.ndarray, lam_e: np.ndarray, charge: float,
                        beta_z: np.ndarray, shape: dict | None = None) -> np.ndarray:
        """Force per unit path-time ratio: F_xy (N) of line density lam_e (C/m)."""
        r2 = np.maximum(np.sum(xy**2, axis=1), 1e-30)
        factor = 2 * COULOMB * lam_e * enclosed_fraction(shape or self.shape, r2) / r2
        return charge * (1 - beta_z * self.beta_e)[:, None] * factor[:, None] * xy

    def kicks(self, beams: list[Species], meta: list[dict], dt: float) -> list:
        """Momentum increments (mc units, x and y) for every beam at the step midpoint."""
        protons = [s for s, info in zip(beams, meta) if info["role"] == "proton"]
        density = self.line_density(protons)
        scale = self.scale
        if self.drift is not None:
            self.drift.advance(protons, density)
        result = []
        for s, info in zip(beams, meta):
            if info["role"] != "proton":
                result.append(None)
                continue
            v = s.velocity_m_s[:, 2]
            ell = self.path_overlap(s.position_m[:, 2], v, dt)
            lam_e = scale * self.density_at(density, s.position_m[:, 2])
            xy, shape = s.position_m[:, :2], None
            if self.orbit is not None:
                xy = xy - self.orbit.xy(s.position_m[:, 2])
            if self.drift is not None:
                xy = xy - self.drift.at(s.position_m[:, 2])
            if self.sigma_track is not None:
                shape = dict(kind=self.cfg.column_profile, sigma=self.sigma_at(s.position_m[:, 2]), **(
                    dict(p=self.cfg.injection_painted_p) if self.cfg.column_profile == "painted" else {}))
            if self.sigma_scale != 1.0:
                base = shape or self.shape
                shape = dict(base, sigma=base["sigma"] * self.sigma_scale) if base["kind"] in ("gaussian", "waterbag", "painted") else dict(
                    base, r_in=base["r_in"] * self.sigma_scale, r_out=base["r_out"] * self.sigma_scale)
            force = self.transverse_kick(xy, lam_e, s.charge_c, v / C, shape)
            dp = force * (ell / v)[:, None] / (s.mass_kg * C)
            if self.drift is not None and self.drift.K:
                dp = dp + self.drift.side_kick(s.position_m[:, 2])
            result.append(dp)
        return result


class ColumnDrift:
    """Per-slice electron-column centroid e (Config.column_centroid_drift; report_column_drift.md).

    The electrons of a column ride with the proton slice they were born under, so their state lives on lab bins of width
    C/(steps_per_turn m) that move m bins per step (dt = T_rev/steps_per_turn exactly). advance() is called once per step at the
    step midpoint (ElectronLens.kicks): it moves the electron slices on, gives fresh electrons (e = 0 on the column axis) to the
    slices that crossed a (sub-)column entrance, and integrates de/ds = i sigma Omega (u - e) exactly for a piecewise-constant u
    (the mean of the slice's proton centroid at the two midpoints), e - u -> (e - u) exp(-i sigma Omega ds). u is relative to the
    column axis (closed orbit if followed), smoothed with a Gaussian of column_drift_smooth_m. Outside the columns e = 0.
    """

    def __init__(self, lens: "ElectronLens", cfg: Config, p: dict):
        self.lens, self.cfg = lens, cfg
        self.length = cfg.circumference_m
        self.m = int(cfg.column_drift_bins_per_step)
        self.nb = cfg.steps_per_turn * self.m
        self.dz = self.length / self.nb
        self.z = (np.arange(self.nb) + 0.5) * self.dz
        track = Racetrack(cfg, p)
        self.sigma = track.field_sign(self.z) * math.copysign(1.0, cfg.solenoid_bz_t)
        self.inside = self.sigma != 0
        cell = self.length / cfg.lattice_periods
        self.K = int(cfg.column_recentre_stations)
        nseg = cfg.column_drift_resets * (self.K + 1)
        sub = float(lens.lengths[0]) / nseg
        sub_index = np.floor((self.z % cell) / sub).clip(0, nseg - 1)
        entry = lens.entrance_of(self.z) if not self.K else np.floor(self.z / cell) * cell + sub_index * sub
        self.id = np.where(self.inside, (np.floor(self.z / cell) * nseg + sub_index).astype(int), -1)
        # path inside the current (sub-)column, capped at one step: the drift length of a slice that has just been born
        self.since = np.where(self.inside, (self.z - entry) % self.length, 0.)
        self.fresh = self.inside & (self.id != np.roll(self.id, self.m))
        self.step_path = self.m * self.dz
        b = abs(cfg.solenoid_bz_t)
        self.coef = cfg.column_drift_omega_scale / (2 * constants.epsilon_0 * b * p["beta"] * p["gamma"]**2 * C)
        self.image = 1 / (math.pi * cfg.column_drift_bore_m**2)
        self.e = np.zeros(self.nb, complex)
        self.overlap = None                # column_drift_omega_overlap: int rho_e rho_p dA on the slice grid (static)
        self.u_prev = None
        self.calls = 0
        self.last = {}
        if self.K:
            # stations: segment starts that are not guns; eh, h: feed-forward model state; kick: this step's proton side kick per bin
            self.gun = self.inside & (sub_index.astype(int) % (self.K + 1) == 0)
            self.station = self.inside & ~self.gun
            self.ff = cfg.column_recentre_mode == "feedforward"
            self.eh = np.zeros(self.nb, complex)
            self.h = np.zeros(self.nb, complex)
            self.kick = np.zeros(self.nb, complex)
            self.side = cfg.column_recentre_side_kick * abs(cfg.solenoid_bz_t) * E / (MP * C * p["gamma"] * p["beta"])   # s B/Brho = 2 s kappa
            self.pmc = p["gamma"] * p["beta"]
            self.rng = np.random.default_rng(cfg.seed * 1000003 + 7919 + cfg.column_recentre_seed)
            nid = int(self.id.max()) + 1
            off = self.rng.normal(0, cfg.column_recentre_offset_m, (nid, 2)) if cfg.column_recentre_offset_m > 0 else np.zeros((nid, 2))
            self.offset = np.where(self.inside, (off[:, 0] + 1j * off[:, 1])[np.maximum(self.id, 0)], 0)
            self.stat = dict(n=0, d2=0.0, k2=0.0)
            self.delay = int(cfg.column_recentre_delay_steps)
            self.hist = [np.zeros(self.nb, complex) for _ in range(self.delay)]     # e after the last J steps (oldest first)

    def nbar(self, sigma) -> np.ndarray:
        """int rho^2 dA of the normalised column profile with rms sigma per plane."""
        kind = self.cfg.column_profile if self.lens.sigma_track is not None else self.lens.shape["kind"]
        sigma = np.asarray(sigma, float)
        if kind == "gaussian":
            return 1 / (4 * math.pi * sigma**2)
        q = 1.0 if kind == "waterbag" else self.cfg.injection_painted_p + 1     # radial density ~ (1 - r^2/R^2)^q
        r2 = 2 * (q + 2) * sigma**2
        return (q + 1)**2 / ((2 * q + 1) * math.pi * r2)

    def slice_centroid(self, protons: list) -> np.ndarray:
        from scipy.ndimage import gaussian_filter1d
        z = np.concatenate([s.position_m[:, 2] for s in protons]) % self.length
        w = np.concatenate([s.population for s in protons])
        xy = np.concatenate([s.position_m[:, :2] for s in protons])
        if self.lens.orbit is not None:
            xy = xy - self.lens.orbit.xy(z)
        b = np.minimum((z / self.dz).astype(int), self.nb - 1)
        wsum = np.bincount(b, w, minlength=self.nb)
        num = np.bincount(b, w * xy[:, 0], minlength=self.nb) + 1j * np.bincount(b, w * xy[:, 1], minlength=self.nb)
        width = self.cfg.column_drift_smooth_m / self.dz
        if width > 0:
            wsum = gaussian_filter1d(wsum, width, mode="wrap")
            num = gaussian_filter1d(num.real, width, mode="wrap") + 1j * gaussian_filter1d(num.imag, width, mode="wrap")
        return np.where(wsum > 1e-12 * max(wsum.max(), 1e-300), num / np.where(wsum > 0, wsum, 1), 0)

    def advance(self, protons: list, density: np.ndarray) -> None:
        u = self.slice_centroid(protons)
        lam = self.lens.density_at(density, self.z)
        if self.cfg.column_drift_omega_overlap:
            if self.overlap is None:
                self.overlap = 1 / (2 * math.pi * (np.asarray(self.lens.sigma_at(self.z), float)**2 + self.lens.matched_sigma_at(self.z)**2))
            nbar = self.overlap
        else:
            nbar = self.nbar(self.lens.sigma_at(self.z))
        omega = self.coef * lam * np.maximum(nbar - self.image, 0.)
        if self.calls == 0:
            ds, uu = np.where(self.inside, np.minimum(self.since, 0.5 * self.step_path), 0.), u
        else:
            self.e = np.roll(self.e, self.m)
            if not self.K:
                self.e[self.fresh] = 0
            else:
                self.recentre(u)
            ds = np.where(self.fresh, np.minimum(self.since, self.step_path), self.step_path)
            uu = np.where(self.fresh, u, 0.5 * (np.roll(self.u_prev, self.m) + u))
        w = np.exp(-1j * self.sigma * omega * ds)
        self.e = np.where(self.inside, uu + (self.e - uu) * w, 0)
        if self.K and self.ff:
            if self.calls == 0:
                self.h = np.where(self.inside, u + self.noise(self.nb), 0)
            self.eh = np.where(self.inside, self.h + (self.eh - self.h) * w, 0)
        if self.K and self.delay:
            self.hist.append(self.e.copy()); self.hist.pop(0)
        self.u_prev = u
        self.calls += 1
        occupied = self.inside & (lam > 0.05 * lam.max())
        if occupied.any():
            self.last = dict(e_rms_m=float(np.sqrt(np.mean(np.abs(self.e[occupied])**2))),
                             u_rms_m=float(np.sqrt(np.mean(np.abs(u[occupied])**2))),
                             u_minus_e_rms_m=float(np.sqrt(np.mean(np.abs(u[occupied] - self.e[occupied])**2))),
                             omega_mean_per_m=float(np.mean(omega[occupied])),
                             omega_max_per_m=float(np.max(omega[occupied])))

    def noise(self, n: int) -> np.ndarray:
        sm = self.cfg.column_recentre_noise_m
        return self.rng.normal(0, sm, n) + 1j * self.rng.normal(0, sm, n) if sm > 0 else np.zeros(n, complex)

    def recentre(self, u: np.ndarray) -> None:
        """Fresh bins (rolled e = the state at the end of the previous segment): guns give e = 0, stations e += d (see Config)."""
        self.kick[:] = 0
        gun, st = self.fresh & self.gun, self.fresh & self.station
        self.e[gun] = 0
        if self.ff:
            self.eh = np.roll(self.eh, self.m); self.h = np.roll(self.h, self.m)
            self.eh[gun] = 0
            self.h[gun] = u[gun] + self.noise(int(gun.sum()))
        if not st.any():
            return
        if self.ff:
            meas = self.eh[st]
        elif self.delay:
            idx = (np.flatnonzero(st) - self.delay * self.m) % self.nb
            meas = self.hist[0][idx] + self.noise(int(st.sum()))
        else:
            meas = self.e[st] + self.noise(int(st.sum()))
        target = self.offset[st] + (u[st] if self.cfg.column_recentre_target == "proton" else 0)
        d = -self.cfg.column_recentre_eta * (meas - target)
        self.e[st] += d
        if self.ff:
            self.eh[st] += d
            self.h[st] = u[st] + self.noise(int(st.sum()))
        self.kick[st] = 1j * self.sigma[st] * self.side * d
        self.stat["n"] += int(st.sum()); self.stat["d2"] += float(np.sum(np.abs(d)**2)); self.stat["k2"] += float(np.sum(np.abs(self.kick[st])**2))

    def side_kick(self, z: np.ndarray) -> np.ndarray:
        """Momentum increments (mc units, x and y) of this step's station side kicks for protons at lab positions z (their bin)."""
        k = self.kick[np.minimum(((np.asarray(z) % self.length) / self.dz).astype(int), self.nb - 1)] * self.pmc
        return np.column_stack((k.real, k.imag))

    def at(self, z: np.ndarray) -> np.ndarray:
        """Electron centroid (n, 2) at lab positions z (linear interpolation between slice centres; 0 outside the columns)."""
        x = (np.asarray(z) % self.length) / self.dz - 0.5
        i = np.floor(x).astype(int)
        f = x - i
        a, b = i % self.nb, (i + 1) % self.nb
        same = self.id[a] == self.id[b]
        e = np.where(same, self.e[a] * (1 - f) + self.e[b] * f, np.where(f < 0.5, self.e[a], self.e[b]))
        return np.column_stack((e.real, e.imag))

    def describe(self) -> dict:
        return dict(bins=self.nb, bin_m=self.dz, bins_per_step=self.m, smooth_m=self.cfg.column_drift_smooth_m,
                    resets_per_column=self.cfg.column_drift_resets, bore_m=self.cfg.column_drift_bore_m,
                    omega_scale=self.cfg.column_drift_omega_scale, steps=self.calls, **self.last,
                    **(dict(stations=self.K, eta=self.cfg.column_recentre_eta, mode=self.cfg.column_recentre_mode, delay_steps=self.delay,
                            station_d_rms_m=float(np.sqrt(self.stat["d2"] / max(self.stat["n"], 1))),
                            side_kick_rms_rad=float(np.sqrt(self.stat["k2"] / max(self.stat["n"], 1))), station_hits=self.stat["n"])
                       if self.K else {}))


def landau_terms(cfg: Config, p: dict, info: dict) -> dict | None:
    """Octupole-like coefficients and chromatic slopes of one beam, or None if all are zero."""
    role = info["role"]
    if role == "electron":
        return None
    a_xx, a_xy, a_yy, xi_x, xi_y = (getattr(cfg, f"{role}_{n}")
                                    for n in ("a_xx", "a_xy", "a_yy", "xi_x", "xi_y"))
    if not (a_xx or a_xy or a_yy or xi_x or xi_y or (cfg.long_enabled() and role == "proton")):
        return None
    qs = np.ravel(p["omega_beta_hminus"] if role == "hminus" else p["omega_beta"]) / p["omega_r"]
    qx, qy = float(qs[0]), float(qs[-1])
    q0 = float(qs[0])
    # Quartic potential (A x^4/4 + B x^2 y^2/2 + C y^4/4); with force
    # F_x = -(A x^3 + B x y^2), F_y = -(C y^3 + B x^2 y) per unit mass, the first-order
    # tune shifts are dQ_x = s2 (3A Jx/4 + B Jy/2), dQ_y = s2 (B Jx/2 + 3C Jy/4),
    # s2 = sigma^2/(Q omega_r^2); invert for A, B, C. With unequal plane tunes A uses Q_x, C uses Q_y and
    # the single cross term B the mean tune (the cross detuning is then exact only on average).
    scale = q0 * p["omega_r"]**2 / cfg.sigma_m**2
    if qx == qy:
        return dict(A=4 * scale * a_xx / 3, B=2 * scale * a_xy, C=4 * scale * a_yy / 3,
                    xi=(xi_x, xi_y), omega_r=p["omega_r"])
    scale_x, scale_y = qx * p["omega_r"]**2 / cfg.sigma_m**2, qy * p["omega_r"]**2 / cfg.sigma_m**2
    return dict(A=4 * scale_x * a_xx / 3, B=(scale_x + scale_y) * a_xy, C=4 * scale_y * a_yy / 3,
                xi=(xi_x, xi_y), omega_r=p["omega_r"])


def landau_extra(s: Species, omega_beta: float, terms: dict) -> np.ndarray:
    """Restoring acceleration (per gamma, x/y) beyond the linear -omega_beta^2 (x,y) term."""
    x, y = s.position_m[:, 0], s.position_m[:, 1]
    extra = np.stack((terms["A"] * x**3 + terms["B"] * x * y**2,
                      terms["C"] * y**3 + terms["B"] * x**2 * y), axis=1)
    delta = getattr(s, "delta", None)
    ratio = getattr(s, "vs_ratio", None)
    if ratio is not None:
        # Longitudinal mode: omega_i = (omega_beta + omega_r xi delta) v_s,i / (beta0 c), the tune
        # Q0 + xi delta per ring turn of the particle itself (T_i = C / v_s,i).
        xi = np.asarray(terms["xi"])
        omega_i = (omega_beta + terms["omega_r"] * xi[None, :] * delta[:, None]) * ratio[:, None]
        extra += (omega_i**2 - omega_beta**2) * s.position_m[:, :2]
    elif delta is not None and any(terms["xi"]):
        xi = np.asarray(terms["xi"])
        # (omega_beta + omega_r xi delta)^2 - omega_beta^2, kept exact rather than linearised
        shift = terms["omega_r"] * xi[None, :] * delta[:, None]
        extra += (2 * omega_beta * shift + shift**2) * s.position_m[:, :2]
    return extra


def push_with_focusing(s: Species, electric: np.ndarray, magnetic: np.ndarray,
                       dt: float, omega_beta: float, backend: PICBackend,
                       terms: dict | None = None) -> np.ndarray:
    """F_ext=-gamma_particle*m*omega_beta²*(x,y,0); native Boris q/m seam.

    Acceleration fields use an artificial q/m=1 SI for the push only: qE/m
    plus F_ext/m, and qB/m. The actual source and observer charges stay physical.
    This applies focusing to a neutral H-minus beam with unchanged inertia.
    ``terms`` (from landau_terms) adds the octupole-like and chromatic forces.
    """
    omega_beta = np.asarray(omega_beta)
    gamma_particle = np.sqrt(1 + np.sum(s.momentum_mc**2, axis=1))
    acceleration = electric * (s.charge_c / s.mass_kg)
    acceleration[:, :2] -= gamma_particle[:, None] * omega_beta**2 * s.position_m[:, :2]
    if terms is not None:
        acceleration[:, :2] -= gamma_particle[:, None] * landau_extra(s, omega_beta, terms)
    scaled_b = magnetic * (s.charge_c / s.mass_kg)
    return backend.to_host(backend.push(s.momentum_mc, acceleration, scaled_b,
                           np.ones(len(s.position_m)), dt)).astype(float)


def with_solenoid(magnetic: np.ndarray, solenoid: tuple | None) -> np.ndarray:
    """The self-consistent magnetic field plus the solenoid's axial field (a copy when present)."""
    if solenoid is None:
        return magnetic
    field = magnetic.copy()
    field[:, 2] += solenoid[0]
    return field


def linear_residual(x: np.ndarray, dp: np.ndarray, u: np.ndarray) -> dict:
    # A full-turn linear map depends on initial position AND momentum. A fit
    # to position alone would mislabel ordinary phase advance as nonlinearity.
    matrix = np.column_stack((np.ones(len(x)), x, u))
    fit = np.linalg.lstsq(matrix, dp, rcond=None)[0]
    residual = dp - matrix @ fit
    return dict(position_slope_kg_s=fit[1:3, :].tolist(),
                momentum_mc_coefficient_kg_m_s=fit[3:, :].tolist(),
                intercept_kg_m_s=fit[0].tolist(), fit_basis="turn-start x,y,u_x,u_y, plus intercept",
                residual_rms_kg_m_s=np.sqrt(np.mean(residual**2, axis=0)).tolist())


def tune_fit(times: np.ndarray, trajectory: np.ndarray, p: dict, bare: float) -> dict:
    """Windowed FFT peak with parabolic interpolation; unaliased subturn samples."""
    if len(times) < 16 or times[-1] - times[0] < 0.75 * p["period_s"]:
        return dict(tune=None, status="insufficient_history")
    centered = trajectory - trajectory.mean(axis=0)
    power = np.sum(abs(np.fft.rfft(centered * np.hanning(len(times))[:, None], axis=0))**2, axis=1)
    frequencies = np.fft.rfftfreq(len(times), (times[-1] - times[0]) / (len(times)-1)) * p["period_s"]
    keep = np.flatnonzero((frequencies > 0.25) & (frequencies < max(6, 2*bare)))
    peak = int(keep[np.argmax(power[keep])])
    offset = 0.
    if 0 < peak < len(power)-1 and np.all(power[peak-1:peak+2] > 0):
        a, b, c = np.log(power[peak-1:peak+2])
        offset = float(np.clip(0.5*(a-c)/(a-2*b+c), -0.5, 0.5)) if a-2*b+c else 0.
    return dict(tune=float(frequencies[peak] + offset*(frequencies[1]-frequencies[0])),
                bin_width=float(frequencies[1]-frequencies[0]),
                status="short_fft_estimate", window_turns=float((times[-1]-times[0])/p["period_s"]))


def snapshot(s: Species, cfg: Config) -> dict:
    x = s.position_m[:, :2]
    slope = s.momentum_mc[:, :2] / abs(s.momentum_mc[:, 2, None])
    centered, ps = x - x.mean(axis=0), slope - slope.mean(axis=0)
    xx, pp, xp = np.mean(centered**2, axis=0), np.mean(ps**2, axis=0), np.mean(centered*ps, axis=0)
    uc = s.momentum_mc[:,:2] - s.momentum_mc[:,:2].mean(axis=0)
    uu, xu = np.mean(uc**2,axis=0), np.mean(centered*uc,axis=0)
    gamma = np.sqrt(1 + np.sum(s.momentum_mc**2, axis=1))
    r = np.linalg.norm(x, axis=1) / cfg.sigma_m
    return dict(rms_xy_m=np.sqrt(xx).tolist(), centroid_xy_m=x.mean(axis=0).tolist(),
                geometric_emittance_m_rad=np.sqrt(np.maximum(0, xx*pp-xp*xp)).tolist(),
                normalized_emittance_m=np.sqrt(np.maximum(0,xx*uu-xu*xu)).tolist(),
                halo_radial_gt3_initial_sigma=float(np.mean(r > 3)),
                halo_radial_gt5_initial_sigma=float(np.mean(r > 5)),
                population=float(s.population.sum()), source_charge_c=float(s.source_charge_c.sum()),
                kinetic_energy_j=float(np.sum(s.population*(gamma-1)*s.mass_kg*C*C)),
                momentum_kg_m_s=np.sum(s.population[:,None]*s.momentum_mc*s.mass_kg*C,axis=0).tolist())


def current_limit(cfg: Config, p: dict, collective_d: np.ndarray, population: float) -> dict:
    """Frozen-profile linear intensity extrapolation of a measured core gradient."""
    bare = cfg.plane_tunes()
    qsq = bare**2 - collective_d
    tune = [math.sqrt(v) if v > 0 else None for v in qsq]
    allowed_d = bare**2 - (bare-cfg.tune_shift_budget)**2
    limit = [population*allowed_d[i]/d if d > 1e-8 else None for i, d in enumerate(collective_d)]
    return dict(core_tune_squared_xy=qsq.tolist(), core_tune_xy=tune,
        measured_defocusing_D_xy=collective_d.tolist(), tune_shift_budget=cfg.tune_shift_budget,
        population_limit_xy=limit, current_limit_A_xy=[n*E/p["period_s"] if n is not None else None for n in limit],
        average_circulating_current_A=population*E/p["period_s"],
        definition="turn-mean near-axis stiffness at moving design bunch centres; frozen profiles and fixed eta; positive downward tune shift only",
        status_xy=["finite_frozen_profile_estimate" if d > 1e-8 else "no_downward_limit_resolved" for d in collective_d])


def drift_beam(s: Species, role: str, h: float, lon, track) -> None:
    """One half-drift of a beam: with longitudinal slip for protons, with the solenoid edge kicks if they live in the drift."""
    if track is not None and track.edge_in_drift and role != "electron":
        track.edge_drift(s, h, lon if (lon and role == "proton") else None)
    elif lon and role == "proton":
        lon.half_drift(s, h)
    else:
        s.position_m += h * s.velocity_m_s


def run(cfg: Config, directory: Path, pilot_steps: int | None = None,
        setup_hook=None, turn_hook=None) -> dict:
    cfg.validate()
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    beams, meta, p = initial_beams(cfg)
    if setup_hook:
        setup_hook(beams, meta, p)       # test seam: rewrite the loaded particles
    dyn = None
    if cfg.columns_enabled() and cfg.column_electrons == "dynamic":
        from .edyn_columns import DynamicColumns
        dyn = DynamicColumns(cfg, p, ElectronLens(cfg, p), Racetrack(cfg, p) if cfg.racetrack() else None)
        dyn.create(beams, meta)
        dyn.update(beams, meta)
    wfb = None
    if cfg.wfb_enable:
        from .wallfb import WallFeedback
        wfb = WallFeedback(cfg, p, ElectronLens(cfg, p), directory)
    solver = RingFields(cfg) if cfg.collective else None
    backend = solver.backend if solver else backend_for(cfg.backend)
    lens = ElectronLens(cfg, p) if cfg.lens_enabled() and cfg.column_electrons == "rigid" else None
    errors = LatticeErrors(cfg, p) if cfg.errors_enabled() else None
    track = Racetrack(cfg, p) if cfg.racetrack() else None
    orbit_ref, damper = None, None
    if cfg.damper_kappa is not None or cfg.diag_bins:
        from .lattice_linear import LinearLattice
        orbit_ref = LinearLattice(cfg, p, track, errors).closed_orbit()
    if cfg.damper_kappa is not None:
        from .damper import Damper
        damper = Damper(cfg, orbit_ref)
    landau = [landau_terms(cfg, p, info) for info in meta]
    lon = Longitudinal(cfg, p) if cfg.long_enabled() else None
    ramp = None
    if cfg.ramp_enabled():
        from .ramp import Ramp
        ramp = Ramp(cfg, p, lon, track, lens if dyn is None else dyn.lens, solver, landau)
        ramp.dyn = dyn
    dt = p["period_s"] / cfg.steps_per_turn
    protons = [s for s, info in zip(beams, meta) if info["role"] == "proton"]
    initial = [s.momentum_mc.copy() for s in beams]
    turn_positions = [s.position_m[:, :2].copy() for s in beams]
    collective_impulse = [np.zeros_like(s.position_m[:, :2]) for s in beams]
    tracker_indices = []
    for info in meta:
        groups = [np.flatnonzero(info["slice_id"] == sl) for sl in range(cfg.slices)]
        ntrack = min(16, *(len(group) for group in groups))
        tracker_indices.append([group[:ntrack] for group in groups])
    trace = [[] for _ in beams]
    lab_times = np.zeros(len(beams))
    times, records, timings = [], [], []
    core_samples = {}
    boundary_max = {}
    initial_energy = {}
    previous_energy = {}
    boundary_log = directory/"boundary.jsonl"
    save(directory / "provenance.json", dict(config=asdict(cfg), physics=p,
        python=sys.executable, command=sys.argv, radiation_reaction="off: diagnostic control",
        **(dict(lattice_errors=errors.describe()) if errors else {}),
        **(dict(racetrack=p.get("racetrack_match") or track.describe()) if track else {}),
        **(dict(longitudinal=lon.describe()) if lon else {}),
        **(dict(ramp=dict(final_energy_mev=cfg.ramp_final_energy_mev, time_s=cfg.ramp_time_s, shape=cfg.ramp_shape,
                          initial_compensation=ramp.f0, initial_tunes_zero_current=ramp.linear_tunes())) if ramp else {}),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), lwi_ring_version=_version(),
        core_root=str(Path(native_kernels.__file__).resolve().parents[2]),
        core_hashes=core_hashes(),
        backend_name=backend.name, kernel_precision=backend.dtype,
        external_mlx_site=os.environ.get("RING_MLX_SITE"),
        initial=[dict(name=s.name, **{k:v for k,v in info.items() if k != "slice_id"},
                      population=float(s.population.sum()), charge_c=float(s.source_charge_c.sum()))
                 for s, info in zip(beams, meta)]))
    initialization_s = time.monotonic() - start
    def record(step: int) -> None:
        if solver:
            boundary = solver.boundary(solver.groups(beams, meta, ramp.t if ramp else step*dt))
        else:
            boundary = {}
        row = dict(step=step, turn=step/cfg.steps_per_turn, lab_time_s=ramp.t if ramp else step*dt, bunches={}, beams={})
        if cfg.diag_bins:
            from .damper import coherent_row
            row["coherent"] = coherent_row(beams, meta, orbit_ref, cfg.diag_bins, cfg)
        if lens is not None and lens.drift is not None:
            row["column_drift"] = lens.drift.describe()
        p_now = dict(p, period_s=cfg.circumference_m / (ramp.beta * C)) if ramp else p
        if ramp:
            row["ramp"] = ramp.diagnose(lens.ramp_diagnostics(protons, ramp.beta) if lens else None, protons)
        if lon:
            row["longitudinal"] = lon.diagnose([s for s, info in zip(beams, meta) if info["role"] == "proton"])
        if dyn:
            row["electrons"] = dyn.diagnose(beams)
        for i, (s, info) in enumerate(zip(beams, meta)):
            if info["role"] == "electron":
                continue
            entry = snapshot(s, cfg)
            entry["slices"] = []
            for sl in range(cfg.slices):
                mask = info["slice_id"] == sl
                if mask.sum() < 4:
                    entry["slices"].append(dict(slice=sl, status="too_few_particles"))
                    continue
                dp = (s.momentum_mc[mask,:2] - initial[i][mask,:2])*s.mass_kg*C
                history = np.asarray(trace[i])[:,sl,:,:] if trace[i] else None
                entry["slices"].append(dict(slice=sl, count=int(mask.sum()),
                    net_dp_linear_fit=linear_residual(turn_positions[i][mask], dp, initial[i][mask,:2]),
                    collective_impulse_linear_fit=linear_residual(turn_positions[i][mask], collective_impulse[i][mask], initial[i][mask,:2]),
                    tune_xy=[tune_fit(np.asarray(times), history[:,axis,:], p_now, cfg.plane_tunes()[axis])
                             for axis in (0,1)] if history is not None else []))
            row["bunches"][s.name] = entry
        for role in ("proton", "hminus"):
            group = [s for s,info in zip(beams,meta) if info["role"] == role]
            if group:
                aggregate = Species(role, np.concatenate([s.position_m for s in group]),
                    np.concatenate([s.momentum_mc for s in group]), group[0].charge_c,
                    group[0].mass_kg, np.concatenate([s.population for s in group]))
                entry = snapshot(aggregate, cfg)
                energy = entry["kinetic_energy_j"]
                initial_energy.setdefault(role, energy)
                previous_energy.setdefault(role, energy)
                entry["kinetic_energy_change_from_initial_j"] = energy-initial_energy[role]
                entry["kinetic_energy_change_this_turn_j"] = energy-previous_energy[role]
                entry["mean_kinetic_energy_change_from_initial_ev"] = (energy-initial_energy[role])/entry["population"]/E
                previous_energy[role] = energy
                entry["transverse_boundary"] = boundary.get(role)
                entry["max_outside_fraction_this_turn"] = boundary_max.get(role, 0.)
                if core_samples.get(role):
                    entry["core_current_limit"] = current_limit(cfg,p,np.mean(core_samples[role],axis=0),entry["population"])
                row["beams"][role] = entry
        core_samples.clear()
        boundary_max.clear()
        records.append(row)
        # Small reduction rewritten atomically; particle history only in local output.
        save(directory / "turns.json", records)
    record(0)
    cadence = cfg.steps_per_turn // cfg.tune_samples_per_turn
    target = cfg.turns * cfg.steps_per_turn
    total = min(target, pilot_steps) if pilot_steps else target
    last_progress = time.monotonic()
    for step in range(1, total+1):
        if cfg.wall_limit_s and time.monotonic()-start > cfg.wall_limit_s:
            raise TimeoutError(f"bounded run exceeded {cfg.wall_limit_s} s")
        if step % 64 == 1 and Path("/private/tmp/compute_cap").exists():
            while int(Path("/private/tmp/compute_cap").read_text()) < int(os.environ.get("RING_REQUIRED_CAP", "2")):
                if cfg.wall_limit_s:
                    raise RuntimeError("compute cap does not permit bounded check")
                print("Waiting for compute_cap headroom", flush=True)
                time.sleep(10)
        tick = time.monotonic()
        if ramp:
            dt = ramp.begin_step(protons)
            t_mid, t_end = ramp.t_mid(), ramp.t + dt
        else:
            t_mid, t_end = (step-0.5)*dt, step*dt
        for s, info in zip(beams, meta):
            drift_beam(s, info["role"], 0.5 * dt, lon, track)
            s.position_m[:,2] %= cfg.circumference_m
        if lon:
            lon.half_ref(dt)
            if solver:
                solver.z_shift = lon.shift(t_mid)
            for s, info in zip(beams, meta):
                if info["role"] == "proton":
                    lon.refresh(s)
        if dyn:
            dyn.update(beams, meta)
        try:
            electric, magnetic, timing = solver.fields(beams, meta, t_mid, measure_core=True) if solver else (
                [np.zeros_like(s.position_m) for s in beams], [np.zeros_like(s.position_m) for s in beams], {})
            if solver:
                for role, value in solver.last_core.items():
                    core_samples.setdefault(role, []).append(value)
                for role, value in solver.last_boundary.items():
                    boundary_max[role] = max(boundary_max.get(role, 0.), value["outside_fraction"])
                with boundary_log.open("a") as log:
                    log.write(json.dumps(dict(step=step, phase="midpoint", beams=solver.last_boundary))+"\n")
        except ValueError as error:
            save(directory/"abort.json",dict(step=step, reason=str(error), boundary=solver.last_boundary if solver else {}))
            raise
        if wfb:
            wfb.apply(beams, meta, electric, t_mid, step, solver.z_shift if solver else 0.0)
        push_start = time.monotonic()
        # All fields use the same midpoint state; every species gets the same dt.
        omegas = [track.omega(s,dt) if track else (ramp.omega(omega_beta_of(p,info)) if ramp else omega_beta_of(p,info))
                  for s,info in zip(beams,meta)]
        landau_step = [ramp.landau_terms(terms) for terms in landau] if ramp else landau
        sols = [track.solenoid(s,dt) if track and info["role"] != "electron" else None for s, info in zip(beams, meta)]
        updated = [dyn.push(s,e,b,dt) if info["role"] == "electron" else
                   push_with_focusing(s,e,with_solenoid(b,sol),dt,om,backend,terms)
                   for s,e,b,om,sol,terms,info in zip(beams,electric,magnetic,omegas,sols,landau_step,meta)]
        lens_du = lens.kicks(beams, meta, dt) if lens else [None]*len(beams)
        for u, du in zip(updated, lens_du):
            if du is not None:
                u[:,:2] += du
        error_du = errors.kicks(beams, meta, dt) if errors else [None]*len(beams)
        if dyn:
            error_du = [None if info["role"] == "electron" else du for info, du in zip(meta, error_du)]
        for u, du in zip(updated, error_du):
            if du is not None:
                u[:,:2] += du[0]
        damp_du = damper.kicks(beams, meta, dt, step) if damper else [None]*len(beams)
        for u, du in zip(updated, damp_du):
            if du is not None:
                u[:,:2] += du[0]
        for u, sol in zip(updated, sols):
            if sol is not None:
                u[:,:2] += sol[1]
        if lon:
            for s, u, info in zip(beams, updated, meta):
                if info["role"] == "proton":
                    lon.kick(s, dt, u)
            lon.update_ref(dt)
        for i, (s, u) in enumerate(zip(beams, updated)):
            if meta[i]["role"] != "electron":
                gamma = np.sqrt(1 + np.sum(s.momentum_mc**2, axis=1))
                external_dp = -gamma[:,None]*s.mass_kg*omegas[i]**2*s.position_m[:,:2]*dt
                if landau_step[i] is not None:
                    external_dp = external_dp - gamma[:,None]*s.mass_kg*landau_extra(s,omegas[i],landau_step[i])*dt
                if sols[i] is not None:
                    v = s.velocity_m_s
                    external_dp = external_dp + s.charge_c*sols[i][0][:,None]*np.column_stack((v[:,1],-v[:,0]))*dt \
                        + sols[i][1]*s.mass_kg*C
                if lens_du[i] is not None:
                    external_dp = external_dp + lens_du[i]*s.mass_kg*C
                if error_du[i] is not None:
                    external_dp = external_dp + error_du[i][0]*s.mass_kg*C
                if damp_du[i] is not None:
                    external_dp = external_dp + damp_du[i][0]*s.mass_kg*C
                collective_impulse[i] += (u[:,:2]-s.momentum_mc[:,:2])*s.mass_kg*C - external_dp
            s.momentum_mc = u
            if error_du[i] is not None:
                s.position_m[:,:2] += error_du[i][1]
            if damp_du[i] is not None:
                s.position_m[:,:2] += damp_du[i][1]
            if sols[i] is not None:
                s.position_m[:,:2] += sols[i][2]
            drift_beam(s, meta[i]["role"], 0.5 * dt, lon, track)
            s.position_m[:,2] %= cfg.circumference_m
            if not (np.isfinite(s.position_m).all() and np.isfinite(u).all()):
                raise ValueError("nonfinite state")
            lab_times[i] += dt
        if lon:
            lon.half_ref(dt)
        if ramp:
            ramp.end_step()
        if damper:
            damper.end_step(step)
        if solver:
            try:
                solver.boundary(solver.groups(beams, meta, t_end))
            except ValueError as error:
                save(directory/"abort.json",dict(step=step, reason=str(error), boundary=solver.last_boundary))
                raise
            with boundary_log.open("a") as log:
                log.write(json.dumps(dict(step=step, phase="end", beams=solver.last_boundary))+"\n")
            for role, value in solver.last_boundary.items():
                boundary_max[role] = max(boundary_max.get(role, 0.), value["outside_fraction"])
        timing.update(push_drift_s=time.monotonic()-push_start, step_s=time.monotonic()-tick)
        timings.append(timing)
        if time.monotonic()-last_progress >= 60:
            recent = float(np.median([t["step_s"] for t in timings[-128:]]))
            progress = dict(step=step, target_steps=total, turn=step/cfg.steps_per_turn,
                            elapsed_s=time.monotonic()-start, recent_step_s=recent,
                            remaining_estimate_s=(total-step)*recent)
            save(directory/"progress.json",progress)
            print(f"step {step}/{total}; {recent:.3f} s/step; remaining estimate {progress['remaining_estimate_s']/86400:.2f} days",flush=True)
            last_progress = time.monotonic()
        if step % cadence == 0:
            times.append(t_end)
            for i,s in enumerate(beams):
                trace[i].append(np.asarray([s.position_m[indices,:2].T
                                           for indices in tracker_indices[i]]))
            # Retain a rolling two-turn tune window.
            if len(times) > 2*cfg.tune_samples_per_turn:
                times.pop(0)
                for hist in trace:
                    hist.pop(0)
        if step % cfg.steps_per_turn == 0:
            if wfb:
                wfb.end_turn(step // cfg.steps_per_turn)
            record(step)
            if turn_hook:
                turn_hook(step, beams, lon)
            if cfg.injection_paint_turns > 1:
                injection.apply_births(beams, [np.arange(len(s.position_m)) for s in beams], step // cfg.steps_per_turn)
            initial = [s.momentum_mc.copy() for s in beams]
            turn_positions = [s.position_m[:,:2].copy() for s in beams]
            for impulse in collective_impulse:
                impulse.fill(0)
            print(f"turn {step//cfg.steps_per_turn}/{cfg.turns}: {time.monotonic()-start:.1f} s", flush=True)
    if total % cfg.steps_per_turn:
        record(total)
    np.savez(directory / "final_particles.npz", **{
        f"{s.name}_{name}": value for s in beams for name,value in (
            ("position_m",s.position_m),("momentum_mc",s.momentum_mc),("population",s.population))})
    result = dict(status="complete" if total==target else "pilot_only", executed_steps=total,
                  target_steps=target, elapsed_s=time.monotonic()-start, initialization_s=initialization_s,
                  target_lab_time_s=target*dt,
                  final_lab_times_s={s.name:float(t) for s,t in zip(beams,lab_times)},
                  timing_mean_s={k:float(np.mean([t.get(k,0) for t in timings])) for k in timings[0]},
                  accounting="source N*q, observer q, inertia m, diagnostics N; external work, fields, radiation, and walls not closed")
    if damper:
        result["damper"] = dict(damper.describe(), **damper.stats())
    save(directory / "summary.json", result)
    return result


def _version() -> str:
    from . import __version__
    return __version__


def main(argv: list[str] | None = None) -> None:
    """lwi-ring-driver: one ring_driver.run case (the copies' ring_driver.py command line, without --checks)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="override a Config field (JSON value)")
    parser.add_argument("--backend", choices=("cpu", "mlx"))
    parser.add_argument("--opposing", choices=("charged", "neutral", "absent"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refine", choices=("dt", "longitudinal", "transverse", "particles"),
                        help="one independent refinement: dt/2, z cells x2, xy cells x1.5, or particles x4")
    parser.add_argument("--steps-per-turn", type=int)
    parser.add_argument("--pilot-steps", type=int, help="bounded initial pilot, not a turn study")
    parser.add_argument("--solver-slots", type=int, choices=(1,2,3), default=1)
    parser.add_argument("--require-acceptance", type=Path, metavar="DIR", help="apply the copies' checks.json gate from DIR")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    with solver_lock(args.solver_slots):
        base = json.loads(args.config.read_text())
        for item in args.set:
            key, value = item.split("=", 1)
            base[key] = json.loads(value)
        cfg = Config(**base)
        if args.backend:
            cfg.backend = args.backend
        if args.opposing:
            cfg.opposing = args.opposing
        if args.steps_per_turn is not None:
            cfg.steps_per_turn = args.steps_per_turn
        if args.refine == "dt":
            cfg.steps_per_turn *= 2
        elif args.refine == "longitudinal":
            cfg.grid = (*cfg.grid[:2], cfg.grid[2]*2)
        elif args.refine == "transverse":
            cfg.grid = (int(cfg.grid[0]*1.5), int(cfg.grid[1]*1.5), cfg.grid[2])
        elif args.refine == "particles":
            cfg.particles_per_bunch *= 4
        if args.require_acceptance:
            require_acceptance(args.require_acceptance)
        if args.pilot_steps is not None and args.pilot_steps < 1:
            raise ValueError("pilot steps must be positive")
        run(cfg, args.output or OUT / time.strftime("%Y%m%d_%H%M%S_ring"), args.pilot_steps)


if __name__ == "__main__":
    main()
