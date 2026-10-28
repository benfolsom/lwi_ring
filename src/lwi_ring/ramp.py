"""Energy ramp for the compensated rapid-cycling synchrotron variant (stage 2b).

Everything here is inert unless ``Config.ramp_final_energy_mev`` is set. One ``Ramp`` object owns the
programmed time dependence of a run and is advanced once per step by the two run loops
(``ring_driver.run`` and ``many_turn.run``):

* the reference energy follows a programme gamma(t) (linear in kinetic energy, or a half-sine in momentum,
  i.e. in the dipole field) from ``energy_mev`` to ``ramp_final_energy_mev`` in ``ramp_time_s``, then holds;
* the step is dt = T_rev(gamma)/steps_per_turn, so a "turn" stays steps_per_turn steps while the revolution
  period shortens (T_rev = C/(beta c));
* the RF follows: the energy gain per turn of the reference particle is set so that gamma tracks the
  programme exactly, the synchronous phase is re-solved from V(phi_s) = dE_turn/e every step, and the voltage
  is constant, held above a minimum-phase floor ("min_phase"), or tabulated;
* quadrupole-like (smooth) focusing is ramped with the rigidity, i.e. it keeps a constant wave number k
  (omega_beta = Q beta c 2 pi/C follows beta); the octupole and chromatic knobs scale the same way;
* in a solenoid lattice the DC solenoid's Larmor wave number K = qB/(2p) falls as 1/(beta gamma); the
  ramped quadrupole in the straights makes up the deficit so that k_q^2 + K^2 stays at its injection value
  (the tunes stay put). ``ramp_solenoid_follow`` = w in [0, 1] ramps the solenoid field itself as
  B(t) = B0 [(1 - w) + w (beta gamma)/(beta gamma)_0]: w = 0 is DC, w = 1 a fully ramped solenoid (no quads needed);
* the electron columns follow the proton velocity (beta_e = beta_p, optionally with a relative error), the
  proton line density (the rigid electron line density is the proton macroparticle density), a programmed
  compensation fraction f(t) and a programmed transverse size;
* the field solver's longitudinal Green function is rebuilt when gamma has moved by ``ramp_regrid_tol``.

What this does not model: the induced azimuthal electric field of a ramping solenoid, power-supply ripple,
the gun voltage offset needed to cancel the electron's own potential well (the nominal electron energy
(gamma - 1) m_e c^2 is reported), eddy-current or magnet-saturation effects, and the injection process.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from scipy import constants

E = constants.elementary_charge
C = constants.c
MP = constants.proton_mass
ME = constants.electron_mass
RP = E**2 / (4 * math.pi * constants.epsilon_0 * MP * C**2)        # classical proton radius

SHAPES = ("linear", "sine")
RF_PROGRAMMES = ("constant", "min_phase", "table")
COMP_PROGRAMMES = ("constant", "taper", "table")
COMP_SCALINGS = ("adiabatic", "fixed_size")
SIGMA_MODES = ("fixed", "adiabatic", "measured")


def gamma_of_energy(energy_mev: float) -> float:
    return 1 + energy_mev * 1e6 * E / (MP * C**2)


def validate(cfg) -> None:
    """Config checks of the ramp options (all inert while ramp_final_energy_mev is None)."""
    if cfg.ramp_final_energy_mev is None:
        if (cfg.ramp_time_s or cfg.ramp_comp_programme != "constant" or cfg.rf_programme != "constant"
                or cfg.ramp_solenoid_follow or cfg.ramp_electron_beta_error or cfg.ramp_column_sigma != "fixed"
                or cfg.ramp_quad_scale != 1.0):
            raise ValueError("ramp options need ramp_final_energy_mev")
        return
    if not (math.isfinite(cfg.ramp_final_energy_mev) and cfg.ramp_final_energy_mev > cfg.energy_mev):
        raise ValueError("ramp_final_energy_mev must exceed energy_mev")
    if not (math.isfinite(cfg.ramp_time_s) and cfg.ramp_time_s > 0):
        raise ValueError("a ramp needs ramp_time_s > 0")
    if cfg.ramp_shape not in SHAPES or cfg.rf_programme not in RF_PROGRAMMES:
        raise ValueError(f"ramp_shape: {SHAPES}; rf_programme: {RF_PROGRAMMES}")
    if cfg.ramp_comp_programme not in COMP_PROGRAMMES or cfg.ramp_comp_scaling not in COMP_SCALINGS \
            or cfg.ramp_column_sigma not in SIGMA_MODES:
        raise ValueError(f"ramp_comp_programme: {COMP_PROGRAMMES}; ramp_comp_scaling: {COMP_SCALINGS}; ramp_column_sigma: {SIGMA_MODES}")
    if not cfg.long_enabled() or not cfg.rf_voltage_v:
        raise ValueError("a ramp needs gamma_t and an RF voltage")
    if cfg.rf_dE_turn_ev is not None:
        raise ValueError("the ramp sets the energy gain per turn itself: leave rf_dE_turn_ev unset")
    if cfg.opposing != "absent" or cfg.column_electrons == "dynamic":
        raise ValueError("the ramp is implemented for a proton-only beam with rigid columns or lenses")
    if cfg.ramp_comp_programme != "constant" and not cfg.lens_enabled():
        raise ValueError("a compensation programme needs electron columns or lenses")
    if cfg.ramp_comp_ring_average and not (cfg.columns_enabled() and cfg.column_local_fraction is not None):
        raise ValueError("ramp_comp_ring_average needs columns with column_local_fraction (the initial local fraction)")
    if cfg.ramp_comp_programme == "taper" and not (cfg.ramp_comp_n_unc0 and cfg.ramp_comp_n_unc0 > 0):
        raise ValueError("the taper programme needs ramp_comp_n_unc0 (N_unc at the injection energy)")
    if cfg.ramp_comp_programme == "table" and not (cfg.ramp_comp_table and all(len(r) == 2 for r in cfg.ramp_comp_table)):
        raise ValueError("ramp_comp_table: [[t_s, f], ...] ascending in time")
    if cfg.rf_programme == "table" and not (cfg.rf_voltage_table and all(len(r) == 2 for r in cfg.rf_voltage_table)):
        raise ValueError("rf_voltage_table: [[t_s, scale], ...] ascending in time")
    if not 0 <= cfg.ramp_solenoid_follow <= 1 or not cfg.ramp_quad_scale > 0:
        raise ValueError("ramp_solenoid_follow in [0, 1]; ramp_quad_scale positive")
    if cfg.ramp_solenoid_follow and not (cfg.racetrack() and cfg.solenoid_bz_t and cfg.solenoid_retune):
        raise ValueError("a ramped solenoid needs a periodic solenoid lattice with solenoid_retune")
    if cfg.racetrack() and cfg.solenoid_bz_t and not cfg.solenoid_retune:
        raise ValueError("a ramp with a solenoid lattice needs solenoid_retune (the quads make up the Larmor focusing)")
    if not 0 < cfg.ramp_regrid_tol <= 0.2:
        raise ValueError("ramp_regrid_tol in (0, 0.2]")
    if not 0 < cfg.rf_phase_max_deg < 90:
        raise ValueError("rf_phase_max_deg in (0, 90)")


def momentum_programme(cfg):
    """gamma(t) of the reference particle."""
    g0, g1 = gamma_of_energy(cfg.energy_mev), gamma_of_energy(cfg.ramp_final_energy_mev)
    p0, p1 = math.sqrt(g0**2 - 1), math.sqrt(g1**2 - 1)

    def gamma_at(t: float) -> float:
        s = min(max(t / cfg.ramp_time_s, 0.0), 1.0)
        if cfg.ramp_shape == "linear":
            return g0 + (g1 - g0) * s
        return math.sqrt(1 + (p0 + (p1 - p0) * 0.5 * (1 - math.cos(math.pi * s)))**2)
    return gamma_at


def voltage_scale(cfg, t: float, de_ev: float) -> float:
    """RF voltage scale at time t for an energy gain per turn de_ev (eV), under cfg.rf_programme."""
    if cfg.rf_programme == "constant":
        return 1.0
    if cfg.rf_programme == "table":
        tt, ss = np.asarray(cfg.rf_voltage_table, float).T
        return float(np.interp(t, tt, ss))
    # min_phase: smallest scale with the synchronous phase at or below rf_phase_max_deg
    from .longitudinal import Waveform
    base = Waveform(cfg.rf_voltage_v, cfg.rf_h2_ratio, cfg.rf_h2_voltage_v, cfg.rf_h2_phase_deg)
    v_at_max = float(base.volts(math.radians(cfg.rf_phase_max_deg)))
    return max(1.0, de_ev / v_at_max) if v_at_max > 0 else 1.0


def initial_rf_state(cfg, p: dict) -> tuple[float, float]:
    """(energy gain per turn in eV, voltage scale) of the reference at t = 0, for the matched bucket load."""
    ramp_gamma = momentum_programme(cfg)
    dt = cfg.circumference_m / (p["beta"] * C)
    de_ev = (ramp_gamma(dt) - ramp_gamma(0.0)) * MP * C**2 / E
    return de_ev, voltage_scale(cfg, 0.0, de_ev)


class Ramp:
    def __init__(self, cfg, p: dict, lon, track, lens, solver, landau, errors=None):
        validate(cfg)
        self.cfg, self.p, self.lon, self.track, self.lens, self.solver = cfg, p, lon, track, lens, solver
        self.landau = landau
        self.gamma_at = momentum_programme(cfg)
        self.t = 0.0                                   # lab time at the start of the next step
        self.beta0, self.gamma0 = p["beta"], p["gamma"]
        self.bg0 = self.beta0 * self.gamma0
        self.base_v1, self.base_v2 = cfg.rf_voltage_v, cfg.rf_h2_voltage_v
        self.v_scale = 1.0
        # programme f is the ring-average fraction when ramp_comp_ring_average; columns then carry f C/L_columns locally
        self.ring_to_local = 1.0
        if cfg.ramp_comp_ring_average and cfg.columns_enabled() and cfg.column_local_fraction is not None:
            self.ring_to_local = cfg.circumference_m / float(lens.lengths.sum())
        self.f = self._initial_fraction()
        self.f0 = self.f
        self.sigma_scale = 1.0
        self.sigma_ref = None                          # rms (x, y) at the last measurement, for "measured"
        self.beta, self.gamma = self.beta0, self.gamma0
        self.b_scale = 1.0
        self.extra_k2 = 0.0
        self.population_alive = None
        self.dt = None
        self.steps_done = 0
        self.set_state()

    # ---- programmes -------------------------------------------------------------------------------------
    def _initial_fraction(self) -> float:
        if self.lens is None:
            return 0.0
        cfg = self.cfg
        return cfg.column_local_fraction / self.ring_to_local if (cfg.columns_enabled() and cfg.column_local_fraction is not None) \
            else cfg.lens_fraction

    def n_unc(self, gamma: float) -> float:
        """Uncompensated intensity limit at the energy of gamma, scaled from its injection value."""
        beta = math.sqrt(1 - gamma**-2)
        if self.cfg.ramp_comp_scaling == "fixed_size":                  # fixed beam size: beta^2 gamma^3
            ratio = (beta**2 * gamma**3) / (self.beta0**2 * self.gamma0**3)
        else:                                                           # conserved normalised emittance: beta gamma^2
            ratio = (beta * gamma**2) / (self.beta0 * self.gamma0**2)
        return self.cfg.ramp_comp_n_unc0 * ratio

    def fraction_at(self, t: float, gamma: float) -> float:
        cfg = self.cfg
        if cfg.ramp_comp_programme == "constant":
            return self.f0
        if cfg.ramp_comp_programme == "table":
            tt, ff = np.asarray(cfg.ramp_comp_table, float).T
            return float(np.interp(t, tt, ff))
        n = cfg.ramp_comp_n_design or self.population_alive
        # residual tune shift (1 - f) dQ_unc(N) held at the uncompensated limit: f = 1 - N_unc(E)/N
        return float(min(max(1 - self.n_unc(gamma) / n, 0.0), cfg.ramp_comp_f_max))

    # ---- per-step update --------------------------------------------------------------------------------
    def set_state(self) -> None:
        """Beta, gamma and the lattice quantities at the current reference energy."""
        lon, cfg = self.lon, self.cfg
        self.gamma = lon.gamma
        self.beta = math.sqrt(1 - self.gamma**-2)
        bg = self.beta * self.gamma
        # solenoid: B(t) and the quadrupole make-up of the Larmor deficit
        w = cfg.ramp_solenoid_follow
        self.b_scale = (1 - w) + w * bg / self.bg0
        if self.track is not None and self.track.larmor:
            larmor_now = self.track.larmor * self.b_scale * self.bg0 / bg
            self.extra_k2 = cfg.ramp_quad_scale * max(self.track.larmor**2 - larmor_now**2, 0.0)
            self.larmor_now = larmor_now
        else:
            self.extra_k2, self.larmor_now = 0.0, 0.0
        if self.track is not None:
            self.track.set_ramp(self.beta, self.b_scale, self.extra_k2)

    def begin_step(self, protons: list) -> float:
        """Programme the step that starts now; returns its dt. Sets reference gain, RF, focusing, columns."""
        lon, cfg, p = self.lon, self.cfg, self.p
        live = [s for s in protons if len(s.population)]
        self.population_alive = float(sum(s.population.sum() for s in live))
        self.set_state()
        if cfg.ramp_column_sigma == "adiabatic" or (cfg.ramp_column_sigma == "measured"
                                                    and self.steps_done % cfg.steps_per_turn == 0 and live):
            self.measure_sigma(live)
        self.steps_done += 1
        t_rev = cfg.circumference_m / (self.beta * C)
        dt = t_rev / cfg.steps_per_turn
        g_next = self.gamma_at(self.t + dt)
        lon.de_turn_j = (g_next - lon.gamma) * MP * C**2 * t_rev / dt
        de_ev = lon.de_turn_j / E
        self.v_scale = voltage_scale(cfg, self.t, de_ev)
        lon.wave.v1, lon.wave.v2 = self.base_v1 * self.v_scale, self.base_v2 * self.v_scale
        from .longitudinal import synchronous_phase
        lon.phi_s = synchronous_phase(lon.wave, lon.de_turn_j / E, math.copysign(1.0, lon.eta), lon.phi_s)
        lon.vs_beta = self.beta
        if self.solver is not None:
            self.solver.set_gamma(self.gamma, cfg.ramp_regrid_tol)
        if self.lens is not None:
            self.f = self.fraction_at(self.t + 0.5 * dt, lon.gamma)
            self.lens.set_ramp(beta_e=self.beta * (1 + cfg.ramp_electron_beta_error), fraction=self.f * self.ring_to_local,
                               sigma_scale=self.sigma_scale)
        self.dt = dt
        return dt

    def end_step(self) -> None:
        self.t += self.dt

    def t_mid(self) -> float:
        return self.t + 0.5 * self.dt

    # ---- scalings for the run loops ---------------------------------------------------------------------
    def omega(self, omega) -> np.ndarray:
        """Smooth-ring omega_beta at the current beta (quadrupole strength k constant)."""
        return np.asarray(omega) * (self.beta / self.beta0)

    def landau_terms(self, terms: dict | None) -> dict | None:
        if terms is None:
            return None
        b = self.beta / self.beta0
        return dict(terms, A=terms["A"] * b**2, B=terms["B"] * b**2, C=terms["C"] * b**2, omega_r=terms["omega_r"] * b)

    def measure_sigma(self, protons: list) -> None:
        """'measured' column size: rms of the protons relative to the injection rms (once per call)."""
        if self.cfg.ramp_column_sigma == "fixed":
            return
        if self.cfg.ramp_column_sigma == "adiabatic":
            self.sigma_scale = math.sqrt(self.bg0 / (self.beta * self.gamma))
            return
        xy = np.concatenate([s.position_m[:, :2] for s in protons])
        w = np.concatenate([s.population for s in protons])
        rms = np.sqrt(np.average((xy - np.average(xy, axis=0, weights=w))**2, axis=0, weights=w))
        if self.sigma_ref is None:
            self.sigma_ref = float(np.sqrt(rms[0] * rms[1]))
        self.sigma_scale = float(np.sqrt(rms[0] * rms[1]) / self.sigma_ref)

    # ---- diagnostics ------------------------------------------------------------------------------------
    def linear_tunes(self) -> list | None:
        """Zero-current tunes of the lattice frozen at the present ramp state (transfer-matrix model)."""
        from dataclasses import replace
        from .lattice_linear import LinearLattice
        cfg, track = self.cfg, self.track
        p = dict(self.p, gamma=self.gamma, beta=self.beta)
        if track is None:
            return [float(q) for q in cfg.plane_tunes()]
        frozen = SimpleNamespace(starts=track.starts, sign=track.sign, ls=track.ls, la=track.la,
                                 k_arc=track.k_arc,
                                 k_straight=np.sqrt(track.k_straight**2 + self.extra_k2))
        frozen_cfg = replace(cfg, solenoid_bz_t=cfg.solenoid_bz_t * self.b_scale)
        return [float(q) for q in LinearLattice(frozen_cfg, p, frozen, None).tunes()]

    def canonical_momentum(self, z: np.ndarray, xy: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Canonical transverse momentum P = u + qA/(mc) in mc units, A = (B_z/2)(-y, x): removes the mechanical swirl of
        the solenoid (a beam with zero canonical angular momentum has kinetic swirl +-q B r/2 inside the field)."""
        if self.track is None or not self.cfg.solenoid_bz_t:
            return u
        kappa = self.cfg.solenoid_bz_t * self.b_scale * self.track.field_sign(z) * E / (2 * MP * C)
        return u + kappa[:, None] * np.column_stack((-xy[:, 1], xy[:, 0]))

    def core_slice(self, protons: list, half_width_m: float = 7.5) -> dict | None:
        """Transverse statistics of the protons within +-half_width_m of the reference position (the same lattice
        location every turn, so the numbers do not move with the bunch's longitudinal breathing), and the canonical
        (swirl-free) normalised emittance of the whole bunch and of the slice. Inside a DC solenoid the kinetic emittance
        of a zero-canonical-angular-momentum beam contains the mechanical swirl, which shrinks relative to the thermal
        momentum as the beam accelerates; the canonical one is the invariant."""
        live = [s for s in protons if len(s.population)]
        if not live:
            return None
        z = np.concatenate([s.position_m[:, 2] for s in live])
        xy_all = np.concatenate([s.position_m[:, :2] for s in live])
        u_all = np.concatenate([s.momentum_mc[:, :2] for s in live])
        p_all = self.canonical_momentum(z, xy_all, u_all)

        def stats(xy, u):
            xc, uc = xy - xy.mean(axis=0), u - u.mean(axis=0)
            xx, uu, xu = np.mean(xc**2, axis=0), np.mean(uc**2, axis=0), np.mean(xc * uc, axis=0)
            return xx, np.sqrt(np.maximum(0, xx * uu - xu * xu))
        rel = (z - self.lon.z_ref + self.cfg.circumference_m / 2) % self.cfg.circumference_m - self.cfg.circumference_m / 2
        mask = np.abs(rel) < half_width_m
        if mask.sum() < 16:
            return None
        xx, eps_kin = stats(xy_all[mask], u_all[mask])
        _, eps_can = stats(xy_all[mask], p_all[mask])
        _, eps_can_all = stats(xy_all, p_all)
        return dict(count=int(mask.sum()), half_width_m=half_width_m, rms_xy_m=np.sqrt(xx).tolist(),
                    normalized_emittance_m=eps_kin.tolist(), canonical_normalized_emittance_m=eps_can.tolist(),
                    bunch_canonical_normalized_emittance_m=eps_can_all.tolist(),
                    halo_gt3_initial_sigma=float(np.mean(np.linalg.norm(xy_all[mask], axis=1) > 3 * self.cfg.sigma_m)))

    def diagnose(self, lens_density: dict | None = None, protons: list | None = None) -> dict:
        gamma, beta = self.gamma, self.beta
        brho = gamma * beta * MP * C / E
        out = dict(time_s=self.t, gamma=gamma, beta=beta, kinetic_energy_mev=(gamma - 1) * MP * C**2 / E / 1e6,
                   brho_tm=brho, revolution_period_s=self.cfg.circumference_m / (beta * C),
                   reference_gain_per_turn_ev=self.lon.de_turn_j / E, rf_voltage_scale=self.v_scale,
                   rf_voltage_v=self.lon.wave.v1, synchronous_phase_deg=math.degrees(self.lon.phi_s),
                   beta_gamma_ratio=beta * gamma / self.bg0, column_sigma_scale=self.sigma_scale,
                   zero_current_tunes=self.linear_tunes(),
                   solenoid_b_scale=self.b_scale, solenoid_larmor_K_per_m=self.larmor_now,
                   quad_extra_k2_per_m2=self.extra_k2,
                   solver_kernel_gamma=self.solver.kernel_gamma if self.solver else None)
        if protons is not None:
            out["core_slice"] = self.core_slice(protons)
        if self.lens is not None:
            out.update(compensation_programmed=self.f, compensation_local_fraction=self.f * self.ring_to_local,
                       electron_kinetic_energy_ev=(gamma - 1) * ME * C**2 / E,
                       electron_beta=self.beta * (1 + self.cfg.ramp_electron_beta_error),
                       **(lens_density or {}))
            if self.cfg.ramp_comp_programme == "taper":
                out["n_unc_at_energy"] = self.n_unc(self.lon.gamma)
        return out
