"""Longitudinal RF and synchrotron motion for the ring PIC drivers (protons only).

Phase convention: a particle at arc position z has RF phase offset
psi = -2 pi h (z - z_ref) / C (folded to one bucket), so a late particle (behind the
reference) has psi > 0 and sees the voltage V(phi_s + psi) per turn, with
V(phi) = V1 sin(phi) + V2 sin(n2 phi + phi2). Per turn, in the small-amplitude limit,
    d psi / dn = 2 pi h eta delta,   d delta / dn = k [V(phi_s + psi) - V(phi_s)],
k = e / (beta^2 gamma m c^2), eta = 1/gamma_t^2 - 1/gamma^2 (slip factor). The motion
conserves W = A delta^2 + Phi(psi), A = pi h |eta|, Phi(psi) = s k int_0^psi [V - V(phi_s)],
s = -sign(eta); the separatrix is the lower barrier of Phi.

In the PIC the kicks are applied each step (distributed RF, dt / T_ref of the turn gain),
du_z = d gamma / beta_z, so the space-charge E_z from the Boris push acts on the same
u_z. Slip enters through the arc speed ds/dt = v_z / (1 + alpha delta), alpha = 1/gamma_t^2.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import constants
from scipy.optimize import brentq
from scipy.special import erf, ndtri
from scipy.stats import qmc

E = constants.elementary_charge
C = constants.c
MP = constants.proton_mass


class Waveform:
    def __init__(self, v1: float, n2: int, v2: float, phase2_deg: float):
        self.v1, self.n2, self.v2, self.phi2 = v1, n2, v2, math.radians(phase2_deg)

    def volts(self, phi):
        return self.v1 * np.sin(phi) + self.v2 * np.sin(self.n2 * phi + self.phi2)

    def slope(self, phi):
        return self.v1 * np.cos(phi) + self.v2 * self.n2 * np.cos(self.n2 * phi + self.phi2)


def synchronous_phase(wave: Waveform, target_v: float, eta_sign: float, guess: float) -> float:
    """Root of V(phi) = target nearest the guess with eta * V'(phi) < 0 (phase-stable)."""
    shift = 3.1e-4                                      # keeps grid points off exact roots
    grid = np.linspace(guess - math.pi + shift, guess + math.pi + shift, 20001)
    f = wave.volts(grid) - target_v
    roots = []
    for i in np.flatnonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0):
        r = brentq(lambda x: wave.volts(x) - target_v, grid[i], grid[i + 1], xtol=1e-14)
        if eta_sign * wave.slope(r) < 0:
            roots.append(r)
    if not roots:
        raise ValueError("no phase-stable synchronous phase: RF voltage too small for the requested ramp")
    return float(min(roots, key=lambda r: abs(r - guess)))


class Bucket:
    """Potential well of the (ψ, δ) motion at the current reference energy."""

    def __init__(self, wave: Waveform, phi_s: float, eta: float, h: int, gamma: float):
        beta2 = 1 - gamma**-2
        self.k = E / (beta2 * gamma * MP * C**2)        # delta gained per turn per volt
        self.A = math.pi * h * abs(eta)
        self.s = -math.copysign(1.0, eta)
        n = 40001
        self.psi = np.linspace(-2 * math.pi, 2 * math.pi, n)
        g = wave.volts(phi_s + self.psi) - wave.volts(phi_s)
        d = self.psi[1] - self.psi[0]
        phi = np.concatenate(([0], np.cumsum(0.5 * (g[1:] + g[:-1]) * d))) * self.s * self.k
        self.phi = phi - np.interp(0.0, self.psi, phi)
        mid = n // 2
        slope = self.s * g                             # Phi' / k
        # Phi' = s k g is > 0 just right of the minimum and < 0 just left of it; the first sign
        # change on each side marks the barrier top.
        right = mid + 1 + int(np.argmax(slope[mid + 1:] < 0)) if np.any(slope[mid + 1:] < 0) else n - 1
        left = mid - 1 - int(np.argmax(slope[mid - 1::-1] > 0)) if np.any(slope[mid - 1::-1] > 0) else 0
        # the first local maxima of Phi on each side; the lower one is the separatrix
        self.w_sep = float(min(self.phi[right], self.phi[left]))
        self.delta_max = math.sqrt(max(self.w_sep, 0) / self.A)
        self.psi_unstable = (float(self.psi[left]), float(self.psi[right]))

    def well(self, w_cut: float) -> tuple[np.ndarray, np.ndarray]:
        """Contiguous psi grid around 0 with Phi < w_cut."""
        mid = len(self.psi) // 2
        inside = self.phi < w_cut
        hi = mid + (int(np.argmin(inside[mid:])) if not inside[mid:].all() else len(inside) - mid)
        lo = mid - (int(np.argmin(inside[mid::-1])) if not inside[mid::-1].all() else mid + 1) + 1
        return self.psi[lo:hi], self.phi[lo:hi]


def matched_sample(bucket: Bucket, fill: float, target_rms_psi: float, count: int, seed: int) -> dict:
    """Quasi-random sample of f(W) ∝ exp(-W/W0) truncated at W = fill W_sep, W0 giving the target rms psi."""
    w_cut = fill * bucket.w_sep
    psi, phi = bucket.well(w_cut)
    dpsi = psi[1] - psi[0]

    def marginal(w0):
        sigma = math.sqrt(w0 / (2 * bucket.A))
        b = np.sqrt(np.maximum(w_cut - phi, 0) / bucket.A)
        rho = np.exp(-phi / w0) * erf(b / (sigma * math.sqrt(2)))
        return rho, sigma, b

    def rms(w0):
        rho = marginal(w0)[0]
        rho = rho / rho.sum()
        m = (rho * psi).sum()
        return math.sqrt((rho * (psi - m) ** 2).sum())

    lo, hi = w_cut * 1e-6, w_cut * 1e6
    if rms(hi) < target_rms_psi:
        raise ValueError(f"target rms {target_rms_psi:.3f} rad exceeds what fill={fill} of this bucket allows "
                         f"({rms(hi):.3f} rad uniform fill): lengthen the RF or raise rf_fill")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if rms(mid) < target_rms_psi else (lo, mid)
    w0 = math.sqrt(lo * hi)
    rho, sigma, b = marginal(w0)
    cdf = np.cumsum(rho)
    cdf = (cdf - 0.5 * rho) / cdf[-1]
    half = count // 2
    m = int(math.log2(half))
    if 2 ** m != half:
        raise ValueError("particle count must be a power of two")
    u = qmc.Sobol(2, scramble=True, seed=seed).random_base2(m)
    psi_s = np.interp(u[:, 0], cdf, psi)
    bound = np.interp(psi_s, psi, b)
    ee = erf(bound / (sigma * math.sqrt(2)))
    delta = sigma * ndtri(0.5 * (1 - ee) + u[:, 1] * ee)
    return dict(psi=np.concatenate((psi_s, psi_s)), delta=np.concatenate((delta, -delta)),
                w0=w0, w_cut=w_cut, sigma_delta=sigma, achieved_rms_psi=rms(w0))


class Longitudinal:
    """Reference-particle state and per-step RF/slip operations for the proton beam."""

    def __init__(self, cfg, p: dict):
        self.cfg, self.p = cfg, p
        self.h = cfg.rf_harmonic or cfg.harmonic
        self.alpha = 1 / cfg.gamma_t**2 if cfg.gamma_t else 0.0
        self.beta0 = p["beta"]
        self.gamma = p["gamma"]
        self.z_ref = 0.0
        self.wave = Waveform(cfg.rf_voltage_v, cfg.rf_h2_ratio, cfg.rf_h2_voltage_v, cfg.rf_h2_phase_deg)
        self.de_turn_j = (cfg.rf_dE_turn_ev or 0.0) * E
        self.vs_beta = self.beta0                       # ramp: reference beta used by the chromatic arc-speed ratio
        if cfg.ramp_final_energy_mev is not None:       # the bucket at t = 0 is that of the programme's first turn
            from .ramp import initial_rf_state
            de_ev, scale = initial_rf_state(cfg, p)
            self.de_turn_j = de_ev * E
            self.wave.v1, self.wave.v2 = self.wave.v1 * scale, self.wave.v2 * scale
        self.has_rf = cfg.rf_voltage_v > 0
        eta0 = self.alpha - 1 / self.gamma**2
        if self.has_rf:
            self.phi_s = synchronous_phase(self.wave, self.de_turn_j / E, math.copysign(1.0, eta0),
                                           math.radians(cfg.rf_phase_s_deg))
        else:
            self.phi_s = 0.0
        self.pending_dgamma = 0.0

    @property
    def beta(self) -> float:
        return math.sqrt(1 - self.gamma**-2)

    @property
    def u_ref(self) -> float:
        return math.sqrt(self.gamma**2 - 1)

    @property
    def eta(self) -> float:
        return self.alpha - 1 / self.gamma**2

    def bucket(self) -> Bucket:
        return Bucket(self.wave, self.phi_s, self.eta, self.h, self.gamma)

    def delta(self, s) -> np.ndarray:
        return np.abs(s.momentum_mc[:, 2]) / self.u_ref - 1

    def half_drift(self, s, h: float) -> None:
        """Advance positions by h, with the arc speed reduced by the path-length factor."""
        v = s.velocity_m_s
        if self.alpha:
            v[:, 2] = v[:, 2] / (1 + self.alpha * self.delta(s))
        s.position_m += h * v

    def half_ref(self, dt: float) -> None:
        self.z_ref += 0.5 * dt * self.beta * C

    def psi_of(self, z: np.ndarray) -> np.ndarray:
        period = self.cfg.circumference_m / self.h
        d = (z - self.z_ref + period / 2) % period - period / 2
        return -2 * math.pi * self.h * d / self.cfg.circumference_m

    def refresh(self, s) -> None:
        """Dynamic delta and arc-speed ratio, used by the chromatic focusing."""
        delta = self.delta(s)
        s.delta = delta
        s.vs_ratio = s.velocity_m_s[:, 2] / (1 + self.alpha * delta) / (self.vs_beta * C)

    def kick(self, s, dt: float, u: np.ndarray) -> None:
        """Gain of this step, V(phi_s + psi) dt / T_ref, added to u[:, 2] (the updated momentum of s,
        evaluated at the step-midpoint position and velocity of s)."""
        if not self.has_rf:
            return
        t_ref = self.cfg.circumference_m / (self.beta * C)
        volts = self.wave.volts(self.phi_s + self.psi_of(s.position_m[:, 2]))
        dgamma = E * volts / (MP * C**2) * dt / t_ref
        u[:, 2] += dgamma * C / s.velocity_m_s[:, 2]

    def update_ref(self, dt: float) -> None:
        if self.de_turn_j:
            t_ref = self.cfg.circumference_m / (self.beta * C)
            self.gamma += self.de_turn_j / (MP * C**2) * dt / t_ref

    def shift(self, lab_time: float) -> float:
        """Reference arc position minus the fixed design-speed position (grid centre offset)."""
        return self.z_ref - self.beta0 * C * lab_time

    def state(self) -> dict:
        return dict(gamma=self.gamma, z_ref=self.z_ref)

    def restore(self, state: dict) -> None:
        self.gamma, self.z_ref = float(state["gamma"]), float(state["z_ref"])

    def describe(self) -> dict:
        b = self.bucket() if self.has_rf else None
        return dict(harmonic=self.h, alpha_c=self.alpha, eta=self.eta, phi_s_rad=self.phi_s,
                    rf_voltage_v=self.cfg.rf_voltage_v, dE_turn_ev=self.de_turn_j / E,
                    qs_small_amplitude=synchrotron_tune(self) if self.has_rf else None,
                    bucket_w_sep=b.w_sep if b else None, bucket_height_delta=b.delta_max if b else None,
                    psi_unstable=b.psi_unstable if b else None)

    def diagnose(self, beams) -> dict:
        """Turn diagnostics of the proton beam: size, momentum spread, emittance proxy, losses from the bucket."""
        z = np.concatenate([s.position_m[:, 2] for s in beams])
        delta = np.concatenate([self.delta(s) for s in beams])
        w = np.concatenate([s.population for s in beams])
        psi = self.psi_of(z)
        mean = lambda a: float(np.average(a, weights=w))
        pm, dm = mean(psi), mean(delta)
        pp, dd, pd = mean((psi - pm) ** 2), mean((delta - dm) ** 2), mean((psi - pm) * (delta - dm))
        out = dict(psi_mean_rad=pm, psi_rms_rad=math.sqrt(pp), delta_mean=dm, delta_rms=math.sqrt(dd),
                   z_rms_m=math.sqrt(pp) * self.cfg.circumference_m / (2 * math.pi * self.h),
                   emittance_proxy_rad=math.sqrt(max(pp * dd - pd * pd, 0)),
                   gamma_ref=self.gamma, z_ref_m=self.z_ref)
        if self.has_rf:
            b = self.bucket()
            wp = b.A * delta**2 + np.interp(psi, b.psi, b.phi)
            out["outside_separatrix_fraction"] = float(np.average(wp > b.w_sep, weights=w))
            out["bucket_height_delta"] = b.delta_max
        return out


def synchrotron_tune(lon: Longitudinal) -> float:
    """Small-amplitude Qs from the local slope: sqrt(2 A k V'(phi_s)) / 2 pi."""
    beta2 = 1 - lon.gamma**-2
    k = E / (beta2 * lon.gamma * MP * C**2)
    return math.sqrt(2 * math.pi * lon.h * abs(lon.eta) * k * abs(lon.wave.slope(lon.phi_s))) / (2 * math.pi)
