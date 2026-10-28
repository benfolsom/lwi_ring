"""Transverse feedback damper and coherent-mode diagnostics (dipole-mode mitigation study, 2026-10-22).

Damper.  A slice-by-slice (azimuth-resolved) proportional transverse damper on the beam centroid.  The beam is cut into `damper_slots`
slices of equal arrival time (slot q = steps q*S ... (q+1)*S - 1 with S = steps_per_turn / slots; a step is C/steps_per_turn = 0.94 m of
path for the 480 m ring at 512 steps per turn).  The pickup, at the middle of arc `damper_pickup_arc`, records for every slot the mean
position of the particles crossing it (minus the zero-current closed orbit there, which is what a real damper removes with a high-pass
filter) plus optional Gaussian reading noise.  The kicker, at the middle of arc `damper_arc`, gives the slot that was read
shift + damper_delay_turns * steps_per_turn steps earlier (shift = path between the two arcs in steps) the slope kick
dx' = kappa_x x_read, dy' = kappa_y y_read.  The planes are independent (lab-frame x and y are uncoupled at the arcs to the few per cent
that the gradient errors cause).  Theory: the amplitude of a coherent oscillation changes by -(kappa/2) sqrt(beta_p beta_k) sin(psi) per
turn, with psi the coherent betatron phase advance from the pickup to the kick (including whole turns), so kappa = G / (sqrt(beta_p beta_k) sin psi)
damps with rate G/2 per turn, damping time 2/G turns.  scripts/csrd_lin.py computes psi and beta from the linear centroid model of the
columns, which depends on the intensity.

Diagnostics.  coherent_row() bins the protons in `diag_bins` lab-position bins and reports the rms and maximum displacement of the bin
centroids from the zero-current closed orbit (the coherent amplitude), the emittance computed after subtracting each bin's centroid and
mean momentum (what remains when the coherent motion is removed), and the rms of the bin centroids per plane.
"""
from __future__ import annotations

import math

import numpy as np


class Damper:
    def __init__(self, cfg, orbit):
        self.cfg = cfg
        self.length = cfg.circumference_m
        self.spt = cfg.steps_per_turn
        self.slots = int(cfg.damper_slots)
        self.per_slot = self.spt // self.slots
        cell = cfg.straight_length_m + cfg.arc_length_m
        mid = lambda a: (a % cfg.lattice_periods) * cell + cfg.straight_length_m + 0.5 * cfg.arc_length_m
        self.zp, self.zk = mid(cfg.damper_pickup_arc), mid(cfg.damper_arc)
        self.orbit = np.asarray(orbit.xy(np.array([self.zp]))[0]) if orbit is not None else np.zeros(2)
        step_path = self.length / self.spt
        self.lag = int(round(((self.zk - self.zp) % self.length) / step_path)) + int(cfg.damper_delay_turns) * self.spt
        if self.lag < self.per_slot:
            raise ValueError("damper: pickup-to-kicker lag shorter than a slot")
        self.kappa = np.asarray(cfg.damper_kappa, float) * np.array([c in cfg.damper_planes for c in "xy"], float)
        self.noise = float(cfg.damper_noise_m)
        self.rng = np.random.default_rng(cfg.seed + 41017)
        self.acc: dict[int, list] = {}
        self.reading: dict[int, np.ndarray] = {}
        self.max_kick = 0.0
        self.sum_kick2 = 0.0
        self.n_kick = 0

    def crossing(self, s, dt: float, z0: float):
        """Indices of the particles that cross lab position z0 in this step and the crossing position (wrap-safe, as LatticeErrors)."""
        z = s.position_m[:, 2]
        v = s.velocity_m_s[:, 2]
        lo, hi = z - v * dt / 2, z + v * dt / 2
        where = z0 + np.floor((hi - z0) / self.length) * self.length
        return np.flatnonzero(where > lo), where

    def kicks(self, beams: list, meta: list, dt: float, step: int) -> list:
        """(momentum increment in mc units, position correction in m) per beam, or None; also accumulates the pickup readings."""
        q_now = (step - 1) // self.per_slot
        q_read = (step - 1 - self.lag) // self.per_slot
        reading = self.reading.get(q_read) if step - 1 - self.lag >= 0 else None
        result = []
        for s, info in zip(beams, meta):
            if info["role"] != "proton" or len(s.population) == 0:
                result.append(None)
                continue
            u = s.momentum_mc
            z = s.position_m[:, 2]
            idx, where = self.crossing(s, dt, self.zp)
            if idx.size:
                at = s.position_m[idx, :2] + u[idx, :2] / u[idx, 2:3] * (where[idx] - z[idx])[:, None]
                a = self.acc.setdefault(q_now, [np.zeros(2), 0])
                a[0] += (at - self.orbit).sum(axis=0)
                a[1] += idx.size
            out = None
            if reading is not None and self.kappa.any():
                idk, wk = self.crossing(s, dt, self.zk)
                if idk.size:
                    angle = np.zeros((len(z), 2))
                    angle[idk] = self.kappa * reading
                    out = (angle * u[:, 2:3], angle * (z - wk)[:, None])
                    self.max_kick = max(self.max_kick, float(np.abs(angle[idk]).max()))
                    self.sum_kick2 += float(np.sum(angle[idk] ** 2))
                    self.n_kick += idk.size
            result.append(out)
        return result

    def end_step(self, step: int) -> None:
        """Close the reading of a slot at its last step."""
        if step % self.per_slot:
            return
        q = step // self.per_slot - 1
        total, count = self.acc.pop(q, [np.zeros(2), 0])
        value = total / count if count else np.zeros(2)
        if self.noise:
            value = value + self.noise * self.rng.standard_normal(2)
        self.reading[q] = value
        for old in [k for k in self.reading if k < q - self.lag // self.per_slot - 4]:
            del self.reading[old]

    def describe(self) -> dict:
        return dict(pickup_z_m=self.zp, kicker_z_m=self.zk, slots=self.slots, delay_turns=self.cfg.damper_delay_turns, lag_steps=self.lag,
                    kappa_xy_rad_per_m=self.kappa.tolist(), noise_m=self.noise, planes=self.cfg.damper_planes)

    def stats(self) -> dict:
        return dict(max_kick_rad=self.max_kick, rms_kick_rad=math.sqrt(self.sum_kick2 / self.n_kick) if self.n_kick else 0.0)


def coherent_row(beams: list, meta: list, orbit, nbins: int, cfg) -> dict:
    """Coherent amplitude and decohered emittance of the protons, from lab-position bins (see the module docstring)."""
    group = [s for s, info in zip(beams, meta) if info["role"] == "proton" and len(s.population)]
    if not group:
        return {}
    pos = np.concatenate([s.position_m for s in group])
    mom = np.concatenate([s.momentum_mc for s in group])
    w = np.concatenate([s.population for s in group])
    z = pos[:, 2] % cfg.circumference_m
    b = np.minimum((z / cfg.circumference_m * nbins).astype(int), nbins - 1)
    xy = pos[:, :2]
    ref = orbit.xy(z) if orbit is not None else 0.0
    wsum = np.bincount(b, w, minlength=nbins)
    ok = wsum > 0
    d = np.zeros((nbins, 2))
    um = np.zeros((nbins, 2))
    xm = np.zeros((nbins, 2))
    for k in range(2):
        d[:, k] = np.bincount(b, w * (xy[:, k] - (ref[:, k] if orbit is not None else 0.0)), minlength=nbins) / np.where(ok, wsum, 1)
        xm[:, k] = np.bincount(b, w * xy[:, k], minlength=nbins) / np.where(ok, wsum, 1)
        um[:, k] = np.bincount(b, w * mom[:, k], minlength=nbins) / np.where(ok, wsum, 1)
    amp2 = np.sum(wsum[:, None] * d ** 2, axis=0) / wsum.sum()
    mag = np.hypot(d[:, 0], d[:, 1])
    xc, uc = xy - xm[b], mom[:, :2] - um[b]
    xx = np.sum(w[:, None] * xc ** 2, axis=0) / w.sum()
    uu = np.sum(w[:, None] * uc ** 2, axis=0) / w.sum()
    xu = np.sum(w[:, None] * xc * uc, axis=0) / w.sum()
    counts = np.bincount(b, minlength=nbins)
    noise2 = np.array([np.sum(wsum / np.maximum(counts, 1) * np.bincount(b, (xy[:, k] - xm[b, k]) ** 2, minlength=nbins) / np.maximum(counts, 1))
                       for k in range(2)]) / wsum.sum() * 1.0
    return dict(bins=nbins, centroid_rms_xy_m=np.sqrt(amp2).tolist(), centroid_rms_m=float(math.sqrt(amp2.sum())),
                centroid_noise_xy_m=np.sqrt(noise2).tolist(),
                centroid_max_m=float(mag[ok].max()),
                incoh_rms_xy_m=np.sqrt(xx).tolist(),
                incoh_normalized_emittance_m=np.sqrt(np.maximum(0, xx * uu - xu * xu)).tolist(),
                **(dict(bin_centroids_m=np.round(d, 7).tolist(), bin_weight=np.round(wsum / wsum.sum(), 6).tolist())
                   if getattr(cfg, "diag_bin_centroids", False) else {}))
