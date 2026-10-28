"""Segmented wall-electrode feedback on the electron columns (csr-wallring, 2026-10-26; report_wall_feedback_ring.md).

The ring-driver counterpart of the WarpX study of report_wall_electrode_feedback.md (scripts/csrwall_column.py, csrwall_el.py): every
column (the straights of the periodic lattice) is lined with a bore of radius b (wfb_bore_m) cut into n azimuthal sectors (wfb_sectors,
sector k centred at 2 pi k / n, contiguous) and into nseg = round(L / wfb_segment_m) equal segments along the column. Each segment is
driven from a pickup wfb_pickup_offset = j segments upstream (for the first j segments: the last j segment lengths of the arc before the
column, which see the protons only).

Pickup. The net (proton + electron) charge is binned in a frame co-moving at the design speed (bins of wfb_bin_m along the ring, so the
two bunches and any debunched beam share the slices, as on a real pickup). At every step the particles inside each pickup region give,
per slice, the moments M_l = sum q z^l (z = x + i y about the chamber axis, l <= max(n + 2, 18)), the induced charge on the n sectors from the exact
circular-bore Green function, and from those the l = 1 and l = 2 estimates (DFT over the sectors with the sinc correction, aliasing of the
n -+ l harmonics included; Bore of csrwall_el.py). A slice's pickup signal is the mean of these per-step estimates over its transit of
the region (the "signal averaged over the previous L of beam travel" of the WarpX study, now slice-resolved). Optional pickup noise
(wfb_noise1_m: rms dipole noise in metres of offset at the bunch-peak line density; wfb_noise2: rms l = 2 noise in units of ellipticity
|Q|/(2 sigma^2) at the peak line density; one draw per slice and pass, quoted after the slice filter), and a slow baseline: the mean
signal of each slice at each pickup over the earlier passes (running mean, then an exponential average over wfb_baseline_turns) is
subtracted, as static closed-orbit and lattice-ellipticity signals must be (0 = no subtraction). The loop drives from turn wfb_start_turn on.

Processing. The slice signals are smoothed with a Gaussian (symmetric, linear-phase) filter of rms sigma = hypot(wfb_smooth_m,
beta c 0.1325 / f), f = wfb_bw_mhz (0: no electronics limit). wfb_smooth_m (-1: 0.707 lab cells of the field grid, the rms of the PIC's
own deposit-and-gather smoothing along the bunch) keeps the loop from driving slice structure that the PIC's electron columns cannot
respond to: there the hold law's proton kick would be pure anti-damping (a first version with 0.25 m slices lost the beam within a turn
of closing the loop). A symmetric filter needs latency: each segment takes the most downstream pickup region that has measured every
slice within 3 sigma behind the slice it reads, moved wfb_pickup_offset - 1 regions further upstream; segments for which even the arc
regions are too close are not driven (the coverage is reported). wfb_delay_ns is the timing error of the drive relative to the beam
(positive: late, a slice gets the signal of the slice beta c tau ahead of it). The electronics latency must fit into the
cable-versus-beam budget of the pickup-to-segment distance; latency beyond it is a positive wfb_delay_ns.

Drive (hold law). A segment carries one voltage pattern at a time: at each instant it is the pattern computed for the slice at the
segment centre, so a particle at x in the segment sees the signal of the slice (x - x_c) behind it (wfb_sweep; off: its own slice). The
uniform field E1 = g1 e^{i phi1} kappa1 D1 / (4 pi eps0 (sp^2 + se^2) gamma^2), D1 the net dipole moment per metre, cancels at g1 = 1 the
mean field of a displaced Gaussian proton beam (rms sp) on a Gaussian electron column (rms se, the gun's size) as seen by co-moving
electrons (1/gamma^2); the quadrupole field E = A2 conj(z), A2 = g2 e^{i phi2} kappa2 M2 / (16 pi eps0 (sp^2 + se^2)^2 gamma^2), M2 the net
l = 2 moment per metre, cancels at g2 = 1 the protons' linear quadrupole field on the electrons (Gaussian source seen by a Gaussian
observer). kappa1, kappa2 rescale for other profiles; the phases are the orientation error of the drive pattern. The pattern is turned
into sector voltages (fundamentals, Bore.voltages; optional amplifier limit wfb_vmax_v on the peak sector voltage |V1| + |V2|), and their
vacuum field (all harmonics up to max(n + 2, 18) of the piecewise-constant wall voltages) is added to the lab electric field of every proton
(direct kick; off with wfb_proton_kick = false) and electron (E x B drift; off with wfb_electron_drive = false) inside a segment.
Segment-end fringe fields and E_z are neglected (2-D electrodes).

Per-turn records go to DIRECTORY/wallfb.jsonl: electrode voltages (rms, 99th percentile and maximum over segments and steps with beam,
lab volts per sector), signal levels, the side kick, and the fraction of the drive limited by wfb_vmax_v.
"""
from __future__ import annotations

import json
import math

import numpy as np

from core.pic.kernels import C

EPS0 = 8.8541878128e-12
QE = 1.602176634e-19


class Bore:
    """Segmented circular bore (conventions of csrwall_el.Bore), with the harmonics truncated at lmax."""

    def __init__(self, nsec: int, bore: float, lmax: int):
        self.n, self.b, self.lmax = nsec, bore, lmax
        self.dth = 2 * math.pi / nsec
        self.th = self.dth * np.arange(nsec)
        th1, th2 = self.th - self.dth / 2, self.th + self.dth / 2
        self.L = np.arange(1, lmax + 1)
        self.I = (np.exp(-1j * np.outer(self.L, th2)) - np.exp(-1j * np.outer(self.L, th1))) / (-1j * self.L[:, None])
        self.J = np.conj(self.I) / math.pi

    def sinc(self, l: int) -> float:
        return math.sin(l * self.dth / 2) / (l * self.dth / 2)

    def sector_charges(self, q0: np.ndarray, M: np.ndarray) -> np.ndarray:
        """q0 (S,) total charge, M (S, lmax) moments sum q z^l -> (S, n) induced charge on the sectors."""
        return -(1 / (2 * math.pi)) * (q0[:, None] * self.dth + 2 * np.real((M / self.b ** self.L) @ self.I))

    def estimate(self, qk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """l = 1, 2 moment estimates (S,) from the sector charges (S, n)."""
        out = []
        for l in (1, 2):
            P = (2.0 / self.n) * (qk @ np.exp(1j * l * self.th))
            out.append(-P * math.pi / (self.dth * self.sinc(l)) * self.b**l)
        return out[0], out[1]

    def voltages(self, E1: complex = 0j, c2: complex = 0j) -> np.ndarray:
        """sector voltages of the uniform field E1 (E_x + i E_y) and the quadrupole potential Re[c2 conj(z)^2] (fundamentals)."""
        V = -self.b * np.real(np.conj(E1) * np.exp(1j * self.th)) / self.sinc(1)
        if self.n >= 4:
            V = V + self.b**2 * np.real(c2 * np.exp(-2j * self.th)) / self.sinc(2)
        return V

    def field_coefficients(self, V: np.ndarray) -> np.ndarray:
        """F[m-1] such that E_x + i E_y = sum_m F[m-1] (conj(z)/b)^(m-1) for the sector voltages V."""
        Phi = self.J @ V
        return -self.L * Phi / self.b


class WallFeedback:
    def __init__(self, cfg, p: dict, lens, directory=None):
        self.cfg = cfg
        self.Cr = cfg.circumference_m
        self.P = cfg.lattice_periods
        self.cell = self.Cr / self.P
        self.Lcol = float(lens.lengths[0])
        self.nseg = max(1, int(round(self.Lcol / cfg.wfb_segment_m)))
        self.Ls = self.Lcol / self.nseg
        self.j = int(cfg.wfb_pickup_offset)
        if self.Cr / cfg.steps_per_turn > self.Ls + 1e-9:
            raise ValueError("wfb: the step path C / steps_per_turn must not exceed the segment length (every slice must be sampled in "
                             "every pickup region)")
        self.ja = max(1, int(math.floor((self.cell - self.Lcol) / self.Ls + 1e-9)))     # pickup regions in the arc before a column
        self.nreg = self.nseg + self.ja
        self.bin = float(cfg.wfb_bin_m)
        self.nb = int(math.ceil(self.Cr / self.bin))
        self.bin = self.Cr / self.nb
        self.vb = p["beta"] * C
        self.gamma = p["gamma"]
        sp, se = cfg.sigma_m, lens.shape["sigma"]
        self.sp2 = sp * sp + se * se
        g2 = self.gamma**2
        self.k1 = cfg.wfb_gain1 * np.exp(1j * math.radians(cfg.wfb_phase1_deg)) * cfg.wfb_kappa1 / (4 * math.pi * EPS0 * self.sp2 * g2)
        self.k2 = cfg.wfb_gain2 * np.exp(1j * math.radians(cfg.wfb_phase2_deg)) * cfg.wfb_kappa2 / (16 * math.pi * EPS0 * self.sp2**2 * g2)
        self.lmax = max(cfg.wfb_sectors + 2, 18)                # field and pickup harmonics: < 1% error to 120 mm for 8 sectors
        self.bore = Bore(cfg.wfb_sectors, cfg.wfb_bore_m, self.lmax)
        # field basis: E1 = 1, E1 = i, c2 = 1, c2 = i
        basis = [self.bore.voltages(1.0 + 0j, 0j), self.bore.voltages(1j, 0j), self.bore.voltages(0j, 1.0 + 0j), self.bore.voltages(0j, 1j)]
        self.F = np.stack([self.bore.field_coefficients(v) for v in basis], axis=1)          # (lmax, 4)
        self.s1, self.s2 = self.bore.sinc(1), self.bore.sinc(2) if cfg.wfb_sectors >= 4 else 1.0
        # slice filter: Gaussian, the PIC's longitudinal resolution (deposit + gather of the quadratic shape, rms 0.707 cell) in quadrature
        # with the electronics bandwidth (Gaussian response, rms time 0.1325 / f_3dB)
        self.sig_pic = cfg.wfb_smooth_m if cfg.wfb_smooth_m >= 0 else 0.7071 * cfg.circumference_m / cfg.grid[2]
        self.sig_bw = self.vb * 0.1325 / (cfg.wfb_bw_mhz * 1e6) if cfg.wfb_bw_mhz > 0 else 0.0
        self.sig = math.hypot(self.sig_pic, self.sig_bw)
        k = np.fft.fftfreq(self.nb, d=self.bin)
        self.kern = np.exp(-0.5 * (2 * math.pi * k * self.sig) ** 2) if self.sig > 0 else None
        self.shift_m = self.vb * cfg.wfb_delay_ns * 1e-9
        # pickup of each driven segment: the most downstream region that has measured every slice the filter needs (3 sigma behind the
        # slice read at the segment centre, or at its entrance without the sweep), moved wfb_pickup_offset - 1 further upstream;
        # segments without such a region upstream of the column (inside the arc) are not driven.
        xr = (np.arange(self.nseg) + (0.5 if cfg.wfb_sweep else 0.0)) * self.Ls + self.shift_m
        rmax = np.floor((xr - 3 * self.sig - self.bin) / self.Ls + 1e-9).astype(np.int64) - 1
        self.rk = np.minimum(rmax, np.arange(self.nseg) - 1) - (self.j - 1)
        self.driven = self.rk >= -self.ja
        self.rk = np.where(self.driven, self.rk, -self.ja)
        if not self.driven.any():
            raise ValueError("wfb: no segment has a causal pickup (filter width, delay and pickup offset against the arc length)")
        self.start_step = int(cfg.wfb_start_turn) * cfg.steps_per_turn
        # tables over (cell of the column, pickup region index r + ja, co-moving slice)
        shape = (self.P, self.nreg, self.nb)
        self.T1, self.T2 = np.zeros(shape, complex), np.zeros(shape, complex)
        self.B1, self.B2 = np.zeros(shape, complex), np.zeros(shape, complex)
        self.N1, self.N2 = np.zeros(shape, complex), np.zeros(shape, complex)
        self.cnt = np.zeros(shape)
        self.last = np.full(shape, -10**9, dtype=np.int64)
        self.npass = np.zeros(shape, dtype=np.int64)
        self.used = np.unique(self.rk[self.driven] + self.ja)
        self.zc = (np.arange(self.nb) + 0.5) * self.bin
        self.rng = np.random.default_rng(cfg.seed * 1000003 + 3571 + cfg.wfb_seed)
        self.lam_pk = None                                 # bunch-peak proton line charge (C/m), from the first step
        self.directory = directory
        self.reset_stats()

    # ---- helpers -----------------------------------------------------------------------------------
    def reset_stats(self):
        self.st = dict(v1=[], v2=[], vp=[], s1=[], s2=[], limited=0, applied=0, kick2=0.0, nkick=0)

    def frame(self, t: float, shift: float, meta: list) -> float:
        """lab position of slice 0's lower edge: half a bunch spacing behind bunch 0's design centre, moving at the design speed."""
        z0 = next(info["design_z_m"] for info in meta if info["role"] == "proton")
        return (z0 + self.vb * t + shift - self.Cr / (2 * self.cfg.harmonic)) % self.Cr

    def slice_of(self, z: np.ndarray, zref: float) -> np.ndarray:
        return (((z - zref) % self.Cr) / self.bin).astype(np.int64).clip(0, self.nb - 1)

    def regions(self, z: np.ndarray):
        """(cell of the column, region index r + ja, valid) for pickup positions z: the column's segments are regions 0..nseg-1, the
        last ja segment lengths of the arc before it regions -ja..-1 (protons only)."""
        c = np.floor(z / self.cell).astype(np.int64) % self.P
        x = z - np.floor(z / self.cell) * self.cell
        incol = x < self.Lcol
        k = np.minimum(np.floor(x / self.Ls).astype(np.int64), self.nseg - 1)
        r = np.where(incol, k, -np.ceil((self.cell - x) / self.Ls).astype(np.int64))
        cd = np.where(incol, c, (c + 1) % self.P)
        ok = r >= -self.ja
        return cd, np.clip(r + self.ja, 0, self.nreg - 1), ok

    # ---- one step -------------------------------------------------------------------------------------
    def apply(self, beams: list, meta: list, electric: list, t: float, step: int, shift: float = 0.0) -> None:
        cfg = self.cfg
        zref = self.frame(t, shift, meta)
        # slices: the pickup region each slice centre is in now; open a new pass where the previous visit is over
        zc = (zref + self.zc) % self.Cr
        cd, drv, ok = self.regions(zc)
        sl = np.flatnonzero(ok)
        cd, drv = cd[sl], drv[sl]
        idx = (cd, drv, sl)
        new = self.last[idx] < step - 2
        if new.any():
            ii = (cd[new], drv[new], sl[new])
            done = self.cnt[ii] > 0
            if cfg.wfb_baseline_turns > 0:
                w = np.where(done, 1.0 / np.maximum(np.minimum(self.npass[ii] + 1, cfg.wfb_baseline_turns), 1), 0.0)
                v1 = np.where(done, self.T1[ii] / np.maximum(self.cnt[ii], 1), 0)
                v2 = np.where(done, self.T2[ii] / np.maximum(self.cnt[ii], 1), 0)
                self.B1[ii] += w * (v1 - self.B1[ii])
                self.B2[ii] += w * (v2 - self.B2[ii])
            self.npass[ii] += done
            self.T1[ii] = 0; self.T2[ii] = 0; self.cnt[ii] = 0
            if cfg.wfb_noise1_m > 0 or cfg.wfb_noise2 > 0:
                self.draw_noise(ii, int(new.sum()))
        self.cnt[idx] += 1
        self.last[idx] = step
        # pickup: net moments per slice of the particles now in a pickup region
        valid = np.zeros(self.nb, bool); valid[sl] = True
        q0 = np.zeros(self.nb); M = np.zeros((self.nb, self.lmax), complex)
        parts = []
        for s, info in zip(beams, meta):
            if info["role"] == "proton":
                w = s.population * QE
            elif info["role"] == "electron":
                w = np.where(s.alive, -s.population * QE, 0.)
            else:
                continue
            live = np.flatnonzero(w != 0)
            zz = s.position_m[live, 2] % self.Cr
            b = self.slice_of(zz, zref)
            xy = s.position_m[live, 0] + 1j * s.position_m[live, 1]
            parts.append((s, info, live, zz, b, xy))
            m = valid[b]
            bb, qq, zl = b[m], w[live][m], xy[m]
            q0 += np.bincount(bb, qq, self.nb)
            zp = np.ones_like(zl)
            for l in range(self.lmax):
                zp = zp * zl
                v = qq * zp
                M[:, l] += np.bincount(bb, v.real, self.nb) + 1j * np.bincount(bb, v.imag, self.nb)
        if self.lam_pk is None:
            pr = [(w_, b_) for (s, info, live, zz, b_, xy) in parts if info["role"] == "proton" for w_ in [s.population[live] * QE]]
            h = sum(np.bincount(b_, w_, self.nb) for w_, b_ in pr)
            k = max(1, int(round(2.0 / self.bin)))
            self.lam_pk = float(np.max(np.convolve(h, np.ones(k) / k, mode="same")) / self.bin)
        qk = self.bore.sector_charges(q0[sl], M[sl])
        e1, e2 = self.bore.estimate(qk)
        self.T1[idx] += e1; self.T2[idx] += e2
        if step < self.start_step:
            return
        # drive: the filtered slice signals of each segment's pickup (per metre), read at each particle's effective slice
        cntd = np.maximum(self.cnt, 1)
        S1 = np.where(self.cnt > 0, self.T1 / cntd, 0) - self.B1 + self.N1
        S2 = np.where(self.cnt > 0, self.T2 / cntd, 0) - self.B2 + self.N2
        S1, S2 = S1[:, self.used], S2[:, self.used]
        if self.kern is not None:
            S1 = np.fft.ifft(np.fft.fft(S1, axis=-1) * self.kern, axis=-1)
            S2 = np.fft.ifft(np.fft.fft(S2, axis=-1) * self.kern, axis=-1)
        S1 /= self.bin; S2 /= self.bin
        row = np.full(self.nreg, -1); row[self.used] = np.arange(len(self.used))
        self.rowk = row[self.rk + self.ja]                  # row of S1/S2 read by each segment
        self.electrode_stats(S1, S2, zref)
        for (s, info, live, zz, b, xy), E in zip(parts, electric_list(parts, beams, electric)):
            c = np.floor(zz / self.cell).astype(np.int64) % self.P
            x = zz - np.floor(zz / self.cell) * self.cell
            inc = x < self.Lcol
            if not inc.any():
                continue
            if info["role"] == "proton" and not cfg.wfb_proton_kick:
                continue
            if info["role"] == "electron" and not cfg.wfb_electron_drive:
                continue
            kk = np.minimum(np.floor(x / self.Ls).astype(np.int64), self.nseg - 1)
            sel = np.flatnonzero(inc & self.driven[kk])
            if not len(sel):
                continue
            c, x, zs, xys = c[sel], x[sel], zz[sel], xy[sel]
            k = kk[sel]
            zeff = zs - ((x - (k + 0.5) * self.Ls) if cfg.wfb_sweep else 0.0) + self.shift_m
            u = ((zeff - zref) % self.Cr) / self.bin - 0.5
            i0 = np.floor(u).astype(np.int64)
            fr = u - i0
            i0 %= self.nb; i1 = (i0 + 1) % self.nb
            rw = self.rowk[k]
            d1 = (1 - fr) * S1[c, rw, i0] + fr * S1[c, rw, i1]
            d2 = (1 - fr) * S2[c, rw, i0] + fr * S2[c, rw, i1]
            E1, A2 = self.k1 * d1, self.k2 * d2
            c2 = -A2 / 2
            if cfg.wfb_vmax_v > 0:
                vpk = self.bore.b * np.abs(E1) / self.s1 + self.bore.b**2 * np.abs(c2) / self.s2
                lim = np.minimum(1.0, cfg.wfb_vmax_v / np.maximum(vpk, 1e-300))
                E1, c2 = E1 * lim, c2 * lim
            r = np.abs(xys)
            rmax = 0.95 * self.bore.b
            zb = np.conj(np.where(r > rmax, xys * rmax / np.maximum(r, 1e-300), xys)) / self.bore.b
            pw = np.ones((len(zb), self.lmax), complex)
            for m in range(1, self.lmax):
                pw[:, m] = pw[:, m - 1] * zb
            G = pw @ self.F
            Ef = E1.real * G[:, 0] + E1.imag * G[:, 1] + c2.real * G[:, 2] + c2.imag * G[:, 3]
            rows = live[sel]
            E[rows, 0] += Ef.real
            E[rows, 1] += Ef.imag
            if info["role"] == "proton":
                self.st["kick2"] += float(np.sum(np.abs(Ef)**2)); self.st["nkick"] += len(Ef)

    def draw_noise(self, ii, n: int) -> None:
        cfg = self.cfg
        if self.lam_pk is None:                            # first step: the line density is not measured yet; no noise on its slices
            return
        f = math.sqrt(2 * math.sqrt(math.pi) * self.sig / self.bin) if self.sig > 0 else 1.0     # rms quoted after the slice filter
        lam = self.lam_pk * self.bin                                                     # peak charge per slice
        s1 = cfg.wfb_noise1_m * lam * f / math.sqrt(2)
        s2 = cfg.wfb_noise2 * 2 * self.cfg.sigma_m**2 * lam * f / math.sqrt(2)
        self.N1[ii] = s1 * (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)) if s1 > 0 else 0
        self.N2[ii] = s2 * (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)) if s2 > 0 else 0

    def electrode_stats(self, S1, S2, zref) -> None:
        """voltage of every electrode now: the pattern of the slice at its centre (sector peak |V1|, |V2| and |V1| + |V2|)."""
        cells = np.arange(self.P)[:, None]
        segs = np.flatnonzero(self.driven)[None, :]
        zmid = cells * self.cell + (segs + 0.5) * self.Ls + self.shift_m
        u = (((zmid - zref) % self.Cr) / self.bin).astype(np.int64) % self.nb
        rw = self.rowk[segs]
        d1 = S1[cells, rw, u]; d2 = S2[cells, rw, u]
        E1 = self.k1 * d1; c2 = -self.k2 * d2 / 2
        v1 = self.bore.b * np.abs(E1) / self.s1
        v2 = self.bore.b**2 * np.abs(c2) / self.s2
        vp = v1 + v2
        occ = vp > 0
        if self.cfg.wfb_vmax_v > 0:
            self.st["limited"] += int(np.sum(vp > self.cfg.wfb_vmax_v))
            vp = np.minimum(vp, self.cfg.wfb_vmax_v)
        self.st["applied"] += int(occ.sum())
        self.st["v1"].append(v1[occ]); self.st["v2"].append(v2[occ]); self.st["vp"].append(vp[occ])
        lam = max(self.lam_pk, 1e-300)
        self.st["s1"].append(np.abs(d1[occ]) / lam)
        self.st["s2"].append(np.abs(d2[occ]) / (lam * 2 * self.cfg.sigma_m**2))

    def end_turn(self, turn: int) -> dict:
        def q(a):
            if not a:
                return dict(rms=0.0, p99=0.0, max=0.0)
            v = np.concatenate(a)
            if not len(v):
                return dict(rms=0.0, p99=0.0, max=0.0)
            return dict(rms=float(np.sqrt(np.mean(v**2))), p99=float(np.percentile(v, 99)), max=float(v.max()))
        st = self.st
        row = dict(turn=turn, v1_V=q(st["v1"]), v2_V=q(st["v2"]), vpeak_V=q(st["vp"]), s1_m=q(st["s1"]), s2_ellipticity=q(st["s2"]),
                   limited_fraction=st["limited"] / max(st["applied"], 1),
                   proton_field_rms_V_per_m=math.sqrt(st["kick2"] / max(st["nkick"], 1)), lam_peak_C_per_m=self.lam_pk)
        if turn == 1 or not getattr(self, "_described", False):
            row["config"] = self.describe(); self._described = True
        if self.directory is not None:
            with (self.directory / "wallfb.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
        self.reset_stats()
        return row

    def describe(self) -> dict:
        cfg = self.cfg
        return dict(segments=self.nseg, segment_m=self.Ls, sectors=cfg.wfb_sectors, bore_m=cfg.wfb_bore_m, pickup_offset=self.j,
                    driven_from_m=float(np.flatnonzero(self.driven)[0] * self.Ls), pickup_region=self.rk.tolist(),
                    smooth_m=dict(pic=self.sig_pic, bandwidth=self.sig_bw, total=self.sig),
                    gain=[cfg.wfb_gain1, cfg.wfb_gain2], phase_deg=[cfg.wfb_phase1_deg, cfg.wfb_phase2_deg], bin_m=self.bin,
                    bw_mhz=cfg.wfb_bw_mhz, drive_shift_m=self.shift_m, baseline_turns=cfg.wfb_baseline_turns,
                    noise=[cfg.wfb_noise1_m, cfg.wfb_noise2], vmax_v=cfg.wfb_vmax_v)


def electric_list(parts, beams, electric):
    """the electric-field array of each part's species (parts hold the species objects)."""
    pos = {id(s): i for i, s in enumerate(beams)}
    return [electric[pos[id(p[0])]] for p in parts]
