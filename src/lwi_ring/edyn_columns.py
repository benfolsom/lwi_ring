"""Self-consistent (strong-strong) electron columns for the native-PIC ring driver.

The rigid columns of ``ElectronLens`` compensate the protons with a prescribed electron charge. Here the
electrons are macroparticles of their own, evolving in the same PIC fields as the protons, so the
electron-proton coupling (the two-stream channel of ``edyn_long_straight``) is present, only slowly.

Model, in the lab frame of the ring driver:

* One electron macroparticle is paired with each proton macroparticle (same index, same longitudinal
  position and longitudinal momentum u_z = gamma beta, since the beams co-move), so the electron line
  density follows the proton bunch exactly at birth. Its weight is c_local times the proton weight,
  with c_local the compensation inside the columns.
* An electron is *born* when its proton enters a column section: transverse position on the column axis
  (the closed orbit when ``column_follow_orbit``) plus a fixed truncated-Gaussian draw scaled to the local
  profile rms; it is *removed* (weight zero) when it leaves the section. Columns are fresh on every
  transit, as in the physical design (electrons are not stored between turns).
* Inside a column the electrons are strongly magnetised (confining field B_z = ``column_b_t``, or the
  proton solenoid field of the periodic lattice with its alternating sign): the cyclotron motion is
  averaged out and the transverse motion is the guiding-centre E x B drift with the effective
  electric field E_eff = F/q = E + v x B_beam, dx/dt = E_eff,y / B_z, dy/dt = -E_eff,x / B_z (the same
  expression as the guiding-centre mode of ``edyn_long_straight``, with the lab-frame force F, which already
  contains the 1/gamma^2 of a co-moving beam). Longitudinally p_z changes by q E_z dt (the electrons are
  free to slide against the protons).
* They are a third species ("electron") of the field solver: they source fields for the protons and
  feel the proton fields, so no rigid lens kick is applied in this mode.

* Re-centring stations (Config.column_recentre_stations = K > 0, monitor mode): K stations at
  L k/(K+1), k = 1..K, inside every column. The electrons that crossed a station during the step are moved rigidly by
  d = -eta (m + n - t), n the monitor noise, t the station offset plus, for column_recentre_target = "proton", the mean offset
  of the paired protons. m is the slice offset measured on the electrons that crossed the station in this step and the one before
  (a causal window of two step paths, 3.75 m at 256 steps per turn, against the 2 m Gaussian smoothing of ColumnDrift): the
  intercept a of the weighted complex fit D = a + c b of the drift displacements D (position minus birth position) on the birth
  offsets b from the column axis. The reference is thus the electron's own birth line: magnetised electrons stay on the straight
  field line through their birth point while the closed orbit inside the solenoid is a Larmor helix (2.4-2.7 mm rms change from
  entrance to station with the 7e-5 rad dipole errors); forcing them onto the local orbit would need static mm shifts whose side
  kicks distort the proton orbit. The static column-orbit offset this leaves is the same with and without stations. A rigid shift of the slice goes into a, the incoherent rotation of the column about its axis
  into c, so a carries neither the sampling offset of the birth draw nor that of the rotation, which would otherwise dominate at
  ~16 macro-electrons per slice (1-2 mm of noise per station, heating the protons through the side kick). The paired protons get
  the side kick x' + i y' += s i sigma side d with the coefficient of ColumnDrift (s = column_recentre_side_kick).

* m = 2 options (all inert at their defaults): dyn_birth_sigma (birth size: the local
  matched size, the gun's fixed size, or the measured size / covariance of the protons entering with the electrons), dyn_resets (fresh
  electrons several times per column), dyn_b_scale (field of the E x B drift only), dyn_gc_scheme ("ab2": second-order Adams-Bashforth in
  the drift velocity), dyn_aperture_kill (protons beyond aperture_m from the column axis are removed and counted, instead of ending the run
  at the grid edge), and at the stations column_recentre_m2 (a strain term q conj(b) in the fit, removed by a quadrupole-like shaping
  field with its own proton kick) and column_recentre_dipole (scale of the dipole correction).

* m = 2 remedy options (inert at their defaults): rem_station_pos (station positions as column fractions, which
  also allows stations together with dyn_resets > 1), rem_delay_steps (each station applies the fit of the same electrons made at a
  pickup J step paths upstream) and rem_m2_noise (noise on the fitted strain q).

* Realistic profiles (inert at their defaults): dyn_profile maps the same birth draw to a painted
  (1 - r^2/R^2)^(p + 1), flat-top (uniform disc) or hollow (uniform annulus) unit-rms profile by its radial CDF (the angle and the
  quantile u = 1 - exp(-|n|^2 / 2) of each standard-normal pair are kept), dyn_size_ratio and dyn_ellipticity apply a fixed linear map
  to the unit draw; the birth-size rules (dyn_birth_sigma) then scale it as for the Gaussian.

* Recycled electrons (Config.recycle_stride = d > 0; inert at the default 0): the electrons leaving a
  column keep their exit offset from the column axis and are re-born, through a parametrised proton-free transport map (rotation,
  centroid phase, centroid and strain cleaning, scraping, current loss; see Config), in column k + d by the same (paired) index, i.e.
  with the same proton slice. Fresh electrons only at stream starts (Recycler.fresh); the slice statistics of the map use the stored
  exits of the window recycle_window_m around the slice (by exit step), all of which have left column k before the slice re-enters.

Not included: cyclotron/Larmor radius (thermal electrons), transverse momentum, electron-electron
collisions, the electrons' own centrifugal and curvature drifts in the arcs.
"""
from __future__ import annotations

import numpy as np

from core.pic.grid import Species
from core.pic.kernels import C

E_CHARGE = 1.602176634e-19
M_ELECTRON = 9.1093837015e-31
TRUNCATION = 4.5
BIRTH_WINDOW_M = 4.0         # dyn_birth_sigma proton | proton_cov | paired*: half-width of the proton sample at an entrance


def profile_unit(normals: np.ndarray, profile: str, p: float, hollow: float) -> np.ndarray:
    """Map standard-normal pairs to a round profile with unit rms per plane (painted, flattop or hollow; see the module doc)."""
    rho2 = np.sum(normals**2, axis=1)
    u = -np.expm1(-rho2 / 2)
    if profile == "flattop":
        r2 = 4 * u
    elif profile == "painted":
        r2 = 2 * (p + 3) * -np.expm1(np.log1p(-u) / (p + 2))
    elif profile == "hollow":
        r2 = 4 / (1 + hollow**2) * (hollow**2 + u * (1 - hollow**2))
    else:
        raise ValueError(f"unknown electron profile {profile}")
    return normals * np.sqrt(r2 / np.maximum(rho2, 1e-300))[:, None]


def truncated_normals(normals: np.ndarray, cut: float = TRUNCATION) -> np.ndarray:
    """Map standard-normal pairs to a 2-D Gaussian truncated at radius `cut` (radial CDF re-scaling)."""
    rho2 = np.sum(normals**2, axis=1)
    u = -np.expm1(-rho2 / 2) * (-np.expm1(-cut**2 / 2))
    new = np.sqrt(-2 * np.log1p(-u))
    return normals * (new / np.sqrt(np.maximum(rho2, 1e-300)))[:, None]


class DynamicColumns:
    """Electron macroparticles co-moving with the proton bunches inside the column sections."""

    def __init__(self, cfg, p: dict, lens, track=None):
        if cfg.lens_profile != "gaussian":
            raise ValueError("dynamic electron columns take their profile from dyn_profile; lens_profile must stay gaussian")
        self.cfg, self.p, self.lens, self.track = cfg, p, lens, track
        total = float(lens.lengths.sum())
        self.local = cfg.column_local_fraction if cfg.column_local_fraction is not None \
            else cfg.lens_fraction * cfg.circumference_m / total
        self.normals: dict[str, np.ndarray] = {}
        self.resets = cfg.electron_resets()              # dyn_resets or its alias column_drift_resets (lwi_ring)
        self.cell = cfg.circumference_m / cfg.lattice_periods if cfg.lattice_periods else cfg.circumference_m
        self.sub = float(lens.lengths[0]) / self.resets
        self.subid: dict[str, np.ndarray] = {}
        self.vprev: dict[str, np.ndarray] = {}
        self.lost = 0.0                                  # proton weight removed by dyn_aperture_kill
        self.total = None
        self.bscale = float(cfg.rem_birth_ratio)          # gun-size factor (rem_birth_ratio, rem_birth_track)
        self.track_on = bool(cfg.rem_birth_track) or cfg.rem_birth_ratio != 1.0
        self.ncall = 0
        self.track_log: list = []
        self.rc = Recycler(cfg, lens) if cfg.recycle_stride else None
        self.field_scale = 1.0                            # ramp: confining-field scale (ramp_solenoid_follow)
        self.ramp_sigma = 1.0                             # ramp: fixed gun size relative to injection (ramp_column_sigma)
        self.m2 = float(cfg.column_recentre_m2)
        self.dip = float(cfg.column_recentre_dipole)
        self.K = int(cfg.column_recentre_stations)
        if self.K:
            if cfg.column_recentre_mode != "monitor":
                raise ValueError("dynamic electron columns support column_recentre_mode = monitor only")
            if cfg.column_sections != "straights" or not cfg.racetrack():
                raise ValueError("dynamic re-centring stations need columns on the straights of a periodic lattice")
            from scipy.constants import elementary_charge as E, proton_mass as MP
            self.cell = cfg.circumference_m / cfg.lattice_periods
            self.seg = float(lens.lengths[0]) / (self.K + 1)
            L = float(lens.lengths[0])
            self.pos = None if cfg.rem_station_pos is None else np.asarray(cfg.rem_station_pos, float) * L
            self.delay = cfg.station_delay_steps()       # rem_delay_steps or its alias column_recentre_delay_steps (lwi_ring)
            if self.delay:
                st = self.pos if self.pos is not None else L * np.arange(1, self.K + 1) / (self.K + 1)
                self.pick = st - self.delay * cfg.circumference_m / cfg.steps_per_turn
                starts = (np.floor(st / (L / self.resets)) * (L / self.resets)) if self.resets > 1 else np.zeros_like(st)
                if np.any(self.pick <= starts + 1e-9) or np.any(self.pick[1:] <= st[:-1]):
                    raise ValueError("rem_delay_steps: every pickup must lie inside the station's (sub-)column, after the previous station")
                self.pend_d: dict[str, np.ndarray] = {}
                self.pend_q: dict[str, np.ndarray] = {}
                self.pend_ok: dict[str, np.ndarray] = {}
                self.psegment: dict[str, np.ndarray] = {}
                self.pwindow: dict[str, tuple] = {}
            if self.pos is not None and self.resets > 1:
                sub = L / self.resets
                if np.any(np.abs(self.pos / sub - np.round(self.pos / sub)) < 1e-6):
                    raise ValueError("rem_station_pos: a station on a sub-column boundary")
            self.qrng = np.random.default_rng(cfg.seed * 1000003 + 104729 + cfg.column_recentre_seed) if cfg.rem_m2_noise > 0 else None
            self.dz = cfg.circumference_m / (cfg.steps_per_turn * cfg.column_drift_bins_per_step)
            self.nbins = cfg.steps_per_turn * cfg.column_drift_bins_per_step
            self.side = cfg.column_recentre_side_kick * abs(cfg.solenoid_bz_t) * E / (MP * C * p["gamma"] * p["beta"])   # as ColumnDrift
            self.pmc = p["gamma"] * p["beta"]
            self.rng = np.random.default_rng(cfg.seed * 1000003 + 7919 + cfg.column_recentre_seed)
            nst = cfg.lattice_periods * self.K
            off = self.rng.normal(0, cfg.column_recentre_offset_m, (nst, 2)) if cfg.column_recentre_offset_m > 0 else np.zeros((nst, 2))
            self.offset = off[:, 0] + 1j * off[:, 1]
            self.birth: dict[str, np.ndarray] = {}         # birth offset from the column axis
            self.birth_abs: dict[str, np.ndarray] = {}     # birth position
            self.segment: dict[str, np.ndarray] = {}
            self.stat = dict(n=0, d2=0.0, k2=0.0, electrons=0, resid2=0.0, rows=0)
            self.window: dict[str, tuple] = {}         # per electron species: the previous step's per-station fit sums

    # ---- energy ramp (ramp.Ramp.begin_step; inert without a ramp) -----------------------------------
    def set_ramp(self, local: float, field_scale: float, sigma_scale: float, gamma: float, beta: float) -> None:
        """Ramp state of the step: local compensation of new births and of live electrons (weights follow the proton population
        each step), confining-field scale, fixed gun size scale, and the station side-kick rigidity."""
        self.local, self.field_scale, self.ramp_sigma = local, field_scale, sigma_scale
        if self.K:
            from scipy.constants import elementary_charge as E, proton_mass as MP
            self.side = self.cfg.column_recentre_side_kick * abs(self.cfg.solenoid_bz_t * field_scale) * E / (MP * C * gamma * beta)
            self.pmc = gamma * beta

    # ---- field of the confining solenoid -------------------------------------------------------
    def bz_at(self, z: np.ndarray) -> np.ndarray:
        """Signed axial field (T) at lab positions z: the lattice solenoid with its polarity, else column_b_t."""
        if self.track is not None and self.cfg.solenoid_bz_t:
            fraction = self.track.overlaps(z, np.ones_like(z), 1.0)
            bz = self.cfg.solenoid_bz_t * (fraction * self.track.sign).sum(axis=1)
            return bz * self.field_scale if self.field_scale != 1.0 else bz
        return np.full(len(z), self.cfg.column_b_t * self.field_scale if self.field_scale != 1.0 else self.cfg.column_b_t)

    # ---- species ---------------------------------------------------------------------------------
    def create(self, beams: list, meta: list) -> None:
        """Append one electron species per proton bunch (all electrons start dead, then `update` births them)."""
        rng_seed = self.cfg.seed + 5003
        new_beams, new_meta = [], []
        for s, info in zip(beams, meta):
            if info["role"] != "proton":
                continue
            n = len(s.position_m)
            rng = np.random.default_rng([rng_seed, info["bunch"]])
            draw = rng.standard_normal((n, 2))
            self.normals[s.name] = truncated_normals(draw)
            cfg = self.cfg
            if cfg.dyn_profile != "gaussian":
                self.normals[s.name] = profile_unit(draw, cfg.dyn_profile, cfg.dyn_painted_p, cfg.dyn_hollow_ratio)
            if cfg.dyn_size_ratio != 1.0 or cfg.dyn_ellipticity:
                t = np.radians(cfg.dyn_ellipse_angle_deg)
                rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                m = cfg.dyn_size_ratio * rot @ np.diag([np.sqrt(1 + cfg.dyn_ellipticity), np.sqrt(1 - cfg.dyn_ellipticity)]) @ rot.T
                self.normals[s.name] = self.normals[s.name] @ m.T
            pos = s.position_m.copy()
            u = np.zeros_like(s.momentum_mc)
            u[:, 2] = s.momentum_mc[:, 2]
            e = Species(f"electron_{info['bunch']}", pos, u, -E_CHARGE, M_ELECTRON, np.full(n, 1e-30))
            e.proton_name = s.name
            e.population = np.zeros(n)
            e.alive = np.zeros(n, bool)
            if self.resets > 1:
                self.subid[e.name] = np.full(n, -1)
            if self.K:
                self.birth[e.name] = np.zeros((n, 2))
                self.birth_abs[e.name] = np.zeros((n, 2))
                self.segment[e.name] = np.full(n, -1)
                if self.delay:
                    self.pend_d[e.name] = np.zeros(n, complex)
                    self.pend_q[e.name] = np.zeros(n, complex)
                    self.pend_ok[e.name] = np.zeros(n, bool)
                    self.psegment[e.name] = np.full(n, -1)
            if self.rc:
                self.rc.create(e.name, n)
            new_beams.append(e)
            new_meta.append(dict(role="electron", bunch=info["bunch"], design_z_m=info["design_z_m"], direction=1,
                                 slice_id=info["slice_id"], matched_variance=0.,
                                 initial_centroid_offset_m=[0., 0., 0.]))
        beams.extend(new_beams)
        meta.extend(new_meta)

    def update(self, beams: list, meta: list) -> None:
        """Birth electrons whose proton is in a section; remove electrons outside every section."""
        by_name = {s.name: s for s in beams}
        if self.rc:
            self.rc.step += 1
        if self.cfg.rem_birth_track:
            if self.ncall % self.cfg.steps_per_turn == 0:
                self.track_size(beams)
            self.ncall += 1
        for e in beams:
            if not hasattr(e, "proton_name"):
                continue
            proton = by_name[e.proton_name]
            if self.cfg.dyn_aperture_kill:
                self.kill(proton, e)
            inside_electron = self.lens.inside(e.position_m[:, 2])
            gone = e.alive & ~inside_electron
            if self.resets > 1:
                gone |= e.alive & (self.sub_index(e.position_m[:, 2]) != self.subid[e.name])
            if self.rc:
                self.rc.store(e, gone)
            e.alive &= ~gone
            born = ~e.alive & self.lens.inside(proton.position_m[:, 2]) & (proton.population > 0)
            if born.any():
                z = proton.position_m[born, 2]
                axis = self.lens.axis_at(z)
                if self.cfg.dyn_birth_sigma == "follow":
                    sigma = self.lens.sigma_at(z)
                    xy = self.normals[proton.name][born] * np.reshape(sigma, (-1, 1))
                else:
                    xy = self.birth_xy(proton, born, z, axis)
                if self.rc:
                    xy = self.rc.reborn(e, proton, born, z, xy)
                e.position_m[born, :2] = xy if axis is None else xy + axis
                if self.resets > 1:
                    self.subid[e.name][born] = self.sub_index(z)
                e.position_m[born, 2] = z
                e.momentum_mc[born, :2] = 0.
                e.momentum_mc[born, 2] = proton.momentum_mc[born, 2]
                e.alive |= born
                if e.name in self.vprev:
                    self.vprev[e.name][1][born] = False
                if self.K:
                    self.birth[e.name][born] = xy
                    self.birth_abs[e.name][born] = e.position_m[born, :2]
            if self.K:
                self.recentre(e, proton, born)
            weight = np.where(e.alive, self.local * proton.population, 0.0)
            if self.rc:
                weight = weight * self.rc.fac[e.name]
            e.population = weight
            idle = ~e.alive
            if idle.any():
                z = e.position_m[idle, 2]
                axis = self.lens.axis_at(z)
                e.position_m[idle, :2] = 0. if axis is None else axis

    # ---- m = 2 helpers ------------------------------------------------------------------------
    def sub_index(self, z: np.ndarray) -> np.ndarray:
        """Global index of the (sub-)column containing lab position z (columns start at the cell origins)."""
        zz = np.asarray(z) % self.cfg.circumference_m
        j = np.floor(zz / self.cell)
        return (j * self.resets + np.clip(np.floor((zz - j * self.cell) / self.sub), 0, self.resets - 1)).astype(int)

    def birth_xy(self, proton: Species, born: np.ndarray, z: np.ndarray, axis) -> np.ndarray:
        """Birth offsets from the axis for dyn_birth_sigma = fixed | proton | proton_cov (protons grouped by the (sub-)column they enter)."""
        nrm = self.normals[proton.name][born]
        if self.cfg.dyn_birth_sigma == "fixed":
            if self.track_on:
                return nrm * (self.lens.shape["sigma"] * self.bscale)
            if self.ramp_sigma != 1.0:
                return nrm * (self.lens.shape["sigma"] * self.ramp_sigma)
            return nrm * self.lens.shape["sigma"]
        rel = proton.position_m[born, :2] - (0. if axis is None else axis)
        group = self.sub_index(z) if self.resets > 1 else np.floor((np.asarray(z) % self.cfg.circumference_m) / self.cell).astype(int)
        out = np.empty_like(nrm)
        # the protons' centroid and covariance at an entrance: all live protons of the bunch within +-BIRTH_WINDOW_M of it (a sample of
        # several thousand at 32768 per bunch; the protons entering in one step alone are too few for a covariance)
        zp = proton.position_m[:, 2] % self.cfg.circumference_m
        live = proton.population > 0
        axp = self.lens.axis_at(zp)
        relp = proton.position_m[:, :2] - (0. if axp is None else axp)
        for g in np.unique(group):
            sel = group == g
            if self.resets > 1:
                ent = (g // self.resets) * self.cell + (g % self.resets) * self.sub
            else:
                ent = g * self.cell
            dz = (zp - ent + 0.5 * self.cfg.circumference_m) % self.cfg.circumference_m - 0.5 * self.cfg.circumference_m
            win = live & (np.abs(dz) < BIRTH_WINDOW_M)
            if win.sum() < 64:
                out[sel] = nrm[sel] * np.reshape(self.lens.sigma_at(z[sel]), (-1, 1))
                continue
            ws = proton.population[win]
            mean = np.average(relp[win], axis=0, weights=ws)
            r = relp[win] - mean
            cov = (r * ws[:, None]).T @ r / ws.sum()
            if self.cfg.dyn_birth_sigma == "paired":
                out[sel] = rel[sel] - mean
            elif self.cfg.dyn_birth_sigma == "paired_fixed":
                out[sel] = (rel[sel] - mean) * (self.lens.shape["sigma"] / np.sqrt(0.5 * np.trace(cov)))
            elif self.cfg.dyn_birth_sigma == "proton":
                out[sel] = nrm[sel] * np.sqrt(0.5 * np.trace(cov))
            else:
                out[sel] = nrm[sel] @ np.linalg.cholesky(cov).T
        return out

    def track_size(self, beams: list) -> None:
        """rem_birth_track: gun size = rem_birth_ratio x the rms (per plane) of all live protons inside the columns, about the axis."""
        sw = sxx = 0.0
        for s in beams:
            if hasattr(s, "proton_name") or s.name.startswith("electron"):
                continue
            z = s.position_m[:, 2]
            live = (s.population > 0) & self.lens.inside(z)
            if not live.any():
                continue
            axis = self.lens.axis_at(z[live])
            r = s.position_m[live, :2] - (0. if axis is None else axis)
            w = s.population[live]
            m = np.average(r, axis=0, weights=w)
            sw += w.sum(); sxx += float(np.sum(w[:, None] * (r - m) ** 2))
        if sw > 0:
            rms = np.sqrt(sxx / sw / 2)
            self.bscale = self.cfg.rem_birth_ratio * rms / self.lens.shape["sigma"]
            self.track_log.append(float(self.bscale))

    def kill(self, proton: Species, e: Species) -> None:
        """dyn_aperture_kill: remove protons beyond aperture_m from the column axis (and their electrons)."""
        if self.total is None:
            self.total = {}
        self.total.setdefault(proton.name, float(proton.population.sum()))
        z = proton.position_m[:, 2]
        axis = self.lens.axis_at(z)
        rel = proton.position_m[:, :2] - (0. if axis is None else axis)
        out = (proton.population > 0) & (np.hypot(rel[:, 0], rel[:, 1]) > self.cfg.aperture_m)
        if out.any():
            self.lost += float(proton.population[out].sum())
            proton.population[out] = 0.
            proton.position_m[out, :2] = 0. if axis is None else axis[out]
            proton.momentum_mc[out, :2] = 0.
            e.alive[out] = False
            e.population[out] = 0.

    # ---- re-centring stations ----------------------------------------------------------------------
    def recentre(self, e: Species, proton: Species, born: np.ndarray) -> None:
        """Move the electron groups that crossed a station since the last call; side-kick their paired protons (see module doc)."""
        z = e.position_m[:, 2] % self.cfg.circumference_m
        if self.pos is None:
            seg = np.where(e.alive, np.clip(np.floor((z % self.cell) / self.seg), 0, self.K).astype(int), -1)
        else:
            seg = np.where(e.alive, np.searchsorted(self.pos, z % self.cell, side="right"), -1)
        prev = self.segment[e.name]
        hit = np.flatnonzero(e.alive & ~born & (prev >= 0) & (seg > prev))
        self.segment[e.name] = seg
        if self.delay:
            # pickups J step paths upstream: fit the electrons crossing them now, keep the correction with each electron
            pseg = np.where(e.alive, np.searchsorted(self.pick, z % self.cell, side="right"), -1)
            pprev = self.psegment[e.name]
            phit = np.flatnonzero(e.alive & ~born & (pprev >= 0) & (pseg > pprev))
            self.psegment[e.name] = pseg
            self.pend_ok[e.name][born] = False
            if len(phit):
                on, dst, dq, station, stats = self.fit(e, proton, phit, pseg, z, self.pwindow)
                self.pend_d[e.name][phit] = dst[station]
                self.pend_q[e.name][phit] = dq[station] if dq is not None else 0
                self.pend_ok[e.name][phit] = True
                self.account(on, dst, dq, stats)
            else:
                self.pwindow[e.name] = None
            if not len(hit):
                return
            ok = self.pend_ok[e.name][hit]
            dh = np.where(ok, self.pend_d[e.name][hit], 0)
            dqh = np.where(ok, self.pend_q[e.name][hit], 0) if self.m2 else None
            self.pend_ok[e.name][hit] = False
            self.apply(e, proton, hit, z[hit], dh, dqh)
            return
        if not len(hit):
            self.window[e.name] = None
            return
        on, dst, dq, station, stats = self.fit(e, proton, hit, seg, z, self.window)
        self.apply(e, proton, hit, z[hit], dst[station], dq[station] if self.m2 else None)
        self.account(on, dst, dq, stats)

    def fit(self, e: Species, proton: Species, hit: np.ndarray, seg: np.ndarray, z: np.ndarray, windows: dict):
        """Per-station fit of the electrons `hit` (crossing station or pickup seg - 1 now) and the previous step's crossers: the dipole
        correction dst and the m = 2 strain correction dq per station (see module doc)."""
        zh = z[hit]
        b = self.birth[e.name][hit]
        rel = e.position_m[hit, :2] - self.birth_abs[e.name][hit]
        D = rel[:, 0] + 1j * rel[:, 1]
        bc = b[:, 0] + 1j * b[:, 1]
        w = e.population[hit]
        nst = self.cfg.lattice_periods * self.K
        station = (np.floor(zh / self.cell).astype(int) % self.cfg.lattice_periods) * self.K + seg[hit] - 1
        cur = (np.bincount(station, w, nst), self.cbin(station, w * D, nst), self.cbin(station, w * bc, nst),
               np.bincount(station, w * np.abs(bc)**2, nst), self.cbin(station, w * D * np.conj(bc), nst),
               np.bincount(station, w * np.abs(D)**2, nst), np.bincount(station, None, nst).astype(float))
        if self.m2:
            pr = proton.position_m[hit, :2] - (self.birth_abs[e.name][hit] - b)        # paired protons from the electrons' birth axis
            pc = pr[:, 0] + 1j * pr[:, 1]
            cur = cur + (self.cbin(station, w * bc**2, nst), self.cbin(station, w * D * bc, nst),
                         self.cbin(station, w * pc, nst), self.cbin(station, w * pc**2, nst))
        prev = windows.get(e.name)
        windows[e.name] = cur
        tot = cur if prev is None else tuple(c + q for c, q in zip(cur, prev))
        W, SD, SB, SBB, SDB, SDD, NN = tot[:7]
        on = np.flatnonzero(cur[0] > 0)
        Wo = W[on]
        Dm, Bm = SD[on] / Wo, SB[on] / Wo
        vb = SBB[on] / Wo - np.abs(Bm)**2
        c = np.where(vb > 0, (SDB[on] / Wo - Dm * np.conj(Bm)) / np.where(vb > 0, vb, 1), 0)
        m = Dm - c * Bm
        resid = np.maximum(SDD[on] / Wo - np.abs(Dm)**2 - np.abs(c)**2 * vb, 0)
        dq = None
        if self.m2:
            # weighted complex least squares D = a + c b + q conj(b) (normal equations per station)
            SB2, SDb, SP, SP2 = (t[on] for t in tot[7:])
            G = np.empty((len(on), 3, 3), complex)
            G[:, 0] = np.stack([Wo, SB[on], np.conj(SB[on])], 1)
            G[:, 1] = np.stack([np.conj(SB[on]), SBB[on], np.conj(SB2)], 1)
            G[:, 2] = np.stack([SB[on], SB2, SBB[on]], 1)
            h = np.stack([SD[on], SDB[on], SDb], 1)
            good = (np.abs(np.linalg.det(G)) > 1e-6 * np.abs(Wo) ** 3 * np.maximum(vb, 1e-30) ** 2) & (NN[on] >= 64)
            theta = np.zeros((len(on), 3), complex)
            if good.any():
                theta[good] = np.linalg.solve(G[good], h[good][..., None])[..., 0]
            m = np.where(good, theta[:, 0], m)
            q = theta[:, 2]
            if self.qrng is not None:
                sq = self.cfg.rem_m2_noise
                q = q + self.qrng.normal(0, sq, len(on)) + 1j * self.qrng.normal(0, sq, len(on))
            qt = 0.
            if self.cfg.column_recentre_m2_target == "proton":
                Pm = SP / Wo
                Qp = SP2 / Wo - Pm**2                       # proton slice ellipse <(p - pbar)^2>
                Qb = SB2 / Wo - Bm**2                       # electron birth ellipse
                qt = np.where(vb > 0, (Qp - Qb) / (2 * np.where(vb > 0, vb, 1)), 0)
            dq = np.zeros(nst, complex)
            dq[on] = np.where(good, -self.m2 * (q - qt), 0)
            dq = np.where(np.isfinite(dq), dq, 0)
            dq = np.where(np.abs(dq) > 0.2, 0.2 * dq / np.maximum(np.abs(dq), 1e-300), dq)
        sm = self.cfg.column_recentre_noise_m
        if sm > 0:
            m = m + self.rng.normal(0, sm, len(on)) + 1j * self.rng.normal(0, sm, len(on))
        target = self.offset[on]
        if self.cfg.column_recentre_target == "proton":
            prel = proton.position_m[hit, :2] - (self.birth_abs[e.name][hit] - b)      # from the electrons' birth axis
            target = target + self.cbin(station, w * (prel[:, 0] + 1j * prel[:, 1]), nst)[on] / cur[0][on]
        dst = np.zeros(nst, complex)
        dst[on] = -self.cfg.column_recentre_eta * (m - target)
        if self.dip != 1.0:
            dst = dst * self.dip
        return on, dst, dq, station, (len(hit), resid, NN)

    def apply(self, e: Species, proton: Species, hit: np.ndarray, zh: np.ndarray, dh: np.ndarray, dqh) -> None:
        """Shift (and, with column_recentre_m2, reshape) the electrons `hit`; kick their paired protons with the same fields."""
        if self.m2:
            # shaping field about the electrons' birth axis: electron displacement dq conj(r), proton kick from the same field
            b = self.birth[e.name][hit]
            axis = self.birth_abs[e.name][hit] - b
            re = e.position_m[hit, :2] - axis
            rp = proton.position_m[hit, :2] - axis
            de = dqh * np.conj(re[:, 0] + 1j * re[:, 1])
            dp_ = dqh * np.conj(rp[:, 0] + 1j * rp[:, 1])
        e.position_m[hit, 0] += dh.real
        e.position_m[hit, 1] += dh.imag
        sigma = np.sign(self.bz_at(zh))
        kick = 1j * sigma * self.side * dh * self.pmc
        if self.m2:
            e.position_m[hit, 0] += de.real
            e.position_m[hit, 1] += de.imag
            kick = kick + self.cfg.column_recentre_m2_kick * 1j * sigma * self.side * dp_ * self.pmc
        proton.momentum_mc[hit, 0] += kick.real
        proton.momentum_mc[hit, 1] += kick.imag

    def account(self, on, dst, dq, stats) -> None:
        nhit, resid, NN = stats
        if self.m2:
            self.stat["q2"] = self.stat.get("q2", 0.) + float(np.sum(np.abs(dq[on])**2))
        self.stat["n"] += len(on); self.stat["electrons"] += nhit
        self.stat["d2"] += float(np.sum(np.abs(dst[on])**2)); self.stat["k2"] += float(np.sum(np.abs(self.side * dst[on])**2))
        self.stat["resid2"] += float(np.sum(resid / NN[on]))          # squared statistical error of the intercept (equal weights)
        self.stat["rows"] += len(on)

    @staticmethod
    def cbin(index, values, n):
        return np.bincount(index, values.real, n) + 1j * np.bincount(index, values.imag, n)

    def describe_recentre(self) -> dict:
        n = max(self.stat["n"], 1)
        return dict(stations=self.K, eta=self.cfg.column_recentre_eta, target=self.cfg.column_recentre_target,
                    noise_m=self.cfg.column_recentre_noise_m, side_kick=self.cfg.column_recentre_side_kick,
                    station_d_rms_m=float(np.sqrt(self.stat["d2"] / n)), side_kick_rms_rad=float(np.sqrt(self.stat["k2"] / n)),
                    station_hits=self.stat["n"], station_electrons=self.stat["electrons"],
                    fit_noise_rms_m=float(np.sqrt(self.stat["resid2"] / max(self.stat["rows"], 1))),
                    **(dict(m2=self.m2, m2_target=self.cfg.column_recentre_m2_target, dipole=self.dip,
                            station_q_rms=float(np.sqrt(self.stat.get("q2", 0.) / n))) if (self.m2 or self.dip != 1.0) else {}))

    # ---- push ------------------------------------------------------------------------------------
    def push(self, e: Species, electric: np.ndarray, magnetic: np.ndarray, dt: float) -> np.ndarray:
        """Guiding-centre step: E x B drift of the position, q E_z dt on the longitudinal momentum.

        Returns the new momentum array (transverse momentum stays zero, as for a guiding centre).
        """
        v = e.velocity_m_s
        force_over_q = electric + np.cross(v, magnetic)
        bz = self.bz_at(e.position_m[:, 2])
        if self.cfg.dyn_b_scale != 1.0:
            bz = bz * self.cfg.dyn_b_scale
        live = e.alive & (bz != 0)          # no confining field, no guiding centre: such electrons stay put
        safe = np.where(bz != 0, bz, 1.0)
        drift = np.zeros((len(v), 2))
        drift[:, 0] = force_over_q[:, 1] / safe
        drift[:, 1] = -force_over_q[:, 0] / safe
        if self.cfg.dyn_gc_scheme == "ab2":
            prev = self.vprev.get(e.name)
            new = drift.copy()
            if prev is not None:
                ok = live & prev[1]
                drift[ok] = 1.5 * drift[ok] - 0.5 * prev[0][ok]
            self.vprev[e.name] = (new, live.copy())
        e.position_m[:, :2] += np.where(live[:, None], drift * dt, 0.)
        u = e.momentum_mc.copy()
        u[:, 2] += np.where(e.alive, e.charge_c * force_over_q[:, 2] * dt / (e.mass_kg * C), 0.)
        return u

    # ---- diagnostics -----------------------------------------------------------------------------
    def diagnose(self, beams: list) -> dict:
        """Electron population, size, centroid and compensation ratio over the alive electrons."""
        by_name = {s.name: s for s in beams}
        rows = {}
        for e in beams:
            if not hasattr(e, "proton_name"):
                continue
            proton = by_name[e.proton_name]
            live = e.alive
            row = dict(alive_fraction=float(live.mean()), electron_population=float(e.population.sum()),
                       proton_population=float(proton.population.sum()))
            if live.any() and e.population[live].sum() > 0:      # (a ramp's taper can set every live electron's weight to 0)
                w = e.population[live]
                xy = e.position_m[live, :2]
                mean = np.average(xy, axis=0, weights=w)
                row.update(centroid_xy_m=mean.tolist(),
                           rms_xy_m=np.sqrt(np.average((xy - mean)**2, axis=0, weights=w)).tolist())
            rows[e.name] = row
        if self.K:
            rows["recentre"] = self.describe_recentre()
        if self.track_on:
            rows["birth_scale"] = dict(scale=float(self.bscale), updates=len(self.track_log))
        if self.rc:
            rows["recycle"] = self.rc.describe()
        if self.cfg.dyn_aperture_kill:
            rows["aperture_kill"] = dict(lost_population=self.lost, lost_fraction=self.lost / max(sum((self.total or {}).values()), 1e-300))
        return rows


class Recycler:
    """Recycled-electron streams (Config.recycle_*; see the module doc). Offsets are from the column axis (the closed orbit when
    column_follow_orbit), complex x + i y internally."""

    def __init__(self, cfg, lens):
        self.cfg, self.lens = cfg, lens
        self.d = int(cfg.recycle_stride)
        self.resets = cfg.electron_resets() if hasattr(cfg, "electron_resets") else 1
        self.ncol = int(cfg.lattice_periods) * self.resets          # units: (sub-)columns, one electron beam each
        self.cell = cfg.circumference_m / cfg.lattice_periods
        self.length = float(lens.lengths[0]) / self.resets
        self.win = max(0, int(round(0.5 * cfg.recycle_window_m / (cfg.circumference_m / cfg.steps_per_turn))))
        self.rot = np.exp(1j * np.radians(cfg.recycle_rotate_deg))
        self.cph = np.exp(1j * np.radians(cfg.recycle_centroid_phase_deg))
        self.rng = np.random.default_rng(cfg.seed * 1000003 + 15485863 + cfg.recycle_seed) if cfg.recycle_noise_m > 0 else None
        self.step = 0
        self.off, self.src, self.when, self.sgen, self.gen, self.fac = {}, {}, {}, {}, {}, {}
        self.stat = dict(fresh=0, recycled=0, scraped_weight=0.0, recycled_weight=0.0, m_in2=0.0, m_out2=0.0, q_in2=0.0, q_out2=0.0,
                         groups=0)

    def fresh(self, c: np.ndarray) -> np.ndarray:
        """Stream starts (gun): the first d columns of the turn, and every recycle_gun_every-th column of a stream."""
        c = np.asarray(c)
        out = c < self.d
        if self.cfg.recycle_gun_every:
            out |= (c // self.d) % self.cfg.recycle_gun_every == 0
        return out

    def create(self, name: str, n: int) -> None:
        # one macro-electron per proton carries d interleaved streams: while one stream's electron of this slice is in a transport,
        # the next column holds another stream's. Stored exits are kept per stream slot (target unit mod d).
        d = self.d
        self.off[name] = np.zeros((d, n), complex)
        self.src[name] = np.full((d, n), -1)
        self.when[name] = np.zeros((d, n), int)
        self.sgen[name] = np.zeros((d, n), int)
        self.gen[name] = np.zeros(n, int)
        self.fac[name] = np.ones(n)

    def column(self, z: np.ndarray) -> np.ndarray:
        """(Sub-)column index of lab position z (inside a column, or in the gap after it)."""
        zz = np.asarray(z) % self.cfg.circumference_m
        j = np.floor(zz / self.cell)
        return ((j * self.resets + np.clip(np.floor((zz - j * self.cell) / self.length), 0, self.resets - 1)).astype(int)) % self.ncol

    def end(self, k: np.ndarray) -> np.ndarray:
        return (k // self.resets) * self.cell + (k % self.resets + 1) * self.length * (1 - 1e-12)

    def store(self, e, gone: np.ndarray) -> None:
        """Electrons leaving unit k: keep their offset at the unit end, in slot k mod d, if unit k + d is served by the same stream."""
        idx = np.flatnonzero(gone)
        if not len(idx):
            return
        # the unit just left: with dyn_resets > 1 an electron is found gone only after it entered the next sub-column, so take its
        # position one step back (without resets the exit is seen in the gap after the column, which column() already maps back)
        k = self.column(e.position_m[idx, 2] - (self.cfg.circumference_m / self.cfg.steps_per_turn if self.resets > 1 else 0.0))
        nxt = k + self.d
        keep = (nxt < self.ncol) & ~self.fresh(nxt)
        slot = k % self.d
        self.src[e.name][slot, idx] = np.where(keep, k, -1)
        if not keep.any():
            return
        idx, k, slot = idx[keep], k[keep], slot[keep]
        axis = self.lens.axis_at(self.end(k))
        xy = e.position_m[idx, :2] - (0. if axis is None else axis)
        self.off[e.name][slot, idx] = xy[:, 0] + 1j * xy[:, 1]
        self.when[e.name][slot, idx] = self.step
        self.sgen[e.name][slot, idx] = self.gen[e.name][idx]

    def reborn(self, e, proton, born: np.ndarray, z: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Birth offsets for the electrons `born` (entering at z, fresh offsets xy): recycled ones get the transport map of their stored
        exit offsets; generation count and charge factor updated."""
        name = e.name
        bidx = np.flatnonzero(born)
        c = self.column(z)
        slot = c % self.d
        rec = (c >= self.d) & ~self.fresh(c) & (self.src[name][slot, bidx] == c - self.d)
        fresh_i = bidx[~rec]
        self.gen[name][fresh_i] = 0
        self.fac[name][fresh_i] = 1.0
        self.stat["fresh"] += len(fresh_i)
        if not rec.any():
            return xy
        xy = xy.copy()
        w_all = proton.population
        for col in np.unique(c[rec]):
            sl = col % self.d
            src, off, when = self.src[name][sl], self.off[name][sl], self.when[name][sl]
            sel = np.flatnonzero(rec & (c == col))
            ii = bidx[sel]
            t0 = int(np.median(when[ii]))
            pool = np.flatnonzero((src == col - self.d) & (np.abs(when - t0) <= self.win) & (w_all > 0))
            loss = (1 - self.cfg.recycle_current_loss) ** (self.sgen[name][sl, pool] + 1)
            w = w_all[pool] * loss
            if w.sum() <= 0:                            # nothing measurable: these electrons are re-born fresh
                self.gen[name][ii] = 0
                self.fac[name][ii] = 1.0
                continue
            o = off[pool]
            m = np.sum(w * o) / w.sum()
            r = o - m
            sxx = np.sum(w * r.real**2) / w.sum(); syy = np.sum(w * r.imag**2) / w.sum(); sxy = np.sum(w * r.real * r.imag) / w.sum()
            S = np.array([[sxx, sxy], [sxy, syy]])
            ro = off[ii] - m
            if self.cfg.recycle_strain_eta:
                St = (1 - self.cfg.recycle_strain_eta) * S + self.cfg.recycle_strain_eta * 0.5 * np.trace(S) * np.eye(2)
                M = _sqrtm(St) @ np.linalg.inv(_sqrtm(S))
                v = M @ np.vstack([ro.real, ro.imag])
                ro = v[0] + 1j * v[1]
                q_out = St[0, 0] - St[1, 1] + 2j * St[0, 1]
            else:
                q_out = S[0, 0] - S[1, 1] + 2j * S[0, 1]
            ro = ro * self.rot
            mm = self.cph * m
            if self.cfg.recycle_centroid_eta:
                noise = 0j
                if self.rng is not None:
                    noise = complex(self.rng.normal(0, self.cfg.recycle_noise_m), self.rng.normal(0, self.cfg.recycle_noise_m))
                mm = mm - self.cfg.recycle_centroid_eta * (mm + noise)
            new = mm + ro
            g = self.sgen[name][sl, ii] + 1
            self.gen[name][ii] = g
            fac = (1 - self.cfg.recycle_current_loss) ** g
            if self.cfg.recycle_scrape_sigma:
                lim = self.cfg.recycle_scrape_sigma * np.sqrt(0.5 * np.trace(S))
                cut = np.abs(ro) > lim
                self.stat["scraped_weight"] += float(np.sum(w_all[ii][cut] * fac[cut]))
                fac = np.where(cut, 0.0, fac)
            self.fac[name][ii] = fac
            xy[sel, 0], xy[sel, 1] = new.real, new.imag
            tr = max(np.trace(S), 1e-300)
            self.stat["recycled"] += len(ii); self.stat["recycled_weight"] += float(np.sum(w_all[ii] * fac)); self.stat["groups"] += 1
            self.stat["m_in2"] += abs(m) ** 2; self.stat["m_out2"] += abs(mm) ** 2
            self.stat["q_in2"] += abs((S[0, 0] - S[1, 1] + 2j * S[0, 1]) / tr) ** 2; self.stat["q_out2"] += abs(q_out / tr) ** 2
        return xy

    def describe(self) -> dict:
        g = max(self.stat["groups"], 1)
        return dict(stride=self.d, fresh_births=self.stat["fresh"], recycled_births=self.stat["recycled"],
                    scraped_weight=self.stat["scraped_weight"], recycled_weight=self.stat["recycled_weight"],
                    centroid_in_rms_m=float(np.sqrt(self.stat["m_in2"] / g)), centroid_out_rms_m=float(np.sqrt(self.stat["m_out2"] / g)),
                    strain_in_rms=float(np.sqrt(self.stat["q_in2"] / g)), strain_out_rms=float(np.sqrt(self.stat["q_out2"] / g)),
                    groups=self.stat["groups"])


def _sqrtm(S: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return V @ np.diag(np.sqrt(np.maximum(w, 1e-300))) @ V.T
