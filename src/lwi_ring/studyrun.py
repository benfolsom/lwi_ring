"""Runner: one driver.run case with m = 0 / m = 2 column diagnostics, optional early stop and profile-shape diagnostic.

usage: lwi-ring-run CONFIG.json NAME [key=value ...] [--diag-every=N] [--stop-growth=X] [--shape] [--no-diag]  -> $LWI_RING_OUT/NAME
Takes the solver lock of the output root (BlockingIOError at once if held); an existing NAME is replaced. Diagnostics are written to
NAME/m2diag.jsonl, one row per turn (none with --no-diag):
  * every N field solves (default 4) the protons (and alive electrons) are binned by column (lattice_periods) and by position in the cell
    (8 bins of cell/8); positions are taken from the column axis (closed orbit) and the bin sums of w, w z, w z^2, w |z|^2 (z = x + i y)
    are formed per sample; per sample and bin the centred moments V = <|z - zbar|^2> and Q = <(z - zbar)^2> (complex ellipse) give,
    accumulated over the turn, the mean V, the mean |Q|/V (ellipticity, with its sampling-noise floor 1/sqrt(n) for reference), and the
    cross terms Q_e conj(Q_p);
  * at the end of each turn the protons' weighted rms, projected emittances and the 4-D rms emittance det(Sigma)^(1/4) of
    (x, Px, y, Py) with canonical momenta P = u + (e B_z / 2 m c) (-y, x) about the machine axis (invariant through the solenoids);
  * --shape: "shape", per column and species, the weighted mean over the 7 in-column bins of <r^4>/<r^2>^2
    about the bin centroid (flat-top 1.33, painted p = -0.5 1.43, Gaussian 2; hollow below 1.33);
  * --stop-growth=X: the run ends after the first turn whose 4-D canonical emittance exceeds (1 + X) times its
    initial value (records up to that turn kept; NAME/stop.json written; status "stopped").
The driver itself is unchanged by the diagnostics (they only read the particles). The wall-feedback record NAME/wallfb.jsonl is written by
the driver when wfb_enable is set.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.constants import c as C, elementary_charge as E, proton_mass as MP

from . import driver as rd


class StopRun(ValueError):
    pass


class ColumnDiagnostics:
    """Moment accumulators and per-turn rows of the study runners (see the module doc)."""

    NS = 8

    def __init__(self, cfg: rd.Config, path: Path, every: int = 4, shape: bool = False, stop_growth: float | None = None):
        self.cfg, self.path, self.every, self.shape, self.stop_growth = cfg, Path(path), every, shape, stop_growth
        p = rd.physics(cfg)
        self.lens = rd.ElectronLens(cfg, p)
        self.track = rd.Racetrack(cfg, p) if cfg.racetrack() else None
        self.NB = cfg.lattice_periods
        self.cell = cfg.circumference_m / self.NB
        self.calls = 0
        self.E4 = []
        self.reset()

    # ---- accumulators ---------------------------------------------------------------
    def reset(self):
        z = lambda dt=float: np.zeros((2, self.NB, self.NS), dt)
        self.n = 0
        self.V, self.absQ, self.noise, self.cnt = z(), z(), z(), z()
        self.Q = z(complex)
        self.QeQp = np.zeros((self.NB, self.NS), complex)
        self.c = z(complex)

    def sample(self, beams):
        cfg, NB, NS, cell = self.cfg, self.NB, self.NS, self.cell
        self.n += 1
        for k, role in enumerate(("proton", "electron")):
            group = [s for s in beams if (s.name.startswith("electron") if role == "electron" else not s.name.startswith("electron"))]
            if not group:
                continue
            pos = np.concatenate([s.position_m for s in group]); w = np.concatenate([s.population for s in group])
            if role == "electron":
                alive = np.concatenate([s.alive for s in group]); w = np.where(alive, w, 0.)
            sel = w > 0
            if not sel.any():
                continue
            pos, w = pos[sel], w[sel]
            zz = pos[:, 2] % cfg.circumference_m
            ax = self.lens.axis_at(zz)
            r = pos[:, :2] - (0. if ax is None else ax)
            zc = r[:, 0] + 1j * r[:, 1]
            j = np.minimum((zz / cell).astype(int), NB - 1)
            b = np.minimum(((zz - j * cell) / cell * NS).astype(int), NS - 1)
            idx = j * NS + b
            m = NB * NS
            W = np.bincount(idx, w, m); N = np.bincount(idx, None, m)
            S1 = np.bincount(idx, w * zc.real, m) + 1j * np.bincount(idx, w * zc.imag, m)
            S2c = w * zc**2
            S2 = np.bincount(idx, S2c.real, m) + 1j * np.bincount(idx, S2c.imag, m)
            SA = np.bincount(idx, w * np.abs(zc)**2, m)
            ok = (N >= 32) & (W > 0)
            Wd = np.where(ok, W, 1.)
            mean = S1 / Wd
            V = np.where(ok, SA / Wd - np.abs(mean)**2, 0.)
            Q = np.where(ok, S2 / Wd - mean**2, 0.)
            sh = (NB, NS)
            self.V[k] += V.reshape(sh); self.Q[k] += Q.reshape(sh); self.c[k] += np.where(ok, mean, 0).reshape(sh)
            self.absQ[k] += np.where(ok & (V > 0), np.abs(Q) / np.where(V > 0, V, 1), 0.).reshape(sh)
            self.noise[k] += np.where(ok, 1 / np.sqrt(np.maximum(N, 1)), 0.).reshape(sh)
            self.cnt[k] += ok.reshape(sh)
            if k == 0:
                self._qp = Q.reshape(sh); self._okp = ok.reshape(sh)
            else:
                both = ok.reshape(sh) & self._okp
                self.QeQp += np.where(both, Q.reshape(sh) * np.conj(self._qp), 0)

    def row(self):
        c = np.maximum(self.cnt, 1)
        d = dict(samples=self.n, count=self.cnt.tolist())
        for k, role in enumerate(("proton", "electron")):
            d[role] = dict(rms_m=np.sqrt(np.maximum(self.V[k] / c[k], 0) / 2).tolist(), ellipticity=(self.absQ[k] / c[k]).tolist(),
                           ellipticity_noise=(self.noise[k] / c[k]).tolist(),
                           Q_mean_re=(self.Q[k].real / c[k]).tolist(), Q_mean_im=(self.Q[k].imag / c[k]).tolist(),
                           centroid_re=(self.c[k].real / c[k]).tolist(), centroid_im=(self.c[k].imag / c[k]).tolist())
        cc = np.maximum(np.minimum(self.cnt[0], self.cnt[1]), 1)
        d["QeQp_re"] = (self.QeQp.real / cc).tolist(); d["QeQp_im"] = (self.QeQp.imag / cc).tolist()
        return d

    # ---- per-turn quantities --------------------------------------------------------------------------
    def emittance4(self, beams):
        cfg, track = self.cfg, self.track
        group = [s for s in beams if not s.name.startswith("electron")]
        pos = np.concatenate([s.position_m for s in group]); u = np.concatenate([s.momentum_mc for s in group])
        w = np.concatenate([s.population for s in group])
        sel = w > 0; pos, u, w = pos[sel], u[sel], w[sel]
        zz = pos[:, 2] % cfg.circumference_m
        if track is not None and cfg.solenoid_bz_t:
            bz = cfg.solenoid_bz_t * (track.overlaps(zz, np.ones_like(zz), 1.0) * track.sign).sum(axis=1)
        else:
            bz = np.zeros_like(zz)
        a = E * bz / (2 * MP * C)
        X = np.column_stack([pos[:, 0], u[:, 0] - a * pos[:, 1], pos[:, 1], u[:, 1] + a * pos[:, 0]])
        m = np.average(X, axis=0, weights=w); Xc = X - m
        S = (Xc * w[:, None]).T @ Xc / w.sum()
        ex = np.sqrt(max(np.linalg.det(S[:2, :2]), 0)); ey = np.sqrt(max(np.linalg.det(S[2:, 2:]), 0))
        # eigen-emittances of the 4-D sigma matrix (|imag eigenvalues of J S|)
        J = np.kron(np.eye(2), np.array([[0, 1], [-1, 0]]))
        ev = np.sort(np.abs(np.linalg.eigvals(J @ S).imag))[::2]
        rr = pos[:, :2] - np.average(pos[:, :2], axis=0, weights=w)
        return dict(emit4d_m=float(max(np.linalg.det(S), 0) ** 0.25), emit_canon_xy_m=[float(ex), float(ey)], eigen_emit_m=ev.tolist(),
                    rms_xy_m=np.sqrt(np.average(rr**2, axis=0, weights=w)).tolist(),
                    alive_fraction=float(w.size / sum(len(s.population) for s in group)),
                    r4_over_r2sq=float(np.average(np.sum(rr**2, 1)**2, weights=w) / np.average(np.sum(rr**2, 1), weights=w)**2))

    def shape4(self, beams):
        cfg, NB, NS, cell = self.cfg, self.NB, self.NS, self.cell
        out = {}
        for role in ("proton", "electron"):
            group = [x for x in beams if (x.name.startswith("electron") if role == "electron" else not x.name.startswith("electron"))]
            if not group:
                continue
            pos = np.concatenate([x.position_m for x in group]); w = np.concatenate([x.population for x in group])
            if role == "electron":
                w = np.where(np.concatenate([x.alive for x in group]), w, 0.)
            sel = w > 0
            pos, w = pos[sel], w[sel]
            zz = pos[:, 2] % cfg.circumference_m
            ax = self.lens.axis_at(zz)
            r = pos[:, :2] - (0. if ax is None else ax)
            zc = r[:, 0] + 1j * r[:, 1]
            j = np.minimum((zz / cell).astype(int), NB - 1)
            b = np.minimum(((zz - j * cell) / cell * NS).astype(int), NS - 1)
            idx = j * NS + b; m = NB * NS
            W = np.bincount(idx, w, m); N = np.bincount(idx, None, m)
            mean = (np.bincount(idx, w * zc.real, m) + 1j * np.bincount(idx, w * zc.imag, m)) / np.where(W > 0, W, 1)
            d2 = np.abs(zc - mean[idx])**2
            r2 = np.bincount(idx, w * d2, m) / np.where(W > 0, W, 1); r4 = np.bincount(idx, w * d2**2, m) / np.where(W > 0, W, 1)
            k = np.where((N >= 64) & (r2 > 0), r4 / np.where(r2 > 0, r2, 1)**2, np.nan).reshape(NB, NS)[:, :7]
            out[role] = np.nanmean(k, axis=1).tolist()
        return out

    # ---- driver seams ----------------------------------------------------------------------------------
    def on_fields(self, beams):
        self.calls += 1
        if self.calls % self.every == 0:
            self.sample(beams)

    def setup_hook(self, beams, meta, pp):
        row = dict(turn=0, **self.emittance4(beams))
        self.E4.append(row["emit4d_m"])
        with self.path.open("w") as f:
            f.write(json.dumps(row) + "\n")

    def turn_hook(self, step, beams, lon):
        row = dict(turn=step // self.cfg.steps_per_turn, **self.emittance4(beams), bins=self.row())
        if self.shape:
            row["shape"] = self.shape4(beams)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.reset()
        if self.stop_growth is not None and self.E4 and row["emit4d_m"] > (1 + self.stop_growth) * self.E4[0]:
            (self.path.parent / "stop.json").write_text(json.dumps(dict(turn=row["turn"], growth4d=row["emit4d_m"] / self.E4[0] - 1)))
            raise StopRun(f"stopped at turn {row['turn']}: 4-D emittance growth above {self.stop_growth}")


def run_case(cfg: rd.Config, out: Path, diagnostics: bool = True, every: int = 4, shape: bool = False,
             stop_growth: float | None = None) -> dict:
    """One driver.run case with the study diagnostics; returns the runner's status dict (complete / stopped / aborted)."""
    out = Path(out)
    if not diagnostics:
        try:
            summary = rd.run(cfg, out)
            return {k: summary.get(k) for k in ("status", "executed_steps", "elapsed_s")}
        except ValueError as err:          # e.g. the beam left the free-space grid: the record up to the abort is the result
            return dict(status="aborted", reason=str(err))
    diag = ColumnDiagnostics(cfg, out / "m2diag.jsonl", every, shape, stop_growth)
    original = rd.RingFields.fields

    def fields(self, beams, meta, lab_time, measure_core=False):
        diag.on_fields(beams)
        return original(self, beams, meta, lab_time, measure_core)

    rd.RingFields.fields = fields
    try:
        summary = rd.run(cfg, out, setup_hook=diag.setup_hook, turn_hook=diag.turn_hook)
        return {k: summary.get(k) for k in ("status", "executed_steps", "elapsed_s")}
    except ValueError as err:
        return dict(status="stopped" if isinstance(err, StopRun) else "aborted", reason=str(err))
    finally:
        rd.RingFields.fields = original


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = [a for a in argv if not a.startswith("--")]
    every, stop_growth, shape, diagnostics = 4, None, False, True
    for a in argv:
        if a.startswith("--diag-every="):
            every = int(a.split("=", 1)[1])
        elif a.startswith("--stop-growth="):
            stop_growth = float(a.split("=", 1)[1])
        elif a == "--shape":
            shape = True
        elif a == "--no-diag":
            diagnostics = False
        elif a.startswith("--out-root="):
            rd.set_output_root(a.split("=", 1)[1])
        elif a.startswith("--"):
            raise SystemExit(f"unknown option {a}")
    if len(args) < 2:
        raise SystemExit(__doc__)
    cfgp, name = Path(args[0]), args[1]
    base = json.loads(cfgp.read_text())
    for a in args[2:]:
        k, v = a.split("=", 1); base[k] = json.loads(v)
    cfg = rd.Config(**base)
    out = rd.OUT / name
    with rd.solver_lock(1):
        if out.exists():
            shutil.rmtree(out)
        print(json.dumps(run_case(cfg, out, diagnostics, every, shape, stop_growth)))


if __name__ == "__main__":
    main()
