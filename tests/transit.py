"""Single-transit harness of the analytic electron-column checks (a compact port of scripts/csrdp_val.py, modes m1 and m2).

Protons are frozen transversely (profile `proton_profile`, displaced by `offset_mm` in x for m1, elliptical with `aspect` for m2), the
electrons are born round on the column axis and drift (E x B) in the driver's own fields while bunch 0 crosses column 0 (B_z = +0.135 T).
Only electrons born at the entrance within `window_m` of the moving bunch peak are analysed.
  m1: the electron centroid circles the proton centroid at Omega = lambda nbar / (2 eps0 |B| beta gamma^2 c) per lab metre, nbar the
      overlap density int rho_p rho_e dA (report_warpx_electron_column.md, report_dynamic_profiles.md);
  m2: fit of dQ_e/ds = i k (a Q_p + b Q_e), k = omega_D / (beta gamma c), omega_D the l = 2 rate of the rms-equivalent uniform column;
      analytic for uniform columns in free space: a = 1, b = f - 2 (report_warpx_m2_response.md).
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.constants import c as C, epsilon_0 as EPS0

from cases import J1


def transit(mode: str, proton_profile: str = "flattop", electron_profile: str = "flattop", particles: int = 2**16,
            grid=(32, 32, 64), steps: int = 20, offset_mm: float = 3.0, aspect: float = 1.1, window_m: float = 4.0) -> dict:
    from lwi_ring import driver as rd, edyn_columns as ed
    base = {k: v for k, v in J1.items() if k not in ("diag_bins", "diag_bin_centroids")}
    base.update(error_dk_rms=0.0, error_dipole_rms_rad=0.0, start_on_closed_orbit=False, column_follow_orbit=False, proton_delta_rms=0.0,
                injection_profile="painted", injection_edge_m=None, column_recentre_stations=0, dyn_birth_sigma="fixed", turns=1,
                proton_xi_x=0.0, proton_xi_y=0.0, particles_per_bunch=particles, grid=list(grid), dyn_profile=electron_profile)
    cfg = rd.Config(**base)
    cfg.validate()
    beams, meta, p = rd.initial_beams(cfg)
    gam, bet, sig = p["gamma"], p["beta"], cfg.sigma_m
    rng = np.random.default_rng(4242)
    for s, info in zip(beams, meta):
        if info["role"] != "proton":
            continue
        g = rng.standard_normal((len(s.position_m), 2))
        unit = ed.truncated_normals(g) if proton_profile == "gaussian" else ed.profile_unit(g, proton_profile, -0.5, 0.5)
        xy = unit * sig
        if mode == "m2":
            xy = xy * [math.sqrt(aspect), 1 / math.sqrt(aspect)]
        if mode == "m1":
            xy = xy + [offset_mm * 1e-3, 0.0]
        s.position_m[:, :2] = xy
        s.momentum_mc[:, :2] = 0.0
        s.momentum_mc[:, 2] = gam * bet
    lens = rd.ElectronLens(cfg, p)
    dyn = ed.DynamicColumns(cfg, p, lens, rd.Racetrack(cfg, p))
    dyn.create(beams, meta)
    solver = rd.RingFields(cfg)
    dt = p["period_s"] / cfg.steps_per_turn
    P = [s for s, i in zip(beams, meta) if i["role"] == "proton"]
    EL = [s for s, i in zip(beams, meta) if i["role"] == "electron"]
    p0, e0 = P[0], EL[0]
    n = len(e0.position_m)
    bxy, bz = np.zeros((n, 2)), np.full(n, np.nan)
    L = float(lens.lengths[0])
    B0 = abs(cfg.solenoid_bz_t * cfg.dyn_b_scale)
    zc = lambda a: a[:, 0] + 1j * a[:, 1]

    def update():
        before = e0.alive.copy()
        dyn.update(beams, meta)
        born = e0.alive & ~before
        bxy[born] = e0.position_m[born, :2]
        bz[born] = e0.position_m[born, 2]

    rows = []
    update()
    for step in range(1, steps + 1):
        t_mid = (step - 0.5) * dt
        for s in beams:
            s.position_m[:, 2] = (s.position_m[:, 2] + 0.5 * dt * s.velocity_m_s[:, 2]) % cfg.circumference_m
        update()
        E, Bf, _ = solver.fields(beams, meta, t_mid)
        zpk = (bet * C * t_mid) % cfg.circumference_m
        sel = e0.alive & (bz < bet * C * dt * 1.01) & (np.abs(e0.position_m[:, 2] - zpk) < window_m)
        psel = (np.abs(p0.position_m[:, 2] - zpk) < window_m) & (p0.position_m[:, 2] < L)
        lamp = float(np.sum(p0.population[psel]) * 1.602176634e-19 / (2 * window_m))
        if sel.sum() > 100 and zpk < L:
            w = e0.population[sel]
            xe, xb = zc(e0.position_m[sel, :2]), zc(bxy[sel])
            xp, wp = zc(p0.position_m[psel, :2]), p0.population[psel]
            u, e, eb = np.average(xp, weights=wp), np.average(xe, weights=w), np.average(xb, weights=w)
            ds = float(np.average(e0.position_m[sel, 2] - bz[sel], weights=w))
            rows.append(dict(ds=ds, u=u, e=e, eb=eb, lamp=lamp, Qp=np.average((xp - u)**2, weights=wp),
                             dQ=np.average((xe - e)**2, weights=w) - np.average((xb - eb)**2, weights=w)))
        for s, Ef, Bb in zip(beams, E, Bf):
            if s in EL:
                s.momentum_mc = dyn.push(s, Ef, Bb, dt)
        for s in beams:
            s.position_m[:, 2] = (s.position_m[:, 2] + 0.5 * dt * s.velocity_m_s[:, 2]) % cfg.circumference_m
    lam = float(np.mean([r["lamp"] for r in rows]))
    s_ = np.array([r["ds"] for r in rows])
    if mode == "m1":
        u, e, eb = (np.array([r[k] for r in rows]) for k in ("u", "e", "eb"))
        slope = float(np.polyfit(s_, np.unwrap(np.angle((e - u) / (eb - u))), 1)[0])
        rr = np.linspace(0, 6 * sig, 6001)

        def dens(shape):
            return np.gradient(rd.enclosed_fraction(shape, rr**2), rr) / (2 * math.pi * np.maximum(rr, 1e-12))

        pshape = dict(kind="gaussian", sigma=sig) if proton_profile == "gaussian" else (
            dict(kind="painted", sigma=sig, p=-0.5) if proton_profile == "painted" else dict(kind="hollow", r_in=0., r_out=2 * sig))
        nbar = float(np.trapezoid(dens(pshape) * dens(rd.dyn_shape(cfg)) * 2 * math.pi * rr, rr))
        omega = lam * nbar / (2 * EPS0 * B0 * bet * gam**2 * C)
        return dict(omega_measured_per_m=abs(slope), omega_analytic_per_m=omega, ratio=abs(slope) / omega, rows=len(rows))
    from scipy.optimize import least_squares
    Qp, dQ = np.array([r["Qp"] for r in rows]), np.array([r["dQ"] for r in rows])
    wD = lam / gam / (2 * EPS0 * math.pi * (2 * sig)**2 * B0)
    k = wD / (bet * gam * C)
    y = dQ / Qp

    def model(th):
        a, b = th[0] + 1j * th[1], th[2] + 1j * th[3]
        return (a / b) * (np.exp(1j * k * b * s_) - 1)

    fit = least_squares(lambda th: np.concatenate([(model(th) - y).real, (model(th) - y).imag]), [0.7, 0, -0.6, 0])
    f = cfg.column_local_fraction
    return dict(a=complex(fit.x[0], fit.x[1]), b=complex(fit.x[2], fit.x[3]), analytic_a=1.0, analytic_b=f - 2, rows=len(rows))


if __name__ == "__main__":
    import sys
    print(json.dumps(transit(sys.argv[1], **{k: json.loads(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}), default=str))
