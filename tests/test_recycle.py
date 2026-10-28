"""Recycled electrons (Config.recycle_*, edyn_columns.Recycler): inert defaults, option checks, the transport map on a known slice, and a
short ring run with two leapfrogging streams."""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cases import DYN0, FAST, J1

CELL, L = 40.0, 35.0


def test_defaults_inert():
    from lwi_ring.driver import RECYCLE_DEFAULTS, Config
    ours = {f.name: f.default for f in dataclasses.fields(Config)}
    assert ours["recycle_stride"] == 0
    for k, v in RECYCLE_DEFAULTS.items():
        assert ours[k] == v, k
    Config(**dict(J1, **DYN0)).validate()


@pytest.mark.parametrize("bad", [
    dict(recycle_rotate_deg=90.0),                       # option without recycle_stride
    dict(recycle_stride=2, column_electrons="rigid"),
    dict(recycle_stride=12),                             # stride must be < lattice_periods
    dict(recycle_stride=24, dyn_resets=2),               # stride must be < sub-columns per turn
    dict(recycle_stride=2, recycle_strain_eta=1.5),
    dict(recycle_stride=2, recycle_current_loss=1.0),
])
def test_conflicts_rejected(bad):
    from lwi_ring.driver import Config
    with pytest.raises(ValueError):
        Config(**dict(J1, **DYN0, **bad)).validate()


def recycler(**kw):
    from lwi_ring.edyn_columns import Recycler
    opts = dict(recycle_stride=2, recycle_gun_every=0, recycle_rotate_deg=0.0, recycle_centroid_phase_deg=0.0, recycle_centroid_eta=0.0,
                recycle_strain_eta=0.0, recycle_current_loss=0.0, recycle_scrape_sigma=0.0, recycle_window_m=4.0, recycle_noise_m=0.0,
                recycle_seed=0)
    opts.update({k: v for k, v in kw.items() if k != "resets"})
    cfg = SimpleNamespace(electron_resets=lambda: kw.get("resets", 1), lattice_periods=12, circumference_m=12 * CELL, steps_per_turn=512, seed=1, **opts)
    lens = SimpleNamespace(lengths=np.full(12, L), axis_at=lambda z: None)
    return Recycler(cfg, lens)


def slice_pass(rc, n=20000, centroid=(1e-3, -0.5e-3), q=0.1, angle=0.3, seed=3):
    """an elliptical, displaced electron slice leaves column 0; returns (stored offsets, re-born offsets in column 2, Recycler)."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, 2)) * 0.03
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    g = g @ (R @ np.diag([np.sqrt(1 + q), np.sqrt(1 - q)]) @ R.T).T + np.array(centroid)
    e = SimpleNamespace(name="electron_0", position_m=np.column_stack([g, np.full(n, L + 0.3)]))
    rc.create(e.name, n)
    rc.step = 7
    rc.store(e, np.ones(n, bool))
    rc.step = 50
    proton = SimpleNamespace(population=np.ones(n))
    z = np.full(n, 2 * CELL + 0.2)
    out = rc.reborn(e, proton, np.ones(n, bool), z, np.zeros((n, 2)))
    return g, out, rc


def moments(xy):
    m = xy.mean(axis=0)
    r = xy - m
    Q = np.mean((r[:, 0] + 1j * r[:, 1]) ** 2)
    return m, Q, np.mean(np.sum(r**2, axis=1))


def test_identity_map():
    g, out, rc = slice_pass(recycler())
    assert np.allclose(out, g, atol=1e-15)
    assert rc.stat["recycled"] == len(g) and rc.stat["fresh"] == 0
    assert np.all(rc.gen["electron_0"] == 1) and np.all(rc.fac["electron_0"] == 1.0)


def test_rotation_reverses_strain():
    g, out, _ = slice_pass(recycler(recycle_rotate_deg=90.0))
    m0, Q0, r0 = moments(g)
    m1, Q1, r1 = moments(out)
    assert np.allclose(m1, m0, atol=1e-12)                 # the ellipse turns about the slice centre
    assert abs(Q1 + Q0) < 1e-12 * abs(Q0) + 1e-18          # 90 degrees: opposite strain
    assert abs(r1 / r0 - 1) < 1e-12


def test_cleaning():
    g, out, _ = slice_pass(recycler(recycle_centroid_eta=1.0, recycle_strain_eta=1.0))
    m1, Q1, r1 = moments(out)
    m0, Q0, r0 = moments(g)
    assert np.abs(m1).max() < 1e-15 and abs(Q1) < 1e-10 * r0
    assert abs(r1 / r0 - 1) < 1e-10                        # the rms (trace) is kept
    g, out, _ = slice_pass(recycler(recycle_centroid_eta=0.5, recycle_centroid_phase_deg=90.0))
    m1 = moments(out)[0]
    m0 = moments(g)[0]
    assert np.allclose(m1[0] + 1j * m1[1], 0.5j * (m0[0] + 1j * m0[1]), atol=1e-15)


def test_loss_scrape_and_streams():
    g, out, rc = slice_pass(recycler(recycle_current_loss=0.01, recycle_scrape_sigma=3.0))
    fac = rc.fac["electron_0"]
    kept = fac > 0
    assert np.allclose(fac[kept], 0.99)
    r = np.hypot(*(g - g.mean(axis=0)).T)
    assert 0 < (~kept).sum() < 0.03 * len(g) and r[~kept].min() > r[kept].max() * 0.9
    rc = recycler(recycle_gun_every=3)
    assert list(np.flatnonzero(rc.fresh(np.arange(12)))) == [0, 1, 6, 7]
    assert list(np.flatnonzero(recycler().fresh(np.arange(12)))) == [0, 1]
    rc = recycler(resets=4)                                 # sub-columns of 8.75 m: units 0..47
    assert rc.ncol == 48 and list(rc.column(np.array([0.1, 8.8, 34.9, 37.0, 40.1, 479.0]))) == [0, 1, 3, 3, 4, 47]
    assert abs(rc.end(np.array([5]))[0] - (40 + 2 * 8.75)) < 1e-9


def test_ring_two_streams(tmp_path):
    """1 turn at the FAST size, leapfrog with the identity map: columns 0 and 1 fresh, all other births recycled."""
    from lwi_ring import driver
    cfg = driver.Config(**dict(J1, **DYN0, **FAST, recycle_stride=2))
    res = driver.run(cfg, tmp_path / "rc")
    rows = json.loads((tmp_path / "rc" / "turns.json").read_text())
    rec = [r["electrons"]["recycle"] for r in rows if "electrons" in r and "recycle" in r["electrons"]][-1]
    assert rec["recycled_births"] > 0 and rec["fresh_births"] > 0
    assert rec["recycled_births"] > rec["fresh_births"]          # start-up turn: slices that began mid-ring are born fresh
    assert res is not None


def test_interleaved_streams():
    """One macro-electron per proton serves both streams: exit 0 -> (fresh in 1) -> exit 1 -> re-born in 2 with the exit-0 state, in 3
    with the exit-1 state."""
    rc = recycler()
    n = 1000
    rng = np.random.default_rng(5)
    e = SimpleNamespace(name="electron_0", position_m=np.zeros((n, 3)))
    rc.create(e.name, n)
    proton = SimpleNamespace(population=np.ones(n))
    allv = np.ones(n, bool)
    states = {}
    for k in range(4):
        z_in = np.full(n, k * CELL + 0.1)
        xy = rc.reborn(e, proton, allv, z_in, rng.standard_normal((n, 2)) * 0.03)
        if k >= 2:
            assert np.allclose(xy, states[k - 2], atol=1e-15), k
        else:
            assert rc.stat["recycled"] == 0
        out = xy + rng.standard_normal((n, 2)) * 1e-3                      # the column changes the state
        states[k] = out
        e.position_m[:, :2] = out
        e.position_m[:, 2] = k * CELL + L + 0.2
        rc.step += 40
        rc.store(e, allv)
    assert rc.stat["recycled"] == 2 * n and rc.stat["fresh"] == 2 * n
