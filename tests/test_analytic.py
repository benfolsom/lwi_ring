"""Analytic checks of the electron-column and lattice physics, run on lwi_ring.

* solenoid area preservation: the one-turn map of the driver's own push through the 12-cell solenoid
  lattice has det = 1 with solenoid_edge_in_drift; the legacy edge treatment does not (1 - det ~ steps^-2), so the check has teeth;
* zero-current identity: with one real proton the self-consistent (dynamic) and rigid columns give the same
  protons (the electrons carry no charge to speak of);
* E x B drift rate (transit harness, mode m1): electrons born on the axis circle a displaced flat-top proton column at
  Omega = lambda nbar / (2 eps0 |B| beta gamma^2 c);
* l = 2 response of a flat-top column (transit harness, mode m2): dQ_e/ds = i k (a Q_p + b Q_e) with a = 1 and
  b = f - 2 for a uniform column in free space; the PIC's edge smoothing makes a few per cent at 64^2
  (a = 0.911, b = -0.944 at the reference resolution).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from cases import FAST, J1

LATTICE = dict(energy_mev=70.0, circumference_m=480.0, harmonic=2, duty=0.3, sigma_m=0.03, tune=4.38, tune_x=4.38, tune_y=4.38,
               straight_length_m=35.0, arc_length_m=5.0, lattice_periods=12, straight_focus_ratio=0.0, solenoid_bz_t=0.135,
               solenoid_polarity="alternating", solenoid_retune=True, opposing="absent", collective=False, particles_per_bunch=32,
               steps_per_turn=256, tune_samples_per_turn=16, turns=1, half_xy_m=0.2, backend="cpu")


def one_turn_map(tmp_path, **options) -> np.ndarray:
    from lwi_ring import driver
    cfg = driver.Config(**LATTICE, **options)
    z0 = 3.0                                   # inside the first straight, away from its edges
    scales = {}

    def hook(beams, meta, p):
        u0 = p["gamma"] * p["beta"]
        for s in beams:
            s.position_m[:] = [0.0, 0.0, z0]
            s.momentum_mc[:] = [0.0, 0.0, u0]
        # particles 1-4 of bunch 0: x, u_x, y, u_y displaced (1 um, slope 1e-7)
        scales["d"] = np.array([1e-6, 1e-7 * u0, 1e-6, 1e-7 * u0])
        b = beams[0]
        b.position_m[1, 0] += scales["d"][0]
        b.momentum_mc[2, 0] += scales["d"][1]
        b.position_m[3, 1] += scales["d"][2]
        b.momentum_mc[4, 1] += scales["d"][3]

    driver.run(cfg, tmp_path / "map", setup_hook=hook)
    with np.load(tmp_path / "map/final_particles.npz") as z:
        x, u = z["proton_0_position_m"], z["proton_0_momentum_mc"]
    state = np.column_stack((x[:5, 0], u[:5, 0], x[:5, 1], u[:5, 1]))
    return ((state[1:] - state[0]) / scales["d"][:, None]).T


def test_solenoid_area_preservation(tmp_path):
    det_new = np.linalg.det(one_turn_map(tmp_path / "new", solenoid_edge_in_drift=True))
    det_old = np.linalg.det(one_turn_map(tmp_path / "old"))
    assert abs(det_new - 1) < 1e-6, det_new
    assert det_old < 0.995, det_old          # the legacy edge kick shrinks phase space (README: 0.985 at 512 steps, P = 6, 0.147 T)


def test_zero_current_identity(tmp_path):
    from lwi_ring import driver
    base = {k: v for k, v in J1.items() if k not in ("diag_bins", "diag_bin_centroids")}
    base.update(FAST, population=1.0, column_recentre_stations=0)
    finals = {}
    for kind in ("dynamic", "rigid"):
        driver.run(driver.Config(**dict(base, column_electrons=kind)), tmp_path / kind)
        with np.load(tmp_path / kind / "final_particles.npz") as z:
            finals[kind] = {k: z[k] for k in z.files if k.startswith("proton")}
    for key, value in finals["rigid"].items():
        tol = 1e-9 if key.endswith("position_m") else 1e-9 * np.abs(value).max()
        assert np.max(np.abs(finals["dynamic"][key] - value)) < tol, key


def test_exb_drift_rate():
    from transit import transit
    # needs the production grid: the ratio is 1.03 at 64^2 x 128, 1.15 at 64^2 x 64 and 1.25 at 32^2 x 128 (edge and entrance smoothing)
    r = transit("m1", "flattop", "flattop", particles=2**17, grid=(64, 64, 128))
    assert abs(r["ratio"] - 1) < 0.05, r


@pytest.mark.slow
def test_l2_response_flattop():
    # 2^19 particles (transit harness, mode m2): at 2^18 the self term b is noise-dominated (-0.74). Result at 2^19, 64^2 x 128:
    # a = 0.880 + 0.070i, b = -0.912 - 0.182i (reference resolution: 0.911, -0.944; analytic 1, f - 2 = -0.914).
    from transit import transit
    r = transit("m2", "flattop", "flattop", particles=2**19, grid=(64, 64, 128))
    assert abs(r["a"] - 1) < 0.2, r
    assert abs(r["b"] - r["analytic_b"]) < 0.2, r
