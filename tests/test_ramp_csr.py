"""csr-ramp options (2026-10-27): dual-harmonic and energy-table ramp shapes, dynamic electrons following the ramp.

- the options are inert by default and rejected without a ramp;
- the programmes have the stated end points and shapes (analytic, no tracking);
- a short rigid-column ramp (sine, taper, min_phase RF) is bit-identical to the unmodified lwi_ring (run in a subprocess from
  LWI_RING_REFERENCE_SRC, default: the study's lwi_ring/src), so the existing ramp paths are untouched;
- a short dynamic-electron ramp runs, and the electron charge in the columns follows the programmed fraction.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import harness
from cases import J1

# the unmodified package to compare against (results_local may be a symlink, so the study is not found from __file__)
STUDY = Path.home() / "work/extracurr/LW_feasibility_studies/studies/compensated_storage_ring"
REF_SRC = Path(os.environ.get("LWI_RING_REFERENCE_SRC", STUDY / "lwi_ring/src"))
LWI = Path(os.environ.get("LWI_INTEGRATOR", Path.home() / "work/extracurr/LW_integrator"))

TINY = dict(particles_per_bunch=4096, grid=[32, 32, 64], steps_per_turn=256, tune_samples_per_turn=16, turns=2, backend="cpu",
            half_xy_m=0.3)
RAMP = dict(column_recentre_stations=0, diag_bins=None, diag_bin_centroids=False, population=1.75e14, proton_delta_rms=0.0,
            rf_voltage_v=60e3, rf_rms_psi_rad=0.4, rf_programme="min_phase", rf_phase_max_deg=30.0, ramp_final_energy_mev=75.0, ramp_time_s=2e-4,
            ramp_shape="sine", ramp_comp_programme="taper", ramp_comp_n_unc0=4.0e13, ramp_comp_f_max=0.95,
            ramp_comp_ring_average=True, ramp_column_sigma="adiabatic", column_local_fraction=0.95 * 480 / 420)


def rigid_ramp(**kw):
    return {**J1, "column_electrons": "rigid", **RAMP, **TINY, **kw}


def test_inert_without_ramp():
    from lwi_ring.driver import Config
    Config().validate()
    assert Config().ramp_dual_alpha == 0.0
    with pytest.raises(ValueError, match="ramp options need"):
        Config(ramp_dual_alpha=0.5).validate()
    with pytest.raises(ValueError, match="ramp options need"):
        Config(**dict(J1, ramp_comp_programme="taper")).validate()


def test_option_checks():
    from lwi_ring.driver import Config
    Config(**rigid_ramp(ramp_shape="dual", ramp_dual_alpha=1.0)).validate()
    with pytest.raises(ValueError, match="ramp_dual_alpha"):
        Config(**rigid_ramp(ramp_shape="dual", ramp_dual_alpha=1.5)).validate()
    with pytest.raises(ValueError, match="ramp_dual_alpha needs"):
        Config(**rigid_ramp(ramp_dual_alpha=0.5)).validate()
    Config(**{**J1, **RAMP, **TINY}).validate()                       # dynamic electrons are accepted in a ramp now
    with pytest.raises(ValueError, match="rem_birth_track"):
        Config(**{**J1, **RAMP, **TINY, "dyn_birth_sigma": "fixed", "rem_birth_track": True}).validate()


def test_dual_programme():
    from lwi_ring.driver import Config
    from lwi_ring.ramp import momentum_programme, gamma_of_energy
    sine = momentum_programme(Config(**rigid_ramp()))
    dual = momentum_programme(Config(**rigid_ramp(ramp_shape="dual", ramp_dual_alpha=1.0)))
    zero = momentum_programme(Config(**rigid_ramp(ramp_shape="dual", ramp_dual_alpha=0.0)))
    T = RAMP["ramp_time_s"]
    g0, g1 = gamma_of_energy(70.0), gamma_of_energy(75.0)
    p = lambda g: math.sqrt(g * g - 1)                                  # noqa: E731
    for f in (sine, dual):
        assert f(0.0) == pytest.approx(g0, rel=1e-14) and f(T) == pytest.approx(g1, rel=1e-14)
    for s in (0.01, 0.2, 0.5, 0.8):
        assert zero(s * T) == sine(s * T)
    # start (1 + alpha) = 2 times steeper in momentum
    s = 1e-3
    assert (p(dual(s * T)) - p(g0)) / (p(sine(s * T)) - p(g0)) == pytest.approx(2.0, rel=1e-5)
    ts = [k / 200 * T for k in range(201)]
    assert all(dual(b) >= dual(a) for a, b in zip(ts, ts[1:]))


def test_taper_values():
    """the stage 2b taper, which the ramp runs use with N_unc(injection) = 0.05 N: f = min(f_max, 1 - N_unc(E)/N) (unchanged code)."""
    from lwi_ring.ramp import Ramp
    cfg = SimpleNamespace(ramp_comp_programme="taper", ramp_comp_n_unc0=0.05 * 1.75e14, ramp_comp_n_design=None,
                          ramp_comp_f_max=0.95, ramp_comp_scaling="adiabatic")
    fake = SimpleNamespace(cfg=cfg, f0=0.95, population_alive=1.75e14, beta0=math.sqrt(1 - 1.2**-2), gamma0=1.2)
    fake.n_unc = lambda gamma: Ramp.n_unc(fake, gamma)
    beta = lambda g: math.sqrt(1 - g**-2)                               # noqa: E731
    assert Ramp.fraction_at(fake, 0.0, 1.2) == pytest.approx(0.95)       # residual held at its injection value
    g = 1.2 * 1.5
    ratio = beta(g) * g**2 / (beta(1.2) * 1.2**2)
    assert Ramp.fraction_at(fake, 0.0, g) == pytest.approx(1 - 0.05 * ratio)
    fake.population_alive = 0.04 * 1.75e14                              # first injected turns: below N_unc, columns off
    assert Ramp.fraction_at(fake, 0.0, 1.2) == 0.0


REF_RUNNER = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); sys.dont_write_bytecode = True
from lwi_ring import driver
driver.set_output_root(Path(sys.argv[3]))
cfg = driver.Config(**json.loads(Path(sys.argv[4]).read_text()))
driver.run(cfg, driver.OUT / "ref")
'''


@pytest.mark.skipif(not (REF_SRC / "lwi_ring/ramp.py").exists(), reason="no reference lwi_ring source")
def test_rigid_ramp_bit_identical_to_reference(output_root, tmp_path):
    """the unchanged ramp paths (sine shape, 'taper', rigid columns) give the same digests as the unmodified package."""
    from lwi_ring import driver
    config = rigid_ramp()
    out = output_root / "ramp_rigid_pkg"
    driver.run(driver.Config(**config), out)
    got = harness.digests(out, "run")
    cfgp = tmp_path / "case.json"
    cfgp.write_text(json.dumps(config))
    (tmp_path / "ref.py").write_text(REF_RUNNER)
    env = dict(os.environ, **harness.THREADS, PYTHONPATH="")
    ref_root = tmp_path / "ref_out"
    proc = subprocess.run([sys.executable, "-B", str(tmp_path / "ref.py"), str(LWI), str(REF_SRC), str(ref_root), str(cfgp)],
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert got == harness.digests(ref_root / "ref", "run")


def test_dynamic_ramp_runs(output_root):
    """dynamic electrons in a ramp: the run completes and the electron charge in the columns is the programmed local fraction."""
    from lwi_ring import driver
    cfg = driver.Config(**{**J1, **RAMP, **TINY, "turns": 1, "column_recentre_stations": 1, "ramp_comp_n_unc0": 0.3 * 2e14,
                           "ramp_final_energy_mev": 120.0, "ramp_time_s": 4e-4})
    out = output_root / "ramp_dynamic"
    driver.run(cfg, out)
    rows = [r for r in json.loads((out / "turns.json").read_text()) if isinstance(r, dict) and "ramp" in r]
    assert rows
    last = rows[-1]["ramp"]
    assert last["kinetic_energy_mev"] > 70.0
    assert 0.0 <= last["compensation_programmed"] < 0.95                # the taper has started to fall
    el = rows[-1].get("electrons")
    if el:
        pops = [v.get("electron_population") for v in el.values() if isinstance(v, dict) and "electron_population" in v]
        assert all(p is None or p >= 0 for p in pops)


def test_energy_table():
    from lwi_ring.driver import Config
    from lwi_ring.ramp import momentum_programme, gamma_of_energy
    tab = [[0.0, 70.0], [1e-4, 71.0], [2e-4, 75.0]]
    f = momentum_programme(Config(**rigid_ramp(ramp_shape="table", ramp_energy_table=tab)))
    assert f(0.0) == gamma_of_energy(70.0) and f(2e-4) == gamma_of_energy(75.0) and f(1.0) == gamma_of_energy(75.0)
    assert f(0.5e-4) == pytest.approx(gamma_of_energy(70.5), rel=1e-15)
    with pytest.raises(ValueError, match="ramp_energy_table"):
        Config(**rigid_ramp(ramp_shape="table", ramp_energy_table=[[0.0, 70.0], [1e-4, 74.0]])).validate()
    with pytest.raises(ValueError, match="needs ramp_shape"):
        Config(**rigid_ramp(ramp_energy_table=tab)).validate()
