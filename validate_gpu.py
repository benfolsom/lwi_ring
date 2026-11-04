#!/usr/bin/env python
"""Check that the GPU backend reproduces the CPU float64 reference on a short, small version of the leading configuration.

usage:  python validate_gpu.py [--backend cupy] [--precision float32] [--turns 2] [--particles 16384] [--out DIR] [--reuse-cpu]

Runs the same case twice (CPU float64, then the GPU backend), compares emittance, losses, ellipticity and centroid diagnostics against
tolerances derived from the Apple-silicon MLX float32 run against CPU float64 (see report.md), and prints PASS/FAIL per quantity plus
timing per step for both. Exit status 0 only if everything passes. Needs PYTHONPATH to hold <tree>/src and the LW_integrator checkout
(or set LW_INTEGRATOR_ROOT); see README_GPU.md. The CPU run takes a few minutes (about 0.14 s/step single-threaded, 1024 steps).
A FAIL on a statistical quantity (centroids, ellipticity) alone is weaker evidence than one on emittance or losses: rerun with
--tol-scale 2 before concluding anything, and send the printed table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
if os.environ.get("LW_INTEGRATOR_ROOT"):
    sys.path.insert(0, os.environ["LW_INTEGRATOR_ROOT"])

# Leading configuration (G400g4_F24_wfb_rc_ends_coll_N175_4x_t150) at 1/16 of the particles and 1/3 of the grid; keys equal to the Config
# defaults are omitted. The primary collimator is narrowed to 2.5 sigma so that the loss bookkeeping is exercised.
BASE = {
    "energy_mev": 400.0, "circumference_m": 480.0, "population": 1552310135480894.5, "harmonic": 2, "duty": 0.3, "sigma_m": 0.03,
    "tune": 8.38, "tune_x": 8.38, "tune_y": 8.38, "particles_per_bunch": 16384, "grid": [32, 32, 64], "half_xy_m": 0.3, "turns": 2,
    "opposing": "absent", "aperture_m": 0.25, "error_superperiods": 6, "error_dk_rms": 0.001, "error_dipole_rms_rad": 7e-05,
    "error_seed": 105, "straight_length_m": 17.5, "arc_length_m": 2.5, "straight_focus_ratio": 0.0,
    "solenoid_bz_t": 0.3489961637143074, "proton_delta_rms": 0.003, "proton_xi_x": -8.38, "proton_xi_y": -8.38, "gamma_t": 12.4,
    "lattice_periods": 24, "solenoid_retune": True, "solenoid_edge_in_drift": True, "column_sections": "straights",
    "column_local_fraction": 1.3221990257480873, "column_follow_orbit": True, "column_sigma_follow": True,
    "column_electrons": "dynamic", "injection_profile": "truncated_gaussian", "injection_edge_m": 0.12,
    "start_on_closed_orbit": True, "diag_bins": 96, "column_recentre_stations": 1, "dyn_birth_sigma": "fixed",
    "dyn_aperture_kill": True, "rem_station_pos": [0.5], "recycle_stride": 2, "recycle_gun_every": 6, "recycle_centroid_eta": 0.9,
    "recycle_noise_m": 0.0001, "diag_bin_centroids": True, "wfb_enable": True, "wfb_gain1": 2.0, "wfb_start_turn": 0,
    "dyn_electron_bore_m": 0.15, "cend_gap_in_m": 1.43, "cend_gap_out_m": 1.43, "cend_reset_gap_m": 1.56, "cend_taper_m": 0.27,
    "cend_taper_offset_m": 0.03, "cend_arms": "Z", "cend_kick_rms_rad": 3e-05, "cend_seed": 1,
    "cend_k1l": [-0.0002727928713730348, -0.0002724363231508524],
    "cend_k1sl": [-0.0022666877409280102, 0.0022677944206182486],
    "coll_elements": [
        {"name": "primary", "s_m": 18.75, "kind": "primary", "radius_m": 0.075, "material": "tungsten", "thickness_m": 0.00017},
        {"name": "secondary_0", "s_m": 22.75, "kind": "secondary", "radius_m": 0.162},
        {"name": "secondary_1", "s_m": 38.75, "kind": "secondary", "radius_m": 0.206},
        {"name": "secondary_2", "s_m": 58.75, "kind": "secondary", "radius_m": 0.206}],
    "coll_arc_aperture_m": 0.25,
    "steps_per_turn": 512,          # fewer steps make a step span two layout boundaries and the driver refuses
}

# Tolerances (relative difference |a-b|/max(|a|,|b|,floor)) for the float32 GPU run against CPU float64 on this case. Set from the
# measured MLX float32 discrepancy (report.md): 2-3x the MLX value. Counts are macro-particle counts.
TOL = {
    "emit4d_m": 0.01, "emit_canon_x_m": 0.04, "emit_canon_y_m": 0.04, "rms_x_m": 0.03, "rms_y_m": 0.03,
    "incoh_norm_emit_x_m": 0.06, "incoh_norm_emit_y_m": 0.06,
    "alive_fraction": 0.002, "electron_alive_fraction": 0.01,
    "ellipticity_proton_mean": 0.05, "ellipticity_electron_mean": 0.05, "r4_over_r2sq_proton": 0.06,
    "centroid_rms_m": 0.10, "wallfb_s2_ellipticity_rms": 0.25, "wallfb_v1_V_rms": 0.10,
    "coll_lost_macro": 0.03, "recycle_fresh_births": 0.001,
}
# Absolute floors added to the denominator (quantities that can be exactly or nearly zero).
FLOOR = {"alive_fraction": 1.0, "electron_alive_fraction": 1.0, "coll_lost_macro": 20.0, "recycle_fresh_births": 20.0,
         "centroid_rms_m": 1e-4}


def finite_mean(values: list) -> float:
    """Mean over the columns where the shape measurement exists (the diagnostic writes NaN/None for under-populated columns)."""
    good = [v for v in values if v is not None and math.isfinite(v)]
    if not good:
        raise SystemExit("shape diagnostic is missing in every column")
    return sum(good) / len(good)


def metrics(directory: Path) -> dict:
    """Pull the comparison quantities out of one run directory (the last turn)."""
    mean = lambda rows: sum(map(sum, rows)) / sum(map(len, rows))
    m2 = [json.loads(line) for line in (directory / "m2diag.jsonl").read_text().splitlines() if line.strip()]
    last = m2[-1]
    turns = json.loads((directory / "turns.json").read_text())[-1]
    ele = turns["electrons"]
    coll = json.loads((directory / "collimation_summary.json").read_text())
    wfb = [json.loads(line) for line in (directory / "wallfb.jsonl").read_text().splitlines() if line.strip()][-1]
    summary = json.loads((directory / "summary.json").read_text())
    out = {
        "emit4d_m": last["emit4d_m"],
        "emit_canon_x_m": last["emit_canon_xy_m"][0], "emit_canon_y_m": last["emit_canon_xy_m"][1],
        "rms_x_m": last["rms_xy_m"][0], "rms_y_m": last["rms_xy_m"][1],
        "incoh_norm_emit_x_m": turns["coherent"]["incoh_normalized_emittance_m"][0],
        "incoh_norm_emit_y_m": turns["coherent"]["incoh_normalized_emittance_m"][1],
        "alive_fraction": last["alive_fraction"],
        "electron_alive_fraction": sum(v["alive_fraction"] for v in ele.values() if "alive_fraction" in v) / sum("alive_fraction" in v for v in ele.values()),
        "ellipticity_proton_mean": mean(last["bins"]["proton"]["ellipticity"]),
        "ellipticity_electron_mean": mean(last["bins"]["electron"]["ellipticity"]),
        "r4_over_r2sq_proton": finite_mean(last["shape"]["proton"]),
        "centroid_rms_m": turns["coherent"]["centroid_rms_m"],
        "wallfb_s2_ellipticity_rms": wfb["s2_ellipticity"]["rms"], "wallfb_v1_V_rms": wfb["v1_V"]["rms"],
        "coll_lost_macro": sum(v["macro"] for v in coll["cumulative"].values()),
        "recycle_fresh_births": ele["recycle"]["fresh_births"],
    }
    t = summary["timing_mean_s"]
    # CUDA launches are asynchronous: deposit_s ends before the kernel does and the next synchronize (in solve) absorbs it, so only the
    # deposit+solve sum and the whole step are comparable between backends.
    out["_timing"] = {k: t[k] for k in ("step_s", "gather_s", "push_drift_s") if k in t}
    out["_timing"]["deposit_plus_solve_s"] = t["deposit_s"] + t["solve_s"]
    out["_timing"]["init_s"] = summary.get("initialization_s")
    out["_timing"]["elapsed_s"] = summary.get("elapsed_s")
    return out


def run_one(driver, studyrun, backend: str, precision: str | None, args, root: Path) -> dict:
    cfg = dict(BASE, particles_per_bunch=args.particles, turns=args.turns, backend=backend)
    if precision:
        cfg["precision"] = precision
    directory = root / f"{backend}{'_' + precision if precision else ''}"
    reusable = False
    if directory.exists() and args.reuse_cpu and backend == "cpu":      # only the CPU reference may be reused, and only if it matches
        try:
            c = json.loads((directory / "provenance.json").read_text())["config"]
            reusable = (c["particles_per_bunch"], c["turns"], c["steps_per_turn"]) == (args.particles, args.turns, BASE["steps_per_turn"]) \
                and json.loads((directory / "summary.json").read_text())["status"] == "complete"
        except (OSError, KeyError, ValueError):
            reusable = False
    if directory.exists() and not reusable:
        import shutil
        shutil.rmtree(directory)
    if not directory.exists():
        t0 = time.time()
        status = studyrun.run_case(driver.Config(**cfg), directory, True, 4, True, None)
        print(f"[{backend}] {status}  wall {time.time() - t0:.1f} s", flush=True)
        if status.get("status") != "complete":
            raise SystemExit(f"{backend} run did not complete: {status}")
    return metrics(directory)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", default="cupy", choices=("cupy", "mlx"))
    ap.add_argument("--precision", default="float32", choices=("float32", "float64"))
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--particles", type=int, default=16384, help="power of two; the truncated-Gaussian injection fails below ~8192")
    ap.add_argument("--out", type=Path, default=HERE / "validate_gpu_out")
    ap.add_argument("--reuse-cpu", action="store_true", help="reuse a matching, completed CPU reference under --out (the GPU run is always redone)")
    ap.add_argument("--tol-scale", type=float, default=1.0)
    ap.add_argument("--print-only", action="store_true", help="print relative differences without PASS/FAIL (used to calibrate TOL)")
    args = ap.parse_args()
    args.out = args.out.resolve()
    os.environ["LWI_RING_OUT"] = str(args.out)          # the driver fixes its output root and Numba cache at import
    args.out.mkdir(parents=True, exist_ok=True)
    from lwi_ring import driver, studyrun
    driver.set_output_root(args.out)

    cpu = run_one(driver, studyrun, "cpu", None, args, args.out)
    gpu = run_one(driver, studyrun, args.backend, args.precision if args.backend == "cupy" else None, args, args.out)
    label = f"{args.backend}/{args.precision if args.backend == 'cupy' else 'float32'}"

    print(f"\n{'quantity':28s} {'cpu/float64':>14s} {label:>14s} {'rel.diff':>10s} {'tol':>8s}")
    failures = 0
    for key, tol in TOL.items():
        a, b = cpu[key], gpu[key]
        rel = abs(a - b) / max(abs(a), abs(b), FLOOR.get(key, 1e-30))
        ok = math.isfinite(rel) and rel <= tol * args.tol_scale
        failures += not ok
        print(f"{key:28s} {a:14.6g} {b:14.6g} {rel:10.3g} {tol * args.tol_scale:8.3g}  {'PASS' if ok or args.print_only else 'FAIL'}")
    print(f"\ntiming per step (mean over run, seconds){'':4s}{'cpu/float64':>14s} {label:>14s} {'speed-up':>9s}")
    for key in cpu["_timing"]:
        a, b = cpu["_timing"][key], gpu["_timing"].get(key)
        if a is None or b is None:
            continue
        print(f"  {key:38s} {a:14.4g} {b:14.4g} {a / b if b else float('nan'):9.2f}")
    print("\n" + ("PASS: all quantities within tolerance" if not failures else f"FAIL: {failures} quantities outside tolerance")
          if not args.print_only else "\n(print-only: no verdict)")
    return 1 if failures and not args.print_only else 0


if __name__ == "__main__":
    sys.exit(main())
