"""Run a bit-identity case with lwi_ring or with a snapshot of a source copy, and digest its outputs (SHA-256 of the raw arrays and of
the canonical JSON of the records; wall-clock fields removed)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from cases import ALL_CASES

# Single-threaded kernels: the copies' bit-identity runs used the same environment (csrm2_bitid.sh).
THREADS = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMBA_NUM_THREADS="1")
MANY_TURN = dict(aperture_m=0.17, tracked_per_beam=2048, samples_per_turn=16, checkpoint_turns=1)


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def jsha(rows) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


def jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digests(directory: Path, entry: str, status: str | None = None) -> dict:
    """Digests of a finished case directory (driver.run / studyrun or many_turn layout)."""
    d = Path(directory)
    out = {}
    if entry == "many_turn":
        with np.load(sorted((d / "checkpoints").glob("turn_*.npz"))[-1]) as z:
            out["mt_ckpt"] = {k: sha(z[k]) for k in z.files if k != "header"}
        rows = jsonl(d / "turns.jsonl")
        for r in rows:
            r.pop("elapsed_s", None)
        out["mt_turns"] = jsha(rows)
        out["mt_losses"] = hashlib.sha256((d / "losses.jsonl").read_bytes()).hexdigest()
        return out
    if (d / "final_particles.npz").exists():
        with np.load(d / "final_particles.npz") as z:
            out["final"] = {k: sha(z[k]) for k in z.files}
    rows = json.loads((d / "turns.json").read_text())
    for r in rows:
        if isinstance(r, dict):
            r.pop("elapsed_s", None)
    out["turns"] = jsha(rows)
    for name in ("m2diag.jsonl", "wallfb.jsonl"):
        if (d / name).exists():
            out[name] = jsha(jsonl(d / name))
    if status is not None:
        out["status"] = status
    return out


# ---- lwi_ring ----------------------------------------------------------------------------------------------------------------------
def run_package(name: str, root: Path) -> dict:
    """Run case `name` with lwi_ring in this process; output below root (the driver's output root)."""
    from lwi_ring import driver, many_turn, studyrun
    source, entry, config, script, flags = ALL_CASES[name]
    driver.set_output_root(root)
    out = driver.OUT / name
    if out.exists():
        shutil.rmtree(out)
    cfg = driver.Config(**json.loads(json.dumps(config)))
    if entry == "run":
        driver.run(cfg, out)
        return digests(out, entry)
    if entry == "many_turn":
        many_turn.run(cfg, out, many_turn.Settings(**MANY_TURN))
        return digests(out, entry)
    opts = dict(diagnostics=script != "csrrcd_run.py", shape=script in ("csrdp_run.py", "csrwr_run.py"), every=4, stop_growth=None)
    for flag in flags:
        if flag.startswith("--stop-growth="):
            opts["stop_growth"] = float(flag.split("=", 1)[1])
    result = studyrun.run_case(cfg, out, **opts)
    return digests(out, entry, result["status"])


# ---- source copies (reference generation, tools/make_references.py) -----------------------------------------------------------------
REF_RUNNER = r'''
import json, sys
from pathlib import Path
drv, cfgp, entry, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
sys.path.insert(0, str(drv)); sys.dont_write_bytecode = True
import ring_driver as rd
cfg = rd.Config(**json.loads(cfgp.read_text()))
out = rd.OUT / name
if entry == "run":
    rd.run(cfg, out)
else:
    import many_turn as mt
    mt.run(cfg, out, mt.Settings(**json.loads(sys.argv[5])))
'''


def run_reference(name: str, study: Path, scratch: Path, python: str, env: dict) -> tuple[dict, dict]:
    """Run case `name` with a scratch snapshot of its source copy (the original is only read). Returns (digests, provenance)."""
    source, entry, config, script, flags = ALL_CASES[name]
    src = study / SOURCE_DIRS[source]
    snap_root = scratch / name
    if snap_root.exists():
        shutil.rmtree(snap_root)
    drv = snap_root / "ring_driver"
    shutil.copytree(src, drv, ignore=shutil.ignore_patterns("__pycache__", "results_local", "bitid_reference", "validation"))
    (snap_root / "scripts").mkdir()
    cfgp = snap_root / "case.json"
    cfgp.write_text(json.dumps(config))
    out_root = snap_root / "results_local/ring_pic"
    if entry == "study":
        shutil.copy2(study / "scripts" / script, snap_root / "scripts" / script)
        cmd = [python, "-B", str(snap_root / "scripts" / script), str(drv), str(cfgp), name, *flags]
    else:
        (snap_root / "ref_case.py").write_text(REF_RUNNER)
        cmd = [python, "-B", str(snap_root / "ref_case.py"), str(drv), str(cfgp), entry, name, json.dumps(MANY_TURN)]
    proc = subprocess.run(cmd, cwd=drv, env=env, capture_output=True, text=True)
    (snap_root / "ref.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"reference run {name} failed:\n{proc.stderr[-3000:]}")
    status = None
    if entry == "study":
        status = json.loads([line for line in proc.stdout.splitlines() if line.startswith("{")][-1])["status"]
    prov = dict(source=source, source_dir=SOURCE_DIRS[source], entry=entry, script=script, flags=flags,
                ring_driver_sha256=hashlib.sha256((src / "ring_driver.py").read_bytes()).hexdigest(),
                source_files_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(src.glob("*.py"))})
    return digests(out_root / name, entry, status), prov


from cases import SOURCES as SOURCE_DIRS  # noqa: E402
