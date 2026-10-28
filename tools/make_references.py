"""Make the bit-identity references: run every case with a scratch snapshot of its source copy, then with lwi_ring, and compare.

usage: python tools/make_references.py [--jobs=1|2] [--study=DIR] [--scratch=DIR] [--slow] [--only=NAME,...] [--no-write]
  --study    the compensated_storage_ring study directory (default: the parent of this package)
  --scratch  where snapshots and outputs go (default: STUDY/results_local/lwi_ring/refs)
Writes tests/data/references.json (reference digests of every case run, with the source files' hashes and the environment) unless
--no-write, and SCRATCH/compare.json (reference and lwi_ring digests side by side). The source copies are only read: each case runs
on a fresh copy of the copy's directory (and of its runner script) inside SCRATCH.

Compute gate (the study's shared-machine rule): a process is started only while fewer than 10 study compute processes are busy (python at
>= 25% CPU whose command mentions csr, ring_driver, many_turn, impactx, xsuite, warpx or lw) and at most --jobs of ours run.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "tests"))
from cases import ALL_CASES, CASES, SLOW_CASES  # noqa: E402
import harness  # noqa: E402

PATTERN = r"csr|ring_driver|many_turn|impactx|xsuite|warpx|lw"
LWI = Path(os.environ.get("LWI_ROOT", str(Path.home() / "work/extracurr/LW_integrator")))
PY = os.environ.get("LWI_PYTHON", str(LWI / ".venv/bin/python"))
LOCK = threading.Lock()


def busy() -> int:
    out = subprocess.run(["ps", "-axo", "%cpu=,command="], capture_output=True, text=True).stdout
    n = 0
    for line in out.splitlines():
        cpu, _, cmd = line.strip().partition(" ")
        try:
            hot = float(cpu) >= 25
        except ValueError:
            continue
        n += hot and "python" in cmd.split(" ")[0].lower() and bool(re.search(PATTERN, cmd, re.I))
    return n


def gate(label: str) -> None:
    while True:
        with LOCK:
            n = busy()
            if n < 10:
                print(time.strftime("%H:%M:%S"), f"start {label} ({n} busy)", flush=True)
                time.sleep(3)          # let the new process show up in ps before the next gate decision
                return
        time.sleep(2)


def env() -> dict:
    e = dict(os.environ, **harness.THREADS, PYTHONDONTWRITEBYTECODE="1")
    e["PYTHONPATH"] = str(LWI)
    return e


PKG_RUNNER = r'''
import json, sys
sys.path[:0] = [sys.argv[3]]
import harness
print("DIGESTS " + json.dumps(harness.run_package(sys.argv[1], __import__("pathlib").Path(sys.argv[2]))))
'''


def run_case(name: str, scratch: Path) -> dict:
    gate(f"case {name}")
    t0 = time.time()
    ref, prov = harness.run_reference(name, STUDY, scratch / "ref", PY, env())
    t1 = time.time()
    # same slot: the lwi_ring run follows the reference immediately (one gated start per case, one busy process of ours at a time)
    e = env()
    e["PYTHONPATH"] = os.pathsep.join([str(LWI), str(PKG / "src")])
    root = scratch / "pkg"
    proc = subprocess.run([PY, "-B", "-c", PKG_RUNNER, name, str(root), str(PKG / "tests")], env=e, capture_output=True, text=True,
                          cwd=scratch)
    (scratch / f"pkg_{name}.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"lwi_ring run {name} failed:\n{proc.stderr[-3000:]}")
    new = json.loads([line for line in proc.stdout.splitlines() if line.startswith("DIGESTS ")][-1][8:])
    t2 = time.time()
    row = dict(name=name, reference=ref, lwi_ring=new, identical=ref == new, provenance=prov, ref_s=t1 - t0, pkg_s=t2 - t1)
    print(time.strftime("%H:%M:%S"), name, "IDENTICAL" if row["identical"] else "DIFFERENT", f"({t1 - t0:.0f} s / {t2 - t1:.0f} s)",
          flush=True)
    return row


def main() -> None:
    global STUDY
    opts = dict(a.split("=", 1) if "=" in a else (a, "1") for a in sys.argv[1:])
    STUDY = Path(opts.get("--study", str(PKG.parent))).resolve()
    scratch = Path(opts.get("--scratch", str(STUDY / "results_local/lwi_ring/refs"))).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    names = list(CASES) + (list(SLOW_CASES) if "--slow" in opts else [])
    if "--only" in opts:
        names = opts["--only"].split(",")
    jobs = min(2, int(opts.get("--jobs", "1")))
    compare_path = scratch / "compare.json"
    results = json.loads(compare_path.read_text()) if compare_path.exists() else {}
    with ThreadPoolExecutor(jobs) as pool:
        for row in pool.map(lambda n: _safe(n, scratch), names):
            results[row["name"]] = row
            compare_path.write_text(json.dumps(results, indent=1))
    if "--no-write" not in opts:
        refp = PKG / "tests/data/references.json"
        refs = json.loads(refp.read_text()) if refp.exists() else dict(cases={})
        import numpy, scipy
        refs["environment"] = dict(python=PY, platform=platform.platform(), machine=platform.machine(), numpy=numpy.__version__,
                                   scipy=scipy.__version__, lwi_root=str(LWI), threads=harness.THREADS,
                                   made=time.strftime("%Y-%m-%d %H:%M"))
        for name, row in results.items():
            if "reference" in row:
                refs["cases"][name] = dict(digests=row["reference"], **row["provenance"])
        refp.write_text(json.dumps(refs, indent=1, sort_keys=True) + "\n")
    bad = [n for n in names if not results.get(n, {}).get("identical")]
    print("all identical" if not bad else f"not identical: {bad}")


def _safe(name: str, scratch: Path) -> dict:
    try:
        return run_case(name, scratch)
    except Exception as err:          # keep going; the failure is recorded
        print(time.strftime("%H:%M:%S"), name, "ERROR", str(err)[-500:], flush=True)
        return dict(name=name, error=str(err)[-3000:], identical=False)


if __name__ == "__main__":
    main()
