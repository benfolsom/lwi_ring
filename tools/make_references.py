"""Regenerate the bit-identity reference digests in tests/data/references.json.

usage: python tools/make_references.py [--sources=DIR] [--scratch=DIR] [--jobs=1|2] [--slow] [--only=NAME,...] [--no-write]
                                       [--busy-pattern=REGEX [--max-busy=10]]

Default mode (from the package): every case is run with lwi_ring itself and its digests replace the stored ones. Use this after an
intentional numerical change, or on another platform build (different NumPy/SciPy/BLAS), where round-off can change the digests. The
stored digests then describe lwi_ring on that build, not the earlier driver versions.

--sources=DIR (optional comparison with the earlier driver versions): each case is also run with a scratch snapshot of the version it
consolidates, and the two digest sets are compared; the stored digests are the earlier version's. DIR holds <label>/ for every label in
tests/cases.py SOURCES (the directory with that version's ring_driver.py and modules) and runners/<script> for the runner scripts
named in the cases. The originals are only read.

Other options:
  --scratch  where snapshots and outputs go (default: a new temporary directory)
  --no-write do not touch tests/data/references.json (SCRATCH/compare.json is still written)
  --busy-pattern  optional compute gate for shared machines: a case starts only while fewer than --max-busy (default 10) python
                  processes at >= 25% CPU match this regular expression (see tools/gated.py); no gate without it
Environment: LWI_ROOT (optional) is a LW Integrator checkout to put on PYTHONPATH when `core` is not installed; LWI_PYTHON is the
interpreter for the case runs (default: this one).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import ALL_CASES, CASES, SLOW_CASES  # noqa: E402
import gated  # noqa: E402
import harness  # noqa: E402

PY = os.environ.get("LWI_PYTHON", sys.executable)
LOCK = threading.Lock()
BUSY_PATTERN: str | None = None
MAX_BUSY = 10


def gate(label: str) -> None:
    if not BUSY_PATTERN:
        return
    while True:
        with LOCK:
            n = gated.busy(BUSY_PATTERN)
            if n < MAX_BUSY:
                print(time.strftime("%H:%M:%S"), f"start {label} ({n} busy)", flush=True)
                time.sleep(3)          # let the new process show up in ps before the next gate decision
                return
        time.sleep(2)


def pythonpath(*extra: Path) -> str:
    lwi = os.environ.get("LWI_ROOT")
    return os.pathsep.join([*map(str, extra), *([lwi] if lwi else [])])


def env(*extra: Path) -> dict:
    e = dict(os.environ, **harness.THREADS, PYTHONDONTWRITEBYTECODE="1")
    e["PYTHONPATH"] = pythonpath(*extra)
    return e


PKG_RUNNER = r'''
import json, sys
sys.path[:0] = [sys.argv[3]]
import harness
print("DIGESTS " + json.dumps(harness.run_package(sys.argv[1], __import__("pathlib").Path(sys.argv[2]))))
'''


def run_package(name: str, scratch: Path) -> dict:
    proc = subprocess.run([PY, "-B", "-c", PKG_RUNNER, name, str(scratch / "pkg"), str(PKG / "tests")], env=env(PKG / "src"),
                          capture_output=True, text=True, cwd=scratch)
    (scratch / f"pkg_{name}.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"lwi_ring run {name} failed:\n{proc.stderr[-3000:]}")
    return json.loads([line for line in proc.stdout.splitlines() if line.startswith("DIGESTS ")][-1][8:])


def run_case(name: str, scratch: Path, sources: Path | None) -> dict:
    gate(f"case {name}")
    t0 = time.time()
    row = dict(name=name)
    if sources is not None:
        row["reference"], row["provenance"] = harness.run_reference(name, sources, scratch / "ref", PY, env())
    t1 = time.time()
    # same gated slot: the lwi_ring run follows the reference immediately
    row["lwi_ring"] = run_package(name, scratch)
    t2 = time.time()
    row["identical"] = row["reference"] == row["lwi_ring"] if sources is not None else True
    row.update(ref_s=t1 - t0, pkg_s=t2 - t1)
    print(time.strftime("%H:%M:%S"), name, ("IDENTICAL" if row["identical"] else "DIFFERENT") if sources is not None else "done",
          f"({t1 - t0:.0f} s / {t2 - t1:.0f} s)", flush=True)
    return row


def main() -> None:
    global BUSY_PATTERN, MAX_BUSY
    opts = dict(a.split("=", 1) if "=" in a else (a, "1") for a in sys.argv[1:])
    sources = Path(opts["--sources"]).resolve() if "--sources" in opts else None
    scratch = Path(opts["--scratch"]).resolve() if "--scratch" in opts else Path(tempfile.mkdtemp(prefix="lwi_ring_refs_"))
    BUSY_PATTERN = opts.get("--busy-pattern")
    MAX_BUSY = int(opts.get("--max-busy", "10"))
    scratch.mkdir(parents=True, exist_ok=True)
    names = list(CASES) + (list(SLOW_CASES) if "--slow" in opts else [])
    if "--only" in opts:
        names = opts["--only"].split(",")
    jobs = min(2, int(opts.get("--jobs", "1")))
    compare_path = scratch / "compare.json"
    results = json.loads(compare_path.read_text()) if compare_path.exists() else {}
    with ThreadPoolExecutor(jobs) as pool:
        for row in pool.map(lambda n: _safe(n, scratch, sources), names):
            results[row["name"]] = row
            compare_path.write_text(json.dumps(results, indent=1))
    if "--no-write" not in opts:
        refp = PKG / "tests/data/references.json"
        refs = json.loads(refp.read_text()) if refp.exists() else dict(cases={})
        import numpy, scipy
        refs["environment"] = dict(python_version=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
                                   numpy=numpy.__version__, scipy=scipy.__version__, threads=harness.THREADS,
                                   made=time.strftime("%Y-%m-%d %H:%M"))
        for name in names:
            row = results.get(name, {})
            if "lwi_ring" not in row:
                continue
            source, entry, _, script, flags = ALL_CASES[name]
            case = refs["cases"].setdefault(name, dict(source=source, entry=entry, script=script, flags=flags))
            if sources is not None:
                case.update(row["provenance"], digests=row["reference"], digests_from="source version")
            else:
                case.update(digests=row["lwi_ring"], digests_from="package")
        refp.write_text(json.dumps(refs, indent=1, sort_keys=True) + "\n")
    bad = [n for n in names if not results.get(n, {}).get("identical")]
    if sources is not None:
        print("all identical" if not bad else f"not identical or failed: {bad}")
    else:
        print("done" if not bad else f"failed: {bad}")


def _safe(name: str, scratch: Path, sources: Path | None) -> dict:
    try:
        return run_case(name, scratch, sources)
    except Exception as err:          # keep going; the failure is recorded
        print(time.strftime("%H:%M:%S"), name, "ERROR", str(err)[-500:], flush=True)
        return dict(name=name, error=str(err)[-3000:], identical=False)


if __name__ == "__main__":
    main()
