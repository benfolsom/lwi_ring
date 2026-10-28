"""lwi_ring against the earlier driver versions it consolidates, bit for bit (digests in data/references.json, made by
tools/make_references.py).

The references hold for the recorded environment (single-threaded CPU kernels, Python 3.12, NumPy 2.4.2, SciPy 1.17.0, macOS arm64).
On another NumPy/SciPy/BLAS build the digests can differ by round-off; regenerate them there with `tools/make_references.py` (from the
package itself) rather than loosening the comparison.
"""
from __future__ import annotations

import json

import pytest

import harness
from cases import CASES, DATA, SLOW_CASES

REFS = json.loads((DATA / "references.json").read_text()) if (DATA / "references.json").exists() else dict(cases={})


def check(name: str, root) -> None:
    ref = REFS["cases"].get(name)
    if ref is None:
        pytest.skip(f"no reference recorded for {name}")
    got = harness.run_package(name, root)
    assert got == ref["digests"], f"{name} differs from its reference digests (version: {ref['source']})"


@pytest.mark.parametrize("name", sorted(CASES))
def test_fast(name, output_root):
    check(name, output_root)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(SLOW_CASES))
def test_slow(name, output_root):
    check(name, output_root)


def test_unified_resets_alias(output_root, monkeypatch):
    """column_drift_resets on dynamic electrons is dyn_resets: same digests as the m2_resets_ab2 case."""
    import cases
    ref = REFS["cases"].get("m2_resets_ab2")
    if ref is None:
        pytest.skip("no reference recorded for m2_resets_ab2")
    src, entry, cfg, script, flags = cases.CASES["m2_resets_ab2"]
    alias = dict(cfg, column_drift_resets=cfg["dyn_resets"])
    alias.pop("dyn_resets")
    monkeypatch.setitem(cases.ALL_CASES, "m2_resets_alias", (src, entry, alias, script, flags))
    assert harness.run_package("m2_resets_alias", output_root) == ref["digests"]
