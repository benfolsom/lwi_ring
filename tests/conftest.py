"""Single-threaded kernels (the references were made so) and an isolated output root per session."""
import os
import tempfile

# The driver fixes its output root and the Numba cache directory at import: point both at a scratch directory before lwi_ring loads.
os.environ.setdefault("LWI_RING_OUT", tempfile.mkdtemp(prefix="lwi_ring_pytest_"))
for _key, _value in dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMBA_NUM_THREADS="1").items():
    os.environ.setdefault(_key, _value)

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def output_root(tmp_path_factory):
    from lwi_ring import driver
    root = tmp_path_factory.mktemp("ring_pic")
    driver.set_output_root(root)
    return root
