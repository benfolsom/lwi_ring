"""lwi_ring: the compensated storage ring's native-PIC ring driver as one versioned package.

Built on the LW Integrator's PIC core (``core.pic``: backend selection, deposit/gather/Boris kernels, ``Species``), used as a library.
Every feature of the study's driver copies is a ``Config`` option; all defaults reproduce the study's ring_driver/ring_driver.py bit
for bit. Modules:

    driver        Config, run (single-turn-record driver, dynamic electrons), RingFields, ElectronLens, ColumnDrift, Racetrack, ...
    many_turn     many-turn tracking with an absorbing aperture, checkpoints and resume
    edyn_columns  self-consistent (dynamic) electron columns
    wallfb        segmented wall-electrode feedback
    damper        slice-by-slice transverse damper and the "coherent" diagnostics
    studyrun      the study runner (m = 0 / m = 2 column diagnostics, early stop, profile-shape diagnostic)
    presets       energy rescaling and field-scaled lattice presets used by the studies
"""
__version__ = "0.2.0"

# Versions of the source copies merged into this release (sha256 of their ring_driver.py, first 12 hex digits).
MERGED_SOURCES = {
    "ring_driver (canonical)": "a0d862d6a3db",
    "dipole": "f90b3ad026be",
    "stage3d": "89e6c604e94e",
    "drift": "e81995b8a06e",
    "recentre_dyn": "4e73604e410c",
    "recentre": "9a366edf0fd1",
    "esize": "6c18d5b64d42",
    "m2 / dyn400": "79cd6c2c8b49",
    "remedy2": "8b2119372f33",
    "remedy": "aa0c6c9b2a7c",
    "dynprof": "a7d451acab31",
    "wallfb": "725d7d88292f",
}
# wallfb.py is the csr-wallring copy as corrected on 2026-10-27 00:53 (l = 1 gain 1/(4 pi eps0 ...), was 1/(2 pi eps0 ...)).
WALLFB_SHA256 = "be3ce4911185"
