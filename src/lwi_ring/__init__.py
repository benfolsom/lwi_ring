"""lwi_ring: a native strong-strong PIC ring driver for space-charge-compensated proton rings, as one versioned package.

Built on the LW Integrator's PIC core (``core.pic``: backend selection, deposit/gather/Boris kernels, ``Species``), used as a library.
Every feature is a ``Config`` option and all defaults reproduce the baseline ring driver bit for bit. Modules:

    driver        Config, run (single-turn-record driver, dynamic electrons), RingFields, ElectronLens, ColumnDrift, Racetrack, ...
    many_turn     many-turn tracking with an absorbing aperture, checkpoints and resume
    edyn_columns  self-consistent (dynamic) electron columns
    wallfb        segmented wall-electrode feedback
    damper        slice-by-slice transverse damper and the "coherent" diagnostics
    studyrun      runner with m = 0 / m = 2 column diagnostics, early stop and profile-shape diagnostic
    presets       energy rescaling and field-scaled lattice presets
"""
__version__ = "0.2.1"
