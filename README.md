# lwi_ring

A native strong–strong particle-in-cell ring driver for space-charge-compensated proton rings, with co-moving magnetised electron
columns (rigid, drifting-centroid or self-consistent), wall-electrode feedback, recycled electrons and energy ramps. It is built on
the PIC core of the [LW Integrator (LWI)](https://github.com/benfolsom/LW_integrator) and is developed alongside a feasibility study
of a compensated rapid-cycling synchrotron; results and reports are published separately as they mature. Licence: GPL-3.0.

The compensated storage ring's ring driver as one versioned Python package. It replaces the diverging copies of `ring_driver/` under
`results_local/*/driver`. It is the study's strong–strong native-PIC ring driver, built on the LW Integrator's PIC core (`core.pic`:
backend selection, deposit, gather, integrated-cell Green function, Boris push, `Species`), used as a library and not modified.

Every feature that a study added in its own copy is an explicit `Config` option. **All defaults reproduce the canonical
`ring_driver/ring_driver.py` bit for bit**, and every copy's feature cases reproduce that copy bit for bit
(`tests/`, `report_lwi_ring_consolidation.md`).

Version 0.1.0 merges the copies listed in `lwi_ring.MERGED_SOURCES`: canonical → dipole → stage3d → drift → recentre_dyn → recentre →
esize, m2 (= dyn400) → remedy2 → remedy, remedy2 → dynprof → wallfb.

Version 0.2.0 adds two study patches, both inert at their defaults (all fast and slow bit-identity cases unchanged):
- **recycled electrons** (`recycle_*`, `report_recycled_electrons.md`): electrons leaving column k pass a proton-free transport
  (strain hold, centroid re-centring, rotation, current loss, scraping, monitor noise) and re-enter column k + `recycle_stride`;
  fresh guns only at stream starts (`tests/test_recycle.py`);
- **ramp** (`ramp_shape` "dual" and "table", `ramp_energy_table`, `ramp_solenoid_follow`, `ramp_column_sigma`, `report_ramp.md`):
  dual-harmonic and tabulated magnet cycles, and self-consistent (dynamic) electrons through an energy ramp (`tests/test_ramp_csr.py`).

## Install

```sh
# the LW Integrator (provides `core`) first, from its checkout, then this package
pip install -e ~/work/extracurr/LW_integrator
pip install -e ~/work/extracurr/LW_feasibility_studies/studies/compensated_storage_ring/lwi_ring[test]
```

The LW Integrator's own `.venv` has no pip. With it, use the source tree directly:

```sh
PY=~/work/extracurr/LW_integrator/.venv/bin/python
export PYTHONPATH=~/work/extracurr/LW_integrator:~/work/extracurr/LW_feasibility_studies/studies/compensated_storage_ring/lwi_ring/src
```

## Modules

| module | contents |
|---|---|
| `lwi_ring.driver` | `Config`, `run` (the single-run driver; dynamic electrons, wall feedback and dampers live here), `RingFields`, `ElectronLens`, `ColumnDrift`, `Racetrack`, `LatticeErrors`, `initial_beams`, `physics`, `solver_lock`, `set_output_root` |
| `lwi_ring.many_turn` | many-turn tracking with an absorbing aperture, checkpoints and resume (rigid columns, centroid drift, dampers) |
| `lwi_ring.edyn_columns` | self-consistent (dynamic) electron columns: births, resets, stations, shaping, profiles, gun-size tracking |
| `lwi_ring.wallfb` | segmented wall-electrode feedback |
| `lwi_ring.damper` | slice-by-slice transverse damper and the `coherent` turn diagnostics |
| `lwi_ring.longitudinal`, `ramp`, `injection`, `lattice_linear` | as in `ring_driver/` (ramp.py: the canonical, newer version) |
| `lwi_ring.studyrun` | the study runner: m = 0 / m = 2 column diagnostics, early stop, profile-shape diagnostic |
| `lwi_ring.presets` | set-up rules that were in config scripts: `rescale_energy` (400 MeV), `FIELD_2X` / `FIELD_1P5X`, `station_positions` |

## Output root

The copies wrote below `<copy>/../results_local/ring_pic` (solver locks, the Numba cache, default run directories, and many_turn's
"output must be below" check). lwi_ring uses `$LWI_RING_OUT` if it is set, otherwise `./results_local/ring_pic` below the working
directory. `driver.set_output_root(path)` changes it at run time. Set `LWI_RING_OUT` to the lane's directory to keep one solver lock
per lane, as the copies did.

## Running

CPU (one thread per process, as the studies run it):

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 LWI_RING_OUT=$PWD/results_local/ring_pic
# a study case with the m = 2 column diagnostics (the csrm2/csrrem/csrdp/csrwr runners)
$PY -B -m lwi_ring.studyrun CONFIG.json NAME [key=JSON ...] [--diag-every=4] [--stop-growth=1.0] [--shape] [--no-diag]
# the plain driver (the copies' ring_driver.py command line, without --checks)
$PY -B -m lwi_ring.driver --config CONFIG.json --output DIR [--set key=JSON] [--pilot-steps N] [--refine dt]
# many turns with an aperture (rigid columns)
$PY -B -m lwi_ring.many_turn --config CONFIG.json --output $LWI_RING_OUT/NAME [--aperture-mm 170]
```

Installed with pip, the same commands are `lwi-ring-run`, `lwi-ring-driver` and `lwi-ring-many-turn`. From Python:

```python
from lwi_ring import driver, studyrun
cfg = driver.Config(**json.load(open("case.json")))
driver.run(cfg, Path("out/case"))                      # or studyrun.run_case(cfg, out, every=4, shape=True, stop_growth=1.0)
```

MLX GPU: set `backend="mlx"` in the config (or `--backend mlx`). The kernels then run in float32 on the GPU through
`core.pic.mlx_backend`. MLX is looked up as an installed package, or else in `$RING_MLX_SITE` (default
`~/.local/share/lw-gpu-spike/lib/pythonX.Y/site-packages`), exactly as in the copies. The studies' GPU interpreter is
`~/compute-data/venvs/csr_mlx/bin/python`. GPU results are not bit-comparable with CPU results (float32 kernels), and the test suite
does not run them: the `mlx` marker is reserved and never selected by default. Use only one GPU job at a time; the study's queues
reserve the GPU.

The run loop still honours `/private/tmp/compute_cap` (it waits while the cap is below `$RING_REQUIRED_CAP`, default 2), as the copies did.

## Tests

```sh
cd lwi_ring
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -B -m pytest            # fast suite (the default)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -B -m pytest -m slow    # the studies' bit-identity size and the 64^2 l = 2 check
```

(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`: the LWI venv has a pytest plugin that needs `_ssl`, which its Python lacks.)

- `test_bitid.py`: for each source copy, short cases (1 turn at 4096 per bunch; slow: 2 turns at 8192, the studies' own
  `csrdrift_bitid.py` size). Each must give the digests (SHA-256 of the final particle arrays, turn records, m2diag, wallfb and
  many_turn records) recorded from that copy. Also checked: the unified `column_drift_resets` alias on dynamic electrons.
- `test_analytic.py`: solenoid area preservation (one-turn map det = 1 with `solenoid_edge_in_drift`, legacy det < 0.995), the
  zero-current identity (dynamic = rigid columns at one proton), the E × B drift rate of electrons about a displaced flat-top column
  (Omega = λ n̄ / (2 ε0 |B| β γ² c); 64² × 128 grid, 1.03), and (slow, 2^19 particles) the l = 2 response of a flat-top column (a = 1, b = f − 2).
- `test_options.py`: the canonical `Config` defaults are kept, aliases are unified, and conflicting options are rejected.

`tools/make_references.py` re-makes `tests/data/references.json`. It runs each case on a scratch snapshot of its source copy and with
lwi_ring, side by side, behind the study's compute gate (fewer than 10 busy study processes, at most 2 of its own). The digests hold
for the recorded environment: macOS arm64, the LWI `.venv` with Python 3.12, NumPy 2.4.2 and SciPy 1.17.0, single-threaded. On
another build, re-make them with the tool rather than loosening the comparison.

## Options by feature, and the copy each came from

All are inert at their defaults. The full documentation of each option is the comment above it in `driver.Config`.

| feature | options | old copy (study, report) |
|---|---|---|
| transverse damper, coherent diagnostics, trim quadrupoles | `damper_kappa`, `damper_pickup_arc`, `damper_arc`, `damper_delay_turns`, `damper_slots`, `damper_noise_m`, `damper_planes`, `diag_bins`, `error_trim_dk` | `results_local/dipole` (report_dipole_mode_mitigation.md) |
| painted injection, waterbag / painted column profile | `injection_profile="painted"`, `injection_painted_p`, `column_profile` | `results_local/stage3d` |
| drifting-centroid columns, resets, flux-fixed size | `column_centroid_drift`, `column_drift_bore_m`, `column_drift_omega_scale`, `column_drift_resets`, `column_drift_bins_per_step`, `column_drift_smooth_m`, `column_sigma_flux`, `diag_bin_centroids` | `results_local/drift/driver` (report_column_drift.md) |
| re-centring stations, feed-forward, noise, offsets, delay | `column_recentre_stations`, `_eta`, `_noise_m`, `_offset_m`, `_mode` (monitor / feedforward), `_target`, `_side_kick`, `_seed`, `_delay_steps` | `results_local/recentre/driver` (report_electron_recentring.md); stations on dynamic electrons: `results_local/recentre_dyn/driver` |
| fixed electron size (overlap drift rate) | `column_drift_omega_overlap` (with `lens_sigma_ratio`, `column_sigma_flux`) | `results_local/esize/driver` (esize's dynamic lanes ran `esize/dyn_driver` = recentre_dyn) |
| self-consistent electrons: birth rule, resets, field proxy, AB2, aperture kill, shaping stations | `dyn_birth_sigma` (follow / fixed / proton / proton_cov / paired / paired_fixed), `dyn_resets`, `dyn_b_scale`, `dyn_gc_scheme`, `dyn_aperture_kill`, `column_recentre_m2`, `_m2_target`, `_m2_kick`, `column_recentre_dipole` | `results_local/m2/driver` (report_electron_m2_and_tune_chaos.md) |
| m = 0 / m = 2 column diagnostics | `studyrun` (m2diag.jsonl), `--diag-every` | `scripts/csrm2_run.py` |
| 400 MeV self-consistent set-up | `presets.rescale_energy(cfg, 400.0)`; driver unchanged | `results_local/dyn400/driver` (byte-identical to m2), `scripts/csrdyn400_cfg.py`, `csrdyn400_run.py` |
| sub-column stations, pickup delay, strain noise; field-scaled lattices | `rem_station_pos`, `rem_delay_steps`, `rem_m2_noise`; `presets.FIELD_2X`, `FIELD_1P5X` (with `solenoid_retune`), `presets.station_positions` | `results_local/remedy2/driver`, `scripts/csrrm2_cfg.py`, `csrrem_cfg.py` (report_m2_remedies.md) |
| gun-size tracking | `rem_birth_ratio`, `rem_birth_track` | `results_local/remedy/driver` |
| early stop | `studyrun --stop-growth=X` | `scripts/csrrem_run.py`, `csrrm2_run.py` |
| painted / flat-top / hollow electrons, size, ellipse, profile-aware match; shape diagnostic | `dyn_profile`, `dyn_painted_p`, `dyn_hollow_ratio`, `dyn_size_ratio`, `dyn_ellipticity`, `dyn_ellipse_angle_deg`, `dyn_match_profile`; `studyrun --shape` | `results_local/dynprof/driver`, `scripts/csrdp_run.py` (report_dynamic_profiles.md) |
| segmented wall-electrode feedback | `wfb_enable`, `wfb_*` (segment, sectors, bore, pickup offset, bin, gains, phases, kappas, delay, bandwidth, smoothing, baseline, start turn, noise, vmax, sweep, diagnostics, seed) | `results_local/wallfb/driver` (csr-wallring, still running) |

### Unified options

| meaning | centroid (rigid) model | dynamic model | lwi_ring |
|---|---|---|---|
| fresh electrons several times per column | `column_drift_resets` | `dyn_resets` | either name works for dynamic electrons (`Config.electron_resets()`); both set and different is an error |
| station reading from a pickup J step paths upstream | `column_recentre_delay_steps` | `rem_delay_steps` | either name works for dynamic electrons (`Config.station_delay_steps()`); both set and different is an error |

In the copies, `column_drift_resets` and `column_recentre_delay_steps` were silently ignored on dynamic electrons. No study config
set them there, so every existing config behaves as before.

Kept separate, with the relation documented: `dyn_b_scale` (field of the E × B drift, dynamic) is the inverse of
`column_drift_omega_scale` (drift rate, centroid model) as a field proxy. `rem_birth_ratio` and `dyn_size_ratio` both scale the fixed
gun size, and they multiply. `rem_birth_ratio` is the factor that `rem_birth_track` follows the protons with.

## Mapping from the old copies

| old | new |
|---|---|
| `sys.path.insert(0, DRIVER_DIR); import ring_driver as rd` | `from lwi_ring import driver as rd` (with `LWI_RING_OUT` = the lane's `results_local/ring_pic`) |
| `import many_turn as mt`, `import edyn_columns as ed`, `import wallfb`, `import damper` | `from lwi_ring import many_turn as mt, edyn_columns as ed, wallfb, damper` |
| `scripts/csrm2_run.py DRV CFG NAME` (also `csrdyn400_run.py`) | `lwi-ring-run CFG NAME` |
| `scripts/csrrem_run.py`, `csrrm2_run.py ... --stop-growth=X` | `lwi-ring-run CFG NAME --stop-growth=X` |
| `scripts/csrdp_run.py`, `csrwr_run.py ...` | `lwi-ring-run CFG NAME --shape [--stop-growth=X]` |
| `scripts/csrrcd_run.py DRV CFG NAME` | `lwi-ring-run CFG NAME --no-diag` |
| `ring_driver.py --config C` (with the checks.json gate) | `lwi-ring-driver --config C [--require-acceptance DIR]` |
| `ring_driver.py --checks`, `stage2_checks.py`, `setup_checks.py` and the other check scripts | stay in `ring_driver/` (not part of the package; the package's gate is its test suite) |
| any copy's config JSON | unchanged: every copy's `Config` field exists under the same name and meaning |

Not merged (out of scope for 0.1.0): the csrk branch (`results_local/csrk`, `tol`, `csr400`, `csrr2`: lattice families with
`lattice_arc_cells`, `lattice_arc_focus`, `lattice_cell_focus`, `lattice_arc_aperture_factor`, `lattice_match_local`). It forked
from stage3d and does not contain the column-drift line, so it needs its own merge.
