# lwi_ring

A native strong–strong particle-in-cell ring driver for space-charge-compensated proton rings, with co-moving magnetised electron
columns (rigid, drifting-centroid or self-consistent), wall-electrode feedback, recycled electrons and energy ramps. It is built on
the PIC core of the [LW Integrator (LWI)](https://github.com/benfolsom/LW_integrator) and is developed alongside a feasibility study
of a compensated rapid-cycling synchrotron; results and reports are published separately as they mature. Licence: GPL-3.0.

Every feature is an explicit option of the `Config` dataclass in `lwi_ring.driver`. All options are inert at their defaults, and the
defaults reproduce the baseline ring driver bit for bit. The full description of each option is the comment above it in `Config`.

## Install

The package needs `core.pic` (backend selection, deposit, gather, Green function, Boris push, `Species`) from the LW Integrator, which
is not on a package index. Install it from a checkout first, then this package:

```sh
git clone https://github.com/benfolsom/LW_integrator
git clone https://github.com/benfolsom/lwi_ring
pip install -e LW_integrator
pip install -e "lwi_ring[test]"
```

Python 3.10 or later; NumPy, SciPy and Numba are installed as dependencies. MLX (`pip install -e "lwi_ring[mlx]"`) is optional and
only needed for the GPU backend.

## Quick start

Run one case from a JSON file of `Config` fields. One thread per process is the usual setting:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export LWI_RING_OUT=$PWD/ring_out          # output root, see below

# plain driver: one run, one record per turn
lwi-ring-driver --config case.json --output out/case [--set key=JSON] [--pilot-steps N] [--refine dt]

# the same run with the m = 0 / m = 2 column diagnostics, early stop and profile-shape diagnostic
lwi-ring-run case.json NAME [key=JSON ...] [--diag-every=4] [--stop-growth=1.0] [--shape] [--no-diag]

# many turns with an absorbing aperture (rigid columns), with checkpoints and resume
lwi-ring-many-turn --config case.json --output $LWI_RING_OUT/NAME [--aperture-mm 170]
```

Without installation, the same commands are `python -m lwi_ring.driver`, `python -m lwi_ring.studyrun` and
`python -m lwi_ring.many_turn`. From Python:

```python
import json
from pathlib import Path
from lwi_ring import driver

cfg = driver.Config(**json.load(open("case.json")))
driver.run(cfg, Path("out/case"))
# or: studyrun.run_case(cfg, out, every=4, shape=True, stop_growth=1.0)
```

Solver locks, the Numba cache, default run directories and the many-turn output check live below the output root: `$LWI_RING_OUT` if
set, otherwise `./results_local/ring_pic` in the working directory. `driver.set_output_root(path)` changes it at run time.

The GPU backend (`backend="mlx"` in the config, or `--backend mlx`) runs the kernels in float32 through `core.pic.mlx_backend`. MLX is
taken from the installed package, or else from `$RING_MLX_SITE`. GPU results are not bit-comparable with CPU results, and the test
suite does not run them (the `mlx` marker is reserved and never selected by default).

## Options by physics feature

The tables list the `Config` fields; the module column shows where the feature is implemented.

### Electron columns: rigid, drifting-centroid and self-consistent

| model | selected by | options |
|---|---|---|
| rigid column | `column_sections` ("straights", "ring" or a list of `[start, end]` in metres) | `column_local_fraction`, `column_follow_orbit`, `column_sigma_follow`, `column_profile` ("gaussian", "waterbag", "painted"), `lens_fraction`, `lens_sigma_ratio` |
| drifting centroid (rigid profile, E × B centroid motion) | `column_centroid_drift` | `column_drift_bore_m`, `column_drift_omega_scale`, `column_drift_omega_overlap`, `column_drift_resets`, `column_drift_bins_per_step`, `column_drift_smooth_m`, `column_sigma_flux`, `diag_bin_centroids` |
| self-consistent (macroparticle electrons in the PIC fields; `edyn_columns`) | `column_electrons="dynamic"`, `column_b_t` | `dyn_birth_sigma` (follow, fixed, proton, proton_cov, paired, paired_fixed), `dyn_resets`, `dyn_b_scale`, `dyn_gc_scheme` (euler, ab2), `dyn_aperture_kill`, `rem_birth_ratio`, `rem_birth_track` |
| electron birth profile (dynamic) | `dyn_profile` (gaussian, painted, flattop, hollow) | `dyn_painted_p`, `dyn_hollow_ratio`, `dyn_size_ratio`, `dyn_ellipticity`, `dyn_ellipse_angle_deg`, `dyn_match_profile`; `lwi-ring-run --shape` adds a profile-shape diagnostic |

### Resets, re-centring and shaping stations

| feature | options |
|---|---|
| fresh electrons several times per column | `column_drift_resets` (drifting centroid) or `dyn_resets` (dynamic); either name works for dynamic electrons, and setting both to different values is an error |
| re-centring stations (drifting-centroid or dynamic electrons) | `column_recentre_stations`, `column_recentre_eta`, `column_recentre_noise_m`, `column_recentre_offset_m`, `column_recentre_mode` (monitor, feedforward), `column_recentre_target` (orbit, proton), `column_recentre_side_kick`, `column_recentre_seed` |
| pickup delay of the stations | `column_recentre_delay_steps` or `rem_delay_steps`; either name works for dynamic electrons, and setting both to different values is an error |
| station positions inside the columns | `rem_station_pos` (fractions of the column length) |
| m = 2 shaping stations (dynamic electrons) | `column_recentre_m2`, `column_recentre_m2_target` (birth, proton), `column_recentre_m2_kick`, `column_recentre_dipole`, `rem_m2_noise` |

`dyn_b_scale` (field of the E × B drift, dynamic model) is the inverse of `column_drift_omega_scale` (drift rate, drifting-centroid
model) as a field proxy. `rem_birth_ratio` and `dyn_size_ratio` both scale the fixed gun size, and they multiply.

### Wall-electrode feedback

Segmented wall electrodes acting on dynamic electrons on the straights of a periodic lattice (`lwi_ring.wallfb`). Enabled by
`wfb_enable`; the options are `wfb_segment_m`, `wfb_sectors`, `wfb_bore_m`, `wfb_pickup_offset`, `wfb_bin_m`, `wfb_gain1/2`,
`wfb_phase1/2_deg`, `wfb_kappa1/2`, `wfb_delay_ns`, `wfb_bw_mhz`, `wfb_smooth_m`, `wfb_baseline_turns`, `wfb_start_turn`,
`wfb_noise1_m`, `wfb_noise2`, `wfb_vmax_v`, `wfb_sweep`, `wfb_proton_kick`, `wfb_electron_drive` and `wfb_seed`. Per-turn records are
written to `wallfb.jsonl`.

### Recycled electrons

Electrons leaving column k pass a proton-free transport and re-enter column k + `recycle_stride`; fresh gun electrons appear only at
the start of a stream. Enabled by `recycle_stride > 0`. The transport acts through `recycle_gun_every`, `recycle_rotate_deg`,
`recycle_centroid_phase_deg`, `recycle_centroid_eta`, `recycle_strain_eta`, `recycle_current_loss`, `recycle_scrape_sigma`,
`recycle_window_m`, `recycle_noise_m` and `recycle_seed`.

### Ramps

Enabled by `ramp_final_energy_mev` (the reference energy follows a programme over `ramp_time_s`; implemented in `lwi_ring.ramp`).
`ramp_shape` is "sine", "dual" (second magnet harmonic, `ramp_dual_alpha`) or "table" (`ramp_energy_table`). Further options:
`rf_programme` (constant, min_phase, table), `rf_voltage_table`, `ramp_comp_programme` (constant, taper, table) with
`ramp_comp_table`, `ramp_comp_n_unc0`, `ramp_comp_n_design`, `ramp_comp_scaling`, `ramp_comp_f_max`, `ramp_comp_ring_average`;
`ramp_column_sigma` (fixed, adiabatic, measured), `ramp_solenoid_follow`, `ramp_electron_beta_error`, `ramp_quad_scale` and
`ramp_regrid_tol`. Dynamic electrons follow the ramp. The helper `presets.rescale_energy(cfg, energy_mev)` moves a case to another
kinetic energy (solenoid field times the rigidity ratio, population times β²γ³), and `presets.FIELD_2X`, `FIELD_1P5X` and
`presets.station_positions` are the field-scaled lattices and station placement used in the feasibility study.

### Dampers, painting and lattice errors

| feature | options |
|---|---|
| transverse damper (slice by slice) | `damper_kappa`, `damper_pickup_arc`, `damper_arc`, `damper_delay_turns`, `damper_slots`, `damper_noise_m`, `damper_planes`; coherent turn diagnostics with `diag_bins` (`lwi_ring.damper`) |
| painted injection | `injection_profile` (gaussian, truncated_gaussian, waterbag, painted), `injection_painted_p`, `injection_cut`, `injection_edge_m`, `injection_paint_turns`, `injection_paint_amax_sigma`, `injection_paint_sigma_ratio`, `injection_paint_phase_deg` |
| lattice errors and trim quadrupoles | `error_superperiods`, `error_count`, `error_dk_rms`, `error_dipole_rms_rad`, `error_seed`, `error_trim_dk` |
| lattice, longitudinal dynamics, Landau-damping detuning | `straight_length_m`, `arc_length_m`, `lattice_periods`, `solenoid_*`; `gamma_t`, `rf_*`; `proton_a_*`, `proton_xi_*`, `proton_delta_rms` |

## Outputs and diagnostics

`lwi-ring-driver` writes to its output directory `turns.json` (one record per turn: emittances, tune estimates, losses and, with
`diag_bins > 0`, a `coherent` entry with the bin-centroid amplitude and decohered emittance), `summary.json`, `provenance.json` (the
full configuration and derived physics), `boundary.jsonl` and `final_particles.npz`. `lwi-ring-run` adds `m2diag.jsonl` (m = 0 and
m = 2 column diagnostics; none with `--no-diag`), `wallfb.jsonl` when the wall feedback is on, and stops early after the first turn
whose 4-D emittance exceeds (1 + `--stop-growth`) times its initial value. `lwi-ring-many-turn` writes `turns.jsonl`, `losses.jsonl`,
`tracked_initial.npz`, checkpoints with `latest_checkpoint.json`, and `summary.json`; it resumes from the latest checkpoint.

## Tests

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest            # fast suite (the default)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest -m slow    # larger bit-identity cases and the 64^2 l = 2 check
```

(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids third-party pytest plugins that may fail to load in a minimal environment.)

- `test_bitid.py`: the bit-identity cases run short configurations (1 turn at 4096 particles per bunch; slow: 2 turns at 8192) and
  compare SHA-256 digests of the final particle arrays and the turn, diagnostic, feedback and many-turn records against stored
  reference digests in `tests/data/references.json`. The references were produced by the earlier, separate ring-driver versions that
  this package consolidates. They hold for the recorded environment (macOS arm64, Python 3.12, NumPy 2.4.2, SciPy 1.17.0,
  single-threaded). On another build, regenerate them with `tools/make_references.py` rather than loosening the comparison; that
  tool needs the original driver versions, which are not part of this repository.
- `test_analytic.py`: solenoid area preservation (one-turn map determinant 1 with `solenoid_edge_in_drift`), the zero-current
  identity (dynamic equals rigid columns at one proton), the E × B drift rate of electrons about a displaced flat-top column
  (64² × 128 grid), and, slow, the l = 2 response of a flat-top column.
- `test_options.py`: the baseline `Config` defaults are kept, aliases are unified, and conflicting options are rejected.
- `test_recycle.py`, `test_ramp_csr.py`: the recycled-electron and ramp options.

## Dependency on the LW Integrator

`lwi_ring` uses `core.pic` from the [LW Integrator](https://github.com/benfolsom/LW_integrator) as a library and does not modify it.
Version 0.12.3 or later is expected (`pip install -e "lwi_ring[lwi]"` requests it if it is available to pip).

## Licence

GPL-3.0; see `LICENSE`.
