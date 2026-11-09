# Running lwi_ring on an NVIDIA GPU (CuPy)

The CUDA backend moves the PIC kernels (deposit, FFT field solve, gather, Boris push) to the GPU. Everything else (collimation,
wall feedback, dynamic electron columns, lattice, diagnostics) stays in NumPy/Numba on one CPU core, so the CPU speed still matters
(see "Expected speed-up" below).

## 1. Install (Ubuntu 22.04/24.04, no GUI needed)

```bash
nvidia-smi                                   # driver installed; note the "CUDA Version" (the driver's maximum, e.g. 12.4)
git clone --branch v0.12.4 https://github.com/benfolsom/LW_integrator
git clone --branch v0.3.0  https://github.com/benfolsom/lwi_ring
python3 -m venv ~/lwi-venv && . ~/lwi-venv/bin/activate      # Python 3.10, 3.11 or 3.12
pip install -U pip
pip install -r lwi_ring/requirements_gpu.txt                 # numpy, scipy, numba, cupy-cuda12x[ctk]
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), cupy.cuda.runtime.runtimeGetVersion())"
```

* `cupy-cuda12x` needs a driver that supports CUDA 12 (driver 525 or later). For an older driver (CUDA 11) replace it with
  `cupy-cuda11x[ctk]`; for CUDA 13 use `cupy-cuda13x[ctk]`. Install only one CuPy wheel: the backend refuses to start if two are present
  or if the wheel does not match the detected toolkit.
* The `[ctk]` extra installs the CUDA runtime, cuFFT and NVRTC libraries from pip, so only the NVIDIA driver is needed. The kernels are
  compiled at run time by NVRTC; a first-call delay of a few seconds is normal.

Then, in every shell that runs jobs:

```bash
. ~/lwi-venv/bin/activate
export PYTHONPATH=$PWD/lwi_ring/src:$PWD/LW_integrator     # the two clones (run from the directory that holds them)
export LWI_RING_OUT=$HOME/ring_out                          # outputs and the Numba cache go here (set before python starts)
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p $LWI_RING_OUT
```

## 2. Validate first (a few minutes)

```bash
cd lwi_ring
python validate_gpu.py --backend cupy --precision float32
```

It runs a short case on the CPU in float64 and on the GPU in float32, compares emittance, losses, ellipticity and centroid against
tolerances measured on Apple-GPU float32 runs, and prints a PASS/FAIL table with timings. Send back the printed table.

## 3. Running a job

A job is a JSON file of `Config` fields. For the GPU it must contain `"backend": "cupy"` and `"precision": "float32"` (`precision` is
accepted only with `cupy`; use `"float64"` for double precision, which is slow on GeForce/RTX cards). Example:
`examples/test_job_gpu.json`.

```bash
cd lwi_ring
nohup python -m lwi_ring.studyrun examples/test_job_gpu.json test_job_gpu --shape --stop-growth=0.5 > test_job_gpu.log 2>&1 &
```

Output goes to `$LWI_RING_OUT/<NAME>/` (an existing directory of that name is replaced). `--stop-growth=0.5` ends the run if the 4-D
emittance grows by 50%. Progress is in `$LWI_RING_OUT/<NAME>/progress.json` and in the log. `key=value` overrides after the name
are accepted, e.g. `turns=5`. Run one job at a time per GPU. A 24 GB card is ample (the largest configurations need a few GB).

### Longer job: leading configuration at 8x resolution

`examples/leading_8x_q838_s1024.json` is the leading 400 MeV configuration at 8 times the standard macroparticles (524 288 per bunch),
tunes 8.38 and 1 024 steps per turn, for 100 turns. It checks whether the low emittance growth and the small halo hold at higher
resolution. Run it like the test job, under its own name:

```bash
nohup python -m lwi_ring.studyrun examples/leading_8x_q838_s1024.json leading_8x --shape --stop-growth=0.5 > leading_8x.log 2>&1 &
```

After about an hour, send the last lines of the log (they give seconds per step and the remaining-time estimate). When it finishes,
send the files listed in section 4. It can be stopped and restarted from the beginning without harm.

### Energy-ramp job: leading configuration, 400 to 1500 MeV

`examples/ramp_leading_1x_q825_s1024.json` ramps the leading configuration (recycled streams, wall strips, column ends, collimation) from
400 to 1500 MeV in about 500 turns (520 turns in all, the last ones at constant energy), with the compensation fraction tapered to hold the
residual space-charge tune shift, the column size shrinking adiabatically, an RF voltage that rises as needed (minimum-phase programme)
and the strip gains and foil physics following the energy. It runs like the other jobs:

```bash
nohup python -m lwi_ring.studyrun examples/ramp_leading_1x_q825_s1024.json ramp_leading_1x --shape --stop-growth=0.5 > ramp_leading_1x.log 2>&1 &
```

The log's seconds per step give the run time (the job has 532 480 steps). `turns.json` carries a `ramp` entry per turn (energy, tunes,
compensation, electron energy) and `electrons.recycle.transport_lag_steps` (constant if the recycling timing holds).

## 4. Files to send back (from `$LWI_RING_OUT/<NAME>/`)

| File | Send |
|---|---|
| `m2diag.jsonl`, `turns.json`, `summary.json`, `provenance.json`, `progress.json`, `stop.json` (if present) | yes |
| `wallfb.jsonl`, `collimation_summary.json`, `collimation.jsonl` | yes |
| the log (`<NAME>.log`) and the output of `nvidia-smi` | yes |
| `boundary.jsonl` | only if asked (large; gzip it) |
| `final_particles.npz` and other `*.npz` | no |

## 5. Expected speed-up

On an Apple M5 GPU (MLX backend) only about 5% of a step of the largest configurations is GPU kernel time; the rest is single-threaded
CPU work (collimation, wall feedback, dynamic electron columns). A faster GPU therefore changes the run time by a few per cent at most,
and a faster CPU core helps more. Several jobs can run side by side on separate CPU cores (with `backend` `"cpu"`), one per core.
