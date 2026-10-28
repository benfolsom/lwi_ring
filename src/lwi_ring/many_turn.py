"""Many-turn ring tracking with absorbing aperture and immutable resume attempts."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from . import driver as _driver, injection
from .driver import (C, Config, ElectronLens, HERE, LatticeErrors, Longitudinal, RingFields, Species, backend_for,
                         core_hashes, drift_beam, initial_beams, landau_terms, omega_beta_of, physics, push_with_focusing, Racetrack, with_solenoid,
                         require_acceptance, save, snapshot, solver_lock)


@dataclass
class Settings:
    aperture_m: float = 0.030
    tracked_per_beam: int = 2048
    samples_per_turn: int = 8
    checkpoint_turns: int = 50

    def validate(self, cfg: Config) -> None:
        cfg.validate()
        if not math.isfinite(self.aperture_m) or self.aperture_m <= 0:
            raise ValueError("aperture must be finite and positive")
        if self.tracked_per_beam < 2048 or self.tracked_per_beam > cfg.harmonic * cfg.particles_per_bunch:
            raise ValueError("track at least 2048 particles per beam, within available count")
        if self.samples_per_turn < 8 or cfg.steps_per_turn % self.samples_per_turn or self.samples_per_turn / 2 <= cfg.plane_tunes().max() + .5:
            raise ValueError("use at least eight equally spaced samples, with Nyquist above bare tune + 0.5")
        if self.checkpoint_turns < 1:
            raise ValueError("checkpoint cadence must be positive")
        if cfg.column_electrons == "dynamic":
            raise ValueError("self-consistent electron columns are implemented in ring_driver.run only")
        extent = cfg.half_extent()
        if cfg.collective and extent < self.aperture_m:
            raise ValueError("field half-extent must cover the aperture; enlarge half_xy_m")


def fingerprints() -> dict:
    return dict(files={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                      for name in ("many_turn.py", "driver.py", "longitudinal.py", "ramp.py")}, core=core_hashes())


def append(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".tmp").open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    path.with_suffix(".tmp").replace(path)


def select_tracked(beams: list[Species], meta: list[dict], cfg: Config, settings: Settings) -> dict:
    """Equal quantile sampling of initial 4D amplitude within every bunch."""
    rng = np.random.default_rng(cfg.seed + 902)
    tracked = {}
    for role in ("proton", "hminus"):
        parts = [(s, info) for s, info in zip(beams, meta) if info["role"] == role]
        if not parts:
            continue
        beta_m = cfg.circumference_m / (2 * np.pi * cfg.plane_tunes(role))
        ids, amplitude, local = [], [], []
        for index, (s, info) in enumerate(parts):
            slopes = s.momentum_mc[:, :2] / abs(s.momentum_mc[:, 2, None])
            a = np.sqrt(np.sum(s.position_m[:, :2]**2 + (beta_m * slopes)**2, axis=1)) / cfg.sigma_m
            count = settings.tracked_per_beam // len(parts) + (index < settings.tracked_per_beam % len(parts))
            order = np.lexsort((rng.random(len(a)), a))
            chosen = order[((np.arange(count) + 0.5) * len(a) / count).astype(int)]
            local.append(chosen)
            ids.append(info["bunch"] * cfg.particles_per_bunch + chosen)
            amplitude.append(a[chosen])
        tracked[role] = dict(id=np.concatenate(ids), amplitude_sigma=np.concatenate(amplitude),
                             by_bunch=local)
    return tracked


def absorb(s: Species, ids: np.ndarray, before: np.ndarray, aperture: float,
           step: int, phase: str, cfg: Config, losses: Path) -> np.ndarray:
    """Intersect the straight half-drift with the circular material boundary."""
    lost = np.sum(s.position_m[:, :2]**2, axis=1) >= aperture**2
    for index in np.flatnonzero(lost):
        start = before[index]
        delta = s.position_m[index] - start
        a = float(delta[:2] @ delta[:2])
        b = float(2 * start[:2] @ delta[:2])
        c = float(start[:2] @ start[:2] - aperture**2)
        fraction = np.clip((-b + math.sqrt(max(0, b*b - 4*a*c))) / (2*a), 0, 1) if a and c < 0 else 0.
        position = start + fraction * delta
        position[2] %= cfg.circumference_m
        base = step - 1 + (0 if phase == "midpoint" else 0.5)
        loss_turn = (base + 0.5 * fraction) / cfg.steps_per_turn if step else 0.
        append(losses, dict(bunch=s.name, particle_id=int(ids[index]), step=step,
            turn=loss_turn, phase=phase, position_m=position.tolist(), population=float(s.population[index]),
            kinetic_energy_j=float(s.population[index] * (math.sqrt(1 + s.momentum_mc[index] @ s.momentum_mc[index]) - 1) * s.mass_kg * C*C)))
    keep = ~lost
    s.position_m = s.position_m[keep]
    s.momentum_mc = s.momentum_mc[keep]
    s.population = s.population[keep]
    s.group_index = s.group_index[keep]
    if hasattr(s, "delta"):
        s.delta = s.delta[keep]
    return ids[keep]


def copy_prefix(source: Path, destination: Path, turn: int, journal_bytes: dict) -> None:
    for name in ("turns.jsonl", "losses.jsonl"):
        # Never parse uncommitted bytes. A killed process can leave its last
        # JSON line or UTF-8 character truncated beyond this durable boundary.
        with (source / name).open("rb") as stream:
            prefix = stream.read(journal_bytes[name])
        if len(prefix) != journal_bytes[name] or (prefix and not prefix.endswith(b"\n")):
            raise ValueError("checkpoint journal prefix is missing or truncated")
        for line in prefix.splitlines():
            json.loads(line)
        (destination / name).write_bytes(prefix)
    shutil.copy2(source / "tracked_initial.npz", destination / "tracked_initial.npz")
    for frame in (source / "tracked").glob("turn_*.npz"):
        if int(frame.stem.split("_")[1]) <= turn:
            # Same machine-local filesystem: immutable frames can be shared.
            os.link(frame, destination / "tracked" / frame.name)


def run(cfg: Config, directory: Path, settings: Settings = Settings(), resume: Path | None = None,
        stop_after_turn: int | None = None) -> dict:
    settings.validate(cfg)
    directory = directory.resolve()
    if not directory.is_relative_to(_driver.OUT.resolve()):
        raise ValueError(f"output must be below the output root {_driver.OUT} (LWI_RING_OUT)")
    state = None
    if resume:
        with np.load(resume, allow_pickle=False) as loaded:
            state = {key: loaded[key] for key in loaded.files}
        header = json.loads(str(state["header"]))
        if header["fingerprints"] != fingerprints() or header["config"] != json.loads(json.dumps(asdict(cfg))) or header["stage2"] != asdict(settings):
            raise ValueError("resume requires identical code, kernels, config, and diagnostics")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "tracked").mkdir()
    (directory / "losses.jsonl").touch()
    started = time.monotonic()
    beams, meta, p = initial_beams(cfg)
    ids = [np.arange(len(s.position_m)) for s in beams]
    tracked = select_tracked(beams, meta, cfg, settings)
    original_pop = {role: sum(s.population.sum() for s, info in zip(beams, meta) if info["role"] == role) for role in tracked}
    start_turn = 0
    carried_elapsed = 0.
    if state is not None:
        start_turn = header["turn"]
        carried_elapsed = header["elapsed_s"]
        if start_turn > cfg.turns:
            raise ValueError("checkpoint beyond final turn")
        copy_prefix(resume.parent.parent, directory, start_turn, header["journal_bytes"])
        if cfg.injection_paint_turns > 1:
            original_pop = {role: injection.injected_population(beams, meta, role, start_turn) for role in tracked}
        for i, s in enumerate(beams):
            for attr in ("position_m", "momentum_mc", "population", "group_index", "delta"):
                if s.name + "_" + attr in state:
                    setattr(s, attr, state[s.name + "_" + attr])
            ids[i] = state[s.name + "_id"]
    else:
        atomic_npz(directory / "tracked_initial.npz", **{role + "_" + key: value
            for role, item in tracked.items() for key, value in item.items() if key != "by_bunch"})
    solver = RingFields(cfg) if cfg.collective else None
    backend = solver.backend if solver else backend_for(cfg.backend)
    lens = ElectronLens(cfg, p) if cfg.lens_enabled() else None
    errors = LatticeErrors(cfg, p) if cfg.errors_enabled() else None
    track = Racetrack(cfg, p) if cfg.racetrack() else None
    landau = {role: landau_terms(cfg, p, dict(role=role)) for role in ("proton", "hminus")}
    lon = Longitudinal(cfg, p) if cfg.long_enabled() else None
    orbit_ref, damper = None, None
    if cfg.damper_kappa is not None or cfg.diag_bins:
        from .lattice_linear import LinearLattice
        orbit_ref = LinearLattice(cfg, p, track, errors).closed_orbit()
    if cfg.damper_kappa is not None:
        from .damper import Damper
        damper = Damper(cfg, orbit_ref)
        if state is not None:
            raise ValueError("resuming a damper run from a checkpoint is not implemented (the damper history is not saved)")
    if lens is not None and lens.drift is not None and state is not None:
        raise ValueError("resuming a column-drift run from a checkpoint is not implemented (the electron centroids are not saved)")
    if lon and state is not None and "longitudinal" in header:
        lon.restore(header["longitudinal"])
    ramp = None
    if cfg.ramp_enabled():
        if state is not None:
            raise ValueError("resuming a ramp run from a checkpoint is not implemented (ramp state is not saved)")
        from .ramp import Ramp
        ramp = Ramp(cfg, p, lon, track, lens, solver, [landau["proton"]])
    initialization_s = time.monotonic() - started
    provenance = dict(config=asdict(cfg), stage2=asdict(settings), physics=p,
        fingerprints=fingerprints(), backend=backend.name, precision=backend.dtype,
        resume_checkpoint=str(resume) if resume else None, command=sys.argv,
        radiation_reaction="off: inherited ring diagnostic model",
        **(dict(lattice_errors=errors.describe()) if errors else {}),
        **(dict(racetrack=p.get("racetrack_match") or track.describe()) if track else {}),
        **(dict(longitudinal=lon.describe()) if lon else {}),
        sampling="initial 4D amplitude quantiles, equally allocated per bunch; unweighted stratified maps",
        accounting="source N*q, observer q, inertia m, diagnostics N; surviving energy omits wall heat, fields, radiation, and external work")
    save(directory / "provenance.json", provenance)

    def record(turn: int, extra: dict | None = None) -> dict:
        row = dict(turn=turn, step=turn * cfg.steps_per_turn, lab_time_s=ramp.t if ramp else turn * p["period_s"],
                   elapsed_s=carried_elapsed + time.monotonic() - started, beams={})
        row.update(extra or {})
        if cfg.diag_bins:
            from .damper import coherent_row
            row["coherent"] = coherent_row(beams, meta, orbit_ref, cfg.diag_bins, cfg)
        if lens is not None and lens.drift is not None:
            row["column_drift"] = lens.drift.describe()
        if ramp:
            live_protons = [s for s, info in zip(beams, meta) if info["role"] == "proton" and len(s.population)]
            row["ramp"] = ramp.diagnose(lens.ramp_diagnostics(live_protons, ramp.beta) if lens else None, live_protons)
        for role in tracked:
            group = [s for s, info in zip(beams, meta) if info["role"] == role and len(s.population)]
            if group:
                combined = Species(role, np.concatenate([s.position_m for s in group]),
                    np.concatenate([s.momentum_mc for s in group]), group[0].charge_c,
                    group[0].mass_kg, np.concatenate([s.population for s in group]))
                entry = snapshot(combined, cfg)
            else:
                entry = dict(population=0., geometric_emittance_m_rad=[None, None], normalized_emittance_m=[None, None])
            if lon and role == "proton" and group:
                row["longitudinal"] = lon.diagnose(group)
            entry["initial_population"] = float(original_pop[role])
            entry["loss_fraction"] = float(1 - entry["population"] / original_pop[role])
            row["beams"][role] = entry
        append(directory / "turns.jsonl", row)
        save(directory / "progress.json", row)
        return row

    def checkpoint(turn: int) -> None:
        journal_bytes = {}
        for name in ("turns.jsonl", "losses.jsonl"):
            with (directory / name).open("rb") as stream:
                os.fsync(stream.fileno())
                journal_bytes[name] = os.fstat(stream.fileno()).st_size
        header = dict(config=asdict(cfg), stage2=asdict(settings), fingerprints=fingerprints(),
                      turn=turn, elapsed_s=carried_elapsed + time.monotonic() - started,
                      journal_bytes=journal_bytes, **(dict(longitudinal=lon.state()) if lon else {}))
        arrays = {s.name + "_" + attr: getattr(s, attr) for s in beams
                  for attr in ("position_m", "momentum_mc", "population", "group_index", "delta")
                  if hasattr(s, attr)}
        arrays.update({s.name + "_id": identity for s, identity in zip(beams, ids)})
        path = directory / "checkpoints" / f"turn_{turn:06d}.npz"
        atomic_npz(path, header=np.asarray(json.dumps(header)), **arrays)
        save(directory / "latest_checkpoint.json", dict(path=str(path), turn=turn))

    if state is None:
        for i, s in enumerate(beams):
            ids[i] = absorb(s, ids[i], s.position_m.copy(), settings.aperture_m, 0, "initial", cfg, directory / "losses.jsonl")
        record(0)
        checkpoint(0)
    dt = p["period_s"] / cfg.steps_per_turn
    cadence = cfg.steps_per_turn // settings.samples_per_turn
    frames = {role: [] for role in tracked}
    last_progress = time.monotonic()
    total = min(stop_after_turn or cfg.turns, cfg.turns)
    try:
        for step in range(start_turn * cfg.steps_per_turn + 1, total * cfg.steps_per_turn + 1):
            if cfg.wall_limit_s and time.monotonic() - started > cfg.wall_limit_s:
                raise TimeoutError("bounded tracking wall limit exceeded")
            if step % 64 == 1:
                cap = Path("/private/tmp/compute_cap")
                while cap.exists() and int(cap.read_text()) < int(os.environ.get("RING_REQUIRED_CAP", "2")):
                    if cfg.wall_limit_s:
                        raise RuntimeError("compute cap does not permit bounded check")
                    print("Waiting for compute_cap headroom", flush=True)
                    time.sleep(10)
            landau_step = landau
            if ramp:
                dt = ramp.begin_step([s for s, info in zip(beams, meta) if info["role"] == "proton"])
                t_mid = ramp.t_mid()
                landau_step = {role: ramp.landau_terms(terms) for role, terms in landau.items()}
            else:
                t_mid = (step - 0.5) * dt
            for phase in ("midpoint", "end"):
                for i, s in enumerate(beams):
                    before = s.position_m.copy()
                    drift_beam(s, meta[i]["role"], 0.5 * dt, lon, track)
                    if not np.isfinite(s.position_m).all():
                        raise ValueError("nonfinite position")
                    ids[i] = absorb(s, ids[i], before, settings.aperture_m, step, phase, cfg, directory / "losses.jsonl")
                    s.position_m[:, 2] %= cfg.circumference_m
                if lon:
                    lon.half_ref(dt)
                if phase == "midpoint":
                    active = [i for i, s in enumerate(beams) if len(s.population)]
                    live, info = [beams[i] for i in active], [meta[i] for i in active]
                    if ramp and solver and lon:
                        solver.z_shift = lon.shift(t_mid)
                    electric, magnetic, _ = solver.fields(live, info, t_mid) if solver and live else (
                        [np.zeros_like(s.position_m) for s in live], [np.zeros_like(s.position_m) for s in live], {})
                    lens_du = lens.kicks(live, info, dt) if lens and live else [None] * len(live)
                    error_du = errors.kicks(live, info, dt) if errors and live else [None] * len(live)
                    damp_du = damper.kicks(live, info, dt, step) if damper and live and phase == "midpoint" else [None] * len(live)
                    if lon:
                        if solver:
                            solver.z_shift = lon.shift(t_mid)
                        for s, meta_i in zip(live, info):
                            if meta_i["role"] == "proton":
                                lon.refresh(s)
                    for s, e, b, meta_i, du, edu, ddu in zip(live, electric, magnetic, info, lens_du, error_du, damp_du):
                        sol = track.solenoid(s, dt) if track else None
                        s.momentum_mc = push_with_focusing(s, e, with_solenoid(b, sol), dt,
                                                         track.omega(s, dt) if track else (
                                                             ramp.omega(omega_beta_of(p, meta_i)) if ramp else omega_beta_of(p, meta_i)),
                                                         backend, landau_step[meta_i["role"]])
                        if sol is not None:
                            s.momentum_mc[:, :2] += sol[1]
                            s.position_m[:, :2] += sol[2]
                        if du is not None:
                            s.momentum_mc[:, :2] += du
                        if edu is not None:
                            s.momentum_mc[:, :2] += edu[0]
                            s.position_m[:, :2] += edu[1]
                        if ddu is not None:
                            s.momentum_mc[:, :2] += ddu[0]
                            s.position_m[:, :2] += ddu[1]
                        if lon and meta_i["role"] == "proton":
                            lon.kick(s, dt, s.momentum_mc)
                        if not np.isfinite(s.momentum_mc).all():
                            raise ValueError("nonfinite momentum")
                    if lon:
                        lon.update_ref(dt)
            if ramp:
                ramp.end_step()
            if damper:
                damper.end_step(step)
            if step % cadence == 0:
                for role, item in tracked.items():
                    frame = np.full((len(item["id"]), 4), np.nan)
                    for i, (s, info) in enumerate(zip(beams, meta)):
                        if info["role"] != role:
                            continue
                        global_ids = info["bunch"] * cfg.particles_per_bunch + ids[i]
                        common, target_indices, source_indices = np.intersect1d(item["id"], global_ids, return_indices=True)
                        u = s.momentum_mc[source_indices]
                        frame[target_indices] = np.column_stack((s.position_m[source_indices, 0],
                            u[:, 0] / abs(u[:, 2]), s.position_m[source_indices, 1], u[:, 1] / abs(u[:, 2])))
                    frames[role].append(frame)
            if step % cfg.steps_per_turn == 0:
                turn = step // cfg.steps_per_turn
                atomic_npz(directory / "tracked" / f"turn_{turn:06d}.npz", **{
                    role: np.asarray(value) for role, value in frames.items()})
                frames = {role: [] for role in tracked}
                k = cfg.hminus_refresh_turns
                if k and turn % k == 0 and turn < total:
                    # The row holds the pre-refresh state; the fresh beam starts next turn.
                    generation = turn // k
                    seed_offset = 7919 * generation
                    record(turn, dict(hminus_refresh=dict(generation=generation,
                        seed=cfg.seed + seed_offset, applied_after_this_record=True)))
                    fresh, fresh_meta, _ = initial_beams(cfg, ("hminus",), seed_offset)
                    for i, (s, info) in enumerate(zip(beams, meta)):
                        if info["role"] == "hminus":
                            replacement = fresh[info["bunch"]]
                            replacement.name = s.name
                            beams[i], meta[i] = replacement, fresh_meta[info["bunch"]]
                            ids[i] = np.arange(len(replacement.position_m))
                else:
                    record(turn)
                if cfg.injection_paint_turns > 1 and turn < total:
                    injection.apply_births(beams, ids, turn)
                    original_pop = {role: injection.injected_population(beams, meta, role, turn) for role in tracked}
                if turn % settings.checkpoint_turns == 0 or turn == total:
                    checkpoint(turn)
                print(f"turn {turn}/{cfg.turns}; elapsed {carried_elapsed + time.monotonic() - started:.1f} s", flush=True)
            if time.monotonic() - last_progress >= 60:
                save(directory / "step_progress.json", dict(step=step, turn=step / cfg.steps_per_turn,
                    target_turns=cfg.turns, elapsed_s=carried_elapsed + time.monotonic() - started))
                print(f"step {step}; turn {step / cfg.steps_per_turn:.3f}", flush=True)
                last_progress = time.monotonic()
    except Exception as error:
        save(directory / "abort.json", dict(reason=str(error), step=step,
            boundary=solver.last_boundary if solver else {}, resume="use the last complete checkpoint in a NEW output directory"))
        raise
    elapsed = carried_elapsed + time.monotonic() - started
    summary = dict(status="complete" if total == cfg.turns else "checkpoint_stop", completed_turns=total,
        target_turns=cfg.turns, elapsed_s=elapsed, initialization_this_attempt_s=initialization_s,
        measured_s_per_turn=((time.monotonic() - started - initialization_s) / (total - start_turn)
                             if total > start_turn else carried_elapsed / max(1, total)),
        final_lab_time_s=ramp.t if ramp else total * p["period_s"])
    summary["projected_1000_turn_hours"] = summary["measured_s_per_turn"] * 1000 / 3600
    if damper:
        summary["damper"] = dict(damper.describe(), **damper.stats())
    save(directory / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """lwi-ring-many-turn: the copies' many_turn.py command line (acceptance gates optional, see driver.require_acceptance)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--aperture-mm", type=float, help="overrides Config.aperture_m; default 30")
    parser.add_argument("--tracked-per-beam", type=int, default=2048)
    parser.add_argument("--samples-per-turn", type=int, help="default: 8, doubled until Nyquist > max bare tune + 0.5")
    parser.add_argument("--solver-slots", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--require-acceptance", type=Path, metavar="DIR", help="apply the copies' checks.json gate from DIR")
    args = parser.parse_args(argv)
    cfg = Config(**json.loads(args.config.read_text()))
    with solver_lock(args.solver_slots):
        if args.require_acceptance:
            require_acceptance(args.require_acceptance)
        aperture = args.aperture_mm / 1000 if args.aperture_mm is not None else (cfg.aperture_m or 0.030)
        samples = args.samples_per_turn
        if samples is None:
            samples = 8
            while samples / 2 <= cfg.plane_tunes().max() + .5:
                samples *= 2
        run(cfg, args.output, Settings(aperture_m=aperture,
            tracked_per_beam=args.tracked_per_beam, samples_per_turn=samples), args.resume)


if __name__ == "__main__":
    main()
