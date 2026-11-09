"""Two-stage betatron collimation: thin primary scatterers, black secondaries, a ring aperture, and loss bookkeeping (Config.coll_*).

Inert unless Config.coll_enabled(). Elements are thin planes at ring positions s_m (lab z), round (the round, solenoid-coupled lattice:
a circular jaw is invariant under the Larmor rotation, so one radius collimates both planes and every azimuth). A proton is tested
where its half-drift crosses the plane (straight-line interpolation of the half-drift, exact outside the solenoids), at its radius from
the centre (the closed orbit by default, coll_centre = "orbit"; "axis" = the design axis x = y = 0).

primary   (kind "primary": material, thickness_m) A proton at r >= radius_m traverses the foil at normal incidence (thickness t):
          * nuclear inelastic interaction with probability 1 - exp(-t / lambda_I): absorbed, counted at the primary;
          * otherwise multiple Coulomb scattering, a Gaussian angle of rms theta0 per plane (Highland, PDG 34.16):
            theta0 = 13.6 MeV / (beta c p) sqrt(t / X0) (1 + 0.038 ln(t / (X0 beta^2))) (z = 1),
            and an energy loss dE = <dE/dx> t (Bethe without density or shell corrections, PDG 34.5) with Bohr straggling
            sigma_E^2 = 0.1569 MeV^2 (rho Z / A t[cm]) (1 - beta^2/2) / (1 - beta^2); the momentum (all three components) is scaled by
            1 + dp/p, dp/p = -dE / (beta^2 E). Single (Rutherford) and nuclear elastic scattering tails are not modelled.
          The physics is evaluated at coll_physics_energy_mev (None = energy_mev) and applied as dimensionless angles, dp/p and
          probabilities: with a higher energy than energy_mev, an energy-scaled model of a ring (same geometric emittance and Twiss)
          receives the angles of the real machine. The kick acts at the plane; the rest of the half-drift is corrected for the new slope.
secondary (kind "secondary") A proton at r >= radius_m is absorbed (black absorber: no out-scattering or punch-through, so the
          captured fraction is an upper bound on the real cleaning efficiency at the secondaries).
aperture  The ring wall: aperture_m (radius, from the centre as above), coll_arc_aperture_m in the arcs of a periodic lattice
          (None = aperture_m). A proton reaching it is an uncontrolled loss, recorded at the crossing point (bins of coll_loss_bin_m).

A lost proton is removed (many_turn) or parked with weight zero on the axis, with its paired electron removed (driver.run, the
dyn_aperture_kill convention; with collimation the dynamic model's own aperture kill is replaced by this one). Every loss is counted
with its population and kinetic energy at the local energy, per element (primary, each secondary, uncontrolled), per turn
(collimation.jsonl) and along the ring (collimation_summary.json).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MP_MEV = 938.27208816
ME_MEV = 0.51099895
K_BETHE = 0.307075                       # MeV cm^2 / mol

# PDG atomic and nuclear properties: Z, A (g/mol), density (g/cm^3), mean excitation I (eV), X0 (g/cm^2), lambda_I (g/cm^2).
MATERIALS = {
    "tungsten": dict(Z=74, A=183.84, rho=19.30, I_ev=727.0, X0=6.76, lam_I=191.9),
    "tantalum": dict(Z=73, A=180.948, rho=16.65, I_ev=718.0, X0=6.82, lam_I=191.6),
    "graphite": dict(Z=6, A=12.011, rho=2.21, I_ev=78.0, X0=42.70, lam_I=85.8),
    "copper": dict(Z=29, A=63.546, rho=8.96, I_ev=322.0, X0=12.86, lam_I=137.3),
    "iron": dict(Z=26, A=55.845, rho=7.874, I_ev=286.0, X0=13.84, lam_I=132.1),
}
KINDS = ("primary", "secondary")


def foil_physics(material: str, thickness_m: float, energy_mev: float) -> dict:
    """Per-pass interaction of a proton of kinetic energy energy_mev with a foil (see the module doc)."""
    m = MATERIALS[material]
    gamma = 1 + energy_mev / MP_MEV
    beta2 = 1 - gamma**-2
    beta = math.sqrt(beta2)
    pc = gamma * beta * MP_MEV
    t_cm = thickness_m * 100
    x_x0 = t_cm * m["rho"] / m["X0"]
    theta0 = 13.6 / (beta * pc) * math.sqrt(x_x0) * (1 + 0.038 * math.log(x_x0 / beta2)) if x_x0 > 0 else 0.
    ratio = ME_MEV / MP_MEV
    tmax = 2 * ME_MEV * beta2 * gamma**2 / (1 + 2 * gamma * ratio + ratio**2)
    i_mev = m["I_ev"] * 1e-6
    dedx = K_BETHE * m["Z"] / m["A"] / beta2 * (0.5 * math.log(2 * ME_MEV * beta2 * gamma**2 * tmax / i_mev**2) - beta2)  # MeV cm^2/g
    de = dedx * m["rho"] * t_cm
    straggle = math.sqrt(0.1569 * m["rho"] * m["Z"] / m["A"] * t_cm * (1 - beta2 / 2) / (1 - beta2))
    total = energy_mev + MP_MEV
    return dict(material=material, thickness_m=thickness_m, energy_mev=energy_mev, theta0_rad=theta0,
                dedx_mev_cm2_g=dedx, de_mean_mev=de, de_straggle_mev=straggle,
                dpp_mean=-de / (beta2 * total), dpp_rms=straggle / (beta2 * total),
                p_nuclear=1 - math.exp(-t_cm * m["rho"] / m["lam_I"]))


def validate(cfg) -> None:
    """Config checks of the coll_* options (called from Config.validate when coll_enabled)."""
    if cfg.aperture_m is None:
        raise ValueError("collimation needs aperture_m (the ring wall)")
    if cfg.coll_arc_aperture_m is not None and (not cfg.racetrack() or not cfg.coll_arc_aperture_m > 0):
        raise ValueError("coll_arc_aperture_m needs a periodic lattice and a positive radius")
    if cfg.coll_centre not in ("orbit", "axis"):
        raise ValueError("coll_centre must be orbit or axis")
    if not cfg.coll_loss_bin_m > 0 or cfg.coll_physics_energy_mev is not None and not cfg.coll_physics_energy_mev > 0:
        raise ValueError("coll_loss_bin_m and coll_physics_energy_mev must be positive")
    names = set()
    wall = max(cfg.aperture_m, cfg.coll_arc_aperture_m or 0.)
    for e in cfg.coll_elements or []:
        if set(e) - {"name", "s_m", "kind", "radius_m", "material", "thickness_m"}:
            raise ValueError(f"collimator {e}: unknown keys")
        if e.get("kind") not in KINDS or not 0 <= e.get("s_m", -1) < cfg.circumference_m or not 0 < e.get("radius_m", 0) < wall:
            raise ValueError(f"collimator {e}: kind primary/secondary, 0 <= s_m < circumference, 0 < radius_m < the ring aperture")
        if e["kind"] == "primary" and (e.get("material") not in MATERIALS or not e.get("thickness_m", 0) > 0):
            raise ValueError(f"primary {e}: material in {sorted(MATERIALS)} and a positive thickness_m")
        if not e.get("name") or e["name"] in names or e["name"] == "uncontrolled":
            raise ValueError("collimators need distinct names (not 'uncontrolled')")
        names.add(e["name"])
    if cfg.collective and cfg.half_extent() < wall:
        raise ValueError("field half-extent must cover the ring aperture; enlarge half_xy_m (keep the cell size)")


class Collimation:
    def __init__(self, cfg, p: dict, orbit=None):
        self.cfg, self.p = cfg, p
        self.C = cfg.circumference_m
        self.orbit = orbit if cfg.coll_centre == "orbit" else None
        self.rng = np.random.default_rng([cfg.seed, cfg.coll_seed, 7177])
        energy = cfg.coll_physics_energy_mev or cfg.energy_mev
        self.elements = sorted((dict(e) for e in cfg.coll_elements or []), key=lambda e: e["s_m"])
        for e in self.elements:
            if e["kind"] == "primary":
                e["physics"] = foil_physics(e["material"], e["thickness_m"], energy)
        if cfg.racetrack():
            self.cell = cfg.circumference_m / cfg.lattice_periods
            self.ls = cfg.straight_length_m if cfg.straight_length_m is not None else self.cell - cfg.arc_length_m
        nb = int(math.ceil(self.C / cfg.coll_loss_bin_m))
        self.bins = np.zeros((nb, 2))                       # uncontrolled: population, kinetic energy (J)
        names = [e["name"] for e in self.elements] + ["uncontrolled"]
        self.tally = {n: dict(population=0., energy_j=0., macro=0) for n in names}
        self.tally_turn = {n: dict(population=0., energy_j=0.) for n in names}
        self.primary_passes = {e["name"]: dict(scattered=0, nuclear=0) for e in self.elements if e["kind"] == "primary"}
        self.initial_population = None
        self.events: list[dict] = []

    def set_ramp(self, gamma: float, tol: float = 0.002) -> None:
        """Ramp (ramp_coll_physics = follow): primary-foil physics at the present kinetic energy, rebuilt when it moved by tol (relative).
        coll_physics_energy_mev, when set, scales with the kinetic energy relative to injection."""
        e_now = (gamma - 1) * MP_MEV
        if abs(e_now - getattr(self, "e_built", self.cfg.energy_mev)) < tol * e_now:
            return
        self.e_built = e_now
        energy = (self.cfg.coll_physics_energy_mev * e_now / self.cfg.energy_mev) if self.cfg.coll_physics_energy_mev else e_now
        self.physics_energy_mev = energy
        for e in self.elements:
            if e["kind"] == "primary":
                e["physics"] = foil_physics(e["material"], e["thickness_m"], energy)

    # ---- geometry ------------------------------------------------------------------------------------------------------------------
    def aperture_at(self, z: np.ndarray) -> np.ndarray:
        a = np.full(len(z), float(self.cfg.aperture_m))
        if self.cfg.coll_arc_aperture_m is not None:
            a[(z % self.C) % self.cell >= self.ls] = self.cfg.coll_arc_aperture_m
        return a

    def centre(self, z: np.ndarray) -> np.ndarray:
        if self.orbit is None:
            return np.zeros((len(z), 2))
        co = self.orbit.at(np.asarray(z) % self.C)
        return co[:, [0, 2]]

    # ---- one half-drift of one proton species ------------------------------------------------------------------------------------
    def process(self, s, before: np.ndarray, step: int, phase: str, partners=()) -> np.ndarray:
        """Test the half-drift before -> s.position_m (z not yet wrapped) against every element and the wall; scatter at primaries.
        Returns the mask of protons lost in this half-drift (the caller removes them, or they are parked here when partners is not
        None: weight zero on the axis, paired electrons removed)."""
        if self.initial_population is None:
            self.initial_population = {}
        self.initial_population.setdefault(s.name, float(s.population.sum()))
        n = len(s.population)
        lost = np.zeros(n, bool)
        alive = s.population > 0
        z0 = before[:, 2]
        dz = s.position_m[:, 2] - z0
        moving = alive & (dz > 0)
        turn = (step - (1.0 if phase == "midpoint" else 0.5)) / self.cfg.steps_per_turn
        for e in self.elements:
            frac = ((e["s_m"] - z0) % self.C) / np.where(moving, dz, 1.)
            idx = np.flatnonzero(moving & ~lost & (frac > 0) & (frac <= 1))
            if not len(idx):
                continue
            f = frac[idx, None]
            xy = before[idx, :2] + f * (s.position_m[idx, :2] - before[idx, :2]) - self.centre(np.full(len(idx), e["s_m"]))
            hit = idx[np.hypot(xy[:, 0], xy[:, 1]) >= e["radius_m"]]
            if not len(hit):
                continue
            if e["kind"] == "secondary":
                self._lose(s, hit, e["name"], e["s_m"], turn, lost)
                continue
            ph = e["physics"]
            nuclear = self.rng.random(len(hit)) < ph["p_nuclear"]
            self.primary_passes[e["name"]]["nuclear"] += int(nuclear.sum())
            self._lose(s, hit[nuclear], e["name"], e["s_m"], turn, lost)
            sc = hit[~nuclear]
            self.primary_passes[e["name"]]["scattered"] += len(sc)
            if len(sc):
                theta = self.rng.standard_normal((len(sc), 2)) * ph["theta0_rad"]
                dpp = ph["dpp_mean"] + ph["dpp_rms"] * self.rng.standard_normal(len(sc))
                u = s.momentum_mc[sc]
                uz = np.abs(u[:, 2])
                u = u * (1 + dpp)[:, None]
                u[:, :2] += theta * (uz * (1 + dpp))[:, None]
                s.momentum_mc[sc] = u
                rest = (1 - frac[sc]) * dz[sc]
                s.position_m[sc, :2] += theta * rest[:, None]
                if hasattr(s, "delta"):
                    s.delta[sc] = s.delta[sc] + dpp
        # the ring wall (uncontrolled)
        z1 = s.position_m[:, 2]
        cand = np.flatnonzero(alive & ~lost)
        rel1 = s.position_m[cand, :2] - self.centre(z1[cand])
        a1 = self.aperture_at(z1[cand])
        out = cand[np.sum(rel1**2, axis=1) >= a1**2]
        if len(out):
            # crossing point on the half-drift (as many_turn.absorb), against the wall at the end point
            start = before[out, :2] - self.centre(z0[out])
            d = s.position_m[out, :2] - self.centre(z1[out]) - start
            a = np.sum(d**2, axis=1)
            b = 2 * np.sum(start * d, axis=1)
            c = np.sum(start**2, axis=1) - self.aperture_at(z1[out])**2
            disc = np.sqrt(np.maximum(b * b - 4 * a * c, 0))
            fr = np.where((a > 0) & (c < 0), np.clip((-b + disc) / (2 * np.where(a > 0, a, 1)), 0, 1), 0.)
            zl = (z0[out] + fr * dz[out]) % self.C
            for k, i in enumerate(out):
                self._lose(s, np.array([i]), "uncontrolled", float(zl[k]), turn, lost)
        if partners is not None and lost.any():
            idx = np.flatnonzero(lost)
            s.population[idx] = 0.
            s.position_m[idx, :2] = 0.
            s.momentum_mc[idx, :2] = 0.
            for e in partners:
                e.alive[idx] = False
                e.population[idx] = 0.
        return lost

    def _lose(self, s, idx: np.ndarray, name: str, where: float, turn: float, lost: np.ndarray) -> None:
        if not len(idx):
            return
        from .driver import C
        u = s.momentum_mc[idx]
        pop = s.population[idx]
        ek = pop * (np.sqrt(1 + np.sum(u**2, axis=1)) - 1) * s.mass_kg * C * C
        lost[idx] = True
        for key in (self.tally[name], self.tally_turn[name]):
            key["population"] += float(pop.sum())
            key["energy_j"] += float(ek.sum())
        self.tally[name]["macro"] += len(idx)
        if name == "uncontrolled":
            b = min(int(where // self.cfg.coll_loss_bin_m), len(self.bins) - 1)
            self.bins[b] += (float(pop.sum()), float(ek.sum()))
        self.events.append(dict(element=name, s_m=float(where), turn=float(turn), bunch=s.name, macro=int(len(idx)),
                                population=float(pop.sum()), kinetic_energy_j=float(ek.sum())))

    # ---- records -------------------------------------------------------------------------------------------------------------------
    def describe(self) -> dict:
        return dict(elements=[{k: v for k, v in e.items()} for e in self.elements], aperture_m=self.cfg.aperture_m,
                    arc_aperture_m=self.cfg.coll_arc_aperture_m, centre=self.cfg.coll_centre,
                    physics_energy_mev=getattr(self, "physics_energy_mev", self.cfg.coll_physics_energy_mev or self.cfg.energy_mev))

    def end_turn(self, turn: int, directory: Path) -> dict:
        total = sum((self.initial_population or {}).values()) or 1.
        row = dict(turn=turn, initial_population=total,
                   per_turn={n: dict(v, fraction=v["population"] / total) for n, v in self.tally_turn.items()},
                   cumulative={n: dict(v, fraction=v["population"] / total) for n, v in self.tally.items()},
                   primary_passes=self.primary_passes)
        with (directory / "collimation.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            for ev in self.events:
                stream.write(json.dumps(dict(event=ev)) + "\n")
        self.events = []
        self.tally_turn = {n: dict(population=0., energy_j=0.) for n in self.tally_turn}
        controlled = sum(v["population"] for n, v in self.tally.items() if n != "uncontrolled")
        lost = controlled + self.tally["uncontrolled"]["population"]
        summary = dict(turn=turn, describe=self.describe(), initial_population=total, cumulative=row["cumulative"],
                       controlled_fraction=controlled / total, uncontrolled_fraction=self.tally["uncontrolled"]["population"] / total,
                       captured_fraction=controlled / lost if lost else None, primary_passes=self.primary_passes,
                       uncontrolled_bins=dict(bin_m=self.cfg.coll_loss_bin_m,
                                              nonzero={str(i): self.bins[i].tolist() for i in np.flatnonzero(self.bins[:, 0])}))
        tmp = directory / "collimation_summary.json.tmp"
        tmp.write_text(json.dumps(summary, indent=1))
        tmp.replace(directory / "collimation_summary.json")
        return summary
