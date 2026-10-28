"""Injection distributions matched to an aperture, and a simple multi-turn painting model.

The driver loads each bunch from a "quiet start": 32 mirror copies of a scrambled Sobol set of
standard normals in (x, y, z, px, py), with exact zero mean and unit rms. The transformations here
act on the four transverse columns (x, px, y, py) as one 4-D vector w, so the result is a function of
the 4-D normalised amplitude |w| only. In a linear lattice such a distribution is stationary (any
function of the two Courant-Snyder invariants is), so truncating it does not create a mismatch by
itself:

* ``truncated_gaussian``: radial CDF of the 4-D Gaussian (chi-square with 4 degrees of freedom,
  F(t) = 1 - exp(-t/2)(1 + t/2)) cut at |w|, then re-scaled to unit rms per column. The cut is solved so
  that the largest 4-D amplitude equals ``injection_cut`` (units of sigma_m) or ``injection_edge_m``
  (metres). In a smooth-focusing ring the radius of every particle then stays below that value for all
  time; in a periodic lattice multiply by sqrt(beta(s)/<beta>).
* ``waterbag``: uniform in the 4-D ball (radius u^(1/4) for uniform u = F(|w|^2)), re-scaled to unit
  rms; the transverse density is parabolic and the hard edge is sqrt(6) sigma_m.

Painting: the particle set is split into K batches (whole mirror groups, so each batch stays
symmetric). Batch k is a small beam of rms ``injection_paint_sigma_ratio`` (truncated at 3) offset in
normalised (x, px) by A_k (cos phi, sin phi) and in (y, py) by A_k (cos(phi+90deg), sin(phi+90deg)),
with A_k = A_max sqrt((k + 1/2)/K) in units of sigma_m, so that the batches have equal area and the
lattice's betatron rotation fills a uniform disc of radius A_max in each plane (rms size
sqrt(A_max^2/4 + ratio^2) sigma_m; the 4-D amplitude of every particle is below sqrt(2) A_max + 3 ratio
in sigma_m units, so choose A_max from the aperture). Batches carry equal charge. Unborn batches carry
zero weight; at the start of turn k their transverse state is reset to the stored birth state and their
weight is switched on (``apply_births``).
"""
from __future__ import annotations

import numpy as np

CHI4_GRID = np.linspace(0.0, 400.0, 400001)


def chi4_cdf(t: np.ndarray) -> np.ndarray:
    return -np.expm1(-t / 2) * 1.0 - (t / 2) * np.exp(-t / 2)


def chi4_inverse(u: np.ndarray) -> np.ndarray:
    return np.interp(u, chi4_cdf(CHI4_GRID), CHI4_GRID)


COLUMNS = [0, 3, 1, 4]       # x, px, y, py inside the quiet-start array


def _renormalise(w: np.ndarray) -> np.ndarray:
    w = w - w.mean(axis=0)
    return w / np.sqrt(np.mean(w**2, axis=0))


def transform(a: np.ndarray, profile: str, cut: float) -> np.ndarray:
    """Copy of the quiet-start array with the transverse columns reshaped (rms 1 each)."""
    if profile == "gaussian":
        return a
    w = a[:, COLUMNS]
    rho2 = np.sum(w**2, axis=1)
    u = chi4_cdf(rho2)
    if profile == "truncated_gaussian":
        new_rho = np.sqrt(chi4_inverse(u * float(chi4_cdf(np.array(cut**2)))))
    elif profile == "painted":
        from scipy.special import betaincinv       # cut is the exponent p here: rho^2 ~ Beta(2, 1 + p) in units of the edge
        new_rho = np.sqrt(betaincinv(2.0, 1.0 + cut, np.clip(u, 1e-12, 1 - 1e-12)))
    else:
        new_rho = np.sqrt(np.sqrt(u)) * 1.0           # radius u^(1/4) in the unit ball
    shaped = _renormalise(w * (new_rho / np.sqrt(rho2))[:, None])
    out = a.copy()
    out[:, COLUMNS] = shaped
    return out


def amplitude_bound(b: np.ndarray) -> float:
    """Largest 4-D normalised amplitude |(x, px, y, py)| of a shaped sample, in units of its rms size.

    In a smooth-focusing ring x^2 + (beta x')^2 and the y analogue are conserved, so the radius
    sqrt(x^2 + y^2) of any particle never exceeds sigma_m times this number."""
    return float(np.sqrt(np.max(np.sum(b[:, COLUMNS]**2, axis=1))))


def cut_for_amplitude(a: np.ndarray, target: float) -> float:
    """Truncation of the underlying Gaussian (before re-normalisation) that gives amplitude bound `target` (in rms units)."""
    lo, hi = 1.0, 12.0
    def bound(cut: float) -> float:
        return amplitude_bound(transform(a, "truncated_gaussian", cut))
    if bound(hi) < target or bound(lo) > target:
        raise ValueError("the requested amplitude cut is outside what a truncated Gaussian can reach for this sample")
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if bound(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def shaped(a: np.ndarray, cfg) -> np.ndarray:
    """Quiet-start array reshaped by the configured injection profile (the same object when gaussian)."""
    if cfg.injection_profile == "gaussian":
        return a
    if cfg.injection_profile == "truncated_gaussian":
        # injection_cut / injection_edge_m are the largest 4-D amplitude in units of the final rms size / metres
        target = cfg.injection_cut if cfg.injection_edge_m is None else cfg.injection_edge_m / cfg.sigma_m
        return transform(a, "truncated_gaussian", cut_for_amplitude(a, target))
    if cfg.injection_profile == "painted":
        return transform(a, "painted", cfg.injection_painted_p)
    return transform(a, cfg.injection_profile, 0.0)


def paint(a: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """(array with every batch placed at its birth state, batch index per particle)."""
    k_total = cfg.injection_paint_turns
    n = len(a)
    nbase = n // 32
    batch = (np.arange(n) % nbase) % k_total
    small = transform(a, "truncated_gaussian", 3.0)
    out = a.copy()
    amp = cfg.injection_paint_amax_sigma * np.sqrt((batch + 0.5) / k_total)
    phi = np.radians(cfg.injection_paint_phase_deg)
    ratio = cfg.injection_paint_sigma_ratio
    out[:, 0] = ratio * small[:, 0] + amp * np.cos(phi)
    out[:, 3] = ratio * small[:, 3] + amp * np.sin(phi)
    out[:, 1] = ratio * small[:, 1] + amp * np.cos(phi + np.pi / 2)
    out[:, 4] = ratio * small[:, 4] + amp * np.sin(phi + np.pi / 2)
    return out, batch


def attach_painting(s, batch: np.ndarray) -> None:
    """Remember each particle's birth state and weight on the species; unborn batches get zero weight."""
    s.paint_batch = batch
    counts = np.bincount(batch)
    # equal charge per batch, whatever the batch sizes: w_i = N_total / (K * n_batch)
    s.paint_weight = s.population.sum() / len(counts) / counts[batch]
    s.paint_state = np.column_stack((s.position_m[:, :2], s.momentum_mc[:, :2]))
    s.population = np.where(batch == 0, s.paint_weight, 0.0)


def apply_births(beams: list, ids: list, turn: int) -> int:
    """Switch on batch `turn` at the start of that turn (call at each turn boundary); returns macroparticles born."""
    born = 0
    for s, idx in zip(beams, ids):
        if not hasattr(s, "paint_batch"):
            continue
        mask = s.paint_batch[idx] == turn
        if mask.any():
            s.position_m[mask, :2] = s.paint_state[idx[mask], :2]
            s.momentum_mc[mask, :2] = s.paint_state[idx[mask], 2:]
            s.population[mask] = s.paint_weight[idx[mask]]
            born += int(mask.sum())
    return born


def injected_population(beams: list, meta: list, role: str, turn: int) -> float:
    """Population switched on by the start of `turn` (batches 0..turn) for a role; the plain total without painting."""
    total = 0.0
    for s, info in zip(beams, meta):
        if info["role"] != role:
            continue
        if hasattr(s, "paint_batch"):
            total += float(s.paint_weight[s.paint_batch <= turn].sum())
        else:
            total += float(s.population.sum())
    return total
