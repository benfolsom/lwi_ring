"""Zero-current linear model of the driver's ring: transfer matrices, tunes and the closed orbit.

The driver's lattice is smooth-focusing (one wave number per plane, or two for a periodic
solenoid lattice with arcs and straights) plus thin kicks: lattice-error gradients and dipoles and,
for solenoids, hard-edge fringe kicks. Between kicks the motion is linear, so the one-turn map is an
affine 4-D map built here from matrix exponentials, independently of the PIC push. It serves three
purposes: (1) the zero-current tunes against which the driver's tracked map is checked; (2) the closed
orbit, which the electron columns follow; (3) a reference for the mode mixing of solenoid lattices.

State (x, x', y, y') with slopes x' = dx/dz. Bulk (wave numbers kx, ky, solenoid K = sign q B / 2p):
    x'' = -kx^2 x + 2 K y',   y'' = -ky^2 y - 2 K x'.
Solenoid edges (hard edge): entry dx' = +K y, dy' = -K x; exit the opposite. Lattice errors, as in
LatticeErrors: dx' = -g x + theta_x, dy' = +g y + theta_y at z_j = (j + 1/2) C/M.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm

E_CHARGE = 1.602176634e-19


def bulk_matrix(kx2: float, ky2: float, k_sol: float, length: float) -> np.ndarray:
    a = np.array([[0, 1, 0, 0], [-kx2, 0, 0, 2 * k_sol], [0, 0, 0, 1], [0, -2 * k_sol, -ky2, 0]], float)
    return expm(a * length)


def affine(m4: np.ndarray, d4: np.ndarray | None = None) -> np.ndarray:
    m = np.eye(5)
    m[:4, :4] = m4
    if d4 is not None:
        m[:4, 4] = d4
    return m


class LinearLattice:
    """Sections, thin kicks and cumulative maps of the driver's zero-current ring."""

    def __init__(self, cfg, p: dict, track=None, errors=None):
        from .driver import MP, C   # deferred: ring_driver imports this module's users
        self.cfg, self.length = cfg, cfg.circumference_m
        tunes = cfg.plane_tunes()
        # sections: (z0, z1, kx2, ky2, K)
        self.sections: list[tuple] = []
        # thin elements at a position: (z, 5x5 matrix)
        self.thin: list[tuple] = []
        brho = p["gamma"] * p["beta"] * MP * C / E_CHARGE           # B rho (T m)
        if track is None:
            k = 2 * math.pi * tunes / self.length
            self.sections.append((0., self.length, k[0]**2, k[1]**2, 0.))
        else:
            kk = cfg.solenoid_bz_t / (2 * brho)
            for j, start in enumerate(track.starts):
                ksol = track.sign[j] * kk
                a, b = start, start + track.ls
                self.sections.append((a, b, track.k_straight[0]**2, track.k_straight[1]**2, ksol))
                self.sections.append((b, start + track.ls + track.la, track.k_arc[0]**2, track.k_arc[1]**2, 0.))
                if kk:
                    self.thin.append((a, self._edge(ksol)))
                    self.thin.append((b, self._edge(-ksol)))
        self.has_errors = errors is not None
        if errors is not None:
            q_ref = float(np.mean(tunes))
            for j in range(errors.count):
                gain = errors.dk[j] * (q_ref * 2 * math.pi / self.length)**2 * errors.spacing
                kick = np.eye(5)
                kick[1, 0], kick[3, 2] = -gain, gain
                kick[1, 4], kick[3, 4] = errors.theta[j, 0], errors.theta[j, 1]
                self.thin.append((errors.location[j], kick))
        self.thin.sort(key=lambda item: item[0])

    @staticmethod
    def _edge(k: float) -> np.ndarray:
        m = np.eye(5)
        m[1, 2], m[3, 0] = k, -k
        return m

    def _section_at(self, z: float) -> tuple:
        for sec in self.sections:
            if sec[0] <= z < sec[1]:
                return sec
        return self.sections[-1]

    def propagate(self, nodes: np.ndarray | None = None) -> tuple[np.ndarray, list]:
        """One-turn 5x5 map from z = 0 and the cumulative maps at the requested nodes (sorted, in [0, C))."""
        nodes = np.array([]) if nodes is None else np.asarray(nodes)
        marks = {0.0, self.length}
        for sec in self.sections:
            marks.update((sec[0], sec[1]))
        marks.update(z for z, _ in self.thin)
        marks.update(nodes.tolist())
        marks = np.array(sorted(marks))
        m = np.eye(5)
        cumulative = {}
        thin = list(self.thin)
        node_set = set(nodes.tolist())
        for a, b in zip(marks[:-1], marks[1:]):
            # elements located at a act before the bulk of [a, b)
            for z, kick in thin:
                if z == a:
                    m = kick @ m
            if a in node_set:
                cumulative[a] = m.copy()
            if b - a > 0:
                sec = self._section_at(0.5 * (a + b))
                m = affine(bulk_matrix(sec[2], sec[3], sec[4], b - a)) @ m
        for z, kick in thin:
            if z == self.length:
                m = kick @ m
        return m, [cumulative.get(z) for z in nodes.tolist()]

    def one_turn(self) -> tuple[np.ndarray, np.ndarray]:
        m, _ = self.propagate()
        return m[:4, :4], m[:4, 4]

    def tunes(self) -> np.ndarray:
        """Fractional zero-current tunes (one of each conjugate pair, ascending)."""
        m4, _ = self.one_turn()
        ev = np.linalg.eigvals(m4)
        return np.sort(np.abs(np.angle(ev)) / (2 * math.pi))[::2]

    def closed_orbit(self, points: int = 4096) -> "ClosedOrbit":
        """Closed orbit (x, x', y, y') on `points` uniform nodes of [0, C)."""
        m, _ = self.propagate()
        m4, d4 = m[:4, :4], m[:4, 4]
        if np.max(np.abs(np.linalg.eigvals(m4))) > 1 + 1e-9 or abs(np.linalg.det(np.eye(4) - m4)) < 1e-10:
            raise ValueError("no closed orbit: the zero-current lattice is unstable or on an integer resonance")
        x0 = np.linalg.solve(np.eye(4) - m4, d4)
        nodes = np.arange(points) * self.length / points
        _, cum = self.propagate(nodes)
        table = np.array([(c @ np.r_[x0, 1.0])[:4] for c in cum])
        return ClosedOrbit(self.length, nodes, table)


class ClosedOrbit:
    """Tabulated closed orbit with periodic linear interpolation in the lab position z."""

    def __init__(self, length: float, nodes: np.ndarray, table: np.ndarray):
        self.length, self.nodes, self.table = length, nodes, table

    def at(self, z: np.ndarray) -> np.ndarray:
        """(n, 4) closed orbit (x, x', y, y') at positions z."""
        zz = np.asarray(z) % self.length
        ext_z = np.r_[self.nodes, self.length]
        ext = np.vstack((self.table, self.table[:1]))
        return np.column_stack([np.interp(zz, ext_z, ext[:, k]) for k in range(4)])

    def xy(self, z: np.ndarray) -> np.ndarray:
        return self.at(z)[:, [0, 2]]

    def slopes(self, z: np.ndarray) -> np.ndarray:
        return self.at(z)[:, [1, 3]]

    def describe(self) -> dict:
        return dict(rms_xy_m=np.sqrt(np.mean(self.table[:, [0, 2]]**2, axis=0)).tolist(),
                    max_abs_xy_m=np.max(np.abs(self.table[:, [0, 2]]), axis=0).tolist(), points=len(self.nodes))
