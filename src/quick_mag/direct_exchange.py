"""Direct d-d exchange between metal sites closer than a cutoff (edge- and face-sharing polyhedra).

Anderson kinetic exchange through the metal-metal overlap itself, no ligand in between:

    J_dd(i, j) = k_dd * G_ij,      G_ij = sum_images sum_ab m_ia m_jb S_ab(r)^2      (J > 0 AFM, unit spins)

- ``m_ia``: the net unpaired spin of orbital a (``polarization_model.occupancy_vector``: Cr3+ t2g 1, eg 0; Fe3+ all
  1), so only half-filled-to-half-filled orbital pairs count (4 t^2 / U, with k_dd absorbing t/S and 1/U).
- ``S_ab``: normalized Slater-type d-d overlaps (``radial_overlap.sto_overlap``, Slater-rule 3d exponents of each
  ion -- the same exponents the p-d range law uses), Slater-Koster combined: each site's five orbitals, in its own
  local d frame (as the bridges use), are projected on the pair's bond frame, ``t_ab = sum_k c_ak c_bk S_|m_k|(r)``.
- ``k_dd``: the one parameter (``PolarizationParameters.k_dd``). Distance and element dependence are the overlaps';
  the cutoff (``dd_cutoff``, default 3.6 A) only bounds the sum -- the overlaps are negligible beyond it.

G is a property of the structure and assignment, so ``exchange_model`` builds it once and the fit (or any parameter
change) only rescales it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from quick_mag.radial_overlap import BOHR_ANG, slater_zeta, sto_overlap
from quick_mag.sk_table import D_ORBITAL_TENSORS, D_ORBITALS

#: |m| about the bond axis of each real d orbital, in ``sk_table.D_ORBITALS`` order.
_M_OF = np.array([{"dxy": 2, "dxz": 1, "dyz": 1, "dz2": 0, "dx2-y2": 2}[o] for o in D_ORBITALS])
_R_GRID = np.linspace(1.8, 4.6, 141)        # Angstrom; overlaps interpolated on it


@lru_cache(maxsize=None)
def _overlap_table(ni: int, zi: float, nj: int, zj: float) -> np.ndarray:
    rb = _R_GRID / BOHR_ANG
    return np.stack([sto_overlap(ni, 2, zi, nj, 2, zj, rb, m=m, normalized=True) for m in (0, 1, 2)])


def dd_overlaps(zeta_i, zeta_j, r: float) -> np.ndarray:
    """Normalized (sigma, pi, delta) d-d overlaps at distance ``r`` (Angstrom); ``zeta_*`` = (n, zeta)."""
    if r > _R_GRID[-1]:
        return np.zeros(3)
    tab = _overlap_table(*zeta_i, *zeta_j)
    return np.array([np.interp(r, _R_GRID, tab[m]) for m in (0, 1, 2)])


def _bond_frame(u: np.ndarray) -> np.ndarray:
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return np.column_stack([e1, e2, u])


def _global_tensors(rotation: Optional[np.ndarray]) -> np.ndarray:
    if rotation is None:
        return D_ORBITAL_TENSORS
    return np.einsum("ab,nbc,dc->nad", rotation, D_ORBITAL_TENSORS, rotation)


def pair_overlap_matrix(Q_i, Q_j, zeta_i, zeta_j, vec) -> np.ndarray:
    """(5, 5) signed d-d overlaps ``S_ab`` of site i's local orbital a with site j's b (one image), ``vec`` from
    i to j (Angstrom); zero beyond the overlap grid."""
    r = float(np.linalg.norm(vec))
    F = _bond_frame(np.asarray(vec, dtype=float) / r)
    Q_k = np.einsum("ab,nbc,dc->nad", F, D_ORBITAL_TENSORS, F)
    c_i = np.einsum("aij,kij->ak", Q_i, Q_k)
    c_j = np.einsum("aij,kij->ak", Q_j, Q_k)
    S = dd_overlaps(zeta_i, zeta_j, r)[_M_OF]
    return np.einsum("ak,bk,k->ab", c_i, c_j, S)


def pair_dd(Q_i, Q_j, m_i, m_j, zeta_i, zeta_j, vec) -> float:
    """``sum_ab m_ia m_jb t_ab^2`` for one metal pair (one image), ``vec`` from i to j (Angstrom).

    ``Q_*``: the site's five d-orbital quadrupole tensors in global coordinates (its local frame applied).
    """
    r = float(np.linalg.norm(vec))
    F = _bond_frame(np.asarray(vec, dtype=float) / r)
    Q_k = np.einsum("ab,nbc,dc->nad", F, D_ORBITAL_TENSORS, F)     # bond-frame orbitals, global coords
    c_i = np.einsum("aij,kij->ak", Q_i, Q_k)
    c_j = np.einsum("aij,kij->ak", Q_j, Q_k)
    S = dd_overlaps(zeta_i, zeta_j, r)[_M_OF]
    t = np.einsum("ak,bk,k->ab", c_i, c_j, S)
    return float(m_i @ (t ** 2) @ m_j)


def dd_geometry(structure, descriptors: Dict, sites: List[int], cutoff: float = 3.6,
                anion_bond_cutoff: float = 3.0) -> np.ndarray:
    """The symmetric (n_sites, n_sites) matrix G over ``sites`` (J_dd = k_dd * G), every periodic image within
    ``cutoff`` summed. Each site uses its own descriptor's state (with ``valence="class"`` that is the state the
    assignment gave it, not the class average the bridges use)."""
    from quick_mag.polarization_model import local_d_frame, occupancy_vector

    index = {s: k for k, s in enumerate(sites)}
    n = len(sites)
    G = np.zeros((n, n))
    if n == 0:
        return G
    occ, zeta, Q = {}, {}, {}
    for s in sites:
        d = descriptors[s]
        occ[s] = occupancy_vector(d)
        zeta[s] = slater_zeta(d.element, int(d.oxidation_state), "d")
        try:
            rot = local_d_frame(structure, s, d.coordination, ligand_cutoff=anion_bond_cutoff)
        except Exception:
            rot = None
        Q[s] = _global_tensors(rot)
    for s in sites:
        if not occ[s].any():
            continue
        x = structure.cartesian_coords[s]
        for nb in structure.neighbors(s, cutoff):
            j = nb.index
            if j not in index or nb.nn_distance < 1e-6 or not occ[j].any():
                continue
            G[index[s], index[j]] += pair_dd(Q[s], Q[j], occ[s], occ[j], zeta[s], zeta[j],
                                             np.asarray(nb.coords) - x)
    return 0.5 * (G + G.T)
