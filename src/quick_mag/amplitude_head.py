"""Per-site amplitude head: how strongly an ion polarizes a ligand, from the ion's atomic descriptors.

There is no per-element amplitude in the kernel. Each bridge end's sigma and pi polarizations are scaled by

    a_m = exp(sum_f theta_m[f] * phi_f(ion)),      m = sigma, pi

with ``phi`` the centred features below: the ion's own descriptors (charge, d filling, spin, d-shell compactness,
charge-transfer energy) and two of its ligand cage (:func:`site_environment`), both continuous in the geometry:
``ln_cn`` = ln(CN / 6) with CN a smooth count of its ligands, and ``ln_nsig`` = ln(N_sigma / 3) with N_sigma the
number of ligands an average sigma-bonding d orbital bonds to (below). A count cannot tell a square plane from a
tetrahedron (CN 4 both); N_sigma can (2.5 vs 4/3). Nothing is fitted per structure or per element.
Bond lengths enter through the range factor, bond by bond. (Weighted Steinhardt invariants of the cage were tried and
removed 2026-11-17: on every magnetic site in the fit data they never moved; a binary ``tet`` label was used briefly
in place of ``ln_cn`` and replaced by it the same day.)

The weights live in ``PolarizationParameters.head_sigma`` / ``head_pi`` ({feature: theta}); a feature left out has weight 0. The overall scale is the ligand's (``w_ligand``,
``jh_ligand``), so a = 1 is the reference ion: octahedral, d5, eg^1.5, Delta = 4 eV (``ln_delta``).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["DELTA", "DELTA_FILLED", "ln_delta", "FEATURES", "ION_FEATURES", "ENV_FEATURES", "ion_features",
           "site_environment", "sigma_load", "head_factors"]

ION_FEATURES = (
    "charge",      # oxidation state - 3
    "d_count",     # d electrons - 5
    "spin",        # S - 2
    "n_eg",        # eg (oct) / e (tet) electrons - 1.5: the sigma-active occupancy
    "ln_zeta",     # ln(Slater d exponent / 2.2): the d shell's compactness
    "ln_delta",    # ln(Delta / 4 eV), Delta the O 2p -> 3d charge-transfer energy (DELTA): covalency
)
# The ligand cage: a smooth count of the site's ligands, CN = sum_j w_j, w_j = 1 for a bond up to SWITCH[0] x R0 (R0
# the Shannon distance) falling smoothly to 0 at SWITCH[1] x R0. A count rather than an overlap weight, because bond
# lengths are already the range factor's, bond by bond (an overlap weight counts them twice: tested, it transfers
# worse). Centred on the octahedron: ln_cn = ln(CN / 6); -2 ln_cn on sigma would be bond-valence (6/CN)^2.
#
# ``ln_nsig``: the cage's sigma Gram matrix over the d shell, G_ab = sum_L w_L A_a(u_L) A_b(u_L) (A = sk_table's sigma
# angular factors, w_L the same smooth bond weights), has trace CN and its eigenvalues are the angular-overlap sigma
# loads of the five d orbitals (octahedron eg 3, 3; tetrahedron t2 4/3 x 3; square plane x2-y2 3, z2 1). Their
# load-weighted mean N_sigma = tr G^2 / tr G is how many ligands a sigma orbital's covalency is shared over; with
# sum_a A_a(u) A_a(u') = P2(u.u') it is a pair sum over the cage,
#     N_sigma = sum_LL' w_L w_L' P2(u_L . u_L')^2 / sum_L w_L = CN [(18/35) q4^2 + (2/7) q2^2 + 1/5]
# (q_l the w-weighted Steinhardt invariants), so it is rotation invariant and needs no local frame or orbital labels.
# Octahedron 3 (ln_nsig = 0), square pyramid 2.6, square plane 2.5, trigonal bipyramid 2.02, trigonal prism 1.80,
# tetrahedron 4/3. Weight -1 on sigma (the derived value: an orbital's sigma covalency shared over N_sigma ligands,
# a_sigma = 3 / N_sigma) gives 2.25 for a tetrahedron, against (6/4)^2.17 = 2.41 from the fitted ln_cn weight, and
# 1.20 for a square plane, where ln_cn gave the tetrahedral 2.41.
ENV_FEATURES = ("ln_cn", "ln_nsig")
FEATURES = ION_FEATURES + ENV_FEATURES
SWITCH = (1.25, 1.45)
NOMINAL_CN = {"oct": 6.0, "tet": 4.0}
NOMINAL_NSIG = {"oct": 3.0, "tet": 4.0 / 3.0}
NSIG_REF = 3.0                           # the octahedron's N_sigma: ln_nsig = 0

# The charge-transfer energy Delta (O 2p -> metal 3d), per ion. In a charge-transfer oxide an end's ligand
# polarization goes as t^2 / Delta, so a pair's superexchange as t^4 / (Delta_i Delta_j U): the ``ln_delta`` feature
# with weight -1 on both channels (the pi channel's Delta differs by the crystal field only). Delta falls along the
# row (MnO ~7.5 eV > NiO ~4.5) and with the charge (FeO ~6.5 > Fe2O3 ~3.3): the M2+ row trend of J and the
# 2+ / 3+ difference in one number, with no per-charge centring.
# Values: the median of the cluster-model (photoemission / XAS) Delta's compiled in the appendix of
# arXiv:2409.09176 (Negative charge transfer energy in correlated compounds), every compound listed for the ion
# (MnO; FeO, Fe3O4; CoO, Co3O4, CoFe2O4; NiO; CuO; Ti2O3, LaTiO3; V2O3, LaVO3; Cr2O3, LaCrO3; LaMnO3; Fe2O3,
# LaFeO3, Fe3O4, CoFe2O4; LaCoO3, Co3O4; LaNiO3; LaCuO3; CaMnO3; SrFeO3). Sources scatter by 1-2 eV (CoO 2.5-6.5).
# Ions with no value take the nearest Z's at the same charge (DELTA_FILLED). Oxide values: other ligands use them
# as they are (their scale is in w_ligand). Delta at or below ~1.5 eV is the negative-Delta (ligand-hole) regime,
# outside this picture: a smooth floor DELTA_FLOOR keeps those ions (Fe4+, Cu3+) from dominating a fit. An ion not
# in the table (no 3d, or Zn) gets ln_delta = 0, the reference DELTA_REF.
# (Tested 2026-11-16 in magnets python/scripts/ct_fit.py / ct_clean_fit.py: fitted free, the weights come out
# sigma -0.98 +- 0.07, pi -0.09 +- 0.15, and pinning both at -1 changes no set's error by more than 0.06 meV/site.
# An ionic Madelung + free-ion-affinity Delta did not reproduce the measured ones (leave-one-out 1.7 eV) and its
# Madelung environment differences made the fit worse.)
_DELTA_LIT = {
    ("Mn", 2): (6.5, 7.0, 9.0, 8.0, 6.5, 8.8),
    ("Fe", 2): (6.0, 6.5, 7.0, 6.5, 6.3, 7.0),
    ("Co", 2): (6.5, 2.5, 3.0, 4.1, 5.7, 5.5, 4.5, 6.5),
    ("Ni", 2): (5.5, 4.5, 4.5, 4.6, 4.0, 3.5, 4.4, 5.3, 6.2),
    ("Cu", 2): (2.2, 2.75, 1.5),
    ("Ti", 3): (6.5, 6.5, 6.0, 4.0),
    ("V", 3): (4.0, 6.0, 5.5, 4.0),
    ("Cr", 3): (5.5, 6.0, 5.2, 5.5),
    ("Mn", 3): (4.5,),
    ("Fe", 3): (3.5, 3.0, 4.5, 2.0, 3.7, 5.5, 2.5, 3.5, 3.0, 2.0),
    ("Co", 3): (2.0, 2.0, 1.5),
    ("Ni", 3): (1.0,),
    ("Cu", 3): (-1.0, -2.9),
    ("Mn", 4): (3.5,),
    ("Fe", 4): (0.0,),
}
_ROW = ("Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu")


def _delta_table():
    t = {k: float(np.median(v)) for k, v in _DELTA_LIT.items()}
    filled = {}
    for q in (2, 3, 4):
        have = [z for z in _ROW if (z, q) in t]
        for z in _ROW:
            if (z, q) not in t:
                near = min(have, key=lambda h: (abs(_ROW.index(h) - _ROW.index(z)), _ROW.index(h)))
                filled[(z, q)] = t[(near, q)]
    return {**t, **filled}, frozenset(filled)


DELTA, DELTA_FILLED = _delta_table()     # (element, charge) -> Delta (eV); the filled keys
DELTA_REF = 4.0                          # eV: ln_delta = 0
DELTA_FLOOR, DELTA_WIDTH = 1.5, 0.5      # eV: smooth floor (softplus) and its width


def ln_delta(element: str, charge: int) -> float:
    """The ``ln_delta`` feature of one ion: ln(Delta_eff / DELTA_REF), Delta_eff = the table's Delta floored
    smoothly at DELTA_FLOOR; 0 for an ion not in the table."""
    d = DELTA.get((element, int(charge)))
    if d is None:
        return 0.0
    eff = DELTA_FLOOR + DELTA_WIDTH * float(np.logaddexp(0.0, (d - DELTA_FLOOR) / DELTA_WIDTH))
    return float(np.log(eff / DELTA_REF))


def ion_features(descriptor, env: np.ndarray = None) -> np.ndarray:
    """The centred feature vector (``FEATURES`` order) of one ion descriptor; ``env`` the site's
    :func:`site_environment` (None = the nominal cage of the descriptor's coordination label: ln(4/6) tet, 0 oct)."""
    from quick_mag.radial_overlap import slater_zeta

    try:
        _n, zeta = slater_zeta(descriptor.element, int(descriptor.oxidation_state), "d")
        lnz = float(np.log(zeta / 2.2))
    except ValueError:                  # an empty d shell (d0)
        lnz = 0.0
    return np.array([
        float(descriptor.oxidation_state) - 3.0,
        float(descriptor.d_count) - 5.0,
        float(descriptor.spin) - 2.0,
        float(descriptor.n_eg) - 1.5,
        lnz,
        ln_delta(descriptor.element, descriptor.oxidation_state),
        *(np.array([np.log(NOMINAL_CN.get(descriptor.coordination, 6.0) / 6.0),
                    np.log(NOMINAL_NSIG.get(descriptor.coordination, NSIG_REF) / NSIG_REF)]) if env is None else env),
    ])


def _smoothstep_down(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def sigma_load(directions: np.ndarray, weights: np.ndarray) -> float:
    """N_sigma of a cage: ``sum_LL' w_L w_L' P2(u_L . u_L')^2 / sum_L w_L`` for ligand directions (n, 3) and bond
    weights (n,) -- the load-weighted mean eigenvalue of the cage's sigma Gram matrix (see ``ENV_FEATURES``).
    Floored at 1 (one ligand); 0 ligands give the reference NSIG_REF."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0 or w.sum() < 1e-6:
        return NSIG_REF
    u = np.asarray(directions, dtype=float)
    u = u / np.linalg.norm(u, axis=1)[:, None]
    p2 = 1.5 * (u @ u.T) ** 2 - 0.5
    return max(float(w @ (p2 * p2) @ w / w.sum()), 1.0)


def site_environment(structure, site: int, element: str, states, spin_state: str, coordination: str,
                     *, cutoff: float = 3.0) -> np.ndarray:
    """``ENV_FEATURES`` of one metal site: ln(CN / 6), CN the smooth count of its ligands, and ln(N_sigma / 3)
    (:func:`sigma_load`) over the same smooth bond weights. ``states`` are the site's (oxidation state, weight)
    pairs and ``coordination`` the Shannon reference they set the switch's R0 by; ``build_bridges`` passes "oct"
    (VI) for every site, since its R0 is that length plus the bond-valence shift of this CN. A site with no ligand in reach counts CN = 1 (the floor keeps a stray geometry from
    sending the amplitude to infinity) and N_sigma = 3."""
    from quick_mag.polarization_model import ANION_SPECIES, r0_metal_ligand_averaged

    symbols = structure.element_symbols()
    nbrs = [n for n in structure.neighbors(site, cutoff) if symbols[n.index] in ANION_SPECIES and n.nn_distance > 1e-8]
    center = structure.cartesian_coords[site]
    w, u = [], []
    for ligand in {symbols[n.index] for n in nbrs}:
        mine = [n for n in nbrs if symbols[n.index] == ligand]
        r = np.array([n.nn_distance for n in mine])
        r0 = r0_metal_ligand_averaged(element, tuple(states), spin_state, ligand, coordination)
        w.extend(_smoothstep_down(r / r0, *SWITCH))
        u.extend(np.asarray(n.coords, dtype=float) - center for n in mine)
    S = float(np.sum(w)) if w else 0.0
    nsig = sigma_load(np.array(u).reshape(-1, 3), np.array(w))
    return np.array([np.log(max(S, 1.0) / 6.0), np.log(nsig / NSIG_REF)])


def head_weights(table: Dict[str, float]) -> np.ndarray:
    unknown = set(table) - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown head feature(s) {sorted(unknown)}; known: {FEATURES}")
    return np.array([float(table.get(f, 0.0)) for f in FEATURES])


def head_factors(params, features: np.ndarray):
    """(a_sigma, a_pi) for one feature vector (F,) or many (n, F)."""
    ws, wp = head_weights(params.head_sigma), head_weights(params.head_pi)
    return np.exp(features @ ws), np.exp(features @ wp)
