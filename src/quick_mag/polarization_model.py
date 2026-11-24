"""Exchange-polarization superexchange model (docs/theory/magnetism-model.md).

Each metal i polarizes the p channels of a bridging ligand L with intensity

    mu_{i->L,p} = sum_a mbar_{i,a} * [a_sigma,i d_sigma(r) W^sigma_{a,p} + gamma_pi^2 a_pi,i d_pi(r) W^pi_{a,p}]

where ``W`` are squared Slater-Koster pd channel weights (``sk_table``) evaluated
in a bridge-adapted orthonormal p frame, ``mbar_{i,a}`` is the shell-averaged net
unpaired spin per d orbital, ``d_m(r)`` the metal-ligand range factor of each channel
(a power law, ``range_exponent``; 1 at ``R0_iL``, the sum of Shannon crystal radii),
and ``a_m,i = exp(theta_m . phi(ion i))`` the ion's amplitude from its atomic descriptors
(``amplitude_head``). There is no per-element amplitude: the overall scale of the
polarization lives in the ligand's ``w_L`` and ``J_H^L``.

The ligand's net per-channel polarization for a collinear configuration is
``P_p = sigma_i mu_{i,p} + sigma_j mu_{j,p}`` (sigma = +-1). A Pauli cost
``E_A = w_L sum_p P_p^2`` penalizes shared channels (AFM) and a ligand Hund term
``E_B = -J_H^L sum_{p != p'} P_p P_p'`` rewards orthogonal-channel polarization
(FM). The configuration-dependent part per bridge is

    J_bridge = 2 (w_L + J_H^L) (mu_i . mu_j) - 2 J_H^L (sum_p mu_{i,p}) (sum_p mu_{j,p})

That is the ``kernel = "diagonal"`` form. The default ``kernel = "coherent"`` is what fourth-order perturbation
theory of the p-d model gives: each end's polarization is the 3x3 matrix ``P = sum_a mbar_a v_a v_a^T`` over the
ligand p shell, ``v_a = sqrt(f_sigma) A_sigma(a) u - sqrt(f_pi) t_pi(a)`` (an orbital's sigma and pi amplitudes
add before squaring; V_pdpi / V_pdsigma < 0), and

    J_bridge = 2 (w_L + J_H^L) tr(P_i P_j) - 2 J_H^L tr(P_i) tr(P_j)

with no bridge frame. ``mu`` is the diagonal of ``P`` with the sigma-pi cross term dropped; the two agree for d5 and
for bonds along the local axes and differ by tens of percent for t2g^3 ions at low symmetry (corundum, spinel A-B).
With ``k_ee > 0`` the ion's empty majority-spin orbitals add the empty-empty (semicovalent) AFM term of
``bridge_J_components``.

Sign convention throughout: J > 0 is AFM, and the model energy of a collinear
configuration is ``E = +1/2 sigma^T J sigma`` with sigma in {+1, 0, -1}. Spin
magnitude lives inside mu (via ``sum_a mbar_a``); do NOT multiply by nominal
moments.

Orbital resolution: ``mbar_a`` is the per-orbital net unpaired spin of a Hund filling over
each site's point-charge crystal-field orbitals (``crystal_field_bases``, ``level_filling``):
the two levels (eg / t2g) from the k = 4 field of the ligand cage, the order inside each from the
k = 2 field of every ion within CF_FIELD_CUTOFF at formal charges, full orbitals lowest, then
half-filled, then empty, degenerate orbitals sharing their filling. No local axes. One rule gives
an eg^1 electron's orbital, an eg^3 hole's, and the t2g order of d1/d2 (and of a minority t2g
electron); on an undistorted cage it is the shell average. No site is classified as octahedral
or tetrahedral: each mixes the two level orderings (octahedral: t2g fills first; tetrahedral:
e first, see ``ion_descriptors``) with the weights of ``coordination_fractions``, from the sign
of the cage's cubic field (``cubic_field_sign``), as oxidation-state fractions are mixed; R0 is
the octahedral Shannon length plus the bond-valence shift ``BOND_VALENCE_B ln(CN / 6)``. Averages
(degenerate levels, templates, oxidation states) are exact averages over microstates, since J is
bilinear in the occupancies.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from quick_mag.ion_descriptors import (
    ANION_SPECIES,
    IonDescriptor,
    ion_descriptor,
    site_ligand_directions,
)
from quick_mag.oxidation_state_energy import min_energies_with_single_valence
from quick_mag.radii import SHANNON_IONIC_RADII
import quick_mag.sk_table as sk_table
T2G_INDICES = tuple(sk_table.D_ORBITALS.index(o) for o in ("dxy", "dxz", "dyz"))
EG_INDICES = tuple(sk_table.D_ORBITALS.index(o) for o in ("dz2", "dx2-y2"))

# Formal anion charge used for the ligand Shannon-radius lookup.
ANION_CHARGES: Dict[str, int] = {
    "O": -2, "S": -2, "Se": -2, "Te": -2, "N": -3,
    "F": -1, "Cl": -1, "Br": -1, "I": -1,
}

# IonDescriptor.coordination -> Shannon table coordination label.
_SHANNON_COORDINATION = {"oct": "VI", "tet": "IV"}

# IonDescriptor.spin_state -> Shannon table spin-state label.
_SHANNON_SPIN_LABEL = {"HS": "High Spin", "LS": "Low Spin"}

# eg splitting (e/A^3, the units of the k=2 field tensor) below which a site counts
# as undistorted, so its orbital order is undetermined and the occupied->empty FM
# channel stays off. An ideal octahedron gives exactly 0 (~1e-17 in floating point);
# a 0.02 A elongation already gives ~1e-3, so this only catches cages with no
# distortion at all -- an idealized builder seed, not a relaxed structure.
# See ``level_filling``.
_CF_DEGENERATE_TOL = 1e-9


@dataclass
class PolarizationParameters:
    """Fit parameters of the polarization model.

    ``w_ligand`` is the ligand Pauli stiffness (it carries the overall scale: the
    metal amplitudes are 1 for a reference ion); ``jh_ligand`` the ligand Hund coupling; ``gamma_pi`` a global pi/sigma
    polarization amplitude ratio (1.0 = no hierarchy).
    """

    w_ligand: Dict[str, float] = field(default_factory=dict)
    jh_ligand: Dict[str, float] = field(default_factory=dict)
    # Per-element double-exchange amplitude factor tau (Anderson-Hasegawa).
    # A DE-active bridge gets hopping amplitude tau_i * tau_j (product
    # combination rule), so same-element pairs see tau^2 and cross-element
    # resonant pairs get a combination for free. Default 0 = term off.
    t_de: Dict[str, float] = field(default_factory=dict)
    # Per-element occupied->empty FM amplitude factor (Kugel-Khomskii FM channel
    # for eg^1 Hund-active-core ions such as Mn3+; product rule j_fm_i*j_fm_j like
    # t_de). An electron hopping into a neighbour's empty eg orbital Hund-aligns
    # with that site's core spins -> FM (see docs/theory/magnetism-model.md).
    # HAND-CALIBRATED, not fit — the d5-heavy training set cannot constrain it.
    # Default 0 = channel off.
    j_fm: Dict[str, float] = field(default_factory=dict)
    # Double-exchange form. "linear" (derived): the carrier's kinetic energy is first order in the effective
    # d-d hopping through the ligand, t_eff ~ t_i t_j cos(theta) / Delta, so
    #   J_DE = -tau_i tau_j sqrt(f_sigma,i f_sigma,j) |cos theta| de_weight,
    # f_sigma each end's sigma intensity (range and amplitude head). "quadratic": the original
    # -tau_i tau_j d_i d_j cos^2(theta) de_weight with the range factors alone. A parameter file without the
    # key is read as "quadratic".
    de_form: str = "linear"
    gamma_pi: float = 1.0
    # Range: a metal's polarization of a ligand (the squared M-L transfer) falls with the M-L distance r as
    # (R0 / r)^n, 1 at the reference distance R0 = the sum of Shannon crystal radii. n is ``range_exponent``:
    # the bounded law below when ``delta_sat`` > 0, else ``range_power`` per metal + ``n_delta`` ln(Delta / 4 eV)
    # (Harrison's universal pd transfer t ~ r^-7/2 gives n = 7 for t^2).
    range_power: Dict[str, float] = field(default_factory=dict)
    range_power_default: float = 7.0
    # Power-law exponent's dependence on the ion's charge-transfer energy: n = n_metal + n_delta ln(Delta / 4 eV)
    # (``amplitude_head.ln_delta``). Ionic ions (large Delta: Mn2+) are perturbative and steep, covalent ones
    # (small Delta: Fe3+) shallow; with it one n serves every metal. 0 = off.
    n_delta: float = 0.0
    # The bounded form of that dependence, used instead of ``range_power`` / ``n_delta`` when ``delta_sat`` > 0:
    #     n = n_min + (n_max - n_min) Delta^2 / (Delta^2 + delta_sat^2)
    # The coupling is of fourth order in the p-d hopping when Delta is large (J ~ t^4) and never of lower order
    # than the first, and t ~ r^-4 (Harrison: r^-3.5), so J runs between r^-16 and r^-4: n between 8 and 2 per end.
    # Delta^2 / (Delta^2 + 4 t^2) is cos^2 of the covalent mixing angle of a p-d bond, so delta_sat plays 2 t.
    delta_sat: float = 0.0
    n_min: float = 2.0
    n_max: float = 8.0
    # Per-site amplitude head (``amplitude_head``): {feature: theta} for the sigma and pi amplitudes of an
    # end, a = exp(theta . phi(ion)). Empty = no head.
    head_sigma: Dict[str, float] = field(default_factory=dict)
    head_pi: Dict[str, float] = field(default_factory=dict)
    w_default: float = 0.03
    jh_default: float = 0.01
    t_de_default: float = 0.0
    j_fm_default: float = 0.0
    # Direct d-d exchange (``direct_exchange``): J_dd = k_dd * G over metal pairs closer than dd_cutoff
    # (Angstrom), G from Slater-type d-d overlaps. 0 = off.
    k_dd: float = 0.0
    dd_cutoff: float = 3.6
    # Bridge kernel: "coherent" = tr(P_i P_j), each orbital's sigma and pi amplitudes added before squaring
    # (rotation invariant, no bridge frame); "diagonal" = the original mu_i . mu_j with sigma and pi squared
    # separately. A parameter file without the key is read as "diagonal" (it was fitted with that kernel).
    kernel: str = "coherent"
    # Empty-empty (semicovalent) exchange. An ion's empty majority-spin orbitals polarize a ligand with the
    # opposite sign to its half-filled ones (the majority-spin ligand electron delocalizes more easily: Hund),
    # in proportion k_ee * s(ion), s the spin weight of ``ee_spin``. Two such polarizations on one ligand channel couple
    # AFM: J_EE = 2 (w_L + J_H^L) k_ee^2 s_i s_j tr(P_i^E P_j^E), reported inside J_SE. 0 = off.
    k_ee: float = 0.0
    # The spin weight s of an empty orbital's polarization: the spin asymmetry x = c p of the empty shell, c its
    # covalency (the end's amplitude already) and p = (n_maj - n_min) / (n_maj + n_min) the asymmetry fraction.
    #   "n"        s = n (unpaired electrons): the files fitted before 2026-11-20 evening
    #   "n_delta"  s = n * 4 eV / Delta: p from the empty orbital's exchange splitting n J_H over the charge-transfer
    #              energy, p ~ 2 delta / Delta (derived; PBE+U's p relative to Cr3+: Ti 0.2, V 0.63, Mn3+ 1.7-2.3,
    #              Mn4+ 1.3-1.4 against n/Delta 0.29, 0.77, 1.63, 1.57 -- magnets ee_asymmetry_check.py)
    #   "n2"       s = n^2 (empirical: PBE+U's x as a whole relative to Cr3+)
    # Default "n_delta"; a parameter file without the key is read as "n" (it was fitted with that rule).
    ee_spin: str = "n_delta"
    # Occupied -> empty t2g (Kugel-Khomskii FM for t2g^1 / t2g^2 acceptors), on top of ``oe_rule``'s channel:
    #   "off"
    #   "signed"  the cross term of the signed polarization H - k_ee E (the derivation's Pi): -2 (w + J_H) k_ee k_t2g
    #             [tr(P_i E^t_j) + tr(E^t_i P_j)], E^t the empty t2g orbitals with the same spin weight as E-E
    #   "kk"      Kugel-Khomskii: -2 (w + J_H) k_t2g [tr(P_i K_j) + tr(K_i P_j)], K the empty t2g orbitals weighted by
    #             n J_H / U of the acceptor (KK_HUND, KK_U: the reference PBE+U's U + J_H)
    # k_t2g = 1 is the derived coefficient (fixed; freeing it only tests the derivation). Default "kk"; a parameter
    # file without the key is read as "off".
    t2g_oe: str = "kk"
    k_t2g: float = 1.0
    # Occupied->empty rule. "product": -j_fm_i j_fm_j [tr(P_i A_j) + tr(A_i P_j)], a factor per element at both
    # ends (the original). "acceptor": the coefficient belongs to the ion that owns the empty orbital,
    #   J_OE = -2 (w_L + J_H^L) [tr(P_i X_j) + tr(X_i P_j)],   X = j_fm[element] A + k_oe E,
    # A the polarization of the ion's empty majority-spin orbitals and E the same weighted by its unpaired
    # electrons (the Hund splitting of the empty orbital grows with the spin it would join). A donor needs no
    # parameter: Mn2+ donates into Cr3+'s empty eg with Cr's coefficient, whatever Mn3+ does as an acceptor.
    # "ionization": donor and acceptor factors computed per ion from the free-ion ionization ladder
    # (``element_data.IONIZATION_ENERGIES``) instead of fitted per element. Moving an electron from donor i (charge
    # q_i) to acceptor j (q_j) costs U_ij = IE_{q_i + 1}(i) - IE_{q_j}(j) - oe_charge (q_i - q_j + 1), the last term
    # the screening of the charge left behind, and
    #   J_OE = -2 (w_L + J_H^L) k_oe [exp(-(U_ij - 3 eV) / oe_decay) tr(P_i A_j) + (i <-> j)].
    # Fe3+ and Ni2+ have the highest next ionization energy of their charge and barely donate; Mn2+, Mn3+ and
    # Mn4+ each get their own factors. Three numbers for every ion.
    oe_rule: str = "product"
    k_oe: float = 0.0
    oe_decay: float = 1.5
    oe_charge: float = 15.0
    # Interference of direct d-d hopping with the hopping through a shared ligand (the Anderson channel adds the
    # two amplitudes before squaring): J_x = k_dx sum_ab m_a m_b S_ab (v_ia . v_jb) per bridge of a pair inside
    # dd_cutoff, S_ab the signed d-d overlap and v the p-d amplitude vectors. Derived sign: k_dx < 0. 0 = off.
    k_dx: float = 0.0
    def get_w(self, ligand: str) -> float:
        return self.w_ligand.get(ligand, self.w_default)

    def get_jh(self, ligand: str) -> float:
        return self.jh_ligand.get(ligand, self.jh_default)

    def get_t_de(self, element: str) -> float:
        return self.t_de.get(element, self.t_de_default)

    def get_j_fm(self, element: str) -> float:
        return self.j_fm.get(element, self.j_fm_default)

    def get_range_power(self, element: str) -> float:
        return self.range_power.get(element, self.range_power_default)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def bridge_frame(u_i: np.ndarray, u_j: np.ndarray, *, tol: float = 1e-6) -> np.ndarray:
    """Bridge-adapted orthonormal p frame (rows e1, e2, e3) for one M-L-M bridge.

    e1 = u_i (metal i's bond direction), e3 normal to the bridge plane, e2
    completes the right-handed set (u_j lies in span(e1, e2)). For (near-)linear
    bridges the plane normal is undefined; fall back to a deterministic global
    axis choice — irrelevant for orbitally isotropic (d5) ions.
    """
    e1 = np.asarray(u_i, dtype=float)
    cross = np.cross(e1, np.asarray(u_j, dtype=float))
    norm = np.linalg.norm(cross)
    if norm > tol:
        e3 = cross / norm
    else:
        axis = np.zeros(3)
        axis[int(np.argmin(np.abs(e1)))] = 1.0
        e3 = axis - (axis @ e1) * e1
        e3 /= np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    return np.array([e1, e2, e3])


# Charge-balanced oxidation distributions within this energy of the best one
# (eV/atom, from the geometry-free oxidation energy model) count as accessible
# when deciding whether mixed valence is forced. Calibrated so that resonant
# hole-transfer variants (~0.1-0.2 eV/atom, e.g. Fe-mixed vs Co-mixed in
# LaSr3FeCoO12) are included while unphysical A-site charge states (La2+, Sr+;
# >0.8 eV/atom) are excluded.
DE_ENERGY_WINDOW: float = 0.3


@lru_cache(maxsize=None)
def _de_active_pairs_cached(
    composition_key: Tuple[Tuple[str, int], ...],
    tm_key: Tuple[str, ...],
) -> frozenset:
    from itertools import combinations

    labels = [element for element, count in composition_key for _ in range(count)]
    # "Is element E (or pair E, E') ever single-valent within the window?" is a
    # constrained minimum-energy question, answered by one DP each — enumerating
    # every in-window assignment is combinatorially hopeless for large cells.
    groups = [frozenset((element,)) for element in tm_key] + [
        frozenset(pair) for pair in combinations(tm_key, 2)
    ]
    try:
        result = min_energies_with_single_valence(labels, groups)
    except Exception:
        result = None
    if result is None:
        return frozenset()
    global_min, constrained_min = result
    cutoff = global_min + DE_ENERGY_WINDOW * len(labels)

    pairs = set()
    for element in tm_key:
        if constrained_min[frozenset((element,))] > cutoff:
            pairs.add(frozenset((element,)))
    for el_a, el_b in combinations(tm_key, 2):
        possible_a = constrained_min[frozenset((el_a,))] <= cutoff
        possible_b = constrained_min[frozenset((el_b,))] <= cutoff
        both = constrained_min[frozenset((el_a, el_b))] <= cutoff
        if possible_a and possible_b and not both:
            pairs.add(frozenset((el_a, el_b)))
    return frozenset(pairs)


def de_active_pairs(structure, tm_elements) -> frozenset:
    """Element pairs whose bridges carry double exchange, gated by charge balance.

    A pair (E, E') is DE-active when no charge-balanced oxidation distribution
    within ``DE_ENERGY_WINDOW`` of the best one makes both elements single-valent:

    - E == E': mixed valence is *forced* (e.g. Mn averaging +3.75 in
      LaSr3Mn4O12) — the classic same-element DE case.
    - E != E': each element *can* be single-valent, but never both at once —
      the itinerant carrier resonates between them (e.g. Fe-mixed <-> Co-mixed
      in LaSr3FeCoO12), the cross-element DE case.

    Materials with an accessible all-single-valent distribution (stoichiometric
    perovskites, hematite, goethite) get no flags, so spurious mixed-valence
    *assignments* from moment matching can no longer switch the term on.
    Composition-cached: repeated geometries of one material cost one enumeration.
    """
    from collections import Counter

    composition = Counter(structure.element_symbols())
    return _de_active_pairs_cached(
        tuple(sorted(composition.items())), tuple(sorted(tm_elements))
    )


def _sphere_quadrature(n_theta: int = 6, n_phi: int = 12):
    """Points and weights on the unit sphere, exact for polynomials up to degree 11 (weights sum to 1)."""
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    st = np.sqrt(1.0 - x ** 2)
    pts = np.stack([np.outer(st, np.cos(phi)), np.outer(st, np.sin(phi)), np.outer(x, np.ones(n_phi))], axis=-1)
    return pts.reshape(-1, 3), np.repeat(w / (2.0 * n_phi), n_phi)


_CF_POINTS, _CF_WEIGHTS = _sphere_quadrature()
# the five global real d orbitals at the quadrature points (angular parts as n^T Q n)
_CF_ORBITALS = np.einsum("gi,aij,gj->ga", _CF_POINTS, sk_table.D_ORBITAL_TENSORS, _CF_POINTS)


def crystal_field_matrices(directions: np.ndarray, weights2: np.ndarray, weights4: np.ndarray):
    """The k = 2 and k = 4 point-charge crystal-field matrices (5, 5) on the global real d orbitals,

        H_k[a, b] = sum_L w_k[L] <a| P_k(n . u_L) |b>      (angular integral, common positive constant dropped)

    for ligand unit vectors ``directions`` (n, 3) with weights ``|q_L| / R_L^3`` (k = 2) and ``|q_L| / R_L^5``
    (k = 4). Positive = the electron is pushed up by the ligand's negative charge. The radial integrals <r^2>,
    <r^4> are not included, so the two matrices are on different scales and are used one after the other
    (``crystal_field_basis``), never added."""
    c = _CF_POINTS @ np.asarray(directions, dtype=float).reshape(-1, 3).T          # (g, n)
    p2 = (1.5 * c ** 2 - 0.5) @ np.asarray(weights2, dtype=float)
    p4 = ((35.0 * c ** 4 - 30.0 * c ** 2 + 3.0) / 8.0) @ np.asarray(weights4, dtype=float)
    h2 = _CF_ORBITALS.T @ ((_CF_WEIGHTS * p2)[:, None] * _CF_ORBITALS)
    h4 = _CF_ORBITALS.T @ ((_CF_WEIGHTS * p4)[:, None] * _CF_ORBITALS)
    return h2, h4


# <dz2| P_2(n . z) |dz2> in ``crystal_field_matrices``' normalization: converts the eg splitting of H_2 to the
# units of the k = 2 field tensor (e / A^3) that ``_CF_DEGENERATE_TOL`` is stated in.
_CF_K2_UNIT = float(crystal_field_matrices(np.array([[0.0, 0.0, 1.0]]), np.ones(1), np.zeros(1))[0][
    sk_table.D_ORBITALS.index("dz2"), sk_table.D_ORBITALS.index("dz2")])


def _site_fields(structure, site_index: int, ligand_cutoff: float = 3.0
                 ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The k = 2 and k = 4 point-charge field matrices of a site's ligand cage within ``ligand_cutoff``
    (``crystal_field_matrices``, weights |q| / R^3 and |q| / R^5); None for a site with no ligand in reach."""
    charges = ANION_CHARGES
    center = structure.cartesian_coords[site_index]
    dirs, w2, w4 = [], [], []
    for nbr in structure.neighbors(site_index, ligand_cutoff):
        if nbr.symbol not in ANION_SPECIES:
            continue
        d = nbr.coords - center
        r = float(np.linalg.norm(d))
        if r < 1e-8:
            continue
        q = abs(charges.get(nbr.symbol, -2))
        dirs.append(d / r); w2.append(q / r ** 3); w4.append(q / r ** 5)
    if not dirs:
        return None
    return crystal_field_matrices(np.array(dirs), np.array(w2), np.array(w4))


def _level_skewness(h: np.ndarray) -> float:
    lam = np.linalg.eigvalsh(h)
    lam = lam - lam.mean()
    norm = float((lam ** 2).sum())
    return 0.0 if norm < 1e-30 else float((lam ** 3).sum() / norm ** 1.5)


# The octahedron's k = 4 level pattern (eg 2 up, t2g 3 down) has skewness +0.1826; the tetrahedron's (e down,
# t2 up) the opposite. ``cubic_field_sign`` is the ratio.
_KAPPA_OCT = _level_skewness(crystal_field_matrices(
    np.vstack([np.eye(3), -np.eye(3)]), np.ones(6), np.ones(6))[1])


def cubic_field_sign(structure, site_index: int, *, ligand_cutoff: float = 3.0) -> float:
    """kappa: the skewness of the site's k = 4 crystal-field levels over the octahedron's, a continuous
    octahedral (+1) / tetrahedral (-1) ordering of the d levels. Square plane +0.78, square pyramid +0.97,
    trigonal bipyramid +0.84; flattening a tetrahedron moves it smoothly from -1 to +0.78. Rotation invariant, no
    labels, no fitted constants. 1 (octahedral ordering, the model's original assumption) for a site with no
    ligand in reach."""
    fields = _site_fields(structure, site_index, ligand_cutoff)
    if fields is None:
        return 1.0
    return _level_skewness(fields[1]) / _KAPPA_OCT


def tetrahedral_fraction(structure, site_index: int, *, ligand_cutoff: float = 3.0) -> float:
    """x_tet = (1 - kappa) / 2 clipped to [0, 1] (``cubic_field_sign``): the weight of the tetrahedral level
    ordering in a site's description. Ideal octahedron 0, ideal tetrahedron 1; real octahedra 0-0.05."""
    k = cubic_field_sign(structure, site_index, ligand_cutoff=ligand_cutoff)
    return float(min(1.0, max(0.0, 0.5 * (1.0 - k))))


# Coordination weights below this are dropped (an ideal octahedron gives ~1e-16 tetrahedral weight).
_COORD_WEIGHT_TOL = 1e-9


def coordination_fractions(structure, site_index: int, *, ligand_cutoff: float = 3.0
                           ) -> Tuple[Tuple[str, float], ...]:
    """``(("oct", 1 - x), ("tet", x))`` with x = ``tetrahedral_fraction``, zero weights dropped. Every bridge end,
    feature and direct-exchange term mixes the two level orderings with these weights, the way oxidation-state
    fractions are mixed: no site is classified."""
    x = tetrahedral_fraction(structure, site_index, ligand_cutoff=ligand_cutoff)
    return tuple((c, w) for c, w in (("oct", 1.0 - x), ("tet", x)) if w > _COORD_WEIGHT_TOL)


def site_orbital_components(structure, site_index: int, descriptor: IonDescriptor, *,
                            ligand_cutoff: float = 3.0, charges=None
                            ) -> List[Tuple[float, np.ndarray, Optional[np.ndarray]]]:
    """``[(weight, occupancy, basis)]`` of one site in its own state: each coordination's level ordering
    (``coordination_fractions``) with its shell-averaged occupancy and crystal-field orbitals. For the
    direct-exchange terms, which are bilinear in the two sites and so mix over the combinations exactly."""
    out = []
    for coord, w in coordination_fractions(structure, site_index, ligand_cutoff=ligand_cutoff):
        d = ion_descriptor(descriptor.element, int(descriptor.oxidation_state), descriptor.spin_state, coord)
        for wb, basis, levels in crystal_field_bases(structure, site_index, coord, ligand_cutoff=ligand_cutoff,
                                                     charges=charges) or [(1.0, None, None)]:
            out.append((w * wb, level_filling(d, levels)[0], basis))
    return out


def crystal_field_basis(
    structure,
    site_index: int,
    coordination: str = "oct",
    *,
    ligand_cutoff: float = 3.0,
    charges=None,
) -> Tuple[Optional[np.ndarray], float]:
    """A site's d orbitals from the point-charge crystal field: ``(basis, eg_splitting)``, the first entry of
    ``crystal_field_bases``. ``basis`` (5, 5) has the orbitals as rows of coefficients on the global real d
    orbitals, in the slots of ``sk_table.D_ORBITALS``: the three of ``T2G_INDICES`` span the t2g (tetrahedral: t2)
    level and the two of ``EG_INDICES`` the eg (e) level, each set lowest k = 2 orbital first. No axes are fitted
    and nothing depends on how the crystal is oriented. ``eg_splitting`` is the k = 2 splitting of the two eg
    orbitals in the units of the field tensor ``sum_L |q_L| / R_L^3 (3 u u^T - 1) / 2``, 0 for an ideal cage.
    ``(None, 0.0)`` for a site with no ligand within ``ligand_cutoff``."""
    bases = crystal_field_bases(structure, site_index, coordination, ligand_cutoff=ligand_cutoff, charges=charges)
    if not bases:
        return None, 0.0
    _w, basis, levels = bases[0]
    e = levels[list(EG_INDICES)]
    return basis, float(e[1] - e[0]) / (2.0 * _CF_K2_UNIT)


# Polarization sets of ``BridgeGeometry.T`` (axis 0): half-filled; empty (unweighted, the acceptor); empty weighted by
# each ``ee_spin`` rule; empty t2g weighted likewise; empty t2g weighted by the Kugel-Khomskii factor.
T_HALF, T_EMPTY = 0, 1
T_EE = {"n": 2, "n2": 3, "n_delta": 4}
T_EE_T2G = {"n": 5, "n2": 6, "n_delta": 7}
T_KK = 8
N_TSETS = 9
# Kugel-Khomskii t2g acceptor: kappa = n J_H / U, U the on-site Coulomb of the reference PBE+U (U_eff + J_H; U_eff as
# the protocol: MP values, Ti 3.5 and Cu 6.5 eV from the x01 choice) and J_H the atomic-like Hund coupling.
KK_HUND = 0.9
KK_U = {el: u + KK_HUND for el, u in {"Ti": 3.5, "V": 3.25, "Cr": 3.7, "Mn": 3.9, "Fe": 5.3, "Co": 3.32, "Ni": 6.2,
                                       "Cu": 6.5}.items()}


# The empty-orbital covalency. An empty orbital's charge-transfer energy is Delta (the table's: O 2p -> the ion's
# lowest empty d level) only if it lies in that lowest level; an empty orbital one level up (the eg of a d1 / d2
# octahedral ion, whose lowest empty level is t2g) costs Delta + 10Dq, so its polarization (the model's amplitude goes
# as 1 / Delta: ln_delta weight -1) is scaled by Delta / (Delta + 10Dq). TEN_DQ: PBE+U's empty-eg covalency c_eg
# (n_maj + n_min) of LaTiO3 / YTiO3 and LaVO3 / YVO3 / V2O3 against Cr3+'s implies 10Dq = 1.2 and 1.8 eV (magnets
# ee_asymmetry_check.py); 1.5 eV, the optical range of 3d 3+ oxides. Tetrahedral splitting 4/9 of it.
TEN_DQ = {"oct": 1.5, "tet": 1.5 * 4.0 / 9.0}


def empty_level_scale(descriptor: IonDescriptor) -> np.ndarray:
    """(5,) factor by slot on a state's empty orbitals (``level_filling``): Delta / (Delta + 10Dq) on the upper level's
    slots when the lower level has an empty orbital too, else 1."""
    from quick_mag.amplitude_head import DELTA_REF, ln_delta

    scale = np.ones(5)
    lower, upper = ((T2G_INDICES, EG_INDICES) if descriptor.coordination != "tet" else (EG_INDICES, T2G_INDICES))
    ehf_lower = descriptor.ehf_t2g if descriptor.coordination != "tet" else descriptor.ehf_eg
    if ehf_lower[0] > 0:
        delta = DELTA_REF * float(np.exp(ln_delta(descriptor.element, descriptor.oxidation_state)))
        scale[list(upper)] = delta / (delta + TEN_DQ.get(descriptor.coordination, TEN_DQ["oct"]))
    return scale


def ee_spin_weights(descriptor: IonDescriptor, unpaired: float) -> Dict[str, float]:
    """The empty-orbital spin weight of one ion state under each ``ee_spin`` rule."""
    from quick_mag.amplitude_head import ln_delta

    n = float(unpaired)
    return {"n": n, "n2": n * n, "n_delta": n * float(np.exp(-ln_delta(descriptor.element, descriptor.oxidation_state)))}

# Two k = 4 levels closer than this (relative to the level spread) are one degenerate level.
_CF_LEVEL_TOL = 1e-6

# Every ion within this distance of a site enters the k = 2 field that orders the orbitals inside each level, at its
# formal charge. The A cations set the t2g order of RTiO3 / RVO3 (the O cage alone gets LaTiO3's occupied orbital
# wrong: overlap with PBE+U's 0.05 against 0.87); stable for any cutoff 4.0-7.0 A (magnets t2g_order_check.py).
CF_FIELD_CUTOFF = 4.5

# Formal charges for atoms an assignment does not cover (``formal_charges``): transition metals come from their
# descriptors, anions from ANION_CHARGES, everything else from here (0 = left out of the field).
_NOMINAL_CHARGES: Dict[str, int] = {
    **{el: 1 for el in ("H", "Li", "Na", "K", "Rb", "Cs", "Ag")},
    **{el: 2 for el in ("Be", "Mg", "Ca", "Sr", "Ba", "Zn", "Cd", "Pb")},
    **{el: 3 for el in ("B", "Al", "Ga", "In", "Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
                        "Er", "Tm", "Yb", "Lu", "Bi")},
    **{el: 4 for el in ("C", "Si", "Ge", "Sn", "Zr", "Hf")},
    **{el: 5 for el in ("P", "As", "Nb", "Ta", "Sb")},
    **{el: 6 for el in ("Mo", "W", "Te")},
}


def formal_charges(structure, descriptors: Optional[Dict[int, IonDescriptor]] = None,
                   site_charges=None) -> np.ndarray:
    """Formal charge of every atom: a descriptor's oxidation state, else the anion charge, else ``site_charges``
    (an assignment's ``site_oxidation_states``), else ``_NOMINAL_CHARGES`` (0 if unknown)."""
    sym = structure.element_symbols()
    q = np.zeros(len(sym))
    for i, el in enumerate(sym):
        if descriptors is not None and i in descriptors:
            q[i] = float(descriptors[i].oxidation_state)
        elif el in ANION_CHARGES:
            q[i] = float(ANION_CHARGES[el])
        elif site_charges is not None and site_charges[i] is not None and np.isfinite(float(site_charges[i])):
            q[i] = float(site_charges[i])
        else:
            q[i] = float(_NOMINAL_CHARGES.get(el, 0))
    return q


def model_charges(structure, descriptors: Dict[int, IonDescriptor], site_charges=None,
                  valence: str = "class") -> np.ndarray:
    """``formal_charges`` as the model uses them: with ``valence="class"`` each transition-metal site carries its
    site class's average charge (``site_class_oxidation_fractions``), so the field, like everything else, cannot
    depend on which atom of a mixed-valent class got which label."""
    q = formal_charges(structure, descriptors, site_charges)
    if valence == "class":
        fr = site_class_oxidation_fractions(descriptors)
        for site, d in descriptors.items():
            q[site] = sum(state * w for state, w in fr[_site_class(d)])
    return q


def site_k2_field(structure, site_index: int, charges: np.ndarray, cutoff: Optional[float] = None) -> np.ndarray:
    """The k = 2 point-charge field matrix (5, 5) of every ion within ``cutoff`` (default CF_FIELD_CUTOFF) at the
    formal ``charges`` (``crystal_field_matrices``' sign: positive = an electron pushed up)."""
    cutoff = CF_FIELD_CUTOFF if cutoff is None else cutoff
    center = structure.cartesian_coords[site_index]
    dirs, w = [], []
    for nbr in structure.neighbors(site_index, cutoff):
        q = float(charges[nbr.index])
        d = nbr.coords - center
        r = float(np.linalg.norm(d))
        if q == 0.0 or r < 1e-8:
            continue
        dirs.append(d / r); w.append(-q / r ** 3)
    if not dirs:
        return np.zeros((5, 5))
    return crystal_field_matrices(np.array(dirs), np.array(w), np.zeros(len(w)))[0]


def _template_bases(h2: np.ndarray, h4: np.ndarray, coordination: str):
    """``[(weight, basis, levels)]`` for one level-ordering template ("oct": eg = the upper two k = 4 levels;
    "tet": e = the lower two). Inside each of the two levels the orbitals diagonalize the k = 2 field ``h2``, lower
    first; ``levels`` (5,) are their k = 2 energies by slot. When a degenerate k = 4 level straddles the template's
    cut (a D2d-flattened tetrahedron's doublet under the octahedral template), which member is "eg" is arbitrary:
    every choice is returned with equal weight, the average over the degenerate microstates."""
    lam, vec = np.linalg.eigh(h4)                            # ascending; columns
    n_low = 2 if coordination == "tet" else 3                # levels below the cut
    tol = _CF_LEVEL_TOL * max(float(lam[-1] - lam[0]), 1e-30)
    lo = n_low
    while lo > 0 and lam[n_low] - lam[lo - 1] < tol:         # the degenerate group around the cut
        lo -= 1
    hi = n_low
    while hi < 5 and lam[hi] - lam[n_low - 1] < tol:
        hi += 1
    group = list(range(lo, hi))
    k_low = n_low - lo                                       # members of the group that go below the cut
    from itertools import combinations

    choices = list(combinations(group, k_low)) if len(group) > 1 and lo < n_low < hi else [tuple(range(lo, n_low))]
    out = []
    for below in choices:
        low = list(range(lo)) + list(below)
        high = [k for k in range(5) if k not in low]
        low_v, high_v = vec[:, low], vec[:, high]
        eg, t2g = (low_v, high_v) if coordination == "tet" else (high_v, low_v)
        e_eg, r_eg = np.linalg.eigh(eg.T @ h2 @ eg)
        e_t, r_t = np.linalg.eigh(t2g.T @ h2 @ t2g)
        basis = np.zeros((5, 5))
        levels = np.zeros(5)
        basis[list(T2G_INDICES)] = (t2g @ r_t).T
        basis[list(EG_INDICES)] = (eg @ r_eg).T
        levels[list(T2G_INDICES)] = e_t
        levels[list(EG_INDICES)] = e_eg
        out.append((1.0 / len(choices), basis, levels))
    return out


def crystal_field_bases(structure, site_index: int, coordination: str = "oct", *, ligand_cutoff: float = 3.0,
                        charges=None) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """A site's d orbitals for one level-ordering template, ``[(weight, basis, levels)]`` (``_template_bases``): the
    two levels from the k = 4 field of the ligand cage within ``ligand_cutoff``, the order inside them from the
    k = 2 field of every ion within CF_FIELD_CUTOFF at the formal ``charges`` (an array over atoms,
    ``formal_charges``; None = the ligand cage alone). One entry unless a degenerate level sits at the template's
    cut; [] with no ligand in reach."""
    fields = _site_fields(structure, site_index, ligand_cutoff)
    if fields is None:
        return []
    h2, h4 = fields
    if charges is not None:
        h2 = site_k2_field(structure, site_index, charges)
    return _template_bases(h2, h4, coordination)


def level_filling(descriptor: IonDescriptor, levels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """``(half, empty)`` (5,) by slot of a site's ``crystal_field_bases`` orbitals: the net unpaired spin and the
    empty (majority-spin) fraction of each orbital. Hund filling inside each level: the level's full orbitals
    (``descriptor.ehf_*``) take its lowest k = 2 orbitals, then the half-filled, then the empty ones, and orbitals
    within _CF_DEGENERATE_TOL share their filling equally (the average over the degenerate microstates). One rule
    for every d count: an eg^1 electron in the lower eg orbital, an eg^3 hole in the upper, Ti3+ / V3+ t2g
    electrons in the lowest t2g, a minority Fe2+ / Co2+ electron likewise. ``levels`` None (or an undistorted
    cage) gives the shell average: H/3 per t2g, H/2 per eg."""
    half, empty = np.zeros(5), np.zeros(5)
    for slots, (n_e, n_h, n_f) in ((list(T2G_INDICES), descriptor.ehf_t2g), (list(EG_INDICES), descriptor.ehf_eg)):
        lv = np.zeros(len(slots)) if levels is None else np.asarray(levels, dtype=float)[slots]
        order = np.argsort(lv, kind="stable")
        h, e = np.zeros(len(slots)), np.zeros(len(slots))
        h[order[n_f:n_f + n_h]] = 1.0
        e[order[n_f + n_h:]] = 1.0
        k = 0
        while k < len(order):                                 # degenerate groups share their filling
            g = [order[k]]
            while k + len(g) < len(order) and lv[order[k + len(g)]] - lv[order[k]] < _CF_DEGENERATE_TOL:
                g.append(order[k + len(g)])
            h[g], e[g] = h[g].mean(), e[g].mean()
            k += len(g)
        half[slots], empty[slots] = h, e
    return half, empty


def occupancy_vector(descriptor: IonDescriptor) -> np.ndarray:
    """The shell-averaged net unpaired spin per d orbital (``level_filling`` with degenerate levels): H_t2g/3 per
    t2g orbital, H_eg/2 per eg orbital (Mn3+ -> [1, 1, 1, 0.5, 0.5])."""
    return level_filling(descriptor)[0]


def empty_vector(descriptor: IonDescriptor) -> np.ndarray:
    """The shell-averaged empty majority-spin fraction per d orbital (Cr3+ -> [0, 0, 0, 1, 1]; Fe3+ -> 0)."""
    return level_filling(descriptor)[1]


@lru_cache(maxsize=None)
def _shannon_crystal_radius(
    element: str, charge: int, spin_label: Optional[str], coordination: str = "VI"
) -> float:
    """Mean Shannon crystal radius with fallback: exact (coordination, spin) ->
    (coordination) -> any coordination at that charge -> any entry for the element.

    ``coordination`` is the Shannon label: "VI" for octahedral, "IV" for
    tetrahedral sites."""
    entry = SHANNON_IONIC_RADII[element]
    for kwargs in (
        {"charge": charge, "coordination": coordination, "spin_state": spin_label},
        {"charge": charge, "coordination": coordination},
        {"charge": charge},
        {},
    ):
        records = entry.find(**kwargs)
        if records:
            return sum(r.crystal_radius for r in records) / len(records)
    raise KeyError(f"No Shannon radius for {element} (charge {charge}).")


# Bond-valence softness (Brown & Altermatt 1985): a bond of valence s has length R_BV - b ln s, so an ion's ideal
# bond is longer by b ln(CN / 6) than its octahedral one. b ln(4/6) = -0.150 A; Shannon IV - VI is -0.14 to -0.185 A
# for every 3d ion that has an IV radius (Mn2+, Fe2+, Fe3+, Co2+, Ni2+, Cu2+, Zn2+, Mn4+, Ti4+), and the table has
# none for Co3+, Cr3+, V3+.
BOND_VALENCE_B = 0.37


def r0_metal_ligand(
    metal: str,
    oxidation_state: int,
    spin_state: str,
    ligand: str,
    coordination: str = "oct",
) -> float:
    """Reference M-L bond length: sum of Shannon crystal radii (Angstrom).

    The metal radius is the one for its ``coordination`` ("oct" -> VI,
    "tet" -> IV); the ligand radius is the tabulated anion radius as before.
    """
    metal_radius = _shannon_crystal_radius(
        metal,
        int(oxidation_state),
        _SHANNON_SPIN_LABEL.get(spin_state.upper()),
        _SHANNON_COORDINATION.get(coordination, "VI"),
    )
    ligand_radius = _shannon_crystal_radius(ligand, ANION_CHARGES.get(ligand, -2), None)
    return metal_radius + ligand_radius


@lru_cache(maxsize=None)
def r0_metal_ligand_averaged(
    metal: str,
    states: Tuple[Tuple[int, float], ...],
    spin_state: str,
    ligand: str,
    coordination: str = "oct",
) -> float:
    """Reference M-L bond length averaged over the states the metal is found in.

    ``states`` is ``((oxidation_state, fraction), ...)`` from
    ``element_oxidation_fractions``. Which *site* carries which valence is an
    arbitrary choice of the assignment expander, so the reference length is taken
    from the element's composition rather than from one site's label.

    ``build_bridges`` calls it with ``coordination="oct"`` (Shannon VI) and adds the cage's bond-valence shift
    ``BOND_VALENCE_B * ln(CN / 6)``, CN the site's smooth ligand count: continuous in the cage, no label.

    This averages the radius, not the range factor ``(R0 / r)^n`` it feeds --
    a linearization of the exponent. It keeps the damping a per-end scalar, which
    is what lets ``build_bridges`` fold the state average into the channel
    intensities instead of carrying every oxidation state through to ``bridge_J``.
    A single-state element reproduces ``r0_metal_ligand`` exactly.
    """
    return sum(
        weight * r0_metal_ligand(metal, state, spin_state, ligand, coordination)
        for state, weight in states
    )


def element_oxidation_fractions(
    descriptors: Dict[int, IonDescriptor],
) -> Dict[str, Tuple[Tuple[int, float], ...]]:
    """Fraction of each oxidation state present, per element, sorted by state.

    LaFeO3 gives ``{"Fe": ((3, 1.0),)}``; a cell whose charge balance forces Mn to
    average +3.75 gives ``{"Mn": ((3, 0.25), (4, 0.75))}``.

    Read off the per-site assignment that was passed in, whose *composition* is the
    oxidation distribution it was expanded from -- the part of it that is not
    arbitrary. Hand-set states in the UI therefore move the fractions, which is the
    intended behaviour: overriding a site says something about the material.
    """
    counts: Dict[str, Dict[int, int]] = {}
    for descriptor in descriptors.values():
        per_element = counts.setdefault(descriptor.element, {})
        state = int(descriptor.oxidation_state)
        per_element[state] = per_element.get(state, 0) + 1
    return {
        element: tuple(
            (state, per_element[state] / float(sum(per_element.values())))
            for state in sorted(per_element)
        )
        for element, per_element in counts.items()
    }


def site_class_oxidation_fractions(
    descriptors: Dict[int, IonDescriptor],
) -> Dict[Tuple[str, str], Tuple[Tuple[int, float], ...]]:
    """``element_oxidation_fractions`` resolved by coordination as well as element.

    Keyed by ``(element, coordination)``. Within one such class which site
    carries which valence is still arbitrary, but across classes it is not: in
    magnetite the tetrahedral Fe are Fe3+ and only the octahedral ones are
    mixed-valent. A structure whose sites of an element share one coordination
    (every perovskite) gets exactly the per-element fractions.
    """
    counts: Dict[Tuple[str, str], Dict[int, int]] = {}
    for descriptor in descriptors.values():
        per_class = counts.setdefault((descriptor.element, descriptor.coordination), {})
        state = int(descriptor.oxidation_state)
        per_class[state] = per_class.get(state, 0) + 1
    return {
        key: tuple(
            (state, per_class[state] / float(sum(per_class.values())))
            for state in sorted(per_class)
        )
        for key, per_class in counts.items()
    }


def _site_class(descriptor: IonDescriptor) -> Tuple[str, str]:
    return descriptor.element, descriptor.coordination


def _de_resonance_weight(
    states_i: Tuple[Tuple[int, float], ...],
    states_j: Tuple[Tuple[int, float], ...],
) -> float:
    """Weight of the valence combinations a carrier can resonate between.

    ``sum_{a != b} sqrt(p_a q_b p_b q_a)``: the configuration ``(i^a, j^b)`` and
    its hopped partner ``(i^b, j^a)`` must both occur. Equals ``1 - sum_a p_a^2``
    when ``p == q``, the original same-element weight.
    """
    p = dict(states_i)
    q = dict(states_j)
    states = sorted(set(p) | set(q))
    total = 0.0
    for a in states:
        for b in states:
            if a != b:
                total += math.sqrt(p.get(a, 0.0) * q.get(b, 0.0) * p.get(b, 0.0) * q.get(a, 0.0))
    return float(total)


@dataclass(frozen=True)
class BridgeGeometry:
    """One M-L-M bridge with precomputed per-end channel intensity vectors.

    ``Bsig_*``/``Bpi_*`` are ``sum_a mbar_a W_{a,p}`` (3,) in the bridge frame;
    everything downstream is elementwise in these plus the damping scalars.
    """

    site_i: int
    site_j: int
    ligand_index: int
    metal_i: str
    metal_j: str
    ligand: str
    r_iL: float
    r_jL: float
    R0_iL: float
    R0_jL: float
    # Half-filled-orbital channel intensities (per bridge frame, (3,)), from each end's ``level_filling``.
    Bsig_i: np.ndarray
    Bpi_i: np.ndarray
    Bsig_j: np.ndarray
    Bpi_j: np.ndarray
    u_iL: np.ndarray = None
    u_jL: np.ndarray = None
    frame: np.ndarray = None
    basis_i: np.ndarray = None   # each end's d orbitals (5, 5), ``crystal_field_basis`` (None = the global ones)
    basis_j: np.ndarray = None
    # True when the end elements form a DE-active pair (charge-balance-forced
    # mixed valence, same-element or cross-element resonant; see de_active_pairs).
    de_active: bool = False
    # Fraction of the end-state combinations that actually carry a hopping carrier:
    # for a same-element pair, the chance the two ends differ in valence. See
    # ``build_bridges``.
    de_weight: float = 1.0
    # Occupied->empty FM channel: open when either end has an empty majority-spin
    # orbital (``level_filling``'s empty orbitals: the upper eg of an eg^1 state, Cr3+'s eg,
    # Ti3+'s upper t2g), with those orbitals' channel intensities on each end. ``bridge_J`` adds
    # -j_fm_i j_fm_j (mu_occ_i . mu_emp_j + mu_emp_i . mu_occ_j). See
    # docs/theory/magnetism-model.md.
    oe_active: bool = False
    Bsig_emp_i: np.ndarray = None
    Bpi_emp_i: np.ndarray = None
    Bsig_emp_j: np.ndarray = None
    Bpi_emp_j: np.ndarray = None
    # Donor intensities of the occupied->empty channel. ``build_bridges`` leaves them
    # None: every occupied orbital donates, so the donor is ``Bsig_i``/``Bpi_i``.
    # Kept so a caller can restrict the donor by hand.
    Bsig_occ_i: np.ndarray = None
    Bpi_occ_i: np.ndarray = None
    Bsig_occ_j: np.ndarray = None
    Bpi_occ_j: np.ndarray = None
    # Direct-exchange interference geometry (4,): sum_ab m_a m_b S_ab x [A_a A_b u_i.u_j, -A_a (u_i . t_b),
    # -(t_a . u_j) A_b, t_a . t_b]; None when the metals are beyond the direct-exchange cutoff.
    G_dx: Optional[np.ndarray] = None
    # Centred atomic features of each end's ion (``amplitude_head.FEATURES``).
    feat_i: Optional[np.ndarray] = None
    feat_j: Optional[np.ndarray] = None
    # Polarization matrices of each end over the ligand p shell, global axes, unit amplitudes: ``T[set, part]``
    # (3, 3, 3, 3) with set = 0 half-filled orbitals, 1 empty majority-spin orbitals (the acceptor), 2 the same
    # weighted by the ``ee_spin`` rules (T_EE), the empty t2g orbitals likewise (T_EE_T2G) and by the Kugel-Khomskii
    # factor (T_KK): N_TSETS sets in all; part = 0 sigma-sigma, 1 pi-pi,
    # 2 sigma-pi interference (sign of V_pdpi / V_pdsigma < 0 included). An end's matrix is
    # f_sigma T[s, 0] + f_pi T[s, 1] + sqrt(f_sigma f_pi) T[s, 2]  (``end_polarization``); the Bsig / Bpi vectors
    # above are the diagonals of parts 0 and 1 in the bridge frame.
    T_i: Optional[np.ndarray] = None
    T_j: Optional[np.ndarray] = None
    # (IE_{q+1}, IE_q, q) of each end's ion, averaged over its states: donor cost, acceptor gain, charge
    # (``oe_rule = "ionization"``).
    ie_i: Tuple[float, float, float] = (0.0, 0.0, 3.0)
    ie_j: Tuple[float, float, float] = (0.0, 0.0, 3.0)

    @property
    def cos_theta(self) -> float:
        return float(self.u_iL @ self.u_jL)


def _end_intensities(
    m_occ: np.ndarray,
    u: np.ndarray,
    frame: np.ndarray,
    basis: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(Bsig, Bpi) channel intensity vectors for one bridge end."""
    w_sigma, w_pi = sk_table.channel_weights(u, frame, basis=basis)
    return m_occ @ w_sigma, m_occ @ w_pi


def _orbital_matrices(weights: np.ndarray, coeffs: np.ndarray, u: np.ndarray,
                      a_sigma: np.ndarray, t_pi: np.ndarray) -> np.ndarray:
    """(3, 3, 3): the sigma-sigma, pi-pi and sigma-pi parts of ``sum_k w_k v_k v_k^T`` for orbitals
    ``coeffs`` (k, 5) on the local d basis, ``v_k = (c_k . A_sigma) u - (c_k . t_pi)`` at unit amplitudes."""
    sa = coeffs @ a_sigma                       # (k,)
    p = coeffs @ t_pi                           # (k, 3)
    out = np.zeros((3, 3, 3))
    out[0] = float(weights @ (sa * sa)) * np.outer(u, u)
    out[1] = (p * weights[:, None]).T @ p
    cross = ((weights * sa)[:, None] * p).sum(axis=0)
    out[2] = -(np.outer(u, cross) + np.outer(cross, u))
    return out


def end_polarization(T: np.ndarray, f_sigma, f_pi, coherent: bool = True) -> np.ndarray:
    """One set's polarization matrix from its parts ``T`` (3, 3, 3) [or (n, 3, 3, 3) with arrays f]."""
    f_sigma, f_pi = np.asarray(f_sigma, dtype=float), np.asarray(f_pi, dtype=float)
    fs, fp = f_sigma[..., None, None], f_pi[..., None, None]
    P = fs * T[..., 0, :, :] + fp * T[..., 1, :, :]
    if coherent:
        P = P + np.sqrt(fs * fp) * T[..., 2, :, :]
    return P


def build_bridges(
    structure,
    descriptors: Dict[int, IonDescriptor],
    *,
    de_pairs: Optional[frozenset] = None,
    anion_bond_cutoff: float = 3.0,
    super_cutoff: float = 7.0,
    valence: str = "class",
    site_charges=None,
    charges: Optional[np.ndarray] = None,
) -> List[BridgeGeometry]:
    """Enumerate M-L-M bridges between descriptor sites with actual bond vectors.

    For each bridging anion, pairs its transition-metal neighbours and records the
    image-resolved M-L unit vectors, precomputing the channel intensities. A metal
    pair sharing several ligands (edge/face sharing) yields one bridge per ligand.

    Each site's d orbitals are those of its ligands' crystal field (``crystal_field_basis``),
    so tilted octahedra keep their eg/t2g character and tetrahedral sites their
    e/t2 character, with no local axes.

    ``de_pairs`` overrides the charge-balance double-exchange gate. When
    None it is computed via ``de_active_pairs``.

    ``valence`` says what a site's oxidation state means. ``"class"`` (the default):
    which site of a mixed-valent class carries which state is arbitrary, so every
    site is the fraction-weighted average over its class's states (below).
    ``"site"``: each site *is* the state it was given -- for states read off a
    calculation (Bader / moments) or set per atom on purpose, where a Mn3+ next to a
    Mn4+ is a statement about those two atoms. The double-exchange gate and its
    resonance weight still come from the class fractions either way: whether
    carriers can hop is a property of the material, not of one labelling of it.

    ``site_charges`` (an assignment's ``site_oxidation_states``) gives the formal charges of the non-transition-metal
    ions in the k = 2 field (``model_charges``; without it they take nominal charges); ``charges`` (per atom)
    overrides the whole set.
    """
    if valence not in ("class", "site"):
        raise ValueError(f"valence must be 'class' or 'site', got {valence!r}.")
    # Which site of a mixed-valent element carries which state is arbitrary (see
    # ``element_oxidation_fractions``), so no site is given a definite valence. Each
    # end is the fraction-weighted average over the states its element is found in,
    # and a bridge is the weighted average over the *combinations* of the two ends'
    # states. Every channel is a product of one factor per end, so that double sum
    # factorizes exactly into the per-end sums built here -- no combination loop and
    # no approximation. A single-valent element leaves everything unchanged.
    #
    # "Its element" is resolved by coordination too (``site_class_oxidation_fractions``):
    # tetrahedral and octahedral sites of one element are not interchangeable.
    fractions = site_class_oxidation_fractions(descriptors)

    def end_fractions(descriptor: IonDescriptor) -> Tuple[Tuple[int, float], ...]:
        if valence == "site":
            return ((int(descriptor.oxidation_state), 1.0),)
        return fractions[_site_class(descriptor)]

    # No site is classified as octahedral or tetrahedral: each mixes the two level orderings with the weights
    # of ``coordination_fractions`` (the cubic field's sign, continuous), exactly as it mixes oxidation states.
    # Every state of a site below carries its ordering's crystal-field orbitals. ``bases`` keeps the dominant
    # ordering's orbitals for the bridge record (``BridgeGeometry.basis_i``: UI, diagnostics).
    coords: Dict[int, Tuple[Tuple[str, float], ...]] = {}
    cf: Dict[Tuple[int, str], List[Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]]] = {}
    bases: Dict[int, Optional[np.ndarray]] = {}
    if charges is None:
        charges = model_charges(structure, descriptors, site_charges, valence)
    for site in descriptors:
        coords[site] = coordination_fractions(structure, site, ligand_cutoff=anion_bond_cutoff)
        for coord, _w in coords[site]:
            cf[(site, coord)] = (crystal_field_bases(structure, site, coord, ligand_cutoff=anion_bond_cutoff,
                                                     charges=charges) or [(1.0, None, None)])
        bases[site] = cf[(site, max(coords[site], key=lambda cw: cw[1])[0])][0][1]
    state_descriptors: Dict[Tuple[str, int, str, str], IonDescriptor] = {}
    for site, descriptor in descriptors.items():
        for state, _weight in end_fractions(descriptor):
            for coord, _w in coords[site]:
                key = (descriptor.element, state, descriptor.spin_state, coord)
                if key not in state_descriptors:
                    state_descriptors[key] = ion_descriptor(*key)

    # Every state of a site: its Hund filling over its crystal-field orbitals (``level_filling``): the half-filled
    # orbitals polarize the ligand (J_SE, and the donor of the occupied->empty channel), the empty majority-spin
    # orbitals are the acceptor (Kanamori: at ~90 degrees a neighbour's electron reaches an empty orbital through the
    # shared ligand p orbital and Hund-aligns there -> FM; at 180 the two use orthogonal p orbitals). Orbital order
    # (eg^1, eg^3, t2g^1, t2g^2, minority t2g) is whatever the field's levels make it; a degenerate level averages.
    # per state: (weight, half (5,), empty (5,), unpaired electrons, crystal-field orbitals)
    site_states: Dict[int, List[tuple]] = {}
    for site, descriptor in descriptors.items():
        states = []
        for state, weight in end_fractions(descriptor):
            for coord, wc in coords[site]:
                alt = state_descriptors[(descriptor.element, state, descriptor.spin_state, coord)]
                for wb, rot, levels in cf[(site, coord)]:
                    half, empty = level_filling(alt, levels)
                    empty = empty * empty_level_scale(alt)          # an empty orbital above another empty level
                    n = float(half.sum())
                    states.append((weight * wc * wb, half, empty, n, rot, ee_spin_weights(alt, n),
                                   n * KK_HUND / KK_U.get(alt.element, 5.0)))
        site_states[site] = states

    def _end_channels(idx, u, frame):
        """``(Bsig, Bpi, Bsig_emp, Bpi_emp)`` of one bridge end: the half-filled orbitals over every state (the
        ligand polarization) and the empty majority-spin orbitals (the acceptor), in the bridge frame."""
        bsig = np.zeros(3); bpi = np.zeros(3)
        bsig_emp = np.zeros(3); bpi_emp = np.zeros(3)
        for weight, half, empty, _unpaired, rot, _sw, _kk in site_states[idx]:
            bs, bp = _end_intensities(half, u, frame, basis=rot)
            bsig = bsig + weight * bs
            bpi = bpi + weight * bp
            if empty.any():
                bse, bpe = _end_intensities(empty, u, frame, basis=rot)
                bsig_emp = bsig_emp + weight * bse
                bpi_emp = bpi_emp + weight * bpe
        return bsig, bpi, bsig_emp, bpi_emp

    _unit = np.eye(5)
    _T2G_MASK = np.zeros(5)
    _T2G_MASK[list(T2G_INDICES)] = 1.0

    def _end_tensor(idx, u):
        """``BridgeGeometry.T`` of one end: the same orbitals as ``_end_channels``, as matrices."""
        T = np.zeros((N_TSETS, 3, 3, 3))
        for weight, half, empty, unpaired, rot, sw, kk in site_states[idx]:
            a_sigma, t_pi = sk_table.pd_amplitudes(u, basis=rot)
            T[T_HALF] += weight * _orbital_matrices(half, _unit, u, a_sigma, t_pi)
            if empty.any():
                acc = _orbital_matrices(empty, _unit, u, a_sigma, t_pi)
                T[T_EMPTY] += weight * acc
                for rule, k in T_EE.items():
                    T[k] += weight * sw[rule] * acc
                empty_t2g = empty * _T2G_MASK
                if empty_t2g.any():
                    acc_t = _orbital_matrices(empty_t2g, _unit, u, a_sigma, t_pi)
                    for rule, k in T_EE_T2G.items():
                        T[k] += weight * sw[rule] * acc_t
                    T[T_KK] += weight * kk * acc_t
        return T

    # The cage (``amplitude_head.site_environment``): ln(CN / 6) and ln(N_sigma / 3), CN counted against the
    # octahedral (Shannon VI) reference length. R0 = that length + BOND_VALENCE_B ln(CN / 6).
    from quick_mag.amplitude_head import site_environment

    envs = {site: site_environment(structure, site, d.element, end_fractions(d), d.spin_state, "oct",
                                   cutoff=anion_bond_cutoff)
            for site, d in descriptors.items()}

    def r0_end(desc, site, ligand):
        return (r0_metal_ligand_averaged(desc.element, end_fractions(desc), desc.spin_state, ligand, "oct")
                + BOND_VALENCE_B * float(envs[site][0]))

    if de_pairs is None:
        de_pairs = de_active_pairs(
            structure, {d.element for d in descriptors.values()}
        )
    symbols = structure.element_symbols()
    anion_indices = [
        i for i, symbol in enumerate(symbols) if symbol in ANION_SPECIES
    ]

    bridges: List[BridgeGeometry] = []
    for ligand_index in anion_indices:
        center_x = structure.cartesian_coords[ligand_index]
        neighbors = [
            nbr
            for nbr in structure.neighbors(ligand_index, anion_bond_cutoff)
            if nbr.index in descriptors
        ]
        for k1 in range(len(neighbors)):
            for k2 in range(k1 + 1, len(neighbors)):
                nbr_i, nbr_j = neighbors[k1], neighbors[k2]
                r_i, r_j = float(nbr_i.nn_distance), float(nbr_j.nn_distance)
                if r_i + r_j > super_cutoff or r_i < 1e-8 or r_j < 1e-8:
                    continue
                u_i = (nbr_i.coords - center_x) / r_i
                u_j = (nbr_j.coords - center_x) / r_j
                frame = bridge_frame(u_i, u_j)
                basis_i = bases.get(nbr_i.index)
                basis_j = bases.get(nbr_j.index)
                bsig_i, bpi_i, bse_i, bpe_i = _end_channels(nbr_i.index, u_i, frame)
                bsig_j, bpi_j, bse_j, bpe_j = _end_channels(nbr_j.index, u_j, frame)
                # Open wherever either end has an empty majority orbital to accept into.
                oe_active = bool(
                    np.abs(bse_i).sum() + np.abs(bpe_i).sum()
                    + np.abs(bse_j).sum() + np.abs(bpe_j).sum() > 1e-12
                )
                desc_i = descriptors[nbr_i.index]
                desc_j = descriptors[nbr_j.index]
                ligand = symbols[ligand_index]
                de_active = frozenset((desc_i.element, desc_j.element)) in de_pairs
                # A carrier can only hop between ends of *different* valence, and
                # the hop needs both the configuration it leaves and the one it
                # makes: (i^a, j^b) <-> (i^b, j^a). With p and q the state
                # fractions of the two ends' site classes, the weight is
                # sum_{a != b} sqrt(p_a q_b * p_b q_a); for two ends of one class
                # that is 1 - sum_a p_a^2, the chance the two ends differ. Across
                # classes it vanishes when either end is single-valent -- magnetite's
                # tetrahedral Fe3+ do not join the octahedral Fe2+/Fe3+ hopping.
                # (A cross-element
                # resonant pair keeps full weight -- "differing valence" says nothing
                # across two elements.) An element the composition forces mixed but
                # whose sites have all been overridden to one state drops to zero,
                # which is the right answer for the structure as it now stands.
                de_weight = 1.0
                if de_active and desc_i.element == desc_j.element:
                    de_weight = _de_resonance_weight(
                        fractions[_site_class(desc_i)], fractions[_site_class(desc_j)]
                    )
                bridges.append(
                    BridgeGeometry(
                        site_i=nbr_i.index,
                        site_j=nbr_j.index,
                        ligand_index=ligand_index,
                        metal_i=desc_i.element,
                        metal_j=desc_j.element,
                        ligand=ligand,
                        r_iL=r_i,
                        r_jL=r_j,
                        R0_iL=r0_end(desc_i, nbr_i.index, ligand),
                        R0_jL=r0_end(desc_j, nbr_j.index, ligand),
                        Bsig_i=bsig_i,
                        Bpi_i=bpi_i,
                        Bsig_j=bsig_j,
                        Bpi_j=bpi_j,
                        u_iL=u_i,
                        u_jL=u_j,
                        frame=frame,
                        basis_i=basis_i,
                        basis_j=basis_j,
                        de_active=de_active,
                        de_weight=de_weight,
                        oe_active=oe_active,
                        Bsig_emp_i=bse_i,
                        Bpi_emp_i=bpe_i,
                        Bsig_emp_j=bse_j,
                        Bpi_emp_j=bpe_j,
                        T_i=_end_tensor(nbr_i.index, u_i),
                        T_j=_end_tensor(nbr_j.index, u_j),
                    )
                )
    _attach_direct_interference(bridges, descriptors, structure=structure, ligand_cutoff=anion_bond_cutoff,
                                charges=charges)
    from quick_mag.amplitude_head import ion_features

    # Like everything else at a bridge end, the features follow the end's states: the site's own
    # with valence="site", the site class's fractions otherwise (so which atom of a mixed-valent
    # class carries which label cannot matter).
    feats = {}
    for site, d in descriptors.items():
        feats[site] = sum(
            weight * wc * ion_features(state_descriptors[(d.element, state, d.spin_state, coord)], envs[site])
            for state, weight in end_fractions(d)
            for coord, wc in coords[site]
        )
    from quick_mag.element_data import ionization_energies

    ies = {}
    for site, d in descriptors.items():
        ladder = ionization_energies(d.element)
        don = acc = q = 0.0
        for state, weight in end_fractions(d):
            state = int(state)
            don += weight * float(ladder[state] if state < len(ladder) and ladder[state] is not None else 1e3)
            acc += weight * float(ladder[state - 1] if 0 < state <= len(ladder) and ladder[state - 1] is not None else 0.0)
            q += weight * state
        ies[site] = (don, acc, q)
    for bridge in bridges:
        object.__setattr__(bridge, "feat_i", feats[bridge.site_i])
        object.__setattr__(bridge, "feat_j", feats[bridge.site_j])
        object.__setattr__(bridge, "ie_i", ies[bridge.site_i])
        object.__setattr__(bridge, "ie_j", ies[bridge.site_j])
    return bridges


def _attach_direct_interference(bridges: List[BridgeGeometry], descriptors: Dict[int, IonDescriptor],
                                cutoff: float = 3.6, *, structure=None, ligand_cutoff: float = 3.0,
                                charges=None) -> None:
    """Fill ``G_dx`` on bridges whose metals are within ``cutoff``: each site in its own state (as ``dd_geometry``),
    mixed over its level orderings (``site_orbital_components``; bilinear, so over the combinations). Without a
    structure each site uses the bridge's recorded orbitals and its descriptor's ordering."""
    from quick_mag.direct_exchange import _global_tensors, pair_overlap_matrix
    from quick_mag.radial_overlap import slater_zeta

    cache: Dict[int, tuple] = {}

    def site(idx, rot):
        if idx not in cache:
            d = descriptors[idx]
            if structure is not None:
                comps = site_orbital_components(structure, idx, d, ligand_cutoff=ligand_cutoff, charges=charges)
            else:
                comps = [(1.0, occupancy_vector(d), rot)]
            cache[idx] = ([(w, m, b, _global_tensors(b)) for w, m, b in comps],
                          slater_zeta(d.element, int(d.oxidation_state), "d"))
        return cache[idx]

    for b in bridges:
        vec = b.u_jL * b.r_jL - b.u_iL * b.r_iL
        if np.linalg.norm(vec) > cutoff:
            continue
        comps_i, z_i = site(b.site_i, b.basis_i)
        comps_j, z_j = site(b.site_j, b.basis_j)
        G = np.zeros(4)
        for w_i, m_i, rot_i, Q_i in comps_i:
            for w_j, m_j, rot_j, Q_j in comps_j:
                if not m_i.any() or not m_j.any():
                    continue
                S = pair_overlap_matrix(Q_i, Q_j, z_i, z_j, vec) * np.outer(m_i, m_j)
                a_i, t_i = sk_table.pd_amplitudes(b.u_iL, basis=rot_i)
                a_j, t_j = sk_table.pd_amplitudes(b.u_jL, basis=rot_j)
                G += w_i * w_j * np.array([
                    float(a_i @ S @ a_j) * float(b.u_iL @ b.u_jL),
                    -float(a_i @ S @ (t_j @ b.u_iL)),
                    -float((t_i @ b.u_jL) @ S @ a_j),
                    float(np.einsum("ax,ab,bx->", t_i, S, t_j)),
                ])
        if np.any(G):
            object.__setattr__(b, "G_dx", G)


# ---------------------------------------------------------------------------
# mu and J assembly
# ---------------------------------------------------------------------------


def end_damping(params: PolarizationParameters, metal: str, r, R0, ln_delta=0.0):
    """The range factor (R0 / r)^n of one bridge end, the same for its sigma and pi channels (scalars or arrays)."""
    return (np.asarray(R0) / np.asarray(r)) ** range_exponent(params, metal, ln_delta)


def oe_transfer_energy(params: PolarizationParameters, donor, acceptor):
    """U of one electron from ``donor`` to ``acceptor`` (each ``(IE_{q+1}, IE_q, q)``, scalars or arrays), eV."""
    return donor[0] - acceptor[1] - params.oe_charge * (donor[2] - acceptor[2] + 1.0)


def range_exponent(params: PolarizationParameters, metal: str, ln_delta):
    """The power-law exponent n of one metal-ligand end; ``ln_delta`` = ln(Delta / 4 eV)."""
    if params.delta_sat > 0.0:
        from quick_mag.amplitude_head import DELTA_REF

        d2 = (DELTA_REF * np.exp(ln_delta)) ** 2
        return params.n_min + (params.n_max - params.n_min) * d2 / (d2 + params.delta_sat ** 2)
    return params.get_range_power(metal) + params.n_delta * ln_delta


def _ln_delta(features) -> float:
    """The ``ln_delta`` entry of an end's head features (0 without features)."""
    if features is None:
        return 0.0
    from quick_mag.amplitude_head import FEATURES

    return float(features[FEATURES.index("ln_delta")])


def bridge_J_components(
    bridge: BridgeGeometry, params: PolarizationParameters
) -> Tuple[float, float, float]:
    """One bridge's coupling split into its three channels (J > 0 AFM).

    Returns ``(J_SE, J_DE, J_OE)``:

    - ``J_SE``, superexchange (Terms A/B), carried by every bridge: ``kernel = "coherent"``
      ``2 (w + J_H) tr(P_i P_j) - 2 J_H tr P_i tr P_j``, ``"diagonal"`` the same on ``mu``. With
      ``k_ee > 0`` it also holds the empty-empty term ``2 (w + J_H) k_ee^2 tr(E_i E_j)`` (AFM), ``E`` the
      polarization of each end's empty majority-spin orbitals weighted by its unpaired electrons: two Cr3+
      on a 180 degree oxygen each take their own majority spin from the p_sigma orbital when antiparallel.
    - ``J_DE``, on DE-active bridges an Anderson-Hasegawa double-exchange term
      ``-tau_i * tau_j * f_i * f_j * cos^2(theta) * de_weight`` (FM; sigma-carrier
      transfer maximal at 180 degrees, zero at 90), scaled by the fraction of the
      ends' valence combinations a carrier can actually hop between.
    - ``J_OE``, on occupied->empty-active bridges (an empty majority-spin orbital
      on either end: eg^1 Mn3+'s CF-empty eg, Cr3+/Mn4+'s empty eg) a
      Kugel-Khomskii / Kanamori FM channel ``-j_fm_i j_fm_j (mu_occ_i . mu_emp_j +
      mu_emp_i . mu_occ_j)`` (an electron hopping into a neighbour's empty
      orbital Hund-aligns with that site's core -> FM; quadratic in the
      intensities, like Term A; donor and acceptor must share a ligand p orbital,
      so for a t2g donor it is largest at 90 degrees and vanishes at 180).

    The two FM channels are product-rule combined, ``x_i * x_j`` for the
    per-element factors, so an unparameterized element drops the channel to zero.
    Split out rather than summed on the spot because the UI plots the three
    against their total: which mechanism makes a pair FM is the question the
    coupling plot is there to answer.
    """
    ds_i = dp_i = end_damping(params, bridge.metal_i, bridge.r_iL, bridge.R0_iL, _ln_delta(bridge.feat_i))
    ds_j = dp_j = end_damping(params, bridge.metal_j, bridge.r_jL, bridge.R0_jL, _ln_delta(bridge.feat_j))
    damp_i, damp_j = ds_i, ds_j        # the sigma channel's: double exchange is a sigma-carrier hop
    if (params.head_sigma or params.head_pi) and bridge.feat_i is not None:
        from quick_mag.amplitude_head import head_factors

        hs_i, hp_i = head_factors(params, bridge.feat_i)
        hs_j, hp_j = head_factors(params, bridge.feat_j)
        ds_i, dp_i, ds_j, dp_j = ds_i * hs_i, dp_i * hp_i, ds_j * hs_j, dp_j * hp_j
    g2 = params.gamma_pi ** 2
    if params.kernel not in ("coherent", "diagonal"):
        raise ValueError(f"kernel must be coherent or diagonal, got {params.kernel!r}")
    have_T = bridge.T_i is not None and bridge.T_j is not None
    coherent = params.kernel == "coherent" and have_T
    w = params.get_w(bridge.ligand)
    jh = params.get_jh(bridge.ligand)
    if coherent:
        P_i = end_polarization(bridge.T_i[0], ds_i, g2 * dp_i)
        P_j = end_polarization(bridge.T_j[0], ds_j, g2 * dp_j)
        j_se = 2.0 * (w + jh) * float(np.sum(P_i * P_j)) - 2.0 * jh * float(np.trace(P_i) * np.trace(P_j))
    else:
        mu_i = ds_i * bridge.Bsig_i + g2 * dp_i * bridge.Bpi_i
        mu_j = ds_j * bridge.Bsig_j + g2 * dp_j * bridge.Bpi_j
        dot = float(mu_i @ mu_j)
        j_se = 2.0 * (w + jh) * dot - 2.0 * jh * float(mu_i.sum() * mu_j.sum())
    if params.k_ee and have_T:
        # empty-empty: both ends' empty majority-spin orbitals on one ligand channel (AFM)
        ke = T_EE[params.ee_spin]
        E_i = end_polarization(bridge.T_i[ke], ds_i, g2 * dp_i, coherent)
        E_j = end_polarization(bridge.T_j[ke], ds_j, g2 * dp_j, coherent)
        j_se += 2.0 * (w + jh) * params.k_ee ** 2 * float(np.sum(E_i * E_j))
    if params.k_dx and bridge.G_dx is not None:
        fs_i, fp_i, fs_j, fp_j = ds_i, g2 * dp_i, ds_j, g2 * dp_j
        j_se += params.k_dx * float(bridge.G_dx @ np.sqrt([fs_i * fs_j, fs_i * fp_j, fp_i * fs_j, fp_i * fp_j]))
    j_de = 0.0
    if bridge.de_active:
        if params.de_form not in ("linear", "quadratic"):
            raise ValueError(f"de_form must be linear or quadratic, got {params.de_form!r}")
        t_pair = params.get_t_de(bridge.metal_i) * params.get_t_de(bridge.metal_j)
        if t_pair != 0.0 and params.de_form == "linear":
            j_de = -(t_pair * float(np.sqrt(ds_i * ds_j)) * abs(bridge.cos_theta) * bridge.de_weight)
        elif t_pair != 0.0:
            j_de = -(
                t_pair * damp_i * damp_j * bridge.cos_theta ** 2 * bridge.de_weight
            )
    j_oe = 0.0
    if bridge.oe_active:
        j_fm = params.get_j_fm(bridge.metal_i) * params.get_j_fm(bridge.metal_j)
        manual = any(v is not None for v in (bridge.Bsig_occ_i, bridge.Bpi_occ_i, bridge.Bsig_occ_j, bridge.Bpi_occ_j))
        if params.oe_rule not in ("product", "acceptor", "ionization"):
            raise ValueError(f"oe_rule must be product, acceptor or ionization, got {params.oe_rule!r}")
        if params.oe_rule == "ionization":
            if have_T and params.k_oe:
                H_i = end_polarization(bridge.T_i[0], ds_i, g2 * dp_i, coherent)
                H_j = end_polarization(bridge.T_j[0], ds_j, g2 * dp_j, coherent)
                A_i = end_polarization(bridge.T_i[1], ds_i, g2 * dp_i, coherent)
                A_j = end_polarization(bridge.T_j[1], ds_j, g2 * dp_j, coherent)
                u_ij, u_ji = oe_transfer_energy(params, bridge.ie_i, bridge.ie_j), oe_transfer_energy(params, bridge.ie_j, bridge.ie_i)
                j_oe = -2.0 * (w + jh) * params.k_oe * float(
                    np.exp(-(u_ij - 3.0) / params.oe_decay) * np.sum(H_i * A_j)
                    + np.exp(-(u_ji - 3.0) / params.oe_decay) * np.sum(A_i * H_j))
        elif params.oe_rule == "acceptor":
            if have_T:
                H_i = end_polarization(bridge.T_i[0], ds_i, g2 * dp_i, coherent)
                H_j = end_polarization(bridge.T_j[0], ds_j, g2 * dp_j, coherent)
                X_i = (params.get_j_fm(bridge.metal_i) * end_polarization(bridge.T_i[1], ds_i, g2 * dp_i, coherent)
                       + params.k_oe * end_polarization(bridge.T_i[T_EE[params.ee_spin]], ds_i, g2 * dp_i, coherent))
                X_j = (params.get_j_fm(bridge.metal_j) * end_polarization(bridge.T_j[1], ds_j, g2 * dp_j, coherent)
                       + params.k_oe * end_polarization(bridge.T_j[T_EE[params.ee_spin]], ds_j, g2 * dp_j, coherent))
                j_oe = -2.0 * (w + jh) * float(np.sum(H_i * X_j) + np.sum(X_i * H_j))
        elif j_fm != 0.0 and coherent and not manual:
            A_i = end_polarization(bridge.T_i[1], ds_i, g2 * dp_i)
            A_j = end_polarization(bridge.T_j[1], ds_j, g2 * dp_j)
            j_oe = -(j_fm * float(np.sum(P_i * A_j) + np.sum(A_i * P_j)))
        elif j_fm != 0.0:
            bsig_occ_i = bridge.Bsig_i if bridge.Bsig_occ_i is None else bridge.Bsig_occ_i
            bpi_occ_i = bridge.Bpi_i if bridge.Bpi_occ_i is None else bridge.Bpi_occ_i
            bsig_occ_j = bridge.Bsig_j if bridge.Bsig_occ_j is None else bridge.Bsig_occ_j
            bpi_occ_j = bridge.Bpi_j if bridge.Bpi_occ_j is None else bridge.Bpi_occ_j
            mu_occ_i = ds_i * bsig_occ_i + g2 * dp_i * bpi_occ_i
            mu_occ_j = ds_j * bsig_occ_j + g2 * dp_j * bpi_occ_j
            mu_emp_i = ds_i * bridge.Bsig_emp_i + g2 * dp_i * bridge.Bpi_emp_i
            mu_emp_j = ds_j * bridge.Bsig_emp_j + g2 * dp_j * bridge.Bpi_emp_j
            j_oe = -(j_fm * float(mu_occ_i @ mu_emp_j + mu_emp_i @ mu_occ_j))
    if params.t2g_oe != "off" and have_T:
        j_oe += t2g_oe_term(params, w, jh, bridge.T_i, bridge.T_j, ds_i, g2 * dp_i, ds_j, g2 * dp_j, coherent)
    return j_se, j_de, j_oe


def t2g_oe_term(params: PolarizationParameters, w, jh, T_i, T_j, fs_i, fp_i, fs_j, fp_j, coherent=True):
    """The occupied -> empty t2g channel (``t2g_oe``; FM, <= 0) for one bridge or, with leading bridge axes, many."""
    if params.t2g_oe == "signed":
        k, c = T_EE_T2G[params.ee_spin], params.k_ee * params.k_t2g
    elif params.t2g_oe == "kk":
        k, c = T_KK, params.k_t2g
    else:
        raise ValueError(f"t2g_oe must be off, signed or kk, got {params.t2g_oe!r}")
    H_i = end_polarization(T_i[..., T_HALF, :, :, :], fs_i, fp_i, coherent)
    H_j = end_polarization(T_j[..., T_HALF, :, :, :], fs_j, fp_j, coherent)
    X_i = end_polarization(T_i[..., k, :, :, :], fs_i, fp_i, coherent)
    X_j = end_polarization(T_j[..., k, :, :, :], fs_j, fp_j, coherent)
    return -2.0 * (w + jh) * c * (np.einsum("...pq,...pq->...", H_i, X_j) + np.einsum("...pq,...pq->...", X_i, H_j))


def bridge_J(bridge: BridgeGeometry, params: PolarizationParameters) -> float:
    """Signed coupling of one bridge (J > 0 AFM): the sum of its three channels.

    See ``bridge_J_components`` for what those channels are.
    """
    j_se, j_de, j_oe = bridge_J_components(bridge, params)
    return j_se + j_de + j_oe


def build_Jeff_matrix(
    bridges: List[BridgeGeometry],
    site_index: Dict[int, int],
    params: PolarizationParameters,
) -> np.ndarray:
    """Symmetric J matrix over compact magnetic-site indices (J > 0 AFM).

    Model energy of a collinear configuration: ``E = +1/2 sigma^T J sigma``.
    """
    n = len(site_index)
    matrix = np.zeros((n, n))
    for bridge in bridges:
        if bridge.site_i not in site_index or bridge.site_j not in site_index:
            continue
        value = bridge_J(bridge, params)
        a, b = site_index[bridge.site_i], site_index[bridge.site_j]
        matrix[a, b] += value
        matrix[b, a] += value
    return matrix


@dataclass(frozen=True)
class PairCoupling:
    """The total coupling between one pair of metal sites, with its provenance.

    ``build_Jeff_matrix`` sums the bridges of a pair into a single matrix element
    and throws away everything that says *why*. This keeps that: which elements
    are coupled, how far apart they are, and through how many ligands at what
    angles -- enough to label and explain a bar in the UI's coupling plot.

    ``j_eff`` is in the model convention (J > 0 AFM), the same as the matrix.
    """

    site_i: int  # structure atom index; site_i < site_j
    site_j: int
    metal_i: str
    metal_j: str
    j_eff: float
    distance: float  # metal-metal distance, Angstrom
    bridge_count: int
    ligands: Tuple[str, ...]
    angles_deg: Tuple[float, ...]  # M-L-M angle, one per bridge
    ligand_indices: Tuple[int, ...] = ()
    #: ``bridge_J_components`` summed over the pair's bridges, so the three add up
    #: to ``j_eff`` (up to float rounding: ``j_eff`` is accumulated bridge by
    #: bridge, exactly as the matrix element is). Defaulted so a caller that only
    #: cares about the total can still build one.
    j_se: float = 0.0  # superexchange; the only channel that can be AFM
    j_de: float = 0.0  # double exchange (<= 0, FM)
    j_oe: float = 0.0  # occupied->empty Kugel-Khomskii (<= 0, FM)
    j_dd: float = 0.0  # direct d-d exchange (>= 0, AFM); only with pair_couplings(..., model=)
    #: Cartesian M-L-M paths, ``(bridge_count, 3, 3)``, ordered
    #: ``[site_i, ligand, site_j]``. Empty when built without coordinates. These are
    #: the *image* positions the bridge actually used, so a path that crosses a cell
    #: boundary is contiguous rather than leaping back across the cell.
    paths: np.ndarray = None


def pair_couplings(
    bridges: List[BridgeGeometry],
    params: PolarizationParameters,
    cartesian_coords: Optional[np.ndarray] = None,
    *,
    model: Optional["ExchangeModel"] = None,
) -> List[PairCoupling]:
    """Per-pair couplings, summed over bridges exactly as ``build_Jeff_matrix`` does.

    Same accumulation, so ``j_eff`` here and the corresponding matrix element are
    the same number by construction rather than by coincidence. The three channels
    of ``bridge_J_components`` are accumulated alongside it, so a pair carries not
    just how strongly it couples but which mechanism did it.

    The geometry is read off the bridge instead of being recomputed from the
    structure: ``u_iL`` and ``u_jL`` both point outwards from the ligand, so
    ``cos_theta`` is the cosine of the M-L-M angle, the law of cosines closes the
    triangle for the metal-metal distance, and stepping ``r_iL`` and ``r_jL`` back
    along those directions recovers the two metals' positions -- as *images*, which
    is what makes a path across a cell boundary contiguous. Pairs bridged more than
    once give the same distance from every bridge, up to rounding; the first is kept.

    ``cartesian_coords`` is the structure's coordinates, needed only to place the
    ligand that anchors each path. Without it the couplings still carry everything
    but ``paths``.

    ``model``: when given, each bridged pair also gets its direct exchange
    ``k_dd * model.dd`` (``j_dd``, added to ``j_eff``), so ``j_eff`` is again the
    model's matrix element. A pair close enough for direct exchange but sharing no
    anion is not listed (in oxides at the default 3.6 A cutoff there is none).
    """
    totals: Dict[Tuple[int, int], float] = {}
    components: Dict[Tuple[int, int], List[float]] = {}
    metals: Dict[Tuple[int, int], Tuple[str, str]] = {}
    distances: Dict[Tuple[int, int], float] = {}
    ligands: Dict[Tuple[int, int], List[str]] = {}
    ligand_indices: Dict[Tuple[int, int], List[int]] = {}
    angles: Dict[Tuple[int, int], List[float]] = {}
    paths: Dict[Tuple[int, int], List[np.ndarray]] = {}

    for bridge in bridges:
        forward = bridge.site_i < bridge.site_j
        key = (
            (bridge.site_i, bridge.site_j) if forward else (bridge.site_j, bridge.site_i)
        )
        j_se, j_de, j_oe = bridge_J_components(bridge, params)
        # The total is accumulated from the bridge's own sum, the same quantity
        # and in the same order as ``build_Jeff_matrix`` accumulates it, so
        # ``j_eff`` stays the matrix element bit for bit.
        totals[key] = totals.get(key, 0.0) + (j_se + j_de + j_oe)
        running = components.setdefault(key, [0.0, 0.0, 0.0])
        running[0] += j_se
        running[1] += j_de
        running[2] += j_oe
        cos_theta = min(max(bridge.cos_theta, -1.0), 1.0)
        if key not in metals:
            metals[key] = (
                (bridge.metal_i, bridge.metal_j)
                if forward
                else (bridge.metal_j, bridge.metal_i)
            )
            distances[key] = math.sqrt(
                max(
                    bridge.r_iL**2
                    + bridge.r_jL**2
                    - 2.0 * bridge.r_iL * bridge.r_jL * cos_theta,
                    0.0,
                )
            )
        ligands.setdefault(key, []).append(bridge.ligand)
        ligand_indices.setdefault(key, []).append(bridge.ligand_index)
        angles.setdefault(key, []).append(math.degrees(math.acos(cos_theta)))
        if cartesian_coords is not None:
            centre = np.asarray(cartesian_coords[bridge.ligand_index], dtype=np.float64)
            ends = (
                (centre + bridge.u_iL * bridge.r_iL, centre + bridge.u_jL * bridge.r_jL)
                if forward
                else (centre + bridge.u_jL * bridge.r_jL, centre + bridge.u_iL * bridge.r_iL)
            )
            paths.setdefault(key, []).append(
                np.array([ends[0], centre, ends[1]], dtype=np.float64)
            )

    j_dd: Dict[Tuple[int, int], float] = {}
    if model is not None and model.dd is not None and params.k_dd:
        idx = model.site_index
        for key in totals:
            if key[0] in idx and key[1] in idx:
                j_dd[key] = params.k_dd * float(model.dd[idx[key[0]], idx[key[1]]])
                totals[key] += j_dd[key]
    return [
        PairCoupling(
            site_i=key[0],
            site_j=key[1],
            metal_i=metals[key][0],
            metal_j=metals[key][1],
            j_eff=totals[key],
            distance=distances[key],
            bridge_count=len(ligands[key]),
            ligands=tuple(ligands[key]),
            angles_deg=tuple(angles[key]),
            ligand_indices=tuple(ligand_indices[key]),
            j_se=components[key][0],
            j_de=components[key][1],
            j_oe=components[key][2],
            j_dd=j_dd.get(key, 0.0),
            paths=(
                np.array(paths[key], dtype=np.float64)
                if key in paths
                else np.zeros((0, 3, 3), dtype=np.float64)
            ),
        )
        for key in totals
    ]


def to_solver_couplings(j_eff: np.ndarray) -> np.ndarray:
    """Convert to the spin-solver convention.

    ``spin_solver`` minimizes ``H = -1/2 sum_ij J_ij m_i m_j`` with J > 0 FM,
    so hand it ``-J_eff`` and feed it UNIT moments (+-1). Spin magnitude is
    already inside mu.

    Prefer :func:`exchange_model`, whose ``J`` is per moment product and runs on the
    actual formal moments.
    """
    return -np.asarray(j_eff)


# ---------------------------------------------------------------------------
# Heisenberg form on actual moments
# ---------------------------------------------------------------------------


def heisenberg_couplings(j_eff: np.ndarray, moments: np.ndarray) -> np.ndarray:
    """``J_eff`` (unit-spin form, J > 0 AFM) rewritten per moment product.

    ``Jt_ij = J_eff_ij / (m_i m_j)`` with ``m`` the formal moments (mu_B), so

        E = 1/2 sum_ij J_eff_ij s_i s_j = 1/2 sum_ij Jt_ij (m_i s_i)(m_j s_j)

    for every sign pattern ``s`` -- the same energies, now on physical moments.
    Every channel's spin dependence is already inside ``J_eff`` (the occupancies of
    the polarization intensities, low spin included), so this is a change of
    convention, exact by construction: it moves the spin magnitude out of the
    coupling and onto the moment, which is where the Heisenberg constants of DFT
    mapping and of experiment put it. A pair with a zero moment on either end gets
    ``Jt = 0`` (it couples to nothing).
    """
    j = np.asarray(j_eff, dtype=float)
    m = np.asarray(moments, dtype=float).reshape(-1)
    if j.shape != (m.size, m.size):
        raise ValueError(f"J shape {j.shape} does not match {m.size} moments.")
    mm = np.outer(m, m)
    out = np.zeros_like(j)
    nz = np.abs(mm) > 1e-12
    out[nz] = j[nz] / mm[nz]
    return out


@dataclass
class ExchangeModel:
    """The exchange model of one structure under one oxidation/spin-state assignment.

    ``sites`` are the magnetic (descriptor) atom indices, in the order of every
    matrix and vector here. ``J`` is the Heisenberg coupling per moment product in
    the model's sign convention (J > 0 AFM; the parameters' energy unit, nominally
    eV, per mu_B^2); ``moments`` the formal spin-only moments ``2S`` (mu_B)
    of each site's descriptor. Energy of a collinear configuration with signs ``s``:

        E = 1/2 sum_ij J_ij (m_i s_i)(m_j s_j)          (:meth:`energy`)

    ``solver_J`` is the same in ``spin_solver``'s convention (J > 0 FM), to be fed
    the signed moments ``moments * s`` -- not unit spins.
    """

    sites: List[int]
    descriptors: Dict[int, IonDescriptor]
    bridges: List[BridgeGeometry]
    j_eff: np.ndarray          # unit-spin form (J > 0 AFM), the old convention
    J: np.ndarray              # per moment product (J > 0 AFM)
    moments: np.ndarray        # formal |m| per site (mu_B)
    # Direct-exchange geometry G (``direct_exchange.dd_geometry``, sites order): j_eff includes k_dd * G.
    dd: Optional[np.ndarray] = None

    @property
    def solver_J(self) -> np.ndarray:
        return -self.J

    @property
    def site_index(self) -> Dict[int, int]:
        return {site: k for k, site in enumerate(self.sites)}

    def energy(self, signs) -> float:
        """Model energy of a collinear configuration (signs per site, in ``sites`` order)."""
        s = np.sign(np.asarray(signs, dtype=float).reshape(-1)) * self.moments
        return float(0.5 * s @ self.J @ s)


def exchange_model(
    structure,
    assignment,
    *,
    params: Optional[PolarizationParameters] = None,
    spin_states: Optional[Dict[int, str]] = None,
    valence: str = "class",
    descriptors: Optional[Dict[int, IonDescriptor]] = None,
) -> ExchangeModel:
    """Descriptors -> bridges -> couplings for one assignment, on actual moments.

    ``spin_states`` (atom index -> ``"HS"``/``"LS"``; default: the assignment's own, as
    ``site_states.ranked_assignments`` attaches them) and ``valence`` (see
    :func:`build_bridges`) are passed through; ``descriptors`` skips building them.
    A structure with no magnetic sites or no bridges gives empty matrices.
    """
    from quick_mag.ion_descriptors import formal_moments, structure_ion_descriptors

    if params is None:
        params = default_params()
    if spin_states is None:
        spin_states = getattr(assignment, "spin_states", None)
    if descriptors is None:
        descriptors = structure_ion_descriptors(structure, assignment, spin_states=spin_states)
    sites = sorted(descriptors)
    moment_map = formal_moments(descriptors)
    moments = np.array([moment_map[s] for s in sites], dtype=float)
    charges = model_charges(structure, descriptors, getattr(assignment, "site_oxidation_states", None), valence)
    bridges = build_bridges(structure, descriptors, valence=valence, charges=charges) if sites else []
    index = {site: k for k, site in enumerate(sites)}
    j_eff = build_Jeff_matrix(bridges, index, params) if bridges else np.zeros((len(sites),) * 2)
    from quick_mag.direct_exchange import dd_geometry

    dd = dd_geometry(structure, descriptors, sites, cutoff=params.dd_cutoff, charges=charges)
    if params.k_dd:
        j_eff = j_eff + params.k_dd * dd
    return ExchangeModel(
        sites=sites,
        descriptors=descriptors,
        bridges=bridges,
        j_eff=j_eff,
        J=heisenberg_couplings(j_eff, moments),
        moments=moments,
        dd=dd,
    )


# ---------------------------------------------------------------------------
# Parameter (de)serialization
# ---------------------------------------------------------------------------

#: The single bundled fitted + hand-calibrated parameter set (the only model).
DEFAULT_PARAMS_PATH = Path(__file__).with_name("exchange_params") / "polarization_params.json"


def params_from_dict(data: Dict) -> PolarizationParameters:
    """Build ``PolarizationParameters`` from a parsed JSON dict."""
    if "kappa" in data or "kappa_default" in data:
        raise ValueError(
            "this parameter file has per-element kappa, which the model no longer has: the scale now "
            "lives in w_ligand / jh_ligand and each ion's amplitude in head_sigma / head_pi. Refit it "
            "(quick_mag.fitting) or fold a single shared kappa k in by hand (w, J_H x k^2; j_fm x k)."
        )
    if data.get("range_model", "power") != "power":
        raise ValueError(
            f"this parameter file was fitted with range_model = {data['range_model']!r}; the model has the power "
            "law only (the exponential and Slater-overlap range laws were removed). Refit it (quick_mag.fitting)."
        )
    fields = {
        "w_ligand", "jh_ligand", "t_de", "j_fm", "gamma_pi",
        "w_default", "jh_default",
        "t_de_default", "j_fm_default",
        "range_power", "range_power_default",
        "head_sigma", "head_pi", "k_dd", "dd_cutoff", "kernel", "k_ee", "ee_spin", "t2g_oe", "k_t2g", "oe_rule", "k_oe", "k_dx", "n_delta", "delta_sat", "n_min", "n_max", "oe_decay", "oe_charge", "de_form",
    }
    values = {k: v for k, v in data.items() if k in fields}
    values.setdefault("kernel", "diagonal")       # files from before the coherent kernel were fitted with this one
    values.setdefault("de_form", "quadratic")     # likewise the double-exchange form
    values.setdefault("ee_spin", "n")             # and the empty-orbital spin weight
    values.setdefault("t2g_oe", "off")            # and no t2g acceptor
    return PolarizationParameters(**values)


def load_params(path) -> PolarizationParameters:
    """Load the fitted parameter set from a JSON file path."""
    return params_from_dict(json.loads(Path(path).read_text()))


def default_params() -> PolarizationParameters:
    """The bundled fitted + hand-calibrated default model parameters."""
    return load_params(DEFAULT_PARAMS_PATH)
