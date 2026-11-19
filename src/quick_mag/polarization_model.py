"""Exchange-polarization superexchange model (docs/theory/magnetism-model.md).

Each metal i polarizes the p channels of a bridging ligand L with intensity

    mu_{i->L,p} = sum_a mbar_{i,a} * [a_sigma,i d_sigma(r) W^sigma_{a,p} + gamma_pi^2 a_pi,i d_pi(r) W^pi_{a,p}]

where ``W`` are squared Slater-Koster pd channel weights (``sk_table``) evaluated
in a bridge-adapted orthonormal p frame, ``mbar_{i,a}`` is the shell-averaged net
unpaired spin per d orbital, ``d_m(r)`` the metal-ligand range factor of each channel
(``PolarizationParameters.range_model``; 1 at ``R0_iL``, the sum of Shannon crystal radii),
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

Orbital resolution: ``mbar_a`` is the per-orbital net unpaired spin from the Hund
shell filling (t2g H/3 on each t2g orbital, eg H/2 on each eg orbital), evaluated
in each site's own local frame (``local_d_frame``: the octahedral axes, or for a
tetrahedral site the axes of the cube its ligands sit on, where the filling order
of the two orbital sets is inverted -- see ``ion_descriptors``). This equals the
average over degenerate microstates, since J is bilinear in the occupancies. For
octahedral eg^1 sites the shell average is replaced by the *coherent* eg orbital
selected by a point-charge crystal field (``crystal_field_eg_orbital``), which
also opens the occupied->empty FM channel of ``bridge_J_components``; for eg^3
sites (Cu2+) by the orbital that holds the hole.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
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
# See ``crystal_field_eg_orbital``.
_CF_DEGENERATE_TOL = 1e-12


@dataclass
class PolarizationParameters:
    """Fit parameters of the polarization model.

    ``alpha`` holds per-element damping exponents for metals AND ligands;
    ``w_ligand`` is the ligand Pauli stiffness (it carries the overall scale: the
    metal amplitudes are 1 for a reference ion); ``jh_ligand`` the ligand Hund coupling; ``gamma_pi`` a global pi/sigma
    polarization amplitude ratio (1.0 = no hierarchy).
    """

    alpha: Dict[str, float] = field(default_factory=dict)
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
    gamma_pi: float = 1.0
    # Range model: how a metal's polarization of a ligand falls with the M-L distance r (it
    # is the squared M-L transfer, so 1 at the reference
    # distance R0 = the sum of Shannon crystal radii):
    #   "exp"    exp(-sqrt(alpha_M alpha_L) (r - R0))           (the original model; alpha fitted)
    #   "power"  (R0 / r)^n_M, n from ``range_power`` (Harrison's universal pd transfer
    #            t ~ r^-7/2 gives n = 7 for t^2)
    #   "sto"    (S_m(r) / S_m(R0))^2 per channel m = sigma, pi: overlaps of the metal-d and
    #            ligand-p Slater-type orbitals, exponents from Slater's rules for each ion
    #            (``radial_overlap``) -- frozen, nothing fitted
    range_model: str = "sto"
    range_power: Dict[str, float] = field(default_factory=dict)
    range_power_default: float = 7.0
    # Per-site amplitude head (``amplitude_head``): {feature: theta} for the sigma and pi amplitudes of an
    # end, a = exp(theta . phi(ion)). Empty = no head.
    head_sigma: Dict[str, float] = field(default_factory=dict)
    head_pi: Dict[str, float] = field(default_factory=dict)
    alpha_default: float = 2.0
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
    # in proportion k_ee * (unpaired electrons of the ion). Two such polarizations on one ligand channel couple
    # AFM: J_EE = 2 (w_L + J_H^L) k_ee^2 n_i n_j tr(P_i^E P_j^E), reported inside J_SE. 0 = off.
    k_ee: float = 0.0

    def get_alpha(self, element: str) -> float:
        return self.alpha.get(element, self.alpha_default)

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


def local_octahedral_frame(
    structure,
    site_index: int,
    *,
    ligand_cutoff: float = 3.0,
    min_pair_dot: float = -0.7,
) -> np.ndarray:
    """Local octahedral d-frame of a metal site from its ligand cage.

    Greedily pairs (near-)antiparallel M-L directions; each pair's difference
    vector is one octahedral axis. The stacked axes are symmetrized to the
    nearest orthonormal set (polar decomposition). Returns the rotation matrix
    whose *columns* are the local axes in global coordinates — the ``rotation``
    argument of ``sk_table.pd_amplitudes``.

    Axis labeling/sign is arbitrary, which is harmless: the shell-averaged
    occupancies weight all t2g (and all eg) orbitals equally, and those shell
    sums are invariant under relabeling the octahedral axes. Falls back to the
    identity (global frame) when fewer than two trans ligand pairs exist
    (non-octahedral or under-coordinated sites).
    """
    center = structure.cartesian_coords[site_index]
    directions: List[np.ndarray] = []
    for neighbor in structure.neighbors(site_index, ligand_cutoff):
        if neighbor.symbol not in ANION_SPECIES:
            continue
        vec = neighbor.coords - center
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            directions.append(vec / norm)

    axes: List[np.ndarray] = []
    used: set = set()
    while len(axes) < 3:
        best_pair = None
        best_dot = min_pair_dot
        for i in range(len(directions)):
            if i in used:
                continue
            for j in range(i + 1, len(directions)):
                if j in used:
                    continue
                dot = float(directions[i] @ directions[j])
                if dot < best_dot:
                    best_dot = dot
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        used.update(best_pair)
        axis = directions[i] - directions[j]
        axes.append(axis / np.linalg.norm(axis))

    if len(axes) < 2:
        return np.eye(3)
    if len(axes) == 2:
        third = np.cross(axes[0], axes[1])
        axes.append(third / np.linalg.norm(third))
    u_svd, _, vt = np.linalg.svd(np.array(axes))
    return (u_svd @ vt).T


def local_tetrahedral_frame(
    structure,
    site_index: int,
    *,
    ligand_cutoff: float = 3.0,
) -> np.ndarray:
    """Local cubic d-frame of a tetrahedral site from its four ligands.

    A tetrahedron's ligands sit on alternate corners of a cube, and the cube's
    axes are the frame in which (dz2, dx2-y2) form the e level and (dxy, dxz,
    dyz) the t2 level. Each of the three ways of splitting the four ligands into
    two pairs gives one axis, ``(u_a + u_b) - (u_c + u_d)`` (exactly a cube axis
    for an ideal tetrahedron); the three are symmetrized to the nearest
    orthonormal set by polar decomposition. Returns the rotation whose *columns*
    are the local axes in global coordinates, like ``local_octahedral_frame``.
    Axis labels and signs are arbitrary, which the shell-averaged occupancies do
    not see. Falls back to the identity when the site does not have exactly four
    ligands.
    """
    directions = site_ligand_directions(structure, site_index, ligand_cutoff=ligand_cutoff)
    if len(directions) != 4:
        return np.eye(3)
    u = directions
    axes = np.array(
        [
            (u[0] + u[1]) - (u[2] + u[3]),
            (u[0] + u[2]) - (u[1] + u[3]),
            (u[0] + u[3]) - (u[1] + u[2]),
        ]
    )
    norms = np.linalg.norm(axes, axis=1)
    if np.any(norms < 1e-8):
        return np.eye(3)
    axes = axes / norms[:, None]
    u_svd, _, vt = np.linalg.svd(axes)
    return (u_svd @ vt).T


def local_d_frame(
    structure,
    site_index: int,
    coordination: str = "oct",
    *,
    ligand_cutoff: float = 3.0,
) -> np.ndarray:
    """The site's local d-orbital frame for its coordination (see ``IonDescriptor``)."""
    if coordination == "tet":
        return local_tetrahedral_frame(structure, site_index, ligand_cutoff=ligand_cutoff)
    return local_octahedral_frame(structure, site_index, ligand_cutoff=ligand_cutoff)


def is_eg_active(descriptor: IonDescriptor) -> bool:
    """True when the eg shell is singly occupied or singly holed (orbital freedom).

    ``n_eg in {1, 3}`` — one eg electron (e.g. Mn3+ d4 HS) or one eg hole (d9) —
    is the Jahn-Teller / orbital-ordering case. ``n_eg`` of 0 (empty) or 2 (full)
    has no eg orbital degree of freedom. Octahedral sites only: the
    crystal-field orbital selection behind it is an octahedral construction.
    """
    return descriptor.coordination == "oct" and descriptor.n_eg in (1, 3)


def is_occ_empty_active(descriptor: IonDescriptor) -> bool:
    """True for an eg^1 ion with a Hund-active core (the occupied->empty FM case).

    One eg electron and one empty eg orbital, plus a spin-polarized core (t2g here)
    for the hopped electron to Hund-align with. An electron hopping into a
    neighbour's empty eg orbital then lowers energy for parallel spins -> FM.
    Excludes eg^3 (a lone hole; filling it leaves no other unpaired spin, so no
    Hund bonus — the FM channel is not electron-hole symmetric; see
    docs/theory/magnetism-model.md). Octahedral sites only, like ``is_eg_active``.

    This is the *coherent* acceptor case (one definite CF-empty eg orbital); every
    other ion accepts through its shell-averaged empty orbitals (``empty_vector``).
    """
    return descriptor.coordination == "oct" and descriptor.n_eg == 1


def is_eg_hole(descriptor: IonDescriptor) -> bool:
    """True for an octahedral eg^3 ion (Cu2+ d9): one eg orbital is full, the other holds the
    hole and is the only half-filled orbital, so which one it is decides every coupling
    (x^2-y^2 in a z-elongated cage: strong in plane, none along z). ``build_bridges`` takes it
    from ``crystal_field_eg_orbital(hole=True)``; on an undistorted cage the shell average stays."""
    return descriptor.coordination == "oct" and descriptor.n_eg == 3


def eg_orbital_director(
    structure, site_index: int, *, ligand_cutoff: float = 3.0
) -> Optional[np.ndarray]:
    """Unit vector along the Jahn-Teller-elongated octahedral axis of a site.

    For an eg^1 ion the occupied eg orbital (a d(3z^2-r^2)-type lobe) points along
    the *long* metal-ligand bonds, so the longest bond direction is the orbital
    director. The structural distortion in a relaxed geometry therefore encodes
    the orbital order directly — no orbital relaxation needed. Sign is irrelevant
    downstream (only ``(director . bond)^2`` is used). Returns None if the site
    has no ligand neighbours.

    Heuristic; superseded by ``crystal_field_eg_orbital`` (a k=2 point-charge
    crystal-field integral) which is continuous, balances all bonds, and vanishes
    for an undistorted octahedron. See docs/theory/magnetism-model.md.
    """
    center = structure.cartesian_coords[site_index]
    bonds = [
        (float(nbr.nn_distance), nbr.coords - center)
        for nbr in structure.neighbors(site_index, ligand_cutoff)
        if nbr.symbol in ANION_SPECIES
    ]
    if not bonds:
        return None
    _, longest = max(bonds, key=lambda b: b[0])
    return longest / np.linalg.norm(longest)


def crystal_field_eg_orbital(
    structure,
    site_index: int,
    rotation: np.ndarray,
    *,
    hole: bool = False,
    ligand_cutoff: float = 3.0,
    charges: Optional[Dict[str, int]] = None,
) -> Optional[np.ndarray]:
    """Occupied (or hole) eg orbital of a site from a k=2 point-charge crystal field.

    Builds the ligand-field tensor ``F = sum_L (q_L / R_L^3)(3 u u^T - I)/2`` in the
    site's local octahedral frame (``rotation``, local->global) and returns the
    coefficient 2-vector on ``(dz2, dx2-y2)`` of the lower eg eigenvector
    (``hole=False``, eg^1 electron) or the upper one (``hole=True``, eg^3 hole).
    Returns None if the site has no ligand neighbours, or if its cage is
    undistorted (see below) — either way the caller falls back to the shell average.

    The eg block is closed form. The k=2 crystal-field matrix element is
    ``H_ab = int (n^T Q_a n)(n^T F n)(n^T Q_b n) dOmega``, a degree-6 polynomial in
    n whose sphere average is exact via the isotropic 6th-moment tensor; since
    ``Q_a``, ``Q_b`` and ``F`` are all symmetric traceless, every pairing with an
    intra-tensor contraction drops out and the eight surviving ones each give the
    same trace:

        <H_ab> = (8/105) tr(Q_a F Q_b)

    Both eg tensors are diagonal, so only ``diag(F)`` survives the trace and (using
    ``tr F = 0``) the eg block reduces, up to the positive constant, to

        [[p, q], [q, -p]],   p = F_zz,   q = (F_yy - F_xx)/sqrt(3)

    whose eigenvectors are the half-angle pair of ``phi = atan2(q, p)``. ``phi`` is
    the usual eg pseudospin angle: ``phi = pi`` is a z-elongated site (occupied
    orbital dz2), ``phi = 0`` a z-compressed one (dx2-y2).

    The k=2 field vanishes for an ideal octahedron, so this is exactly the
    distortion-induced orbital order and ``r = hypot(p, q)`` is the splitting. At
    ``r = 0`` the two eg orbitals are degenerate and there is no orbital order to
    report, so this returns None rather than an arbitrary member of the degenerate
    pair: ``build_bridges`` then leaves the site on its shell-averaged occupancy with
    the FM channel off, which is how every non-orbital ion class is already handled.
    See docs/theory/magnetism-model.md.
    """
    charges = ANION_CHARGES if charges is None else charges
    center = structure.cartesian_coords[site_index]
    F = np.zeros((3, 3))
    have = False
    for nbr in structure.neighbors(site_index, ligand_cutoff):
        if nbr.symbol not in ANION_SPECIES:
            continue
        d = nbr.coords - center
        r = float(np.linalg.norm(d))
        if r < 1e-8:
            continue
        u = d / r
        q = abs(charges.get(nbr.symbol, -2))
        F += (q / r ** 3) * (3.0 * np.outer(u, u) - np.eye(3)) / 2.0
        have = True
    if not have:
        return None
    rotation = np.asarray(rotation)
    f_xx, f_yy, f_zz = np.diag(rotation.T @ F @ rotation)
    p = float(f_zz)
    q = float(f_yy - f_xx) / math.sqrt(3.0)
    if math.hypot(p, q) < _CF_DEGENERATE_TOL:
        return None                             # degenerate: no orbital order
    half = 0.5 * math.atan2(q, p)
    if hole:
        return np.array([math.cos(half), math.sin(half)])
    return np.array([-math.sin(half), math.cos(half)])


def empty_vector(descriptor: IonDescriptor) -> np.ndarray:
    """Shell-averaged empty fraction per d orbital, ordered as sk_table.D_ORBITALS.

    E_t2g/3 on each t2g orbital and E_eg/2 on each eg orbital (high spin fills
    every orbital singly first, so these are the empty *majority-spin* orbitals):
    the acceptor side of the occupied->empty channel for any ion that is not eg^1
    (Cr3+ -> [0, 0, 0, 1, 1]; Fe3+ -> 0).
    """
    m = np.zeros(5)
    m[list(T2G_INDICES)] = descriptor.ehf_t2g[0] / 3.0
    m[list(EG_INDICES)] = descriptor.ehf_eg[0] / 2.0
    return m


def coherent_eg_intensity(
    coeff: np.ndarray,
    u: np.ndarray,
    frame: np.ndarray,
    rotation: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(Bsig, Bpi) channel intensities of one *coherent* d orbital (coeff on the
    5 real d orbitals, local frame), for the occupied->empty channel.

    Unlike ``_end_intensities`` (an incoherent per-orbital sum over occupancies),
    this squares the *coherent* amplitude of a single superposition orbital, so the
    cross-terms that carry orbital directionality are retained:

        Bsig[p] = (A_sigma^psi * (e_p . u))^2 ,  Bpi[p] = ((sum_a c_a t_pi_a) . e_p)^2
    """
    a_sigma, t_pi = sk_table.pd_amplitudes(u, rotation=rotation)
    coeff = np.asarray(coeff, dtype=float)
    a_sig_psi = float(coeff @ a_sigma)
    t_pi_psi = coeff @ t_pi                       # (3,)
    c = np.asarray(frame) @ np.asarray(u)         # (3,) components of u on frame
    return (a_sig_psi * c) ** 2, (t_pi_psi @ np.asarray(frame).T) ** 2


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

    This averages the radius, not the damping ``exp(-alpha (r - R0))`` it feeds --
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


def occupancy_vector(descriptor: IonDescriptor) -> np.ndarray:
    """Shell-averaged net unpaired spin per d orbital, ordered as sk_table.D_ORBITALS.

    H_t2g/3 on each t2g orbital and H_eg/2 on each eg orbital. Only singly occupied
    orbitals carry net spin, so this is the exact average of "is orbital a singly
    occupied?" over the degenerate Hund microstates in
    ``descriptor.microstates`` — closed form, which is why those microstates are
    never enumerated here (Mn3+ -> [1, 1, 1, 0.5, 0.5]).

    The average is spherically symmetric within each shell, so the couplings it
    feeds carry no orbital-order information. For eg^1 Hund-active-core sites
    ``build_bridges`` replaces the averaged eg part with the *coherent* orbital
    picked out by the crystal field (``crystal_field_eg_orbital``); elsewhere
    replacing this with a non-uniform weighting over ``descriptor.microstates`` is
    the intended extension point if orbital order should come from the electronic
    structure rather than the geometry.
    """
    m = np.zeros(5)
    m[list(T2G_INDICES)] = descriptor.ehf_t2g[1] / 3.0
    m[list(EG_INDICES)] = descriptor.ehf_eg[1] / 2.0
    return m


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
    # Occupied-orbital channel intensities (per bridge frame, (3,)). For an eg^1
    # Hund-active-core ion the eg part is the *coherent* CF-ordered orbital (see
    # build_bridges); otherwise the shell-averaged occupancy.
    Bsig_i: np.ndarray
    Bpi_i: np.ndarray
    Bsig_j: np.ndarray
    Bpi_j: np.ndarray
    u_iL: np.ndarray = None
    u_jL: np.ndarray = None
    frame: np.ndarray = None
    rot_i: np.ndarray = None   # local d-frame rotations (None = global frame)
    rot_j: np.ndarray = None
    # True when the end elements form a DE-active pair (charge-balance-forced
    # mixed valence, same-element or cross-element resonant; see de_active_pairs).
    de_active: bool = False
    # Fraction of the end-state combinations that actually carry a hopping carrier:
    # for a same-element pair, the chance the two ends differ in valence. See
    # ``build_bridges``.
    de_weight: float = 1.0
    # Occupied->empty FM channel: open when either end has an empty majority-spin
    # orbital (the coherent CF-empty eg of an eg^1 state; the shell-averaged empty
    # orbitals of any other, e.g. Cr3+ eg), with those orbitals' channel intensities
    # on each end. ``bridge_J`` adds
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
    # Frozen Slater-type-orbital damping of each end, (sigma, pi): (S_m(r)/S_m(R0))^2.
    # Used when ``range_model == "sto"``; see ``radial_overlap``.
    sto_i: Tuple[float, float] = (1.0, 1.0)
    sto_j: Tuple[float, float] = (1.0, 1.0)
    # Centred atomic features of each end's ion (``amplitude_head.FEATURES``).
    feat_i: Optional[np.ndarray] = None
    feat_j: Optional[np.ndarray] = None
    # Polarization matrices of each end over the ligand p shell, global axes, unit amplitudes: ``T[set, part]``
    # (3, 3, 3, 3) with set = 0 half-filled orbitals, 1 empty majority-spin orbitals (the acceptor), 2 the same
    # weighted by each state's unpaired electrons (the empty-empty term); part = 0 sigma-sigma, 1 pi-pi,
    # 2 sigma-pi interference (sign of V_pdpi / V_pdsigma < 0 included). An end's matrix is
    # f_sigma T[s, 0] + f_pi T[s, 1] + sqrt(f_sigma f_pi) T[s, 2]  (``end_polarization``); the Bsig / Bpi vectors
    # above are the diagonals of parts 0 and 1 in the bridge frame.
    T_i: Optional[np.ndarray] = None
    T_j: Optional[np.ndarray] = None

    @property
    def cos_theta(self) -> float:
        return float(self.u_iL @ self.u_jL)


def _end_intensities(
    m_occ: np.ndarray,
    u: np.ndarray,
    frame: np.ndarray,
    rotation: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(Bsig, Bpi) channel intensity vectors for one bridge end."""
    w_sigma, w_pi = sk_table.channel_weights(u, frame, rotation=rotation)
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
) -> List[BridgeGeometry]:
    """Enumerate M-L-M bridges between descriptor sites with actual bond vectors.

    For each bridging anion, pairs its transition-metal neighbours and records the
    image-resolved M-L unit vectors, precomputing the channel intensities. A metal
    pair sharing several ligands (edge/face sharing) yields one bridge per ligand.

    Each site's d orbitals are evaluated in its own local frame (``local_d_frame``),
    so tilted octahedra keep their eg/t2g character and tetrahedral sites their
    e/t2 character.

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

    rotations: Dict[int, np.ndarray] = {
        site: local_d_frame(
            structure, site, descriptor.coordination, ligand_cutoff=anion_bond_cutoff
        )
        for site, descriptor in descriptors.items()
    }
    state_descriptors: Dict[Tuple[str, int, str, str], IonDescriptor] = {}
    for descriptor in descriptors.values():
        for state, _weight in end_fractions(descriptor):
            key = (
                descriptor.element, state, descriptor.spin_state, descriptor.coordination
            )
            if key not in state_descriptors:
                state_descriptors[key] = ion_descriptor(*key)

    # Occupied->empty FM channel setup. Every state's empty majority-spin orbitals are
    # its acceptor (Kanamori: at ~90 degrees a neighbour's t2g electron reaches an empty
    # eg through the shared ligand p orbital and Hund-aligns there -> FM; at 180 the two
    # use orthogonal p orbitals). For eg^1 Hund-active-core states the coherent
    # CF-ordered occupied eg orbital replaces the averaged eg part of the occupancy,
    # and the orthogonal (empty) eg orbital is the acceptor. t2g stays averaged.
    # The CF orbital is read off the ligand cage alone, so it is shared by every
    # state of a site; only whether a state can use it depends on the valence.
    eg_mask = np.zeros(5)
    eg_mask[list(EG_INDICES)] = 1.0
    # per state: (weight, occupancy, coherent occupied eg orbital, acceptor, eg electrons in it, unpaired electrons)
    site_states: Dict[int, List[tuple]] = {}
    for site, descriptor in descriptors.items():
        alternatives = [
            (weight, state_descriptors[
                (descriptor.element, state, descriptor.spin_state, descriptor.coordination)
            ])
            for state, weight in end_fractions(descriptor)
        ]
        ev = ev_hole = None
        if any(is_occ_empty_active(alt) for _weight, alt in alternatives):
            ev = crystal_field_eg_orbital(
                structure, site, rotations[site], hole=False,
                ligand_cutoff=anion_bond_cutoff,
            )
        if any(is_eg_hole(alt) for _weight, alt in alternatives):
            ev_hole = crystal_field_eg_orbital(
                structure, site, rotations[site], hole=True,
                ligand_cutoff=anion_bond_cutoff,
            )
        states = []
        for weight, alt in alternatives:
            occupancy = occupancy_vector(alt)
            unpaired = float(occupancy.sum())
            if is_eg_hole(alt):
                # eg^3: the hole's orbital is the half-filled one; nothing is empty to accept into.
                if ev_hole is None:
                    states.append((weight, occupancy, None, None, 0.0, unpaired))
                else:
                    occ5 = np.zeros(5); occ5[EG_INDICES[0]] = ev_hole[0]; occ5[EG_INDICES[1]] = ev_hole[1]
                    states.append((weight, occupancy * (1.0 - eg_mask), occ5, None, float(alt.ehf_eg[1]), unpaired))
                continue
            if not is_occ_empty_active(alt):
                # Kanamori acceptor: the state's empty majority-spin orbitals, shell-averaged
                # (Cr3+ / Mn4+ d3: both eg; d5-d9 high spin: none).
                states.append((weight, occupancy, None, empty_vector(alt), 0.0, unpaired))
                continue
            if ev is None:
                # eg^1 on an undistorted cage: no orbital order, so no definite empty orbital.
                states.append((weight, occupancy, None, None, 0.0, unpaired))
                continue
            occ5 = np.zeros(5); occ5[EG_INDICES[0]] = ev[0]; occ5[EG_INDICES[1]] = ev[1]
            emp5 = np.zeros(5); emp5[EG_INDICES[0]] = -ev[1]; emp5[EG_INDICES[1]] = ev[0]
            states.append(
                (
                    weight,
                    occupancy * (1.0 - eg_mask),   # eg zeroed, coherent part added below
                    occ5,
                    emp5,
                    float(alt.ehf_eg[1]),
                    unpaired,
                )
            )
        site_states[site] = states

    def _end_channels(idx, u, frame, rot):
        """Weighted channel intensities for one bridge end.

        ``(Bsig, Bpi, Bsig_emp, Bpi_emp)``: the occupied orbitals over every state
        (the ligand polarization: J_SE's factor and the occupied->empty donor) and
        the empty majority-spin orbitals over every state (the acceptor): the
        coherent CF-empty eg orbital of an eg^1 state, the shell-averaged empty
        orbitals of any other.
        """
        bsig = np.zeros(3); bpi = np.zeros(3)
        bsig_emp = np.zeros(3); bpi_emp = np.zeros(3)
        for weight, occupancy, occ5, emp, h_eg, _unpaired in site_states[idx]:
            bs, bp = _end_intensities(occupancy, u, frame, rotation=rot)
            if occ5 is not None:
                bso, bpo = coherent_eg_intensity(occ5, u, frame, rotation=rot)
                bs = bs + h_eg * bso
                bp = bp + h_eg * bpo
                if emp is not None:
                    bse, bpe = coherent_eg_intensity(emp, u, frame, rotation=rot)
                    bsig_emp = bsig_emp + weight * h_eg * bse
                    bpi_emp = bpi_emp + weight * h_eg * bpe
            elif emp is not None and emp.any():
                bse, bpe = _end_intensities(emp, u, frame, rotation=rot)
                bsig_emp = bsig_emp + weight * bse
                bpi_emp = bpi_emp + weight * bpe
            bsig = bsig + weight * bs
            bpi = bpi + weight * bp
        return bsig, bpi, bsig_emp, bpi_emp

    _unit = np.eye(5)

    def _end_tensor(idx, u, rot):
        """``BridgeGeometry.T`` of one end: the same orbitals as ``_end_channels``, as matrices."""
        a_sigma, t_pi = sk_table.pd_amplitudes(u, rotation=rot)
        T = np.zeros((3, 3, 3, 3))
        for weight, occupancy, occ5, emp, h_eg, unpaired in site_states[idx]:
            occ = _orbital_matrices(occupancy, _unit, u, a_sigma, t_pi)
            acc = None
            if occ5 is not None:
                occ = occ + _orbital_matrices(np.array([h_eg]), occ5[None, :], u, a_sigma, t_pi)
                if emp is not None:
                    acc = _orbital_matrices(np.array([h_eg]), emp[None, :], u, a_sigma, t_pi)
            elif emp is not None and emp.any():
                acc = _orbital_matrices(emp, _unit, u, a_sigma, t_pi)
            T[0] += weight * occ
            if acc is not None:
                T[1] += weight * acc
                T[2] += weight * unpaired * acc
        return T

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
                rot_i = rotations.get(nbr_i.index)
                rot_j = rotations.get(nbr_j.index)
                bsig_i, bpi_i, bse_i, bpe_i = _end_channels(nbr_i.index, u_i, frame, rot_i)
                bsig_j, bpi_j, bse_j, bpe_j = _end_channels(nbr_j.index, u_j, frame, rot_j)
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
                        R0_iL=r0_metal_ligand_averaged(
                            desc_i.element,
                            end_fractions(desc_i),
                            desc_i.spin_state,
                            ligand,
                            desc_i.coordination,
                        ),
                        R0_jL=r0_metal_ligand_averaged(
                            desc_j.element,
                            end_fractions(desc_j),
                            desc_j.spin_state,
                            ligand,
                            desc_j.coordination,
                        ),
                        Bsig_i=bsig_i,
                        Bpi_i=bpi_i,
                        Bsig_j=bsig_j,
                        Bpi_j=bpi_j,
                        u_iL=u_i,
                        u_jL=u_j,
                        frame=frame,
                        rot_i=rot_i,
                        rot_j=rot_j,
                        de_active=de_active,
                        de_weight=de_weight,
                        oe_active=oe_active,
                        Bsig_emp_i=bse_i,
                        Bpi_emp_i=bpe_i,
                        Bsig_emp_j=bse_j,
                        Bpi_emp_j=bpe_j,
                        T_i=_end_tensor(nbr_i.index, u_i, rot_i),
                        T_j=_end_tensor(nbr_j.index, u_j, rot_j),
                    )
                )
    _attach_sto_damping(bridges, descriptors, end_fractions)
    from quick_mag.amplitude_head import ion_features

    # Like everything else at a bridge end, the features follow the end's states: the site's own
    # with valence="site", the site class's fractions otherwise (so which atom of a mixed-valent
    # class carries which label cannot matter).
    from quick_mag.amplitude_head import site_environment

    feats = {}
    for site, d in descriptors.items():
        states = end_fractions(d)
        env = site_environment(structure, site, d.element, states, d.spin_state, d.coordination,
                               cutoff=anion_bond_cutoff)
        feats[site] = sum(
            weight * ion_features(state_descriptors[(d.element, state, d.spin_state, d.coordination)], env)
            for state, weight in states
        )
    for bridge in bridges:
        object.__setattr__(bridge, "feat_i", feats[bridge.site_i])
        object.__setattr__(bridge, "feat_j", feats[bridge.site_j])
    return bridges


def _attach_sto_damping(bridges: List[BridgeGeometry], descriptors: Dict[int, IonDescriptor], end_fractions) -> None:
    """Fill each bridge end's ``sto_i`` / ``sto_j``: the (sigma, pi) STO damping (S(r)/S(R0))^2,
    averaged over the end's oxidation-state fractions (a state's exponent depends on its charge).
    Vectorised over every end sharing (element, states, ligand)."""
    from quick_mag.radial_overlap import pd_damping, slater_zeta

    if not bridges:
        return
    groups: Dict[tuple, List[Tuple[int, int]]] = {}
    for b, bridge in enumerate(bridges):
        for end, site in ((0, bridge.site_i), (1, bridge.site_j)):
            d = descriptors[site]
            key = (d.element, tuple(end_fractions(d)), bridge.ligand)
            groups.setdefault(key, []).append((b, end))
    for (element, fractions, ligand), ends in groups.items():
        r = np.array([bridges[b].r_iL if e == 0 else bridges[b].r_jL for b, e in ends])
        r0 = np.array([bridges[b].R0_iL if e == 0 else bridges[b].R0_jL for b, e in ends])
        n_p, z_p = slater_zeta(ligand, ANION_CHARGES.get(ligand, -2), "p")
        ds = np.zeros(len(ends)); dp = np.zeros(len(ends))
        for state, weight in fractions:
            n_d, z_d = slater_zeta(element, state, "d")
            s_, p_ = pd_damping(n_d, z_d, n_p, z_p, r, r0)
            ds += weight * s_; dp += weight * p_
        for k, (b, e) in enumerate(ends):
            # BridgeGeometry is frozen; this is still part of its construction
            object.__setattr__(bridges[b], "sto_i" if e == 0 else "sto_j", (float(ds[k]), float(dp[k])))


# ---------------------------------------------------------------------------
# mu and J assembly
# ---------------------------------------------------------------------------


def end_damping(params: PolarizationParameters, metal: str, ligand: str, r, R0, sto=(1.0, 1.0)):
    """(d_sigma, d_pi) of one bridge end (scalars or arrays): see ``PolarizationParameters.range_model``."""
    model = params.range_model
    if model == "exp":
        d = np.exp(-np.sqrt(params.get_alpha(metal) * params.get_alpha(ligand)) * (np.asarray(r) - R0))
        return d, d
    if model == "power":
        d = (np.asarray(R0) / np.asarray(r)) ** params.get_range_power(metal)
        return d, d
    if model == "sto":
        return sto[0], sto[1]
    raise ValueError(f"range_model must be exp, power or sto, got {model!r}")


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
    ds_i, dp_i = end_damping(params, bridge.metal_i, bridge.ligand, bridge.r_iL, bridge.R0_iL, bridge.sto_i)
    ds_j, dp_j = end_damping(params, bridge.metal_j, bridge.ligand, bridge.r_jL, bridge.R0_jL, bridge.sto_j)
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
        E_i = end_polarization(bridge.T_i[2], ds_i, g2 * dp_i, coherent)
        E_j = end_polarization(bridge.T_j[2], ds_j, g2 * dp_j, coherent)
        j_se += 2.0 * (w + jh) * params.k_ee ** 2 * float(np.sum(E_i * E_j))
    j_de = 0.0
    if bridge.de_active:
        t_pair = params.get_t_de(bridge.metal_i) * params.get_t_de(bridge.metal_j)
        if t_pair != 0.0:
            j_de = -(
                t_pair * damp_i * damp_j * bridge.cos_theta ** 2 * bridge.de_weight
            )
    j_oe = 0.0
    if bridge.oe_active:
        j_fm = params.get_j_fm(bridge.metal_i) * params.get_j_fm(bridge.metal_j)
        manual = any(v is not None for v in (bridge.Bsig_occ_i, bridge.Bpi_occ_i, bridge.Bsig_occ_j, bridge.Bpi_occ_j))
        if j_fm != 0.0 and coherent and not manual:
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
    return j_se, j_de, j_oe


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
    bridges = build_bridges(structure, descriptors, valence=valence) if sites else []
    index = {site: k for k, site in enumerate(sites)}
    j_eff = build_Jeff_matrix(bridges, index, params) if bridges else np.zeros((len(sites),) * 2)
    from quick_mag.direct_exchange import dd_geometry

    dd = dd_geometry(structure, descriptors, sites, cutoff=params.dd_cutoff)
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
    fields = {
        "alpha", "w_ligand", "jh_ligand", "t_de", "j_fm", "gamma_pi",
        "alpha_default", "w_default", "jh_default",
        "t_de_default", "j_fm_default",
        "range_model", "range_power", "range_power_default",
        "head_sigma", "head_pi", "k_dd", "dd_cutoff", "kernel", "k_ee",
    }
    values = {k: v for k, v in data.items() if k in fields}
    values.setdefault("kernel", "diagonal")       # files from before the coherent kernel were fitted with this one
    return PolarizationParameters(**values)


def load_params(path) -> PolarizationParameters:
    """Load the fitted parameter set from a JSON file path."""
    return params_from_dict(json.loads(Path(path).read_text()))


def default_params() -> PolarizationParameters:
    """The bundled fitted + hand-calibrated default model parameters."""
    return load_params(DEFAULT_PARAMS_PATH)
