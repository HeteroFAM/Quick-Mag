"""d-shell ion descriptors for the exchange model.

From an element, oxidation state and site coordination, derive the d-shell
picture the exchange model needs:

    oxidation state -> d-count -> (n_t2g, n_eg) shell filling
                    -> per-shell (E, H, F) decomposition
                    -> microstates over the named d-orbitals
                    -> ion spin S = (H_t2g + H_eg) / 2

A microstate is an occupancy vector over the five named d-orbitals (electrons per
orbital, in {0, 1, 2}); the (E, H, F) label of an orbital is ``{0: "E", 1: "H",
2: "F"}``.

Coordination
------------
The two shells are always the same two *orbital sets* -- (dxy, dxz, dyz) and
(dz2, dx2-y2) in the site's local cubic frame -- and ``n_t2g``/``n_eg`` count the
electrons in each. What the coordination changes is which set lies lower:

* ``"oct"`` (six ligands along the frame axes): t2g below eg, the default.
* ``"tet"`` (four ligands on alternate corners of the frame's cube, e.g. the
  spinel A site): the order inverts -- e = (dz2, dx2-y2) lies below
  t2 = (dxy, dxz, dyz) -- so Hund filling puts the first electrons into the
  (dz2, dx2-y2) set. Tetrahedral splittings are ~4/9 of octahedral ones, so
  tetrahedral ions are taken as high spin.

For HS d5 and d10 the two fillings coincide; for Co2+ (d7) they do not:
octahedral t2g^5 eg^2 carries its three unpaired electrons as one t2g + two eg,
tetrahedral e^4 t2^3 carries all three in t2.

Microstates and the exchange model
----------------------------------
Hund filling fixes how many orbitals per shell are empty/half/full, but not which
named orbital plays which role — every distinct assignment is a degenerate
microstate (Mn3+ t2g^3 eg^1 has two: the lone eg electron in dz2 or in dx2-y2).

**The exchange model does not currently consume these.** It takes the closed-form
average over them instead — ``polarization_model.occupancy_vector``, which puts
H_t2g/3 on each t2g orbital and H_eg/2 on each eg orbital. For degenerate
microstates that is exactly their mean, so nothing is lost relative to a uniform
weighting; what is lost is orbital-order information. For eg^1 ions the model
recovers that from the geometry instead, by picking the occupied eg orbital out of
a point-charge crystal field (``polarization_model.crystal_field_eg_orbital``).

``enumerate_shell_microstates`` and ``IonDescriptor.microstates`` are kept as the
hook for per-microstate weighting (Boltzmann weights, or resolving each microstate
into its own coupling and combining afterwards). Until something consumes them,
they are built and discarded on every solve — cheap (<= 6 per ion), but do not
mistake their presence for orbital resolution in the couplings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product as cartesian_product
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from quick_mag.electron_configurations import get_ionic_electron_configuration

# Named d-orbitals per shell. Order is fixed and used as the microstate key set.
T2G_ORBITALS: Tuple[str, ...] = ("dxy", "dxz", "dyz")
EG_ORBITALS: Tuple[str, ...] = ("dz2", "dx2-y2")
D_ORBITALS: Tuple[str, ...] = T2G_ORBITALS + EG_ORBITALS

# Site coordinations with their own crystal-field ordering (see module docstring).
COORDINATIONS: Tuple[str, ...] = ("oct", "tet")

# Transition-metal d-block elements that carry a partially-fillable d shell.
TRANSITION_METALS: frozenset[str] = frozenset(
    {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    }
)

# Bridging anions that mediate superexchange.
ANION_SPECIES: frozenset[str] = frozenset(
    {"O", "S", "N", "F", "Cl", "Br", "I", "Se", "Te"}
)

Microstate = Dict[str, int]


def d_electron_count(element: str, oxidation_state: int) -> int:
    """Number of d electrons for the ion (highest-n d subshell of its ionic config)."""
    configuration = get_ionic_electron_configuration(element, oxidation_state)
    d_subshells = [(n, count) for n, orbital, count in configuration if orbital == "d"]
    if not d_subshells:
        return 0
    # Valence d shell = the highest principal quantum number carrying d electrons.
    _, count = max(d_subshells, key=lambda item: item[0])
    if not 0 <= count <= 10:
        raise ValueError(f"Unphysical d-count {count} for {element}{oxidation_state:+d}.")
    return count


def shell_filling(
    d_count: int, spin: str = "HS", coordination: str = "oct"
) -> Tuple[int, int]:
    """Return ``(n_t2g, n_eg)`` for a d-count. High-spin by default.

    ``n_t2g`` counts electrons in (dxy, dxz, dyz) and ``n_eg`` in (dz2, dx2-y2).
    With ``coordination="tet"`` the (dz2, dx2-y2) pair is the lower (e) level, so
    it fills first.
    """
    if not 0 <= d_count <= 10:
        raise ValueError(f"d_count must be in [0, 10], got {d_count}.")
    spin = spin.upper()
    coordination = _validated_coordination(coordination)
    if coordination == "tet":
        if spin == "HS":
            # Hund filling: e^2 t2^3 singly, then pair e before t2.
            if d_count <= 2:
                return 0, d_count
            if d_count <= 5:
                return d_count - 2, 2
            if d_count <= 7:
                return 3, d_count - 3
            return d_count - 4, 4
        if spin == "LS":
            n_e = min(d_count, 4)
            return d_count - n_e, n_e
        raise ValueError(f"spin must be 'HS' or 'LS', got {spin!r}.")
    if spin == "HS":
        # Hund filling: t2g^3 eg^2 singly, then pair t2g before eg.
        if d_count <= 3:
            return d_count, 0
        if d_count <= 5:
            return 3, d_count - 3
        if d_count <= 8:
            return d_count - 2, 2
        return 6, d_count - 6
    if spin == "LS":
        # Strong field: fill t2g completely before touching eg.
        n_t2g = min(d_count, 6)
        return n_t2g, d_count - n_t2g
    raise ValueError(f"spin must be 'HS' or 'LS', got {spin!r}.")


def shell_ehf(n_shell: int, degeneracy: int) -> Tuple[int, int, int]:
    """(E, H, F) for ``n_shell`` electrons in a shell of orbital degeneracy ``g``."""
    empty = max(degeneracy - n_shell, 0)
    full = max(n_shell - degeneracy, 0)
    half = degeneracy - empty - full
    return empty, half, full


def _canonical_shell_occupancy(n_shell: int, degeneracy: int) -> List[int]:
    """Hund canonical per-orbital occupancy: doubly filled first, then singly, then empty."""
    empty, half, full = shell_ehf(n_shell, degeneracy)
    return [2] * full + [1] * half + [0] * empty


def enumerate_shell_microstates(
    n_shell: int, orbitals: Tuple[str, ...]
) -> List[Microstate]:
    """All distinct label assignments of a shell to its named orbitals.

    The count equals the multinomial ``g! / (E! H! F!)``. These microstates are
    degenerate under Hund filling; see the module docstring for why the exchange
    model averages over them rather than weighting them individually.
    """
    occupancy = _canonical_shell_occupancy(n_shell, len(orbitals))
    seen: set[Tuple[int, ...]] = set()
    microstates: List[Microstate] = []
    for permutation in permutations(occupancy):
        if permutation in seen:
            continue
        seen.add(permutation)
        microstates.append(dict(zip(orbitals, permutation)))
    return microstates


def _validated_coordination(coordination: str) -> str:
    value = str(coordination).strip().lower()
    if value not in COORDINATIONS:
        raise ValueError(f"coordination must be one of {COORDINATIONS}, got {coordination!r}.")
    return value


@dataclass
class IonDescriptor:
    """d-shell descriptor for a single transition-metal ion.

    ``n_t2g``/``n_eg`` (and ``ehf_t2g``/``ehf_eg``) refer to the (dxy, dxz, dyz)
    and (dz2, dx2-y2) orbital sets in the site's local frame; for a tetrahedral
    site those are the t2 and e levels (see the module docstring).
    """

    element: str
    oxidation_state: int
    d_count: int
    spin_state: str
    n_t2g: int
    n_eg: int
    ehf_t2g: Tuple[int, int, int]
    ehf_eg: Tuple[int, int, int]
    spin: float
    # Degenerate Hund microstates. Nothing consumes these yet — the exchange model
    # uses their average via ``polarization_model.occupancy_vector``; see the
    # module docstring.
    microstates: List[Microstate] = field(default_factory=list)
    coordination: str = "oct"

    @property
    def n_microstates(self) -> int:
        return len(self.microstates)

    def __repr__(self) -> str:
        if self.coordination == "tet":
            shells = f"tet e^{self.n_eg} t2^{self.n_t2g}"
        else:
            shells = f"t2g^{self.n_t2g} eg^{self.n_eg}"
        return (
            f"IonDescriptor({self.element}{self.oxidation_state:+d}, d{self.d_count} "
            f"{self.spin_state}, {shells}, S={self.spin:g}, "
            f"{self.n_microstates} microstate(s))"
        )


def ion_descriptor(
    element: str, oxidation_state: int, spin: str = "HS", coordination: str = "oct"
) -> IonDescriptor:
    """Build the full ion descriptor for ``element`` in the given oxidation state."""
    spin = spin.upper()
    coordination = _validated_coordination(coordination)
    d_count = d_electron_count(element, oxidation_state)
    n_t2g, n_eg = shell_filling(d_count, spin, coordination)
    ehf_t2g = shell_ehf(n_t2g, len(T2G_ORBITALS))
    ehf_eg = shell_ehf(n_eg, len(EG_ORBITALS))
    total_spin = (ehf_t2g[1] + ehf_eg[1]) / 2.0

    t2g_states = enumerate_shell_microstates(n_t2g, T2G_ORBITALS)
    eg_states = enumerate_shell_microstates(n_eg, EG_ORBITALS)
    microstates = [
        {**t2g_state, **eg_state}
        for t2g_state, eg_state in cartesian_product(t2g_states, eg_states)
    ]

    return IonDescriptor(
        element=element,
        oxidation_state=int(oxidation_state),
        d_count=d_count,
        spin_state=spin,
        n_t2g=n_t2g,
        n_eg=n_eg,
        ehf_t2g=ehf_t2g,
        ehf_eg=ehf_eg,
        spin=total_spin,
        microstates=microstates,
        coordination=coordination,
    )


def site_ligand_directions(
    structure,
    site_index: int,
    *,
    ligand_cutoff: float = 3.0,
    shell_tolerance: float = 1.3,
) -> np.ndarray:
    """Unit vectors to a site's first-shell anions, ``(n, 3)``.

    The first shell is every anion within ``ligand_cutoff`` and within
    ``shell_tolerance`` times the shortest metal-anion distance, so a second
    anion shell that happens to fall inside the cutoff is not counted.
    """
    center = structure.cartesian_coords[site_index]
    bonds = [
        (float(nbr.nn_distance), nbr.coords - center)
        for nbr in structure.neighbors(site_index, ligand_cutoff)
        if nbr.symbol in ANION_SPECIES and nbr.nn_distance > 1e-8
    ]
    if not bonds:
        return np.zeros((0, 3))
    shortest = min(distance for distance, _ in bonds)
    return np.array(
        [vec / distance for distance, vec in bonds if distance <= shell_tolerance * shortest]
    )


def site_coordination(
    structure,
    site_index: int,
    *,
    ligand_cutoff: float = 3.0,
    shell_tolerance: float = 1.3,
    trans_dot: float = -0.7,
) -> str:
    """``"tet"`` for a four-coordinate site with no trans ligand pair, else ``"oct"``.

    Four ligands with no (near-)antiparallel pair is a tetrahedron (ideal
    L-M-L cosine -1/3); four with trans pairs is square planar, which keeps the
    octahedral orbital ordering, as do five- and six-coordinate sites. Anything
    unrecognised falls back to ``"oct"``, the model's original assumption.
    """
    directions = site_ligand_directions(
        structure, site_index, ligand_cutoff=ligand_cutoff, shell_tolerance=shell_tolerance
    )
    if len(directions) != 4:
        return "oct"
    dots = directions @ directions.T
    np.fill_diagonal(dots, 1.0)
    return "oct" if float(dots.min()) < trans_dot else "tet"


def structure_ion_descriptors(
    structure,
    assignment,
    spin: str = "HS",
    *,
    spin_states: Optional[Mapping[int, str]] = None,
    ligand_cutoff: float = 3.0,
) -> Dict[int, IonDescriptor]:
    """Map each transition-metal site of an ``OxidationStateAssignment`` to its descriptor.

    ``assignment`` only needs a ``site_oxidation_states`` array aligned with
    ``structure``. Non-transition-metal sites (A-site cations, anions) are skipped.
    Each site's coordination (``site_coordination``) picks its crystal-field
    ordering.

    Spin state: ``spin_states`` (atom index -> ``"HS"``/``"LS"``, usually from
    ``quick_mag.site_states``) sets it per site; a site it does not name takes
    ``spin``, except that a tetrahedral site defaults to high spin (its splitting is
    ~4/9 of the octahedral one) -- a tetrahedral site is low spin only when named.
    """
    site_ox = assignment.site_oxidation_states
    per_site = {int(k): validated_spin(v) for k, v in (spin_states or {}).items()}
    descriptors: Dict[int, IonDescriptor] = {}
    for index, symbol in enumerate(structure.element_symbols()):
        if symbol not in TRANSITION_METALS:
            continue
        coordination = site_coordination(structure, index, ligand_cutoff=ligand_cutoff)
        site_spin = per_site.get(index, "HS" if coordination == "tet" else spin)
        descriptors[index] = ion_descriptor(
            symbol,
            int(site_ox[index]),
            site_spin,
            coordination,
        )
    return descriptors


def validated_spin(value: str) -> str:
    """``"HS"`` or ``"LS"`` (case-insensitive), else ValueError."""
    spin = str(value).strip().upper()
    if spin not in ("HS", "LS"):
        raise ValueError(f"spin state must be 'HS' or 'LS', got {value!r}.")
    return spin


def formal_moments(descriptors: Mapping[int, IonDescriptor]) -> Dict[int, float]:
    """Spin-only moment ``2S`` (mu_B, the number of unpaired d electrons) of every descriptor site.

    Follows each descriptor's spin state, so a low-spin Co3+ carries 0 and a
    low-spin Co2+ 1 -- unlike ``magnetic_moments.get_magnetic_moment``, which is
    the free-ion high-spin count.
    """
    return {int(site): 2.0 * float(d.spin) for site, d in descriptors.items()}
