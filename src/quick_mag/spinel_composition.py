"""Occupancy rules for spinel builds: inversion, high-entropy mixes, site preference.

Everything here maps a :class:`~quick_mag.structure.SpinelGenerationParameters`
onto element labels for a :class:`~quick_mag.spinel_builder.SpinelBuild`, or
helps choose the parameters in the first place:

* :func:`spinel_atomic_labels` -- the labels for every canonical site, for
  either formula mode. Deterministic: the same parameters always give the same
  labels.
* :func:`exact_count_labels` -- place *exactly* the composition a set of
  fractions implies on ``n`` sites (largest-remainder rounding, then a seeded
  shuffle). High-entropy spinels default to this rather than independent draws,
  because an A:B ratio that drifts from 1:2 is a different, charged compound.
* :func:`distribute_cations` -- split a cation pool over the tetrahedral and
  octahedral sublattices given per-cation site preferences, which is how a
  composition like (Co,Cr,Fe,Mn,Ni)3O4 becomes per-sublattice mixes.
* :func:`sublattice_oxidation_constraints` -- turn the per-sublattice
  oxidation-state restrictions stored on the parameters into the per-atom
  constraints the oxidation-state search takes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from quick_mag.element_data import is_valid_symbol

# Gas constant in kJ/(mol K), for site-preference energies given in kJ/mol.
GAS_CONSTANT_KJ_PER_MOL_K = 8.314462618e-3
DEFAULT_TEMPERATURE_K = 1273.15

# A preference is either a hard placement or a site energy
# ``E_tet - E_oct`` in kJ/mol (negative favours the tetrahedral site).
SitePreference = Union[str, float]


def _normalize_symbol(raw: str) -> str:
    symbol = str(raw).strip()
    if not symbol:
        raise ValueError("Element symbols cannot be empty.")
    symbol = symbol[0].upper() + symbol[1:].lower()
    if not is_valid_symbol(symbol):
        raise ValueError(f"'{raw}' is not a valid element symbol.")
    return symbol


def normalized_mix(entries: Sequence[Tuple[str, float]], *, name: str) -> List[Tuple[str, float]]:
    """Validated ``(symbol, fraction)`` pairs, fractions summing to 1.

    Repeated symbols are merged (first occurrence keeps its place) and
    zero-weight rows dropped, so equivalent mixes sample identically.
    """
    merged: Dict[str, float] = {}
    for raw_symbol, raw_fraction in entries:
        symbol = _normalize_symbol(raw_symbol)
        fraction = float(raw_fraction)
        if fraction < 0.0 or not math.isfinite(fraction):
            raise ValueError(f"{name} fraction for {symbol} must be a non-negative number.")
        merged[symbol] = merged.get(symbol, 0.0) + fraction
    total = sum(merged.values())
    if total <= 0.0:
        raise ValueError(f"{name} needs at least one element with a positive fraction.")
    return [(symbol, value / total) for symbol, value in merged.items() if value > 0.0]


def exact_counts(fractions: Sequence[float], count: int) -> List[int]:
    """Integer counts summing to ``count`` closest to ``fractions * count``.

    Largest-remainder (Hamilton) rounding; ties go to the earlier entry, so the
    result is a deterministic function of the input order.
    """
    count = int(count)
    weights = np.asarray(fractions, dtype=float)
    if count <= 0 or weights.size == 0:
        return [0] * int(weights.size)
    ideal = weights / weights.sum() * count
    floors = np.floor(ideal + 1e-12).astype(int)
    remaining = count - int(floors.sum())
    remainders = ideal - floors
    # Stable sort on -remainder keeps earlier entries first among ties.
    order = sorted(range(len(weights)), key=lambda i: (-round(remainders[i], 12), i))
    for i in order[:remaining]:
        floors[i] += 1
    return [int(value) for value in floors]


def _rng(seed_parts: Tuple[object, ...]) -> np.random.Generator:
    from quick_mag.generation import deterministic_seed

    return np.random.default_rng(deterministic_seed(*seed_parts))


def exact_count_labels(
    entries: Sequence[Tuple[str, float]],
    count: int,
    *,
    seed_parts: Tuple[object, ...],
    name: str = "mix",
) -> List[str]:
    """``count`` labels with exactly the composition ``entries`` implies, shuffled."""
    if count <= 0:
        return []
    mix = normalized_mix(entries, name=name)
    counts = exact_counts([fraction for _, fraction in mix], count)
    labels = [symbol for (symbol, _), n in zip(mix, counts) for _ in range(n)]
    rng = _rng(seed_parts + (tuple(mix), int(count)))
    return [labels[int(index)] for index in rng.permutation(len(labels))]


def iid_labels(
    entries: Sequence[Tuple[str, float]],
    count: int,
    *,
    seed_parts: Tuple[object, ...],
    name: str = "mix",
) -> List[str]:
    """``count`` labels drawn independently from ``entries`` (composition fluctuates)."""
    if count <= 0:
        return []
    mix = normalized_mix(entries, name=name)
    rng = _rng(seed_parts + (tuple(mix), int(count)))
    choices = rng.choice(len(mix), size=int(count), p=[fraction for _, fraction in mix])
    return [mix[int(index)][0] for index in choices]


def _role_counts(build) -> Dict[str, int]:
    return {
        "A": len(build.a_site_indices),
        "B": len(build.b_site_indices),
        "X": len(build.x_site_indices),
    }


def inversion_swap_count(inversion: float, n_tetrahedral: int) -> int:
    """Number of A/B pairs exchanged for an inversion degree on ``n_tetrahedral`` sites."""
    return int(math.floor(float(inversion) * int(n_tetrahedral) + 0.5))


def spinel_atomic_labels(build, params) -> List[str]:
    """Element label for every canonical site of ``build``, per ``params``.

    ``build`` is the :class:`~quick_mag.spinel_builder.SpinelBuild` that
    ``params`` describes; the labels come back in the build's canonical
    ``(A, B, X)`` order.
    """
    counts = _role_counts(build)
    base_seed = (
        "spinel",
        tuple(build.n_cells),
        build.cell,
        int(params.seed),
        int(params.sample_index),
    )

    if params.formula_mode == "high_entropy":
        draw = exact_count_labels if params.sampling == "exact" else iid_labels
        labels: List[str] = []
        for role, entries in (
            ("A", params.high_entropy_a_sites),
            ("B", params.high_entropy_b_sites),
            ("X", params.high_entropy_x_sites),
        ):
            labels += draw(
                entries,
                counts[role],
                seed_parts=base_seed + (role,),
                name=f"{role}-site mix",
            )
        return labels

    a_symbol = _normalize_symbol(params.a_site_element)
    b_symbol = _normalize_symbol(params.b_site_element)
    x_symbol = _normalize_symbol(params.x_site_element)
    a_labels = [a_symbol] * counts["A"]
    b_labels = [b_symbol] * counts["B"]
    n_swap = inversion_swap_count(params.inversion, counts["A"])
    if n_swap:
        # The permutations ignore the inversion degree on purpose: the sites
        # inverted at x are a prefix of those inverted at any larger x.
        tet_order = _rng(base_seed + ("inversion", "A")).permutation(counts["A"])
        oct_order = _rng(base_seed + ("inversion", "B")).permutation(counts["B"])
        for index in tet_order[:n_swap]:
            a_labels[int(index)] = b_symbol
        for index in oct_order[:n_swap]:
            b_labels[int(index)] = a_symbol
    return a_labels + b_labels + [x_symbol] * counts["X"]


# ---------------------------------------------------------------------------
# Cation distribution over the two sublattices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CationDistribution:
    """A cation pool split over the spinel sublattices, per formula unit AB2X4.

    ``tetrahedral[el]`` + ``octahedral[el]`` is the element's amount in the pool
    (normalized to 3 cations per formula unit); the tetrahedral amounts sum to 1
    and the octahedral ones to 2. ``a_sites``/``b_sites`` are the same numbers
    as site fractions -- what :class:`~quick_mag.structure.SpinelGenerationParameters`
    takes as ``high_entropy_a_sites``/``high_entropy_b_sites``.
    """

    tetrahedral: Dict[str, float]
    octahedral: Dict[str, float]

    @property
    def a_sites(self) -> List[Tuple[str, float]]:
        return [(el, amount) for el, amount in self.tetrahedral.items() if amount > 1e-12]

    @property
    def b_sites(self) -> List[Tuple[str, float]]:
        return [(el, amount / 2.0) for el, amount in self.octahedral.items() if amount > 1e-12]

    def tetrahedral_fraction(self, element: str) -> float:
        """Share of ``element``'s atoms that sit on the tetrahedral sublattice."""
        total = self.tetrahedral.get(element, 0.0) + self.octahedral.get(element, 0.0)
        return self.tetrahedral.get(element, 0.0) / total if total > 0.0 else 0.0

    def inversion(self, octahedral_elements: Sequence[str]) -> float:
        """Inversion degree: amount of the named "B-type" cations on A sites."""
        return float(sum(self.tetrahedral.get(el, 0.0) for el in octahedral_elements))


def _preference_weight(preference: SitePreference, temperature: float) -> float:
    """Boltzmann weight of the tetrahedral site: ``inf`` (tet only) .. 0 (oct only)."""
    if isinstance(preference, str):
        key = preference.strip().lower()
        if key in ("tet", "tetrahedral", "a"):
            return math.inf
        if key in ("oct", "octahedral", "b"):
            return 0.0
        try:
            preference = float(key)
        except ValueError:
            raise ValueError(
                f"Site preference '{preference}' must be 'tet', 'oct', or an energy in kJ/mol."
            )
    energy = float(preference)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    exponent = -energy / (GAS_CONSTANT_KJ_PER_MOL_K * float(temperature))
    return math.exp(min(max(exponent, -700.0), 700.0))


def distribute_cations(
    cations: Sequence[Tuple[str, float]],
    preferences: Optional[Mapping[str, SitePreference]] = None,
    *,
    temperature: float = DEFAULT_TEMPERATURE_K,
) -> CationDistribution:
    """Split a cation pool over the tetrahedral (1) and octahedral (2) sites.

    ``cations`` are ``(symbol, amount)`` pairs, normalized to 3 cations per
    formula unit. ``preferences`` maps an element to ``"tet"``/``"oct"`` (it
    sits only there) or to a site energy ``E_tet - E_oct`` in kJ/mol at
    ``temperature``; unlisted elements have no preference.

    The soft placements are the ideal-solution equilibrium of the two
    sublattices: for every element ``y_A / y_B = lambda * exp(-dE / RT)`` with
    one ``lambda`` fixed by the site count, i.e.
    ``t_i = n_i * (lambda w_i / 2) / (1 + lambda w_i / 2)`` with
    ``sum_i t_i = 1``. With every preference zero this is the random
    (statistical) distribution, each element 1/3 tetrahedral.
    """
    pool = normalized_mix(cations, name="cation pool")
    amounts = {symbol: 3.0 * fraction for symbol, fraction in pool}
    prefs = {_normalize_symbol(el): value for el, value in dict(preferences or {}).items()}
    unknown = sorted(set(prefs) - set(amounts))
    if unknown:
        raise ValueError(f"Site preference given for elements not in the pool: {', '.join(unknown)}.")
    weights = {el: _preference_weight(prefs.get(el, 0.0), temperature) for el in amounts}

    hard_tet = sum(amounts[el] for el, w in weights.items() if math.isinf(w))
    hard_oct = sum(amounts[el] for el, w in weights.items() if w == 0.0)
    soft = [el for el, w in weights.items() if 0.0 < w < math.inf]
    tol = 1e-9
    if hard_tet > 1.0 + tol:
        raise ValueError(
            f"Cations restricted to tetrahedral sites need {hard_tet:.3f} per formula "
            "unit, but there is only 1."
        )
    if hard_oct > 2.0 + tol:
        raise ValueError(
            f"Cations restricted to octahedral sites need {hard_oct:.3f} per formula "
            "unit, but there are only 2."
        )

    tet = {el: (amounts[el] if math.isinf(weights[el]) else 0.0) for el in amounts}
    need = 1.0 - hard_tet
    if not soft:
        if abs(need) > 1e-6:
            raise ValueError(
                "Every cation has a hard site preference and they do not fill the "
                f"sublattices exactly ({hard_tet:.3f} tetrahedral of 1, {hard_oct:.3f} "
                "octahedral of 2)."
            )
    elif need > tol:
        def filled(log_lambda: float) -> float:
            total = 0.0
            for el in soft:
                x = math.exp(min(log_lambda, 700.0)) * weights[el] / 2.0
                total += amounts[el] * x / (1.0 + x)
            return total

        lo, hi = -800.0, 800.0
        for _ in range(400):
            mid = 0.5 * (lo + hi)
            if filled(mid) < need:
                lo = mid
            else:
                hi = mid
        lam = math.exp(min(0.5 * (lo + hi), 700.0))
        for el in soft:
            x = lam * weights[el] / 2.0
            tet[el] = amounts[el] * x / (1.0 + x)
        # Remove the bisection residue so the sublattices fill exactly.
        scale = need / sum(tet[el] for el in soft)
        for el in soft:
            tet[el] *= scale
    oct_ = {el: amounts[el] - tet[el] for el in amounts}
    return CationDistribution(tetrahedral=tet, octahedral=oct_)


# ---------------------------------------------------------------------------
# Oxidation-state constraints and summaries on built structures
# ---------------------------------------------------------------------------


def structure_site_roles(structure) -> List[str]:
    """Spinel sublattice role (A/B/X) of every atom, in structure order."""
    params = getattr(structure, "generation_parameters", None)
    if params is None or getattr(params, "family", "") != "spinel":
        raise ValueError(f"'{structure.name}' has no spinel builder provenance.")
    roles = list(params.site_roles)
    if len(roles) != structure.atom_count:
        raise ValueError(
            f"'{structure.name}' records {len(roles)} site roles for "
            f"{structure.atom_count} atoms."
        )
    return roles


def sublattice_oxidation_constraints(
    structure,
    sublattice_states: Optional[Mapping[str, Mapping[str, Sequence[int]]]] = None,
) -> Dict[int, FrozenSet[int]]:
    """Per-atom oxidation-state constraints from per-sublattice restrictions.

    ``sublattice_states`` maps a role to ``{element: allowed states}``; it
    defaults to the ``sublattice_oxidation_states`` stored on the structure's
    parameters. The result plugs into ``constraints=`` of
    :func:`quick_mag.oxidation_state_energy.enumerate_site_group_assignments`,
    which keeps each constrained set of atoms as its own group -- so Fe on the
    tetrahedral sites and Fe on the octahedral sites can be given different
    valences, as in Fe3O4.
    """
    if sublattice_states is None:
        sublattice_states = structure.generation_parameters.sublattice_oxidation_states
    roles = structure_site_roles(structure)
    symbols = structure.element_symbols()
    constraints: Dict[int, FrozenSet[int]] = {}
    for role, per_element in dict(sublattice_states).items():
        for element, states in dict(per_element).items():
            allowed = frozenset(int(state) for state in states)
            if not allowed:
                raise ValueError(f"No oxidation states allowed for {element} on {role} sites.")
            for atom, (atom_role, symbol) in enumerate(zip(roles, symbols)):
                if atom_role == role and symbol == element:
                    constraints[atom] = allowed
    return constraints


def sublattice_composition(structure) -> Dict[str, Dict[str, int]]:
    """Element counts per sublattice role, e.g. ``{"A": {"Co": 8}, ...}``."""
    roles = structure_site_roles(structure)
    summary: Dict[str, Dict[str, int]] = {}
    for role, symbol in zip(roles, structure.element_symbols()):
        per_role = summary.setdefault(role, {})
        per_role[symbol] = per_role.get(symbol, 0) + 1
    return summary


def format_sublattice_composition(structure) -> str:
    """One-line ``A: Co8 | B: Fe16 | X: O32`` summary of :func:`sublattice_composition`."""
    parts = []
    for role, counts in sublattice_composition(structure).items():
        body = " ".join(f"{el}{n}" for el, n in counts.items())
        parts.append(f"{role}: {body}")
    return " | ".join(parts)
