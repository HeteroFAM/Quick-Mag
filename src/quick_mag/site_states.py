"""Oxidation and spin states set by the user: rules and per-atom values, resolved onto atoms.

The oxidation search (``oxidation_state_energy``) already takes per-atom and per-element
constraints, and the descriptors (``ion_descriptors``) a per-site spin state. This module is
how a person or a script *says* them, the way they think about a structure -- "Co is 2+ on the
tetrahedral sites", "the octahedral Co3+ are low spin", "these atoms are Mn4+" -- and turns that
into the two things the pipeline consumes:

    constraints(structure)              atom index -> allowed oxidation states (pins and subsets)
    spin_states(structure, site_ox)     atom index -> "HS" / "LS"

Rules
-----
``SELECTOR=VALUE``. The selector names atoms; the value is what they are.

    Co=2            every Co is 2+
    Co@tet=2        Co on tetrahedral sites (``ion_descriptors.site_coordination``)
    Co@A=2          Co on sites labelled A: the ``labels`` given, else the builder's
                    site roles (spinels: A / B / X)
    Mn@B=3|4        an allowed subset (also ``{3,4}`` or ``3,4``); the search picks the split
    #4,7,10-12=3    atoms by index (0-based; ranges inclusive)
    Co3+@oct=LS     spin rules may also select by oxidation state (``3+``, ``+3``)
    Co=LS           spin values are HS / LS

Later rules win over earlier ones where they overlap; per-atom arrays (``site_oxidation``,
``site_spin``: None / 0-length string / NaN = not set) win over every rule. An oxidation rule
that selects by charge is an error (the charge is what it sets); a rule that selects nothing
warns, since one rule set is often applied to many structures.

Spin states: a site no spin rule names takes ``default_spin`` (HS) -- octahedral and
tetrahedral alike, except that a tetrahedral site is low spin only when a rule names it (see
``structure_ion_descriptors``). Low spin changes the orbital occupancies the exchange model
reads (LS Co3+ t2g^6 carries nothing; LS Co2+ and LS Ni3+ are eg^1), and with them the formal
moment ``2S``; for d1-d3 and d8-d10 octahedral ions HS and LS coincide.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from quick_mag.ion_descriptors import (
    TRANSITION_METALS,
    d_electron_count,
    site_coordination,
    validated_spin,
)

__all__ = [
    "Selector",
    "SiteStates",
    "parse_selector",
    "parse_oxidation_rule",
    "parse_spin_rule",
]

_COORDINATIONS = ("tet", "oct")
_SELECTOR = re.compile(
    r"^(?P<el>[A-Z][a-z]?)(?P<charge>(?:\d+[+-])|(?:[+-]\d+))?(?:@(?P<where>[A-Za-z0-9_]+))?$"
)


@dataclass(frozen=True)
class Selector:
    """Which atoms a rule names. ``indices`` alone, or an element with optional filters."""

    element: Optional[str] = None
    charge: Optional[int] = None          # spin rules only: atoms in this oxidation state
    where: Optional[str] = None           # "tet" / "oct", or a site label
    indices: Optional[Tuple[int, ...]] = None
    text: str = ""

    def matches(
        self,
        atom: int,
        symbol: str,
        *,
        label: Optional[str],
        coordination,
        oxidation: Optional[int] = None,
    ) -> bool:
        if self.indices is not None:
            return atom in self.indices
        if symbol != self.element:
            return False
        if self.charge is not None and oxidation != self.charge:
            return False
        if self.where is None:
            return True
        if self.where in _COORDINATIONS:
            return coordination(atom) == self.where
        return label == self.where


def _parse_charge(text: str) -> int:
    text = text.strip()
    if text.endswith(("+", "-")):
        return int(text[:-1]) * (1 if text.endswith("+") else -1)
    return int(text)


def parse_selector(text: str) -> Selector:
    """``Co``, ``Co@tet``, ``Co3+@B``, ``#1,4,7-9``."""
    raw = text.strip()
    if raw.startswith("#"):
        indices: List[int] = []
        for part in raw[1:].split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                if hi < lo:
                    raise ValueError(f"empty index range {part!r} in {text!r}")
                indices.extend(range(lo, hi + 1))
            else:
                indices.append(int(part))
        if not indices:
            raise ValueError(f"no atom indices in {text!r}")
        return Selector(indices=tuple(sorted(set(indices))), text=raw)
    m = _SELECTOR.match(raw)
    if m is None:
        raise ValueError(
            f"cannot read selector {text!r}: expected Element[charge][@tet|@oct|@label] or #indices"
        )
    charge = _parse_charge(m.group("charge")) if m.group("charge") else None
    return Selector(element=m.group("el"), charge=charge, where=m.group("where"), text=raw)


def _split_rule(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"a rule is SELECTOR=VALUE, got {text!r}")
    left, right = text.split("=", 1)
    return left.strip(), right.strip()


def _parse_states(text: str) -> FrozenSet[int]:
    body = text.strip().strip("{}")
    parts = [p for p in re.split(r"[|,\s]+", body) if p]
    if not parts:
        raise ValueError(f"no oxidation state in {text!r}")
    return frozenset(_parse_charge(p) for p in parts)


def parse_oxidation_rule(text: str) -> Tuple[Selector, FrozenSet[int]]:
    """``"Co@tet=2"`` -> (selector, {2}); ``"Mn@B=3|4"`` -> (selector, {3, 4})."""
    left, right = _split_rule(text)
    selector = parse_selector(left)
    if selector.charge is not None:
        raise ValueError(f"an oxidation rule cannot select by charge: {text!r}")
    states = _parse_states(right)
    if selector.element is not None:
        for state in states:
            _check_ion(selector.element, state, text)
    return selector, states


def parse_spin_rule(text: str) -> Tuple[Selector, str]:
    """``"Co3+@oct=LS"`` -> (selector, "LS")."""
    left, right = _split_rule(text)
    return parse_selector(left), validated_spin(right)


def _valence_electrons(element: str) -> int:
    """Outer s + d electrons of the neutral atom (Co: 3d7 4s2 -> 9)."""
    from quick_mag.electron_configurations import get_ionic_electron_configuration

    config = get_ionic_electron_configuration(element, 0)
    d = [(n, c) for n, orb, c in config if orb == "d"]
    s = [(n, c) for n, orb, c in config if orb == "s"]
    n_d, c_d = max(d) if d else (0, 0)
    c_s = sum(c for n, c in s if n == n_d + 1)
    return int(c_d + c_s)


def _check_ion(element: str, state: int, context: str) -> None:
    """A transition-metal state the model can describe: 0 <= state <= its outer s + d count.

    The configuration tables happily strip core electrons (they give Co+12 as d0 with a
    3p3 core), so the bound is checked here rather than left to them.
    """
    if element not in TRANSITION_METALS:
        return
    state = int(state)
    try:
        top = _valence_electrons(element)
        d_electron_count(element, state)
    except Exception as exc:  # the tables cannot build it
        raise ValueError(f"{element}{state:+d} is not an ion the model can describe ({context}): {exc}")
    if not 0 <= state <= top:
        raise ValueError(
            f"{element}{state:+d} is not an ion the model can describe ({context}): "
            f"{element} states run from 0 to +{top}"
        )


def _is_unset(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


@dataclass
class SiteStates:
    """Oxidation and spin states as rules plus optional per-atom values.

    Build with :meth:`parse` from rule strings, or fill the fields directly. ``labels``
    (per-atom site labels for ``@label`` selectors) is optional; without it the
    builder's site roles are used when the structure has them.
    """

    oxidation_rules: List[Tuple[Selector, FrozenSet[int]]] = field(default_factory=list)
    spin_rules: List[Tuple[Selector, str]] = field(default_factory=list)
    site_oxidation: Optional[Sequence] = None
    site_spin: Optional[Sequence] = None
    labels: Optional[Sequence[str]] = None
    default_spin: str = "HS"

    @classmethod
    def parse(
        cls,
        oxidation: Sequence[str] = (),
        spin: Sequence[str] = (),
        **kwargs,
    ) -> "SiteStates":
        return cls(
            oxidation_rules=[parse_oxidation_rule(r) for r in oxidation],
            spin_rules=[parse_spin_rule(r) for r in spin],
            **kwargs,
        )

    @property
    def empty(self) -> bool:
        return not (
            self.oxidation_rules
            or self.spin_rules
            or (self.site_oxidation is not None and not all(_is_unset(v) for v in self.site_oxidation))
            or (self.site_spin is not None and not all(_is_unset(v) for v in self.site_spin))
        )

    # -- resolution ----------------------------------------------------------------------

    def _labels(self, structure) -> Optional[List[str]]:
        if self.labels is not None:
            labels = [str(v) for v in self.labels]
            if len(labels) != structure.atom_count:
                raise ValueError(f"{len(labels)} site labels for {structure.atom_count} atoms")
            return labels
        try:
            from quick_mag.spinel_composition import structure_site_roles

            return [str(v) for v in structure_site_roles(structure)]
        except Exception:
            return None

    def _coordination(self, structure):
        cache: Dict[int, str] = {}

        def coordination(atom: int) -> str:
            if atom not in cache:
                cache[atom] = site_coordination(structure, atom)
            return cache[atom]

        return coordination

    def _per_atom(self, values, n: int, name: str):
        if values is None:
            return None
        values = list(values)
        if len(values) != n:
            raise ValueError(f"{len(values)} {name} values for {n} atoms")
        return values

    def constraints(self, structure) -> Dict[int, FrozenSet[int]]:
        """Atom index -> allowed oxidation states, for ``constraints=`` of the oxidation search."""
        symbols = structure.element_symbols()
        labels = self._labels(structure)
        coordination = self._coordination(structure)
        out: Dict[int, FrozenSet[int]] = {}
        for selector, states in self.oxidation_rules:
            if selector.where is not None and selector.where not in _COORDINATIONS and labels is None:
                raise ValueError(f"rule {selector.text!r} names a site label but the structure has none")
            hit = 0
            for atom, symbol in enumerate(symbols):
                if selector.matches(
                    atom, symbol, label=labels[atom] if labels else None, coordination=coordination
                ):
                    if selector.indices is not None:
                        for state in states:
                            _check_ion(symbol, state, selector.text)
                    out[atom] = states
                    hit += 1
            if not hit:
                warnings.warn(f"oxidation rule {selector.text!r} selects no atoms in {structure.name}")
        per_atom = self._per_atom(self.site_oxidation, len(symbols), "site_oxidation")
        if per_atom is not None:
            for atom, value in enumerate(per_atom):
                if _is_unset(value):
                    continue
                state = int(round(float(value)))
                _check_ion(symbols[atom], state, f"site_oxidation[{atom}]")
                out[atom] = frozenset([state])
        return out

    def spin_states(self, structure, site_oxidation_states) -> Dict[int, str]:
        """Atom index -> "HS"/"LS" for every atom a spin rule or ``site_spin`` names.

        ``site_oxidation_states`` (the assignment being solved) is what charge-selecting
        rules (``Co3+=LS``) test against.
        """
        symbols = structure.element_symbols()
        site_ox = np.asarray(site_oxidation_states).reshape(-1)
        labels = self._labels(structure)
        coordination = self._coordination(structure)
        out: Dict[int, str] = {}
        for selector, spin in self.spin_rules:
            if selector.where is not None and selector.where not in _COORDINATIONS and labels is None:
                raise ValueError(f"rule {selector.text!r} names a site label but the structure has none")
            hit = 0
            for atom, symbol in enumerate(symbols):
                if selector.matches(
                    atom, symbol, label=labels[atom] if labels else None,
                    coordination=coordination, oxidation=int(site_ox[atom]),
                ):
                    out[atom] = spin
                    hit += 1
            if not hit:
                warnings.warn(f"spin rule {selector.text!r} selects no atoms in {structure.name}")
        per_atom = self._per_atom(self.site_spin, len(symbols), "site_spin")
        if per_atom is not None:
            for atom, value in enumerate(per_atom):
                if not _is_unset(value):
                    out[atom] = validated_spin(value)
        return out

    def check_pinned_charge(self, structure, charge: int = 0) -> None:
        """Raise a readable error when every cation is pinned and the pins do not balance.

        The oxidation search would just return nothing; this says by how much. Anions are
        taken at their closed-shell charge (``STANDARD_ANION_STATES``).
        """
        from quick_mag.oxidation_state_energy import STANDARD_ANION_STATES

        pins = {a: next(iter(s)) for a, s in self.constraints(structure).items() if len(s) == 1}
        total = 0
        for atom, symbol in enumerate(structure.element_symbols()):
            if atom in pins:
                total += pins[atom]
            elif symbol in STANDARD_ANION_STATES:
                total += STANDARD_ANION_STATES[symbol]
            else:
                return  # a free cation can still balance it
        if total != int(charge):
            raise ValueError(
                f"the pinned oxidation states of {structure.name} sum to {total:+d} "
                f"(anions at their closed-shell charge), not the cell charge {int(charge):+d}"
            )


# ---------------------------------------------------------------------------------------
# The pipeline's front end: ranked assignments that honour the states
# ---------------------------------------------------------------------------------------


def ranked_assignments(
    structure,
    states: Optional[SiteStates] = None,
    *,
    charge: int = 0,
    max_mixing: int = 2,
    top_k: int = 1,
    base_constraints: Optional[Mapping[int, FrozenSet[int]]] = None,
):
    """The ``top_k`` lowest-energy oxidation assignments that honour ``states``, one site
    ordering each, with each assignment's ``magnetic_moments`` set to its descriptors'
    formal moments (spin states included) and the spin states attached as
    ``assignment.spin_states``.

    ``base_constraints`` (e.g. a builder's sublattice restrictions) apply under the user's:
    an atom the user constrains takes the user's set. Without any constraint this is the
    unconstrained search, as before.
    """
    from dataclasses import replace

    from quick_mag.ion_descriptors import formal_moments, structure_ion_descriptors
    from quick_mag.magnetic_moments import (
        expand_distribution_to_site_assignments,
        expand_group_assignments_to_site_assignments,
    )
    from quick_mag.oxidation_state_energy import (
        enumerate_oxidation_states_by_energy,
        enumerate_site_group_assignments,
    )

    states = states or SiteStates()
    constraints: Dict[int, FrozenSet[int]] = dict(base_constraints or {})
    constraints.update(states.constraints(structure))
    labels = structure.element_symbols()
    out = []
    if constraints:
        states.check_pinned_charge(structure, charge)
        ranked = enumerate_site_group_assignments(
            labels, charge=charge, max_mixing=max_mixing, constraints=constraints, top_k=max(1, top_k)
        )
        for group_assignment in ranked:
            out.extend(expand_group_assignments_to_site_assignments([group_assignment], structure, max_assignments=1))
    else:
        for distribution, _energy in enumerate_oxidation_states_by_energy(
            labels, charge=charge, max_mixing=max_mixing, top_k=max(1, top_k)
        ):
            found = expand_distribution_to_site_assignments([distribution], structure)
            if found:
                out.append(found[0])
    result = []
    for assignment in out:
        spins = states.spin_states(structure, assignment.site_oxidation_states)
        descriptors = structure_ion_descriptors(structure, assignment, states.default_spin, spin_states=spins)
        moments = np.array(assignment.magnetic_moments, dtype=float, copy=True)
        for site, value in formal_moments(descriptors).items():
            moments[site] = value
        result.append(replace(
            assignment,
            magnetic_moments=moments,
            spin_states={site: d.spin_state for site, d in descriptors.items()},
        ))
    return result
