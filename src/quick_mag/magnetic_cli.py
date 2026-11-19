#!/usr/bin/env python3
"""Command-line pipeline: structure -> oxidation states -> exchange -> spin configs.

Given a structure file, this predicts oxidation states, builds the exchange-coupling
matrix from the polarization model, and searches for low-energy collinear spin
configurations. The canonical perovskite orderings (G, C, F, A) are always scored
as *reference* configurations, independently of the solver's search, so their
energies are reported even when the solver does not land on them.

Usage:
    python magnetic_cli.py STRUCTURE.cif [--charge 0] [--top-k 1] ...
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from quick_mag.structure import ChemicalStructure, build_from_generation_parameters
from quick_mag.structure_utils import read_structure
from quick_mag.magnetic_moments import format_oxidation_distribution
from quick_mag.polarization_model import exchange_model
from quick_mag.site_states import SiteStates, ranked_assignments
from quick_mag.spin_solver import (
    NO_NONZERO_MAGNETIC_MOMENTS_MESSAGE,
    solve_for_assignment,
)
from quick_mag.classify_spin_structure import (
    classify_structure,
    site_indexing_from_generation_parameters,
    site_indexing_from_magnetic_sublattice,
)
from quick_mag.reference_configs import named_reference_spin_configs


def _resolve_site_indexing(structure: ChemicalStructure, magnetic_site_indices):
    """Best-available perovskite B-site indexing, or ``None`` for non-perovskites.

    Prefers builder provenance (``generation_parameters``); otherwise fits the
    magnetic sublattice onto a B grid via ``site_indexing_from_magnetic_sublattice``.
    """
    params = getattr(structure, "generation_parameters", None)
    if params is not None and getattr(params, "family", "perovskite") != "perovskite":
        # A spinel (or any other non-perovskite build) has no B grid to classify on.
        return None
    if params is not None:
        try:
            build = build_from_generation_parameters(params)
            indexing = site_indexing_from_generation_parameters(params, build)
            if indexing.b_site_indices.size > 0:
                return indexing
        except Exception:
            pass
    return site_indexing_from_magnetic_sublattice(structure, magnetic_site_indices)


def _label_config(structure, config, magnetic_site_indices, site_indexing) -> str:
    """Classify a compact spin config as F/A/C/G/E/Other, or "" if not classifiable."""
    if site_indexing is None:
        return ""
    try:
        tmp = ChemicalStructure.with_zero_magnetic_moments(
            name=structure.name,
            lattice=np.array(structure.lattice, dtype=np.float64, copy=True),
            cartesian_coords=np.array(structure.cartesian_coords, dtype=np.float64, copy=True),
            atomic_labels=list(structure.atomic_labels),
            is_periodic=structure.is_periodic,
        )
        compact = np.asarray(config.all_moments, dtype=float).reshape(-1)
        for i, site in enumerate(magnetic_site_indices):
            if 0 <= int(site) < structure.atom_count and i < compact.size:
                tmp.magnetic_moments[int(site), 2] = compact[i]
        return classify_structure(tmp, site_indexing=site_indexing).label
    except Exception:
        return ""


def _report_reference_configs(named_configs) -> None:
    """Print the reference-ordering energy table (lowest reference = ΔE 0).

    A and C single out one axis of the B grid, so each is scored in all three
    orientations — ``A(a)`` stacks its ferromagnetic planes along a, ``C(a)`` runs
    ferromagnetic chains along a, and so on. On a cubic cell the three coincide;
    on a distorted one they are different states, and reporting only one of them
    would miss the ordering the solver actually settles into.
    """
    print("\n  Reference magnetic configurations (scored independently of the solve):")
    if not named_configs:
        print("    unavailable — magnetic sublattice is not a perovskite B grid.")
        return
    e_min = min(config.energy for _name, config in named_configs)
    ordered = sorted(named_configs, key=lambda pair: pair[1].energy)
    print(f"    {'type':>6}  {'energy':>14}  {'ΔE':>12}  {'magnetization':>13}")
    for name, config in ordered:
        print(
            f"    {name:>6}  {config.energy:>14.6f}  {config.energy - e_min:>12.6f}"
            f"  {config.magnetization:>13.3f}"
        )


def _report_solved_configs(
    structure, all_states, magnetic_site_indices, site_indexing, max_configs
) -> None:
    """Print ranked solved configurations with their F/A/C/G/E/Other labels."""
    print("\n  Solved low-energy spin configurations:")
    if not all_states:
        print("    (none found)")
        return
    e_min = all_states[0].energy
    shown = all_states if max_configs <= 0 else all_states[:max_configs]
    print(
        f"    {'#':>3}  {'energy':>14}  {'ΔE':>12}  {'magnetization':>13}"
        f"  {'n_unpaired':>10}  {'type':>6}"
    )
    for rank, config in enumerate(shown, start=1):
        label = _label_config(structure, config, magnetic_site_indices, site_indexing)
        print(
            f"    {rank:>3}  {config.energy:>14.6f}  {config.energy - e_min:>12.6f}"
            f"  {config.magnetization:>13.3f}  {config.n_unpaired:>10.2f}  {label:>6}"
        )
    if max_configs > 0 and len(all_states) > max_configs:
        print(f"    ... {len(all_states) - max_configs} more (raise --max-configs to see them)")


def _report_spin_states(model) -> None:
    """One line per (element, state, coordination, spin) class: count and formal moment."""
    classes = {}
    for site, m in zip(model.sites, model.moments):
        d = model.descriptors[site]
        key = (d.element, d.oxidation_state, d.coordination, d.spin_state)
        count, _m = classes.get(key, (0, m))
        classes[key] = (count + 1, m)
    parts = [
        f"{n}x{el}{ox:+d}({coord},{spin}) {m:g} muB"
        for (el, ox, coord, spin), (n, m) in sorted(classes.items())
    ]
    print("  magnetic sites: " + ", ".join(parts))


def _run_assignment(structure, assignment, args) -> None:
    print(f"\n{'=' * 80}")
    print(
        "Oxidation-state assignment:  "
        f"{format_oxidation_distribution(assignment.distributions)}"
    )
    print(f"  oxidation-model energy: {assignment.total_energy:.4f}")

    model = exchange_model(structure, assignment, valence=getattr(args, "valence", "class"))
    if not model.sites:
        print("  No transition-metal magnetic sites in this assignment; skipping.")
        return
    magnetic_site_indices = model.sites
    if not model.bridges:
        print("  No exchange couplings (no M-L-M bridges found); skipping solve.")
        return
    j_matrix = model.solver_J
    _report_spin_states(model)

    site_indexing = _resolve_site_indexing(structure, magnetic_site_indices)

    # J is per moment product, so the solver runs on the actual formal moments (2S of
    # each site's descriptor, low spin included). Reference and solved configs share
    # them, so their energies are comparable.
    moments = np.zeros(structure.atom_count)
    moments[magnetic_site_indices] = model.moments
    unit_assignment = replace(assignment, magnetic_moments=moments)

    # Reference G/C/F/A configurations, in every orientation of the B grid —
    # always reported, independent of the solve.
    reference = named_reference_spin_configs(
        structure, unit_assignment, j_matrix, magnetic_site_indices, site_indexing
    )
    _report_reference_configs(reference)

    n_mag = len(magnetic_site_indices)
    method = "exact" if n_mag <= args.exact_max_sites else "optimizer"
    print(f"\n  Solving {n_mag} magnetic sites with method='{method}'.")
    try:
        _base_states, all_states = solve_for_assignment(
            unit_assignment,
            j_matrix,
            magnetic_site_indices=magnetic_site_indices,
            method=method,
            collinear=True,
            n_trials=args.n_trials,
            n_steps=args.n_steps,
        )
    except ValueError as exc:
        if str(exc) == NO_NONZERO_MAGNETIC_MOMENTS_MESSAGE:
            print("  This assignment has no unpaired electrons on any site; nothing to solve.")
            return
        raise

    _report_solved_configs(
        structure, all_states, magnetic_site_indices, site_indexing, args.max_configs
    )

    # Ground state across both the solved search and the reference orderings.
    candidates = list(all_states) + [config for _name, config in reference]
    if candidates:
        ground = min(candidates, key=lambda c: c.energy)
        label = _label_config(structure, ground, magnetic_site_indices, site_indexing)
        tag = f" ({label})" if label else ""
        print(f"\n  Ground state (solved + reference): E = {ground.energy:.6f}{tag}")


SOLVE_DESCRIPTION = (
    "Predict oxidation states, exchange couplings, and low-energy magnetic "
    "configurations for a structure. Always reports the canonical G/C/F/A "
    "reference orderings."
)


def configure_solve_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the ``solve`` pipeline's arguments to ``parser`` (reused by the CLI)."""
    parser.add_argument(
        "structures", type=Path, nargs="*",
        help="Structure file(s): .cif (P1) or .vasp/POSCAR. Omit when the "
        "structures come from an earlier stage of a '::' chain.",
    )
    parser.add_argument("--charge", type=int, default=0, help="Net cell charge (default 0).")
    parser.add_argument(
        "--max-mixing", type=int, default=2,
        help="Max distinct oxidation states per element (default 2).",
    )
    parser.add_argument(
        "--top-k", type=int, default=1,
        help="Number of lowest-energy oxidation distributions to solve (default 1).",
    )
    parser.add_argument(
        "--exact-max-sites", type=int, default=16,
        help="Use exact Ising enumeration when magnetic sites <= this, else the optimizer (default 16).",
    )
    parser.add_argument("--n-trials", type=int, default=30, help="Optimizer restarts (default 30).")
    parser.add_argument("--n-steps", type=int, default=250, help="Optimizer steps per restart (default 250).")
    parser.add_argument(
        "--max-configs", type=int, default=20,
        help="Cap on solved configs printed (0 = no cap; default 20).",
    )
    parser.add_argument(
        "--ox", action="append", default=[], metavar="RULE",
        help="Oxidation state rule SELECTOR=STATES, repeatable: Co=2, Co@tet=2, Co@A=2 "
        "(site label), Mn@B=3|4 (allowed subset), #4,7-9=3 (atom indices, 0-based). "
        "Later rules win; everything not pinned is balanced by the oxidation model.",
    )
    parser.add_argument(
        "--spin", action="append", default=[], metavar="RULE",
        help="Spin state rule SELECTOR=HS|LS, repeatable: Co3+@oct=LS, Ni=LS, #12=LS. "
        "Unnamed sites are high spin.",
    )
    parser.add_argument(
        "--valence", choices=("class", "site"), default="class",
        help="'class' (default): a mixed-valent site is the average of its element/"
        "coordination class's states; 'site': each site is the state it was assigned "
        "(use with per-atom --ox pins).",
    )
    return parser


def _provenance_oxidation_constraints(structure: ChemicalStructure) -> dict:
    """Per-atom oxidation constraints recorded by the builder, if any.

    Only spinels record them so far (``--oxidation`` on ``build-spinel``): a
    restriction such as "Fe is +3 on the tetrahedral sites" has to reach the
    oxidation search as per-atom constraints, because the element-level search
    cannot tell the two Fe sublattices apart.
    """
    params = getattr(structure, "generation_parameters", None)
    if params is None or getattr(params, "family", "perovskite") != "spinel":
        return {}
    if not params.sublattice_oxidation_states:
        return {}
    from quick_mag.spinel_composition import sublattice_oxidation_constraints

    try:
        return sublattice_oxidation_constraints(structure)
    except ValueError:
        # Site roles that no longer line up with the atoms (e.g. a reordered
        # copy) cannot be trusted; fall back to the unconstrained search.
        return {}


def solve_structure(structure: ChemicalStructure, args) -> int:
    """Run the full magnetism pipeline for one structure.

    Oxidation constraints are the builder's sublattice restrictions (spinels) with the
    user's ``--ox`` rules over them; ``--spin`` rules set spin states per site.
    """
    labels = structure.element_symbols()
    print(f"Structure: {structure.name}  ({structure.atom_count} atoms)")
    print(f"  elements: {', '.join(sorted(set(labels)))}")

    try:
        states = SiteStates.parse(getattr(args, "ox", []) or [], getattr(args, "spin", []) or [])
        base = _provenance_oxidation_constraints(structure)
        assignments = ranked_assignments(
            structure, states, charge=args.charge, max_mixing=args.max_mixing,
            top_k=max(1, args.top_k), base_constraints=base,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    if not assignments:
        print("No charge-balanced oxidation-state assignments satisfy the constraints.")
        return 1
    if base:
        print(f"  Sublattice oxidation constraints on {len(base)} atom(s) (builder).")
    user = states.constraints(structure)
    if user:
        print(f"  User oxidation constraints on {len(user)} atom(s) (--ox).")
    print(f"  Top {len(assignments)} oxidation assignment(s) will be solved.")
    for assignment in assignments:
        _run_assignment(structure, assignment, args)
    return 0


def run_solve(args, structures=None) -> int:
    """Solve every structure named on the command line, or handed over by a chain.

    ``solve`` is always the last stage of a ``::`` chain — spin data feeds nothing
    else — so it writes no structures and simply reports.
    """
    if structures is None:
        if not args.structures:
            print(
                "Error: no structure given. Pass one or more structure files, or "
                "chain solve after a build/chgnet stage with '::'."
            )
            return 1
        structures = [read_structure(path) for path in args.structures]

    status = 0
    for index, structure in enumerate(structures):
        if index:
            print()
        status = solve_structure(structure, args) or status
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=SOLVE_DESCRIPTION)
    configure_solve_parser(parser)
    args = parser.parse_args(argv)
    return run_solve(args)


if __name__ == "__main__":
    raise SystemExit(main())
