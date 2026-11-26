#!/usr/bin/env python3
"""Command-line pipeline: structure -> oxidation states -> exchange -> spin configs.

Given a structure file, this predicts oxidation states, builds the exchange-coupling
matrix from the polarization model, and searches for low-energy collinear spin
configurations. The canonical perovskite orderings (G, C, F, A) are always scored
as *reference* configurations, independently of the solver's search, so their
energies are reported even when the solver does not land on them.

An extended xyz structure may carry the states itself, per atom: an ``oxidation`` column
(0 = not given) and a ``spin_state`` column (``HS`` / ``LS``). Atoms without one are worked
out as usual; ``--ox`` / ``--spin`` rules are applied over what the file says.

``--out DIR`` writes the results as files (:mod:`quick_mag.solve_output`) besides printing
them.

Usage:
    python magnetic_cli.py STRUCTURE.cif [--charge 0] [--top-k 1] ...
"""

from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import numpy as np

from quick_mag.structure import ChemicalStructure, build_from_generation_parameters
from quick_mag.structure_utils import read_structure_frames
from quick_mag.extxyz_io import (
    Frame,
    frame_oxidation_states,
    frame_site_labels,
    frame_spin_states,
)
from quick_mag.solve_output import AssignmentResult, StructureResult, write_solve_output
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


def _run_assignment(structure, assignment, args) -> AssignmentResult:
    """Solve one oxidation assignment: print the report and return what was found."""
    result = AssignmentResult(assignment=assignment)
    print(f"\n{'=' * 80}")
    print(
        "Oxidation-state assignment:  "
        f"{format_oxidation_distribution(assignment.distributions)}"
    )
    print(f"  oxidation-model energy: {assignment.total_energy:.4f}")

    model = exchange_model(structure, assignment, valence=getattr(args, "valence", "class"))
    if not model.sites:
        print("  No transition-metal magnetic sites in this assignment; skipping.")
        result.note = "no magnetic sites"
        return result
    magnetic_site_indices = model.sites
    if not model.bridges:
        print("  No exchange couplings (no M-L-M bridges found); skipping solve.")
        result.note = "no exchange couplings"
        return result
    j_matrix = model.solver_J
    _report_spin_states(model)

    site_indexing = _resolve_site_indexing(structure, magnetic_site_indices)

    # J is per moment product, so the solver runs on the actual formal moments (2S of
    # each site's descriptor, low spin included). Reference and solved configs share
    # them, so their energies are comparable.
    moments = np.zeros(structure.atom_count)
    moments[magnetic_site_indices] = model.moments
    unit_assignment = replace(assignment, magnetic_moments=moments)
    result.assignment = unit_assignment
    result.model = model

    # Reference G/C/F/A configurations, in every orientation of the B grid —
    # always reported, independent of the solve.
    reference = named_reference_spin_configs(
        structure, unit_assignment, j_matrix, magnetic_site_indices, site_indexing
    )
    _report_reference_configs(reference)
    result.reference = list(reference)

    n_mag = len(magnetic_site_indices)
    method = "exact" if n_mag <= args.exact_max_sites else "optimizer"
    print(f"\n  Solving {n_mag} magnetic sites with method='{method}'.")
    result.method = method
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
            result.note = "no unpaired electrons"
            return result
        raise
    result.solved = list(all_states)
    if getattr(args, "out", None) is not None:
        # Only the written files need a label per configuration; the report labels the
        # rows it prints itself.
        written = result.solved if args.max_configs <= 0 else result.solved[: args.max_configs]
        result.solved_labels = [
            _label_config(structure, config, magnetic_site_indices, site_indexing)
            for config in written
        ]

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
    return result


SOLVE_DESCRIPTION = (
    "Predict oxidation states, exchange couplings, and low-energy magnetic "
    "configurations for a structure. Always reports the canonical G/C/F/A "
    "reference orderings."
)


def configure_solve_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the ``solve`` pipeline's arguments to ``parser`` (reused by the CLI)."""
    parser.add_argument(
        "structures", type=Path, nargs="*",
        help="Structure file(s): .cif (P1), .vasp/POSCAR or extended xyz (.xyz, every "
        "frame; its 'oxidation' and 'spin_state' columns set the states per atom). Omit "
        "when the structures come from an earlier stage of a '::' chain.",
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
        help="Cap on solved configs printed, and written by --out (0 = no cap; default 20).",
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
    parser.add_argument(
        "--ignore-file-states", action="store_true",
        help="Solve as if an extended xyz input had no 'oxidation' / 'spin_state' columns.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, metavar="DIR",
        help="Also write the results into DIR: configs.xyz (every configuration as an "
        "extended xyz frame), sites.csv, exchange.csv, exchange.graphs.json, summary.json "
        "and chemlab.json, which lists them with their units (spin energies in meV, J in meV/mu_B^2).",
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


def _site_states(args, structure: ChemicalStructure, frame: Optional[Frame]):
    """The ``--ox`` / ``--spin`` rules over the states the structure file carried.

    Returns the states and how many atoms the file gave an oxidation state / a spin state.
    """
    oxidation = getattr(args, "ox", []) or []
    spin = getattr(args, "spin", []) or []
    if frame is None or getattr(args, "ignore_file_states", False):
        return SiteStates.parse(oxidation, spin), 0, 0
    site_oxidation = frame_oxidation_states(frame)
    site_spin = frame_spin_states(frame)
    states = SiteStates.parse(
        oxidation,
        spin,
        site_oxidation=site_oxidation,
        site_spin=site_spin,
        labels=frame_site_labels(frame),
        rules_over_sites=True,
    )
    n_oxidation = sum(1 for v in site_oxidation or [] if v is not None)
    n_spin = sum(1 for v in site_spin or [] if v is not None)
    return states, n_oxidation, n_spin


def solve_structure(structure: ChemicalStructure, args, frame: Optional[Frame] = None) -> int:
    """Run the full magnetism pipeline for one structure and print its report (0 = solved)."""
    return 0 if solve_structure_result(structure, args, frame) is not None else 1


def solve_structure_result(
    structure: ChemicalStructure, args, frame: Optional[Frame] = None
) -> Optional[StructureResult]:
    """:func:`solve_structure`, returning what was found; ``None`` when it cannot be solved.

    Oxidation constraints are the builder's sublattice restrictions (spinels), then the
    per-atom states of the structure file (``frame``: an extended xyz ``oxidation`` column),
    then the user's ``--ox`` rules, each over the one before. Spin states are the file's
    ``spin_state`` column with the ``--spin`` rules over it.
    """
    labels = structure.element_symbols()
    print(f"Structure: {structure.name}  ({structure.atom_count} atoms)")
    print(f"  elements: {', '.join(sorted(set(labels)))}")

    try:
        states, n_file_oxidation, n_file_spin = _site_states(args, structure, frame)
        base = _provenance_oxidation_constraints(structure)
        assignments = ranked_assignments(
            structure, states, charge=args.charge, max_mixing=args.max_mixing,
            top_k=max(1, args.top_k), base_constraints=base,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return None
    if not assignments:
        print("No charge-balanced oxidation-state assignments satisfy the constraints.")
        return None
    if base:
        print(f"  Sublattice oxidation constraints on {len(base)} atom(s) (builder).")
    if n_file_oxidation:
        print(f"  Oxidation states given for {n_file_oxidation} atom(s) (structure file).")
    if n_file_spin:
        print(f"  Spin states given for {n_file_spin} atom(s) (structure file).")
    rules = SiteStates.parse(getattr(args, "ox", []) or [], labels=states.labels).constraints(structure)
    if rules:
        print(f"  User oxidation constraints on {len(rules)} atom(s) (--ox).")
    print(f"  Top {len(assignments)} oxidation assignment(s) will be solved.")
    result = StructureResult(
        structure=structure, frame=frame,
        pinned_oxidation=n_file_oxidation, pinned_spin=n_file_spin,
    )
    for assignment in assignments:
        result.assignments.append(_run_assignment(structure, assignment, args))
    return result


def run_solve(args, structures=None) -> int:
    """Solve every structure named on the command line, or handed over by a chain.

    ``solve`` is always the last stage of a ``::`` chain — spin data feeds nothing
    else — so it writes no structures; it reports, and with ``--out`` writes its
    results as files.
    """
    if structures is None:
        if not args.structures:
            print(
                "Error: no structure given. Pass one or more structure files, or "
                "chain solve after a build/chgnet stage with '::'."
            )
            return 1
        frames: List[Frame] = []
        for path in args.structures:
            try:
                frames.extend(read_structure_frames(path))
            except (OSError, ValueError) as exc:
                print(f"Error: {path}: {exc}")
                return 1
    else:
        frames = [Frame(structure=structure) for structure in structures]

    status = 0
    results: List[StructureResult] = []
    for index, frame in enumerate(frames):
        if index:
            print()
        # Only an extended xyz frame carries per-atom columns; an empty one says nothing.
        carried = frame if (frame.arrays or frame.info) else None
        result = solve_structure_result(frame.structure, args, carried)
        if result is None:
            status = 1
        else:
            results.append(result)

    out_dir = getattr(args, "out", None)
    if out_dir is not None and results:
        manifest = write_solve_output(
            Path(out_dir).expanduser(), results,
            max_configs=args.max_configs,
            command="quick-mag " + " ".join(shlex.quote(a) for a in sys.argv[1:]),
        )
        print(f"\nWrote results -> {manifest.parent} ({manifest.name} lists them)")
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=SOLVE_DESCRIPTION)
    configure_solve_parser(parser)
    args = parser.parse_args(argv)
    return run_solve(args)


if __name__ == "__main__":
    raise SystemExit(main())
