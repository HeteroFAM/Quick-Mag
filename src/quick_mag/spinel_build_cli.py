#!/usr/bin/env python3
"""Command-line spinel builder: ``quick-mag build-spinel ...``.

The spinel counterpart of ``quick-mag build`` (see :mod:`quick_mag.build_cli`),
exposing :func:`quick_mag.generation.generate_spinel` and
:func:`quick_mag.generation.generate_high_entropy_spinel`. It shares that
command's conventions:

* **Scans** -- ``--a``, ``--c``, ``--u``, ``--inversion`` and ``--n-cells-*``
  accept an inclusive ``start:stop:num_steps`` spec. Scanned axes form a
  Cartesian grid, or advance in lockstep with ``--zip``.
* **Element combinations** -- in ``--formula spinel`` the ``--a-site`` /
  ``--b-site`` / ``--x-site`` flags take comma-separated lists, and the full
  Cartesian product is built.
* **High-entropy sampling** -- ``--formula high_entropy`` takes per-sublattice
  ``El:weight`` mixes (``--a-sites`` tetrahedral, ``--b-sites`` octahedral,
  ``--x-sites`` anion), *or* one ``--cations`` pool that ``--site-preference``
  splits over the two sublattices. ``--num-samples`` draws that many
  reproducible occupancy realizations and ``--seed`` shifts the family.

The generated set is ``element-combinations x scan-points x samples``.
``--n-cells-*`` count repeats of the chosen ``--cell`` (the 56-atom
conventional cell by default, or the 14-atom primitive one).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from quick_mag.build_cli import (
    DEFAULT_OUTPUT_DIR,
    parse_element_list,
    parse_high_entropy_distribution,
    parse_scan_spec,
)
from quick_mag.export_utils import (
    FORMAT_CHOICES,
    FORMAT_HELP,
    export_options,
    export_structure,
    sanitize_filename,
)
from quick_mag.generation import (
    generate_high_entropy_spinel,
    generate_spinel,
    normalize_element_symbol,
)
from quick_mag.spinel_builder import (
    CELL_KINDS,
    DEFAULT_LATTICE_CONSTANT,
    DEFAULT_U,
)
from quick_mag.spinel_composition import (
    DEFAULT_TEMPERATURE_K,
    distribute_cations,
    format_sublattice_composition,
)
from quick_mag.structure import SPINEL_SAMPLING_MODES

_STRUCTURAL_AXES = ("a", "c", "u", "inversion", "n_cells_x", "n_cells_y", "n_cells_z")
_AXIS_TAG = {
    "a": "a", "c": "c", "u": "u", "inversion": "x",
    "n_cells_x": "nx", "n_cells_y": "ny", "n_cells_z": "nz",
}
_INTEGER_AXES = {"n_cells_x", "n_cells_y", "n_cells_z"}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_site_preferences(value) -> Dict[str, object]:
    """Parse ``El:tet,El:oct,El:-12.5`` into ``distribute_cations`` preferences.

    A number is the site energy ``E_tet - E_oct`` in kJ/mol (negative favours
    the tetrahedral site).
    """
    preferences: Dict[str, object] = {}
    if value is None:
        return preferences
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"--site-preference entry '{token}' should be El:tet, El:oct, or El:ENERGY."
            )
        symbol_text, _, pref_text = token.partition(":")
        symbol = normalize_element_symbol(symbol_text)
        pref = pref_text.strip().lower()
        if pref in ("tet", "oct"):
            preferences[symbol] = pref
            continue
        try:
            preferences[symbol] = float(pref)
        except ValueError:
            raise ValueError(
                f"--site-preference '{token}': expected 'tet', 'oct', or an energy in kJ/mol."
            )
    return preferences


def parse_oxidation_restrictions(values) -> Dict[str, Dict[str, Tuple[int, ...]]]:
    """Parse repeated ``ROLE:El=s1,s2`` flags into ``sublattice_oxidation_states``.

    E.g. ``--oxidation A:Fe=3 --oxidation B:Fe=2,3`` for magnetite.
    """
    result: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    for raw in values or []:
        text = str(raw).strip()
        role_text, sep, rest = text.partition(":")
        element_text, sep2, states_text = rest.partition("=")
        if not sep or not sep2:
            raise ValueError(f"--oxidation '{raw}' should look like ROLE:El=STATES, e.g. A:Fe=3.")
        role = role_text.strip().upper()
        if role not in ("A", "B", "X"):
            raise ValueError(f"--oxidation '{raw}': role must be A, B, or X.")
        element = normalize_element_symbol(element_text)
        try:
            states = tuple(sorted({int(s.strip().rstrip("+")) for s in states_text.split(",") if s.strip()}))
        except ValueError:
            raise ValueError(f"--oxidation '{raw}': states must be integers, e.g. B:Fe=2,3.")
        if not states:
            raise ValueError(f"--oxidation '{raw}' lists no states.")
        result.setdefault(role, {})[element] = states
    return result


def _structural_axes(args: argparse.Namespace) -> Dict[str, list]:
    axes: Dict[str, list] = {
        "a": parse_scan_spec(args.a, name="--a"),
        "c": parse_scan_spec(args.c, name="--c") if args.c is not None else [None],
        "u": parse_scan_spec(args.u, name="--u"),
        "inversion": parse_scan_spec(args.inversion, name="--inversion"),
    }
    for axis in ("x", "y", "z"):
        axes[f"n_cells_{axis}"] = parse_scan_spec(
            getattr(args, f"n_cells_{axis}"), name=f"--n-cells-{axis}", integer=True
        )
    for value in axes["inversion"]:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--inversion must lie in [0, 1]; got {value}.")
    return axes


def _structural_points(axes: Dict[str, list], zip_scans: bool) -> List[Dict[str, object]]:
    if zip_scans:
        scanned = [name for name in _STRUCTURAL_AXES if len(axes[name]) > 1]
        lengths = {len(axes[name]) for name in scanned}
        if len(lengths) > 1:
            detail = ", ".join(f"{name}={len(axes[name])}" for name in scanned)
            raise ValueError(f"--zip requires all scanned axes to share num_steps; got {detail}.")
        length = lengths.pop() if scanned else 1
        return [
            {
                name: (axes[name][i] if len(axes[name]) > 1 else axes[name][0])
                for name in _STRUCTURAL_AXES
            }
            for i in range(length)
        ]
    return [
        dict(zip(_STRUCTURAL_AXES, combo))
        for combo in itertools.product(*(axes[name] for name in _STRUCTURAL_AXES))
    ]


def _high_entropy_mixes(args: argparse.Namespace) -> Dict[str, list]:
    x_sites = parse_high_entropy_distribution(args.x_sites, name="--x-sites")
    if args.cations is not None:
        if args.a_sites is not None or args.b_sites is not None:
            raise ValueError("Give either --cations or --a-sites/--b-sites, not both.")
        distribution = distribute_cations(
            parse_high_entropy_distribution(args.cations, name="--cations"),
            parse_site_preferences(args.site_preference),
            temperature=float(args.temperature),
        )
        return {"a_sites": distribution.a_sites, "b_sites": distribution.b_sites, "x_sites": x_sites}
    if args.site_preference is not None:
        raise ValueError("--site-preference only applies together with --cations.")
    if args.a_sites is None or args.b_sites is None:
        raise ValueError(
            "--formula high_entropy needs --a-sites and --b-sites, or a --cations pool."
        )
    return {
        "a_sites": parse_high_entropy_distribution(args.a_sites, name="--a-sites"),
        "b_sites": parse_high_entropy_distribution(args.b_sites, name="--b-sites"),
        "x_sites": x_sites,
    }


def _format_mix(entries) -> str:
    return ",".join(f"{symbol}:{fraction:.4g}" for symbol, fraction in entries)


def _structure_name(
    *,
    root: str,
    elem_combo: Dict[str, str],
    point: Dict[str, object],
    varying_axes: List[str],
    sample: int,
    num_samples: int,
    append_elements: bool,
    seed: int,
) -> str:
    parts = [root]
    if append_elements:
        parts.append("".join(elem_combo.values()))
    for axis in varying_axes:
        value = point[axis]
        if value is None:
            continue
        tag = _AXIS_TAG[axis]
        parts.append(f"{tag}{int(value)}" if axis in _INTEGER_AXES else f"{tag}{float(value):.3g}")
    if seed:
        parts.append(f"seed{int(seed)}")
    if num_samples > 1:
        width = len(str(num_samples - 1))
        parts.append(f"s{sample:0{width}d}")
    return "_".join(part for part in parts if part)


# ---------------------------------------------------------------------------
# Structure construction
# ---------------------------------------------------------------------------


def build_spinel_structures(args: argparse.Namespace) -> List:
    """Expand a parsed ``build-spinel`` namespace into the structures it describes.

    Pure: nothing is written to disk. Raises ``ValueError`` for any invalid
    argument combination.
    """
    if args.sampling not in SPINEL_SAMPLING_MODES:
        raise ValueError(f"--sampling must be one of {SPINEL_SAMPLING_MODES}.")
    oxidation = parse_oxidation_restrictions(args.oxidation)
    axes = _structural_axes(args)
    points = _structural_points(axes, args.zip_scans)

    high_entropy = args.formula == "high_entropy"
    if high_entropy:
        if any(value != 0.0 for value in axes["inversion"]):
            raise ValueError(
                "--inversion applies to --formula spinel; a high-entropy spinel's "
                "cation distribution is set by its sublattice mixes (or --cations "
                "with --site-preference)."
            )
        mixes: Optional[Dict[str, list]] = _high_entropy_mixes(args)
        element_combos: List[Dict[str, str]] = [{}]
    else:
        stray = [
            flag
            for flag, given in (
                ("--a-sites", args.a_sites is not None),
                ("--b-sites", args.b_sites is not None),
                ("--cations", args.cations is not None),
                ("--site-preference", args.site_preference is not None),
                ("--sampling", args.sampling != "exact"),
            )
            if given
        ]
        if stray:
            raise ValueError(
                f"{', '.join(stray)} only apply to --formula high_entropy."
            )
        mixes = None
        sites = ("a_site", "b_site", "x_site")
        per_site = [
            parse_element_list(getattr(args, site), name=f"--{site.replace('_', '-')}")
            for site in sites
        ]
        element_combos = [dict(zip(sites, combo)) for combo in itertools.product(*per_site)]

    # Inversion draws, like high-entropy ones, are sampled, so --num-samples and
    # --seed apply to both modes; they only change anything when something is
    # actually drawn.
    num_samples = max(1, int(args.num_samples))
    seed = int(args.seed)
    varying_axes = [name for name in _STRUCTURAL_AXES if len(axes[name]) > 1]
    append_elements = (
        not high_entropy and (args.name is not None) and len(element_combos) > 1
    )

    used_names: set[str] = set()
    structures = []
    for elem_combo in element_combos:
        if args.name is not None:
            root = args.name
        elif high_entropy:
            root = "HES"
        else:
            root = f"{elem_combo['a_site']}{elem_combo['b_site']}2{elem_combo['x_site']}4"
        for point in points:
            common = dict(
                a=float(point["a"]),
                c=None if point["c"] is None else float(point["c"]),
                u=float(point["u"]),
                n_cells_x=int(point["n_cells_x"]),
                n_cells_y=int(point["n_cells_y"]),
                n_cells_z=int(point["n_cells_z"]),
                cell=args.cell,
                seed=seed,
                sublattice_oxidation_states=oxidation,
            )
            for sample in range(num_samples):
                name = _structure_name(
                    root=root,
                    elem_combo=elem_combo,
                    point=point,
                    varying_axes=varying_axes,
                    sample=sample,
                    num_samples=num_samples,
                    append_elements=append_elements,
                    seed=seed,
                )
                unique = name
                suffix = 2
                while sanitize_filename(unique) in used_names:
                    unique = f"{name}_{suffix}"
                    suffix += 1
                used_names.add(sanitize_filename(unique))

                if high_entropy:
                    assert mixes is not None
                    structure = generate_high_entropy_spinel(
                        unique,
                        a_sites=mixes["a_sites"],
                        b_sites=mixes["b_sites"],
                        x_sites=mixes["x_sites"],
                        sampling=args.sampling,
                        sample_index=sample,
                        **common,
                    )
                else:
                    structure = generate_spinel(
                        unique,
                        inversion=float(point["inversion"]),
                        sample_index=sample,
                        **elem_combo,
                        **common,
                    )
                structures.append(structure)
    return structures


def report_spinel_build(args: argparse.Namespace, structures: List, *, write: bool = True) -> None:
    """Print one line per structure (with its sublattice occupancy) and write it."""
    to_disk = (write or args.output_dir is not None) and not args.dry_run
    out_dir = Path(args.output_dir or DEFAULT_OUTPUT_DIR).expanduser()
    if to_disk:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.formula == "high_entropy" and structures:
        params = structures[0].generation_parameters
        print(f"  tetrahedral (A) mix: {_format_mix(params.high_entropy_a_sites)}")
        print(f"  octahedral  (B) mix: {_format_mix(params.high_entropy_b_sites)}")

    verb = "would build" if args.dry_run else ("wrote" if to_disk else "built")
    for structure in structures:
        if to_disk:
            export_structure(structure, out_dir, **export_options(args))
        print(
            f"  {verb} {structure.name} ({structure.atom_count} atoms; "
            f"{format_sublattice_composition(structure)})"
        )
    where = f" under {out_dir}" if to_disk else ""
    print(f"Done. {verb} {len(structures)} structure(s){where}.")


def run_spinel_build(args: argparse.Namespace) -> int:
    try:
        structures = build_spinel_structures(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    report_spinel_build(args, structures)
    return 0


SPINEL_BUILD_DESCRIPTION = (
    "Build spinel (AB2X4) structures -- normal, (partially) inverse, or "
    "high-entropy -- and write them as CIFs. Structural variables and the "
    "inversion degree accept inclusive 'start:stop:num_steps' scans, and "
    "single-element site flags accept comma-separated element lists."
)


def configure_spinel_build_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the ``build-spinel`` command's arguments to ``parser``."""
    parser.add_argument(
        "--formula", choices=["spinel", "high_entropy"], default="spinel",
        help="'spinel' for A B2 X4 with an inversion degree, or 'high_entropy' "
        "for per-sublattice element mixes (default spinel).",
    )

    sites = parser.add_argument_group("elements, --formula spinel (comma lists -> Cartesian product)")
    sites.add_argument("--a-site", default="Co", help="Tetrahedral (A) element(s) (default Co).")
    sites.add_argument("--b-site", default="Fe", help="Octahedral (B) element(s) (default Fe).")
    sites.add_argument("--x-site", default="O", help="Anion element(s) (default O).")
    sites.add_argument(
        "--inversion", default="0.0",
        help="Inversion degree x in (A_{1-x}B_x)[A_x B_{2-x}]X4: 0 normal, 1 inverse "
        "(default 0; scannable).",
    )

    he = parser.add_argument_group("high-entropy sublattices ('El:weight' comma lists)")
    he.add_argument("--a-sites", default=None, help="Tetrahedral (A) sublattice mix.")
    he.add_argument("--b-sites", default=None, help="Octahedral (B) sublattice mix.")
    he.add_argument("--x-sites", default="O", help="Anion mix (default O).")
    he.add_argument(
        "--cations", default=None,
        help="One cation pool (e.g. Co,Cr,Fe,Mn,Ni) to split over both sublattices "
        "instead of --a-sites/--b-sites.",
    )
    he.add_argument(
        "--site-preference", default=None,
        help="Per-cation placement for --cations: El:tet, El:oct, or El:ENERGY with "
        "ENERGY = E_tet - E_oct in kJ/mol (negative favours tetrahedral). "
        "Unlisted cations have no preference.",
    )
    he.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE_K,
        help=f"Temperature (K) for energy site preferences (default {DEFAULT_TEMPERATURE_K}).",
    )
    he.add_argument(
        "--sampling", choices=list(SPINEL_SAMPLING_MODES), default="exact",
        help="'exact' places exactly the mix's composition (rounded to whole atoms); "
        "'iid' draws every site independently (default exact).",
    )

    sampling = parser.add_argument_group("sampling (high-entropy mixes and inversion)")
    sampling.add_argument(
        "--num-samples", type=int, default=1,
        help="Occupancy realizations to draw per structure (default 1).",
    )
    sampling.add_argument(
        "--seed", type=int, default=0,
        help="Base seed for occupancy sampling (default 0).",
    )

    struct = parser.add_argument_group(
        "structural variables (scalar or 'start:stop:num_steps' scan)"
    )
    struct.add_argument(
        "--a", default=str(DEFAULT_LATTICE_CONSTANT),
        help=f"Conventional cubic edge a in Angstrom (default {DEFAULT_LATTICE_CONSTANT}).",
    )
    struct.add_argument("--c", default=None, help="Conventional edge c for a tetragonal cell (default: follow a).")
    struct.add_argument(
        "--u", default=str(DEFAULT_U),
        help=f"Anion parameter u, origin choice 2 (ideal 0.25; default {DEFAULT_U}).",
    )
    struct.add_argument(
        "--cell", choices=list(CELL_KINDS), default="conventional",
        help="Repeat the 56-atom conventional cell or the 14-atom primitive cell "
        "(default conventional).",
    )
    for axis in ("x", "y", "z"):
        struct.add_argument(
            f"--n-cells-{axis}", default="1",
            help=f"Repeats of --cell along {axis} (default 1).",
        )

    parser.add_argument(
        "--oxidation", action="append", metavar="ROLE:El=STATES",
        help="Restrict an element's oxidation states on one sublattice (repeatable), "
        "e.g. --oxidation A:Fe=3 --oxidation B:Fe=2,3. Stored with the structure and "
        "honoured by a chained 'solve'.",
    )
    parser.add_argument(
        "--zip", dest="zip_scans", action="store_true",
        help="Advance scanned axes in lockstep instead of a Cartesian grid.",
    )
    parser.add_argument("--name", default=None, help="Base name for outputs.")
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help=f"Directory to write structures into (default {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--format", choices=sorted(FORMAT_CHOICES), default="cif", help=FORMAT_HELP)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the structures that would be built without writing files.",
    )
    return parser


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=SPINEL_BUILD_DESCRIPTION)
    configure_spinel_build_parser(parser)
    args = parser.parse_args(argv)
    return run_spinel_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
