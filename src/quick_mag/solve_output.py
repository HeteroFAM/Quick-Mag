"""Machine-readable output of ``quick-mag solve --out DIR``.

``solve`` prints a report; this writes the same results as files another program can
read without parsing that report:

    configs.xyz            every spin configuration as an extended xyz frame: the reference
                           orderings and the solved ones, of every oxidation assignment solved.
                           Comment line: ``assignment`` (1-based), ``source`` (reference | solved),
                           ``rank`` (1 = lowest solved; 0 for a reference), ``label`` (G, C(a), ...;
                           the ferromagnetic ordering is written ``FM`` in every file here, a
                           bare ``F`` being extended xyz for "false"),
                           ``energy``, ``delta_energy`` (meV; above the lowest of its assignment),
                           ``magnetization``, ``n_unpaired``, ``degeneracy``.
                           Columns: ``magmom`` (signed, mu_B), ``oxidation``, ``spin_state`` -- the
                           states the model used, whether they were given or worked out -- and every
                           column the input file carried.
    sites.csv              one row per atom, in the input's order: the oxidation state, spin state
                           and formal moment of each assignment.
    exchange.csv           the couplings, one row per coupled pair and assignment.
    exchange.graphs.json   the same as graphs on the atoms ("chemlab-graphs/1"): an edge per
                           coupled pair, its value J in meV/mu_B^2 (> 0 antiferromagnetic).
    summary.json           per assignment: the oxidation distribution, energies, the ground state.
    chemlab.json           what the files above are (kind, units), for a program that collects
                           results by reading one manifest.

With several input structures every file name is prefixed with the structure's position and
name (``001_LaMnO3_configs.xyz``) and one ``chemlab.json`` lists them all.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quick_mag.export_utils import sanitize_filename
from quick_mag.extxyz_io import (
    OXIDATION_COLUMN,
    SPIN_COLUMN,
    Frame,
    write_extxyz,
)
from quick_mag.magnetic_moments import format_oxidation_distribution
from quick_mag.structure import ChemicalStructure

MANIFEST_NAME = "chemlab.json"
MANIFEST_VERSION = 1

#: The units of what is written. The model's energies are in the exchange parameters'
#: unit, nominally eV (``polarization_model.ExchangeModel``); the files give the spin
#: energies and the couplings in meV, the scale they live on. J is per product of moments.
ENERGY_UNIT = "meV"
ENERGY_SCALE = 1000.0          # model energy (eV) -> written energy (meV)
MOMENT_UNIT = "mu_B"
COUPLING_UNIT = "meV/mu_B^2"
DIMENSIONLESS = "[1]"

# Comment-line keys this module sets on a configuration's frame; the input's own keys of
# the same name do not survive.
_CONFIG_KEYS = (
    "assignment", "source", "rank", "label", "energy", "delta_energy",
    "magnetization", "n_unpaired", "degeneracy",
)


@dataclass
class AssignmentResult:
    """What ``solve`` found for one oxidation assignment of one structure."""

    assignment: Any                                   # OxidationStateAssignment (formal moments set)
    model: Any = None                                 # ExchangeModel, or None when nothing is magnetic
    reference: List[Tuple[str, Any]] = field(default_factory=list)   # (name, SpinConfig)
    solved: List[Any] = field(default_factory=list)                  # SpinConfig, lowest first
    solved_labels: List[str] = field(default_factory=list)
    method: str = ""
    note: str = ""                                    # why there are no configurations, when there are none


@dataclass
class StructureResult:
    structure: ChemicalStructure
    frame: Optional[Frame] = None                     # the input frame, when it came from extended xyz
    assignments: List[AssignmentResult] = field(default_factory=list)
    pinned_oxidation: int = 0                         # atoms whose oxidation state came with the file
    pinned_spin: int = 0


def file_label(label: str) -> str:
    """A configuration's label as written to files: ``F`` becomes ``FM`` (see the module
    docstring), an empty one ``-``."""
    label = (label or "").strip()
    if label in ("F", "T"):
        return label + "M" if label == "F" else label + "_"
    return label or "-"


def _full_moments(structure: ChemicalStructure, sites: Sequence[int], config) -> np.ndarray:
    """A configuration's signed moments on every atom, as (n_atoms, 3) along z."""
    moments = np.zeros((structure.atom_count, 3))
    values = np.asarray(config.all_moments, dtype=float)
    if values.ndim == 2:                              # per-site vectors
        for k, site in enumerate(sites):
            if k < len(values):
                moments[int(site), : values.shape[1]] = values[k]
        return moments
    values = values.reshape(-1)
    for k, site in enumerate(sites):
        if k < values.size:
            moments[int(site), 2] = values[k]
    return moments


def _spin_column(structure: ChemicalStructure, assignment) -> List[str]:
    states = getattr(assignment, "spin_states", None) or {}
    return [str(states.get(atom, "-")) for atom in range(structure.atom_count)]


def _oxidation_column(structure: ChemicalStructure, assignment) -> np.ndarray:
    return np.asarray(assignment.site_oxidation_states).reshape(-1).astype(int)[: structure.atom_count]


def _config_frames(result: StructureResult, max_configs: int) -> List[Frame]:
    structure = result.structure
    carried_info = dict(getattr(result.frame, "info", None) or {})
    carried_arrays = dict(getattr(result.frame, "arrays", None) or {})
    for key in _CONFIG_KEYS:
        carried_info.pop(key, None)
    frames: List[Frame] = []
    for number, item in enumerate(result.assignments, start=1):
        if item.model is None:
            continue
        sites = list(item.model.sites)
        solved = item.solved if max_configs <= 0 else item.solved[:max_configs]
        entries = [("reference", 0, name, config) for name, config in item.reference]
        entries += [
            ("solved", rank, item.solved_labels[rank - 1] if rank - 1 < len(item.solved_labels) else "", config)
            for rank, config in enumerate(solved, start=1)
        ]
        if not entries:
            continue
        lowest = min(config.energy for _s, _r, _l, config in entries)
        arrays = dict(carried_arrays)
        arrays[OXIDATION_COLUMN] = _oxidation_column(structure, item.assignment)
        arrays[SPIN_COLUMN] = np.array(_spin_column(structure, item.assignment))
        for source, rank, label, config in entries:
            copy = ChemicalStructure(
                name=structure.name,
                lattice=np.array(structure.lattice, dtype=float, copy=True),
                cartesian_coords=np.array(structure.cartesian_coords, dtype=float, copy=True),
                atomic_labels=list(structure.atomic_labels),
                magnetic_moments=_full_moments(structure, sites, config),
                is_periodic=structure.is_periodic,
                periodic_axes=structure.periodic_axes,
                generation_parameters=structure.generation_parameters,
                geometry_matches_generation=structure.geometry_matches_generation,
            )
            info = dict(carried_info)
            info.update({
                "assignment": number,
                "source": source,
                "rank": rank,
                "label": file_label(label),
                "energy": float(config.energy) * ENERGY_SCALE,
                "delta_energy": float(config.energy - lowest) * ENERGY_SCALE,
                "magnetization": float(config.magnetization),
                "n_unpaired": float(config.n_unpaired),
                "degeneracy": int(getattr(config, "degeneracy", 1)),
            })
            frames.append(Frame(structure=copy, info=info, arrays=arrays))
    return frames


def _minimum_image_distance(structure: ChemicalStructure, i: int, j: int) -> float:
    delta = np.asarray(structure.cartesian_coords[j], dtype=float) - np.asarray(structure.cartesian_coords[i], dtype=float)
    if not structure.is_periodic:
        return float(np.linalg.norm(delta))
    lattice = np.asarray(structure.lattice, dtype=float)
    frac = np.linalg.solve(lattice.T, delta)
    axes = structure.periodic_axes or (True, True, True)
    best = np.inf
    # The nearest image is within one cell of the rounded fractional difference.
    base = np.array([np.round(f) if periodic else 0.0 for f, periodic in zip(frac, axes)])
    shifts = [(-1, 0, 1) if periodic else (0,) for periodic in axes]
    for a in shifts[0]:
        for b in shifts[1]:
            for c in shifts[2]:
                image = frac - base + np.array([a, b, c], dtype=float)
                best = min(best, float(np.linalg.norm(image @ lattice)))
    return best


def _coupled_pairs(model, threshold: float = 1e-12) -> List[Tuple[int, int, float]]:
    """(atom i, atom j, J in meV/mu_B^2) for every coupled pair, i < j, J > 0 antiferromagnetic."""
    sites = list(model.sites)
    J = np.asarray(model.J, dtype=float)
    pairs = []
    for a in range(len(sites)):
        for b in range(a + 1, len(sites)):
            if abs(J[a, b]) > threshold:
                pairs.append((int(sites[a]), int(sites[b]), float(J[a, b]) * ENERGY_SCALE))
    return pairs


def _write_sites(path: Path, result: StructureResult) -> List[str]:
    structure = result.structure
    symbols = structure.element_symbols()
    many = len(result.assignments) > 1
    header = ["atom", "species"]
    columns: List[List[Any]] = []
    for number, item in enumerate(result.assignments, start=1):
        suffix = f"_{number}" if many else ""
        header += [f"oxidation{suffix}", f"spin_state{suffix}", f"moment{suffix}"]
        moments = np.asarray(item.assignment.magnetic_moments, dtype=float).reshape(-1)
        columns.append([
            _oxidation_column(structure, item.assignment),
            _spin_column(structure, item.assignment),
            moments,
        ])
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for atom in range(structure.atom_count):
            row: List[Any] = [atom, symbols[atom]]
            for oxidation, spin, moments in columns:
                row += [int(oxidation[atom]), spin[atom], f"{float(moments[atom]):g}"]
            writer.writerow(row)
    return header


def _write_exchange(csv_path: Path, graphs_path: Path, result: StructureResult) -> int:
    structure = result.structure
    symbols = structure.element_symbols()
    graphs: Dict[str, Any] = {}
    count = 0
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["assignment", "i", "j", "species_i", "species_j", "J", "distance"])
        for number, item in enumerate(result.assignments, start=1):
            if item.model is None:
                continue
            pairs = _coupled_pairs(item.model)
            edges = []
            for i, j, value in pairs:
                writer.writerow([
                    number, i, j, symbols[i], symbols[j], repr(value),
                    f"{_minimum_image_distance(structure, i, j):.6f}",
                ])
                edges.append({"a": i, "b": j, "value": value, "kind": "afm" if value > 0 else "fm"})
            count += len(pairs)
            largest = max((abs(v) for _i, _j, v in pairs), default=0.0)
            graph: Dict[str, Any] = {"nodes": "atoms", "colormap": "diverging", "edges": edges}
            if largest > 0:
                graph["range"] = [-largest, largest]
            graphs["exchange" if number == 1 else f"exchange_{number}"] = graph
    graphs_path.write_text(json.dumps({"format": "chemlab-graphs/1", "graphs": graphs}, indent=1) + "\n")
    return count


def _summary(result: StructureResult) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "structure": result.structure.name,
        "atoms": int(result.structure.atom_count),
        "oxidation_from_file": int(result.pinned_oxidation),
        "spin_states_from_file": int(result.pinned_spin),
        "assignments": [],
    }
    for number, item in enumerate(result.assignments, start=1):
        entry: Dict[str, Any] = {
            "assignment": number,
            "oxidation": format_oxidation_distribution(item.assignment.distributions),
            "oxidation_energy": float(item.assignment.total_energy),
            "magnetic_sites": 0 if item.model is None else len(item.model.sites),
            "method": item.method,
            "references": {file_label(name): float(config.energy) * ENERGY_SCALE for name, config in item.reference},
            "solved": len(item.solved),
        }
        if item.note:
            entry["note"] = item.note
        candidates = [("solved", label, config) for label, config in zip(
            item.solved_labels + [""] * len(item.solved), item.solved)]
        candidates += [("reference", name, config) for name, config in item.reference]
        if candidates:
            source, label, ground = min(candidates, key=lambda c: c[2].energy)
            entry["ground"] = {
                "energy": float(ground.energy) * ENERGY_SCALE,
                "label": file_label(label),
                "source": source,
                "magnetization": float(ground.magnetization),
            }
        out["assignments"].append(entry)
    return out


def write_solve_output(
    out_dir: Path, results: Sequence[StructureResult], *, max_configs: int = 0, command: str = ""
) -> Path:
    """Write every result into ``out_dir`` and return the manifest's path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    many = len(results) > 1
    items: List[Dict[str, Any]] = []
    for position, result in enumerate(results, start=1):
        stem = sanitize_filename(result.structure.name)
        file_prefix = f"{position:03d}_{stem}_" if many else ""
        item_prefix = f"{position:03d}_{stem}." if many else ""

        frames = _config_frames(result, max_configs)
        if frames:
            name = f"{file_prefix}configs.xyz"
            write_extxyz(out_dir / name, frames)
            items.append({
                "name": f"{item_prefix}configs", "kind": "structures", "file": name,
                "frames": len(frames),
                "units": {
                    "energy": ENERGY_UNIT, "delta_energy": ENERGY_UNIT,
                    "magnetization": MOMENT_UNIT, "n_unpaired": DIMENSIONLESS,
                    "magmom": MOMENT_UNIT, OXIDATION_COLUMN: DIMENSIONLESS,
                },
            })

        name = f"{file_prefix}sites.csv"
        header = _write_sites(out_dir / name, result)
        units = {
            column: (MOMENT_UNIT if column.startswith("moment") else DIMENSIONLESS)
            for column in header
            if column.startswith(("moment", "oxidation"))
        }
        items.append({"name": f"{item_prefix}sites", "kind": "table", "level": "atoms", "file": name, "units": units})

        if any(item.model is not None for item in result.assignments):
            pairs_name = f"{file_prefix}exchange.csv"
            graphs_name = f"{file_prefix}exchange.graphs.json"
            pairs = _write_exchange(out_dir / pairs_name, out_dir / graphs_name, result)
            items.append({
                "name": f"{item_prefix}exchange", "kind": "graph", "level": "node", "file": graphs_name,
                "units": {"value": COUPLING_UNIT},
            })
            items.append({
                "name": f"{item_prefix}exchange_pairs", "kind": "table", "file": pairs_name, "rows": pairs,
                "units": {"J": COUPLING_UNIT, "distance": "angstrom"},
            })

        name = f"{file_prefix}summary.json"
        (out_dir / name).write_text(json.dumps(_summary(result), indent=1) + "\n")
        items.append({
            "name": f"{item_prefix}summary", "kind": "values", "level": "node", "file": name,
            "units": {"energy": ENERGY_UNIT, "oxidation_energy": "unknown", "magnetization": MOMENT_UNIT},
        })

    manifest = {"chemlab-result": MANIFEST_VERSION, "program": "quick-mag solve", "items": items}
    if command:
        manifest["command"] = command
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return path
