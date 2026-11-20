"""Fit systems from the magnetic-configuration HDF5 sets (``quick_magnetic_configs/assets/*_configs.h5``).

Layout of a set: ``/<material>/<structure>/config*``. A structure group carries one geometry (attrs ``poscar``,
``is_md``, ``is_strained``, ...); its configs carry ``converged`` / ``is_outlier`` attrs and datasets
``magnetic_moments``, ``energies/free_energy``, ``spin_configuration``. Structure names look like
``2x2x2``, ``2x2x2__md_base_md_step00500``, ``superhex_strain_a1.0_b0.0_c0.0``,
``2x2x3__md_base_md_step00500_strain_a-1.0_b0.0_c0.0``: a supercell, an optional MD snapshot, an optional strain.

Each structure becomes one :class:`~quick_mag.fitting.FitSystem`:

* configurations: converged, not flagged outliers; then dropped if any magnetic site's |m| strays more than
  ``max_moment_dev`` from that site's median over the structure's configs (a changed charge or spin state is
  not a spin flip), then trimmed to those within ``max_span`` eV/site of the lowest (high-energy
  configurations are where Heisenberg fails first). A structure needs ``min_configs`` survivors.
* oxidation and spin states: the ``top_k`` charge-balanced assignments (under ``states``, if given) times each
  of ``spin_options`` (spin-rule lists for :class:`~quick_mag.site_states.SiteStates`), keeping the one whose
  formal moments best match the DFT moments up to a common scale (covalency shrinks every DFT moment).
* signs: of the DFT moments on the model's magnetic sites (not the requested ``spin_configuration``).

Building bridges is the slow part (STO overlaps included); ``cache`` pickles the systems, keyed by the file's
size and mtime and the loader's options.
"""

from __future__ import annotations

import hashlib
import io
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from quick_mag.fitting import FitSystem, fit_system

__all__ = ["StructureTag", "parse_structure_name", "load_h5_systems", "valence_split"]

_STRAIN = re.compile(r"strain_a(-?[\d.]+)_b(-?[\d.]+)_c(-?[\d.]+)")
_MD = re.compile(r"md_step(\d+)")


@dataclass(frozen=True)
class StructureTag:
    material: str
    supercell: str                 # 2x2x2 / 2x2x3 / superhex
    md_step: Optional[int]         # None for the relaxed geometry
    strain: tuple                  # (a, b, c) in percent; (0, 0, 0) unstrained

    @property
    def strain_label(self) -> str:
        return "unstrained" if not any(self.strain) else "a{:+g}b{:+g}c{:+g}".format(*self.strain)

    @property
    def volume_strain(self) -> float:
        """Linear volume strain, percent (sum of the axial strains)."""
        return float(sum(self.strain))


def parse_structure_name(material: str, name: str) -> StructureTag:
    m = _STRAIN.search(name)
    strain = tuple(float(v) for v in m.groups()) if m else (0.0, 0.0, 0.0)
    md = _MD.search(name)
    return StructureTag(material, re.split(r"__|_strain", name)[0], int(md.group(1)) if md else None, strain)


def _structure(poscar: str, name: str):
    from ase.io import read

    from quick_mag.structure import ChemicalStructure

    atoms = read(io.StringIO(poscar), format="vasp")
    return ChemicalStructure.with_zero_magnetic_moments(
        name=name, lattice=np.array(atoms.cell), cartesian_coords=atoms.positions.copy(),
        atomic_labels=list(atoms.get_chemical_symbols()), is_periodic=True)


def _choose_assignment(structure, dft_abs_m, states, spin_options, top_k):
    """The (assignment, score) whose formal moments match |m_DFT| best after a common scale."""
    from dataclasses import replace

    from quick_mag.site_states import SiteStates, ranked_assignments

    best = None
    for spins in spin_options:
        st = states if spins is None else replace(states or SiteStates(), spin_rules=SiteStates.parse([], spins).spin_rules)
        try:
            cands = ranked_assignments(structure, st, top_k=top_k)
        except ValueError:
            continue
        for a in cands:
            formal = np.abs(np.asarray(a.magnetic_moments, dtype=float))
            mag = formal > 1e-8
            if not mag.any():
                continue
            c = float(dft_abs_m[mag] @ formal[mag] / (formal[mag] @ formal[mag]))
            res = dft_abs_m - c * formal
            # a site the assignment calls non-magnetic but DFT finds magnetic is a full miss
            score = float(np.sqrt(np.mean(res ** 2)))
            if best is None or score < best[1]:
                best = (a, score, c)
    if best is None:
        raise ValueError("no oxidation/spin assignment")
    return best


def load_h5_systems(
    path,
    *,
    states=None,
    spin_options: Sequence[Optional[Sequence[str]]] = (None,),
    top_k: int = 4,
    valence: str = "site",
    max_moment_dev: float = 0.5,
    max_span: float = 0.3,
    min_configs: int = 2,
    energy: str = "free_energy",
    select: Optional[Callable[[StructureTag], bool]] = None,
    group: Callable[[StructureTag], str] = lambda t: t.strain_label,
    cache: Optional[str] = None,
    max_valence_split: Optional[int] = 1,
    verbose: bool = False,
) -> List[FitSystem]:
    """FitSystems for the structures of one HDF5 set (see the module docstring).

    Each system's ``name`` is ``<material>/<structure>``, its ``group`` is ``group(tag)`` (the strain by
    default, for leave-one-strain-out), and ``system.tag`` / ``system.assignment_score`` /
    ``system.moment_scale`` record where it came from and how well the chosen states match DFT.

    ``max_valence_split``: drop a structure whose chosen states put one element in two oxidation states further
    apart than this (None keeps all). The moment match can read a distorted all-Fe3+ geometry as Fe2+ + Fe4+
    (hematite / goethite MD snapshots), and a disproportionated Fe3+/Fe5+ compound (LaSr3Fe4O12) as Fe2+/Fe4+;
    mixed valence one apart (Fe2+/Fe3+, Mn3+/Mn4+) is kept.
    """
    import h5py

    from quick_mag.polarization_model import exchange_model

    from quick_mag.amplitude_head import DELTA, FEATURES

    path = Path(path)
    key = None
    if cache:
        st = path.stat()
        # the head features are pickled in the systems: their definition is part of the key
        feats = (FEATURES, tuple(sorted(DELTA.items())))
        opts = repr((str(path.resolve()), st.st_size, st.st_mtime, states, tuple(map(lambda s: tuple(s) if s else s, spin_options)),
                     top_k, valence, max_moment_dev, max_span, min_configs, energy, feats, "T-v3"))
        key = hashlib.sha1(opts.encode()).hexdigest()[:16]
        cpath = Path(cache) / f"{path.stem}-{key}.pkl"
        if cpath.exists():
            systems = pickle.loads(cpath.read_bytes())
            for s in systems:
                s.group = group(s.tag)
            return [s for s in systems if (select is None or select(s.tag)) and _split_ok(s, max_valence_split)]

    systems: List[FitSystem] = []
    stats: Dict[str, int] = {}

    def skip(why):
        stats[why] = stats.get(why, 0) + 1

    with h5py.File(path, "r") as f:
        for material in f:
            for sname, g in f[material].items():
                tag = parse_structure_name(material, sname)
                if select is not None and not cache and not select(tag):
                    continue
                cfgs = [g[c] for c in g if c.startswith("config")
                        and bool(g[c].attrs.get("converged", False)) and not bool(g[c].attrs.get("is_outlier", False))]
                if len(cfgs) < min_configs:
                    skip("too few converged configs"); continue
                mom = np.array([c["magnetic_moments"][()] for c in cfgs], dtype=float)
                ene = np.array([float(c["energies"][energy][()]) for c in cfgs])
                structure = _structure(str(g.attrs["poscar"]), f"{material}/{sname}")
                med = np.median(np.abs(mom), axis=0)
                try:
                    a, score, scale = _choose_assignment(structure, med, states, spin_options, top_k)
                except ValueError:
                    skip("no assignment"); continue
                model = exchange_model(structure, a, valence=valence)
                sites = np.asarray(model.sites)
                dev = np.abs(np.abs(mom[:, sites]) - med[sites][None, :]).max(axis=1)
                keep = dev <= max_moment_dev
                n_mag = int((model.moments > 1e-8).sum())
                if keep.sum() >= min_configs:
                    e = ene[keep]
                    keep[np.flatnonzero(keep)[(e - e.min()) / n_mag > max_span]] = False
                if keep.sum() < min_configs:
                    skip("too few configs after moment/span filters"); continue
                signs = np.sign(mom[keep][:, sites])
                labels = [cfgs[k].name.rsplit("/", 1)[-1] for k in np.flatnonzero(keep)]
                s = fit_system(model, signs, ene[keep], name=f"{material}/{sname}", group=group(tag), labels=labels)
                s.tag = tag
                s.assignment_score = score
                s.moment_scale = scale
                s.n_dropped = int(len(cfgs) - keep.sum())
                systems.append(s)
                if verbose:
                    print(f"  {s.name}: {keep.sum()}/{len(cfgs)} configs, states fit {score:.2f} muB (scale {scale:.2f})")
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        cpath.write_bytes(pickle.dumps(systems))
        systems = [s for s in systems if select is None or select(s.tag)]
    n = len(systems)
    systems = [s for s in systems if _split_ok(s, max_valence_split)]
    if n > len(systems):
        stats["valence split"] = n - len(systems)
    if verbose:
        print(f"{path.name}: {len(systems)} systems; skipped {stats}")
    return systems


def valence_split(system: FitSystem) -> int:
    """The widest gap between two oxidation states one element takes at the system's bridge ends."""
    from quick_mag.amplitude_head import FEATURES

    ic = FEATURES.index("charge")
    states: Dict[str, set] = {}
    for end in ("i", "j"):
        for el, q in zip(getattr(system, f"metal_{end}"), getattr(system, f"feat_{end}")[:, ic]):
            states.setdefault(str(el), set()).add(int(round(float(q))))
    return max((max(v) - min(v) for v in states.values()), default=0)


def _split_ok(system: FitSystem, limit: Optional[int]) -> bool:
    return limit is None or valence_split(system) <= limit
