from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

from quick_mag.cif_io import read_cif
from quick_mag.extxyz_io import Frame, is_extxyz_path, read_frames
from quick_mag.structure import ChemicalStructure
from quick_mag.vasp_io import read_poscar

SUPPORTED_FORMATS = (
    "VASP/POSCAR (.vasp, POSCAR, CONTCAR), P1 .cif and extended xyz (.xyz, .extxyz)"
)


def read_structure(path: Union[str, Path]) -> ChemicalStructure:
    """Read a structure file into a :class:`ChemicalStructure`.

    Supports VASP/POSCAR (``.vasp``, ``POSCAR``, ``CONTCAR``), P1 ``.cif`` and
    extended xyz (``.xyz``, ``.extxyz``; the first frame -- :func:`read_structure_frames`
    gives every frame and what else the file carried).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".cif":
        return read_cif(path)
    if suffix == ".vasp" or path.name.upper() in {"POSCAR", "CONTCAR"}:
        return read_poscar(path)
    if is_extxyz_path(path):
        return read_frames(path)[0].structure
    raise ValueError(
        f"Unsupported structure file '{path.name}'. Supported formats: {SUPPORTED_FORMATS}."
    )


def read_structure_frames(path: Union[str, Path]) -> List[Frame]:
    """Every structure a file holds, each as a :class:`~quick_mag.extxyz_io.Frame`.

    An extended xyz file gives one frame per structure with its comment-line keys
    (``Frame.info``) and extra per-atom columns (``Frame.arrays``: oxidation states,
    spin states, a caller's own columns). Any other format gives a single frame with
    neither.
    """
    path = Path(path)
    if is_extxyz_path(path):
        return read_frames(path)
    return [Frame(structure=read_structure(path))]
