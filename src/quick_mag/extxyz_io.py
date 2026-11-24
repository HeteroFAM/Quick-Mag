"""Extended xyz for quick_mag structures.

Extended xyz (one or many frames per file) is the format that carries everything a program
on either side of ``quick-mag`` may want to say about a structure, per frame on the comment
line and per atom as columns:

    <natoms>
    Lattice="ax ay az bx by bz cx cy cz" Properties=species:S:1:pos:R:3[:label:S:1][:magmom:R:1]...
        pbc="T T T" name=LaFeO3 [qm_gen=<base64 json>] [key=value ...]
    La  0.0 0.0 0.0  ...

- ``Lattice`` rows are the lattice vectors. A frame without one is a molecule in a large box.
- ``label`` is written only when an atomic label differs from its element ("Fe3+").
- ``magmom`` is the signed collinear moment (the z component of the moments, mu_B), written
  only when some moment is non-zero. On reading, ``magmoms`` and ``initial_magmoms`` are taken
  as the same thing, with one component (along z) or three.
- ``oxidation`` (Int, 0 = not given), ``spin_state`` (``HS`` / ``LS``, anything else = not
  given) and ``site`` (a label for ``@label`` rules) are the per-atom states ``quick-mag solve``
  reads (:func:`site_states_from_frame`).
- ``qm_gen`` is the builder's ``generation_parameters`` as base64 JSON, so a structure written
  out and read back keeps its provenance. Tools that do not know it ignore it.
- Every other ``key=value`` on the comment line and every other per-atom column is kept
  (``Frame.info`` / ``Frame.arrays``) and written back by :func:`write_extxyz`, so columns a
  caller added (site keys, say) survive a trip through a quick-mag command.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from quick_mag.structure import (
    ChemicalStructure,
    generation_parameters_from_json,
    generation_parameters_to_json,
)

PathLike = Union[str, Path]

_ELEMENT = re.compile(r"[A-Z][a-z]?")

#: Per-atom columns with a meaning to quick_mag.
MOMENT_COLUMNS = ("magmom", "magmoms", "initial_magmoms")
OXIDATION_COLUMN = "oxidation"
SPIN_COLUMN = "spin_state"
LABEL_COLUMN = "site"
EXTXYZ_SUFFIXES = (".xyz", ".extxyz")


@dataclass
class Frame:
    """One extxyz frame: the structure plus everything else the file carried."""

    structure: ChemicalStructure
    info: Dict[str, Any] = field(default_factory=dict)       # comment-line keys (not Lattice/Properties/pbc/name/qm_gen)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)  # extra per-atom columns


# ---------------------------------------------------------------------------- writing


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (list, tuple, np.ndarray)):
        return '"' + " ".join(_fmt_value(v) for v in np.asarray(value).reshape(-1).tolist()) + '"'
    text = str(value)
    if text == "" or re.search(r'[\s="]', text):
        return '"' + text.replace('"', "'") + '"'
    return text


def _encode_params(structure: ChemicalStructure) -> Optional[str]:
    params = getattr(structure, "generation_parameters", None)
    if params is None:
        return None
    payload = generation_parameters_to_json(params)
    payload["_geometry_matches_generation"] = bool(structure.geometry_matches_generation)
    raw = json.dumps(payload, separators=(",", ":"), default=_json_default).encode()
    return base64.b64encode(raw).decode()


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(f"not JSON serialisable: {type(x)}")


def _periodic_axes(structure: ChemicalStructure):
    axes = getattr(structure, "periodic_axes", None)
    if axes is None:
        return (bool(structure.is_periodic),) * 3
    return tuple(bool(a) for a in axes)


def extxyz_text(
    structure: ChemicalStructure,
    *,
    info: Optional[Dict[str, Any]] = None,
    arrays: Optional[Dict[str, np.ndarray]] = None,
    moments: bool = True,
    provenance: bool = True,
) -> str:
    """One extxyz frame for ``structure`` (see the module docstring for the layout).

    ``info`` adds comment-line keys, ``arrays`` per-atom columns (float, int, or str arrays of
    length natoms; 2-D float arrays become multi-component columns, e.g. forces).
    """
    n = structure.atom_count
    symbols = structure.element_symbols()
    labels = list(structure.atomic_labels)
    coords = np.asarray(structure.cartesian_coords, dtype=float)

    columns: List[tuple] = [("species", "S", 1, symbols), ("pos", "R", 3, coords)]
    if any(lab != sym for lab, sym in zip(labels, symbols)):
        columns.append(("label", "S", 1, labels))
    mz = np.asarray(structure.magnetic_moments, dtype=float).reshape(n, 3)[:, 2] if n else np.zeros(0)
    if moments and np.any(np.abs(mz) > 1e-12):
        columns.append(("magmom", "R", 1, mz))
    for name, values in (arrays or {}).items():
        arr = np.asarray(values)
        if len(arr) != n:
            raise ValueError(f"column {name!r} has {len(arr)} rows for {n} atoms")
        if arr.dtype.kind in "USO":
            columns.append((name, "S", 1, [str(v) for v in arr]))
        elif arr.dtype.kind in "iub":
            columns.append((name, "I", 1 if arr.ndim == 1 else arr.shape[1], arr.astype(int)))
        else:
            columns.append((name, "R", 1 if arr.ndim == 1 else arr.shape[1], arr.astype(float)))

    props = ":".join(f"{name}:{kind}:{width}" for name, kind, width, _ in columns)
    header = [
        "Lattice=" + _fmt_value(np.asarray(structure.lattice, dtype=float)),
        f"Properties={props}",
        "pbc=" + _fmt_value(list(_periodic_axes(structure))),
        "name=" + _fmt_value(structure.name),
    ]
    for key, value in (info or {}).items():
        if key in ("Lattice", "Properties", "pbc", "name", "qm_gen"):
            raise ValueError(f"info key {key!r} is reserved")
        header.append(f"{key}={_fmt_value(value)}")
    enc = _encode_params(structure) if provenance else None
    if enc:
        header.append(f"qm_gen={enc}")

    lines = [str(n), " ".join(header)]
    for i in range(n):
        parts = []
        for _name, kind, width, values in columns:
            v = values[i]
            if kind == "S":
                parts.append(f"{v:<4s}")
            elif kind == "I":
                parts.extend(str(int(x)) for x in np.atleast_1d(v))
            else:
                parts.extend(f"{float(x):16.10f}" for x in np.atleast_1d(v))
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def write_extxyz(
    path: PathLike,
    structures: Union[ChemicalStructure, Frame, Sequence[Union[ChemicalStructure, Frame]]],
    *,
    append: bool = False,
    if_changed: bool = False,
    **kwargs,
) -> Path:
    """Write one or many structures (or :class:`Frame` s) as extxyz frames. ``if_changed``
    leaves an existing file with the same text untouched (no new mtime for watchers)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(structures, (ChemicalStructure, Frame)):
        structures = [structures]
    chunks = []
    for item in structures:
        if isinstance(item, Frame):
            chunks.append(extxyz_text(item.structure, info=item.info, arrays=item.arrays, **kwargs))
        else:
            chunks.append(extxyz_text(item, **kwargs))
    text = "".join(chunks)
    if if_changed and not append and path.exists() and path.read_text() == text:
        return path
    with open(path, "a" if append else "w") as fh:
        fh.write(text)
    return path


# ---------------------------------------------------------------------------- reading


def _parse_comment(line: str) -> Dict[str, str]:
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.whitespace = " \t"
    lexer.commenters = ""
    out: Dict[str, str] = {}
    for token in lexer:
        if "=" in token:
            k, v = token.split("=", 1)
            out[k] = v
        else:
            out[token] = "T"
    return out


def _parse_value(text: str):
    parts = text.split()
    def one(t):
        if t in ("T", "True", "true"):
            return True
        if t in ("F", "False", "false"):
            return False
        try:
            return int(t)
        except ValueError:
            pass
        try:
            return float(t)
        except ValueError:
            return t
    if len(parts) > 1:
        vals = [one(p) for p in parts]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            return np.array(vals, dtype=float)
        if all(isinstance(v, bool) for v in vals):
            return vals
        return text
    return one(text) if parts else ""


def _decode_params(text: str):
    payload = json.loads(base64.b64decode(text.encode()).decode())
    matches = bool(payload.pop("_geometry_matches_generation", False))
    return generation_parameters_from_json(payload), matches


def _parse_frames(text: str, source: str) -> List[Frame]:
    lines = text.splitlines()
    frames: List[Frame] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            n = int(lines[i].split()[0])
        except ValueError as exc:
            raise ValueError(f"{source}: line {i + 1}: expected an atom count") from exc
        comment = _parse_comment(lines[i + 1]) if i + 1 < len(lines) else {}
        body = lines[i + 2 : i + 2 + n]
        if len(body) != n:
            raise ValueError(f"{source}: frame at line {i + 1} is truncated")
        i += 2 + n

        props = comment.pop("Properties", "species:S:1:pos:R:3").split(":")
        spec = [(props[k], props[k + 1], int(props[k + 2])) for k in range(0, len(props), 3)]
        cols: Dict[str, list] = {name: [] for name, _, _ in spec}
        for row in body:
            fields = row.split()
            j = 0
            for name, kind, width in spec:
                chunk = fields[j : j + width]
                j += width
                if kind == "S":
                    cols[name].append(chunk[0])
                elif kind == "I":
                    cols[name].append([int(x) for x in chunk])
                elif kind == "L":
                    cols[name].append([x in ("T", "True", "1") for x in chunk])
                else:
                    cols[name].append([float(x) for x in chunk])
        species = cols.pop("species", None) or cols.pop("element")
        pos = np.asarray(cols.pop("pos"), dtype=float).reshape(n, 3)
        labels = cols.pop("label", None) or [s if _ELEMENT.fullmatch(s) else s for s in species]

        lattice_text = comment.pop("Lattice", None)
        pbc_text = comment.pop("pbc", None)
        if lattice_text is not None:
            lattice = np.asarray(lattice_text.split(), dtype=float).reshape(3, 3)
            axes = tuple(_parse_value(pbc_text)) if pbc_text else (True, True, True)
            if isinstance(axes[0], (int, float)) and not isinstance(axes[0], bool):
                axes = tuple(bool(a) for a in axes)
        else:
            lattice = np.eye(3) * 100.0
            axes = (False, False, False)
        name = comment.pop("name", None) or f"{Path(source).stem}_{len(frames)}"

        params, matches = (None, True)
        if "qm_gen" in comment:
            try:
                params, matches = _decode_params(comment.pop("qm_gen"))
            except Exception:
                params, matches = None, True
        structure = ChemicalStructure.with_zero_magnetic_moments(
            name=name,
            lattice=lattice,
            cartesian_coords=pos,
            atomic_labels=list(labels),
            is_periodic=any(axes),
            periodic_axes=tuple(bool(a) for a in axes),
            generation_parameters=params,
            geometry_matches_generation=matches if params is not None else True,
        )
        for column in MOMENT_COLUMNS:
            if column not in cols:
                continue
            m = np.asarray(cols.pop(column), dtype=float)
            if m.shape[1] == 1:
                structure.magnetic_moments[:, 2] = m[:, 0]
            else:
                structure.magnetic_moments[:, :] = m[:, :3]
            break
        arrays = {}
        for k, v in cols.items():
            arr = np.asarray(v)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            arrays[k] = arr
        info = {k: _parse_value(v) for k, v in comment.items()}
        frames.append(Frame(structure=structure, info=info, arrays=arrays))
    return frames


def is_extxyz_path(path: PathLike) -> bool:
    return Path(path).suffix.lower() in EXTXYZ_SUFFIXES


def read_frames(path_or_text: Union[PathLike, str], *, text: bool = False) -> List[Frame]:
    """Every frame of an extxyz file (or of the text itself with ``text=True``)."""
    if text:
        return _parse_frames(str(path_or_text), "<text>")
    path = Path(path_or_text)
    frames = _parse_frames(path.read_text(), str(path))
    if not frames:
        raise ValueError(f"{path}: no frames")
    return frames


def read_extxyz(path: PathLike, index: int = 0) -> ChemicalStructure:
    """The structure of frame ``index`` of an extxyz file."""
    return read_frames(path)[index].structure


# ---------------------------------------------------------------------------- site states


def _column(frame: Optional[Frame], name: str):
    arrays = getattr(frame, "arrays", None) or {}
    return arrays.get(name)


def frame_oxidation_states(frame: Optional[Frame]) -> Optional[List[Optional[int]]]:
    """The ``oxidation`` column as one value per atom (``None`` = not given), or ``None``
    when the frame has no such column."""
    column = _column(frame, OXIDATION_COLUMN)
    if column is None:
        return None
    out: List[Optional[int]] = []
    for value in np.asarray(column).reshape(len(column), -1)[:, 0]:
        try:
            state = int(round(float(value)))
        except (TypeError, ValueError):
            state = 0
        out.append(state if state != 0 else None)
    return out


def frame_spin_states(frame: Optional[Frame]) -> Optional[List[Optional[str]]]:
    """The ``spin_state`` column as ``"HS"`` / ``"LS"`` / ``None`` per atom, or ``None``
    when the frame has no such column."""
    column = _column(frame, SPIN_COLUMN)
    if column is None:
        return None
    out: List[Optional[str]] = []
    for value in column:
        text = str(value).strip().upper()
        out.append(text if text in ("HS", "LS") else None)
    return out


def frame_site_labels(frame: Optional[Frame]) -> Optional[List[str]]:
    column = _column(frame, LABEL_COLUMN)
    return None if column is None else [str(v) for v in column]


__all__ = [
    "Frame",
    "MOMENT_COLUMNS",
    "OXIDATION_COLUMN",
    "SPIN_COLUMN",
    "LABEL_COLUMN",
    "EXTXYZ_SUFFIXES",
    "extxyz_text",
    "write_extxyz",
    "read_frames",
    "read_extxyz",
    "is_extxyz_path",
    "frame_oxidation_states",
    "frame_spin_states",
    "frame_site_labels",
]
