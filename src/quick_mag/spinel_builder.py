"""Spinel (AB2X4) lattice builder.

The perovskite builder lays out a grid of corner-sharing octahedra; a spinel has
no such grid, so this builder works from the space group instead. The cubic
spinel is Fd-3m (No. 227), written here in ITA *origin choice 2* (inversion
centre at the origin):

* ``A`` -- tetrahedral cations on 8a  ``(1/8, 1/8, 1/8)``
* ``B`` -- octahedral cations on 16d  ``(1/2, 1/2, 1/2)``
* ``X`` -- anions on 32e ``(u, u, u)``

``u`` is the one free coordinate. In origin choice 2 the ideal cubic-close-packed
value is ``u = 1/4``; real oxide spinels sit around 0.255-0.265 (MgAl2O4 ~0.262),
and larger ``u`` expands the tetrahedra at the expense of the octahedra. Papers
that use origin choice 1 quote ``u + 1/8`` (ideal 3/8); see
:func:`u_from_origin_choice_1`.

Two cells are available:

* ``"conventional"`` -- the 56-atom cubic cell (8 A, 16 B, 32 X).
* ``"primitive"`` -- the 14-atom rhombohedral cell (2 A, 4 B, 8 X), spanned by
  the face-centring vectors.

and either is repeated ``n_cells`` times along its own lattice vectors. A
tetragonal cell (``c != a``, e.g. a Jahn-Teller-distorted spinel) is built by
scaling the cubic fractional coordinates along ``c``. That is the
I4_1/amd structure only in its simplest form: the real tetragonal structure has
two independent anion coordinates, which a relaxation recovers.

Sites are addressed by the same :class:`~quick_mag.perovskite_builder.SiteKey`
the perovskite builder uses, read differently: ``(i, j, k)`` is the index of the
repeated cell and ``vertex`` is the index of the site within that cell's basis
for its role. Every site belongs to exactly one cell, so -- unlike the
corner-shared perovskite oxygens -- no site has two names.

Only fully periodic builds are supported: a spinel slab needs a choice of
termination and a treatment of its polar surfaces that a simple cut does not
make.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from quick_mag.perovskite_builder import SiteKey, periodic_axes

# Ideal (cubic close-packed) anion parameter in origin choice 2.
IDEAL_U = 0.25
# Default anion parameter: a typical oxide spinel rather than the ideal value,
# which leaves the tetrahedral bonds unphysically short.
DEFAULT_U = 0.26
# A typical oxide spinel lattice constant (Angstrom), e.g. MgAl2O4 ~8.08,
# NiFe2O4 ~8.34, CoFe2O4 ~8.39, ZnFe2O4 ~8.44.
DEFAULT_LATTICE_CONSTANT = 8.4

SPINEL_ROLES: Tuple[str, ...] = ("A", "B", "X")
ROLE_DESCRIPTIONS = {
    "A": "tetrahedral (8a)",
    "B": "octahedral (16d)",
    "X": "anion (32e)",
}
CELL_KINDS: Tuple[str, ...] = ("conventional", "primitive")

# Fd-3m face-centring translations.
_FCC_TRANSLATIONS = np.array(
    [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
    dtype=float,
)

# Primitive (rhombohedral) cell vectors in units of the conventional cell.
_PRIMITIVE_IN_CONVENTIONAL = np.array(
    [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
    dtype=float,
)

_COORD_TOL = 1e-6


def u_from_origin_choice_1(u_origin_1: float) -> float:
    """Convert an origin-choice-1 anion parameter (ideal 3/8) to origin choice 2."""
    return float(u_origin_1) - 0.125


def u_to_origin_choice_1(u: float) -> float:
    """Convert this module's origin-choice-2 ``u`` to origin choice 1."""
    return float(u) + 0.125


def _orbit(representatives: Sequence[Sequence[float]]) -> np.ndarray:
    """Representatives plus their face-centred copies, wrapped into [0, 1)."""
    reps = np.asarray(representatives, dtype=float)
    points = (reps[None, :, :] + _FCC_TRANSLATIONS[:, None, :]).reshape(-1, 3)
    return _wrap(points)


def _wrap(points: np.ndarray) -> np.ndarray:
    wrapped = np.mod(points, 1.0)
    # Fold values that round to 1 back onto 0 so equal sites compare equal.
    wrapped[np.isclose(wrapped, 1.0, atol=_COORD_TOL)] = 0.0
    return wrapped


def conventional_basis(u: float = DEFAULT_U) -> dict:
    """Fractional coordinates of every site in the 56-atom conventional cell.

    Returned as ``{"A": (8, 3), "B": (16, 3), "X": (32, 3)}`` in a fixed,
    documented order (orbit representatives in ITA order, each followed by its
    face-centred copies), which is the order the site keys index.
    """
    x = float(u)
    a_reps = [(1 / 8, 1 / 8, 1 / 8), (7 / 8, 3 / 8, 3 / 8)]
    b_reps = [(1 / 2, 1 / 2, 1 / 2), (1 / 4, 3 / 4, 0.0), (3 / 4, 0.0, 1 / 4), (0.0, 1 / 4, 3 / 4)]
    x_reps = [
        (x, x, x),
        (-x + 3 / 4, -x + 1 / 4, x + 1 / 2),
        (-x + 1 / 4, x + 1 / 2, -x + 3 / 4),
        (x + 1 / 2, -x + 3 / 4, -x + 1 / 4),
        (-x, -x, -x),
        (x + 1 / 4, x + 3 / 4, -x + 1 / 2),
        (x + 3 / 4, -x + 1 / 2, x + 1 / 4),
        (-x + 1 / 2, x + 1 / 4, x + 3 / 4),
    ]
    return {"A": _orbit(a_reps), "B": _orbit(b_reps), "X": _orbit(x_reps)}


def _primitive_basis(u: float) -> dict:
    """The conventional basis folded into the primitive cell, duplicates dropped.

    The first conventional site of each primitive-cell equivalence class is the
    one kept, so the order is a deterministic function of the conventional order.
    """
    to_primitive = np.linalg.inv(_PRIMITIVE_IN_CONVENTIONAL.T)
    basis = {}
    for role, points in conventional_basis(u).items():
        folded = _wrap(points @ to_primitive.T)
        kept: List[np.ndarray] = []
        for point in folded:
            if not any(_same_fractional_site(point, other) for other in kept):
                kept.append(point)
        basis[role] = np.asarray(kept, dtype=float)
    return basis


def _same_fractional_site(p: np.ndarray, q: np.ndarray) -> bool:
    delta = np.asarray(p) - np.asarray(q)
    delta -= np.round(delta)
    return bool(np.all(np.abs(delta) < 1e-6))


def cell_basis(u: float = DEFAULT_U, cell: str = "conventional") -> dict:
    """Per-role fractional coordinates within one ``cell`` (see ``CELL_KINDS``)."""
    cell = _validated_cell(cell)
    if cell == "conventional":
        return conventional_basis(u)
    return _primitive_basis(u)


def cell_lattice(a: float, c: float | None = None, cell: str = "conventional") -> np.ndarray:
    """Row-vector lattice of one ``cell`` for edge ``a`` (and ``c``, default ``a``)."""
    cell = _validated_cell(cell)
    edge_a = float(a)
    edge_c = edge_a if c is None else float(c)
    if edge_a <= 0.0 or edge_c <= 0.0:
        raise ValueError("Spinel lattice constants must be positive.")
    conventional = np.diag([edge_a, edge_a, edge_c])
    if cell == "conventional":
        return conventional
    return _PRIMITIVE_IN_CONVENTIONAL @ conventional


def basis_counts(cell: str = "conventional") -> Tuple[int, int, int]:
    """``(n_a, n_b, n_x)`` sites in one cell."""
    return (8, 16, 32) if _validated_cell(cell) == "conventional" else (2, 4, 8)


def _validated_cell(cell: str) -> str:
    cell = str(cell).strip().lower()
    if cell not in CELL_KINDS:
        raise ValueError(f"Unknown spinel cell '{cell}'; expected one of {CELL_KINDS}.")
    return cell


def _validated_n_cells(n_cells) -> Tuple[int, int, int]:
    values = tuple(int(value) for value in n_cells)
    if len(values) != 3 or min(values) < 1:
        raise ValueError("n_cells must be three positive integers.")
    return values  # type: ignore[return-value]


def _validated_u(u: float) -> float:
    value = float(u)
    # Outside this window the anion collides with a cation or leaves its cage;
    # it is far wider than any real spinel (~0.24-0.27).
    if not 0.2 <= value <= 0.3:
        raise ValueError(
            f"Spinel anion parameter u={value} is outside [0.2, 0.3] "
            "(origin choice 2; ideal 0.25). Did you pass an origin-choice-1 value?"
        )
    return value


def canonical_spinel_site_keys(n_cells, cell: str = "conventional") -> List[SiteKey]:
    """Site keys in the order :func:`build_spinel` emits its sites.

    All A sites, then all B, then all X; within a role, cell-major over
    ``(i, j, k)`` and then basis index.
    """
    nx, ny, nz = _validated_n_cells(n_cells)
    keys: List[SiteKey] = []
    for role, count in zip(SPINEL_ROLES, basis_counts(cell)):
        keys += [
            SiteKey(role, i, j, k, vertex)
            for i in range(nx)
            for j in range(ny)
            for k in range(nz)
            for vertex in range(count)
        ]
    return keys


def canonical_spinel_index_of_key(key: SiteKey, n_cells, cell: str = "conventional") -> int:
    """Position of ``key`` in :func:`canonical_spinel_site_keys`, or -1."""
    nx, ny, nz = _validated_n_cells(n_cells)
    counts = basis_counts(cell)
    if key.role not in SPINEL_ROLES:
        return -1
    role_index = SPINEL_ROLES.index(key.role)
    i, j, k, vertex = int(key.i), int(key.j), int(key.k), int(key.vertex)
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz and 0 <= vertex < counts[role_index]):
        return -1
    n_repeat = nx * ny * nz
    offset = sum(counts[r] * n_repeat for r in range(role_index))
    return offset + ((i * ny + j) * nz + k) * counts[role_index] + vertex


@dataclass(frozen=True)
class SpinelBuild:
    """A fully occupied spinel supercell in canonical ``(A, B, X)`` order.

    ``lattice`` holds the supercell vectors as rows; ``all_sites`` are cartesian
    and lie inside that cell. ``keys[n]`` addresses ``all_sites[n]``.
    """

    lattice: np.ndarray
    all_sites: np.ndarray
    a_site_indices: np.ndarray
    b_site_indices: np.ndarray
    x_site_indices: np.ndarray
    keys: Tuple[SiteKey, ...]
    n_cells: Tuple[int, int, int]
    cell: str
    u: float

    @property
    def a_sites(self) -> np.ndarray:
        return self.all_sites[self.a_site_indices]

    @property
    def b_sites(self) -> np.ndarray:
        return self.all_sites[self.b_site_indices]

    @property
    def x_sites(self) -> np.ndarray:
        return self.all_sites[self.x_site_indices]

    @property
    def site_roles(self) -> List[str]:
        return [key.role for key in self.keys]

    def role_indices(self, role: str) -> np.ndarray:
        return {
            "A": self.a_site_indices,
            "B": self.b_site_indices,
            "X": self.x_site_indices,
        }[role]


def build_spinel(
    a: float = DEFAULT_LATTICE_CONSTANT,
    c: float | None = None,
    u: float = DEFAULT_U,
    n_cells=(1, 1, 1),
    cell: str = "conventional",
    periodic=True,
) -> SpinelBuild:
    """Lay out an ideal spinel supercell.

    ``a``/``c`` are the *conventional* cubic (or tetragonal) edge lengths in
    Angstrom even when ``cell="primitive"``; ``n_cells`` repeats the chosen cell
    along its own lattice vectors.
    """
    if not all(periodic_axes(periodic)):
        raise ValueError(
            "Spinel builds are periodic along every axis; finite (slab or "
            "cluster) spinels are not supported yet."
        )
    cell = _validated_cell(cell)
    u = _validated_u(u)
    n_cells = _validated_n_cells(n_cells)
    nx, ny, nz = n_cells

    unit = cell_lattice(a, c, cell)
    basis = cell_basis(u, cell)
    offsets = np.array(
        [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)],
        dtype=float,
    )

    blocks = []
    for role in SPINEL_ROLES:
        frac = basis[role]
        # (cell, basis) order: cell-major, then basis index -- the key order.
        tiled = (offsets[:, None, :] + frac[None, :, :]).reshape(-1, 3)
        blocks.append(tiled @ unit)
    all_sites = np.vstack(blocks)

    n_a, n_b, n_x = (len(block) for block in blocks)
    return SpinelBuild(
        lattice=np.diag(n_cells).astype(float) @ unit,
        all_sites=all_sites,
        a_site_indices=np.arange(0, n_a, dtype=int),
        b_site_indices=np.arange(n_a, n_a + n_b, dtype=int),
        x_site_indices=np.arange(n_a + n_b, n_a + n_b + n_x, dtype=int),
        keys=tuple(canonical_spinel_site_keys(n_cells, cell)),
        n_cells=n_cells,
        cell=cell,
        u=u,
    )


def ideal_bond_lengths(a: float, u: float = DEFAULT_U) -> Tuple[float, float]:
    """Cubic-spinel ``(R_tet, R_oct)`` cation-anion distances for edge ``a``.

    Closed forms in origin choice 1 (``u1 = u + 1/8``):
    ``R_tet = a sqrt(3) (u1 - 1/4)`` and
    ``R_oct = a sqrt(3 u1^2 - 11/4 u1 + 43/64)``.
    """
    u1 = u_to_origin_choice_1(u)
    r_tet = float(a) * np.sqrt(3.0) * (u1 - 0.25)
    r_oct = float(a) * np.sqrt(3.0 * u1 * u1 - 2.75 * u1 + 43.0 / 64.0)
    return float(r_tet), float(r_oct)
