"""Tests for the spinel lattice builder (``quick_mag.spinel_builder``)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.generation import generate_spinel  # noqa: E402
from quick_mag.perovskite_builder import SiteKey  # noqa: E402
from quick_mag.spinel_builder import (  # noqa: E402
    IDEAL_U,
    build_spinel,
    canonical_spinel_index_of_key,
    canonical_spinel_site_keys,
    cell_basis,
    conventional_basis,
    ideal_bond_lengths,
    u_from_origin_choice_1,
    u_to_origin_choice_1,
)
from quick_mag.structure import ChemicalStructure  # noqa: E402


def _structure(build, labels=None):
    roles = [key.role for key in build.keys]
    symbols = {"A": "Mg", "B": "Al", "X": "O"}
    return ChemicalStructure.with_zero_magnetic_moments(
        name="spinel",
        lattice=build.lattice,
        cartesian_coords=build.all_sites,
        atomic_labels=labels or [symbols[role] for role in roles],
        is_periodic=True,
    )


def _coordination(structure, index, cutoff=2.3):
    return sorted(n.symbol for n in structure.neighbors(index, cutoff))


class ConventionalCellTest(unittest.TestCase):
    def test_site_counts_are_8_16_32(self):
        build = build_spinel(8.08, u=0.262)
        self.assertEqual(
            (len(build.a_sites), len(build.b_sites), len(build.x_sites)), (8, 16, 32)
        )

    def test_basis_sites_are_distinct(self):
        for role, points in conventional_basis(0.262).items():
            rounded = {tuple(np.round(p, 6)) for p in points}
            self.assertEqual(len(rounded), len(points), role)

    def test_ideal_u_puts_anions_on_an_fcc_lattice(self):
        # At the ideal u every anion is 1/4 cell from its neighbours along <110>/2.
        build = build_spinel(8.0, u=IDEAL_U)
        structure = _structure(build)
        x0 = int(build.x_site_indices[0])
        anion_shell = [
            n.nn_distance for n in structure.neighbors(x0, 2.9) if n.symbol == "O"
        ]
        self.assertEqual(len(anion_shell), 12)
        np.testing.assert_allclose(anion_shell, 8.0 / (2.0 * np.sqrt(2.0)), atol=1e-9)

    def test_coordination_tetrahedral_octahedral_and_anion(self):
        build = build_spinel(8.08, u=0.262)
        structure = _structure(build)
        for index in build.a_site_indices:
            self.assertEqual(_coordination(structure, int(index)), ["O"] * 4)
        for index in build.b_site_indices:
            self.assertEqual(_coordination(structure, int(index)), ["O"] * 6)
        for index in build.x_site_indices:
            self.assertEqual(_coordination(structure, int(index)), ["Al", "Al", "Al", "Mg"])

    def test_bond_lengths_follow_u(self):
        for u in (0.25, 0.255, 0.262, 0.27):
            build = build_spinel(8.08, u=u)
            structure = _structure(build)
            r_tet, r_oct = ideal_bond_lengths(8.08, u)
            tet = [n.nn_distance for n in structure.neighbors(int(build.a_site_indices[3]), 2.3)]
            oct_ = [n.nn_distance for n in structure.neighbors(int(build.b_site_indices[5]), 2.3)]
            np.testing.assert_allclose(tet, r_tet, atol=1e-9)
            np.testing.assert_allclose(oct_, r_oct, atol=1e-9)

    def test_no_two_sites_overlap_across_the_boundary(self):
        build = build_spinel(8.08, u=0.262, n_cells=(2, 1, 1))
        structure = _structure(build)
        for index in range(structure.atom_count):
            neighbours = structure.neighbors(index, 1.5)
            self.assertEqual(neighbours, [], f"site {index} has a neighbour inside 1.5 A")

    def test_tetragonal_cell_scales_c(self):
        build = build_spinel(8.0, c=9.0, u=0.26)
        np.testing.assert_allclose(build.lattice, np.diag([8.0, 8.0, 9.0]))
        frac = np.linalg.solve(build.lattice.T, build.all_sites.T).T
        cubic = build_spinel(8.0, u=0.26)
        cubic_frac = np.linalg.solve(cubic.lattice.T, cubic.all_sites.T).T
        np.testing.assert_allclose(frac, cubic_frac, atol=1e-12)


class PrimitiveCellTest(unittest.TestCase):
    def test_primitive_cell_has_14_atoms(self):
        basis = cell_basis(0.26, "primitive")
        self.assertEqual({role: len(p) for role, p in basis.items()}, {"A": 2, "B": 4, "X": 8})

    def test_primitive_cell_is_rhombohedral_with_a_quarter_of_the_volume(self):
        build = build_spinel(8.0, cell="primitive")
        lengths = np.linalg.norm(build.lattice, axis=1)
        np.testing.assert_allclose(lengths, 8.0 / np.sqrt(2.0))
        a, b = build.lattice[0], build.lattice[1]
        self.assertAlmostEqual(
            np.degrees(np.arccos(a @ b / (lengths[0] * lengths[1]))), 60.0
        )
        self.assertAlmostEqual(abs(np.linalg.det(build.lattice)), 8.0 ** 3 / 4.0)

    def test_primitive_cell_has_the_same_local_environments(self):
        build = build_spinel(8.08, u=0.262, cell="primitive", n_cells=(2, 2, 2))
        structure = _structure(build)
        r_tet, r_oct = ideal_bond_lengths(8.08, 0.262)
        for index in build.a_site_indices:
            distances = [n.nn_distance for n in structure.neighbors(int(index), 2.3)]
            np.testing.assert_allclose(distances, [r_tet] * 4, atol=1e-9)
        for index in build.b_site_indices:
            distances = [n.nn_distance for n in structure.neighbors(int(index), 2.3)]
            np.testing.assert_allclose(distances, [r_oct] * 6, atol=1e-9)


class SiteKeyTest(unittest.TestCase):
    def test_keys_are_in_build_order_and_invert(self):
        for cell in ("conventional", "primitive"):
            keys = canonical_spinel_site_keys((2, 1, 3), cell)
            build = build_spinel(8.0, n_cells=(2, 1, 3), cell=cell)
            self.assertEqual(list(build.keys), keys)
            self.assertEqual(len(keys), len(build.all_sites))
            for n, key in enumerate(keys):
                self.assertEqual(canonical_spinel_index_of_key(key, (2, 1, 3), cell), n)

    def test_out_of_range_keys_have_no_index(self):
        for key in (SiteKey("A", 0, 0, 0, 8), SiteKey("B", 1, 0, 0, 0), SiteKey("Q", 0, 0, 0, 0)):
            self.assertEqual(canonical_spinel_index_of_key(key, (1, 1, 1)), -1)

    def test_roles_are_contiguous_blocks(self):
        build = build_spinel(8.0, n_cells=(1, 2, 1))
        roles = build.site_roles
        self.assertEqual(roles, ["A"] * 16 + ["B"] * 32 + ["X"] * 64)


class ValidationTest(unittest.TestCase):
    def test_finite_builds_are_refused(self):
        with self.assertRaises(ValueError):
            build_spinel(8.0, periodic=(True, True, False))

    def test_origin_choice_1_u_is_caught(self):
        with self.assertRaises(ValueError):
            build_spinel(8.0, u=0.385)
        self.assertAlmostEqual(u_from_origin_choice_1(0.385), 0.26)
        self.assertAlmostEqual(u_to_origin_choice_1(0.26), 0.385)

    def test_bad_cell_and_supercell(self):
        with self.assertRaises(ValueError):
            build_spinel(8.0, cell="hexagonal")
        with self.assertRaises(ValueError):
            build_spinel(8.0, n_cells=(0, 1, 1))


class GenerateSpinelTest(unittest.TestCase):
    def test_normal_spinel_stoichiometry_and_provenance(self):
        structure = generate_spinel("MgAl2O4", a_site="Mg", b_site="Al", a=8.08, u=0.262)
        self.assertEqual(structure.atom_count, 56)
        symbols = structure.element_symbols()
        self.assertEqual(symbols.count("Mg"), 8)
        self.assertEqual(symbols.count("Al"), 16)
        self.assertEqual(symbols.count("O"), 32)
        params = structure.generation_parameters
        self.assertEqual(params.family, "spinel")
        self.assertEqual(params.site_roles, ["A"] * 8 + ["B"] * 16 + ["X"] * 32)
        np.testing.assert_array_equal(params.permutation, np.arange(56))
        np.testing.assert_allclose(structure.lattice, params.supercell_lattice())
        self.assertTrue(structure.is_periodic)


if __name__ == "__main__":
    unittest.main()
