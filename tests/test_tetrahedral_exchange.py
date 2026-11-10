"""Coordination-aware exchange: tetrahedral sites (spinel A sites) in the polarization model."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.generation import (  # noqa: E402
    generate_inverse_spinel,
    generate_normal_spinel,
    generate_single_perovskite,
)
from quick_mag.ion_descriptors import (  # noqa: E402
    ion_descriptor,
    shell_filling,
    site_coordination,
    structure_ion_descriptors,
)
from quick_mag.magnetic_moments import (  # noqa: E402
    expand_group_assignments_to_site_assignments,
)
from quick_mag.oxidation_state_energy import enumerate_site_group_assignments  # noqa: E402
from quick_mag.polarization_model import (  # noqa: E402
    _de_resonance_weight,
    _shannon_crystal_radius,
    build_bridges,
    build_Jeff_matrix,
    default_params,
    is_eg_active,
    is_occ_empty_active,
    local_d_frame,
    occupancy_vector,
    site_class_oxidation_fractions,
)
from quick_mag.spinel_composition import sublattice_oxidation_constraints  # noqa: E402
from quick_mag.structure import ChemicalStructure  # noqa: E402


def _assignment(structure):
    constraints = sublattice_oxidation_constraints(structure)
    ranked = enumerate_site_group_assignments(
        structure.element_symbols(), constraints=constraints, top_k=1
    )
    return expand_group_assignments_to_site_assignments(ranked, structure, max_assignments=1)[0]


def _cobalt_ferrite():
    return generate_normal_spinel(
        "CoFe2O4", a_site="Co", b_site="Fe", a=8.39, u=0.256,
        sublattice_oxidation_states={"A": {"Co": (2,)}, "B": {"Fe": (3,)}},
    )


def _rotation(seed=3):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _rotated(structure, rotation):
    return ChemicalStructure.with_zero_magnetic_moments(
        name=structure.name,
        lattice=structure.lattice @ rotation.T,
        cartesian_coords=structure.cartesian_coords @ rotation.T,
        atomic_labels=list(structure.atomic_labels),
        is_periodic=structure.is_periodic,
        generation_parameters=structure.generation_parameters,
    )


def _j_matrix(structure, assignment):
    descriptors = structure_ion_descriptors(structure, assignment)
    sites = sorted(descriptors)
    index = {site: i for i, site in enumerate(sites)}
    return build_Jeff_matrix(build_bridges(structure, descriptors), index, default_params())


def _is_signed_permutation(matrix, atol=1e-8):
    rounded = np.round(matrix)
    return np.allclose(matrix, rounded, atol=atol) and np.allclose(
        np.abs(rounded) @ np.ones(3), 1.0
    )


class ShellFillingTest(unittest.TestCase):
    # (n_t2, n_e) = electrons in (dxy, dxz, dyz) and in (dz2, dx2-y2).
    TET_HS = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 2), 4: (2, 2), 5: (3, 2),
              6: (3, 3), 7: (3, 4), 8: (4, 4), 9: (5, 4), 10: (6, 4)}

    def test_tetrahedral_high_spin_fills_e_first(self):
        for d_count, expected in self.TET_HS.items():
            self.assertEqual(shell_filling(d_count, "HS", "tet"), expected, d_count)

    def test_octahedral_filling_is_unchanged(self):
        self.assertEqual(shell_filling(7, "HS"), (5, 2))
        self.assertEqual(shell_filling(7, "HS", "oct"), (5, 2))

    def test_high_spin_moment_does_not_depend_on_coordination(self):
        for d_count in range(11):
            n_t, n_e = shell_filling(d_count, "HS", "tet")
            m_t, m_e = shell_filling(d_count, "HS", "oct")
            unpaired = lambda n, g: min(n, 2 * g - n)  # noqa: E731
            self.assertEqual(unpaired(n_t, 3) + unpaired(n_e, 2),
                             unpaired(m_t, 3) + unpaired(m_e, 2), d_count)

    def test_tetrahedral_cobalt_puts_its_spin_in_t2(self):
        descriptor = ion_descriptor("Co", 2, "HS", "tet")
        np.testing.assert_allclose(occupancy_vector(descriptor), [1, 1, 1, 0, 0])
        self.assertEqual(descriptor.spin, 1.5)
        self.assertIn("tet e^4 t2^3", repr(descriptor))
        octahedral = ion_descriptor("Co", 2, "HS", "oct")
        np.testing.assert_allclose(occupancy_vector(octahedral), [1 / 3] * 3 + [1, 1])

    def test_unknown_coordination_is_rejected(self):
        with self.assertRaises(ValueError):
            ion_descriptor("Fe", 3, "HS", "trigonal")


class CoordinationDetectionTest(unittest.TestCase):
    def test_spinel_sites(self):
        structure = _cobalt_ferrite()
        roles = structure.generation_parameters.site_roles
        for index, role in enumerate(roles):
            if role == "A":
                self.assertEqual(site_coordination(structure, index), "tet")
            elif role == "B":
                self.assertEqual(site_coordination(structure, index), "oct")

    def test_perovskite_b_site_is_octahedral(self):
        structure = generate_single_perovskite(
            "LaFeO3", a_site="La", b_site="Fe", x_site="O", a=3.93,
            n_cells_x=2, n_cells_y=2, n_cells_z=2,
        )
        symbols = structure.element_symbols()
        for index, symbol in enumerate(symbols):
            if symbol == "Fe":
                self.assertEqual(site_coordination(structure, index), "oct")

    def test_square_planar_is_not_tetrahedral(self):
        coords = [[0, 0, 0], [1.95, 0, 0], [-1.95, 0, 0], [0, 1.95, 0], [0, -1.95, 0]]
        structure = ChemicalStructure.with_zero_magnetic_moments(
            name="CuO4", lattice=np.eye(3) * 20.0, cartesian_coords=coords,
            atomic_labels=["Cu", "O", "O", "O", "O"], is_periodic=False,
        )
        self.assertEqual(site_coordination(structure, 0), "oct")

    def test_tetrahedral_sites_are_always_high_spin(self):
        structure = _cobalt_ferrite()
        descriptors = structure_ion_descriptors(structure, _assignment(structure), spin="LS")
        roles = structure.generation_parameters.site_roles
        for site, descriptor in descriptors.items():
            expected = ("tet", "HS") if roles[site] == "A" else ("oct", "LS")
            self.assertEqual((descriptor.coordination, descriptor.spin_state), expected)


class TetrahedralFrameTest(unittest.TestCase):
    def test_ideal_tetrahedron_frame_is_the_cube(self):
        structure = _cobalt_ferrite()
        frame = local_d_frame(structure, 0, "tet")
        self.assertTrue(_is_signed_permutation(frame), frame)

    def test_frame_rotates_with_the_structure(self):
        structure = _cobalt_ferrite()
        rotation = _rotation()
        frame = local_d_frame(structure, 0, "tet")
        rotated = local_d_frame(_rotated(structure, rotation), 0, "tet")
        self.assertTrue(_is_signed_permutation(frame.T @ rotation.T @ rotated))

    def test_couplings_do_not_depend_on_orientation(self):
        # With the old global-frame fallback the tetrahedral sites' orbitals did
        # not turn with the crystal, so this failed for any A-site ion with an
        # anisotropic occupancy (Co2+ here).
        structure = _cobalt_ferrite()
        assignment = _assignment(structure)
        reference = _j_matrix(structure, assignment)
        rotated = _j_matrix(_rotated(structure, _rotation()), assignment)
        np.testing.assert_allclose(rotated, reference, atol=1e-10)


class RadiusAndOrbitalGatingTest(unittest.TestCase):
    def test_tetrahedral_reference_length_uses_the_iv_radius(self):
        structure = generate_normal_spinel(
            "MnFe2O4", a_site="Mn", b_site="Fe", a=8.5, u=0.26,
            sublattice_oxidation_states={"A": {"Mn": (2,)}, "B": {"Fe": (3,)}},
        )
        bridges = build_bridges(
            structure, structure_ion_descriptors(structure, _assignment(structure))
        )
        o_radius = _shannon_crystal_radius("O", -2, None)
        expected = {
            "Mn": _shannon_crystal_radius("Mn", 2, "High Spin", "IV") + o_radius,
            "Fe": _shannon_crystal_radius("Fe", 3, "High Spin", "VI") + o_radius,
        }
        ends = {(b.metal_i, b.R0_iL) for b in bridges} | {(b.metal_j, b.R0_jL) for b in bridges}
        self.assertEqual({metal for metal, _ in ends}, {"Mn", "Fe"})
        for metal, r0 in ends:
            self.assertAlmostEqual(r0, expected[metal])
        self.assertLess(
            expected["Mn"], _shannon_crystal_radius("Mn", 2, "High Spin", "VI") + o_radius
        )

    def test_eg_orbital_channels_are_octahedral_only(self):
        self.assertTrue(is_eg_active(ion_descriptor("Cu", 2, "HS", "oct")))
        self.assertFalse(is_eg_active(ion_descriptor("Cu", 2, "HS", "tet")))
        self.assertTrue(is_occ_empty_active(ion_descriptor("Mn", 3, "HS", "oct")))
        self.assertFalse(is_occ_empty_active(ion_descriptor("Fe", 2, "HS", "tet")))


class SiteClassValenceTest(unittest.TestCase):
    def test_magnetite_fractions_are_resolved_by_sublattice(self):
        structure = generate_inverse_spinel(
            "Fe3O4", a_site="Fe", b_site="Fe", a=8.396, u=0.2549,
            sublattice_oxidation_states={"A": {"Fe": (3,)}, "B": {"Fe": (2, 3)}},
        )
        descriptors = structure_ion_descriptors(structure, _assignment(structure))
        fractions = site_class_oxidation_fractions(descriptors)
        self.assertEqual(fractions[("Fe", "tet")], ((3, 1.0),))
        self.assertEqual(fractions[("Fe", "oct")], ((2, 0.5), (3, 0.5)))

    def test_double_exchange_weight(self):
        mixed = ((2, 0.5), (3, 0.5))
        self.assertAlmostEqual(_de_resonance_weight(mixed, mixed), 0.5)
        third = ((3, 0.25), (4, 0.75))
        self.assertAlmostEqual(_de_resonance_weight(third, third), 1 - 0.25 ** 2 - 0.75 ** 2)
        # A single-valent end cannot take part in the resonance.
        self.assertAlmostEqual(_de_resonance_weight(((3, 1.0),), mixed), 0.0)


if __name__ == "__main__":
    unittest.main()
