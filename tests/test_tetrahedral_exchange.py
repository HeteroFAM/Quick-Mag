"""Coordination-aware exchange: tetrahedral sites (spinel A sites) in the polarization model."""

import os
import sys
import unittest

import numpy as np
import pytest

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
    crystal_field_basis,
    level_filling,
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
    def test_ideal_tetrahedron_levels_are_e_and_t2(self):
        # e = (dz2, dx2-y2) and t2 = (dxy, dxz, dyz) on the axes of the cube the ligands sit on, which are the
        # crystal axes here: the crystal-field orbitals span exactly those two sets.
        structure = _cobalt_ferrite()
        basis, _split = crystal_field_basis(structure, 0, "tet")
        np.testing.assert_allclose(basis @ basis.T, np.eye(5), atol=1e-10)
        np.testing.assert_allclose(basis[:3, 3:], 0.0, atol=1e-8)
        np.testing.assert_allclose(basis[3:, :3], 0.0, atol=1e-8)

    def test_levels_turn_with_the_structure(self):
        from quick_mag.sk_table import orbital_tensors

        structure = _cobalt_ferrite()
        rotation = _rotation()
        for site, label in ((0, "tet"), (len(structure.atomic_labels) // 4, "oct")):
            if structure.atomic_labels[site] == "O":
                continue
            Q = orbital_tensors(crystal_field_basis(structure, site, label)[0])
            Qr = orbital_tensors(crystal_field_basis(_rotated(structure, rotation), site, label)[0])
            for shell in (slice(0, 3), slice(3, 5)):       # the projector on each level turns with the crystal
                proj = np.einsum("aij,akl->ijkl", Q[shell], Q[shell])
                proj_r = np.einsum("aij,akl->ijkl", Qr[shell], Qr[shell])
                turned = np.einsum("im,jn,ko,lp,mnop->ijkl", rotation, rotation, rotation, rotation, proj)
                np.testing.assert_allclose(proj_r, turned, atol=1e-8)

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
    def test_tetrahedral_reference_length_is_the_bond_valence_one(self):
        """R0 = the octahedral (Shannon VI) length + b ln(CN / 6): continuous in the cage, no label. At CN 4 the
        shift (-0.150 A) is the Shannon IV - VI difference to within 0.02 A."""
        from quick_mag.polarization_model import BOND_VALENCE_B

        structure = generate_normal_spinel(
            "MnFe2O4", a_site="Mn", b_site="Fe", a=8.5, u=0.26,
            sublattice_oxidation_states={"A": {"Mn": (2,)}, "B": {"Fe": (3,)}},
        )
        bridges = build_bridges(
            structure, structure_ion_descriptors(structure, _assignment(structure))
        )
        o_radius = _shannon_crystal_radius("O", -2, None)
        expected = {
            "Mn": _shannon_crystal_radius("Mn", 2, "High Spin", "VI") + o_radius + BOND_VALENCE_B * np.log(4 / 6),
            "Fe": _shannon_crystal_radius("Fe", 3, "High Spin", "VI") + o_radius,
        }
        ends = {(b.metal_i, b.R0_iL) for b in bridges} | {(b.metal_j, b.R0_jL) for b in bridges}
        self.assertEqual({metal for metal, _ in ends}, {"Mn", "Fe"})
        for metal, r0 in ends:
            self.assertAlmostEqual(r0, expected[metal], places=6)
        iv = _shannon_crystal_radius("Mn", 2, "High Spin", "IV") + o_radius
        self.assertLess(abs(expected["Mn"] - iv), 0.025)

    def test_level_filling_follows_the_k2_order_inside_each_level(self):
        """One rule for every d count: full orbitals lowest, then half-filled, then empty, inside each level."""
        lv = np.array([0.1, -0.2, 0.3, -0.5, 0.5])            # t2g slots 0-2, eg slots 3-4
        half, empty = level_filling(ion_descriptor("Mn", 3, "HS", "oct"), lv)     # t2g^3 eg^1
        np.testing.assert_allclose(half, [1, 1, 1, 1, 0]); np.testing.assert_allclose(empty, [0, 0, 0, 0, 1])
        half, empty = level_filling(ion_descriptor("Cu", 2, "HS", "oct"), lv)     # eg^3: the hole in the upper eg
        np.testing.assert_allclose(half, [0, 0, 0, 0, 1]); np.testing.assert_allclose(empty, 0)
        half, empty = level_filling(ion_descriptor("Ti", 3, "HS", "oct"), lv)     # t2g^1: the lowest t2g
        np.testing.assert_allclose(half, [0, 1, 0, 0, 0]); np.testing.assert_allclose(empty, [1, 0, 1, 1, 1])
        half, _ = level_filling(ion_descriptor("V", 3, "HS", "oct"), lv)          # t2g^2
        np.testing.assert_allclose(half, [1, 1, 0, 0, 0])
        half, _ = level_filling(ion_descriptor("Fe", 2, "HS", "oct"), lv)         # minority t2g electron: lowest
        np.testing.assert_allclose(half, [1, 0, 1, 1, 1])
        # degenerate levels share the filling: the shell average
        half, empty = level_filling(ion_descriptor("Ti", 3, "HS", "oct"), np.array([0.0, 0.0, 0.0, -0.5, 0.5]))
        np.testing.assert_allclose(half, [1 / 3] * 3 + [0, 0])
        half, empty = level_filling(ion_descriptor("V", 3, "HS", "oct"), np.array([-0.1, 0.2, 0.2, 0.0, 0.0]))
        np.testing.assert_allclose(half, [1, 0.5, 0.5, 0, 0]); np.testing.assert_allclose(empty, [0, 0.5, 0.5, 1, 1])


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


# ---- continuous level ordering (no oct/tet classification) ----------------------------------------------------

def _cage_structure(directions, r=1.9, element="Co"):
    from quick_mag.structure import ChemicalStructure

    u = np.asarray(directions, float)
    u = u / np.linalg.norm(u, axis=1)[:, None]
    coords = np.vstack([np.zeros(3), r * u])
    labels = [element] + ["O"] * len(u)
    return ChemicalStructure(name="cage", lattice=np.eye(3) * 30.0, cartesian_coords=coords, atomic_labels=labels,
                             magnetic_moments=np.zeros((len(labels), 3)), is_periodic=False)


def _flattened_tet(max_angle_deg):
    t = np.radians(max_angle_deg / 2)
    s, c = np.sin(t), np.cos(t)
    return [[s, 0, c], [-s, 0, c], [0, s, -c], [0, -s, -c]]


def test_tetrahedral_fraction_of_ideal_and_flattened_cages():
    from quick_mag.polarization_model import tetrahedral_fraction

    I = np.eye(3)
    assert tetrahedral_fraction(_cage_structure(np.vstack([I, -I])), 0) == pytest.approx(0.0, abs=1e-9)
    assert tetrahedral_fraction(_cage_structure(_flattened_tet(109.4712)), 0) == pytest.approx(1.0, abs=1e-6)
    assert tetrahedral_fraction(_cage_structure(_flattened_tet(180.0)), 0) == pytest.approx(0.11, abs=0.01)
    xs = [tetrahedral_fraction(_cage_structure(_flattened_tet(a)), 0) for a in np.linspace(109.4712, 180.0, 60)]
    assert all(b <= a + 1e-9 for a, b in zip(xs, xs[1:]))          # monotonic: tetrahedron -> square plane
    assert max(abs(b - a) for a, b in zip(xs, xs[1:])) < 0.06       # and continuous


def test_coupling_is_continuous_across_the_old_trans_pair_threshold(monkeypatch):
    """The old rule called a four-coordinate site square planar once an L-M-L cosine fell below -0.7 (134.4 deg)
    and switched its level ordering there. Two corner-sharing Co3+ cages flattened through that angle now give a
    coupling that moves smoothly. (D2d puts a k = 4 doublet at the octahedral template's cut here: both choices
    are averaged, ``crystal_field_bases``.) The field is the cages' own: in a bare two-cage cluster the k = 2 levels
    inside t2g cross along the path, and an orbital-order switch at a crossing is a step by design."""
    import quick_mag.polarization_model as PM

    monkeypatch.setattr(PM, "CF_FIELD_CUTOFF", 2.6)
    from quick_mag.ion_descriptors import ion_descriptor
    from quick_mag.polarization_model import bridge_J, build_bridges, load_params, DEFAULT_PARAMS_PATH
    from quick_mag.structure import ChemicalStructure

    params = load_params(DEFAULT_PARAMS_PATH)

    def j_at(angle):
        u = np.asarray(_flattened_tet(angle), float)
        u = u / np.linalg.norm(u, axis=1)[:, None]
        m1, r = np.zeros(3), 1.9
        o = m1 + r * u[0]
        m2 = o + r * u[0]                         # second cage through the shared O, mirrored
        coords = [m1, m2, o] + [m1 + r * v for v in u[1:]] + [m2 - r * v for v in u[1:]]
        labels = ["Co", "Co"] + ["O"] * 7
        st = ChemicalStructure(name="pair", lattice=np.eye(3) * 30.0, cartesian_coords=np.array(coords),
                               atomic_labels=labels, magnetic_moments=np.zeros((9, 3)), is_periodic=False)
        desc = {0: ion_descriptor("Co", 3, "HS", "oct"), 1: ion_descriptor("Co", 3, "HS", "oct")}
        (b,) = [b for b in build_bridges(st, desc, valence="site") if b.ligand_index == 2]
        return bridge_J(b, params)

    js = [j_at(a) for a in np.linspace(128.0, 139.0, 23)]   # the k = 4 levels themselves cross near 140.3 deg
    steps = np.abs(np.diff(js))
    assert steps.max() < 1.2 * np.median(steps)


def test_a_degenerate_level_at_the_cut_is_averaged():
    from quick_mag.polarization_model import crystal_field_bases

    st = _cage_structure(_flattened_tet(135.0))
    oct_choices = crystal_field_bases(st, 0, "oct")
    assert len(oct_choices) == 2 and sum(w for w, _b, _s in oct_choices) == pytest.approx(1.0)
    # the averaged eg projector is a function of the cage alone (rotation covariant)
    def eg_projector(choices):
        from quick_mag.polarization_model import EG_INDICES
        return sum(w * b[list(EG_INDICES)].T @ b[list(EG_INDICES)] for w, b, _s in choices)
    P = eg_projector(oct_choices)
    assert np.trace(P) == pytest.approx(2.0)
    assert len(crystal_field_bases(_cage_structure(np.vstack([np.eye(3), -np.eye(3)])), 0, "oct")) == 1
