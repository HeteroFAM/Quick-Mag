"""Tests for spinel occupancy rules, provenance, and oxidation constraints."""

import os
import sys
import unittest
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.generation import (  # noqa: E402
    generate_high_entropy_perovskite,
    generate_high_entropy_spinel,
    generate_inverse_spinel,
    generate_normal_spinel,
    generate_spinel,
    spinel_structure_from_parameters,
)
from quick_mag.magnetic_moments import (  # noqa: E402
    expand_group_assignments_to_site_assignments,
)
from quick_mag.oxidation_state_energy import enumerate_site_group_assignments  # noqa: E402
from quick_mag.spinel_builder import SpinelBuild  # noqa: E402
from quick_mag.spinel_composition import (  # noqa: E402
    distribute_cations,
    exact_count_labels,
    exact_counts,
    sublattice_composition,
    sublattice_oxidation_constraints,
)
from quick_mag.structure import (  # noqa: E402
    PerovskiteGenerationParameters,
    SpinelGenerationParameters,
    build_from_generation_parameters,
    generation_family,
    generation_parameters_from_json,
    generation_parameters_to_json,
)

FIVE = ["Co", "Cr", "Fe", "Mn", "Ni"]


def _role_symbols(structure, role):
    roles = structure.generation_parameters.site_roles
    return [s for s, r in zip(structure.element_symbols(), roles) if r == role]


class ExactCountTest(unittest.TestCase):
    def test_counts_sum_and_round_to_nearest(self):
        self.assertEqual(exact_counts([0.2] * 5, 64), [13, 13, 13, 13, 12])
        self.assertEqual(exact_counts([0.5, 0.25, 0.25], 8), [4, 2, 2])
        self.assertEqual(sum(exact_counts([0.37, 0.41, 0.22], 101)), 101)

    def test_exact_labels_have_exactly_the_composition(self):
        labels = exact_count_labels([("Co", 1), ("Ni", 3)], 16, seed_parts=("t",))
        self.assertEqual(labels.count("Co"), 4)
        self.assertEqual(labels.count("Ni"), 12)

    def test_exact_labels_are_reproducible_and_seeded(self):
        a = exact_count_labels([(el, 1) for el in FIVE], 40, seed_parts=("t", 0))
        b = exact_count_labels([(el, 1) for el in FIVE], 40, seed_parts=("t", 0))
        c = exact_count_labels([(el, 1) for el in FIVE], 40, seed_parts=("t", 1))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(sorted(a), sorted(c))


class InversionTest(unittest.TestCase):
    def _build(self, inversion, **kwargs):
        return generate_spinel(
            "CFO", a_site="Co", b_site="Fe", a=8.39, inversion=inversion,
            n_cells_x=2, n_cells_y=1, n_cells_z=1, **kwargs,
        )

    def test_normal_and_inverse_limits(self):
        normal = generate_normal_spinel("n", a_site="Zn", b_site="Fe")
        self.assertEqual(set(_role_symbols(normal, "A")), {"Zn"})
        self.assertEqual(set(_role_symbols(normal, "B")), {"Fe"})
        inverse = generate_inverse_spinel("i", a_site="Ni", b_site="Fe")
        self.assertEqual(set(_role_symbols(inverse, "A")), {"Fe"})
        b = _role_symbols(inverse, "B")
        self.assertEqual((b.count("Ni"), b.count("Fe")), (8, 8))

    def test_partial_inversion_swaps_the_rounded_number_of_pairs(self):
        structure = self._build(0.75)  # 16 tetrahedral sites -> 12 swaps
        self.assertEqual(_role_symbols(structure, "A").count("Fe"), 12)
        self.assertEqual(_role_symbols(structure, "B").count("Co"), 12)
        self.assertEqual(structure.element_symbols().count("Co"), 16)

    def test_inverted_sites_are_nested_across_inversion_degrees(self):
        def inverted(x):
            labels = self._build(x).element_symbols()
            return {i for i, s in enumerate(labels) if s == "Fe" and i < 16}

        low, high = inverted(0.25), inverted(0.5)
        self.assertEqual(len(low), 4)
        self.assertTrue(low < high)

    def test_sample_index_redraws_the_arrangement(self):
        a = self._build(0.5).element_symbols()
        b = self._build(0.5, sample_index=1).element_symbols()
        self.assertNotEqual(a, b)
        self.assertEqual(sorted(a), sorted(b))

    def test_inversion_outside_unit_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            self._build(1.5)


class HighEntropyTest(unittest.TestCase):
    def test_exact_sampling_matches_each_sublattice_mix(self):
        structure = generate_high_entropy_spinel(
            "HES",
            a_sites=[("Zn", 0.5), ("Mg", 0.5)],
            b_sites=[(el, 0.2) for el in FIVE],
            n_cells_x=2, n_cells_y=2, n_cells_z=2,
        )
        self.assertEqual(structure.atom_count, 448)
        composition = sublattice_composition(structure)
        self.assertEqual(composition["A"], {"Zn": 32, "Mg": 32} if list(composition["A"])[0] == "Zn" else {"Mg": 32, "Zn": 32})
        self.assertEqual(sorted(composition["B"].values()), [25, 25, 26, 26, 26])
        self.assertEqual(composition["X"], {"O": 256})

    def test_iid_sampling_is_available_and_differs(self):
        kwargs = dict(
            a_sites=[("Zn", 1.0)], b_sites=[(el, 0.2) for el in FIVE],
            n_cells_x=2, n_cells_y=2, n_cells_z=2,
        )
        exact = generate_high_entropy_spinel("e", **kwargs)
        iid = generate_high_entropy_spinel("i", sampling="iid", **kwargs)
        self.assertEqual(iid.generation_parameters.sampling, "iid")
        self.assertNotEqual(exact.element_symbols(), iid.element_symbols())

    def test_cation_pool_with_preferences(self):
        structure = generate_high_entropy_spinel(
            "HES",
            cations=[(el, 1.0) for el in FIVE],
            site_preferences={"Cr": "oct"},
            n_cells_x=2, n_cells_y=2, n_cells_z=2,
        )
        self.assertNotIn("Cr", _role_symbols(structure, "A"))
        self.assertEqual(len(set(_role_symbols(structure, "A"))), 4)

    def test_pool_and_explicit_mixes_are_exclusive(self):
        with self.assertRaises(ValueError):
            generate_high_entropy_spinel(
                "x", cations=[("Co", 1)], a_sites=[("Co", 1)], b_sites=[("Fe", 1)]
            )

    def test_perovskite_high_entropy_sampling_is_unchanged(self):
        # The perovskite builder keeps its independent draws, so saved
        # high-entropy perovskites regenerate exactly as before.
        structure = generate_high_entropy_perovskite(
            "HEA",
            a_sites=[("La", 1.0)],
            b_sites=[(el, 0.2) for el in FIVE],
            x_sites=[("O", 1.0)],
            a=3.9, n_cells_x=2, n_cells_y=2, n_cells_z=2,
        )
        self.assertFalse(hasattr(structure.generation_parameters, "sampling"))


class DistributeCationsTest(unittest.TestCase):
    def _check_fill(self, dist):
        self.assertAlmostEqual(sum(dist.tetrahedral.values()), 1.0, places=9)
        self.assertAlmostEqual(sum(dist.octahedral.values()), 2.0, places=9)

    def test_no_preference_is_the_random_distribution(self):
        dist = distribute_cations([(el, 1) for el in FIVE])
        self._check_fill(dist)
        for el in FIVE:
            self.assertAlmostEqual(dist.tetrahedral_fraction(el), 1.0 / 3.0)

    def test_hard_preferences(self):
        dist = distribute_cations([("Zn", 1), ("Fe", 2)], {"Zn": "tet"})
        self._check_fill(dist)
        self.assertAlmostEqual(dist.tetrahedral["Zn"], 1.0)
        self.assertAlmostEqual(dist.inversion(["Fe"]), 0.0)

        dist = distribute_cations([("Ni", 1), ("Fe", 2)], {"Ni": "oct"})
        self._check_fill(dist)
        self.assertAlmostEqual(dist.inversion(["Fe"]), 1.0)

    def test_energy_preference_follows_ideal_sublattice_equilibrium(self):
        temperature = 1000.0
        dist = distribute_cations([("Mg", 1), ("Al", 2)], {"Mg": -20.0}, temperature=temperature)
        self._check_fill(dist)
        # Exchange equilibrium: (y_Mg^A y_Al^B) / (y_Al^A y_Mg^B) = exp(20 / RT).
        y_a = {el: dist.tetrahedral[el] for el in ("Mg", "Al")}
        y_b = {el: dist.octahedral[el] / 2.0 for el in ("Mg", "Al")}
        ratio = (y_a["Mg"] * y_b["Al"]) / (y_a["Al"] * y_b["Mg"])
        self.assertAlmostEqual(ratio, np.exp(20.0 / (8.314462618e-3 * temperature)), places=6)

    def test_impossible_constraints_are_refused(self):
        with self.assertRaises(ValueError):
            distribute_cations([("Zn", 2), ("Fe", 1)], {"Zn": "tet"})
        with self.assertRaises(ValueError):
            distribute_cations([("Cr", 2.5), ("Fe", 0.5)], {"Cr": "oct"})
        with self.assertRaises(ValueError):
            distribute_cations([("Co", 1)], {"Ni": "oct"})
        with self.assertRaises(ValueError):
            distribute_cations([("Co", 1)], {"Co": "sideways"})


class ProvenanceTest(unittest.TestCase):
    def test_json_round_trip_regenerates_the_identical_structure(self):
        original = generate_high_entropy_spinel(
            "HES",
            cations=[(el, 1.0) for el in FIVE],
            site_preferences={"Cr": "oct"},
            a=8.3, c=8.5, u=0.258, cell="primitive",
            n_cells_x=2, n_cells_y=2, n_cells_z=2, seed=7, sample_index=3,
            sublattice_oxidation_states={"B": {"Mn": (3, 4)}},
        )
        payload = generation_parameters_to_json(original.generation_parameters)
        self.assertEqual(payload["family"], "spinel")
        params = generation_parameters_from_json(payload)
        self.assertIsInstance(params, SpinelGenerationParameters)
        rebuilt = spinel_structure_from_parameters(params, name="HES")
        self.assertEqual(rebuilt.atomic_labels, original.atomic_labels)
        np.testing.assert_allclose(rebuilt.cartesian_coords, original.cartesian_coords)
        np.testing.assert_allclose(rebuilt.lattice, original.lattice)
        self.assertEqual(params.sublattice_oxidation_states, {"B": {"Mn": (3, 4)}})

    def test_perovskite_payloads_are_tagged_and_untagged_ones_still_load(self):
        params = PerovskiteGenerationParameters(
            center=np.zeros(3), n_oct_x=0, n_oct_y=0, n_oct_z=0,
            center_to_vertex_distance_x=2.0, center_to_vertex_distance_y=2.0,
            center_to_vertex_distance_z=2.0, tilt_system="a0a0a0",
            tilt_angle_x_deg=0.0, tilt_angle_y_deg=0.0, tilt_angle_z_deg=0.0,
            periodic=True, a_site_element="La", b_site_element="Fe", x_site_element="O",
        )
        payload = generation_parameters_to_json(params)
        self.assertEqual(payload["family"], "perovskite")
        payload.pop("family")
        loaded = generation_parameters_from_json(payload)
        self.assertIsInstance(loaded, PerovskiteGenerationParameters)
        self.assertEqual(generation_family(loaded), "perovskite")
        with self.assertRaises(ValueError):
            generation_parameters_from_json({"family": "garnet"})

    def test_build_dispatches_on_family(self):
        structure = generate_spinel("s", a_site="Co", b_site="Fe")
        self.assertIsInstance(
            build_from_generation_parameters(structure.generation_parameters), SpinelBuild
        )

    def test_perovskite_only_consumers_refuse_spinels(self):
        from quick_mag.classify_spin_structure import site_indexing_from_generation_parameters
        from quick_mag.magnetic_cli import _resolve_site_indexing
        from quick_mag.remote.protocol import ProtocolError, check_structure_against_params

        structure = generate_spinel("s", a_site="Co", b_site="Fe")
        params = structure.generation_parameters
        with self.assertRaises(ValueError):
            site_indexing_from_generation_parameters(
                params, build_from_generation_parameters(params)
            )
        self.assertIsNone(_resolve_site_indexing(structure, list(range(24))))
        with self.assertRaises(ProtocolError):
            check_structure_against_params(structure, {"calculation": "reconstruct"})

    def test_unknown_mode_and_sampling_are_rejected(self):
        with self.assertRaises(ValueError):
            SpinelGenerationParameters(formula_mode="garnet")
        with self.assertRaises(ValueError):
            SpinelGenerationParameters(sampling="sobol")
        with self.assertRaises(ValueError):
            SpinelGenerationParameters(sublattice_oxidation_states={"T": {"Fe": (3,)}})


class SublatticeOxidationTest(unittest.TestCase):
    def test_magnetite_puts_fe3_on_every_tetrahedral_site(self):
        structure = generate_inverse_spinel(
            "Fe3O4", a_site="Fe", b_site="Fe", a=8.396, u=0.2549,
            sublattice_oxidation_states={"A": {"Fe": (3,)}, "B": {"Fe": (2, 3)}},
        )
        constraints = sublattice_oxidation_constraints(structure)
        self.assertEqual(len(constraints), 24)
        ranked = enumerate_site_group_assignments(
            structure.element_symbols(), constraints=constraints, top_k=1
        )
        assignment = expand_group_assignments_to_site_assignments(
            ranked, structure, max_assignments=1
        )[0]
        states = assignment.site_oxidation_states
        roles = structure.generation_parameters.site_roles
        tet = [int(s) for s, r in zip(states, roles) if r == "A"]
        oct_ = [int(s) for s, r in zip(states, roles) if r == "B"]
        self.assertEqual(set(tet), {3})
        self.assertEqual((oct_.count(2), oct_.count(3)), (8, 8))

    def test_explicit_restrictions_override_the_stored_ones(self):
        structure = generate_normal_spinel("ZnFe2O4", a_site="Zn", b_site="Fe")
        constraints = sublattice_oxidation_constraints(structure, {"B": {"Fe": [3]}})
        self.assertEqual(sorted(constraints), list(range(8, 24)))

    def test_structures_without_spinel_provenance_are_refused(self):
        structure = generate_spinel("s", a_site="Co", b_site="Fe")
        structure.generation_parameters = replace(
            structure.generation_parameters, site_roles=["A"]
        )
        with self.assertRaises(ValueError):
            sublattice_oxidation_constraints(structure)


if __name__ == "__main__":
    unittest.main()
