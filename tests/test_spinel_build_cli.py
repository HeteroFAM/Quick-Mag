"""Tests for ``quick-mag build-spinel`` and its place in ``::`` chains."""

import argparse
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag import cli, spinel_build_cli  # noqa: E402
from quick_mag.structure_utils import read_structure  # noqa: E402


def _args(*argv):
    parser = argparse.ArgumentParser()
    spinel_build_cli.configure_spinel_build_parser(parser)
    return parser.parse_args(list(argv))


def _build(*argv):
    return spinel_build_cli.build_spinel_structures(_args(*argv))


class BuildSpinelTest(unittest.TestCase):
    def test_default_is_one_normal_cobalt_ferrite_cell(self):
        (structure,) = _build()
        self.assertEqual(structure.name, "CoFe2O4")
        self.assertEqual(structure.atom_count, 56)

    def test_element_lists_expand_as_a_cartesian_product(self):
        structures = _build("--a-site", "Co,Ni,Zn", "--b-site", "Fe,Cr")
        self.assertEqual(len(structures), 6)
        self.assertIn("ZnCr2O4", [s.name for s in structures])

    def test_scans_and_zip(self):
        grid = _build("--u", "0.25:0.27:3", "--inversion", "0:1:2")
        self.assertEqual(len(grid), 6)
        zipped = _build("--u", "0.25:0.27:3", "--a", "8.3:8.5:3", "--zip")
        self.assertEqual(len(zipped), 3)
        self.assertEqual(
            [round(s.generation_parameters.u, 3) for s in zipped], [0.25, 0.26, 0.27]
        )
        with self.assertRaises(ValueError):
            _build("--u", "0.25:0.27:3", "--a", "8.3:8.5:2", "--zip")

    def test_primitive_cell_and_supercell(self):
        (structure,) = _build("--cell", "primitive", "--n-cells-x", "2")
        self.assertEqual(structure.atom_count, 28)

    def test_high_entropy_mixes_and_samples(self):
        structures = _build(
            "--formula", "high_entropy",
            "--a-sites", "Zn:1,Mg:1", "--b-sites", "Co,Cr,Fe,Mn,Ni",
            "--n-cells-x", "2", "--num-samples", "3",
        )
        self.assertEqual([s.name for s in structures], ["HES_s0", "HES_s1", "HES_s2"])
        self.assertEqual(len({tuple(s.atomic_labels) for s in structures}), 3)

    def test_high_entropy_cation_pool(self):
        (structure,) = _build(
            "--formula", "high_entropy", "--cations", "Co,Cr,Fe,Mn,Ni",
            "--site-preference", "Cr:oct,Mn:-5", "--n-cells-x", "2",
        )
        roles = structure.generation_parameters.site_roles
        tet = {s for s, r in zip(structure.element_symbols(), roles) if r == "A"}
        self.assertNotIn("Cr", tet)

    def test_invalid_combinations(self):
        with self.assertRaises(ValueError):
            _build("--formula", "high_entropy")
        with self.assertRaises(ValueError):
            _build("--formula", "high_entropy", "--cations", "Co,Fe", "--a-sites", "Co")
        with self.assertRaises(ValueError):
            _build("--formula", "high_entropy", "--cations", "Co,Fe", "--inversion", "0.5")
        with self.assertRaises(ValueError):
            _build("--site-preference", "Cr:oct")
        with self.assertRaises(ValueError):
            _build("--inversion", "1.2")
        with self.assertRaises(ValueError):
            _build("--oxidation", "T:Fe=3")

    def test_oxidation_restrictions_are_recorded(self):
        (structure,) = _build("--oxidation", "A:Fe=3", "--oxidation", "B:Fe=2,3+")
        self.assertEqual(
            structure.generation_parameters.sublattice_oxidation_states,
            {"A": {"Fe": (3,)}, "B": {"Fe": (2, 3)}},
        )

    def test_writes_readable_structures(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                status = spinel_build_cli.main(
                    ["--a-site", "Mg", "--b-site", "Al", "--a", "8.08", "-o", tmp]
                )
            self.assertEqual(status, 0)
            written = read_structure(Path(tmp) / "MgAl2O4.cif")
            self.assertEqual(sorted(set(written.element_symbols())), ["Al", "Mg", "O"])
            self.assertEqual(written.atom_count, 56)


class SpinelChainTest(unittest.TestCase):
    def _stages(self, argv):
        parser = cli.build_parser()
        return [parser.parse_args(segment) for segment in cli.split_chain(argv)]

    def test_build_spinel_may_start_a_chain(self):
        cli.validate_chain(self._stages(["build-spinel", "::", "solve"]))

    def test_build_spinel_must_be_first(self):
        with self.assertRaises(ValueError):
            cli.validate_chain(self._stages(["build", "::", "build-spinel"]))

    def test_chained_solve_honours_sublattice_oxidation(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = cli.main([
                "build-spinel", "--a-site", "Fe", "--b-site", "Fe", "--inversion", "1",
                "--oxidation", "A:Fe=3", "--oxidation", "B:Fe=2,3",
                "::", "solve", "--n-trials", "2", "--n-steps", "20", "--max-configs", "1",
            ])
        self.assertEqual(status, 0)
        text = out.getvalue()
        self.assertIn("Sublattice oxidation constraints on 24 atom(s)", text)
        self.assertIn("8xFe+2 + 16xFe+3", text)


if __name__ == "__main__":
    unittest.main()
