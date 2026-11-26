"""Extended xyz in and out, the states a file carries, and ``solve --out``."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.export_utils import export_structure  # noqa: E402
from quick_mag.extxyz_io import (  # noqa: E402
    Frame,
    extxyz_text,
    frame_oxidation_states,
    frame_spin_states,
    read_frames,
    write_extxyz,
)
from quick_mag.magnetic_cli import configure_solve_parser, run_solve  # noqa: E402
from quick_mag.site_states import SiteStates  # noqa: E402
from quick_mag.structure_utils import read_structure, read_structure_frames  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
LAMNO3 = ASSETS / "A_type" / "LaMnO3_222.cif"


def lamno3():
    return read_structure(LAMNO3)


def solve_args(*argv):
    return configure_solve_parser(argparse.ArgumentParser()).parse_args(list(argv))


# ---- reading and writing ---------------------------------------------------------------


def test_a_structure_round_trips_with_its_moments(tmp_path):
    st = lamno3()
    st.magnetic_moments[8:16, 2] = [4, -4, 4, -4, 4, -4, 4, -4]
    path = write_extxyz(tmp_path / "s.xyz", st)
    back = read_structure(path)
    assert back.atomic_labels == st.atomic_labels
    assert np.allclose(back.lattice, st.lattice)
    assert np.allclose(back.cartesian_coords, st.cartesian_coords, atol=1e-9)
    assert np.allclose(back.magnetic_moments, st.magnetic_moments)
    assert back.is_periodic and back.name == st.name


def test_extra_keys_and_columns_are_kept(tmp_path):
    st = lamno3()
    n = st.atom_count
    frame = Frame(
        structure=st,
        info={"energy": -1.25, "chemlab_build_node": "a553658eb20d", "title": "two words"},
        arrays={
            "site_key": np.array([f"B:{i}" for i in range(n)]),
            "oxidation": np.arange(n) % 4,
            "forces": np.arange(3 * n, dtype=float).reshape(n, 3),
        },
    )
    path = write_extxyz(tmp_path / "f.xyz", [frame, frame])
    frames = read_frames(path)
    assert len(frames) == 2
    got = frames[1]
    assert got.info["energy"] == -1.25
    assert got.info["chemlab_build_node"] == "a553658eb20d"
    assert got.info["title"] == "two words"
    assert list(got.arrays["site_key"]) == [f"B:{i}" for i in range(n)]
    assert got.arrays["forces"].shape == (n, 3)
    assert frame_oxidation_states(got)[:5] == [None, 1, 2, 3, None]


def test_a_file_written_elsewhere_is_read():
    # Three-component moments under another name, a logical column, no name key.
    text = (
        "2\n"
        'Lattice="4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0" '
        "Properties=species:S:1:pos:R:3:initial_magmoms:R:3:orbit:S:1:fixed:L:1:spin_state:S:1 "
        'pbc="T T F" note="made by hand"\n'
        "Fe 0.0 0.0 0.0  0.0 0.0  4.0  B T HS\n"
        "O  2.0 0.0 0.0  0.0 0.0  0.0  X F -\n"
    )
    (frame,) = read_frames(text, text=True)
    st = frame.structure
    assert st.element_symbols() == ["Fe", "O"]
    assert st.magnetic_moments[0, 2] == 4.0
    assert tuple(st.periodic_axes) == (True, True, False)
    assert frame.info["note"] == "made by hand"
    assert list(frame.arrays["orbit"]) == ["B", "X"]
    assert frame_spin_states(frame) == ["HS", None]
    assert frame_oxidation_states(frame) is None


def test_a_truncated_frame_is_an_error():
    with pytest.raises(ValueError, match="truncated"):
        read_frames("3\nLattice=\"1 0 0 0 1 0 0 0 1\"\nFe 0 0 0\n", text=True)


def test_read_structure_frames_wraps_other_formats():
    (frame,) = read_structure_frames(LAMNO3)
    assert frame.structure.atom_count == 40 and not frame.arrays and not frame.info


def test_unsupported_files_are_still_refused(tmp_path):
    path = tmp_path / "x.pdb"
    path.write_text("")
    with pytest.raises(ValueError, match="extended xyz"):
        read_structure(path)


def test_export_structure_writes_xyz(tmp_path):
    st = lamno3()
    written = export_structure(st, tmp_path, formats=("xyz",), xyz_info={"energy": 2.0})
    assert written["xyz"] == 1 and written["cif"] == 0
    (frame,) = read_frames(tmp_path / f"{st.name}.xyz")
    assert frame.info["energy"] == 2.0


# ---- which wins: a rule or a per-atom value --------------------------------------------


def test_per_atom_values_win_over_rules_by_default():
    st = lamno3()
    symbols = st.element_symbols()
    per_atom = [4 if s == "Mn" else None for s in symbols]
    states = SiteStates.parse(["Mn=3"], site_oxidation=per_atom)
    mn = symbols.index("Mn")
    assert states.constraints(st)[mn] == frozenset([4])


def test_rules_win_over_file_values_when_asked():
    st = lamno3()
    symbols = st.element_symbols()
    mn = symbols.index("Mn")
    states = SiteStates.parse(
        ["Mn=3"], ["Mn=HS"],
        site_oxidation=[4 if s == "Mn" else None for s in symbols],
        site_spin=["LS" if s == "Mn" else None for s in symbols],
        rules_over_sites=True,
    )
    assert states.constraints(st)[mn] == frozenset([3])
    assert states.spin_states(st, np.full(st.atom_count, 3))[mn] == "HS"
    # Without a rule naming them, the file's values stand.
    only_file = SiteStates(
        site_oxidation=[4 if s == "Mn" else None for s in symbols],
        site_spin=["LS" if s == "Mn" else None for s in symbols],
        rules_over_sites=True,
    )
    assert only_file.constraints(st)[mn] == frozenset([4])
    assert only_file.spin_states(st, np.full(st.atom_count, 4))[mn] == "LS"


# ---- solve -----------------------------------------------------------------------------


def write_with_states(path, *, spin="LS", extra=True):
    st = lamno3()
    symbols = st.element_symbols()
    arrays = {
        "oxidation": np.array([3 if s in ("La", "Mn") else 0 for s in symbols]),
        "spin_state": np.array([spin if s == "Mn" else "-" for s in symbols]),
    }
    if extra:
        arrays["site_key"] = np.array([f"k{i}" for i in range(len(symbols))])
    return write_extxyz(path, Frame(structure=st, info={"chemlab_build_node": "abc"}, arrays=arrays))


def test_solve_takes_the_states_a_file_carries(tmp_path, capsys):
    path = write_with_states(tmp_path / "ls.xyz")
    assert run_solve(solve_args(str(path), "--max-configs", "2")) == 0
    out = capsys.readouterr().out
    assert "Oxidation states given for 16 atom(s) (structure file)." in out
    assert "Spin states given for 8 atom(s) (structure file)." in out
    assert "8xMn+3(oct,LS) 2 muB" in out


def test_solve_can_ignore_them_and_rules_go_over_them(tmp_path, capsys):
    path = write_with_states(tmp_path / "ls.xyz")
    assert run_solve(solve_args(str(path), "--max-configs", "1", "--ignore-file-states")) == 0
    out = capsys.readouterr().out
    assert "structure file" not in out
    assert "8xMn+3(oct,HS) 4 muB" in out

    assert run_solve(solve_args(str(path), "--max-configs", "1", "--spin", "Mn=HS")) == 0
    out = capsys.readouterr().out
    assert "Spin states given for 8 atom(s) (structure file)." in out
    assert "8xMn+3(oct,HS) 4 muB" in out


def test_a_plain_file_solves_as_before(capsys):
    assert run_solve(solve_args(str(LAMNO3), "--max-configs", "1")) == 0
    out = capsys.readouterr().out
    assert "structure file" not in out and "Wrote results" not in out


def test_solve_out_writes_what_the_manifest_lists(tmp_path, capsys):
    path = write_with_states(tmp_path / "ls.xyz")
    out_dir = tmp_path / "out"
    assert run_solve(solve_args(str(path), "--max-configs", "3", "--out", str(out_dir))) == 0
    assert "Wrote results" in capsys.readouterr().out

    manifest = json.loads((out_dir / "chemlab.json").read_text())
    assert manifest["chemlab-result"] == 1
    items = {item["name"]: item for item in manifest["items"]}
    assert set(items) == {"configs", "sites", "exchange", "exchange_pairs", "summary"}
    for item in items.values():
        assert (out_dir / item["file"]).is_file()
    assert items["configs"]["kind"] == "structures"
    assert items["configs"]["units"]["energy"] == "meV"
    assert items["sites"]["level"] == "atoms"
    assert items["exchange"]["units"]["value"] == "meV/mu_B^2"
    assert 0.01 < max(abs(e["value"]) for e in json.loads((out_dir / "exchange.graphs.json").read_text())["graphs"]["exchange"]["edges"]) < 100.0

    frames = read_frames(out_dir / "configs.xyz")
    assert len(frames) == items["configs"]["frames"]
    solved = [f for f in frames if f.info["source"] == "solved"]
    reference = [f for f in frames if f.info["source"] == "reference"]
    assert [f.info["rank"] for f in solved] == [1, 2, 3]
    assert {"G", "FM"} <= {f.info["label"] for f in reference}
    assert solved[0].info["delta_energy"] >= 0.0
    assert min(f.info["delta_energy"] for f in frames) == 0.0
    first = solved[0]
    symbols = first.structure.element_symbols()
    mn = [i for i, s in enumerate(symbols) if s == "Mn"]
    assert np.allclose(np.abs(first.structure.magnetic_moments[mn, 2]), 2.0)   # low spin
    assert list(first.arrays["spin_state"][mn]) == ["LS"] * 8
    assert set(np.asarray(first.arrays["oxidation"])[mn]) == {3}
    # what the input carried comes back
    assert list(first.arrays["site_key"]) == [f"k{i}" for i in range(len(symbols))]
    assert first.info["chemlab_build_node"] == "abc"

    with open(out_dir / "sites.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(symbols)
    assert rows[mn[0]]["spin_state"] == "LS" and rows[mn[0]]["oxidation"] == "3"
    assert float(rows[mn[0]]["moment"]) == 2.0

    graph = json.loads((out_dir / "exchange.graphs.json").read_text())
    assert graph["format"] == "chemlab-graphs/1"
    edges = graph["graphs"]["exchange"]["edges"]
    assert edges and all(e["a"] in mn and e["b"] in mn and e["a"] < e["b"] for e in edges)
    with open(out_dir / "exchange.csv") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == len(edges) == items["exchange_pairs"]["rows"]
    assert all(float(p["distance"]) < 8.0 for p in pairs)

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oxidation_from_file"] == 16 and summary["spin_states_from_file"] == 8
    assert summary["assignments"][0]["ground"]["energy"] == pytest.approx(
        min(f.info["energy"] for f in frames)
    )


def test_several_structures_get_prefixed_files(tmp_path):
    st = lamno3()
    path = write_extxyz(tmp_path / "two.xyz", [st, st])
    out_dir = tmp_path / "out"
    assert run_solve(solve_args(str(path), "--max-configs", "1", "--out", str(out_dir))) == 0
    manifest = json.loads((out_dir / "chemlab.json").read_text())
    names = [item["name"] for item in manifest["items"]]
    assert f"001_{st.name}.configs" in names and f"002_{st.name}.configs" in names
    assert (out_dir / f"002_{st.name}_configs.xyz").is_file()


def test_a_bad_oxidation_state_in_the_file_is_reported(tmp_path, capsys):
    st = lamno3()
    symbols = st.element_symbols()
    path = write_extxyz(
        tmp_path / "bad.xyz",
        Frame(structure=st, arrays={"oxidation": np.array([9 if s == "La" else 0 for s in symbols])}),
    )
    assert run_solve(solve_args(str(path), "--out", str(tmp_path / "out"))) == 1
    out = capsys.readouterr().out
    assert "Error:" in out or "No charge-balanced" in out
    assert not (tmp_path / "out").exists()


# ---- chgnet ----------------------------------------------------------------------------


def test_chgnet_xyz_extras_keep_the_input_columns():
    from quick_mag.chgnet_cli import _xyz_extras

    st = lamno3()
    n = st.atom_count

    class Result:
        final_structure = st
        energy = -300.0
        energy_per_atom = -7.5
        max_force = 0.01
        calculation = "single-point"
        converged = True
        forces = np.zeros((n, 3))
        magnetic_moments = -np.ones(n)

    frame = Frame(structure=st, info={"rank": 2}, arrays={"site_key": np.array(["k"] * n)})
    extras = _xyz_extras(Result, frame)
    assert extras["xyz_info"]["rank"] == 2 and extras["xyz_info"]["energy"] == -300.0
    assert set(extras["xyz_arrays"]) == {"site_key", "forces", "chgnet_absm"}
    assert np.all(extras["xyz_arrays"]["chgnet_absm"] == 1.0)
    text = extxyz_text(st, info=extras["xyz_info"], arrays=extras["xyz_arrays"])
    (back,) = read_frames(text, text=True)
    assert back.info["converged"] is True and back.arrays["forces"].shape == (n, 3)
