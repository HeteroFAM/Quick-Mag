from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quick_mag.generation import (  # noqa: E402
    generate_double_perovskite,
    generate_high_entropy_perovskite,
    generate_single_perovskite,
)
from quick_mag.ion_descriptors import structure_ion_descriptors  # noqa: E402
from quick_mag.magnetic_moments import (  # noqa: E402
    expand_distribution_to_site_assignments,
)
from quick_mag.oxidation_state_energy import (  # noqa: E402
    enumerate_oxidation_states_by_energy,
)
from quick_mag.polarization_model import (  # noqa: E402
    bridge_J,
    bridge_J_components,
    build_Jeff_matrix,
    build_bridges,
    default_params,
    element_oxidation_fractions,
    pair_couplings,
)
from quick_mag.structure import ChemicalStructure  # noqa: E402


def _bridges(structure):
    """(bridges, params, compact index map) for a structure's best assignment."""
    ranked = enumerate_oxidation_states_by_energy(
        structure.element_symbols(), charge=0, max_mixing=2, top_k=5
    )
    assignment = expand_distribution_to_site_assignments(
        [distribution for distribution, _energy in ranked], structure
    )[0]
    descriptors = structure_ion_descriptors(structure, assignment)
    return (
        build_bridges(structure, descriptors),
        default_params(),
        {site: i for i, site in enumerate(sorted(descriptors))},
    )


def _couplings(structure):
    """(pairs, J matrix, compact index map) for a structure's best assignment."""
    bridges, params, site_index = _bridges(structure)
    return (
        pair_couplings(bridges, params),
        build_Jeff_matrix(bridges, site_index, params),
        site_index,
    )


@pytest.fixture(scope="module")
def lafeo3():
    return generate_single_perovskite(
        "LaFeO3",
        a_site="La",
        b_site="Fe",
        x_site="O",
        a=4.0,
        n_cells_x=2,
        n_cells_y=2,
        n_cells_z=2,
    )


@pytest.fixture(scope="module")
def la2femno6():
    return generate_double_perovskite(
        "La2FeMnO6",
        a_site="La",
        b_site="Fe",
        b2_site="Mn",
        x_site="O",
        a=4.0,
        n_cells_x=2,
        n_cells_y=2,
        n_cells_z=2,
    )


def test_pair_couplings_sum_to_the_matrix(lafeo3):
    """Every pair's j_eff is exactly the matrix element it was summed into."""
    pairs, matrix, site_index = _couplings(lafeo3)
    assert pairs
    for pair in pairs:
        a, b = site_index[pair.site_i], site_index[pair.site_j]
        assert pair.j_eff == pytest.approx(matrix[a, b], abs=1e-12)
        assert matrix[a, b] == pytest.approx(matrix[b, a], abs=1e-12)


def test_pair_couplings_cover_exactly_the_nonzero_elements(lafeo3):
    """No bridged pair is missing, and no unbridged pair is invented."""
    pairs, matrix, site_index = _couplings(lafeo3)
    listed = {(site_index[p.site_i], site_index[p.site_j]) for p in pairs}
    nonzero = {
        (i, j)
        for i in range(matrix.shape[0])
        for j in range(i + 1, matrix.shape[0])
        if matrix[i, j] != 0.0
    }
    assert listed == nonzero


def test_pairs_are_unique_and_ordered(lafeo3):
    """One entry per pair, always with site_i < site_j."""
    pairs, _matrix, _site_index = _couplings(lafeo3)
    keys = [(p.site_i, p.site_j) for p in pairs]
    assert len(keys) == len(set(keys))
    assert all(i < j for i, j in keys)


def test_distance_matches_the_minimum_image_separation(lafeo3):
    """The law-of-cosines distance off the bridge equals the real M-M distance."""
    pairs, _matrix, _site_index = _couplings(lafeo3)
    fractional = lafeo3.fractional_coords
    for pair in pairs:
        delta = (
            fractional[pair.site_j] - fractional[pair.site_i] + 0.5
        ) % 1.0 - 0.5
        expected = float(np.linalg.norm(delta @ lafeo3.lattice))
        assert pair.distance == pytest.approx(expected, abs=1e-6)


def test_metal_symbols_follow_the_site_order(lafeo3):
    """metal_i/metal_j name the elements at site_i/site_j, not the bridge's order."""
    pairs, _matrix, _site_index = _couplings(lafeo3)
    symbols = lafeo3.element_symbols()
    for pair in pairs:
        assert pair.metal_i == symbols[pair.site_i]
        assert pair.metal_j == symbols[pair.site_j]


def test_bridge_provenance_is_recorded(lafeo3):
    """Ligands and angles are one per bridge, and bridge_count agrees."""
    pairs, _matrix, _site_index = _couplings(lafeo3)
    for pair in pairs:
        assert pair.bridge_count == len(pair.ligands) == len(pair.angles_deg)
        assert pair.bridge_count >= 1
        assert set(pair.ligands) == {"O"}
        assert all(0.0 <= angle <= 180.0 for angle in pair.angles_deg)


def test_cubic_lafeo3_nearest_neighbour_coupling_is_antiferromagnetic(lafeo3):
    """A 180-degree Fe(3+)-O-Fe(3+) bridge is AFM, which is J > 0 in this convention."""
    pairs, _matrix, _site_index = _couplings(lafeo3)
    assert all(pair.j_eff > 0.0 for pair in pairs)


def test_rock_salt_double_perovskite_couples_only_across_species(la2femno6):
    """Every B site's six bridged neighbours are B', so Fe-Mn is the only pairing.

    Fe-Fe and Mn-Mn are second neighbours in the rock-salt ordering, with no single
    ligand between them, so they are absent rather than small.
    """
    pairs, _matrix, _site_index = _couplings(la2femno6)
    pairings = {
        " - ".join(sorted((pair.metal_i, pair.metal_j))) for pair in pairs
    }
    assert pairings == {"Fe - Mn"}


def test_mixed_b_site_yields_several_element_pairings():
    """A disordered B site gives the plot more than one category to colour."""
    structure = generate_high_entropy_perovskite(
        "LaFeMnO3",
        a_sites=[("La", 1.0)],
        b_sites=[("Fe", 0.5), ("Mn", 0.5)],
        x_sites=[("O", 1.0)],
        a=4.0,
        n_cells_x=2,
        n_cells_y=2,
        n_cells_z=2,
        seed=0,
    )
    pairs, _matrix, _site_index = _couplings(structure)
    pairings = {
        " - ".join(sorted((pair.metal_i, pair.metal_j))) for pair in pairs
    }
    assert len(pairings) > 1
    assert pairings <= {"Fe - Fe", "Fe - Mn", "Mn - Mn"}


def test_pair_couplings_is_empty_without_bridges():
    """No bridges, no pairs -- and no exception."""
    assert pair_couplings([], default_params()) == []


@pytest.fixture(scope="module")
def lasr3mn4o12():
    """A composition charge balance forces into mixed valence: Mn averages +3.75.

    This is the double-exchange case named in ``de_active_pairs`` -- no in-window
    oxidation distribution makes Mn single-valent, so its bridges carry the term.
    """
    return generate_high_entropy_perovskite(
        "LaSr3Mn4O12",
        a_sites=[("La", 0.25), ("Sr", 0.75)],
        b_sites=[("Mn", 1.0)],
        x_sites=[("O", 1.0)],
        a=3.9,
        n_cells_x=2,
        n_cells_y=2,
        n_cells_z=2,
        seed=0,
    )


@pytest.fixture(scope="module")
def lamno3_jahn_teller():
    """LaMnO3 with the ab-plane oxygens shifted into a Jahn-Teller pattern.

    Mn(3+) is eg^1, but a perfectly cubic cage leaves the two eg orbitals
    degenerate, so no orbital is resolved and the occupied->empty channel stays
    shut. Alternating long and short Mn-O bonds in the plane resolve it -- the
    same thing relaxing the structure does, in one line and without CHGNet.
    """
    ideal = generate_single_perovskite(
        "LaMnO3",
        a_site="La",
        b_site="Mn",
        x_site="O",
        a=4.0,
        n_cells_x=2,
        n_cells_y=2,
        n_cells_z=2,
    )
    coords = np.array(ideal.cartesian_coords, dtype=float)
    for index, element in enumerate(ideal.element_symbols()):
        if element != "O":
            continue
        # A bridging oxygen sits half a cell along the axis it bridges; push it
        # off centre, in opposite senses on neighbouring bridges.
        cell = coords[index] / 4.0
        offsets = [abs(value - round(value)) > 0.25 for value in cell]
        if offsets[0]:
            coords[index][0] += 0.16 * (
                1 if int(round(cell[1] + cell[2])) % 2 == 0 else -1
            )
        elif offsets[1]:
            coords[index][1] += 0.16 * (
                1 if int(round(cell[0] + cell[2])) % 2 == 0 else -1
            )
    return ChemicalStructure(
        name="LaMnO3-jt",
        lattice=ideal.lattice,
        cartesian_coords=coords,
        atomic_labels=list(ideal.atomic_labels),
        magnetic_moments=np.array(ideal.magnetic_moments),
        is_periodic=True,
    )


def test_bridge_components_sum_to_the_bridge_coupling(lafeo3):
    """Splitting J into its channels does not change what J is."""
    bridges, params, _site_index = _bridges(lafeo3)
    assert bridges
    for bridge in bridges:
        assert sum(bridge_J_components(bridge, params)) == bridge_J(bridge, params)


def test_pair_components_sum_to_j_eff(lafeo3, lasr3mn4o12, lamno3_jahn_teller):
    """The three bars of a pair add up to its total bar, in every regime."""
    for structure in (lafeo3, lasr3mn4o12, lamno3_jahn_teller):
        pairs, _matrix, _site_index = _couplings(structure)
        assert pairs
        for pair in pairs:
            assert pair.j_se + pair.j_de + pair.j_oe == pytest.approx(
                pair.j_eff, abs=1e-12
            )


def test_half_filled_shells_are_pure_superexchange(lafeo3):
    """Fe(3+) d5: no mixed valence and no eg^1, so only the SE channel is open."""
    pairs, _matrix, _site_index = _couplings(lafeo3)
    assert pairs
    for pair in pairs:
        assert pair.j_de == 0.0
        assert pair.j_oe == 0.0
        assert pair.j_se == pair.j_eff


def test_forced_mixed_valence_opens_double_exchange(lasr3mn4o12):
    """Double exchange fires, and it is what turns a bridge ferromagnetic."""
    pairs, _matrix, _site_index = _couplings(lasr3mn4o12)
    de_pairs = [pair for pair in pairs if pair.j_de != 0.0]
    assert de_pairs
    assert all(pair.j_de < 0.0 for pair in de_pairs)
    # The AFM superexchange is still there; DE overwhelms it.
    flipped = [pair for pair in de_pairs if pair.j_se > 0.0 > pair.j_eff]
    assert flipped


def test_jahn_teller_distortion_opens_the_occupied_empty_channel(lamno3_jahn_teller):
    """Resolving the eg orbital is what switches the Kugel-Khomskii term on.

    The ideal cubic cell leaves it shut, which is why a distorted structure has to
    be found before this channel can be seen at all.
    """
    ideal_pairs, _matrix, _site_index = _couplings(
        generate_single_perovskite(
            "LaMnO3",
            a_site="La",
            b_site="Mn",
            x_site="O",
            a=4.0,
            n_cells_x=2,
            n_cells_y=2,
            n_cells_z=2,
        )
    )
    assert all(pair.j_oe == 0.0 for pair in ideal_pairs)

    pairs, _matrix, _site_index = _couplings(lamno3_jahn_teller)
    oe_pairs = [pair for pair in pairs if pair.j_oe != 0.0]
    assert oe_pairs
    assert all(pair.j_oe < 0.0 for pair in oe_pairs)
    assert all(pair.j_de == 0.0 for pair in pairs)


def _descriptors_for_states(structure, site_states):
    """Descriptors from a hand-built per-site oxidation array."""
    from types import SimpleNamespace

    return structure_ion_descriptors(
        structure, SimpleNamespace(site_oxidation_states=np.asarray(site_states))
    )


def test_oxidation_fractions_read_the_composition(lafeo3, lasr3mn4o12):
    """One entry per state an element is found in, weighted by how often."""
    single = _descriptors_for_states(lafeo3, _site_states(lafeo3))
    assert element_oxidation_fractions(single)["Fe"] == ((3, 1.0),)

    doped = _descriptors_for_states(lasr3mn4o12, _site_states(lasr3mn4o12))
    assert element_oxidation_fractions(doped)["Mn"] == ((3, 0.25), (4, 0.75))


def _site_states(structure):
    """The per-site oxidation array of the structure's best assignment."""
    ranked = enumerate_oxidation_states_by_energy(
        structure.element_symbols(), charge=0, max_mixing=2, top_k=5
    )
    assignment = expand_distribution_to_site_assignments(
        [distribution for distribution, _energy in ranked], structure
    )[0]
    return np.asarray(assignment.site_oxidation_states)


def test_couplings_do_not_depend_on_which_site_got_which_state(lasr3mn4o12):
    """The whole point: no site carries a valence, only the element does.

    Which Mn site the expander labels 3+ and which 4+ is arbitrary -- it falls back
    to a sequential slice of the site list when no Wyckoff ordering fits. Rotating
    those labels among the Mn sites must leave every coupling untouched.
    """
    states = _site_states(lasr3mn4o12)
    metals = [
        index
        for index, symbol in enumerate(lasr3mn4o12.element_symbols())
        if symbol == "Mn"
    ]
    rotated = states.copy()
    rotated[metals] = np.roll(states[metals], 3)
    assert not np.array_equal(states[metals], rotated[metals])

    params = default_params()
    matrices = []
    for array in (states, rotated):
        descriptors = _descriptors_for_states(lasr3mn4o12, array)
        site_index = {site: i for i, site in enumerate(sorted(descriptors))}
        matrices.append(
            build_Jeff_matrix(build_bridges(lasr3mn4o12, descriptors), site_index, params)
        )
    assert np.array_equal(matrices[0], matrices[1])


def test_double_exchange_is_weighted_by_the_mixed_valence_fraction(lasr3mn4o12):
    """DE rides only the combinations whose two ends differ in valence."""
    bridges, params, _site_index = _bridges(lasr3mn4o12)
    de_bridges = [bridge for bridge in bridges if bridge.de_active]
    assert de_bridges
    # Mn at 25% 3+ / 75% 4+: 1 - (0.25^2 + 0.75^2).
    for bridge in de_bridges:
        assert bridge.de_weight == pytest.approx(0.375)
    # Every Mn-Mn bridge of this cell is geometrically identical, and now nothing
    # else distinguishes them either.
    values = {round(bridge_J_components(bridge, params)[1], 12) for bridge in de_bridges}
    assert len(values) == 1


@pytest.fixture(scope="module")
def lasr3mn4o12_jahn_teller(lasr3mn4o12):
    """The doped manganite with the same ab-plane oxygen shifts as LaMnO3.

    Mn is both mixed-valent and orbitally resolved here, so a bridge end holds an
    eg^1 state (Mn3+, 25%) beside an eg^0 one (Mn4+, 75%) -- the only case where
    the occupied->empty donor is a strict subset of what polarizes the ligand.
    """
    coords = np.array(lasr3mn4o12.cartesian_coords, dtype=float)
    for index, element in enumerate(lasr3mn4o12.element_symbols()):
        if element != "O":
            continue
        cell = coords[index] / 3.9
        offsets = [abs(value - round(value)) > 0.25 for value in cell]
        if offsets[0]:
            coords[index][0] += 0.16 * (
                1 if int(round(cell[1] + cell[2])) % 2 == 0 else -1
            )
        elif offsets[1]:
            coords[index][1] += 0.16 * (
                1 if int(round(cell[0] + cell[2])) % 2 == 0 else -1
            )
    return ChemicalStructure(
        name="LaSr3Mn4O12-jt",
        lattice=lasr3mn4o12.lattice,
        cartesian_coords=coords,
        atomic_labels=list(lasr3mn4o12.atomic_labels),
        magnetic_moments=np.array(lasr3mn4o12.magnetic_moments),
        is_periodic=True,
    )


def test_every_state_donates_and_mn4_accepts_in_the_occupied_empty_channel(
    lasr3mn4o12_jahn_teller,
):
    """Mixed Mn3+/Mn4+: every occupied orbital donates (the donor is J_SE's factor),
    and Mn4+'s empty eg joins Mn3+'s CF-empty eg orbital as the acceptor."""
    bridges, params, _site_index = _bridges(lasr3mn4o12_jahn_teller)
    oe_bridges = [bridge for bridge in bridges if bridge.oe_active]
    assert oe_bridges
    for bridge in oe_bridges:
        assert bridge.Bsig_occ_i is None and bridge.Bpi_occ_i is None
        assert np.abs(bridge.Bsig_emp_i).sum() + np.abs(bridge.Bsig_emp_j).sum() > 0.0
    assert all(bridge_J_components(b, params)[2] < 0.0 for b in oe_bridges)


def _two_cr_bridge(angle_deg):
    """The one bridge through the shared O of ``_two_cr_structure``."""
    from quick_mag.ion_descriptors import ion_descriptor
    from quick_mag.polarization_model import build_bridges

    st = _two_cr_structure(angle_deg)
    desc = {1: ion_descriptor("Cr", 3, "HS", "oct"), 2: ion_descriptor("Cr", 3, "HS", "oct")}
    return [b for b in build_bridges(st, desc, valence="site") if b.ligand_index == 0]


def _two_cr_structure(angle_deg):
    """Two Cr3+ octahedra sharing one O, M-O-M angle ``angle_deg`` (bond 2.0 A)."""
    r = 2.0
    t = np.radians(angle_deg)
    o = np.zeros(3)
    m1 = np.array([r, 0.0, 0.0])
    m2 = r * np.array([np.cos(t), np.sin(t), 0.0])
    coords = [o, m1, m2]
    labels = ["O", "Cr", "Cr"]
    for m in (m1, m2):           # complete each octahedron around its own bond axis
        u = (o - m) / r
        a = np.array([0.0, 0.0, 1.0])
        v = np.cross(u, a); v /= np.linalg.norm(v)
        w = np.cross(u, v)
        for d in (-u, v, -v, w, -w):
            coords.append(m + r * d)
            labels.append("O")
    return ChemicalStructure(name=f"Cr2-{angle_deg}", lattice=np.eye(3) * 30.0,
                             cartesian_coords=np.array(coords), atomic_labels=labels,
                             magnetic_moments=np.zeros((len(labels), 3)), is_periodic=False)


def test_cr3_kanamori_channel_is_ferromagnetic_at_90_and_off_at_180():
    """d3: the t2g -> empty-eg hop needs one shared O p orbital that is pi to one Cr and
    sigma to the other, which exists at 90 degrees and not at 180."""
    params = default_params()
    (b90,) = _two_cr_bridge(90.0)
    (b180,) = _two_cr_bridge(180.0)
    assert b90.oe_active and b180.oe_active
    assert bridge_J_components(b90, params)[2] < 0.0
    assert abs(bridge_J_components(b180, params)[2]) < 1e-12


def test_direct_exchange_couples_close_cr_pairs_and_enters_j_eff():
    """G > 0 for two Cr3+ sharing an edge (2.83 A), 0 once they are beyond the cutoff,
    and pair_couplings(model=) adds k_dd * G to the pair so j_eff is the matrix element."""
    from dataclasses import replace

    from quick_mag.direct_exchange import dd_geometry
    from quick_mag.ion_descriptors import ion_descriptor

    (b90,) = _two_cr_bridge(90.0)
    desc = {1: ion_descriptor("Cr", 3, "HS", "oct"), 2: ion_descriptor("Cr", 3, "HS", "oct")}
    st = _two_cr_structure(90.0)
    G = dd_geometry(st, desc, [1, 2], cutoff=3.6)
    assert G[0, 1] > 0.0 and G[0, 1] == G[1, 0] and G[0, 0] == 0.0
    assert dd_geometry(st, desc, [1, 2], cutoff=2.5)[0, 1] == 0.0

    from types import SimpleNamespace

    params = replace(default_params(), k_dd=0.2)
    model = SimpleNamespace(dd=G, site_index={1: 0, 2: 1})
    (with_dd,) = pair_couplings([b90], params, model=model)
    (bare,) = pair_couplings([b90], params)
    assert with_dd.j_dd == pytest.approx(0.2 * G[0, 1])
    assert with_dd.j_eff == pytest.approx(bare.j_eff + with_dd.j_dd)
