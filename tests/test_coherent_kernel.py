"""The coherent bridge kernel tr(P_i P_j), the empty-empty term and the eg^3 hole orbital."""

from dataclasses import replace

import numpy as np
import pytest

from quick_mag.fitting import fit_system, j_eff_matrix
from quick_mag.ion_descriptors import ion_descriptor
from quick_mag.polarization_model import (
    bridge_J_components,
    build_bridges,
    build_Jeff_matrix,
    default_params,
    exchange_model,
    params_from_dict,
)
from quick_mag.structure import ChemicalStructure
import quick_mag.polarization_model as PM


def _pair(element, charge, angle_deg, **kwargs):
    """The one bridge through the shared O of ``_pair_structure``."""
    st, desc = _pair_structure(element, charge, angle_deg, **kwargs)
    (bridge,) = [b for b in build_bridges(st, desc, valence="site") if b.ligand_index == 0]
    return bridge


def _pair_structure(element, charge, angle_deg, *, r=2.0, long=None, rotation=None, twist=0.0):
    """Two octahedra of ``element`` sharing one O at M-O-M angle ``angle_deg``. ``long``: length of the two
    bonds of each octahedron normal to the M-O-M plane (a Jahn-Teller elongation); ``twist``: each octahedron
    turned by this angle (degrees) about an axis off its bonds, so no bond lies along a local d axis."""
    t = np.radians(angle_deg)
    o = np.zeros(3)
    metals = [r * np.array([1.0, 0.0, 0.0]), r * np.array([np.cos(t), np.sin(t), 0.0])]
    coords, labels = [o, *metals], ["O", element, element]
    for m in metals:
        u = (o - m) / r
        v = np.cross(u, [0.0, 0.0, 1.0]); v /= np.linalg.norm(v)
        w = np.cross(u, v)
        for d, length in ((-u, r), (v, r), (-v, r), (w, long or r), (-w, long or r)):
            coords.append(m + length * d)
            labels.append("O")
    coords = np.array(coords)
    if twist:
        # shear the outer ligands of each octahedron (a trigonal-like distortion): bonds leave the local axes
        k = np.radians(twist)
        for n, m in enumerate(metals):
            for q in range(5):
                idx = 3 + 5 * n + q
                d = coords[idx] - m
                axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
                d = d * np.cos(k) + np.cross(axis, d) * np.sin(k) + axis * (axis @ d) * (1 - np.cos(k))
                coords[idx] = m + d
    keep = [k for k in range(len(coords)) if all(np.linalg.norm(coords[k] - coords[m]) > 1e-6 for m in range(k))]
    coords, labels = coords[keep], [labels[k] for k in keep]      # 90 degrees: the octahedra share an edge, one O each
    if rotation is not None:
        coords = coords @ np.asarray(rotation).T
    st = ChemicalStructure(name=f"{element}2-{angle_deg}", lattice=np.eye(3) * 40.0, cartesian_coords=coords,
                           atomic_labels=labels, magnetic_moments=np.zeros((len(labels), 3)), is_periodic=False)
    desc = {1: ion_descriptor(element, charge, "HS", "oct"), 2: ion_descriptor(element, charge, "HS", "oct")}
    return st, desc


def _random_rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


COHERENT = replace(default_params(), kernel="coherent")
DIAGONAL = replace(default_params(), kernel="diagonal")


@pytest.mark.parametrize("angle", [180.0, 150.0, 120.0, 90.0])
def test_kernels_agree_for_a_half_filled_shell(angle):
    """d5: sum over the whole d shell, so the sigma-pi cross term cancels at every geometry."""
    b = _pair("Fe", 3, angle, twist=12.0)
    assert bridge_J_components(b, COHERENT)[0] == pytest.approx(bridge_J_components(b, DIAGONAL)[0], rel=1e-9)


def test_kernels_agree_when_bonds_lie_on_the_local_axes():
    b = _pair("Cr", 3, 130.0)
    assert bridge_J_components(b, COHERENT)[0] == pytest.approx(bridge_J_components(b, DIAGONAL)[0], rel=1e-9)


def test_sigma_pi_interference_moves_a_t2g_ion_off_its_axes():
    b = _pair("Cr", 3, 130.0, twist=12.0)
    coherent, diagonal = bridge_J_components(b, COHERENT)[0], bridge_J_components(b, DIAGONAL)[0]
    assert abs(coherent - diagonal) > 0.02 * abs(diagonal)


@pytest.mark.parametrize("angle", [180.0, 137.0])
def test_the_coherent_kernel_is_rotation_invariant(angle):
    """No bridge frame: a rigid rotation of the structure changes nothing, linear bridges included."""
    params = replace(COHERENT, k_ee=0.3)
    ref = bridge_J_components(_pair("Cr", 3, angle, twist=9.0), params)
    for seed in (1, 2, 3):
        rot = bridge_J_components(_pair("Cr", 3, angle, twist=9.0, rotation=_random_rotation(seed)), params)
        assert rot == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_empty_empty_exchange_is_afm_at_180_and_gone_at_90():
    """Two Cr3+ (empty eg) share the O p_sigma orbital at 180 degrees and use orthogonal ones at 90."""
    on, off = replace(COHERENT, k_ee=0.3), COHERENT
    d180 = bridge_J_components(_pair("Cr", 3, 180.0), on)[0] - bridge_J_components(_pair("Cr", 3, 180.0), off)[0]
    d90 = bridge_J_components(_pair("Cr", 3, 90.0), on)[0] - bridge_J_components(_pair("Cr", 3, 90.0), off)[0]
    assert d180 > 0.0
    assert abs(d90) < 1e-9 * d180
    d135 = bridge_J_components(_pair("Cr", 3, 135.0), on)[0] - bridge_J_components(_pair("Cr", 3, 135.0), off)[0]
    assert d135 == pytest.approx(0.5 * d180, rel=1e-6)          # cos^2(theta)


def test_empty_empty_exchange_needs_unpaired_electrons_and_empty_orbitals():
    on = replace(COHERENT, k_ee=0.3)
    fe = _pair("Fe", 3, 180.0)       # d5: nothing empty
    assert bridge_J_components(fe, on)[0] == pytest.approx(bridge_J_components(fe, COHERENT)[0], rel=1e-12)
    ti = _pair("Ti", 4, 180.0)       # d0: everything empty, no spin to split it
    assert bridge_J_components(ti, on)[0] == 0.0


def test_the_eg3_hole_sits_in_x2_y2_of_an_elongated_octahedron(monkeypatch):
    """Cu2+ with its long bonds normal to the Cu-O-Cu line: the hole's x2-y2 points at the shared O with weight
    3/4 instead of the shell average 1/2, so the 180 degree coupling is (3/4 / 1/2)^2 = 2.25 times larger.
    A two-octahedron cluster has no lattice of counter-ions, so the orbital order comes from its cage alone here."""
    import quick_mag.polarization_model as PM

    monkeypatch.setattr(PM, "CF_FIELD_CUTOFF", 2.6)
    flat = bridge_J_components(_pair("Cu", 2, 180.0, r=1.95), COHERENT)[0]
    jt = bridge_J_components(_pair("Cu", 2, 180.0, r=1.95, long=2.40), COHERENT)[0]
    # the elongation also weakens the cubic field's octahedral pattern: 3.3 % tetrahedral ordering
    # (``tetrahedral_fraction``), which spreads the hole over t2 and pulls the ratio below 2.25
    from quick_mag.polarization_model import tetrahedral_fraction

    st, _ = _pair_structure("Cu", 2, 180.0, r=1.95, long=2.40)
    assert tetrahedral_fraction(st, 1) == pytest.approx(0.0327, abs=1e-3)
    assert 2.1 < jt / flat < 2.25
    b = _pair("Cu", 2, 180.0, r=1.95, long=2.40)
    assert not b.oe_active                                       # a hole has no empty orbital to accept into


def test_the_fit_vectorization_follows_the_kernel_and_k_ee():
    st, desc = _pair_structure("Cr", 3, 140.0, twist=10.0)
    for params in (replace(COHERENT, k_ee=0.4), replace(DIAGONAL, k_ee=0.4), COHERENT):
        model = exchange_model(st, None, params=params, descriptors=desc, valence="site")
        system = fit_system(model, np.array([[1, 1], [1, -1]]), np.zeros(2))
        index = {s: k for k, s in enumerate(model.sites)}
        reference = build_Jeff_matrix(model.bridges, index, params) + params.k_dd * model.dd
        assert np.allclose(j_eff_matrix(system, params), reference, rtol=1e-10, atol=1e-14)


def test_parameter_files_without_a_kernel_keep_the_one_they_were_fitted_with():
    assert params_from_dict({"w_ligand": {"O": 0.01}}).kernel == "diagonal"
    assert params_from_dict({"kernel": "coherent", "k_ee": 0.2}).k_ee == 0.2
    with pytest.raises(ValueError):
        bridge_J_components(_pair("Fe", 3, 180.0), replace(default_params(), kernel="frame"))


def test_the_acceptor_rule_needs_no_parameter_on_the_donor():
    """Cr3+ (empty eg) next to Fe3+ (nothing empty) at 90 degrees: FM through Cr's coefficient alone, and the
    vectorized form agrees."""
    st, desc = _pair_structure("Cr", 3, 90.0)
    desc = {1: desc[1], 2: ion_descriptor("Fe", 3, "HS", "oct")}
    labels = list(st.atomic_labels); labels[2] = "Fe"
    st = ChemicalStructure(name="CrFe", lattice=st.lattice, cartesian_coords=st.cartesian_coords, atomic_labels=labels,
                           magnetic_moments=np.zeros((len(labels), 3)), is_periodic=False)
    params = replace(COHERENT, oe_rule="acceptor", j_fm={"Cr": 0.5}, k_oe=0.0, ee_spin="n", t2g_oe="off")
    (b,) = [x for x in build_bridges(st, desc, valence="site") if x.ligand_index == 0]
    assert bridge_J_components(b, params)[2] < 0.0
    assert bridge_J_components(b, replace(params, j_fm={"Fe": 0.5}))[2] == 0.0
    by_spin = bridge_J_components(b, replace(params, j_fm={}, k_oe=0.5 / 3.0))[2]      # Cr3+: three unpaired electrons (ee_spin "n")
    assert by_spin == pytest.approx(bridge_J_components(b, params)[2], rel=1e-9)
    model = exchange_model(st, None, params=params, descriptors=desc, valence="site")
    system = fit_system(model, np.array([[1, 1], [1, -1]]), np.zeros(2))
    index = {s: k for k, s in enumerate(model.sites)}
    assert np.allclose(j_eff_matrix(system, params), build_Jeff_matrix(model.bridges, index, params) + params.k_dd * model.dd,
                       rtol=1e-10, atol=1e-14)


def test_direct_interference_only_inside_the_cutoff_and_vectorizes():
    far = _pair("Cr", 3, 180.0)
    assert far.G_dx is None
    st, desc = _pair_structure("Cr", 3, 92.0, twist=8.0)        # 2.88 A apart
    params = replace(COHERENT, k_dx=-0.7, k_ee=0.2)
    (near,) = [x for x in build_bridges(st, desc, valence="site") if x.ligand_index == 0]
    assert near.G_dx is not None and np.abs(near.G_dx).sum() > 0
    assert bridge_J_components(near, params)[0] != bridge_J_components(near, replace(params, k_dx=0.0))[0]
    model = exchange_model(st, None, params=params, descriptors=desc, valence="site")
    system = fit_system(model, np.array([[1, 1], [1, -1]]), np.zeros(2))
    index = {s: k for k, s in enumerate(model.sites)}
    assert np.allclose(j_eff_matrix(system, params), build_Jeff_matrix(model.bridges, index, params) + params.k_dd * model.dd,
                       rtol=1e-10, atol=1e-14)


def test_the_ionization_rule_ranks_donors_by_their_next_ionization_energy():
    """Into Cr3+'s empty eg at 90 degrees: Mn2+ and Co2+ donate, Ni2+ less, Fe3+ hardly; and it vectorizes."""
    params = replace(COHERENT, oe_rule="ionization", k_oe=1.0, oe_decay=1.5, oe_charge=15.0, j_fm={})
    out = {}
    for el, q in (("Mn", 2), ("Co", 2), ("Ni", 2), ("Fe", 3)):
        st, desc = _pair_structure("Cr", 3, 90.0)
        labels = list(st.atomic_labels); labels[2] = el
        st = ChemicalStructure(name="x", lattice=st.lattice, cartesian_coords=st.cartesian_coords, atomic_labels=labels,
                               magnetic_moments=np.zeros((len(labels), 3)), is_periodic=False)
        desc = {1: desc[1], 2: ion_descriptor(el, q, "HS", "oct")}
        (b,) = [x for x in build_bridges(st, desc, valence="site") if x.ligand_index == 0]
        t = float(np.sum(b.T_j[0][:2].sum(0) * b.T_i[1][:2].sum(0)))   # donor (end j) x Cr acceptor geometry, unit amplitudes
        assert b.oe_active and t > 0
        from quick_mag.polarization_model import oe_transfer_energy
        out[el] = float(oe_transfer_energy(params, b.ie_j, b.ie_i))
        assert bridge_J_components(b, params)[2] < 0.0
    assert out["Co"] < out["Ni"] < out["Fe"] and out["Mn"] < out["Ni"]
    model = exchange_model(st, None, params=params, descriptors=desc, valence="site")
    system = fit_system(model, np.array([[1, 1], [1, -1]]), np.zeros(2))
    index = {s: k for k, s in enumerate(model.sites)}
    assert np.allclose(j_eff_matrix(system, params), build_Jeff_matrix(model.bridges, index, params) + params.k_dd * model.dd,
                       rtol=1e-10, atol=1e-14)


def test_linear_double_exchange_goes_as_cos_theta_and_the_sigma_amplitudes():
    """Two mixed-valent Mn (3+/4+) through one O: the derived form falls as |cos theta|, the original as cos^2,
    both vanish at 90 degrees, and the fit's vectorization follows the form."""
    out = {}
    for angle in (180.0, 120.0, 90.0):
        st, _ = _pair_structure("Mn", 3, angle)
        desc = {1: ion_descriptor("Mn", 3, "HS", "oct"), 2: ion_descriptor("Mn", 4, "HS", "oct")}
        (b,) = [x for x in build_bridges(st, desc, de_pairs=frozenset({frozenset({"Mn"})}), valence="class") if x.ligand_index == 0]
        assert b.de_active and b.de_weight > 0
        for form in ("linear", "quadratic"):
            params = replace(COHERENT, de_form=form, t_de={"Mn": 0.3})
            out[(form, angle)] = bridge_J_components(b, params)[1]
            if angle == 120.0:
                model = exchange_model(st, None, params=params, descriptors=desc, valence="class")
                system = fit_system(model, np.array([[1, 1], [1, -1]]), np.zeros(2))
                index = {s: k for k, s in enumerate(model.sites)}
                ref = build_Jeff_matrix(model.bridges, index, params) + params.k_dd * model.dd
                if any(x.de_active for x in model.bridges):
                    assert np.allclose(j_eff_matrix(system, params), ref, rtol=1e-10, atol=1e-14)
    assert out[("linear", 180.0)] < 0 and out[("quadratic", 180.0)] < 0
    assert out[("linear", 120.0)] / out[("linear", 180.0)] == pytest.approx(0.5, rel=1e-6)
    assert out[("quadratic", 120.0)] / out[("quadratic", 180.0)] == pytest.approx(0.25, rel=1e-6)
    assert abs(out[("linear", 90.0)]) < 1e-9 and abs(out[("quadratic", 90.0)]) < 1e-9


def test_the_crystal_field_gives_the_levels_and_the_jahn_teller_orbital_without_axes():
    """Ideal cage: t2g / eg exactly, no eg splitting. Elongated along the bonds normal to the M-O-M plane (z here):
    the electron's orbital is dz2, the hole's dx2-y2. Twisted: still three + two orthonormal orbitals."""
    from quick_mag.polarization_model import crystal_field_basis

    st, _ = _pair_structure("Mn", 3, 180.0)
    basis, split = crystal_field_basis(st, 1, "oct")
    assert split == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(basis[:3, 3:], 0.0, atol=1e-10) and np.allclose(basis[3:, :3], 0.0, atol=1e-10)
    st, _ = _pair_structure("Mn", 3, 180.0, long=2.2)
    basis, split = crystal_field_basis(st, 1, "oct")
    assert split > 1e-3
    assert np.allclose(np.abs(basis[3]), [0, 0, 0, 1, 0], atol=1e-8)      # lower eg orbital: dz2, along the long bonds
    assert np.allclose(np.abs(basis[4]), [0, 0, 0, 0, 1], atol=1e-8)
    st, _ = _pair_structure("Mn", 3, 140.0, long=2.2, twist=9.0)
    basis, _split = crystal_field_basis(st, 1, "oct")
    assert np.allclose(basis @ basis.T, np.eye(5), atol=1e-10)


@pytest.mark.parametrize("element, charge", [("Mn", 3), ("Cu", 2), ("Ni", 2), ("V", 3)])
@pytest.mark.parametrize("seed", [0, 1])
def test_orbital_ordered_and_partly_filled_ions_do_not_depend_on_orientation(element, charge, seed):
    """An eg^1 electron, an eg^3 hole and unequal t2g / eg occupancies on distorted, twisted cages: every term is
    the same after the whole structure is turned (the orbitals come from the cage, not from axes)."""
    params = replace(COHERENT, k_ee=0.3, j_fm={element: 0.2})
    ref = bridge_J_components(_pair(element, charge, 140.0, long=2.2, twist=9.0), params)
    rot = bridge_J_components(_pair(element, charge, 140.0, long=2.2, twist=9.0, rotation=_random_rotation(seed)), params)
    assert rot == pytest.approx(ref, rel=1e-8, abs=1e-12)


def test_the_bounded_range_exponent_stays_between_its_limits_and_vectorizes():
    """n = n_min + (n_max - n_min) Delta^2 / (Delta^2 + delta_sat^2): 2 for a covalent ion, 8 for an ionic one,
    halfway at Delta = delta_sat, and the fitting module evaluates the same number."""
    from quick_mag.amplitude_head import DELTA_REF
    from quick_mag.polarization_model import range_exponent

    params = replace(COHERENT, delta_sat=6.0, n_min=2.0, n_max=8.0, n_delta=3.0)
    n = range_exponent(params, "Fe", np.log(np.array([1e-3, 6.0, 1e5]) / DELTA_REF))
    assert n == pytest.approx([2.0, 5.0, 8.0], abs=1e-4)
    st, desc = _pair_structure("Cr", 3, 150.0, twist=7.0)
    model = exchange_model(st, None, params=params, descriptors=desc, valence="site")
    system = fit_system(model, np.array([[1, 1], [1, -1]]), np.zeros(2))
    index = {s: k for k, s in enumerate(model.sites)}
    assert np.allclose(j_eff_matrix(system, params), build_Jeff_matrix(model.bridges, index, params) + params.k_dd * model.dd,
                       rtol=1e-9, atol=1e-12)
    assert not np.allclose(j_eff_matrix(system, params), j_eff_matrix(system, replace(params, delta_sat=0.0)))


def test_files_of_the_removed_range_laws_are_refused():
    from quick_mag.polarization_model import params_from_dict

    for model in ("exp", "sto", "sto_abs"):
        with pytest.raises(ValueError, match="power"):
            params_from_dict({"range_model": model, "gamma_pi": 0.4})
    assert params_from_dict({"range_model": "power", "gamma_pi": 0.4}).gamma_pi == 0.4


@pytest.mark.parametrize("rule", ["n", "n2", "n_delta"])
@pytest.mark.parametrize("t2g_oe", ["off", "signed", "kk"])
def test_spin_weight_rules_and_the_t2g_acceptor_vectorize(rule, t2g_oe):
    """Ti3+ / V3+ pairs (empty t2g): every ee_spin rule and t2g acceptor form gives the same couplings bridge by bridge
    and vectorized; the t2g channel is FM, and ~0 for an ion with no empty t2g (Cr3+)."""
    params = replace(COHERENT, k_ee=0.2, ee_spin=rule, t2g_oe=t2g_oe)
    for element in ("Ti", "V", "Cr"):
        st, desc = _pair_structure(element, 3, 150.0, long=2.1, twist=6.0)
        model = exchange_model(st, None, params=params, descriptors=desc, valence="site")
        system = fit_system(model, np.array([[1, 1], [1, -1]]), np.zeros(2))
        index = {s: k for k, s in enumerate(model.sites)}
        assert np.allclose(j_eff_matrix(system, params), build_Jeff_matrix(model.bridges, index, params) + params.k_dd * model.dd,
                           rtol=1e-10, atol=1e-14)
        if t2g_oe != "off":
            base = replace(params, t2g_oe="off")
            d = [bridge_J_components(b, params)[2] - bridge_J_components(b, base)[2] for b in model.bridges]
            if element == "Cr":                # t2g^3: no empty t2g (only the cage's small tetrahedral admixture)
                assert np.all(np.abs(d) < 1e-4)
            else:
                assert all(x < 0.0 for x in d)


def test_the_n_delta_spin_weight_divides_by_the_charge_transfer_energy():
    from quick_mag.amplitude_head import DELTA, ln_delta
    from quick_mag.polarization_model import ee_spin_weights

    w = ee_spin_weights(ion_descriptor("V", 3, "HS", "oct"), 2.0)
    assert w["n"] == 2.0 and w["n2"] == 4.0
    assert w["n_delta"] == pytest.approx(2.0 * np.exp(-ln_delta("V", 3)))       # Delta with its smooth floor
    assert w["n_delta"] == pytest.approx(2.0 * 4.0 / DELTA[("V", 3)], rel=1e-3)


def test_empty_eg_above_an_empty_t2g_costs_ten_dq():
    """Ti3+ / V3+: their lowest empty level is t2g, so an empty eg orbital costs Delta + 10Dq; Cr3+ (t2g full-half)
    and Mn3+ keep 1 on every slot."""
    from quick_mag.amplitude_head import DELTA_REF, ln_delta
    from quick_mag.polarization_model import EG_INDICES, TEN_DQ, empty_level_scale

    for el, q in (("Ti", 3), ("V", 3)):
        d = DELTA_REF * np.exp(ln_delta(el, q))
        sc = empty_level_scale(ion_descriptor(el, q, "HS", "oct"))
        assert np.allclose(sc[list(EG_INDICES)], d / (d + TEN_DQ["oct"])) and np.allclose(np.delete(sc, list(EG_INDICES)), 1.0)
    for el, q in (("Cr", 3), ("Mn", 3), ("Mn", 4), ("Fe", 3)):
        assert np.allclose(empty_level_scale(ion_descriptor(el, q, "HS", "oct")), 1.0)
