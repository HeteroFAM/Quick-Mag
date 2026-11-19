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


def test_the_eg3_hole_sits_in_x2_y2_of_an_elongated_octahedron():
    """Cu2+ with its long bonds normal to the Cu-O-Cu line: the hole's x2-y2 points at the shared O with weight
    3/4 instead of the shell average 1/2, so the 180 degree coupling is (3/4 / 1/2)^2 = 2.25 times larger."""
    flat = bridge_J_components(_pair("Cu", 2, 180.0, r=1.95), COHERENT)[0]
    jt = bridge_J_components(_pair("Cu", 2, 180.0, r=1.95, long=2.40), COHERENT)[0]
    assert jt / flat == pytest.approx(2.25, rel=1e-6)
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
