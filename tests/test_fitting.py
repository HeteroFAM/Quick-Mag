"""quick_mag.fitting: the vectorized model equals the reference one, and fits recover parameters."""

import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.fitting import (  # noqa: E402
    ParamSpec,
    diagnose,
    fit,
    fit_system,
    identifiability,
    j_eff_matrix,
    leave_one_group_out,
    predict,
)
from quick_mag.polarization_model import default_params, exchange_model  # noqa: E402
from quick_mag.site_states import ranked_assignments  # noqa: E402

from test_site_states import _base, co3o4, lamno3, magnetite  # noqa: E402


def _system(make, n=12, seed=0, params=None, **kw):
    st = make()
    (a,) = ranked_assignments(st, base_constraints=_base(st))
    model = exchange_model(st, a, params=params, **kw)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(model.sites)))
    energies = np.array([model.energy(s) for s in signs])
    return model, fit_system(model, signs, energies, name=st.name)


@pytest.mark.parametrize("make", [lamno3, magnetite, co3o4])
def test_vectorized_couplings_are_the_reference_ones(make):
    model, system = _system(make)
    assert np.allclose(j_eff_matrix(system, default_params()), model.j_eff, rtol=1e-12, atol=1e-14)
    assert np.allclose(predict(system, default_params()), system.energies, rtol=1e-12, atol=1e-14)


def test_site_valence_systems_match_too():
    model, system = _system(magnetite, valence="site")
    assert np.allclose(j_eff_matrix(system, default_params()), model.j_eff, rtol=1e-12, atol=1e-14)


def test_a_fit_recovers_the_parameters_that_made_the_data():
    true = default_params()
    true = replace(true, w_ligand={**true.w_ligand, "O": 0.04}, jh_ligand={**true.jh_ligand, "O": 0.012},
                   head_sigma={**true.head_sigma, "ln_cn": -1.5})
    systems = [_system(magnetite, params=true, seed=1)[1], _system(lamno3, params=true, seed=2)[1],
               _system(co3o4, params=true, seed=5)[1]]
    spec = ParamSpec(free=["w[O]", "jh[O]", "head_sigma[ln_cn]"], systems=systems)
    res = fit(systems, spec, reg_weight=1e-10)
    got = dict(zip(spec.names, res.x))
    assert got["w[O]"] == pytest.approx(0.04, rel=1e-3)
    assert got["jh[O]"] == pytest.approx(0.012, rel=1e-3)
    assert got["head_sigma[ln_cn]"] == pytest.approx(-1.5, rel=1e-3)
    assert diagnose(systems, res.params)["rmse_meV_per_site"] < 1e-3
    ident = identifiability(systems, spec, res.x)
    assert np.isfinite(ident["condition"]) and set(ident["stderr"]) == set(spec.names)


def test_gauges_cannot_be_freed():
    systems = [_system(lamno3)[1]]
    with pytest.raises(ValueError, match="unknown parameter kind"):
        ParamSpec(free=["alpha[O]"], systems=systems)           # removed with the exponential range law
    with pytest.raises(ValueError, match="unknown parameter kind"):
        ParamSpec(free=["kappa[Mn]"], systems=systems)          # removed: the scale is w's


def test_legacy_kappa_files_are_refused():
    from quick_mag.polarization_model import params_from_dict

    with pytest.raises(ValueError, match="no longer"):
        params_from_dict({"kappa": {"Fe": 0.15}, "gamma_pi": 0.4})


def test_leave_one_group_out_runs():
    systems = [_system(magnetite, seed=s)[1] for s in (1, 2)]
    systems[1] = replace(systems[1], group="other")
    out = leave_one_group_out(systems, ["w[O]"])
    assert out["all"] < 1e-3


# ---- range models --------------------------------------------------------------------


@pytest.mark.parametrize("law", [dict(delta_sat=0.0, n_delta=2.0, range_power={"Fe": 6.0}), dict(delta_sat=5.0)])
@pytest.mark.parametrize("make", [magnetite, co3o4])
def test_range_laws_vectorize_too(make, law):
    p = replace(default_params(), **law)
    model, system = _system(make, params=p)
    assert np.allclose(j_eff_matrix(system, p), model.j_eff, rtol=1e-12, atol=1e-14)


def test_sto_damping_is_one_at_r0_and_steeper_for_pi():
    from quick_mag.radial_overlap import pd_damping, slater_zeta

    nd, zd = slater_zeta("Fe", 3, "d")
    np_, zp = slater_zeta("O", -2, "p")
    ds, dp = pd_damping(nd, zd, np_, zp, [1.9, 2.0, 2.1], 2.0)
    assert ds[1] == pytest.approx(1.0) and dp[1] == pytest.approx(1.0)
    assert ds[0] > 1 > ds[2] and dp[0] / dp[2] > ds[0] / ds[2]


def test_a_shared_range_power_is_one_parameter():
    p = replace(default_params(), delta_sat=0.0)
    systems = [_system(magnetite, params=p)[1], _system(lamno3, params=p)[1]]
    spec = ParamSpec(free=["w[O]", "n[*]"], systems=systems)
    assert spec.names.count("n[*]") == 1
    assert spec.unpack(spec.pack(p)).range_power == pytest.approx({m: p.range_power_default for m in spec.metals})


# ---- amplitude head ------------------------------------------------------------------


def _headed():
    return replace(default_params(), head_sigma={"charge": 0.3, "ln_cn": 0.5}, head_pi={"spin": 0.1})


@pytest.mark.parametrize("make", [magnetite, co3o4])
def test_the_head_vectorizes(make):
    p = _headed()
    model, system = _system(make, params=p)
    assert np.allclose(j_eff_matrix(system, p), model.j_eff, rtol=1e-12, atol=1e-14)
    assert not np.allclose(model.j_eff, _system(make)[0].j_eff)


def test_a_fit_recovers_head_weights():
    true = _headed()
    start = replace(true, head_sigma={"charge": 0.0, "ln_cn": 0.0})
    systems = [_system(magnetite, params=true, seed=3)[1], _system(co3o4, params=true, seed=4)[1]]
    spec = ParamSpec(free=["head_sigma[charge]", "head_sigma[ln_cn]"], systems=systems, base=start)
    # tight tolerances and a negligible prior: charge and ln_cn are nearly collinear on these two cells, so the
    # direction is shallow (gtol stops early) and even a 1e-10 prior biases it by ~0.2 %
    res = fit(systems, spec, reg_weight=1e-16, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    got = dict(zip(spec.names, res.x))
    assert got["head_sigma[charge]"] == pytest.approx(0.3, rel=1e-3)
    assert got["head_sigma[ln_cn]"] == pytest.approx(0.5, rel=1e-3)


def test_the_shipped_head_scores_ions_as_fitted():
    from quick_mag.amplitude_head import head_factors

    p = default_params()
    model, _ = _system(magnetite)               # Fe3+ on both sites; the tetrahedral cage is the difference
    ends = {}
    for b in model.bridges:
        for s, f in ((b.site_i, b.feat_i), (b.site_j, b.feat_j)):
            d = model.descriptors[s]
            if d.oxidation_state == 3 or d.coordination == "tet":
                ends.setdefault(d.coordination, head_factors(p, f)[0])
    assert ends["tet"] / ends["oct"] > 2.0      # tetrahedral ions polarize harder (fewer bonds share the cation)


# ---- coordination number ------------------------------------------------------------


def test_smooth_coordination_number_tells_tetrahedra_from_octahedra():
    from quick_mag.amplitude_head import FEATURES

    model, _ = _system(co3o4)
    feats = {b.site_i: b.feat_i for b in model.bridges} | {b.site_j: b.feat_j for b in model.bridges}
    k = FEATURES.index("ln_cn")
    assert "tet" not in FEATURES
    for s, d in model.descriptors.items():
        if s in feats:
            want = np.log(4 / 6) if d.coordination == "tet" else 0.0
            assert feats[s][k] == pytest.approx(want, abs=0.05)


def test_sigma_load_of_ideal_cages():
    from quick_mag.amplitude_head import sigma_load

    I = np.eye(3)
    s3 = np.sqrt(3) / 2
    cages = {
        "oct": (np.vstack([I, -I]), 3.0),
        "tet": (np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]), 4 / 3),
        "square plane": (np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]), 2.5),
        "square pyramid": (np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]]), 2.6),
        "trigonal bipyramid": (np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-0.5, s3, 0], [-0.5, -s3, 0]]), 323 / 160),
    }
    rng = np.random.default_rng(0)
    for name, (u, want) in cages.items():
        assert sigma_load(u, np.ones(len(u))) == pytest.approx(want, abs=1e-12), name
        # rotation invariant
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        assert sigma_load(u @ q.T, np.ones(len(u))) == pytest.approx(want, abs=1e-12), name
    # = tr G^2 / tr G of the cage's sigma Gram matrix over the d shell (sk_table's sigma angular factors)
    from quick_mag.sk_table import pd_amplitudes

    for name, (u, want) in cages.items():
        uu = u / np.linalg.norm(u, axis=1)[:, None]
        A = np.array([pd_amplitudes(v)[0] for v in uu])
        G = A.T @ A
        assert np.trace(G @ G) / np.trace(G) == pytest.approx(want, abs=1e-9), name
    # continuous: a bond switching off moves an octahedron smoothly to a square pyramid
    oct_u = cages["oct"][0]
    loads = [sigma_load(oct_u, np.r_[np.ones(5), w]) for w in np.linspace(1, 0, 11)]
    assert loads[0] == pytest.approx(3.0) and loads[-1] == pytest.approx(2.6)
    assert np.all(np.abs(np.diff(loads)) < 0.1)


def test_sigma_load_tells_square_planes_from_tetrahedra():
    """CN is 4 for both; the sigma load is not, and the feature reaches the bridge ends."""
    from quick_mag.amplitude_head import FEATURES, NSIG_REF

    model, _ = _system(co3o4)
    feats = {b.site_i: b.feat_i for b in model.bridges} | {b.site_j: b.feat_j for b in model.bridges}
    k = FEATURES.index("ln_nsig")
    for s, d in model.descriptors.items():
        if s in feats:
            want = np.log((4 / 3) / NSIG_REF) if d.coordination == "tet" else 0.0
            assert feats[s][k] == pytest.approx(want, abs=0.05)


# ---- charge-transfer feature ---------------------------------------------------------


def test_ln_delta_follows_the_charge_transfer_trends():
    from quick_mag.amplitude_head import DELTA_FLOOR, DELTA_REF, ln_delta

    row = [ln_delta(el, 2) for el in ("Mn", "Fe", "Co", "Ni", "Cu")]
    assert all(a > b for a, b in zip(row, row[1:]))           # Delta falls along the M2+ row
    assert ln_delta("Fe", 2) > ln_delta("Fe", 3)                # and with the charge
    assert ln_delta("Zn", 2) == 0.0 and ln_delta("Ru", 4) == 0.0  # not in the table: the reference
    # the floor: a negative-Delta ion stays finite, just above DELTA_FLOOR
    assert np.log(DELTA_FLOOR / DELTA_REF) < ln_delta("Cu", 3) < np.log((DELTA_FLOOR + 0.1) / DELTA_REF)


def test_the_valence_split_filter_reads_the_bridge_ends():
    from quick_mag.amplitude_head import FEATURES
    from quick_mag.fitting_h5 import valence_split

    _model, system = _system(magnetite, valence="site")
    assert valence_split(system) <= 1                           # Fe2+ / Fe3+: one apart, kept
    ic = FEATURES.index("charge")
    fe = np.flatnonzero(system.metal_i == "Fe")
    system.feat_i[fe[0], ic] = 4 - 3                            # one end read as Fe4+ next to Fe2+
    system.feat_i[fe[1], ic] = 2 - 3
    assert valence_split(system) == 2


def test_normalized_sto_overlaps_match_closed_forms_and_limits():
    import numpy as np
    import pytest

    from quick_mag.radial_overlap import slater_zeta, sto_overlap

    z, R = 1.3, np.array([0.5, 1.0, 2.0, 4.0])
    exact = np.exp(-z * R) * (1 + z * R + (z * R) ** 2 / 3)          # 1s-1s, equal exponents
    assert np.allclose(sto_overlap(1, 0, z, 1, 0, z, R, normalized=True), exact, atol=1e-10)
    n, zc = slater_zeta("Cr", 3, "d")
    near = [float(sto_overlap(n, 2, zc, n, 2, zc, [0.05], m=m, normalized=True)[0]) for m in (0, 1, 2)]
    assert np.allclose(near, 1.0, atol=5e-3)                          # identical orbitals at R -> 0
    sig, pi, delta = (float(sto_overlap(n, 2, zc, n, 2, zc, [5.0], m=m, normalized=True)[0]) for m in (0, 1, 2))
    assert sig > 0 > pi and delta > 0 and abs(delta) < abs(pi) < abs(sig)   # Slater-Koster signs and order
    with pytest.raises(ValueError):
        sto_overlap(2, 1, 1.0, 3, 2, 1.0, [2.0], m=2)                 # no delta on a p orbital


@pytest.mark.parametrize("make", [magnetite, co3o4])
def test_direct_exchange_vectorizes(make):
    """The fit's J (k_dd * FitSystem.dd added) is the model's, with direct exchange on."""
    p = replace(default_params(), k_dd=0.3)
    model, system = _system(make, params=p)
    assert system.dd is not None and np.any(system.dd)
    assert np.allclose(j_eff_matrix(system, p), model.j_eff, rtol=1e-12, atol=1e-14)
