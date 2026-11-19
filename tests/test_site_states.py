"""User-set oxidation and spin states (site_states) and the moment convention (exchange_model)."""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.generation import generate_normal_spinel, generate_single_perovskite  # noqa: E402
from quick_mag.ion_descriptors import formal_moments, structure_ion_descriptors  # noqa: E402
from quick_mag.polarization_model import (  # noqa: E402
    build_bridges,
    build_Jeff_matrix,
    default_params,
    exchange_model,
    heisenberg_couplings,
    to_solver_couplings,
)
from quick_mag.site_states import (  # noqa: E402
    SiteStates,
    parse_oxidation_rule,
    parse_selector,
    parse_spin_rule,
    ranked_assignments,
)
from quick_mag.spin_solver import compute_config_energy  # noqa: E402


def co3o4():
    return generate_normal_spinel("Co3O4", a_site="Co", b_site="Co", a=8.08, u=0.263)


def magnetite():
    return generate_normal_spinel(
        "FeFe2O4", a_site="Fe", b_site="Fe", a=8.39, u=0.255,
        sublattice_oxidation_states={"A": {"Fe": (3,)}, "B": {"Fe": (2, 3)}},
    )


def lamno3():
    return generate_single_perovskite(
        "LaMnO3", a_site="La", b_site="Mn", x_site="O", a=3.95,
        n_cells_x=2, n_cells_y=2, n_cells_z=2,
    )


def roles(structure):
    from quick_mag.spinel_composition import structure_site_roles

    return [str(r) for r in structure_site_roles(structure)]


# ---- parsing -------------------------------------------------------------------------


class TestParsing:
    def test_selectors(self):
        s = parse_selector("Co3+@oct")
        assert (s.element, s.charge, s.where) == ("Co", 3, "oct")
        assert parse_selector("Mn@B").where == "B"
        assert parse_selector("Fe+2").charge == 2
        assert parse_selector("#4,7-9").indices == (4, 7, 8, 9)

    def test_oxidation_values(self):
        assert parse_oxidation_rule("Co@tet=2")[1] == frozenset({2})
        for text in ("Mn=3|4", "Mn={3,4}", "Mn=3,4", "Mn=+3|+4"):
            assert parse_oxidation_rule(text)[1] == frozenset({3, 4}), text

    def test_bad_rules(self):
        with pytest.raises(ValueError):
            parse_oxidation_rule("Co3+=2")          # an oxidation rule cannot select by charge
        with pytest.raises(ValueError):
            parse_oxidation_rule("Co")              # no value
        with pytest.raises(ValueError):
            parse_spin_rule("Co=IS")
        with pytest.raises(ValueError):
            parse_selector("cobalt")
        with pytest.raises(ValueError):
            parse_oxidation_rule("Co=12")           # not an ion the tables can build


# ---- oxidation constraints -----------------------------------------------------------


class TestConstraints:
    def test_label_and_coordination_select_the_same_atoms(self):
        st = co3o4()
        by_label = SiteStates.parse(["Co@A=2"]).constraints(st)
        by_coord = SiteStates.parse(["Co@tet=2"]).constraints(st)
        assert by_label == by_coord
        assert len(by_label) == roles(st).count("A") == 8

    def test_later_rules_and_per_atom_values_win(self):
        st = co3o4()
        b = [i for i, r in enumerate(roles(st)) if r == "B"]
        site_ox = [None] * st.atom_count
        site_ox[b[0]] = 2
        c = SiteStates.parse(["Co=2|3", "Co@B=3"], site_oxidation=site_ox).constraints(st)
        assert c[b[1]] == frozenset({3})
        assert c[b[0]] == frozenset({2})             # the per-atom value beats every rule
        assert all(c[i] == frozenset({2, 3}) for i, r in enumerate(roles(st)) if r == "A")

    def test_a_rule_that_selects_nothing_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SiteStates.parse(["Ni=2"]).constraints(co3o4())
        assert any("selects no atoms" in str(w.message) for w in caught)

    def test_pins_are_honoured_and_the_rest_rebalanced(self):
        st = magnetite()
        b = [i for i, r in enumerate(roles(st)) if r == "B"]
        site_ox = [None] * st.atom_count
        for i in b[:8]:
            site_ox[i] = 2
        (a,) = ranked_assignments(st, SiteStates(site_oxidation=site_ox), top_k=1)
        ox = a.site_oxidation_states
        assert all(ox[i] == 2 for i in b[:8])
        assert all(ox[i] == 3 for i in b[8:])

    def test_unbalanced_pins_say_by_how_much(self):
        st = co3o4()
        with pytest.raises(ValueError, match="sum to"):
            ranked_assignments(st, SiteStates.parse(["Co=3"]))


# ---- spin states ---------------------------------------------------------------------


class TestSpinStates:
    def test_low_spin_co3_carries_no_moment(self):
        st = co3o4()
        (hs,) = ranked_assignments(st, SiteStates.parse(["Co@A=2", "Co@B=3"]))
        (ls,) = ranked_assignments(st, SiteStates.parse(["Co@A=2", "Co@B=3"], ["Co3+@oct=LS"]))
        B = [i for i, r in enumerate(roles(st)) if r == "B"]
        A = [i for i, r in enumerate(roles(st)) if r == "A"]
        assert np.allclose(hs.magnetic_moments[B], 4.0)
        assert np.allclose(ls.magnetic_moments[B], 0.0)
        assert np.allclose(ls.magnetic_moments[A], 3.0)
        assert all(ls.spin_states[i] == "LS" for i in B)
        assert all(ls.spin_states[i] == "HS" for i in A)
        # LS Co3+ is t2g^6: it couples to nothing
        model = exchange_model(st, ls)
        idx = model.site_index
        assert np.allclose(model.J[np.ix_([idx[i] for i in B], range(len(model.sites)))], 0.0)

    def test_low_spin_co2_is_eg1(self):
        st = co3o4()
        B = [i for i, r in enumerate(roles(st)) if r == "B"]
        (a,) = ranked_assignments(st, SiteStates.parse(["Co@A=2", "Co@B=3"]))
        a.site_oxidation_states[B[0]] = 2          # hand a site a different state, then name its spin
        d = structure_ion_descriptors(st, a, spin_states={B[0]: "LS"})
        assert (d[B[0]].n_t2g, d[B[0]].n_eg, d[B[0]].spin) == (6, 1, 0.5)
        assert formal_moments(d)[B[0]] == 1.0

    def test_tetrahedral_sites_are_low_spin_only_when_named(self):
        st = co3o4()
        (a,) = ranked_assignments(st, SiteStates.parse(["Co@A=2", "Co@B=3"]))
        A = [i for i, r in enumerate(roles(st)) if r == "A"]
        assert all(d.spin_state == "HS" for i, d in structure_ion_descriptors(st, a, "LS").items() if i in A)
        named = structure_ion_descriptors(st, a, spin_states={A[0]: "LS"})
        assert named[A[0]].spin_state == "LS"


# ---- moment convention ---------------------------------------------------------------


def _old_pipeline(structure, assignment):
    descriptors = structure_ion_descriptors(structure, assignment)
    sites = sorted(descriptors)
    bridges = build_bridges(structure, descriptors)
    params = default_params()
    j_eff = build_Jeff_matrix(bridges, {s: k for k, s in enumerate(sites)}, params)
    from quick_mag.direct_exchange import dd_geometry

    j_eff = j_eff + params.k_dd * dd_geometry(structure, descriptors, sites, cutoff=params.dd_cutoff)
    unit = (np.abs(assignment.magnetic_moments[sites]) > 1e-8).astype(float)
    return to_solver_couplings(j_eff), unit


@pytest.mark.parametrize("make", [lamno3, magnetite, co3o4])
def test_energies_are_unchanged_by_the_moment_convention(make):
    st = make()
    (a,) = ranked_assignments(st, None, top_k=1, base_constraints=_base(st))
    model = exchange_model(st, a)
    j_old, unit = _old_pipeline(st, a)
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.choice([-1.0, 1.0], size=len(model.sites))
        e_old = compute_config_energy(j_old, s * unit)
        e_new = compute_config_energy(model.solver_J, s * model.moments)
        assert e_new == pytest.approx(e_old, rel=1e-10, abs=1e-12)
        assert model.energy(s) == pytest.approx(e_old, rel=1e-10, abs=1e-12)


def _base(st):
    from quick_mag.spinel_composition import sublattice_oxidation_constraints

    try:
        return sublattice_oxidation_constraints(st)
    except Exception:
        return {}


def test_heisenberg_couplings_divide_by_the_moment_product():
    j = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    m = np.array([2.0, 4.0, 0.0])
    jt = heisenberg_couplings(j, m)
    assert jt[0, 1] == pytest.approx(2.0 / 8.0)
    assert jt[0, 2] == 0.0 and jt[1, 2] == 0.0


# ---- definite valence ----------------------------------------------------------------


def test_site_valence_matches_class_valence_when_single_valent():
    st = lamno3()
    (a,) = ranked_assignments(st)
    assert np.allclose(exchange_model(st, a, valence="site").J, exchange_model(st, a).J)


def test_site_valence_resolves_a_mixed_valent_class():
    st = magnetite()
    (a,) = ranked_assignments(st, base_constraints=_base(st))
    cls = exchange_model(st, a)
    site = exchange_model(st, a, valence="site")
    assert not np.allclose(cls.J, site.J)
    # double exchange is gated by the class either way
    de_cls = [(b.site_i, b.site_j, b.de_weight) for b in cls.bridges if b.de_active]
    de_site = [(b.site_i, b.site_j, b.de_weight) for b in site.bridges if b.de_active]
    assert de_cls == de_site and de_cls


# ---- CLI -----------------------------------------------------------------------------


def test_cli_rules_reach_the_solve(tmp_path, capsys):
    from quick_mag.magnetic_cli import configure_solve_parser, solve_structure
    import argparse

    args = configure_solve_parser(argparse.ArgumentParser()).parse_args(
        ["--ox", "Co@A=2", "--ox", "Co@B=3", "--spin", "Co3+@oct=LS", "--exact-max-sites", "0",
         "--n-trials", "2", "--n-steps", "20"]
    )
    status = solve_structure(co3o4(), args)
    out = capsys.readouterr().out
    assert status == 0
    assert "User oxidation constraints on 24 atom(s)" in out
    assert "16xCo+3(oct,LS) 0 muB" in out
    assert "8xCo+2(tet,HS) 3 muB" in out
