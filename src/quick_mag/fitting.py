"""Fitting the exchange-polarization model's atomic parameters to collinear DFT energies.

Every coupling stays a function of atomic parameters (``PolarizationParameters``: w and J_H per
ligand, gamma_pi, the amplitude head's weights on ion descriptors, the range law's n / delta_sat,
t_de and j_fm) and the geometry -- the fit moves those, never a pair J -- so what it learns
transfers to structures it has not seen.

Data
----
A :class:`FitSystem` is one structure under one oxidation / spin-state assignment (the states
DFT actually found, set through ``quick_mag.site_states``) with several collinear
configurations at that one geometry and their DFT energies. Its bridges are built once
(:func:`fit_system`) and vectorized, so a residual evaluation is array arithmetic. The model
energy of a configuration is exactly :meth:`polarization_model.ExchangeModel.energy` (tested).

Loss: per system, the configurations' energies relative to their mean, per magnetic site
(eV/site), model minus DFT, times the system's weight (default 1/sqrt(n_configs), so a system
with many configurations does not dominate); plus a pull of each free parameter toward its prior
(``reg_weight * ((x - prior)/scale)^2``). Only energy differences within a system enter -- one
assignment, one geometry -- so neither a lattice-energy difference nor a valence change can
leak into the fit (filter configurations whose moments changed before building systems).

Parameters and stages
---------------------
:class:`ParamSpec` names the free parameters (``"w[O]"``, ``"jh[O]"``, ``"gamma_pi"``, ``"k_dd"``, ``"k_ee"``,
``"head_sigma[ln_cn]"``, ``"head_pi[*]"``, ``"n[*]"`` = one shared exponent for every metal of the
data, ``"delta_sat"``, ...); everything else is held at the base parameter set. A staged fit is a
sequence of specs, each starting from the last one's result (:func:`fit_staged`).

Diagnostics (:func:`diagnose`): energy RMSE (meV/site), ground-state match rate and mean
Spearman rho; :func:`leave_one_group_out`; :func:`identifiability` -- the eigen-decomposition of
J^T J of the data residuals at the solution, its condition number, the parameter combinations
of the weakest directions and per-parameter standard errors.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from quick_mag.polarization_model import (
    ExchangeModel,
    PolarizationParameters,
    default_params,
    N_TSETS,
    T_EE,
)

__all__ = [
    "FitSystem",
    "ParamSpec",
    "FitResult",
    "fit_system",
    "predict",
    "fit",
    "fit_staged",
    "diagnose",
    "identifiability",
    "leave_one_group_out",
]


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------


@dataclass
class FitSystem:
    """One structure + assignment with its configurations, bridges vectorized."""

    name: str
    n_sites: int
    signs: np.ndarray          # (n_configs, n_sites) in {+1, 0, -1}; 0 on zero-moment sites
    energies: np.ndarray       # (n_configs,) DFT energies (eV)
    weight: float
    group: str                 # for leave-one-group-out (e.g. the composition)
    # per bridge
    idx_i: np.ndarray
    idx_j: np.ndarray
    metal_i: np.ndarray        # element symbols (object arrays)
    metal_j: np.ndarray
    ligand: np.ndarray
    dr_i: np.ndarray           # r_iL - R0_iL
    dr_j: np.ndarray
    rr_i: np.ndarray           # r_iL / R0_iL
    rr_j: np.ndarray
    feat_i: np.ndarray         # (n_bridges, F): amplitude-head features of each end's ion
    feat_j: np.ndarray
    Bsig_i: np.ndarray
    Bpi_i: np.ndarray
    Bsig_j: np.ndarray
    Bpi_j: np.ndarray
    de_env: np.ndarray         # cos^2(theta) * de_weight on DE-active bridges, else 0
    oe_env: np.ndarray         # 1 on occupied->empty-active bridges
    Bsig_occ_i: np.ndarray
    Bpi_occ_i: np.ndarray
    Bsig_occ_j: np.ndarray
    Bpi_occ_j: np.ndarray
    Bsig_emp_i: np.ndarray
    Bpi_emp_i: np.ndarray
    Bsig_emp_j: np.ndarray
    Bpi_emp_j: np.ndarray
    labels: List[str] = field(default_factory=list)   # configuration names
    dd: Optional[np.ndarray] = None   # direct-exchange geometry G (n_sites, n_sites); J += k_dd * G
    # (n_bridges, 3, 3, 3, 3): ``BridgeGeometry.T`` of each end (the coherent kernel and the empty-empty term)
    T_i: Optional[np.ndarray] = None
    T_j: Optional[np.ndarray] = None
    G_dx: Optional[np.ndarray] = None     # (n_bridges, 4): direct-exchange interference geometry (0 beyond the cutoff)
    ie_i: Optional[np.ndarray] = None     # (n_bridges, 3): (IE_{q+1}, IE_q, q) of each end (oe_rule "ionization")
    ie_j: Optional[np.ndarray] = None
    de_env1: Optional[np.ndarray] = None  # |cos(theta)| * de_weight on DE-active bridges (de_form "linear")

    @property
    def elements(self) -> set:
        return set(self.metal_i) | set(self.metal_j) | set(self.ligand)

    @property
    def metals(self) -> set:
        """Metals that polarize a ligand here (an end with any sigma/pi intensity; not d0 / d10)."""
        on_i = (np.abs(self.Bsig_i).sum(1) + np.abs(self.Bpi_i).sum(1)) > 0
        on_j = (np.abs(self.Bsig_j).sum(1) + np.abs(self.Bpi_j).sum(1)) > 0
        return set(self.metal_i[on_i]) | set(self.metal_j[on_j])

    @property
    def ligands(self) -> set:
        return set(self.ligand)

    @property
    def de_elements(self) -> set:
        on = self.de_env > 0
        return set(self.metal_i[on]) | set(self.metal_j[on])

    @property
    def oe_elements(self) -> set:
        on = self.oe_env > 0
        return set(self.metal_i[on]) | set(self.metal_j[on])


def fit_system(
    model: ExchangeModel,
    signs,
    energies,
    *,
    name: str = "",
    group: Optional[str] = None,
    weight: Optional[float] = None,
    labels: Sequence[str] = (),
) -> FitSystem:
    """A :class:`FitSystem` from an exchange model and its configurations.

    ``signs``: (n_configs, n_sites) signs on ``model.sites`` (any sign of a zero-moment site
    is ignored); ``energies``: the DFT energy of each (eV).
    """
    signs = np.sign(np.asarray(signs, dtype=float))
    if signs.ndim != 2 or signs.shape[1] != len(model.sites):
        raise ValueError(f"signs must be (n_configs, {len(model.sites)}), got {signs.shape}")
    signs = signs * (model.moments > 1e-12)[None, :]
    energies = np.asarray(energies, dtype=float).reshape(-1)
    if energies.size != signs.shape[0]:
        raise ValueError("one energy per configuration")
    index = model.site_index
    bridges = [b for b in model.bridges if b.site_i in index and b.site_j in index]

    def arr(get):
        return np.array([get(b) for b in bridges], dtype=float).reshape(len(bridges), -1)

    def occ(b, end):
        sig = getattr(b, f"Bsig_occ_{end}")
        pi = getattr(b, f"Bpi_occ_{end}")
        return (getattr(b, f"Bsig_{end}") if sig is None else sig,
                getattr(b, f"Bpi_{end}") if pi is None else pi)

    zeros = np.zeros(3)
    n = signs.shape[0]
    return FitSystem(
        name=name,
        n_sites=len(model.sites),
        signs=signs,
        energies=energies,
        weight=float(1.0 / math.sqrt(n) if weight is None else weight) if n > 1 else 0.0,
        group=group if group is not None else name,
        idx_i=np.array([index[b.site_i] for b in bridges], dtype=int),
        idx_j=np.array([index[b.site_j] for b in bridges], dtype=int),
        metal_i=np.array([b.metal_i for b in bridges], dtype=object),
        metal_j=np.array([b.metal_j for b in bridges], dtype=object),
        ligand=np.array([b.ligand for b in bridges], dtype=object),
        dr_i=np.array([b.r_iL - b.R0_iL for b in bridges], dtype=float),
        dr_j=np.array([b.r_jL - b.R0_jL for b in bridges], dtype=float),
        rr_i=np.array([b.r_iL / b.R0_iL for b in bridges], dtype=float),
        rr_j=np.array([b.r_jL / b.R0_jL for b in bridges], dtype=float),
        feat_i=np.array([b.feat_i for b in bridges], dtype=float).reshape(len(bridges), -1),
        feat_j=np.array([b.feat_j for b in bridges], dtype=float).reshape(len(bridges), -1),
        Bsig_i=arr(lambda b: b.Bsig_i),
        Bpi_i=arr(lambda b: b.Bpi_i),
        Bsig_j=arr(lambda b: b.Bsig_j),
        Bpi_j=arr(lambda b: b.Bpi_j),
        de_env=np.array([b.cos_theta ** 2 * b.de_weight if b.de_active else 0.0 for b in bridges]),
        oe_env=np.array([1.0 if b.oe_active else 0.0 for b in bridges]),
        Bsig_occ_i=arr(lambda b: occ(b, "i")[0]),
        Bpi_occ_i=arr(lambda b: occ(b, "i")[1]),
        Bsig_occ_j=arr(lambda b: occ(b, "j")[0]),
        Bpi_occ_j=arr(lambda b: occ(b, "j")[1]),
        Bsig_emp_i=arr(lambda b: b.Bsig_emp_i if b.oe_active else zeros),
        Bpi_emp_i=arr(lambda b: b.Bpi_emp_i if b.oe_active else zeros),
        Bsig_emp_j=arr(lambda b: b.Bsig_emp_j if b.oe_active else zeros),
        Bpi_emp_j=arr(lambda b: b.Bpi_emp_j if b.oe_active else zeros),
        labels=list(labels),
        dd=None if getattr(model, "dd", None) is None else np.array(model.dd, dtype=float),
        T_i=(np.array([b.T_i for b in bridges], dtype=float).reshape(len(bridges), N_TSETS, 3, 3, 3)
             if all(b.T_i is not None for b in bridges) else None),
        T_j=(np.array([b.T_j for b in bridges], dtype=float).reshape(len(bridges), N_TSETS, 3, 3, 3)
             if all(b.T_j is not None for b in bridges) else None),
        G_dx=np.array([np.zeros(4) if b.G_dx is None else b.G_dx for b in bridges], dtype=float).reshape(len(bridges), 4),
        ie_i=np.array([b.ie_i for b in bridges], dtype=float).reshape(len(bridges), 3),
        ie_j=np.array([b.ie_j for b in bridges], dtype=float).reshape(len(bridges), 3),
        de_env1=np.array([abs(b.cos_theta) * b.de_weight if b.de_active else 0.0 for b in bridges]),
    )


# ---------------------------------------------------------------------------------------
# Prediction (vectorized bridge_J_components)
# ---------------------------------------------------------------------------------------


def _lookup(table: Dict[str, float], default: float, keys: np.ndarray) -> np.ndarray:
    return np.array([table.get(k, default) for k in keys], dtype=float)


def _damping(system: FitSystem, params: PolarizationParameters, end: str):
    """(d_sigma, d_pi) per bridge for one end, as ``polarization_model.end_damping``."""
    from quick_mag.amplitude_head import FEATURES
    from quick_mag.polarization_model import range_exponent

    ln_delta = getattr(system, f"feat_{end}")[:, FEATURES.index("ln_delta")]
    if params.delta_sat > 0.0:
        n = range_exponent(params, None, ln_delta)
    else:
        n = _lookup(params.range_power, params.range_power_default, getattr(system, f"metal_{end}")) + params.n_delta * ln_delta
    d = getattr(system, f"rr_{end}") ** (-n)
    return d, d


def j_eff_matrix(system: FitSystem, params: PolarizationParameters) -> np.ndarray:
    """The unit-spin coupling matrix (J > 0 AFM) of ``system`` under ``params``: the same
    numbers as ``build_Jeff_matrix`` over the same bridges."""
    ds_i, dp_i = _damping(system, params, "i")
    ds_j, dp_j = _damping(system, params, "j")
    damp_i, damp_j = ds_i, ds_j                     # double exchange: the sigma channel
    if params.head_sigma or params.head_pi:
        from quick_mag.amplitude_head import head_factors

        hs_i, hp_i = head_factors(params, system.feat_i)
        hs_j, hp_j = head_factors(params, system.feat_j)
        ds_i, dp_i, ds_j, dp_j = ds_i * hs_i, dp_i * hp_i, ds_j * hs_j, dp_j * hp_j
    g2 = params.gamma_pi ** 2
    if params.kernel not in ("coherent", "diagonal"):
        raise ValueError(f"kernel must be coherent or diagonal, got {params.kernel!r}")
    from quick_mag.polarization_model import end_polarization

    have_T = system.T_i is not None and system.T_j is not None
    coherent = params.kernel == "coherent" and have_T
    fs_i, fp_i, fs_j, fp_j = ds_i, g2 * dp_i, ds_j, g2 * dp_j          # per bridge
    ds_i, dp_i, ds_j, dp_j = ds_i[:, None], dp_i[:, None], ds_j[:, None], dp_j[:, None]
    w = _lookup(params.w_ligand, params.w_default, system.ligand)
    jh = _lookup(params.jh_ligand, params.jh_default, system.ligand)
    if coherent:
        P_i = end_polarization(system.T_i[:, 0], fs_i, fp_i)
        P_j = end_polarization(system.T_j[:, 0], fs_j, fp_j)
        j = (2.0 * (w + jh) * np.einsum("bpq,bpq->b", P_i, P_j)
             - 2.0 * jh * np.einsum("bpp->b", P_i) * np.einsum("bpp->b", P_j))
    else:
        mu_i = ds_i * system.Bsig_i + g2 * dp_i * system.Bpi_i
        mu_j = ds_j * system.Bsig_j + g2 * dp_j * system.Bpi_j
        j = 2.0 * (w + jh) * np.einsum("bp,bp->b", mu_i, mu_j) - 2.0 * jh * mu_i.sum(1) * mu_j.sum(1)
    if params.k_ee and have_T:
        from quick_mag.polarization_model import T_EE

        E_i = end_polarization(system.T_i[:, T_EE[params.ee_spin]], fs_i, fp_i, coherent)
        E_j = end_polarization(system.T_j[:, T_EE[params.ee_spin]], fs_j, fp_j, coherent)
        j = j + 2.0 * (w + jh) * params.k_ee ** 2 * np.einsum("bpq,bpq->b", E_i, E_j)
    if params.k_dx and system.G_dx is not None:
        amp = np.sqrt(np.stack([fs_i * fs_j, fs_i * fp_j, fp_i * fs_j, fp_i * fp_j], axis=1))
        j = j + params.k_dx * np.einsum("bk,bk->b", system.G_dx, amp)
    if np.any(system.de_env):
        t = (_lookup(params.t_de, params.t_de_default, system.metal_i)
             * _lookup(params.t_de, params.t_de_default, system.metal_j))
        if params.de_form not in ("linear", "quadratic"):
            raise ValueError(f"de_form must be linear or quadratic, got {params.de_form!r}")
        if params.de_form == "linear":
            env1 = system.de_env1 if system.de_env1 is not None else np.sqrt(system.de_env)
            j = j - t * np.sqrt(fs_i * fs_j) * env1
        else:
            j = j - t * damp_i * damp_j * system.de_env
    if np.any(system.oe_env):
        jf = (_lookup(params.j_fm, params.j_fm_default, system.metal_i)
              * _lookup(params.j_fm, params.j_fm_default, system.metal_j))
        if params.oe_rule not in ("product", "acceptor", "ionization"):
            raise ValueError(f"oe_rule must be product, acceptor or ionization, got {params.oe_rule!r}")
        if params.oe_rule == "ionization":
            overlap = 0.0
            if have_T and system.ie_i is not None:
                from quick_mag.polarization_model import oe_transfer_energy

                H_i = end_polarization(system.T_i[:, 0], fs_i, fp_i, coherent)
                H_j = end_polarization(system.T_j[:, 0], fs_j, fp_j, coherent)
                A_i = end_polarization(system.T_i[:, 1], fs_i, fp_i, coherent)
                A_j = end_polarization(system.T_j[:, 1], fs_j, fp_j, coherent)
                u_ij = oe_transfer_energy(params, system.ie_i.T, system.ie_j.T)
                u_ji = oe_transfer_energy(params, system.ie_j.T, system.ie_i.T)
                overlap = 2.0 * (w + jh) * params.k_oe * (
                    np.exp(-(u_ij - 3.0) / params.oe_decay) * np.einsum("bpq,bpq->b", H_i, A_j)
                    + np.exp(-(u_ji - 3.0) / params.oe_decay) * np.einsum("bpq,bpq->b", A_i, H_j))
            jf = 1.0
        elif params.oe_rule == "acceptor":
            overlap = 0.0
            if have_T:
                H_i = end_polarization(system.T_i[:, 0], fs_i, fp_i, coherent)
                H_j = end_polarization(system.T_j[:, 0], fs_j, fp_j, coherent)
                c_i = _lookup(params.j_fm, params.j_fm_default, system.metal_i)[:, None, None]
                c_j = _lookup(params.j_fm, params.j_fm_default, system.metal_j)[:, None, None]
                X_i = (c_i * end_polarization(system.T_i[:, 1], fs_i, fp_i, coherent)
                       + params.k_oe * end_polarization(system.T_i[:, T_EE[params.ee_spin]], fs_i, fp_i, coherent))
                X_j = (c_j * end_polarization(system.T_j[:, 1], fs_j, fp_j, coherent)
                       + params.k_oe * end_polarization(system.T_j[:, T_EE[params.ee_spin]], fs_j, fp_j, coherent))
                overlap = 2.0 * (w + jh) * (np.einsum("bpq,bpq->b", H_i, X_j) + np.einsum("bpq,bpq->b", X_i, H_j))
            jf = 1.0
        elif coherent:
            A_i = end_polarization(system.T_i[:, 1], fs_i, fp_i)
            A_j = end_polarization(system.T_j[:, 1], fs_j, fp_j)
            overlap = np.einsum("bpq,bpq->b", P_i, A_j) + np.einsum("bpq,bpq->b", A_i, P_j)
        else:
            mo_i = ds_i * system.Bsig_occ_i + g2 * dp_i * system.Bpi_occ_i
            mo_j = ds_j * system.Bsig_occ_j + g2 * dp_j * system.Bpi_occ_j
            me_i = ds_i * system.Bsig_emp_i + g2 * dp_i * system.Bpi_emp_i
            me_j = ds_j * system.Bsig_emp_j + g2 * dp_j * system.Bpi_emp_j
            overlap = np.einsum("bp,bp->b", mo_i, me_j) + np.einsum("bp,bp->b", me_i, mo_j)
        j = j - jf * system.oe_env * overlap
    if params.t2g_oe != "off" and have_T:
        from quick_mag.polarization_model import t2g_oe_term

        j = j + t2g_oe_term(params, w, jh, system.T_i, system.T_j, fs_i, fp_i, fs_j, fp_j, coherent)
    out = np.zeros((system.n_sites, system.n_sites))
    np.add.at(out, (system.idx_i, system.idx_j), j)
    np.add.at(out, (system.idx_j, system.idx_i), j)
    if params.k_dd and system.dd is not None:
        out = out + params.k_dd * system.dd
    return out


def predict(system: FitSystem, params: PolarizationParameters) -> np.ndarray:
    """Model energy of every configuration: ``1/2 s^T J_eff s`` (= ``ExchangeModel.energy``)."""
    j = j_eff_matrix(system, params)
    return 0.5 * np.einsum("ci,ij,cj->c", system.signs, j, system.signs)


# ---------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------

_FIELDS = {
    "w": ("w_ligand", "w_default"),
    "jh": ("jh_ligand", "jh_default"),
    "t_de": ("t_de", "t_de_default"),
    "j_fm": ("j_fm", "j_fm_default"),
    "n": ("range_power", "range_power_default"),
    "head_sigma": ("head_sigma", None),
    "head_pi": ("head_pi", None),
}
_SCALARS = ("gamma_pi", "k_dd", "k_ee", "k_oe", "k_t2g", "k_dx", "n_delta", "delta_sat", "oe_decay", "oe_charge")      # parameters that are one number, not a table
# (lower, upper, prior scale as a fraction of |prior|, floor of the scale)
_BOUNDS = {
    "w": (1e-4, 10.0, 0.5, 0.005),
    "jh": (0.0, 10.0, 0.5, 0.002),
    "t_de": (0.0, 3.0, 0.5, 0.05),
    "j_fm": (0.0, 5.0, 0.5, 0.1),
    "gamma_pi": (0.0, 2.0, 0.5, 0.1),
    "k_dd": (0.0, 50.0, 0.5, 0.1),
    "k_ee": (0.0, 5.0, 0.5, 0.1),
    "k_oe": (0.0, 5.0, 0.5, 0.1),
    "k_t2g": (0.0, 20.0, 0.5, 0.5),
    "k_dx": (-50.0, 50.0, 0.5, 0.1),
    "n_delta": (-20.0, 20.0, 0.5, 1.0),
    "delta_sat": (0.5, 40.0, 0.5, 1.0),
    "oe_decay": (0.2, 30.0, 0.5, 0.5),
    "oe_charge": (0.0, 40.0, 0.5, 3.0),
    "n": (0.0, 30.0, 0.5, 2.0),
    "head_sigma": (-5.0, 5.0, 0.5, 0.5),
    "head_pi": (-5.0, 5.0, 0.5, 0.5),
}


@dataclass
class _Free:
    name: str                    # as written: "w[O]", "n[*]", "head_pi[n_eg]", "gamma_pi"
    kind: str                    # w / jh / n / head_sigma / ... / gamma_pi
    targets: Tuple[str, ...]     # elements it sets (empty for gamma_pi)


@dataclass
class ParamSpec:
    """Which parameters a fit moves, their priors, bounds and prior scales.

    ``free``: names like ``"w[O]"``, ``"jh[*]"`` (one per ligand of the data), ``"gamma_pi"``,
    ``"head_sigma[ln_cn]"``, ``"head_pi[*]"`` (one per amplitude-head feature), ``"t_de[Mn]"``,
    ``"n[*]"`` (*one* shared range exponent for every metal: tied). Glob
    patterns expand against ``systems``; ``n[*]`` is the exception, a tie, as
    the data rarely separates per-metal ranges (use ``n[Fe]`` etc. to free one alone).
    ``base``: the parameter set everything not free is held at, and the priors.
    """

    free: Sequence[str]
    systems: Sequence[FitSystem]
    base: PolarizationParameters = field(default_factory=default_params)
    prior_scale: Optional[Dict[str, float]] = None   # override the scale for a name

    def __post_init__(self) -> None:
        metals = sorted(set().union(*(s.metals for s in self.systems))) if self.systems else []
        ligands = sorted(set().union(*(s.ligands for s in self.systems))) if self.systems else []
        de = sorted(set().union(*(s.de_elements for s in self.systems))) if self.systems else []
        self.metals, self.ligands, self.de = metals, ligands, de
        frees: List[_Free] = []
        for raw in self.free:
            name = raw.replace(" ", "")
            if name in _SCALARS:
                frees.append(_Free(name, name, ()))
                continue
            if "[" not in name or not name.endswith("]"):
                raise ValueError(f"cannot read parameter {raw!r}")
            kind, pattern = name[:-1].split("[", 1)
            if kind not in _FIELDS:
                raise ValueError(f"unknown parameter kind {kind!r} in {raw!r}")
            universe = ligands if kind in ("w", "jh") else (de if kind == "t_de" else metals)
            if kind in ("head_sigma", "head_pi"):
                from quick_mag.amplitude_head import FEATURES

                universe = list(FEATURES)
            if kind == "n" and pattern == "*":
                frees.append(_Free(f"{kind}[*]", kind, tuple(metals)))
                continue
            hits = [e for e in universe if fnmatch.fnmatchcase(e, pattern)]
            if not hits:
                raise ValueError(f"{raw!r} matches no element of the data ({universe})")
            for e in hits:
                frees.append(_Free(f"{kind}[{e}]", kind, (e,)))
        names = [f.name for f in frees]
        if len(set(names)) != len(names):
            raise ValueError(f"a parameter is freed twice: {names}")
        self._frees = frees

    @property
    def names(self) -> List[str]:
        return [f.name for f in self._frees]

    def _value(self, params: PolarizationParameters, f: _Free) -> float:
        if f.kind in _SCALARS:
            return float(getattr(params, f.kind))
        table, default = _FIELDS[f.kind]
        fallback = 0.0 if default is None else getattr(params, default)
        values = [getattr(params, table).get(e, fallback) for e in f.targets]
        return float(np.mean(values))

    def pack(self, params: PolarizationParameters) -> np.ndarray:
        return np.array([self._value(params, f) for f in self._frees])

    @property
    def prior(self) -> np.ndarray:
        return self.pack(self.base)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array([_BOUNDS[f.kind][0] for f in self._frees])
        hi = np.array([_BOUNDS[f.kind][1] for f in self._frees])
        return lo, hi

    @property
    def scale(self) -> np.ndarray:
        out = []
        for f, p in zip(self._frees, self.prior):
            _lo, _hi, frac, floor = _BOUNDS[f.kind]
            s = max(abs(p) * frac, floor)
            if self.prior_scale and f.name in self.prior_scale:
                s = float(self.prior_scale[f.name])
            out.append(s)
        return np.array(out)

    def unpack(self, x) -> PolarizationParameters:
        p = replace(
            self.base,
            w_ligand=dict(self.base.w_ligand), jh_ligand=dict(self.base.jh_ligand),
            t_de=dict(self.base.t_de), j_fm=dict(self.base.j_fm),
            range_power=dict(self.base.range_power),
            head_sigma=dict(self.base.head_sigma), head_pi=dict(self.base.head_pi),
        )
        for f, v in zip(self._frees, np.asarray(x, dtype=float)):
            if f.kind in _SCALARS:
                setattr(p, f.kind, float(v))
                continue
            table = getattr(p, _FIELDS[f.kind][0])
            for e in f.targets:
                table[e] = float(v)
        return p


# ---------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------


def _data_residuals(params: PolarizationParameters, systems: Sequence[FitSystem]) -> np.ndarray:
    parts = []
    for s in systems:
        if s.weight == 0.0:
            continue
        e_m = predict(s, params)
        r = ((e_m - e_m.mean()) - (s.energies - s.energies.mean())) / s.n_sites
        parts.append(s.weight * r)
    return np.concatenate(parts) if parts else np.zeros(0)


def _residuals(x, spec: ParamSpec, systems, reg_weight: float) -> np.ndarray:
    data = _data_residuals(spec.unpack(x), systems)
    reg = math.sqrt(reg_weight) * (np.asarray(x) - spec.prior) / spec.scale
    return np.concatenate([data, reg])


@dataclass
class FitResult:
    params: PolarizationParameters
    spec: ParamSpec
    x: np.ndarray
    cost: float
    success: bool
    message: str
    at_bound: List[str] = field(default_factory=list)

    def table(self) -> str:
        lines = []
        for name, p0, v in zip(self.spec.names, self.spec.prior, self.x):
            flag = "  (at bound)" if name in self.at_bound else ""
            lines.append(f"    {name:14s} {p0:9.4f} -> {v:9.4f}{flag}")
        return "\n".join(lines)


def fit(
    systems: Sequence[FitSystem],
    spec: ParamSpec,
    *,
    reg_weight: float = 1e-4,
    x0: Optional[np.ndarray] = None,
    **least_squares_kwargs,
) -> FitResult:
    """Least squares over ``spec``'s free parameters (bounded, scaled by the Jacobian)."""
    from scipy.optimize import least_squares

    lo, hi = spec.bounds
    x0 = np.clip(spec.prior if x0 is None else np.asarray(x0, dtype=float), lo + 1e-12, hi - 1e-12)
    res = least_squares(_residuals, x0, bounds=(lo, hi), x_scale="jac",
                        args=(spec, systems, reg_weight), **least_squares_kwargs)
    at = [n for n, v, a, b in zip(spec.names, res.x, lo, hi) if v <= a + 1e-9 or v >= b - 1e-9]
    return FitResult(spec.unpack(res.x), spec, res.x, float(res.cost), bool(res.success), str(res.message), at)


def fit_staged(
    systems: Sequence[FitSystem],
    stages: Sequence[Sequence[str]],
    *,
    base: Optional[PolarizationParameters] = None,
    reg_weight: float = 1e-4,
    log=print,
) -> List[Tuple[FitResult, Dict[str, float], dict]]:
    """Fit stage by stage, each starting from (and pulled toward) the previous result's
    parameters. Returns ``[(result, diagnose(...), identifiability(...)), ...]``."""
    params = base or default_params()
    out = []
    for k, free in enumerate(stages):
        spec = ParamSpec(free=free, systems=systems, base=params)
        res = fit(systems, spec, reg_weight=reg_weight)
        diag = diagnose(systems, res.params)
        ident = identifiability(systems, spec, res.x)
        if log:
            log(f"stage {k + 1}: free {spec.names}\n{res.table()}\n"
                f"    RMSE {diag['rmse_meV_per_site']:.2f} meV/site, ground match {diag['ground_match']:.0%}, "
                f"rho {diag['spearman']:.3f}; cond {ident['condition']:.3g}")
        params = res.params
        out.append((res, diag, ident))
    return out


# ---------------------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------------------


def diagnose(systems: Sequence[FitSystem], params: PolarizationParameters) -> Dict[str, float]:
    """RMSE of within-system energy differences (meV per magnetic site), how often the model's
    lowest configuration is DFT's, and the mean Spearman rho of the orderings."""
    from scipy.stats import spearmanr

    sq, match, n, rhos = [], 0, 0, []
    for s in systems:
        if len(s.energies) < 2:
            continue
        n += 1
        e_m = predict(s, params)
        d = ((e_m - e_m.mean()) - (s.energies - s.energies.mean())) / s.n_sites
        sq.extend(d ** 2)
        match += int(np.argmin(e_m) == np.argmin(s.energies))
        if len(s.energies) > 2:
            rho = spearmanr(e_m, s.energies).statistic
            if np.isfinite(rho):
                rhos.append(float(rho))
    return {
        "systems": float(n),
        "rmse_meV_per_site": float(np.sqrt(np.mean(sq)) * 1000.0) if sq else float("nan"),
        "ground_match": match / n if n else float("nan"),
        "spearman": float(np.mean(rhos)) if rhos else float("nan"),
    }


def _jacobian(fun, x, eps=1e-6) -> np.ndarray:
    f0 = fun(x)
    jac = np.zeros((f0.size, x.size))
    for k in range(x.size):
        h = eps * max(1.0, abs(x[k]))
        xp = x.copy()
        xp[k] += h
        xm = x.copy()
        xm[k] -= h
        jac[:, k] = (fun(xp) - fun(xm)) / (2 * h)
    return jac


def identifiability(systems: Sequence[FitSystem], spec: ParamSpec, x) -> dict:
    """How well the data alone determines each direction of parameter space at ``x``.

    From the data-residual Jacobian (no prior term), in parameters scaled by their prior
    scale: the singular values (``condition`` = largest / smallest), the weakest directions
    as parameter combinations, and per-parameter standard errors (residual variance times
    the diagonal of (J^T J)^-1, in parameter units; inf where the data says nothing).
    """
    x = np.asarray(x, dtype=float)
    scale = spec.scale

    def fun(z):
        return _data_residuals(spec.unpack(z * scale), systems)

    z = x / scale
    jac = _jacobian(fun, z)
    r = fun(z)
    u, sv, vt = np.linalg.svd(jac, full_matrices=False)
    dof = max(len(r) - len(x), 1)
    s2 = float(r @ r) / dof
    live = sv > 1e-12 * sv.max()
    inv = (vt[live].T / sv[live] ** 2) @ vt[live]
    se = np.sqrt(np.clip(np.diag(inv) * s2, 0, None)) * scale
    # a parameter with a share in a direction the data does not see is undetermined; the rest keep their errors
    se[np.any(np.abs(vt[~live]) > 1e-8, axis=0)] = np.inf
    weakest = []
    for k in np.argsort(sv)[:3]:
        v = vt[k]
        order = np.argsort(-np.abs(v))[:4]
        weakest.append({"sv": float(sv[k]),
                        "combination": " ".join(f"{v[i]:+.2f}*{spec.names[i]}" for i in order if abs(v[i]) > 0.05)})
    return {
        "singular_values": sv.tolist(),
        "condition": float(sv.max() / max(sv.min(), 1e-300)) if sv.size else float("nan"),
        "weakest": weakest,
        "stderr": dict(zip(spec.names, se.tolist())),
    }


def leave_one_group_out(
    systems: Sequence[FitSystem],
    free: Sequence[str],
    *,
    base: Optional[PolarizationParameters] = None,
    reg_weight: float = 1e-4,
) -> Dict[str, float]:
    """Refit without each group and score the held-out group: RMSE (meV/site) per group and
    overall -- what the fit predicts for compositions it has not seen."""
    base = base or default_params()
    groups = sorted({s.group for s in systems})
    out, sq = {}, []
    for g in groups:
        train = [s for s in systems if s.group != g]
        test = [s for s in systems if s.group == g]
        res = fit(train, ParamSpec(free=free, systems=train, base=base), reg_weight=reg_weight)
        r = _data_residuals(res.params, [replace(s, weight=1.0) for s in test])
        out[g] = float(np.sqrt(np.mean(r ** 2)) * 1000.0) if r.size else float("nan")
        sq.extend(r ** 2)
    out["all"] = float(np.sqrt(np.mean(sq)) * 1000.0) if sq else float("nan")
    return out
