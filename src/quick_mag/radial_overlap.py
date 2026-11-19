"""Metal-d / ligand-p two-centre overlaps of Slater-type orbitals: the frozen range of the exchange kernel.

The polarization a metal puts on a bridging ligand scales as the square of their transfer, and the transfer as their
overlap along each channel (sigma: m = 0, pi: |m| = 1 about the bond). With Slater-type orbitals

    phi(r) = r^(n-1) exp(-zeta r) Y_lm,

the overlap's shape in R is fixed by the two exponents; its size is absorbed into the fitted amplitudes, so only
the *ratio* to a reference distance is used (``sto_overlap(..., normalized=True)`` gives true overlaps, for d-d
direct exchange):

    d_m(R) = (S_m(R) / S_m(R0))^2       (m = sigma, pi; R0 = the Shannon M-L distance, as for the other range models)

Exponents (:func:`slater_zeta`) follow Slater's rules for the ion's configuration (the electron-configuration
tables of quick-mag): a free-ion estimate, chosen to be tested against strain data rather than trusted.

Overlaps are computed in prolate spheroidal coordinates (A at z = -R/2, B at z = +R/2, lambda = (r_a + r_b)/R,
mu = (r_a - r_b)/R) with Gauss-Laguerre in lambda - 1 and Gauss-Legendre in mu: the integrand is a polynomial times
an exponential, so modest grids are exact to ~1e-12 (checked against the closed-form 1s-1s overlap in the tests).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np

__all__ = ["slater_zeta", "sto_overlap", "pd_overlaps", "pd_damping"]

# Slater's effective principal quantum numbers
_N_EFF = {1: 1.0, 2: 2.0, 3: 3.0, 4: 3.7, 5: 4.0, 6: 4.2}
_L = {"s": 0, "p": 1, "d": 2, "f": 3}


def _groups(config):
    """Slater groups in order: (1s)(2s,2p)(3s,3p)(3d)(4s,4p)(4d)(4f)(5s,5p)... with their electron counts."""
    counts = {}
    for n, orb, c in config:
        key = (n, "sp") if orb in ("s", "p") else (n, orb)
        counts[key] = counts.get(key, 0) + c
    order = []
    for n in range(1, 8):                 # (1s)(2s,2p)(3s,3p)(3d)(4s,4p)(4d)(4f)...: by n, then sp < d < f
        for kind in ("sp", "d", "f"):
            if (n, kind) in counts:
                order.append(((n, kind), counts[(n, kind)]))
    return order


def slater_zeta(element: str, oxidation_state: int, shell: str) -> Tuple[int, float]:
    """``(n, zeta)`` (bohr^-1) of the ion's outermost ``shell`` ("d" or "p") by Slater's rules.

    Fe3+ 3d: (3, 2.20); O2- 2p: (2, 1.925). ``n`` is the integer principal quantum number used for the STO's
    radial power; ``zeta`` uses Slater's effective n*.
    """
    from quick_mag.electron_configurations import get_ionic_electron_configuration

    Z = sum(c for _n, _o, c in get_ionic_electron_configuration(element, 0))
    config = get_ionic_electron_configuration(element, int(oxidation_state))
    l_want = _L[shell]
    shells = [(n, orb, c) for n, orb, c in config if _L[orb] == l_want and c > 0]
    if not shells:
        if shell == "d" and oxidation_state > 0:
            # an empty d shell (d0): the orbital an added electron would occupy, screened as in the d1 ion
            return slater_zeta(element, int(oxidation_state) - 1, shell)
        raise ValueError(f"{element}{int(oxidation_state):+d} has no occupied {shell} shell")
    n_t = max(n for n, _o, _c in shells)
    target = (n_t, "sp") if shell in ("s", "p") else (n_t, shell)
    groups = _groups(config)
    keys = [k for k, _ in groups]
    idx = keys.index(target)
    sigma = 0.0
    for k, (key, count) in enumerate(groups):
        if k == idx:
            sigma += (count - 1) * (0.30 if key == (1, "sp") else 0.35)
        elif k < idx:
            # s/p electrons: 0.85 from shell n-1, 1.00 from deeper; d/f electrons: 1.00 from every group to the left
            sigma += (0.85 if target[1] == "sp" and key[0] == n_t - 1 else 1.00) * count
    return n_t, (Z - sigma) / _N_EFF[n_t]


@lru_cache(maxsize=8)
def _grids(n_lag: int, n_leg: int):
    x, wx = np.polynomial.laguerre.laggauss(n_lag)
    u, wu = np.polynomial.legendre.leggauss(n_leg)
    return x, wx, u, wu


# real-harmonic angular normalization of the factors used below: theta part x cos(m phi), integrated over the sphere
_ANG_NORM = {(0, 0): 1.0 / math.sqrt(4 * math.pi),
             (1, 0): math.sqrt(3 / (4 * math.pi)), (1, 1): math.sqrt(3 / (4 * math.pi)),
             (2, 0): math.sqrt(5 / (16 * math.pi)), (2, 1): math.sqrt(15 / (4 * math.pi)),
             (2, 2): math.sqrt(15 / (16 * math.pi))}


def sto_overlap(na, la, za, nb, lb, zb, R, m: int = 0, *, normalized: bool = False, n_lag: int = 48,
                n_leg: int = 48) -> np.ndarray:
    """Overlap of STO (na, la, m) on A and (nb, lb, m) on B at separations ``R`` (bohr, array), both quantized along
    the same axis A -> B (the Slater-Koster convention: d_sigma-d_sigma > 0, d_pi-d_pi < 0, d_delta-d_delta > 0 at
    large R). Angular factors: s 1; p_sigma cos t, p_pi sin t; d_sigma 3cos^2 t - 1, d_pi sin t cos t,
    d_delta sin^2 t (each times cos(m phi)).

    ``normalized=False`` (the default) leaves out every constant, so only ratios at different R mean anything (the
    p-d range law). ``normalized=True`` gives the true overlap of normalized orbitals, comparable across channels
    and exponents (S -> 1 as R -> 0 for identical orbitals)."""
    m = abs(int(m))
    if la > 2 or lb > 2:
        raise ValueError("l <= 2")
    if m > min(la, lb):
        raise ValueError(f"m = {m} needs l >= {m} on both centres (got {la}, {lb}); the overlap would be zero")
    R = np.atleast_1d(np.asarray(R, dtype=float))
    x, wx, u, wu = _grids(n_lag, n_leg)
    h = R[:, None, None] / 2.0
    p = h * (za + zb)                                # exp(-p lambda) = exp(-p) exp(-p t)
    lam = 1.0 + x[None, :, None] / p                 # (R, lag, 1)
    mu = u[None, None, :]
    ra, rb = h * (lam + mu), h * (lam - mu)
    cos_a = (lam * mu + 1.0) / (lam + mu)
    cos_b = (lam * mu - 1.0) / (lam - mu)
    rho = h * np.sqrt(np.clip((lam ** 2 - 1.0) * (1.0 - mu ** 2), 0.0, None))
    sin_a, sin_b = rho / ra, rho / rb

    def ang(l, c, s):
        if l == 0:
            return np.ones_like(c)
        if l == 1:
            return c if m == 0 else s
        return (3 * c * c - 1, s * c, s * s)[m]

    f = ra ** (na - 1) * rb ** (nb - 1) * ang(la, cos_a, sin_a) * ang(lb, cos_b, sin_b)
    f = f * np.exp(-h * (za - zb) * mu)              # the remaining exponential: exp(-(za ra + zb rb)) / exp(-p lam)
    jac = h ** 3 * (lam ** 2 - mu ** 2)
    integral = np.einsum("rij,j,i->r", f * jac / p, wu, wx) * np.exp(-p[:, 0, 0])
    if normalized:
        radial = ((2 * za) ** (na + 0.5) / math.sqrt(math.factorial(2 * na))
                  * (2 * zb) ** (nb + 0.5) / math.sqrt(math.factorial(2 * nb)))
        integral = integral * radial * _ANG_NORM[(la, m)] * _ANG_NORM[(lb, m)] * (2 * math.pi if m == 0 else math.pi)
    return integral


def pd_overlaps(nd, zd, np_, zp, R) -> Tuple[np.ndarray, np.ndarray]:
    """(sigma, pi) overlap shapes of a metal d STO and a ligand p STO at separations R (bohr)."""
    s = sto_overlap(nd, 2, zd, np_, 1, zp, R, m=0)
    p = sto_overlap(nd, 2, zd, np_, 1, zp, R, m=1)
    return s, p


BOHR_ANG = 0.529177210903


def pd_damping(nd, zd, np_, zp, r_ang, r0_ang) -> Tuple[np.ndarray, np.ndarray]:
    """(d_sigma, d_pi) = (S(r)/S(r0))^2 per channel; r and r0 in Angstrom (arrays of equal length, or r0 scalar)."""
    r = np.atleast_1d(np.asarray(r_ang, dtype=float)) / BOHR_ANG
    r0 = np.broadcast_to(np.asarray(r0_ang, dtype=float), r.shape) / BOHR_ANG
    s, p = pd_overlaps(nd, zd, np_, zp, r)
    s0, p0 = pd_overlaps(nd, zd, np_, zp, r0)
    return (s / s0) ** 2, (p / p0) ** 2
