"""
Admissible rate-curve family theory.

Implements fixed-share, fresh-depositor, and combined protection guarantees,
including boundary curves, admissibility checks, and endpoint bounds.

All quantities in decimal form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate
from src.curves import PiecewiseLinearCurve, SingleKinkCurve


# ---------------------------------------------------------------------------
# Fixed-share protection
# ---------------------------------------------------------------------------

def fixed_share_K(alpha_bar: float) -> float:
    """K = (1 - ᾱ) / ᾱ for a fixed protection share ᾱ."""
    return (1.0 - alpha_bar) / alpha_bar


def fixed_share_alpha(K: float) -> float:
    """ᾱ = 1 / (1 + K)."""
    return 1.0 / (1.0 + K)


def endpoint_K_star(O: float, R_o: float, R_max: float) -> float:
    """K* = ln(R_max/R_o) / ln(1/O).

    Maximum K such that R_K(1) = R_o*(1/O)^K ≤ R_max.
    Equivalently, the largest admissible elasticity for a constant-elasticity
    power branch connecting (O, R_o) to (1, R_max).
    """
    return np.log(R_max / R_o) / np.log(1.0 / O)


def alpha_max(O: float, R_o: float, R_max: float) -> float:
    """α̂_max = 1/(1+K*) — maximum protectable fixed share."""
    K_star = endpoint_K_star(O, R_o, R_max)
    return fixed_share_alpha(K_star)


def fixed_share_boundary_curve(O: float, R_o: float, K: float, U_grid: np.ndarray) -> np.ndarray:
    """R_K(U) = R_o * (U/O)^K  for U ≥ O.

    This is the boundary curve for fixed-share protection ᾱ = 1/(1+K).
    """
    return R_o * (U_grid / O) ** K


def fixed_share_elasticity_envelope(alpha_bar: float) -> float:
    """E(U) = K = (1-ᾱ)/ᾱ  (constant for fixed-share profile)."""
    return fixed_share_K(alpha_bar)


# ---------------------------------------------------------------------------
# Fresh-depositor protection
# ---------------------------------------------------------------------------

def fresh_depositor_profile(U: float | np.ndarray) -> float | np.ndarray:
    """Global sufficient protection profile ᾱ(U) = 1 - U."""
    return 1.0 - np.asarray(U)


def fresh_depositor_envelope(U: float | np.ndarray) -> float | np.ndarray:
    """E(U) = U / (1 - U)  for U < 1."""
    U = np.asarray(U, dtype=float)
    return U / (1.0 - U)


def fresh_depositor_boundary(U: np.ndarray, U_a: float, Rb_a: float) -> np.ndarray:
    """R_b(U) = R_b(U_a) * (1-U_a)/(1-U)  (fresh-depositor boundary curve)."""
    return Rb_a * (1.0 - U_a) / (1.0 - U)


def fresh_depositor_exact_condition(u_0: float, u: float, slope: float, Rb_u: float) -> bool:
    """Check (u_0 - u) * R_b'(u) ≤ R_b(u).

    True if the fresh-depositor exact condition is satisfied at u.
    """
    return (u_0 - u) * slope <= Rb_u


def fresh_depositor_piecewise_check(
    knots_u: list[float],
    knots_r: list[float],
) -> pd.DataFrame:
    """Inspect each adjacent knot pair for fresh-depositor admissibility.

    Rule: R_H / R_L ≤ 1 + (u_H - u_L) / (1 - u_L)
    equivalently: slope m ≤ R_L / (1 - u_L)

    Parameters
    ----------
    knots_u : utilization knots (ascending, decimal form)
    knots_r : borrow-APR knots at each utilization knot

    Returns
    -------
    DataFrame with columns [u_L, u_H, R_L, R_H, ratio, cap, admissible]
    """
    rows = []
    for i in range(len(knots_u) - 1):
        u_L, u_H = knots_u[i], knots_u[i + 1]
        R_L, R_H = knots_r[i], knots_r[i + 1]
        du = u_H - u_L
        cap = 1.0 + du / (1.0 - u_L)   # max admissible ratio R_H/R_L
        ratio = R_H / R_L
        m = (R_H - R_L) / du
        slope_cap = R_L / (1.0 - u_L)
        rows.append({
            "u_L": u_L,
            "u_H": u_H,
            "R_L": R_L,
            "R_H": R_H,
            "m": m,
            "m_cap (R_L/(1-u_L))": slope_cap,
            "R_H/R_L": ratio,
            "cap (1+du/(1-u_L))": cap,
            "admissible": ratio <= cap + 1e-12,
        })
    return pd.DataFrame(rows)


def appendix_a_knots() -> tuple[list[float], list[float]]:
    """Piecewise-linear fresh-depositor example from Appendix A.

    Knots: 0%→5%, 80%→9%, 90%→13.5%, 95%→20%, 98%→32%, 99.5%→56%,
           99.99%→110%, flat to 100%.
    """
    us = [0.00, 0.80, 0.90, 0.95, 0.98, 0.995, 0.9999, 1.00]
    rs = [0.05, 0.09, 0.135, 0.20, 0.32, 0.56,  1.10,   1.10]
    return us, rs


# ---------------------------------------------------------------------------
# Combined guarantee
# ---------------------------------------------------------------------------

def combined_profile(
    U: float | np.ndarray,
    alpha_bar: float,
) -> float | np.ndarray:
    """ᾱ_both(U) = max{ᾱ, 1-U}."""
    U = np.asarray(U, dtype=float)
    return np.maximum(alpha_bar, 1.0 - U)


def combined_envelope(
    U: float | np.ndarray,
    K: float,
) -> float | np.ndarray:
    """E_both(U) = min{K, U/(1-U)}."""
    U = np.asarray(U, dtype=float)
    return np.minimum(K, U / (1.0 - U))


def combined_boundary_F(U: float | np.ndarray, K: float) -> float | np.ndarray:
    """Combined boundary shape function F(U).

    t = K/(K+1)
    F(U) = 1/(1-U)              for 0 ≤ U ≤ t
    F(U) = (K+1) * (U/t)^K     for t < U ≤ 1
    """
    U = np.asarray(U, dtype=float)
    t = K / (K + 1.0)
    below = 1.0 / (1.0 - U)
    above = (K + 1.0) * (U / t) ** K
    return np.where(U <= t, below, above)


def combined_base_bound(K: float, R_max: float) -> float:
    """Base-rate lower bound R_b(0) ≥ R_max * t^K / (K+1).

    t = K/(K+1)
    """
    t = K / (K + 1.0)
    return R_max * t ** K / (K + 1.0)


# ---------------------------------------------------------------------------
# Integral admissibility bound
# ---------------------------------------------------------------------------

def integral_admissibility_bound(
    U: float,
    U_a: float,
    Rb_a: float,
    envelope_fn,
) -> float:
    """Upper bound R_b(U) ≤ R_b(U_a) * exp(∫_{U_a}^{U} E(v)/v dv).

    Parameters
    ----------
    U          : query utilization
    U_a        : anchor utilization
    Rb_a       : borrow APR at anchor
    envelope_fn: callable E(v) → elasticity envelope at v
    """
    integrand = lambda v: envelope_fn(v) / v
    integral, _ = integrate.quad(integrand, U_a, U)
    return Rb_a * np.exp(integral)


# ---------------------------------------------------------------------------
# Section 2 / Canonical example parameters
# ---------------------------------------------------------------------------

O_CANON = 0.80
R_O_CANON = 0.05
R_MAX_CANON = 1.00


def canonical_K_star() -> float:
    """K* ≈ 13.43 for canonical O=0.8, R_o=0.05, R_max=1.00."""
    return endpoint_K_star(O_CANON, R_O_CANON, R_MAX_CANON)


def canonical_alpha_max() -> float:
    """α̂_max ≈ 6.93% for canonical parameters."""
    return alpha_max(O_CANON, R_O_CANON, R_MAX_CANON)
