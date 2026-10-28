"""
Single-kink power repair construction for deposit-monotonicity enforcement.

Given a standard single-kink curve and a target fixed-share K, constructs
the minimal-change repaired curve that enforces η(U) ≤ K on the replaced
interval (O, U_c) by using a power branch R_o*(U/O)^K, then rejoins the
original steep line at the catch-up utilization U_c.

All values in decimal form.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from src.curves import SingleKinkCurve, RepairedSingleKinkCurve
from src.admissibility import fixed_share_K


def find_catchup_utilization(
    O: float,
    R_o: float,
    R_max: float,
    K: float,
    bracket: tuple[float, float] = None,
) -> float:
    """Find U_c such that R_o*(U_c/O)^K = R_o + m*(U_c - O).

    U_c is the first utilization above O where the power branch meets
    the original steep linear branch.

    Parameters
    ----------
    O    : kink utilization
    R_o  : borrow APR at kink
    R_max: maximum borrow APR
    K    : power exponent (= (1-ᾱ)/ᾱ)
    bracket: search bracket; defaults to (O+1e-9, 1.0)

    Returns
    -------
    U_c : catch-up utilization ∈ (O, 1]
    """
    m = (R_max - R_o) / (1.0 - O)

    def f(U: float) -> float:
        power_branch = R_o * (U / O) ** K
        steep_branch = R_o + m * (U - O)
        return power_branch - steep_branch

    if bracket is None:
        lo, hi = O + 1e-9, 1.0 - 1e-9

    # Check that f changes sign on [lo, hi]
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi > 0:
        # No crossing: power branch always above steep branch — return 1.0
        return 1.0

    U_c = brentq(f, lo, hi, xtol=1e-12, rtol=1e-12)
    return float(U_c)


def build_repaired_curve(
    O: float = 0.80,
    R_o: float = 0.05,
    R_max: float = 1.00,
    alpha_bar: float = 0.05,
) -> tuple[RepairedSingleKinkCurve, float, float]:
    """Construct the minimal-change power repair for a target protection share.

    Parameters
    ----------
    O        : kink utilization
    R_o      : borrow APR at kink
    R_max    : maximum borrow APR
    alpha_bar: target fixed-share protection (e.g. 0.05 for 5%)

    Returns
    -------
    (repaired_curve, K, U_c)
    """
    K = fixed_share_K(alpha_bar)
    U_c = find_catchup_utilization(O, R_o, R_max, K)
    curve = RepairedSingleKinkCurve(O=O, R_o=R_o, R_max=R_max, K=K, U_c=U_c)
    return curve, K, U_c


def income_derivative_at_U(
    U: float,
    curve,
    alpha: float,
) -> float:
    """Compute dI/dδ = U * R_b(U) * [1 - α*(1+η(U))]."""
    Rb = float(curve.rate(U))
    eta = curve.elasticity(U, use_left=True)
    return U * Rb * (1.0 - alpha * (1.0 + eta))


def verify_repair_protects(
    repaired: RepairedSingleKinkCurve,
    alpha_bar: float,
    U_test_points: list[float] | None = None,
) -> bool:
    """Return True if α_safe(U) ≥ ᾱ for all tested points on repaired branch."""
    if U_test_points is None:
        U_test_points = list(np.linspace(repaired.O + 1e-6, repaired.U_c - 1e-6, 200))
    ok = True
    for U in U_test_points:
        alpha_safe = repaired.alpha_safe(U)
        if alpha_safe < alpha_bar - 1e-8:
            ok = False
            break
    return ok
