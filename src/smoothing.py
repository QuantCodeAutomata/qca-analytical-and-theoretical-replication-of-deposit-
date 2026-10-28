"""
Appendix E smoothing formulas for C1 branch family.

Works in x = ln(U/O) and uses a smoothstep transition from kink elasticity
η_0 to plateau elasticity η_max over a width w.

Paper parameters (canonical):
  O = 0.80, R_o = 0.05, R_max = 1.00
  η_0 = 1   (elasticity at kink, equal to 1 for the standard proportional ramp)
  L = ln(1/O) = ln(1/0.80) ≈ 0.2231
  A = ln(R_max/R_o) = ln(100/5) ≈ 2.9957

All values in decimal form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Smooth-step function and derivatives
# ---------------------------------------------------------------------------

def smoothstep(t: float | np.ndarray) -> float | np.ndarray:
    """s(t) = 3t^2 - 2t^3  (cubic Hermite smoothstep, s(0)=0, s(1)=1)."""
    t = np.asarray(t, dtype=float)
    return 3.0 * t ** 2 - 2.0 * t ** 3


def smoothstep_deriv(t: float | np.ndarray) -> float | np.ndarray:
    """s'(t) = 6t - 6t^2 = 6t(1-t)."""
    t = np.asarray(t, dtype=float)
    return 6.0 * t * (1.0 - t)


# ---------------------------------------------------------------------------
# Elasticity profile η(x; w)
# ---------------------------------------------------------------------------

def elasticity_profile(
    x: float | np.ndarray,
    eta_0: float,
    eta_max: float,
    w: float,
    L: float,
) -> float | np.ndarray:
    """Piecewise elasticity η(x) defined in Appendix E.

    x = ln(U/O) ∈ [0, L] where L = ln(1/O).

    η(x) = η_0 + (η_max - η_0) * s(x/w)   for 0 ≤ x ≤ w
    η(x) = η_max                             for w < x ≤ L

    Parameters
    ----------
    x      : log-utilization (scalar or array)
    eta_0  : elasticity at kink (x=0)
    eta_max: plateau elasticity
    w      : smoothing width in x-space
    L      : total range L = ln(1/O)
    """
    x = np.asarray(x, dtype=float)
    transition = eta_0 + (eta_max - eta_0) * smoothstep(x / w) if w > 0 else eta_0 * np.ones_like(x)
    if w <= 0:
        return eta_max * np.ones_like(x)
    return np.where(x <= w, transition, eta_max)


# ---------------------------------------------------------------------------
# η_max(w) and its derivative
# ---------------------------------------------------------------------------

def eta_max_from_width(w: float, eta_0: float, L: float, A: float) -> float:
    """η_max(w) = (A - η_0 * w/2) / (L - w/2).

    Derived by enforcing ∫_0^L η(x) dx = A (endpoint constraint).

    Parameters
    ----------
    w    : smoothing width (0 ≤ w ≤ L)
    eta_0: elasticity at kink
    L    : ln(1/O)
    A    : ln(R_max/R_o)
    """
    assert w <= L + 1e-12, f"Width w={w} must not exceed L={L}"
    denom = L - w / 2.0
    if abs(denom) < 1e-15:
        raise ValueError("Denominator L - w/2 is zero.")
    return (A - eta_0 * w / 2.0) / denom


def d_eta_max_dw(w: float, eta_0: float, L: float, A: float) -> float:
    """d η_max / dw = L * (K* - η_0) / [2 * (L - w/2)^2].

    K* = A/L (the pure plateau case, w=0).
    """
    K_star = A / L
    denom = L - w / 2.0
    return L * (K_star - eta_0) / (2.0 * denom ** 2)


def eta_max_at_w0(L: float, A: float) -> float:
    """η_max when w=0 (no smoothing) = A/L = K*."""
    return A / L


def feasibility_width_bound(K: float, K_star: float, eta_0: float, L: float) -> float:
    """Maximum width for a target fixed-share K to remain feasible.

    w ≤ 2L(K - K*) / (K - η_0)

    Parameters
    ----------
    K     : target elasticity (= (1-ᾱ)/ᾱ for protected share ᾱ)
    K_star: endpoint bound A/L
    eta_0 : kink elasticity
    L     : ln(1/O)
    """
    if K <= K_star:
        return 0.0   # no width feasible for K ≤ K*
    return 2.0 * L * (K - K_star) / (K - eta_0)


# ---------------------------------------------------------------------------
# Table 5 replication
# ---------------------------------------------------------------------------

def table5(
    widths: list[float] | None = None,
    O: float = 0.80,
    R_o: float = 0.05,
    R_max: float = 1.00,
    eta_0: float = 1.0,
) -> pd.DataFrame:
    """Reproduce Appendix E Table 5: width → η_max → protected share.

    Parameters
    ----------
    widths: list of smoothing widths w to evaluate
    O, R_o, R_max, eta_0: canonical parameters
    """
    L = np.log(1.0 / O)
    A = np.log(R_max / R_o)
    K_star = A / L

    if widths is None:
        widths = [0.0, 0.05, 0.10, L * (1 - 1e-6), L]

    rows = []
    for w in widths:
        if w >= L:
            # Degenerate: full smoothing, no plateau; η_max = η_0 if w=L
            eta_m = eta_0  # entire range is transition, η_max determined differently
            note = "full-width (degenerate)"
        else:
            try:
                eta_m = eta_max_from_width(w, eta_0, L, A)
            except ValueError:
                eta_m = float("nan")
                note = "infeasible"
            else:
                note = ""
        alpha_safe = 1.0 / (1.0 + eta_m) if not np.isnan(eta_m) else float("nan")
        rows.append({
            "w": w,
            "w/L": w / L,
            "eta_max": eta_m,
            "alpha_safe_%": alpha_safe * 100 if not np.isnan(alpha_safe) else float("nan"),
            "note": note,
        })

    df = pd.DataFrame(rows)
    df["K_star"] = K_star
    df["L"] = L
    df["A"] = A
    return df


# ---------------------------------------------------------------------------
# Rate curve from smoothed elasticity profile
# ---------------------------------------------------------------------------

def rate_from_elasticity_profile(
    U_grid: np.ndarray,
    O: float,
    R_o: float,
    eta_0: float,
    eta_max: float,
    w: float,
) -> np.ndarray:
    """Compute R_b(U) above kink by integrating the elasticity profile.

    d ln R_b = η(x) / U dU = η(x) dx  (since x = ln U/O → dx = dU/U)
    R_b(U) = R_o * exp(∫_0^{ln(U/O)} η(x) dx)

    Parameters
    ----------
    U_grid  : array of utilizations ≥ O
    O, R_o  : kink parameters
    eta_0, eta_max, w: elasticity profile parameters
    """
    L = np.log(1.0 / O)
    results = np.empty_like(U_grid)
    for j, U in enumerate(U_grid):
        if U <= O:
            results[j] = R_o * (U / O)   # ramp branch
            continue
        x = np.log(U / O)
        # Numerical integration ∫_0^x η(t) dt
        def integrand(t: float) -> float:
            return float(elasticity_profile(t, eta_0, eta_max, w, L))
        from scipy import integrate
        val, _ = integrate.quad(integrand, 0.0, x)
        results[j] = R_o * np.exp(val)
    return results
