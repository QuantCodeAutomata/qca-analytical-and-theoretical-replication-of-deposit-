"""
Fixed-kink repairs for multiple-kink borrow-rate curves.

Constructs four repair strategies:
1. Floor repair   — minimum base rate for admissibility (propagate caps downward)
2. Raise-only     — hold top endpoint, raise lower knots minimally
3. Lower-only     — hold base endpoint, lower upper knots maximally
4. Mixed repair   — constrained quadratic minimization (requires cvxpy)

All values in decimal form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.multikink_caps import fresh_cap, fixed_share_cap, combined_cap


# ---------------------------------------------------------------------------
# Helper: get cap for each segment
# ---------------------------------------------------------------------------

def _caps_array(
    us: list[float],
    K: float,
    guarantee: str = "both",
) -> np.ndarray:
    """Return cap c_i for each segment i = 0..n-1."""
    caps = []
    for i in range(len(us) - 1):
        u_L, u_H = us[i], us[i + 1]
        if guarantee == "fresh":
            c = fresh_cap(u_L, u_H)
        elif guarantee == "fixed":
            c = fixed_share_cap(u_L, u_H, K)
        else:
            c = combined_cap(u_L, u_H, K)
        caps.append(c)
    return np.array(caps)


# ---------------------------------------------------------------------------
# 1. Floor repair
# ---------------------------------------------------------------------------

def floor_repair(
    us: list[float],
    rs_top: float,
    K: float,
    guarantee: str = "both",
) -> np.ndarray:
    """Minimum base rate achievable with fixed top endpoint rs_top.

    Propagates backward: r_{n} = rs_top, r_{i} = r_{i+1} / c_i.
    This gives the minimum feasible knot vector.

    Returns
    -------
    np.ndarray of length len(us) — minimum admissible rates
    """
    n = len(us) - 1  # number of segments
    caps = _caps_array(us, K, guarantee)
    r_floor = np.empty(n + 1)
    r_floor[n] = rs_top
    for i in range(n - 1, -1, -1):
        r_floor[i] = r_floor[i + 1] / caps[i]
    return r_floor


# ---------------------------------------------------------------------------
# 2. Raise-only repair
# ---------------------------------------------------------------------------

def raise_only_repair(
    us: list[float],
    rs_baseline: list[float],
    K: float,
    guarantee: str = "both",
) -> np.ndarray:
    """Hold top endpoint, raise lower knots only as needed.

    r_n^+ = r_n^0  (top endpoint preserved)
    r_i^+ = max{r_i^0, r_{i+1}^+ / c_i}

    Returns
    -------
    np.ndarray of raised rates
    """
    n = len(us) - 1
    caps = _caps_array(us, K, guarantee)
    r = np.array(rs_baseline, dtype=float)
    for i in range(n - 1, -1, -1):
        r[i] = max(r[i], r[i + 1] / caps[i])
    return r


# ---------------------------------------------------------------------------
# 3. Lower-only repair
# ---------------------------------------------------------------------------

def lower_only_repair(
    us: list[float],
    rs_baseline: list[float],
    K: float,
    guarantee: str = "both",
) -> np.ndarray:
    """Hold base endpoint, lower upper knots only as needed.

    r_0^- = r_0^0  (base endpoint preserved)
    r_{i+1}^- = min{r_{i+1}^0, c_i * r_i^-}

    Returns
    -------
    np.ndarray of lowered rates
    """
    n = len(us) - 1
    caps = _caps_array(us, K, guarantee)
    r = np.array(rs_baseline, dtype=float)
    for i in range(n):
        r[i + 1] = min(r[i + 1], caps[i] * r[i])
    return r


# ---------------------------------------------------------------------------
# 4. Mixed repair (constrained quadratic)
# ---------------------------------------------------------------------------

def mixed_repair(
    us: list[float],
    rs_baseline: list[float],
    K: float,
    guarantee: str = "both",
    fix_base: bool = True,
    fix_top: bool = True,
) -> np.ndarray:
    """Constrained quadratic repair: minimize weighted squared deviations.

    Objective: Σ_i w_i * (r_i - r_i^0)^2
    Constraints: r_{i+1} ≤ c_i * r_i  for all segments i
    with optional fixed endpoints.

    Weights: trapezoidal utilization interval widths w_i = (u_{i+1} - u_i)/2 + ...
    using equal contribution from adjacent segments.

    Requires cvxpy; falls back to raise-only if not available.

    Returns
    -------
    np.ndarray of optimized rates
    """
    try:
        import cvxpy as cp
    except ImportError:
        # Graceful fallback
        return raise_only_repair(us, rs_baseline, K, guarantee)

    us_arr = np.array(us, dtype=float)
    r0 = np.array(rs_baseline, dtype=float)
    n = len(us_arr) - 1
    caps = _caps_array(us, K, guarantee)

    # Trapezoidal weights: each knot shares half of its adjacent segment widths
    widths = np.diff(us_arr)
    w = np.zeros(n + 1)
    w[0] = widths[0] / 2.0
    w[-1] = widths[-1] / 2.0
    for i in range(1, n):
        w[i] = (widths[i - 1] + widths[i]) / 2.0

    r = cp.Variable(n + 1)
    objective = cp.Minimize(cp.sum(cp.multiply(w, cp.square(r - r0))))

    constraints = [r[i + 1] <= caps[i] * r[i] for i in range(n)]
    constraints += [r >= 0]
    if fix_base:
        constraints.append(r[0] == r0[0])
    if fix_top:
        constraints.append(r[n] == r0[n])

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        prob.solve(verbose=False)

    if r.value is not None:
        return np.maximum(r.value, 0.0)

    # Infeasible with fix_base=True — relax base endpoint and retry
    if fix_base:
        constraints_relaxed = [r[i + 1] <= caps[i] * r[i] for i in range(n)]
        constraints_relaxed += [r >= 0]
        if fix_top:
            constraints_relaxed.append(r[n] == r0[n])
        prob2 = cp.Problem(objective, constraints_relaxed)
        try:
            prob2.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob2.solve(verbose=False)
        if r.value is not None:
            return np.maximum(r.value, 0.0)

    # Final fallback: raise-only
    return raise_only_repair(us, rs_baseline, K, guarantee)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def repair_summary_table(
    us: list[float],
    rs_baseline: list[float],
    K: float,
    guarantee: str = "both",
) -> pd.DataFrame:
    """Build a side-by-side comparison of all four repair strategies."""
    r_base = np.array(rs_baseline)
    r_floor_b = floor_repair(us, float(rs_baseline[-1]), K, guarantee)
    r_raise = raise_only_repair(us, rs_baseline, K, guarantee)
    r_lower = lower_only_repair(us, rs_baseline, K, guarantee)
    r_mixed = mixed_repair(us, rs_baseline, K, guarantee)

    df = pd.DataFrame({
        "u": us,
        "baseline": r_base,
        "floor": r_floor_b,
        "raise_only": r_raise,
        "lower_only": r_lower,
        "mixed": r_mixed,
    })
    return df
