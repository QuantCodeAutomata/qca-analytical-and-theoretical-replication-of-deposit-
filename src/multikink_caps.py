"""
Multiple-kink fixed-kink repair: segment-cap admissibility checks.

For each adjacent pair (u_L, R_L) → (u_H, R_H) on a piecewise-linear
borrow-rate curve, computes admissibility caps for:
  - fresh-depositor guarantee
  - fixed-share guarantee with parameter K
  - combined guarantee (both simultaneously)

All values in decimal form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fresh_cap(u_L: float, u_H: float) -> float:
    """Cap c^fresh = 1 + (u_H - u_L) / (1 - u_L).

    A segment ratio R_H/R_L must not exceed this to satisfy the
    fresh-depositor guarantee on the fixed-kink grid.
    """
    du = u_H - u_L
    return 1.0 + du / (1.0 - u_L)


def fixed_share_cap(u_L: float, u_H: float, K: float) -> float:
    """Cap c^fixed = min{1 + K*Δu/u_L,  u_H / (u_H - K*Δu)}.

    Denominators ≤ 0 or conditions that are nonrestrictive are handled by
    returning infinity (no restriction) for that term.
    """
    du = u_H - u_L

    # Term 1: 1 + K*Δu / u_L
    if u_L <= 0:
        term1 = np.inf
    else:
        term1 = 1.0 + K * du / u_L

    # Term 2: u_H / (u_H - K*Δu)
    denom2 = u_H - K * du
    if denom2 <= 0:
        term2 = np.inf   # nonrestrictive (power branch can't reach u_H with this K)
    else:
        term2 = u_H / denom2

    cap = min(term1, term2)
    return float(cap)


def combined_cap(u_L: float, u_H: float, K: float) -> float:
    """Combined cap = min(c^fresh, c^fixed)."""
    return min(fresh_cap(u_L, u_H), fixed_share_cap(u_L, u_H, K))


def segment_admissibility_table(
    knots_u: list[float],
    knots_r: list[float],
    K: float,
    guarantee: str = "both",
) -> pd.DataFrame:
    """Compute admissibility caps and violations for each segment.

    Parameters
    ----------
    knots_u  : utilization knots (ascending)
    knots_r  : borrow APR at each knot
    K        : fixed-share parameter (1-ᾱ)/ᾱ
    guarantee: 'fresh', 'fixed', or 'both'

    Returns
    -------
    DataFrame with columns per segment
    """
    rows = []
    for i in range(len(knots_u) - 1):
        u_L, u_H = knots_u[i], knots_u[i + 1]
        R_L, R_H = knots_r[i], knots_r[i + 1]
        ratio = R_H / R_L

        c_f = fresh_cap(u_L, u_H)
        c_k = fixed_share_cap(u_L, u_H, K)
        c_b = combined_cap(u_L, u_H, K)

        if guarantee == "fresh":
            cap = c_f
        elif guarantee == "fixed":
            cap = c_k
        else:
            cap = c_b

        rows.append({
            "u_L": u_L,
            "u_H": u_H,
            "R_L": R_L,
            "R_H": R_H,
            "R_H/R_L": ratio,
            "c_fresh": c_f,
            "c_fixed": c_k,
            "c_both": c_b,
            "cap (selected)": cap,
            "admissible": ratio <= cap + 1e-10,
            "violation": max(0.0, ratio - cap),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table 4 USDT knot grid
# ---------------------------------------------------------------------------

def usdt_knots() -> tuple[list[float], list[float]]:
    """USDT curve from Table 4.

    Knots: 0%→0.5%, 80%→5.25%, 85%→9.4875%, 90%→17.9625%,
           95%→30.675%, 99%→47.625%, 99.5%→68.8125%, 100%→90%
    """
    us = [0.00, 0.80, 0.85, 0.90, 0.95, 0.99, 0.995, 1.00]
    rs = [0.005, 0.0525, 0.094875, 0.179625, 0.30675, 0.47625, 0.688125, 0.90]
    return us, rs
