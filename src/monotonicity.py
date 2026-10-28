"""
Deposit-monotonicity analysis for DeFi lending markets.

Implements the derivative identity dI/dδ = U R_b(U)[1 - α(1+η(U))]
and reproduces the Section 2 single-kink illustrative failure example.

All values are in decimal form (5% → 0.05, 83% → 0.83, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

from src.curves import RateCurve, SingleKinkCurve


# ---------------------------------------------------------------------------
# Core lending-state quantities
# ---------------------------------------------------------------------------

def utilization(B: float, T: float) -> float:
    """U = B / T  (total borrows / total supply)."""
    return B / T


def supply_apr(U: float, Rb: float) -> float:
    """R_s(U) = U * R_b(U)."""
    return U * Rb


def supplier_income(b: float, U: float, Rb: float) -> float:
    """Annualized income I = b * U * R_b(U)."""
    return b * U * Rb


def income_derivative(U: float, Rb: float, alpha: float, eta: float) -> float:
    """dI/dδ = U * R_b(U) * [1 - α*(1 + η(U))].

    This is the exact identity for a marginal additional deposit δ by the
    same supplier while aggregate borrowings B remain fixed.

    Parameters
    ----------
    U    : current utilization B/T
    Rb   : borrow APR at U
    alpha: supplier pool share b/T
    eta  : elasticity η(U) = (U/R_b) * dR_b/dU (using left/below slope)
    """
    return U * Rb * (1.0 - alpha * (1.0 + eta))


def sign_condition_ok(alpha: float, eta: float) -> bool:
    """Return True if dI/dδ ≥ 0, i.e. α*(1+η) ≤ 1."""
    return alpha * (1.0 + eta) <= 1.0


# ---------------------------------------------------------------------------
# Section 2 single-kink baseline and failure example
# ---------------------------------------------------------------------------

# Paper parameters (Section 2)
SEC2_O = 0.80
SEC2_R_o = 0.05
SEC2_R_max = 1.00
SEC2_T_TOTAL = 10_000_000.0   # total supply when at equilibrium utilization
SEC2_B = 8_300_000.0          # total borrowings (fixed during supplier tx)


def build_section2_curve() -> SingleKinkCurve:
    """Standard single-kink curve with Section 2 paper parameters."""
    return SingleKinkCurve(O=SEC2_O, R_o=SEC2_R_o, R_max=SEC2_R_max)


def state_at_supplier_balance(
    b: float,
    B: float,
    T_other: float,
    curve: RateCurve,
) -> dict:
    """Compute lending-state quantities for a supplier balance b.

    The pool has total supply T = T_other + b (other suppliers + this one).
    Borrowings B are fixed during the transaction.

    Parameters
    ----------
    b       : supplier balance
    B       : total pool borrowings (fixed)
    T_other : total supply contributed by all *other* suppliers
    curve   : borrow-rate curve

    Returns
    -------
    dict with keys: b, T, U, alpha, Rb, Rs, I, eta, alpha_safe, dI_ddelta
    """
    T = T_other + b
    U = B / T
    alpha = b / T
    Rb = float(curve.rate(U))
    Rs = supply_apr(U, Rb)
    I = supplier_income(b, U, Rb)
    eta = curve.elasticity(U, use_left=True)
    alpha_safe = 1.0 / (1.0 + eta)
    dI_dd = income_derivative(U, Rb, alpha, eta)
    return {
        "b": b,
        "T": T,
        "U": U,
        "alpha": alpha,
        "Rb": Rb,
        "Rs": Rs,
        "I": I,
        "eta": eta,
        "alpha_safe": alpha_safe,
        "dI_ddelta": dI_dd,
    }


def reproduce_section2_key_states(curve: SingleKinkCurve | None = None) -> pd.DataFrame:
    """Reproduce the three key supplier-balance states from Section 2.

    States:
      1. Initial: b = 1,000,000 (U ≈ 83%, above kink)
      2. Post-deposit: b = 1,375,000, total supply increased by 375,000
         → utilization returns to kink O = 80%
      3. Post-withdrawal of ~302,000: b = 698,000, higher income

    Borrowings are fixed at B = 8,300,000.

    Returns
    -------
    pd.DataFrame with one row per state
    """
    if curve is None:
        curve = build_section2_curve()

    B = SEC2_B

    # State 1: initial equilibrium
    b1 = 1_000_000.0
    # Total supply T_1: given B = 8.3M and b_1 = 1M, U = 83% → T_1 = B/0.83 ≈ 10.0M
    T1 = B / 0.83
    T_other = T1 - b1

    s1 = state_at_supplier_balance(b1, B, T_other, curve)

    # State 2: after depositing 375,000 → T increases by 375,000
    b2 = b1 + 375_000.0
    s2 = state_at_supplier_balance(b2, B, T_other, curve)

    # State 3: after *withdrawing* ~302,000 from state 1
    b3 = b1 - 302_000.0
    s3 = state_at_supplier_balance(b3, B, T_other, curve)

    rows = [s1, s2, s3]
    labels = ["initial (b=1M)", "post-deposit (+375k)", "post-withdrawal (-302k)"]
    df = pd.DataFrame(rows)
    df.insert(0, "state", labels)
    return df


def sweep_supplier_balance(
    B: float = SEC2_B,
    T_other: float | None = None,
    curve: RateCurve | None = None,
    n_points: int = 500,
) -> pd.DataFrame:
    """Sweep supplier balance b over a feasible range and compute lending state.

    Feasible range: b > 0 and U = B/(T_other+b) < 1  (b > B - T_other).
    Also ensures U > 0 (b > 0).

    Parameters
    ----------
    B        : fixed total borrowings
    T_other  : supply from all other suppliers; defaults to T1 - b1 from Sec 2
    curve    : rate curve; defaults to Section 2 single-kink curve
    n_points : number of sweep points

    Returns
    -------
    pd.DataFrame with columns [b, U, alpha, Rb, Rs, I, eta, alpha_safe, dI_ddelta]
    """
    if curve is None:
        curve = build_section2_curve()
    if T_other is None:
        T1 = B / 0.83
        T_other = T1 - 1_000_000.0

    # Minimum b: must have T = T_other + b > B (so U < 1)
    b_min = max(1.0, B - T_other + 1.0)
    # Maximum b: practical ceiling, e.g. 20M
    b_max = 20_000_000.0

    bs = np.linspace(b_min, b_max, n_points)
    records = [state_at_supplier_balance(float(b), B, T_other, curve) for b in bs]
    return pd.DataFrame(records)


def compute_elasticity_diagnostics(
    curve: SingleKinkCurve | None = None,
    Us: list[float] | None = None,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Compute elasticity and monotonicity diagnostics at selected utilizations.

    Reproduces representative values from the paper's steepness discussion
    (e.g., η ≈ 20.5 at U = 83%, α(1+η) ≈ 2.15 for α = 10%).
    """
    if curve is None:
        curve = build_section2_curve()
    if Us is None:
        Us = [0.80, 0.8001, 0.83, 1.00]

    rows = []
    for U in Us:
        Rb = float(curve.rate(U))
        eta_left = curve.elasticity(U, use_left=True)
        eta_right = curve.elasticity(U, use_left=False)
        alpha_safe = 1.0 / (1.0 + eta_left)
        alpha_1_eta = alpha * (1.0 + eta_left)
        rows.append({
            "U": U,
            "Rb": Rb,
            "eta_left": eta_left,
            "eta_right": eta_right,
            "alpha_safe_%": alpha_safe * 100,
            "alpha*(1+eta) [alpha=10%]": alpha_1_eta,
            "dI_ddelta_positive": alpha_1_eta <= 1.0,
        })
    return pd.DataFrame(rows)
