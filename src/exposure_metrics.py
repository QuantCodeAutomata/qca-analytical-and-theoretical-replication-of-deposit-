"""
Exposure metrics for Drift supplier-exposure study (exp_2).

Computes:
  - protected share α_safe(U) = 1/(1+η(U)) for each market-hour
  - supplier share = supplier_balance / total_deposits
  - exposure flag: supplier_share > α_safe(U)

Uses the below-kink slope convention for deposit monotonicity.
All values in decimal form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.curves import PiecewiseLinearCurve


# ---------------------------------------------------------------------------
# Rate curve construction from Drift config
# ---------------------------------------------------------------------------

def rate_curve_from_config(
    optimal_util: float,
    min_borrow_rate: float,
    optimal_borrow_rate: float,
    max_borrow_rate: float,
) -> PiecewiseLinearCurve:
    """Build a two-segment PiecewiseLinearCurve from Drift-style config.

    Drift uses a two-kink (or single-kink) curve:
      [0, optimal_util] → [min_borrow_rate, optimal_borrow_rate]
      [optimal_util, 1] → [optimal_borrow_rate, max_borrow_rate]

    Parameters
    ----------
    optimal_util       : kink utilization (O)
    min_borrow_rate    : borrow APR at U=0 (or slope-start)
    optimal_borrow_rate: borrow APR at kink
    max_borrow_rate    : borrow APR at U=1
    """
    us = [0.0, optimal_util, 1.0]
    rs = [min_borrow_rate, optimal_borrow_rate, max_borrow_rate]
    return PiecewiseLinearCurve(us, rs)


def alpha_safe_from_state(
    total_borrows: float,
    total_supply: float,
    optimal_util: float,
    min_borrow_rate: float,
    optimal_borrow_rate: float,
    max_borrow_rate: float,
) -> tuple[float, float, float]:
    """Compute U, η(U), and α_safe(U) from a market state.

    Returns (U, eta, alpha_safe)
    """
    if total_supply <= 0:
        return 0.0, 0.0, 1.0
    U = min(total_borrows / total_supply, 1.0)
    curve = rate_curve_from_config(
        optimal_util, min_borrow_rate, optimal_borrow_rate, max_borrow_rate
    )
    eta = curve.elasticity(U, use_left=True)
    alpha_safe = 1.0 / (1.0 + eta)
    return U, eta, alpha_safe


# ---------------------------------------------------------------------------
# Hourly exposure computation
# ---------------------------------------------------------------------------

def compute_hourly_exposure(
    market_df: pd.DataFrame,
    supplier_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute exposure flags for all suppliers at each hour.

    Parameters
    ----------
    market_df   : DataFrame with columns [hour, U, alpha_safe, total_supply]
                  one row per hour
    supplier_df : DataFrame with columns [hour, supplier_id, balance]

    Returns
    -------
    DataFrame with columns [hour, supplier_id, balance, share, alpha_safe, exposed]
    """
    merged = supplier_df.merge(market_df[["hour", "alpha_safe", "total_supply"]], on="hour")
    merged["share"] = merged["balance"] / merged["total_supply"]
    merged["exposed"] = merged["share"] > merged["alpha_safe"]
    return merged


# ---------------------------------------------------------------------------
# Market-level diagnostics (reproduce Table 2 style)
# ---------------------------------------------------------------------------

def market_diagnostics(
    hourly_df: pd.DataFrame,
    optimal_util: float = 0.80,
    alpha_safe_threshold: float = 0.05,
) -> dict:
    """Compute market-level diagnostic fractions.

    Parameters
    ----------
    hourly_df : DataFrame with columns [hour, U, alpha_safe]
    optimal_util: kink/threshold for "above kink" definition
    alpha_safe_threshold: threshold for α_safe < X% flag

    Returns
    -------
    dict with diagnostic keys
    """
    total_hours = len(hourly_df)
    above_kink = (hourly_df["U"] > optimal_util).sum()
    alpha_safe_below = (hourly_df["alpha_safe"] < alpha_safe_threshold).sum()
    above_kink_mask = hourly_df["U"] > optimal_util
    median_alpha_above = hourly_df.loc[above_kink_mask, "alpha_safe"].median() if above_kink > 0 else float("nan")

    return {
        "total_hours": total_hours,
        "above_kink_pct": 100.0 * above_kink / total_hours,
        "alpha_safe_below_threshold_pct": 100.0 * alpha_safe_below / total_hours,
        "median_alpha_safe_above_kink_pct": 100.0 * median_alpha_above if not np.isnan(median_alpha_above) else float("nan"),
    }


def exposure_fractions(exposure_df: pd.DataFrame) -> dict:
    """Compute supplier-level exposure fractions.

    Parameters
    ----------
    exposure_df: DataFrame from compute_hourly_exposure

    Returns
    -------
    dict with fraction of hours with at least one lender exposed
    """
    hours = exposure_df["hour"].unique()
    total_hours = len(hours)
    hours_with_any = exposure_df.groupby("hour")["exposed"].any().sum()
    return {
        "total_hours": total_hours,
        "hours_with_any_exposed": int(hours_with_any),
        "pct_hours_with_any_exposed": 100.0 * hours_with_any / total_hours,
    }
