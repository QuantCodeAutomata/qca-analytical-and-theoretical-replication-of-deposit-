"""
Experiment 2: Empirical replication of Drift SOL and USDC supplier-exposure study.

NOTE: Real Drift onchain data is required for a full replication.  This script
provides:
  - The complete analytical pipeline (curve parsing, exposure metrics, LOCF
    hourly state construction, balance propagation)
  - Synthetic data generation mimicking the paper's reported statistics, for
    structural and pipeline validation
  - Placeholder targets from Table 2 for comparison

When real data is available, replace `generate_synthetic_market_states` and
the synthetic supplier DataFrame with actual ingested data.

All values in decimal form; rates as APR decimals.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.curves import PiecewiseLinearCurve
from src.exposure_metrics import (
    rate_curve_from_config, alpha_safe_from_state,
    compute_hourly_exposure, market_diagnostics, exposure_fractions,
)
from src.drift_market import (
    generate_synthetic_market_states_two_period, propagate_balance,
)

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

# Paper Table 2 targets (approximate)
TABLE2_TARGETS = {
    "SOL": {
        "above_kink_pct_full":       5.4,
        "alpha_safe_lt_5pct_full":   0.3,
        "median_alpha_safe_above":   9.9,
        "above_kink_pct_obs":       12.4,
        "lender_exposed_pct":       11.4,
        "supplier_exposed_pct":     11.4,
    },
    "USDC": {
        "above_kink_pct_full":      36.6,
        "alpha_safe_lt_5pct_full":   3.0,
        "median_alpha_safe_above":   8.4,
        "above_kink_pct_obs":       26.0,
        "lender_exposed_pct":        7.9,
        "supplier_exposed_pct":     11.6,
    },
}


# ---------------------------------------------------------------------------
# 1. Market state pipeline
# ---------------------------------------------------------------------------

def build_market_hourly(market: str, start: str, end: str) -> pd.DataFrame:
    """Build hourly market states with utilization and α_safe for a Drift market.

    Uses synthetic data calibrated to match the paper's per-period statistics.
    Replace with real Drift onchain data for a full empirical replication.
    """
    df = generate_synthetic_market_states_two_period(
        market, full_start=start, obs_start="2025-12-01", full_end=end
    )

    # Vectorized α_safe computation using below-kink slope convention
    O_arr  = df["optimal_util"].values
    R_o_arr= df["optimal_borrow_rate"].values
    R_max_arr = df["max_borrow_rate"].values
    U_arr  = df["U"].values

    # Piecewise-linear two-segment curve: slope below = R_o/O, slope above = (R_max-R_o)/(1-O)
    m_below = R_o_arr / O_arr
    m_above = (R_max_arr - R_o_arr) / (1.0 - O_arr)

    # Apply below-kink convention: always use slope of the segment below current U
    # At U ≤ O: below slope; at U > O: above-kink segment slope
    slope = np.where(U_arr <= O_arr, m_below, m_above)
    Rb = np.where(U_arr <= O_arr,
                  R_o_arr * (U_arr / O_arr),
                  R_o_arr + m_above * (U_arr - O_arr))
    Rb = np.where(Rb <= 0, 1e-9, Rb)
    eta = U_arr * slope / Rb
    alpha_safe = 1.0 / (1.0 + eta)

    df = df.copy()
    df["alpha_safe"] = alpha_safe
    df["eta"] = eta
    df["hour"] = df["timestamp"]
    return df


def validate_curve_against_charged(hourly_df: pd.DataFrame) -> dict:
    """Compare reconstructed borrow APR against protocol-charged APR."""
    O = hourly_df["optimal_util"].values
    R_o = hourly_df["optimal_borrow_rate"].values
    R_max = hourly_df["max_borrow_rate"].values
    U = hourly_df["U"].values
    m_above = (R_max - R_o) / (1.0 - O)
    curve_apr = np.where(U <= O, R_o * (U / O), R_o + m_above * (U - O))
    diffs = np.abs(curve_apr - hourly_df["charged_borrow_apr"].values)
    return {
        "mean_abs_diff": float(diffs.mean()),
        "pct_within_1pct": float((diffs < 0.01).mean() * 100),
        "max_diff": float(diffs.max()),
    }


# ---------------------------------------------------------------------------
# 2. Synthetic supplier data
# ---------------------------------------------------------------------------

def build_synthetic_suppliers(
    hourly_df: pd.DataFrame,
    market: str,
    seed: int = 123,
) -> pd.DataFrame:
    """Build a synthetic supplier balance table calibrated to approximate Table 2.

    Shares chosen so that the key-lender exposure fraction ≈ paper target:
      SOL : lender share ~10%  → exposed ≈ 11.4% of obs-window hours
      USDC: lender share ~4.5% → exposed ≈  7.9% of obs-window hours

    A "trader" (non-lender) with a higher share approximates the
    any-supplier exposed fraction (SOL 11.4%, USDC 11.6%).

    NOTE: Exact replication requires real per-account Drift data.
    """
    rng = np.random.default_rng(seed)
    hours = hourly_df["hour"].values
    supply_arr = hourly_df["total_supply"].values

    # Analytically calibrated shares (see exp/exp2_drift.py docstring and calibration script)
    #
    # SOL: key lender share 13.83% → exposed when U < 0.940 (91.9% of above-kink obs hours)
    #      → 12.4% × 0.919 ≈ 11.4% lender exposure   (paper target: 11.4%)
    #      paper max-share 20% → modelled here as a single lender whose share can peak
    #      at 20% when the pool shrinks (not modelled separately; report 20% in max_share)
    #
    # USDC: key lender share 7.08% → exposed when U < 0.855 (30.4% of above-kink obs hours)
    #       → 26.0% × 0.304 ≈ 7.9% lender exposure   (paper target:  7.9%)
    #       trader share 8.05% → 26.0% × 0.446 ≈ 11.6% any-supplier exposure
    #                                              (paper target: 11.6%)
    if market == "SOL":
        lender_shares = [0.1383]          # key lender: 13.83% typical share
        trader_shares = [0.1383]          # trader share same → any-supplier ≈ lender
    else:  # USDC
        lender_shares = [0.0708]          # key lender: 7.08%
        trader_shares = [0.0805]          # trader: 8.05% → any-supplier 11.6%

    all_shares  = lender_shares + trader_shares
    is_lender   = [True] * len(lender_shares) + [False] * len(trader_shares)

    rows = []
    for i, (share, lender_flag) in enumerate(zip(all_shares, is_lender)):
        supplier_id = f"lender_{i:03d}" if lender_flag else f"trader_{i:03d}"
        for hour, supply in zip(hours, supply_arr):
            rows.append({
                "hour": hour,
                "supplier_id": supplier_id,
                "balance": share * supply,
                "is_lender": lender_flag,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Exposure computation
# ---------------------------------------------------------------------------

def compute_exposure_for_market(
    market: str,
    full_start: str = "2024-11-11",
    obs_start: str = "2025-12-01",
    obs_end: str = "2026-04-01",
) -> dict:
    """Run the full exposure pipeline for one Drift market."""
    print(f"\n  Processing market: {market}")

    # Build full-window hourly states
    full_df = build_market_hourly(market, start=full_start, end=obs_end)
    val_metrics = validate_curve_against_charged(full_df)
    print(f"    Curve validation: {val_metrics['pct_within_1pct']:.1f}% within 1%")

    # Full-window market diagnostics
    full_diag = market_diagnostics(full_df, optimal_util=0.80, alpha_safe_threshold=0.05)

    # Balance-observable window
    obs_df = full_df[full_df["hour"] >= pd.Timestamp(obs_start)].copy()
    obs_diag = market_diagnostics(obs_df, optimal_util=0.80, alpha_safe_threshold=0.05)

    # Synthetic suppliers
    supplier_df = build_synthetic_suppliers(obs_df, market=market)
    lender_df = supplier_df[supplier_df["is_lender"]].copy()

    # Merge with market state
    market_hour_df = obs_df[["hour", "alpha_safe", "total_supply", "U"]].copy()

    # All suppliers
    all_exp = compute_hourly_exposure(market_hour_df, supplier_df[["hour", "supplier_id", "balance"]])
    all_fracs = exposure_fractions(all_exp)

    # Lenders only
    lender_exp = compute_hourly_exposure(market_hour_df, lender_df[["hour", "supplier_id", "balance"]])
    lender_fracs = exposure_fractions(lender_exp)

    # Longest-exposed single supplier
    exp_hours_per = lender_exp[lender_exp["exposed"]].groupby("supplier_id").size()
    max_hours = int(exp_hours_per.max()) if len(exp_hours_per) > 0 else 0
    most_exposed_id = exp_hours_per.idxmax() if len(exp_hours_per) > 0 else None
    sim_max_share = float(
        lender_exp[lender_exp["supplier_id"] == most_exposed_id]["share"].max()
    ) if most_exposed_id else 0.0

    # Paper-stated peak shares (when pool was at its smallest)
    PAPER_MAX_SHARE = {"SOL": 0.20, "USDC": 0.115}
    paper_max_share = PAPER_MAX_SHARE.get(market, sim_max_share)

    return {
        "market": market,
        "full_diag": full_diag,
        "obs_diag": obs_diag,
        "val_metrics": val_metrics,
        "all_fracs": all_fracs,
        "lender_fracs": lender_fracs,
        "max_exposed_hours": max_hours,
        "max_share_pct": paper_max_share * 100,   # paper-stated peak share
        "sim_max_share_pct": sim_max_share * 100,  # simulation (fixed pool)
        "obs_df": obs_df,
        "all_exp": all_exp,
        "lender_exp": lender_exp,
    }


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

def plot_market_diagnostics(results: dict) -> None:
    """Plot utilization and α_safe time series for both markets."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle("Drift Market Exposure Analysis (Synthetic Data)", fontsize=13, fontweight="bold")

    for row_idx, market in enumerate(["SOL", "USDC"]):
        obs_df = results[market]["obs_df"]
        ts = pd.to_datetime(obs_df["hour"])

        # Left: utilization
        ax = axes[row_idx, 0]
        ax.plot(ts, obs_df["U"] * 100, lw=0.8, alpha=0.7, label=f"{market} Utilization")
        ax.axhline(80, color="red", ls="--", lw=1.2, label="Kink O=80%")
        ax.set_ylabel("Utilization (%)"); ax.set_title(f"{market}: Utilization")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Right: α_safe
        ax2 = axes[row_idx, 1]
        ax2.plot(ts, obs_df["alpha_safe"] * 100, lw=0.8, alpha=0.7, color="green",
                 label=f"{market} α_safe")
        ax2.axhline(5, color="red", ls="--", lw=1.2, label="5% threshold")
        ax2.set_ylabel("α_safe (%)"); ax2.set_title(f"{market}: Protected Share α_safe(U)")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "fig4_drift_diagnostics.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  Saved: fig4_drift_diagnostics.png")


def plot_exposure_timeseries(results: dict) -> None:
    """Plot exposure flag time series for each market."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle("Supplier Exposure Flags (Synthetic Drift Data)", fontsize=13, fontweight="bold")

    for i, market in enumerate(["SOL", "USDC"]):
        lender_exp = results[market]["lender_exp"]
        hourly_exposed = lender_exp.groupby("hour")["exposed"].any().reset_index()
        hourly_exposed["hour"] = pd.to_datetime(hourly_exposed["hour"])

        ax = axes[i]
        ax.fill_between(hourly_exposed["hour"], hourly_exposed["exposed"].astype(int),
                        alpha=0.5, color="red", label="Hour with ≥1 exposed lender")
        ax.set_ylabel("Exposure"); ax.set_title(f"{market}: Hourly Exposure Flag")
        ax.set_ylim(0, 1.2); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "fig5_exposure_timeseries.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  Saved: fig5_exposure_timeseries.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("EXPERIMENT 2: Drift Supplier Exposure (Synthetic Data)")
    print("NOTE: Uses synthetic data; replace with real Drift data for")
    print("      full empirical replication.")
    print("=" * 60)

    results = {}
    summary_rows = []

    for market in ["SOL", "USDC"]:
        res = compute_exposure_for_market(market)
        results[market] = res
        targets = TABLE2_TARGETS[market]

        full_d = res["full_diag"]
        obs_d = res["obs_diag"]
        lf = res["lender_fracs"]
        af = res["all_fracs"]

        print(f"\n  === {market} Full-Window Results ===")
        print(f"    Above kink: {full_d['above_kink_pct']:.1f}%  (paper target: ~{targets['above_kink_pct_full']}%)")
        print(f"    α_safe < 5%: {full_d['alpha_safe_below_threshold_pct']:.2f}%  (target: ~{targets['alpha_safe_lt_5pct_full']}%)")
        print(f"    Median α_safe (above kink): {full_d['median_alpha_safe_above_kink_pct']:.2f}%  (target: ~{targets['median_alpha_safe_above']}%)")

        obs_above = obs_d["above_kink_pct"]
        print(f"\n  === {market} Balance-Observable Window ===")
        print(f"    Above kink: {obs_above:.1f}%  (target: ~{targets['above_kink_pct_obs']}%)")
        print(f"    Lender exposed: {lf['pct_hours_with_any_exposed']:.1f}%  (target: ~{targets['lender_exposed_pct']}%)")
        print(f"    Any supplier exposed: {af['pct_hours_with_any_exposed']:.1f}%  (target: ~{targets['supplier_exposed_pct']}%)")
        print(f"    Most exposed lender: {res['max_exposed_hours']} hours, "
              f"sim share {res['sim_max_share_pct']:.1f}%, paper peak share {res['max_share_pct']:.1f}%")

        summary_rows.append({
            "market": market,
            "above_kink_pct_full": round(full_d["above_kink_pct"], 2),
            "paper_target_above_kink_full": targets["above_kink_pct_full"],
            "alpha_safe_lt5_pct": round(full_d["alpha_safe_below_threshold_pct"], 2),
            "paper_target_alpha_lt5": targets["alpha_safe_lt_5pct_full"],
            "median_alpha_safe_above_kink_pct": round(full_d["median_alpha_safe_above_kink_pct"] or 0, 2),
            "paper_target_median": targets["median_alpha_safe_above"],
            "above_kink_pct_obs": round(obs_above, 2),
            "paper_target_above_kink_obs": targets["above_kink_pct_obs"],
            "lender_exposed_pct": round(lf["pct_hours_with_any_exposed"], 2),
            "paper_target_lender": targets["lender_exposed_pct"],
            "supplier_exposed_pct": round(af["pct_hours_with_any_exposed"], 2),
            "paper_target_supplier": targets["supplier_exposed_pct"],
            "max_exposed_hours": res["max_exposed_hours"],
            "paper_peak_share_pct": round(res["max_share_pct"], 2),
        })

    # Plots
    print("\n=== Generating Plots ===")
    plot_market_diagnostics(results)
    plot_exposure_timeseries(results)

    # Save results
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS / "exp2_exposure_summary.csv", index=False)
    print(f"\n  Saved: exp2_exposure_summary.csv")
    print("\n✓ Experiment 2 complete (synthetic data).")

    return summary_df


if __name__ == "__main__":
    main()
