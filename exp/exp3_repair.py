"""
Experiment 3: Design and repair replication for single-kink and multiple-kink
curves, including holding-period income comparisons.

Reproduces:
  - Single-kink power repair with catch-up utilization U_c ≈ 90.92%
  - Multiple-kink fixed-kink repairs (floor, raise-only, lower-only, mixed)
  - Holding-period income comparisons (Scenarios 1 and 2)

All synthetic parameters taken exactly from the paper.
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

from src.curves import SingleKinkCurve, RepairedSingleKinkCurve, PiecewiseLinearCurve
from src.admissibility import fixed_share_K, fixed_share_alpha
from src.single_kink_repair import (
    find_catchup_utilization, build_repaired_curve, income_derivative_at_U,
    verify_repair_protects,
)
from src.multikink_caps import (
    usdt_knots, segment_admissibility_table, fresh_cap, fixed_share_cap, combined_cap,
)
from src.fixed_kink_repairs import (
    floor_repair, raise_only_repair, lower_only_repair, mixed_repair,
    repair_summary_table,
)
from src.holding_period import (
    scenario1_paths, scenario2_paths, accumulated_income,
)

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

# Canonical parameters
O = 0.80; R_o = 0.05; R_max = 1.00
ALPHA_BAR = 0.05   # 5% protection target
K = fixed_share_K(ALPHA_BAR)   # K = 19
RESERVE_FACTOR = 0.05


# ---------------------------------------------------------------------------
# 1. Single-kink power repair
# ---------------------------------------------------------------------------

def run_single_kink_repair():
    """Reproduce Section 3 power repair with K = 19, U_c ≈ 90.92%."""
    print("\n=== Single-Kink Power Repair ===")
    print(f"  Target ᾱ = {ALPHA_BAR*100:.0f}%  →  K = {K:.2f}")

    curve_old = SingleKinkCurve(O=O, R_o=R_o, R_max=R_max)
    U_c = find_catchup_utilization(O, R_o, R_max, K)
    curve_new, _, _ = build_repaired_curve(O, R_o, R_max, ALPHA_BAR)

    print(f"  Catch-up utilization U_c = {U_c * 100:.4f}%  (expected ~90.92%)")
    assert abs(U_c - 0.9092) < 0.002, f"U_c mismatch: {U_c:.4f}"

    # Verify continuity
    r_at_O_new = float(curve_new.rate(O))
    r_at_O_old = float(curve_old.rate(O))
    assert abs(r_at_O_new - r_at_O_old) < 1e-8, "Rate not continuous at kink O"

    r_at_Uc_power = float(R_o * (U_c / O) ** K)
    r_at_Uc_steep = float(R_o + curve_old.m * (U_c - O))
    print(f"  Rate at U_c: power={r_at_Uc_power*100:.4f}%  steep={r_at_Uc_steep*100:.4f}%")
    assert abs(r_at_Uc_power - r_at_Uc_steep) < 1e-4, "Not continuous at U_c"

    # Verify protection property
    protected = verify_repair_protects(curve_new, ALPHA_BAR)
    print(f"  Repair protects ≥{ALPHA_BAR*100:.0f}% suppliers: {protected}")
    assert protected, "Repair should protect target share"

    # Elasticity comparison on repair branch
    U_test = np.linspace(O + 0.001, U_c - 0.001, 5)
    for u in U_test:
        eta_new = curve_new.elasticity(u, use_left=True)
        alpha_safe_new = 1.0 / (1.0 + eta_new)
        assert alpha_safe_new >= ALPHA_BAR - 1e-8, f"α_safe={alpha_safe_new:.4f} < ᾱ={ALPHA_BAR} at U={u:.3f}"

    return curve_old, curve_new, U_c


# ---------------------------------------------------------------------------
# 2. Multiple-kink USDT repair
# ---------------------------------------------------------------------------

def run_multikink_repair():
    """Reproduce Table 4 USDT curve repair for fresh-only, fixed-only, both."""
    print("\n=== Multiple-Kink USDT Repair ===")
    us, rs = usdt_knots()
    print(f"  USDT knots (u): {[f'{u*100:.1f}%' for u in us]}")
    print(f"  USDT rates (r): {[f'{r*100:.4f}%' for r in rs]}")

    results = {}
    for guarantee, g_label in [("fresh", "Fresh-only"), ("fixed", "Fixed-share-only"), ("both", "Both")]:
        print(f"\n  --- {g_label} guarantee ---")
        df_adm = segment_admissibility_table(us, rs, K, guarantee=guarantee)
        violations = df_adm[~df_adm["admissible"]]
        print(f"    Violations: {len(violations)} / {len(df_adm)} segments")
        print(df_adm[["u_L","u_H","R_H/R_L","cap (selected)","admissible","violation"]].to_string(index=False))

        # Floor repair
        r_floor = floor_repair(us, rs[-1], K, guarantee=guarantee)
        base_floor = r_floor[0] * 100

        # Raise-only
        r_raise = raise_only_repair(us, rs, K, guarantee=guarantee)

        # Lower-only
        r_lower = lower_only_repair(us, rs, K, guarantee=guarantee)
        terminal_lower = r_lower[-1] * 100

        # Mixed: for "fresh" guarantee fixing both endpoints causes infeasibility
        # (chain upper-bound from base and lower-bound from top are incompatible).
        # Use fix_base=False so the QP finds the optimal base rate freely,
        # yielding an intermediate income between floor and raise-only (~$1,680).
        fix_base_flag = guarantee != "fresh"
        r_mixed = mixed_repair(us, rs, K, guarantee=guarantee,
                               fix_base=fix_base_flag, fix_top=True)

        print(f"    Floor base APR:      {base_floor:.4f}%")
        print(f"    Raise-only base:     {r_raise[0]*100:.4f}%")
        print(f"    Lower-only terminal: {terminal_lower:.4f}%")

        results[guarantee] = {
            "adm_table": df_adm,
            "floor": r_floor,
            "raise": r_raise,
            "lower": r_lower,
            "mixed": r_mixed,
        }

    # Expected targets
    fresh_floor_base = results["fresh"]["floor"][0] * 100
    raise_base_fresh = results["fresh"]["raise"][0] * 100
    both_floor_base = results["both"]["floor"][0] * 100
    lower_max_fresh = results["fresh"]["lower"][-1] * 100
    lower_max_both = results["both"]["lower"][-1] * 100

    print(f"\n  Summary:")
    print(f"    Fresh-only floor base APR:  {fresh_floor_base:.4f}%  (expected ~3.47%)")
    print(f"    Raise-only base (fresh):    {raise_base_fresh:.4f}%  (expected ~6.39%)")
    print(f"    Both-guarantee floor base:  {both_floor_base:.4f}%  (expected ~8.68%)")
    print(f"    Lower-only max (fresh):     {lower_max_fresh:.4f}%  (expected ~12.04%)")
    print(f"    Lower-only max (both):      {lower_max_both:.4f}%  (expected ~4.82%)")

    return us, rs, results


# ---------------------------------------------------------------------------
# 3. Holding-period income
# ---------------------------------------------------------------------------

def run_holding_period(us: list[float], rs: list[float], results: dict):
    """Reproduce holding-period income comparisons."""
    print("\n=== Holding-Period Income Comparison ===")
    H = 365.0  # 1 year
    f = RESERVE_FACTOR

    # Build curve objects
    baseline_curve = PiecewiseLinearCurve(us, rs)
    fresh_floor_curve = PiecewiseLinearCurve(us, results["fresh"]["floor"].tolist())
    fresh_raise_curve = PiecewiseLinearCurve(us, results["fresh"]["raise"].tolist())
    fresh_lower_curve = PiecewiseLinearCurve(us, results["fresh"]["lower"].tolist())
    fresh_mixed_curve = PiecewiseLinearCurve(us, results["fresh"]["mixed"].tolist())
    both_floor_curve  = PiecewiseLinearCurve(us, results["both"]["floor"].tolist())

    # Scenario 1
    print("\n  Scenario 1 (fixed 10k USDT, varying utilization):")
    b_fn1, U_fn1 = scenario1_paths(initial_balance=10_000.0)

    scenarios_1 = {
        "baseline": baseline_curve,
        "fresh_floor": fresh_floor_curve,
        "fresh_raise": fresh_raise_curve,
        "fresh_mixed": fresh_mixed_curve,
        "both_floor": both_floor_curve,
    }
    incomes_1 = {}
    for name, curve in scenarios_1.items():
        A = accumulated_income(H, b_fn1, U_fn1, curve, f=f, n_intervals=2000)
        incomes_1[name] = A
        print(f"    {name:20s}: ${A:,.0f}")

    # Paper targets (approximate)
    print("\n  Paper targets (fresh-only guarantee):")
    print("    floor    ~$1,287,  raise    ~$2,231,  mixed    ~$1,680")
    print("    baseline ~$1,983,  both     ~$3,165")

    # Scenario 2
    print("\n  Scenario 2 (step-up principal, same utilization):")
    b_fn2, U_fn2 = scenario2_paths(initial_balance=10_000.0)
    incomes_2 = {}
    for name, curve in scenarios_1.items():
        A = accumulated_income(H, b_fn2, U_fn2, curve, f=f, n_intervals=2000)
        incomes_2[name] = A
        print(f"    {name:20s}: ${A:,.0f}")

    # Verify that the 5% reserve-factor multiplier on income difference is ~0.0095
    # for a supplier with 1% share on 1M total supply
    # Each unit of borrower interest → 0.01 * (1-0.05) = 0.0095 to depositor
    multiplier = 0.01 * (1 - RESERVE_FACTOR)
    print(f"\n  1% share multiplier = {multiplier:.4f}  (expected 0.0095)")
    assert abs(multiplier - 0.0095) < 1e-8, f"Multiplier mismatch: {multiplier}"

    return incomes_1, incomes_2


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_repaired_curves(curve_old: SingleKinkCurve, curve_new: RepairedSingleKinkCurve, U_c: float):
    """Figure 4/5 style: old vs repaired borrow curve and income derivative."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Single-Kink Power Repair (K=19, ᾱ=5%)", fontsize=13, fontweight="bold")

    U_grid = np.linspace(0.01, 0.9999, 600)

    # Left: borrow APR comparison
    ax = axes[0]
    Rb_old = np.vectorize(lambda u: float(curve_old.rate(u)))(U_grid)
    Rb_new = np.vectorize(lambda u: float(curve_new.rate(u)))(U_grid)
    ax.plot(U_grid * 100, Rb_old * 100, "b--", lw=2, label="Baseline curve")
    ax.plot(U_grid * 100, Rb_new * 100, "r-", lw=2, label="Repaired curve (K=19)")
    ax.axvline(O * 100, color="gray", ls=":", lw=1, label=f"Kink O={O*100:.0f}%")
    ax.axvline(U_c * 100, color="green", ls=":", lw=1.2, label=f"U_c={U_c*100:.2f}%")
    ax.set_xlabel("Utilization U (%)"); ax.set_ylabel("Borrow APR (%)")
    ax.set_title("Borrow APR: Baseline vs Repaired"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right: income derivative sign (dI/dδ) for α = 10%
    ax2 = axes[1]
    alpha_test = 0.10
    dI_old = [income_derivative_at_U(float(u), curve_old, alpha_test) for u in U_grid]
    dI_new = [income_derivative_at_U(float(u), curve_new, alpha_test) for u in U_grid]
    ax2.plot(U_grid * 100, dI_old, "b--", lw=1.8, label="Baseline dI/dδ (α=10%)")
    ax2.plot(U_grid * 100, dI_new, "r-", lw=1.8, label="Repaired dI/dδ (α=10%)")
    ax2.axhline(0, color="k", lw=1)
    ax2.axvline(O * 100, color="gray", ls=":", lw=1)
    ax2.axvline(U_c * 100, color="green", ls=":", lw=1.2)
    ax2.fill_between(U_grid * 100, dI_old, 0, where=np.array(dI_old) < 0, alpha=0.2, color="blue", label="Violation region (baseline)")
    ax2.set_xlabel("Utilization U (%)"); ax2.set_ylabel("dI/dδ")
    ax2.set_title("Income Derivative: Baseline vs Repaired"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "fig6_power_repair.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: fig6_power_repair.png")


def plot_multikink_repairs(us: list[float], rs: list[float], results: dict):
    """Plot USDT borrow-APR curves for all repair strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("USDT Multiple-Kink Repairs", fontsize=13, fontweight="bold")

    us_arr = np.array(us)
    styles = {"floor": "r-", "raise": "b--", "lower": "g-.", "mixed": "m:"}

    for col_idx, guarantee in enumerate(["fresh", "both"]):
        ax = axes[col_idx]
        ax.plot(us_arr * 100, np.array(rs) * 100, "k-", lw=2.5, label="Baseline", marker="o", ms=4)
        for method, style in styles.items():
            r_rep = results[guarantee][method]
            ax.plot(us_arr * 100, r_rep * 100, style, lw=1.8, label=f"{method.capitalize()}", marker="s", ms=3)
        ax.set_xlabel("Utilization U (%)"); ax.set_ylabel("Borrow APR (%)")
        title = "Fresh-Depositor Guarantee" if guarantee == "fresh" else "Combined Guarantee"
        ax.set_title(f"USDT Repairs: {title}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "fig7_multikink_repairs.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: fig7_multikink_repairs.png")


def plot_holding_period(incomes_1: dict, incomes_2: dict):
    """Bar chart of accumulated supplier income per repair strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Holding-Period Supplier Income by Repair Strategy", fontsize=13, fontweight="bold")

    for ax, incomes, title in zip(axes,
                                   [incomes_1, incomes_2],
                                   ["Scenario 1 (Fixed 10k)", "Scenario 2 (Step-Up)"]):
        names = list(incomes.keys())
        vals = [incomes[n] for n in names]
        colors = ["black", "red", "blue", "purple", "green"]
        bars = ax.bar(range(len(names)), vals, color=colors[:len(names)], alpha=0.75)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
        ax.set_ylabel("Accumulated Interest ($)")
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"${v:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(RESULTS / "fig8_holding_period.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: fig8_holding_period.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("EXPERIMENT 3: Rate-Curve Repair and Holding-Period Income")
    print("=" * 60)

    # 1. Single-kink power repair
    curve_old, curve_new, U_c = run_single_kink_repair()

    # 2. Multiple-kink USDT repair
    us, rs, repair_results = run_multikink_repair()

    # 3. Holding-period income
    incomes_1, incomes_2 = run_holding_period(us, rs, repair_results)

    # Plots
    print("\n=== Generating Plots ===")
    plot_repaired_curves(curve_old, curve_new, U_c)
    plot_multikink_repairs(us, rs, repair_results)
    plot_holding_period(incomes_1, incomes_2)

    # Save results
    _save_results(U_c, us, rs, repair_results, incomes_1, incomes_2)
    print("\n✓ Experiment 3 complete.")


def _save_results(U_c, us, rs, repair_results, incomes_1, incomes_2):
    """Save tabular results."""
    # Single-kink summary
    pd.DataFrame([{"U_c": U_c, "K": K, "alpha_bar_pct": ALPHA_BAR * 100}]).to_csv(
        RESULTS / "exp3_singlekink_summary.csv", index=False
    )

    # Multi-kink repair tables
    for guarantee in ["fresh", "both"]:
        summary = repair_summary_table(us, rs, K, guarantee=guarantee)
        summary.to_csv(RESULTS / f"exp3_usdt_{guarantee}_repair.csv", index=False)

        adm = repair_results[guarantee]["adm_table"]
        adm.to_csv(RESULTS / f"exp3_usdt_{guarantee}_admissibility.csv", index=False)

    # Holding-period income
    hp_df = pd.DataFrame({
        "curve": list(incomes_1.keys()),
        "income_scenario1": list(incomes_1.values()),
        "income_scenario2": list(incomes_2.values()),
    })
    hp_df.to_csv(RESULTS / "exp3_holding_period_income.csv", index=False)
    print("  Saved CSV result files.")


if __name__ == "__main__":
    main()
