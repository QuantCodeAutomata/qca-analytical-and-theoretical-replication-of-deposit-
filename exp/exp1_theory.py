"""
Experiment 1: Analytical and theoretical replication of deposit monotonicity,
elasticity condition, and admissible rate-curve families.

Reproduces:
  - Section 2 illustrative failure example
  - Derivative identity dI/dδ = U R_b(U)[1 - α(1+η(U))]
  - Elasticity as the key monotonicity measure
  - Fixed-share, fresh-depositor, and combined admissible curve families
  - Appendix E smoothing formulas

All paper-stated parameter values are used exactly.
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
import matplotlib.gridspec as gridspec
from pathlib import Path

from src.curves import SingleKinkCurve, PowerKinkCurve, PiecewiseLinearCurve
from src.monotonicity import (
    reproduce_section2_key_states,
    sweep_supplier_balance,
    compute_elasticity_diagnostics,
    SEC2_B, SEC2_O, SEC2_R_o, SEC2_R_max,
    build_section2_curve,
)
from src.admissibility import (
    endpoint_K_star, alpha_max,
    fixed_share_boundary_curve, fixed_share_K, fixed_share_alpha,
    fresh_depositor_profile, fresh_depositor_envelope, fresh_depositor_boundary,
    fresh_depositor_piecewise_check, appendix_a_knots,
    combined_profile, combined_envelope, combined_boundary_F, combined_base_bound,
    integral_admissibility_bound,
    canonical_K_star, canonical_alpha_max,
)
from src.smoothing import (
    eta_max_from_width, d_eta_max_dw, feasibility_width_bound, table5,
    rate_from_elasticity_profile,
)

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Section 2 failure reproduction
# ---------------------------------------------------------------------------

def run_section2_failure():
    """Reproduce Section 2 illustrative failure mode."""
    print("\n=== Section 2: Illustrative Failure ===")
    curve = build_section2_curve()
    df = reproduce_section2_key_states(curve)
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    # Additional diagnostics
    B = SEC2_B
    T1 = B / 0.83
    T_other = T1 - 1_000_000.0

    # Paper key numbers check
    s1 = df[df.state == "initial (b=1M)"].iloc[0]
    s2 = df[df.state == "post-deposit (+375k)"].iloc[0]
    s3 = df[df.state == "post-withdrawal (-302k)"].iloc[0]

    print(f"\nState 1 (initial):       U={s1.U:.4f}  Rb={s1.Rb:.4f}  Rs={s1.Rs:.4f}  I={s1.I:,.0f}")
    print(f"State 2 (post-deposit):  U={s2.U:.4f}  Rb={s2.Rb:.4f}  Rs={s2.Rs:.4f}  I={s2.I:,.0f}")
    print(f"State 3 (post-withdraw): U={s3.U:.4f}  Rb={s3.Rb:.4f}  Rs={s3.Rs:.4f}  I={s3.I:,.0f}")

    # Verify deposit lowers income
    assert s2.I < s1.I, "Deposit should lower income in the failure example"
    # Verify withdrawal raises income
    assert s3.I > s1.I, "Withdrawal should raise income in the failure example"

    return df


# ---------------------------------------------------------------------------
# 2. Derivative identity verification
# ---------------------------------------------------------------------------

def run_derivative_identity():
    """Verify the identity dI/dδ = U R_b(U)[1 - α(1+η(U))] numerically."""
    print("\n=== Derivative Identity Verification ===")
    curve = build_section2_curve()
    B = SEC2_B
    T1 = B / 0.83
    T_other = T1 - 1_000_000.0

    # Numerical derivative via finite differences
    b0 = 1_000_000.0
    eps = 1.0  # $1 deposit

    T0 = T_other + b0
    T_eps = T_other + b0 + eps

    I0 = b0 * (B / T0) * float(curve.rate(B / T0))
    I_eps = (b0 + eps) * (B / T_eps) * float(curve.rate(B / T_eps))
    dI_numeric = (I_eps - I0) / eps

    # Analytic formula
    U0 = B / T0
    Rb0 = float(curve.rate(U0))
    alpha0 = b0 / T0
    eta0 = curve.elasticity(U0, use_left=True)
    dI_analytic = U0 * Rb0 * (1.0 - alpha0 * (1.0 + eta0))

    print(f"  Numerical  dI/dδ = {dI_numeric:.6f}")
    print(f"  Analytic   dI/dδ = {dI_analytic:.6f}")
    print(f"  Difference        = {abs(dI_numeric - dI_analytic):.2e}")

    assert abs(dI_numeric - dI_analytic) < 0.01, "Derivative identity mismatch"
    return dI_numeric, dI_analytic


# ---------------------------------------------------------------------------
# 3. Elasticity diagnostics
# ---------------------------------------------------------------------------

def run_elasticity_diagnostics():
    """Reproduce paper elasticity discussion values."""
    print("\n=== Elasticity Diagnostics ===")
    curve = build_section2_curve()
    df = compute_elasticity_diagnostics(curve, Us=[0.80, 0.8001, 0.83, 1.00], alpha=0.10)
    print(df.to_string(index=False))

    # Verify representative values
    row_83 = df[df.U == 0.83].iloc[0]
    print(f"\n  U=0.83: eta={row_83['eta_left']:.2f}, α(1+η)={row_83['alpha*(1+eta) [alpha=10%]']:.4f}")
    assert row_83["eta_left"] > 15.0, f"Expected η > 15 at U=83%, got {row_83['eta_left']:.2f}"

    # Check just above kink
    row_kink_plus = df[df.U == 0.8001].iloc[0]
    print(f"  U=0.8001 (right slope): eta_right={row_kink_plus['eta_right']:.2f}")

    # Alpha safe at 1.00
    row_100 = df[df.U == 1.00].iloc[0]
    print(f"  U=1.00: eta={row_100['eta_left']:.4f}")

    return df


# ---------------------------------------------------------------------------
# 4. Fixed-share protection
# ---------------------------------------------------------------------------

def run_fixed_share():
    """Reproduce fixed-share boundary curves and endpoint bounds."""
    print("\n=== Fixed-Share Protection ===")
    O = SEC2_O; R_o = SEC2_R_o; R_max = SEC2_R_max

    K_star = canonical_K_star()
    alpha_max_val = canonical_alpha_max()
    print(f"  K*             = {K_star:.4f}  (expected ~13.43)")
    print(f"  α̂_max          = {alpha_max_val * 100:.4f}%  (expected ~6.93%)")

    assert abs(K_star - 13.43) < 0.05, f"K* mismatch: {K_star}"
    assert abs(alpha_max_val - 0.0693) < 0.005, f"α̂_max mismatch: {alpha_max_val}"

    # Build boundary curves for K = 5, 13.43, 19
    U_grid = np.linspace(O, 1.0, 300)
    curves_data = {}
    for alpha_b in [0.05, 1.0 / (1 + K_star), 0.10]:
        K = fixed_share_K(alpha_b)
        R_K = fixed_share_boundary_curve(O, R_o, K, U_grid)
        curves_data[f"α={alpha_b*100:.1f}%, K={K:.2f}"] = R_K

    # Check 5% protected share implies K = 19
    K_5pct = fixed_share_K(0.05)
    print(f"\n  5% protection → K = {K_5pct:.2f}  (expected 19)")
    assert abs(K_5pct - 19.0) < 0.001, f"K for 5% mismatch: {K_5pct}"

    return K_star, alpha_max_val, curves_data, U_grid


# ---------------------------------------------------------------------------
# 5. Fresh-depositor protection
# ---------------------------------------------------------------------------

def run_fresh_depositor():
    """Reproduce fresh-depositor boundary, Appendix A admissibility table."""
    print("\n=== Fresh-Depositor Protection ===")

    # Boundary curve: R_b(U) = R_b(U_a)(1-U_a)/(1-U)
    U_a = SEC2_O; Rb_a = SEC2_R_o
    U_grid = np.linspace(0.0, 0.999, 300)
    Rb_fd = fresh_depositor_boundary(U_grid, U_a, Rb_a)

    # Appendix A piecewise check
    us_a, rs_a = appendix_a_knots()
    df_check = fresh_depositor_piecewise_check(us_a, rs_a)
    print("\n  Appendix A admissibility table:")
    print(df_check.to_string(index=False))

    # Global sufficient profile alpha_bar(U) = 1 - U
    U_test = np.array([0.5, 0.8, 0.9, 0.95])
    alpha_bar_fd = fresh_depositor_profile(U_test)
    E_fd = fresh_depositor_envelope(U_test)
    print(f"\n  Profile ᾱ(U) = 1-U at U={U_test}:")
    print(f"    ᾱ(U) = {alpha_bar_fd}")
    print(f"    E(U) = U/(1-U) = {E_fd}")

    return df_check, Rb_fd, U_grid


# ---------------------------------------------------------------------------
# 6. Combined guarantees
# ---------------------------------------------------------------------------

def run_combined():
    """Reproduce combined profile, envelope, boundary, and base bound."""
    print("\n=== Combined Guarantees ===")
    alpha_bar = 0.05
    K = fixed_share_K(alpha_bar)
    t = K / (K + 1.0)
    print(f"  K = {K:.2f}, t = {t:.4f}")

    U_grid = np.linspace(0.001, 0.999, 300)
    alpha_both = combined_profile(U_grid, alpha_bar)
    E_both = combined_envelope(U_grid, K)
    F_both = combined_boundary_F(U_grid, K)

    base_bound = combined_base_bound(K, SEC2_R_max)
    print(f"  Combined base bound R_b(0) ≥ {base_bound * 100:.4f}%")

    return U_grid, alpha_both, E_both


# ---------------------------------------------------------------------------
# 7. Appendix E smoothing table
# ---------------------------------------------------------------------------

def run_smoothing():
    """Reproduce Appendix E Table 5."""
    print("\n=== Appendix E Smoothing ===")
    O = 0.80; R_o = 0.05; R_max = 1.00; eta_0 = 1.0
    L = np.log(1.0 / O)
    A = np.log(R_max / R_o)
    K_star = A / L

    widths = [0.0, 0.05, 0.10, 0.1382, L]
    df = table5(widths=widths, O=O, R_o=R_o, R_max=R_max, eta_0=eta_0)
    print(df.to_string(index=False))

    # Verify w=0: η_max = K* ≈ 13.43
    row_w0 = df[df.w == 0.0].iloc[0]
    print(f"\n  w=0: η_max = {row_w0['eta_max']:.4f}  (expected K* ≈ {K_star:.4f})")
    assert abs(row_w0["eta_max"] - K_star) < 0.01, "η_max at w=0 should equal K*"

    # Feasibility width for K = 19
    K19 = 19.0
    w_feas = feasibility_width_bound(K19, K_star, eta_0, L)
    print(f"  Feasibility width for K=19: w ≤ {w_feas:.4f}")

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_section2_and_income(sweep_df: pd.DataFrame) -> None:
    """Figure 1 style: borrow curve and income vs supplier balance."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Section 2: Single-Kink Failure Example", fontsize=13, fontweight="bold")

    # Left: borrow APR curve
    ax = axes[0]
    U_grid = np.linspace(0.01, 0.9999, 500)
    curve = build_section2_curve()
    Rb = np.vectorize(lambda u: float(curve.rate(u)))(U_grid)
    ax.plot(U_grid * 100, Rb * 100, "b-", lw=2, label="Borrow APR R_b(U)")
    ax.plot(U_grid * 100, np.vectorize(lambda u: float(curve.supply_apr(u)))(U_grid) * 100,
            "g--", lw=1.5, label="Supply APR R_s(U) = U·R_b(U)")
    ax.axvline(80, color="gray", ls=":", lw=1, label="Kink O=80%")
    ax.axvline(83, color="red", ls="--", lw=1, alpha=0.7, label="U=83% (initial)")
    ax.set_xlabel("Utilization U (%)")
    ax.set_ylabel("APR (%)")
    ax.set_title("Borrow and Supply APR Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: income vs supplier balance
    ax2 = axes[1]
    ax2.plot(sweep_df["b"] / 1e6, sweep_df["I"], "b-", lw=2, label="Annualized Income I(b)")
    ax2.axvline(1.0, color="red", ls="--", lw=1.2, label="b=1M (initial)")
    ax2.axvline(1.375, color="orange", ls="--", lw=1.2, label="b=1.375M (post-deposit)")
    ax2.axvline(0.698, color="green", ls="--", lw=1.2, label="b=698k (post-withdrawal)")
    ax2.set_xlabel("Supplier Balance b (millions)")
    ax2.set_ylabel("Annualized Income ($)")
    ax2.set_title("Supplier Income vs Balance\n(borrowings fixed at $8.3M)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "fig1_section2_failure.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: fig1_section2_failure.png")


def plot_admissible_curves(K_star: float, U_grid: np.ndarray, curves_data: dict) -> None:
    """Fixed-share and fresh-depositor boundary curves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Admissible Rate-Curve Families", fontsize=13, fontweight="bold")

    O = SEC2_O; R_o = SEC2_R_o; R_max = SEC2_R_max

    # Left: fixed-share boundary curves
    ax = axes[0]
    colors = ["blue", "green", "red"]
    for (label, R_K), col in zip(curves_data.items(), colors):
        ax.plot(U_grid * 100, R_K * 100, lw=2, color=col, label=label)

    # Original steep branch
    curve = build_section2_curve()
    Rb_orig = np.vectorize(lambda u: float(curve.rate(u)))(U_grid)
    ax.plot(U_grid * 100, Rb_orig * 100, "k--", lw=1.5, label="Baseline (steep)")
    ax.axvline(80, color="gray", ls=":", lw=1)
    ax.set_xlabel("Utilization U (%)"); ax.set_ylabel("Borrow APR (%)")
    ax.set_title(f"Fixed-Share Boundary Curves\nK*≈{K_star:.2f}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right: fresh-depositor boundary
    ax2 = axes[1]
    U_fd = np.linspace(0.0, 0.95, 300)
    Rb_fd = fresh_depositor_boundary(U_fd, O, R_o)
    ax2.plot(U_fd * 100, Rb_fd * 100, "purple", lw=2, label="R_b(U)=R_o(1-O)/(1-U)")
    profile = fresh_depositor_profile(U_fd) * 100
    env = fresh_depositor_envelope(U_fd)
    ax2.plot(U_fd * 100, profile, "b--", lw=1.5, label="ᾱ(U)=1-U (%)")
    ax2_r = ax2.twinx()
    ax2_r.plot(U_fd * 100, env, "g-.", lw=1.2, label="E(U)=U/(1-U)")
    ax2_r.set_ylabel("Elasticity envelope E(U)", color="green")
    ax2.set_xlabel("Utilization U (%)"); ax2.set_ylabel("APR / Share (%)")
    ax2.set_title("Fresh-Depositor Boundary Curve")
    ax2.legend(loc="upper left", fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "fig2_admissible_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: fig2_admissible_curves.png")


def plot_smoothing(df_smooth: pd.DataFrame) -> None:
    """Appendix E: smoothing width vs plateau elasticity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Appendix E: C1 Smoothing of Single-Kink Curve", fontsize=13, fontweight="bold")

    O = 0.80; R_o = 0.05; R_max = 1.00; eta_0 = 1.0
    L = np.log(1.0 / O)
    A = np.log(R_max / R_o)

    widths_cont = np.linspace(0.0, L * 0.99, 200)
    eta_maxs = [eta_max_from_width(w, eta_0, L, A) for w in widths_cont]
    alpha_safes = [1.0 / (1.0 + em) * 100 for em in eta_maxs]

    # Left: η_max vs w
    ax = axes[0]
    ax.plot(widths_cont, eta_maxs, "b-", lw=2, label="η_max(w)")
    ax.axhline(A / L, color="k", ls="--", lw=1, label=f"K*={A/L:.2f} (w=0)")
    if not df_smooth.empty:
        ax.scatter(df_smooth["w"], df_smooth["eta_max"], color="red", s=60, zorder=5, label="Table 5 points")
    ax.set_xlabel("Smoothing width w"); ax.set_ylabel("Plateau elasticity η_max")
    ax.set_title("η_max vs Smoothing Width")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Right: protected share vs w
    ax2 = axes[1]
    ax2.plot(widths_cont, alpha_safes, "g-", lw=2)
    if not df_smooth.empty:
        finite_mask = df_smooth["alpha_safe_%"].notna() & np.isfinite(df_smooth["alpha_safe_%"])
        ax2.scatter(df_smooth.loc[finite_mask, "w"],
                    df_smooth.loc[finite_mask, "alpha_safe_%"],
                    color="red", s=60, zorder=5)
    ax2.set_xlabel("Smoothing width w"); ax2.set_ylabel("Protected share ᾱ (%)")
    ax2.set_title("Protected Share vs Smoothing Width")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "fig3_smoothing.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: fig3_smoothing.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("EXPERIMENT 1: Theoretical Replication")
    print("=" * 60)

    # 1. Section 2 failure
    df_sec2 = run_section2_failure()

    # 2. Derivative identity
    dI_num, dI_an = run_derivative_identity()

    # 3. Elasticity diagnostics
    df_elast = run_elasticity_diagnostics()

    # 4. Fixed-share
    K_star, alpha_max_val, curves_data, U_grid = run_fixed_share()

    # 5. Fresh-depositor
    df_fresh, Rb_fd, U_fd_grid = run_fresh_depositor()

    # 6. Combined
    U_comb, alpha_both, E_both = run_combined()

    # 7. Smoothing
    df_smooth = run_smoothing()

    # Plots
    print("\n=== Generating Plots ===")
    sweep_df = sweep_supplier_balance()
    plot_section2_and_income(sweep_df)
    plot_admissible_curves(K_star, U_grid, curves_data)
    plot_smoothing(df_smooth)

    # Save summary results
    _save_results(df_sec2, df_elast, df_fresh, df_smooth, K_star, alpha_max_val)

    print("\n✓ Experiment 1 complete.")


def _save_results(df_sec2, df_elast, df_fresh, df_smooth, K_star, alpha_max_val):
    """Save key numerical results to CSV for RESULTS.md."""
    df_sec2.to_csv(RESULTS / "exp1_section2_states.csv", index=False)
    df_elast.to_csv(RESULTS / "exp1_elasticity_diagnostics.csv", index=False)
    df_fresh.to_csv(RESULTS / "exp1_appendix_a_admissibility.csv", index=False)
    df_smooth.to_csv(RESULTS / "exp1_smoothing_table5.csv", index=False)
    summary = pd.DataFrame([{
        "K_star": K_star,
        "alpha_max_pct": alpha_max_val * 100,
        "K_5pct_target": fixed_share_K(0.05),
    }])
    summary.to_csv(RESULTS / "exp1_fixed_share_summary.csv", index=False)
    print("  Saved CSV result files.")


if __name__ == "__main__":
    main()
