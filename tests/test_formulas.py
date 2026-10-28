"""
Tests for core lending-market formulas.

Verifies:
  - Derivative identity dI/dδ = U R_b(U)[1 - α(1+η(U))]
  - Monotonicity sign condition
  - Section 2 failure example values
  - Elasticity computations
  - Edge cases (kink, full utilization)
"""

import pytest
import numpy as np
from src.curves import SingleKinkCurve, PiecewiseLinearCurve
from src.monotonicity import (
    utilization, supply_apr, supplier_income, income_derivative,
    sign_condition_ok, reproduce_section2_key_states,
    sweep_supplier_balance, build_section2_curve,
    SEC2_B, SEC2_O, SEC2_R_o, SEC2_R_max,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def canonical_curve():
    return SingleKinkCurve(O=0.80, R_o=0.05, R_max=1.00)


@pytest.fixture
def section2_state():
    """Section 2 initial state: b=1M, B=8.3M, T≈10.0M."""
    B = SEC2_B
    T1 = B / 0.83
    T_other = T1 - 1_000_000.0
    return {"B": B, "T_other": T_other, "b": 1_000_000.0}


# ---------------------------------------------------------------------------
# Basic quantity tests
# ---------------------------------------------------------------------------

def test_utilization_basic():
    assert abs(utilization(8.3e6, 10.0e6) - 0.83) < 1e-10


def test_supply_apr():
    assert abs(supply_apr(0.83, 0.1925) - 0.83 * 0.1925) < 1e-10


def test_supplier_income():
    b = 1_000_000
    U = 0.83
    Rb = 0.1925
    I = supplier_income(b, U, Rb)
    assert abs(I - b * U * Rb) < 0.01


def test_income_derivative_identity():
    """Verify dI/dδ numerically matches the analytic formula."""
    curve = build_section2_curve()
    B = SEC2_B
    T1 = B / 0.83
    T_other = T1 - 1_000_000.0
    b0 = 1_000_000.0
    eps = 0.01  # tiny deposit

    T0 = T_other + b0
    T_eps = T_other + b0 + eps
    I0 = b0 * (B / T0) * float(curve.rate(B / T0))
    I_eps = (b0 + eps) * (B / T_eps) * float(curve.rate(B / T_eps))
    dI_numeric = (I_eps - I0) / eps

    U0 = B / T0
    Rb0 = float(curve.rate(U0))
    alpha0 = b0 / T0
    eta0 = curve.elasticity(U0, use_left=True)
    dI_analytic = income_derivative(U0, Rb0, alpha0, eta0)

    assert abs(dI_numeric - dI_analytic) < 0.1, (
        f"Derivative mismatch: numeric={dI_numeric:.4f}, analytic={dI_analytic:.4f}"
    )


def test_income_derivative_sign_condition():
    """α * (1+η) ≤ 1 ↔ dI/dδ ≥ 0."""
    U = 0.83
    Rb = 0.1925
    eta = 20.5
    # Small supplier: α = 1%
    assert sign_condition_ok(0.01, eta) == (0.01 * (1 + eta) <= 1.0)
    # Large supplier: α = 10% should violate with η ≈ 20.5
    assert not sign_condition_ok(0.10, 20.5)
    # Zero share always OK
    assert sign_condition_ok(0.0, 100.0)


def test_sign_condition_range():
    """α_safe(U) = 1/(1+η) is in (0, 1]."""
    for eta in [0.5, 1.0, 5.0, 20.5, 76.0]:
        alpha_safe = 1.0 / (1.0 + eta)
        assert 0 < alpha_safe <= 1, f"α_safe={alpha_safe} out of range"


# ---------------------------------------------------------------------------
# Section 2 failure reproduction
# ---------------------------------------------------------------------------

def test_section2_deposit_reduces_income():
    """Depositing 375k reduces income from ~159,775 to ~55,000."""
    df = reproduce_section2_key_states()
    s1 = df[df.state == "initial (b=1M)"].iloc[0]
    s2 = df[df.state == "post-deposit (+375k)"].iloc[0]
    assert s2.I < s1.I, "Deposit should reduce income"
    assert s1.I > 100_000, f"Initial income too low: {s1.I}"
    assert s2.I < 100_000, f"Post-deposit income too high: {s2.I}"


def test_section2_withdrawal_increases_income():
    """Withdrawing ~302k increases income above initial."""
    df = reproduce_section2_key_states()
    s1 = df[df.state == "initial (b=1M)"].iloc[0]
    s3 = df[df.state == "post-withdrawal (-302k)"].iloc[0]
    assert s3.I > s1.I, "Withdrawal should increase income"


def test_section2_utilization_values(canonical_curve):
    """Check approximate utilization targets from paper."""
    B = SEC2_B
    T1 = B / 0.83
    T_other = T1 - 1_000_000.0

    b1 = 1_000_000.0
    T1_actual = T_other + b1
    U1 = B / T1_actual
    assert abs(U1 - 0.83) < 0.005, f"Initial U mismatch: {U1}"

    b2 = 1_375_000.0
    T2 = T_other + b2
    U2 = B / T2
    assert abs(U2 - 0.80) < 0.005, f"Post-deposit U mismatch: {U2}"


# ---------------------------------------------------------------------------
# Single-kink curve tests
# ---------------------------------------------------------------------------

def test_single_kink_rate_at_kink(canonical_curve):
    """Rate at kink should be exactly R_o."""
    Rb_at_O = float(canonical_curve.rate(0.80))
    assert abs(Rb_at_O - 0.05) < 1e-10


def test_single_kink_rate_at_zero(canonical_curve):
    """Rate at U=0 should be 0 (ramp starts at 0)."""
    assert float(canonical_curve.rate(0.0)) == pytest.approx(0.0, abs=1e-10)


def test_single_kink_rate_at_full(canonical_curve):
    """Rate at U=1 should be R_max."""
    assert float(canonical_curve.rate(1.0)) == pytest.approx(1.00, abs=1e-8)


def test_single_kink_elasticity_ramp(canonical_curve):
    """On the ramp below O, η(U) = 1 (linear proportional ramp)."""
    for U in [0.1, 0.4, 0.79]:
        eta = canonical_curve.elasticity(U, use_left=True)
        assert abs(eta - 1.0) < 1e-8, f"Expected η=1 on ramp, got {eta} at U={U}"


def test_single_kink_elasticity_steep(canonical_curve):
    """On the steep branch, η(U) = U*m/R_b(U) > 1."""
    m = canonical_curve.m
    for U in [0.85, 0.90, 0.95]:
        Rb = float(canonical_curve.rate(U))
        eta_expected = U * m / Rb
        eta_actual = canonical_curve.elasticity(U, use_left=True)
        assert abs(eta_actual - eta_expected) < 1e-8, f"Elasticity mismatch at U={U}"


def test_single_kink_elasticity_at_kink_leftward():
    """At kink, left-slope elasticity should be 1 (ramp elasticity)."""
    curve = SingleKinkCurve(O=0.80, R_o=0.05, R_max=1.00)
    eta_left = curve.elasticity(0.80, use_left=True)
    assert abs(eta_left - 1.0) < 1e-8


def test_elasticity_paper_values():
    """Reproduce paper's stated elasticity values near kink and at 83%."""
    curve = build_section2_curve()
    # η at U = 0.83 (above kink, steep branch)
    eta_83 = curve.elasticity(0.83, use_left=True)
    # Paper states ~20.5; verify roughly correct
    assert 15 <= eta_83 <= 30, f"η at U=83% should be ~20.5, got {eta_83:.2f}"

    # α(1+η) for α=10% at U=83%: should be > 1 (violation)
    alpha = 0.10
    assert alpha * (1 + eta_83) > 1.0, "Should violate monotonicity"


def test_alpha_safe_bounds():
    """α_safe(U) ∈ (0, 1] for all valid U."""
    curve = build_section2_curve()
    for U in np.linspace(0.01, 0.99, 50):
        alpha_safe = curve.alpha_safe(U)
        assert 0 < alpha_safe <= 1, f"α_safe={alpha_safe} out of bounds at U={U}"


# ---------------------------------------------------------------------------
# Piecewise-linear curve tests
# ---------------------------------------------------------------------------

def test_piecewise_linear_continuity():
    """PiecewiseLinearCurve is continuous at all knots."""
    us = [0.0, 0.8, 0.9, 1.0]
    rs = [0.05, 0.09, 0.20, 0.90]
    curve = PiecewiseLinearCurve(us, rs)
    for u, r in zip(us, rs):
        assert abs(float(curve.rate(u)) - r) < 1e-8, f"Not continuous at u={u}"


def test_piecewise_linear_interpolation():
    """Rate at midpoint of segment is correct linear interpolation."""
    us = [0.0, 1.0]
    rs = [0.0, 1.0]
    curve = PiecewiseLinearCurve(us, rs)
    assert abs(float(curve.rate(0.5)) - 0.5) < 1e-10


def test_piecewise_elasticity_formula():
    """η(U) = U * m / R_b(U) on each segment."""
    us = [0.8, 0.9, 1.0]
    rs = [0.05, 0.20, 0.90]
    curve = PiecewiseLinearCurve(us, rs)
    U = 0.85  # mid-segment
    m = (0.20 - 0.05) / (0.9 - 0.8)  # = 1.5
    Rb = 0.05 + m * (U - 0.80)
    eta_expected = U * m / Rb
    eta_actual = curve.elasticity(U, use_left=True)
    assert abs(eta_actual - eta_expected) < 1e-8


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_income_derivative_zero_share():
    """α=0 → dI/dδ = U*R_b > 0 always."""
    dI = income_derivative(0.83, 0.1925, 0.0, 20.5)
    assert dI > 0


def test_income_derivative_full_share():
    """α=1 with large η → strongly negative dI/dδ."""
    eta = 20.0
    dI = income_derivative(0.83, 0.1925, 1.0, eta)
    assert dI < 0


def test_single_kink_invalid_params():
    """Invalid params should raise AssertionError."""
    with pytest.raises(AssertionError):
        SingleKinkCurve(O=1.1, R_o=0.05, R_max=1.00)
    with pytest.raises(AssertionError):
        SingleKinkCurve(O=0.80, R_o=0.0, R_max=1.00)
    with pytest.raises(AssertionError):
        SingleKinkCurve(O=0.80, R_o=0.05, R_max=0.04)


def test_sweep_supplier_balance_monotone_near_kink():
    """Income should have a trough at/near the kink."""
    df = sweep_supplier_balance(n_points=200)
    # Find b closest to kink (U=80%)
    B = SEC2_B
    T_other = B / 0.83 - 1_000_000.0
    b_kink = B / 0.80 - T_other  # b such that U = 80%
    # Income should be lower near the kink than at large b
    idx_kink = (df["b"] - b_kink).abs().idxmin()
    income_near_kink = df.loc[idx_kink, "I"]
    income_far = df[df["b"] > b_kink * 1.5]["I"].max()
    assert income_far > income_near_kink, "Income should be higher further from kink"
