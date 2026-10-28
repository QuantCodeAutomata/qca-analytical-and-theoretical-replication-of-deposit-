"""
Tests for admissibility theory: fixed-share, fresh-depositor, combined,
and smoothing formulas.
"""

import pytest
import numpy as np
from src.admissibility import (
    fixed_share_K, fixed_share_alpha, endpoint_K_star, alpha_max,
    fixed_share_boundary_curve,
    fresh_depositor_profile, fresh_depositor_envelope, fresh_depositor_boundary,
    fresh_depositor_exact_condition, fresh_depositor_piecewise_check,
    appendix_a_knots,
    combined_profile, combined_envelope, combined_boundary_F, combined_base_bound,
    integral_admissibility_bound,
    canonical_K_star, canonical_alpha_max,
)
from src.smoothing import (
    eta_max_from_width, d_eta_max_dw, eta_max_at_w0, feasibility_width_bound,
    table5, smoothstep,
)


# ---------------------------------------------------------------------------
# Fixed-share: K and alpha conversions
# ---------------------------------------------------------------------------

def test_fixed_share_K_roundtrip():
    """K = (1-ᾱ)/ᾱ and ᾱ = 1/(1+K) are inverses."""
    for alpha in [0.01, 0.05, 0.10, 0.50]:
        K = fixed_share_K(alpha)
        alpha_back = fixed_share_alpha(K)
        assert abs(alpha_back - alpha) < 1e-10, f"Roundtrip failed for ᾱ={alpha}"


def test_fixed_share_5pct_gives_K19():
    """5% protection → K = 19 exactly."""
    K = fixed_share_K(0.05)
    assert abs(K - 19.0) < 1e-10


def test_fixed_share_K_is_nonneg():
    """K ≥ 0 for ᾱ ∈ (0, 1)."""
    for alpha in np.linspace(0.01, 0.99, 50):
        K = fixed_share_K(alpha)
        assert K >= 0, f"K < 0 for ᾱ={alpha}"


# ---------------------------------------------------------------------------
# Endpoint K* and α̂_max
# ---------------------------------------------------------------------------

def test_canonical_K_star():
    """K* ≈ 13.43 for canonical O=0.8, R_o=0.05, R_max=1.00."""
    K_star = canonical_K_star()
    assert abs(K_star - 13.43) < 0.05, f"K* mismatch: {K_star:.4f}"


def test_canonical_alpha_max():
    """α̂_max ≈ 6.93% for canonical parameters."""
    a_max = canonical_alpha_max()
    assert abs(a_max - 0.0693) < 0.005, f"α̂_max mismatch: {a_max:.4f}"


def test_K_star_formula():
    """K* = ln(R_max/R_o) / ln(1/O)."""
    O, R_o, R_max = 0.80, 0.05, 1.00
    K_star_expected = np.log(R_max / R_o) / np.log(1.0 / O)
    K_star = endpoint_K_star(O, R_o, R_max)
    assert abs(K_star - K_star_expected) < 1e-10


def test_alpha_max_formula():
    """α̂_max = 1/(1+K*)."""
    O, R_o, R_max = 0.80, 0.05, 1.00
    K_star = endpoint_K_star(O, R_o, R_max)
    a_max = alpha_max(O, R_o, R_max)
    assert abs(a_max - 1.0 / (1.0 + K_star)) < 1e-10


# ---------------------------------------------------------------------------
# Fixed-share boundary curve
# ---------------------------------------------------------------------------

def test_boundary_curve_at_kink():
    """R_K(O) = R_o for any K."""
    O, R_o, K = 0.80, 0.05, 19.0
    Rb = fixed_share_boundary_curve(O, R_o, K, np.array([O]))
    assert abs(float(Rb[0]) - R_o) < 1e-10


def test_boundary_curve_is_increasing():
    """R_K(U) is increasing in U for K > 0."""
    O, R_o, K = 0.80, 0.05, 19.0
    U_grid = np.linspace(O, 1.0, 100)
    Rb = fixed_share_boundary_curve(O, R_o, K, U_grid)
    assert np.all(np.diff(Rb) >= 0), "Boundary curve should be non-decreasing"


def test_boundary_curve_endpoint_K_star():
    """R_K*(1) = R_max exactly."""
    O, R_o, R_max = 0.80, 0.05, 1.00
    K_star = endpoint_K_star(O, R_o, R_max)
    Rb_at_1 = fixed_share_boundary_curve(O, R_o, K_star, np.array([1.0]))
    assert abs(float(Rb_at_1[0]) - R_max) < 1e-6


# ---------------------------------------------------------------------------
# Fresh-depositor profile
# ---------------------------------------------------------------------------

def test_fresh_depositor_profile_at_U():
    """ᾱ(U) = 1 - U at test points."""
    Us = [0.0, 0.5, 0.80, 0.95]
    profiles = fresh_depositor_profile(np.array(Us))
    expected = 1.0 - np.array(Us)
    np.testing.assert_allclose(profiles, expected, atol=1e-10)


def test_fresh_depositor_envelope():
    """E(U) = U/(1-U) at test points."""
    Us = np.array([0.5, 0.8, 0.9])
    E = fresh_depositor_envelope(Us)
    expected = Us / (1.0 - Us)
    np.testing.assert_allclose(E, expected, atol=1e-10)


def test_fresh_depositor_boundary_shape():
    """R_b(U) ∝ 1/(1-U): doubling (1-U) halves the rate."""
    U_a = 0.80; Rb_a = 0.05
    Rb_U = fresh_depositor_boundary(np.array([0.90]), U_a, Rb_a)
    # At U=0.9: (1-0.8)/(1-0.9) = 0.2/0.1 = 2 → rate doubles
    assert abs(float(Rb_U[0]) - 2 * Rb_a) < 1e-8


def test_fresh_depositor_exact_condition_pass():
    """(u_0 - u)*R_b'(u) ≤ R_b(u) for flat curve."""
    assert fresh_depositor_exact_condition(0.9, 0.8, slope=0.0, Rb_u=0.05)


def test_fresh_depositor_exact_condition_fail():
    """Large slope should fail the exact condition."""
    # (0.9 - 0.8) * 10.0 = 1.0 > 0.05
    assert not fresh_depositor_exact_condition(0.9, 0.8, slope=10.0, Rb_u=0.05)


def test_appendix_a_admissibility():
    """Appendix A knot set should be admissible under fresh-depositor rule."""
    us, rs = appendix_a_knots()
    df = fresh_depositor_piecewise_check(us, rs)
    # All segments should be admissible (cap ≥ ratio)
    assert df["admissible"].all(), (
        f"Admissibility failures:\n{df[~df['admissible']]}"
    )


# ---------------------------------------------------------------------------
# Combined guarantee
# ---------------------------------------------------------------------------

def test_combined_profile_max():
    """ᾱ_both(U) = max{ᾱ, 1-U}."""
    alpha_bar = 0.05
    Us = np.array([0.1, 0.5, 0.90, 0.96])
    expected = np.maximum(alpha_bar, 1.0 - Us)
    np.testing.assert_allclose(combined_profile(Us, alpha_bar), expected, atol=1e-10)


def test_combined_envelope_min():
    """E_both(U) = min{K, U/(1-U)}."""
    K = 19.0
    Us = np.array([0.1, 0.5, 0.90, 0.99])
    expected = np.minimum(K, Us / (1.0 - Us))
    np.testing.assert_allclose(combined_envelope(Us, K), expected, atol=1e-10)


def test_combined_base_bound():
    """Base bound formula R_b(0) ≥ R_max * t^K / (K+1)."""
    K = 19.0; R_max = 1.00
    t = K / (K + 1.0)
    expected = R_max * t ** K / (K + 1.0)
    assert abs(combined_base_bound(K, R_max) - expected) < 1e-10


def test_combined_boundary_F_continuity():
    """F(U) is continuous at t = K/(K+1)."""
    K = 19.0
    t = K / (K + 1.0)
    eps = 1e-7
    F_below = float(combined_boundary_F(np.array([t - eps]), K)[0])
    F_above = float(combined_boundary_F(np.array([t + eps]), K)[0])
    assert abs(F_below - F_above) < 0.01, f"F not continuous at t={t:.4f}"


# ---------------------------------------------------------------------------
# Integral admissibility bound
# ---------------------------------------------------------------------------

def test_integral_admissibility_constant_envelope():
    """For constant E(U) = K, integral bound = R_a * (U/U_a)^K."""
    K = 5.0; U_a = 0.80; Rb_a = 0.05
    U = 0.90
    bound = integral_admissibility_bound(U, U_a, Rb_a, lambda v: K)
    expected = Rb_a * (U / U_a) ** K
    assert abs(bound - expected) < 1e-6, f"Bound mismatch: {bound:.6f} vs {expected:.6f}"


# ---------------------------------------------------------------------------
# Smoothing formulas
# ---------------------------------------------------------------------------

def test_smoothstep_endpoints():
    """s(0) = 0, s(1) = 1."""
    assert smoothstep(0.0) == pytest.approx(0.0)
    assert smoothstep(1.0) == pytest.approx(1.0)


def test_smoothstep_monotone():
    """Smoothstep is non-decreasing on [0,1]."""
    t = np.linspace(0, 1, 100)
    s = smoothstep(t)
    assert np.all(np.diff(s) >= -1e-12)


def test_eta_max_w0_equals_K_star():
    """At w=0, η_max = K* = A/L."""
    O, R_o, R_max = 0.80, 0.05, 1.00
    L = np.log(1.0 / O)
    A = np.log(R_max / R_o)
    K_star = A / L
    eta_m = eta_max_from_width(0.0, eta_0=1.0, L=L, A=A)
    assert abs(eta_m - K_star) < 1e-10


def test_eta_max_increases_with_w():
    """Smoother transitions require larger plateau elasticity."""
    O, R_o, R_max, eta_0 = 0.80, 0.05, 1.00, 1.0
    L = np.log(1.0 / O)
    A = np.log(R_max / R_o)
    widths = [0.0, 0.05, 0.10]
    eta_maxs = [eta_max_from_width(w, eta_0, L, A) for w in widths]
    # Should be monotonically increasing in w (since K* > η_0)
    assert all(eta_maxs[i] <= eta_maxs[i + 1] for i in range(len(eta_maxs) - 1))


def test_table5_structure():
    """Table 5 has expected columns and number of rows."""
    df = table5(widths=[0.0, 0.05, 0.10])
    assert "eta_max" in df.columns
    assert "alpha_safe_%" in df.columns
    assert len(df) == 3


def test_table5_alpha_safe_in_range():
    """Protected share must be in (0, 100) for all finite rows."""
    df = table5(widths=[0.0, 0.05, 0.10])
    finite_mask = df["alpha_safe_%"].notna() & np.isfinite(df["alpha_safe_%"])
    valid = df.loc[finite_mask, "alpha_safe_%"]
    assert (valid > 0).all()
    assert (valid < 100).all()


def test_feasibility_width_zero_for_K_below_K_star():
    """K ≤ K* means no width is feasible (return 0)."""
    O, R_o, R_max, eta_0 = 0.80, 0.05, 1.00, 1.0
    L = np.log(1.0 / O)
    A = np.log(R_max / R_o)
    K_star = A / L
    w_feas = feasibility_width_bound(K_star - 0.01, K_star, eta_0, L)
    assert w_feas == 0.0


def test_d_eta_max_dw_positive():
    """Since K* > η_0, d η_max/dw > 0 (increasing)."""
    O, R_o, R_max, eta_0 = 0.80, 0.05, 1.00, 1.0
    L = np.log(1.0 / O)
    A = np.log(R_max / R_o)
    deriv = d_eta_max_dw(0.05, eta_0, L, A)
    assert deriv > 0
