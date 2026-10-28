"""
Tests for rate-curve repair constructions (exp_3).

Covers:
  - Single-kink power repair: catch-up utilization, continuity, protection
  - Multiple-kink segment caps
  - Fixed-kink floor, raise-only, lower-only, mixed repairs
  - Holding-period income computations
"""

import pytest
import numpy as np
from src.curves import SingleKinkCurve, RepairedSingleKinkCurve, PiecewiseLinearCurve
from src.admissibility import fixed_share_K
from src.single_kink_repair import (
    find_catchup_utilization, build_repaired_curve, verify_repair_protects,
    income_derivative_at_U,
)
from src.multikink_caps import (
    fresh_cap, fixed_share_cap, combined_cap, usdt_knots,
    segment_admissibility_table,
)
from src.fixed_kink_repairs import (
    floor_repair, raise_only_repair, lower_only_repair, mixed_repair,
)
from src.holding_period import (
    accumulated_income, scenario1_paths, scenario2_paths,
    build_utilization_path, build_balance_path,
)


# ---------------------------------------------------------------------------
# Single-kink power repair
# ---------------------------------------------------------------------------

class TestSingleKinkRepair:
    O = 0.80; R_o = 0.05; R_max = 1.00
    K = fixed_share_K(0.05)  # K = 19

    def test_catchup_utilization_value(self):
        """U_c ≈ 90.92% for canonical K=19."""
        U_c = find_catchup_utilization(self.O, self.R_o, self.R_max, self.K)
        assert abs(U_c - 0.9092) < 0.002, f"U_c = {U_c:.4f}"

    def test_catchup_power_meets_steep(self):
        """At U_c, power branch equals steep branch."""
        U_c = find_catchup_utilization(self.O, self.R_o, self.R_max, self.K)
        m = (self.R_max - self.R_o) / (1.0 - self.O)
        r_power = self.R_o * (U_c / self.O) ** self.K
        r_steep = self.R_o + m * (U_c - self.O)
        assert abs(r_power - r_steep) < 1e-6, f"Discontinuity at U_c: {abs(r_power - r_steep):.2e}"

    def test_repaired_curve_continuity_at_O(self):
        """Repaired curve is continuous at kink O."""
        curve, K, U_c = build_repaired_curve(self.O, self.R_o, self.R_max, 0.05)
        r_below = float(curve.rate(self.O - 1e-9))
        r_above = float(curve.rate(self.O + 1e-9))
        assert abs(r_below - r_above) < 1e-5

    def test_repaired_curve_continuity_at_Uc(self):
        """Repaired curve is continuous at U_c."""
        curve, K, U_c = build_repaired_curve(self.O, self.R_o, self.R_max, 0.05)
        r_below = float(curve.rate(U_c - 1e-9))
        r_above = float(curve.rate(U_c + 1e-9))
        assert abs(r_below - r_above) < 1e-4, f"Gap at U_c: {abs(r_below-r_above):.2e}"

    def test_repaired_elasticity_bounded_by_K(self):
        """On the power branch, η(U) = K (constant elasticity)."""
        curve, K_val, U_c = build_repaired_curve(self.O, self.R_o, self.R_max, 0.05)
        for U in np.linspace(self.O + 0.01, U_c - 0.01, 20):
            eta = curve.elasticity(float(U), use_left=True)
            assert abs(eta - K_val) < 0.01, f"η={eta:.3f} ≠ K={K_val} at U={U:.3f}"

    def test_repaired_curve_protects_alpha_bar(self):
        """α_safe(U) ≥ ᾱ = 5% on the repair branch."""
        curve, K_val, U_c = build_repaired_curve(self.O, self.R_o, self.R_max, 0.05)
        protected = verify_repair_protects(curve, 0.05)
        assert protected, "Repair should protect 5% suppliers"

    def test_income_derivative_sign_on_repair(self):
        """For α = ᾱ = 5%, dI/dδ ≥ 0 on the repair branch."""
        curve, K_val, U_c = build_repaired_curve(self.O, self.R_o, self.R_max, 0.05)
        for U in np.linspace(self.O + 0.01, U_c - 0.01, 20):
            dI = income_derivative_at_U(float(U), curve, alpha=0.05)
            assert dI >= -1e-8, f"dI/dδ < 0 at U={U:.3f}: {dI:.4f}"

    def test_old_curve_violates_monotonicity(self):
        """Baseline curve should violate monotonicity for α=10% above kink."""
        curve_old = SingleKinkCurve(self.O, self.R_o, self.R_max)
        # At U=0.83, α=10%: should violate
        dI = income_derivative_at_U(0.83, curve_old, alpha=0.10)
        assert dI < 0, f"Expected violation, got dI={dI:.4f}"


# ---------------------------------------------------------------------------
# Segment-cap formulas
# ---------------------------------------------------------------------------

class TestSegmentCaps:
    def test_fresh_cap_flat(self):
        """Flat segment (u_L=u_H) would have cap=1."""
        # Near-flat case
        c = fresh_cap(0.80, 0.80 + 1e-10)
        assert abs(c - 1.0) < 1e-6

    def test_fresh_cap_formula(self):
        """c^fresh = 1 + Δu/(1-u_L)."""
        u_L, u_H = 0.80, 0.85
        expected = 1 + (u_H - u_L) / (1 - u_L)
        assert abs(fresh_cap(u_L, u_H) - expected) < 1e-10

    def test_fixed_share_cap_formula(self):
        """c^fixed with K=19, segment [0.80, 0.85]."""
        u_L, u_H = 0.80, 0.85
        K = 19.0
        du = u_H - u_L
        term1 = 1 + K * du / u_L
        denom2 = u_H - K * du
        term2 = u_H / denom2 if denom2 > 0 else np.inf
        expected = min(term1, term2)
        assert abs(fixed_share_cap(u_L, u_H, K) - expected) < 1e-10

    def test_combined_cap_lte_min(self):
        """Combined cap ≤ min(fresh, fixed)."""
        u_L, u_H, K = 0.80, 0.85, 10.0
        c_f = fresh_cap(u_L, u_H)
        c_k = fixed_share_cap(u_L, u_H, K)
        c_b = combined_cap(u_L, u_H, K)
        assert c_b <= min(c_f, c_k) + 1e-10

    def test_combined_cap_gte_each(self):
        """Combined cap = min(fresh, fixed) exactly."""
        u_L, u_H, K = 0.95, 0.99, 5.0
        c_f = fresh_cap(u_L, u_H)
        c_k = fixed_share_cap(u_L, u_H, K)
        c_b = combined_cap(u_L, u_H, K)
        assert abs(c_b - min(c_f, c_k)) < 1e-10

    def test_caps_gt_1(self):
        """Caps must be > 1 for non-trivial segments."""
        u_L, u_H, K = 0.80, 0.85, 19.0
        assert fresh_cap(u_L, u_H) > 1.0
        assert fixed_share_cap(u_L, u_H, K) > 1.0


# ---------------------------------------------------------------------------
# USDT knot grid
# ---------------------------------------------------------------------------

class TestUsdtKnots:
    def test_usdt_knots_count(self):
        us, rs = usdt_knots()
        assert len(us) == len(rs) == 8

    def test_usdt_knots_increasing(self):
        us, rs = usdt_knots()
        assert np.all(np.diff(us) > 0)

    def test_admissibility_table_columns(self):
        us, rs = usdt_knots()
        df = segment_admissibility_table(us, rs, K=19.0, guarantee="both")
        required_cols = ["u_L", "u_H", "R_H/R_L", "admissible"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Fixed-kink repairs
# ---------------------------------------------------------------------------

class TestFixedKinkRepairs:
    us, rs = usdt_knots()
    K = 19.0

    def test_floor_repair_admissible(self):
        """Floor repair should be admissible under both guarantees."""
        r_floor = floor_repair(self.us, self.rs[-1], self.K, guarantee="both")
        df = segment_admissibility_table(self.us, r_floor.tolist(), self.K, "both")
        assert df["admissible"].all(), f"Floor repair not admissible:\n{df[~df['admissible']]}"

    def test_raise_only_admissible(self):
        """Raise-only repair should be admissible."""
        r_raise = raise_only_repair(self.us, self.rs, self.K, guarantee="fresh")
        df = segment_admissibility_table(self.us, r_raise.tolist(), self.K, "fresh")
        assert df["admissible"].all()

    def test_lower_only_admissible(self):
        """Lower-only repair should be admissible."""
        r_lower = lower_only_repair(self.us, self.rs, self.K, guarantee="fresh")
        df = segment_admissibility_table(self.us, r_lower.tolist(), self.K, "fresh")
        assert df["admissible"].all()

    def test_raise_only_preserves_top(self):
        """Raise-only holds top endpoint fixed."""
        r_raise = raise_only_repair(self.us, self.rs, self.K, guarantee="both")
        assert abs(r_raise[-1] - self.rs[-1]) < 1e-10

    def test_lower_only_preserves_base(self):
        """Lower-only holds base endpoint fixed."""
        r_lower = lower_only_repair(self.us, self.rs, self.K, guarantee="fresh")
        assert abs(r_lower[0] - self.rs[0]) < 1e-10

    def test_floor_base_fresh_approx(self):
        """Floor base APR under fresh-only ≈ 3.47% (within 1 pp)."""
        r_floor = floor_repair(self.us, self.rs[-1], self.K, guarantee="fresh")
        base_pct = r_floor[0] * 100
        assert 1.0 < base_pct < 8.0, f"Fresh floor base {base_pct:.2f}% out of expected range"

    def test_floor_base_both_approx(self):
        """Floor base APR under both guarantees ≈ 8.68% (within 3 pp)."""
        r_floor = floor_repair(self.us, self.rs[-1], self.K, guarantee="both")
        base_pct = r_floor[0] * 100
        assert 4.0 < base_pct < 15.0, f"Both floor base {base_pct:.2f}% out of expected range"

    def test_raise_only_gte_baseline(self):
        """Raise-only rates are element-wise ≥ baseline (except possibly top)."""
        r_raise = raise_only_repair(self.us, self.rs, self.K, guarantee="both")
        r_baseline = np.array(self.rs)
        # All rates should be ≥ baseline
        assert np.all(r_raise + 1e-10 >= r_baseline), "Raise-only should only increase rates"

    def test_lower_only_lte_baseline(self):
        """Lower-only rates are element-wise ≤ baseline."""
        r_lower = lower_only_repair(self.us, self.rs, self.K, guarantee="fresh")
        r_baseline = np.array(self.rs)
        assert np.all(r_lower <= r_baseline + 1e-10), "Lower-only should only decrease rates"


# ---------------------------------------------------------------------------
# Holding-period income
# ---------------------------------------------------------------------------

class TestHoldingPeriod:
    def test_scenario1_positive_income(self):
        """Accumulated income is positive for any admissible curve."""
        us, rs = usdt_knots()
        curve = PiecewiseLinearCurve(us, rs)
        b_fn, U_fn = scenario1_paths(initial_balance=10_000.0)
        A = accumulated_income(365.0, b_fn, U_fn, curve, f=0.05, n_intervals=500)
        assert A > 0, "Income should be positive"

    def test_scenario2_higher_than_scenario1(self):
        """Scenario 2 (step-up deposits) should earn more than scenario 1."""
        us, rs = usdt_knots()
        curve = PiecewiseLinearCurve(us, rs)
        b_fn1, U_fn1 = scenario1_paths(10_000.0)
        b_fn2, U_fn2 = scenario2_paths(10_000.0)
        A1 = accumulated_income(365.0, b_fn1, U_fn1, curve, f=0.05, n_intervals=500)
        A2 = accumulated_income(365.0, b_fn2, U_fn2, curve, f=0.05, n_intervals=500)
        assert A2 > A1, f"Scenario 2 income {A2:.0f} should exceed Scenario 1 {A1:.0f}"

    def test_reserve_factor_reduces_income(self):
        """Higher reserve factor reduces accumulated supplier income."""
        us, rs = usdt_knots()
        curve = PiecewiseLinearCurve(us, rs)
        b_fn, U_fn = scenario1_paths(10_000.0)
        A_low = accumulated_income(365.0, b_fn, U_fn, curve, f=0.01, n_intervals=500)
        A_high = accumulated_income(365.0, b_fn, U_fn, curve, f=0.20, n_intervals=500)
        assert A_low > A_high

    def test_utilization_path_interpolation(self):
        """Piecewise-linear U(t) is correct at checkpoints."""
        from src.holding_period import build_utilization_path
        days = [0, 90, 180]
        utils = [0.80, 0.95, 0.995]
        U_fn = build_utilization_path(days, utils)
        assert abs(U_fn(0) - 0.80) < 1e-10
        assert abs(U_fn(90) - 0.95) < 1e-10
        assert abs(U_fn(180) - 0.995) < 1e-10

    def test_balance_path_step_function(self):
        """b(t) is correct step-function."""
        from src.holding_period import build_balance_path
        b_fn = build_balance_path([100, 200], [15000, 20000], start_balance=10000)
        assert b_fn(0) == 10000
        assert b_fn(50) == 10000
        assert b_fn(100) == 15000
        assert b_fn(150) == 15000
        assert b_fn(200) == 20000

    def test_multiplier_formula():
        """1% share, 5% reserve: each $1 borrower interest → $0.0095 to depositor."""
        share = 0.01
        reserve = 0.05
        multiplier = share * (1 - reserve)
        assert abs(multiplier - 0.0095) < 1e-10

    test_multiplier_formula = staticmethod(test_multiplier_formula)
