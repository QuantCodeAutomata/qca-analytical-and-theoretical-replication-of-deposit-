"""
Rate-curve classes for DeFi lending-market deposit-monotonicity analysis.

Supports evaluation of R_b(U), left/right derivative, and elasticity η(U)
for single-kink and piecewise-linear curves.

All quantities are in decimal form (e.g. 5% → 0.05, 80% → 0.80).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class RateCurve:
    """Abstract base for borrow-rate curves."""

    def rate(self, U: float | np.ndarray) -> float | np.ndarray:
        """Return R_b(U) for scalar or array U."""
        raise NotImplementedError

    def slope_left(self, U: float) -> float:
        """Return left derivative dR_b/dU at U (slope of segment below U)."""
        raise NotImplementedError

    def slope_right(self, U: float) -> float:
        """Return right derivative dR_b/dU at U (slope of segment above U)."""
        raise NotImplementedError

    def elasticity(self, U: float, use_left: bool = True) -> float:
        """Return η(U) = U / R_b(U) * dR_b/dU.

        When a deposit is made utilization *falls*, so we always use the
        slope of the segment *below* U (i.e. the left derivative).  Pass
        use_left=False only for diagnostic purposes.
        """
        Rb = self.rate(float(U))
        if use_left:
            dR = self.slope_left(U)
        else:
            dR = self.slope_right(U)
        return float(U) * dR / Rb

    def alpha_safe(self, U: float) -> float:
        """Return α_safe(U) = 1 / (1 + η(U)).

        Uses below-kink (left) slope per the deposit-monotonicity convention.
        """
        eta = self.elasticity(U, use_left=True)
        return 1.0 / (1.0 + eta)

    def supply_apr(self, U: float | np.ndarray) -> float | np.ndarray:
        """Return supply APR R_s(U) = U * R_b(U)."""
        return np.asarray(U) * self.rate(U)


# ---------------------------------------------------------------------------
# Single-kink (standard) curve
# ---------------------------------------------------------------------------

class SingleKinkCurve(RateCurve):
    """Standard single-kink borrow-rate curve.

    R_b(U) = R_o * (U / O)           for U ≤ O  (linear ramp)
    R_b(U) = R_o + m * (U - O)       for U > O  (steep linear branch)

    where m = (R_max - R_o) / (1 - O).
    """

    def __init__(self, O: float, R_o: float, R_max: float) -> None:
        """
        Parameters
        ----------
        O    : optimal utilization (kink point)
        R_o  : borrow APR at kink
        R_max: maximum borrow APR (at U = 1)
        """
        assert 0 < O < 1, "O must be in (0,1)"
        assert R_o > 0, "R_o must be positive"
        assert R_max > R_o, "R_max must exceed R_o"
        self.O = O
        self.R_o = R_o
        self.R_max = R_max
        self.m = (R_max - R_o) / (1.0 - O)

    def rate(self, U: float | np.ndarray) -> float | np.ndarray:
        U = np.asarray(U, dtype=float)
        below = self.R_o * (U / self.O)
        above = self.R_o + self.m * (U - self.O)
        return np.where(U <= self.O, below, above)

    def slope_left(self, U: float) -> float:
        """Slope of the segment whose right endpoint is U (or at U if interior)."""
        U = float(U)
        if U <= self.O:
            return self.R_o / self.O          # slope of ramp below kink
        else:
            return self.m                     # slope of steep branch

    def slope_right(self, U: float) -> float:
        """Slope of the segment whose left endpoint is U."""
        U = float(U)
        if U < self.O:
            return self.R_o / self.O
        else:
            return self.m

    def elasticity(self, U: float, use_left: bool = True) -> float:
        """η(U) using left or right slope.

        At the kink O:  left slope gives η = 1 (proportional ramp),
        right slope gives η = O * m / R_o (much larger).
        A deposit lowers U, so left slope is appropriate.
        """
        U = float(U)
        Rb = float(self.rate(U))
        dR = self.slope_left(U) if use_left else self.slope_right(U)
        return U * dR / Rb


# ---------------------------------------------------------------------------
# Power-law curve (used for fixed-share protection branch)
# ---------------------------------------------------------------------------

class PowerKinkCurve(RateCurve):
    """Boundary curve R_K(U) = R_o * (U/O)^K for U ≥ O.

    This is the maximal borrow-rate curve that maintains a fixed-share
    protection guarantee ᾱ = 1/(1+K).
    """

    def __init__(self, O: float, R_o: float, K: float) -> None:
        self.O = O
        self.R_o = R_o
        self.K = K

    def rate(self, U: float | np.ndarray) -> float | np.ndarray:
        U = np.asarray(U, dtype=float)
        return self.R_o * (U / self.O) ** self.K

    def slope_left(self, U: float) -> float:
        U = float(U)
        return self.K * self.R_o * (U / self.O) ** (self.K - 1) / self.O

    def slope_right(self, U: float) -> float:
        return self.slope_left(U)

    def elasticity(self, U: float, use_left: bool = True) -> float:
        """η(U) = K everywhere on the power branch (constant elasticity)."""
        return self.K


# ---------------------------------------------------------------------------
# Piecewise-linear curve
# ---------------------------------------------------------------------------

class PiecewiseLinearCurve(RateCurve):
    """General piecewise-linear borrow-rate curve defined by knot pairs.

    Parameters
    ----------
    utilizations : array-like of length n+1, strictly increasing, first entry 0 or ≥ 0
    rates        : array-like of length n+1, borrow APRs at each knot
    """

    def __init__(self, utilizations: list[float], rates: list[float]) -> None:
        us = np.asarray(utilizations, dtype=float)
        rs = np.asarray(rates, dtype=float)
        assert us.shape == rs.shape, "utilizations and rates must have same length"
        assert np.all(np.diff(us) > 0), "utilization knots must be strictly increasing"
        assert np.all(rs >= 0), "all rates must be non-negative"
        self.us = us
        self.rs = rs
        # Segment slopes: m[i] = (r[i+1]-r[i]) / (u[i+1]-u[i])
        self._slopes = np.diff(rs) / np.diff(us)

    def _segment_index(self, U: float) -> int:
        """Return index i such that us[i] <= U < us[i+1] (or last segment)."""
        U = float(U)
        idx = np.searchsorted(self.us, U, side='right') - 1
        return int(np.clip(idx, 0, len(self._slopes) - 1))

    def rate(self, U: float | np.ndarray) -> float | np.ndarray:
        scalar = np.isscalar(U)
        U = np.atleast_1d(np.asarray(U, dtype=float))
        out = np.empty_like(U)
        for j, u in enumerate(U):
            i = int(np.clip(np.searchsorted(self.us, u, side='right') - 1,
                            0, len(self._slopes) - 1))
            out[j] = self.rs[i] + self._slopes[i] * (u - self.us[i])
        return float(out[0]) if scalar else out

    def slope_left(self, U: float) -> float:
        """Slope of segment whose right endpoint is at or after U.

        For deposit monotonicity (deposit lowers U) we need slope of the
        segment immediately below U, which is slope_left.
        """
        U = float(U)
        if U <= self.us[0]:
            return self._slopes[0]
        i = int(np.searchsorted(self.us, U, side='left')) - 1
        i = int(np.clip(i, 0, len(self._slopes) - 1))
        return float(self._slopes[i])

    def slope_right(self, U: float) -> float:
        """Slope of segment immediately to the right of U."""
        U = float(U)
        i = int(np.searchsorted(self.us, U, side='right')) - 1
        i = int(np.clip(i, 0, len(self._slopes) - 1))
        return float(self._slopes[i])

    def elasticity(self, U: float, use_left: bool = True) -> float:
        """η(U) = U * slope / R_b(U) using left or right slope."""
        U = float(U)
        Rb = float(self.rate(U))
        if Rb <= 0:
            return 0.0
        dR = self.slope_left(U) if use_left else self.slope_right(U)
        return U * dR / Rb

    def segment_slopes(self) -> np.ndarray:
        """Return array of segment slopes m_i."""
        return self._slopes.copy()

    def n_segments(self) -> int:
        return len(self._slopes)


# ---------------------------------------------------------------------------
# Repaired single-kink curve  (used in exp_3)
# ---------------------------------------------------------------------------

class RepairedSingleKinkCurve(RateCurve):
    """Single-kink curve with a power-law repair on (O, U_c).

    R_new(U) = R_o * U/O              for U ≤ O
               R_o * (U/O)^K          for O < U < U_c
               R_o + m * (U - O)      for U_c ≤ U ≤ 1
    """

    def __init__(self, O: float, R_o: float, R_max: float, K: float, U_c: float) -> None:
        self.O = O
        self.R_o = R_o
        self.R_max = R_max
        self.K = K
        self.U_c = U_c
        self.m = (R_max - R_o) / (1.0 - O)

    def rate(self, U: float | np.ndarray) -> float | np.ndarray:
        U = np.asarray(U, dtype=float)
        ramp   = self.R_o * (U / self.O)
        power  = self.R_o * (U / self.O) ** self.K
        steep  = self.R_o + self.m * (U - self.O)
        result = np.where(U <= self.O, ramp,
                 np.where(U < self.U_c, power, steep))
        return result

    def slope_left(self, U: float) -> float:
        U = float(U)
        if U <= self.O:
            return self.R_o / self.O
        elif U < self.U_c:
            return self.K * self.R_o * (U / self.O) ** (self.K - 1) / self.O
        else:
            return self.m

    def slope_right(self, U: float) -> float:
        U = float(U)
        if U < self.O:
            return self.R_o / self.O
        elif U <= self.U_c:
            return self.K * self.R_o * (U / self.O) ** (self.K - 1) / self.O
        else:
            return self.m

    def elasticity(self, U: float, use_left: bool = True) -> float:
        U = float(U)
        Rb = float(self.rate(U))
        dR = self.slope_left(U) if use_left else self.slope_right(U)
        return U * dR / Rb
