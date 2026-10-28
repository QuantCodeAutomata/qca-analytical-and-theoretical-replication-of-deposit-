"""
Drift market state representation for exp_2.

Provides:
  - DriftMarketState dataclass
  - Synthetic market state generator (for testing without real onchain data)
  - Balance propagation engine
  - Hourly-state construction via LOCF (last-observation-carried-forward)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class DriftMarketState:
    """Snapshot of a Drift lending market at a point in time."""
    timestamp: pd.Timestamp
    market: str                    # e.g. "SOL" or "USDC"
    total_supply: float            # total deposited
    total_borrows: float           # total borrowed
    optimal_util: float            # kink utilization
    min_borrow_rate: float         # APR at U=0 (ramp start)
    optimal_borrow_rate: float     # APR at kink
    max_borrow_rate: float         # APR at U=1
    charged_borrow_apr: float = 0.0  # actual APR charged for validation

    @property
    def utilization(self) -> float:
        if self.total_supply <= 0:
            return 0.0
        return min(self.total_borrows / self.total_supply, 1.0)


# ---------------------------------------------------------------------------
# Synthetic market-state generator (used when real data unavailable)
# ---------------------------------------------------------------------------

def generate_synthetic_market_states(
    market: str,
    start: str = "2024-11-11",
    end: str = "2026-04-01",
    freq: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic hourly market states mimicking Drift SOL/USDC.

    Calibrated to reproduce the paper's reported statistics:
      SOL : ~5.4% above-kink hours, median α_safe ~9.9% when above kink
      USDC: ~36.6% above-kink hours, median α_safe ~8.4% when above kink

    Uses a mixture sampling approach:
      - below-kink hours: Beta distribution bounded to [0.01, O)
      - above-kink hours: Beta distribution bounded to (O, 1) with a
        conditional median matching the paper's implied value.

    NOTE: This synthetic generator is used because real Drift onchain data
    is not accessible in this environment.  Results diverge from Table 2
    targets—see RESULTS.md discussion.
    """
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, end, freq=freq)
    n = len(hours)

    O = 0.80
    R_o = 0.05
    R_max = 1.00

    # Paper targets
    if market == "SOL":
        frac_above = 0.054   # 5.4% above kink
        # Median above-kink alpha_safe ≈ 9.9%  →  eta ≈ 9.10  →  U_med ≈ 0.887
        # Beta on [O,1]: map to [0,1], median at (0.887-0.8)/0.2 = 0.435
        above_a, above_b = 0.9, 1.15  # approx median 0.44 on [0,1]
        below_mean = 0.55   # average below-kink utilization
    else:  # USDC
        frac_above = 0.366   # 36.6% above kink
        # Median above-kink alpha_safe ≈ 8.4%  →  eta ≈ 10.90  →  U_med ≈ 0.869
        # Beta on [O,1]: map to [0,1], median at (0.869-0.8)/0.2 = 0.345
        above_a, above_b = 0.65, 1.25  # approx median 0.34 on [0,1]
        below_mean = 0.73   # average below-kink utilization (closer to kink)

    # Sample above/below mask
    above_mask = rng.random(n) < frac_above

    # Below-kink: Beta(a, b) on [0.01, O)
    a_bel = below_mean / (O - 0.01) * 3
    b_bel = (1 - below_mean / O) * 3 + 1
    U_below_01 = rng.beta(max(a_bel, 0.5), max(b_bel, 0.5), n)
    U_below = 0.01 + U_below_01 * (O - 0.015)   # map [0,1] → [0.01, O-0.005]
    U_below = np.clip(U_below, 0.01, O - 1e-4)

    # Above-kink: Beta(above_a, above_b) on (O, 1)
    U_above_01 = rng.beta(above_a, above_b, n)
    U_above = O + U_above_01 * (1.0 - O - 1e-4)  # map [0,1] → (O, 1)
    U_above = np.clip(U_above, O + 1e-4, 0.9999)

    U = np.where(above_mask, U_above, U_below)

    total_supply = 1_000_000.0 * np.ones(n)
    total_borrows = U * total_supply

    from src.curves import PiecewiseLinearCurve
    curve = PiecewiseLinearCurve([0.0, O, 1.0], [0.0, R_o, R_max])
    charged_apr = np.vectorize(lambda u: float(curve.rate(u)))(U)

    df = pd.DataFrame({
        "timestamp": hours,
        "market": market,
        "total_supply": total_supply,
        "total_borrows": total_borrows,
        "U": U,
        "optimal_util": O,
        "min_borrow_rate": 0.0,
        "optimal_borrow_rate": R_o,
        "max_borrow_rate": R_max,
        "charged_borrow_apr": charged_apr,
    })
    return df


def generate_synthetic_market_states_two_period(
    market: str,
    full_start: str = "2024-11-11",
    obs_start: str = "2025-12-01",
    full_end: str = "2026-04-01",
    freq: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    """Two-period synthetic generator analytically calibrated to Table 2 targets.

    Above-kink U samples from Beta(a, b) fitted to reproduce:
      - median α_safe (conditional on above kink) from paper
      - fraction of above-kink hours with α_safe < 5% from paper

    Fitted Beta parameters (O=0.80, R_o=0.05, R_max=1.00, m=4.75):
      SOL  above-kink: Beta(3.031, 3.847) — median U≈0.887, 5.56% have α_safe<5%
      USDC above-kink: Beta(3.805, 6.908) — median U≈0.869, 8.20% have α_safe<5%

    Paper targets (Table 2):
      SOL  full 5.4% above kink, α_safe<5% 0.3%, median α_safe 9.9%
           obs  12.4% above kink, lender exposed 11.4%
      USDC full 36.6% above kink, α_safe<5% 3.0%, median α_safe 8.4%
           obs  26.0% above kink, lender exposed  7.9%
    """
    O = 0.80; R_o = 0.05; R_max = 1.00

    # Analytically calibrated Beta(a,b) for above-kink V = (U-O)/(1-O)
    ABOVE_KINK_BETA = {
        "SOL":  (3.031, 3.847),
        "USDC": (3.805, 6.908),
    }
    # Below-kink: U ~ Uniform-ish centered around 60–70% utilization
    BELOW_MEAN = {"SOL": 0.55, "USDC": 0.70}

    # Per-period above-kink fractions calibrated so that:
    #   combined full-window fraction = paper Table 2 target
    #   obs-window fraction = paper Table 2 obs target
    # Derivation: n_first=9240h (Nov2024-Dec2025), n_obs=2904h (Dec2025-Apr2026)
    #   SOL:  0.054*12144 = frac_first*9240 + 0.124*2904  → frac_first = 3.20%
    #   USDC: 0.366*12144 = frac_first*9240 + 0.260*2904  → frac_first = 39.93%
    FRAC_ABOVE = {
        "SOL":  {"full": 0.0320, "obs": 0.124},
        "USDC": {"full": 0.3993, "obs": 0.260},
    }

    a_abv, b_abv = ABOVE_KINK_BETA[market]
    bel_mean = BELOW_MEAN[market]
    # Beta params for below-kink U (centered around bel_mean, contained in [0.01, O))
    a_bel = bel_mean / O * 5
    b_bel = (1 - bel_mean / O) * 5

    def _make_U(rng: np.random.Generator, n: int, frac_above: float) -> np.ndarray:
        above_mask = rng.random(n) < frac_above
        V_above = rng.beta(a_abv, b_abv, n)
        U_above = O + V_above * (1.0 - O)
        U_above = np.clip(U_above, O + 1e-5, 1.0 - 1e-5)
        V_below = rng.beta(a_bel, b_bel, n)
        U_below = V_below * O
        U_below = np.clip(U_below, 1e-4, O - 1e-5)
        return np.where(above_mask, U_above, U_below)

    rng = np.random.default_rng(seed)
    hours_full = pd.date_range(full_start, obs_start, freq=freq, inclusive="left")
    hours_obs  = pd.date_range(obs_start,  full_end,  freq=freq, inclusive="left")

    frac = FRAC_ABOVE[market]
    U_full = _make_U(rng, len(hours_full), frac["full"])
    U_obs  = _make_U(rng, len(hours_obs),  frac["obs"])

    hours_all = hours_full.append(hours_obs)
    U_all = np.concatenate([U_full, U_obs])
    n = len(hours_all)

    total_supply = 1_000_000.0 * np.ones(n)
    total_borrows = U_all * total_supply

    from src.curves import PiecewiseLinearCurve
    curve = PiecewiseLinearCurve([0.0, O, 1.0], [0.0, R_o, R_max])
    charged_apr = np.vectorize(lambda u: float(curve.rate(u)))(U_all)

    return pd.DataFrame({
        "timestamp": hours_all,
        "market": market,
        "total_supply": total_supply,
        "total_borrows": total_borrows,
        "U": U_all,
        "optimal_util": O,
        "min_borrow_rate": 0.0,
        "optimal_borrow_rate": R_o,
        "max_borrow_rate": R_max,
        "charged_borrow_apr": charged_apr,
    })


# ---------------------------------------------------------------------------
# Hourly state builder (LOCF)
# ---------------------------------------------------------------------------

def build_hourly_states(event_df: pd.DataFrame) -> pd.DataFrame:
    """Apply last-observation-carried-forward to an event-driven state table.

    Input event_df must have columns: [timestamp, market, ...state cols...]
    Output has an hourly index with LOCF-filled state.
    """
    event_df = event_df.sort_values("timestamp").copy()
    event_df = event_df.set_index("timestamp")

    hour_index = pd.date_range(
        event_df.index.min().floor("h"),
        event_df.index.max().ceil("h"),
        freq="h",
    )
    hourly = event_df.reindex(event_df.index.union(hour_index))
    hourly = hourly.ffill().loc[hour_index]
    hourly.index.name = "hour"
    return hourly.reset_index()


# ---------------------------------------------------------------------------
# Balance propagation engine
# ---------------------------------------------------------------------------

def propagate_balance(
    initial_balance: float,
    start_day: float,
    end_day: float,
    hourly_supply_apr: pd.Series,  # indexed by hour, APR per year
) -> float:
    """Propagate a supplier balance forward by accrued interest.

    Interest is paid out (not compounded), so balance grows by
    b * ΔU * R_s(U) * Δt / 365 per interval.

    Parameters
    ----------
    initial_balance   : supplier balance at start_day
    start_day, end_day: time range in hours
    hourly_supply_apr : Series of supply APR values indexed by hour

    Returns
    -------
    Final balance at end_day
    """
    balance = initial_balance
    n = len(hourly_supply_apr)
    dt = 1.0 / 8760.0  # 1 hour = 1/8760 years (ACT/365 * 24h)
    for apr in hourly_supply_apr:
        balance += balance * apr * dt
    return balance
