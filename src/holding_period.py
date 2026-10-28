"""
Holding-period supplier-income comparisons.

Computes accumulated supplier interest A(H) = ∫_0^H (1-f) b(t) U(t) R_b(U(t)) dt

Interest is paid out (not compounded), so b(t) evolves only from explicit
deposit/withdrawal events.  U(t) is imposed exogenously (piecewise-linear
interpolation between daily checkpoints).

All values: rates in decimal, balances in monetary units.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate


# ---------------------------------------------------------------------------
# Piecewise-linear utilization path
# ---------------------------------------------------------------------------

def build_utilization_path(
    days: list[int],
    utils: list[float],
) -> callable:
    """Build a piecewise-linear U(t) interpolant from checkpoint data.

    Parameters
    ----------
    days  : days from start [0, 90, 180, ...]
    utils : utilization at each checkpoint

    Returns
    -------
    Callable U(t) → float for t in [0, H]
    """
    days_arr = np.array(days, dtype=float)
    utils_arr = np.array(utils, dtype=float)

    def U_t(t: float) -> float:
        return float(np.interp(t, days_arr, utils_arr))

    return U_t


# ---------------------------------------------------------------------------
# Piecewise-constant balance path
# ---------------------------------------------------------------------------

def build_balance_path(
    events_days: list[int],
    event_balances: list[float],
    start_balance: float = 0.0,
) -> callable:
    """Build a step-function b(t) from deposit/withdrawal events.

    Parameters
    ----------
    events_days    : days from start at which balance changes
    event_balances : new balance after each event
    start_balance  : initial balance before first event

    Returns
    -------
    Callable b(t) → float
    """
    days_arr = np.array([0.0] + list(events_days), dtype=float)
    bals_arr = np.array([start_balance] + list(event_balances), dtype=float)

    def b_t(t: float) -> float:
        idx = int(np.searchsorted(days_arr, t, side='right')) - 1
        idx = max(0, min(idx, len(bals_arr) - 1))
        return float(bals_arr[idx])

    return b_t


# ---------------------------------------------------------------------------
# Accumulated income integration
# ---------------------------------------------------------------------------

def accumulated_income(
    H: float,
    b_fn,
    U_fn,
    curve,
    f: float = 0.05,
    n_intervals: int = 1000,
) -> float:
    """Compute A(H) = ∫_0^H (1-f) b(t) U(t) R_b(U(t)) dt.

    Uses numerical integration with uniform sub-intervals (trapezoidal rule).

    Parameters
    ----------
    H          : horizon in days (e.g. 365)
    b_fn       : callable b(t) supplier balance at day t
    U_fn       : callable U(t) utilization at day t
    curve      : borrow-rate curve with .rate(U) method
    f          : reserve factor (e.g. 0.05 for 5%)
    n_intervals: number of quadrature sub-intervals

    Returns
    -------
    Accumulated interest in monetary units (per ACT/365 basis)
    """
    ts = np.linspace(0.0, H, n_intervals + 1)
    vals = np.array([
        (1.0 - f) * b_fn(t) * U_fn(t) * float(curve.rate(U_fn(t))) / 365.0
        for t in ts
    ])
    return float(np.trapz(vals, ts))


def income_difference(
    H: float,
    b_fn,
    U_fn,
    curve1,
    curve2,
    f: float = 0.05,
    n_intervals: int = 1000,
) -> float:
    """ΔA(H) = A_1(H) - A_2(H) using the same b(t) and U(t) path."""
    A1 = accumulated_income(H, b_fn, U_fn, curve1, f, n_intervals)
    A2 = accumulated_income(H, b_fn, U_fn, curve2, f, n_intervals)
    return A1 - A2


# ---------------------------------------------------------------------------
# Scenario 1 (fixed principal, piecewise-linear utilization)
# ---------------------------------------------------------------------------

def scenario1_paths(
    initial_balance: float = 10_000.0,
) -> tuple[callable, callable]:
    """Scenario 1: fixed 10,000 USDT deposit.

    Utilization path: day 0→80%, 90→95%, 180→99.5%, 270→85%, 365→80%.
    Balance: constant.
    """
    days  = [0, 90, 180, 270, 365]
    utils = [0.80, 0.95, 0.995, 0.85, 0.80]
    U_fn = build_utilization_path(days, utils)
    b_fn = build_balance_path([], [], start_balance=initial_balance)
    return b_fn, U_fn


# ---------------------------------------------------------------------------
# Scenario 2 (step-up principal, same utilization)
# ---------------------------------------------------------------------------

def scenario2_paths(
    initial_balance: float = 10_000.0,
    add_amount: float = 20_000.0,
    add_days: list[int] | None = None,
) -> tuple[callable, callable]:
    """Scenario 2: starts with initial_balance, adds add_amount on specified days.

    Default add days: 60, 120, 180, 240, 300.
    Utilization path: same as Scenario 1.
    """
    if add_days is None:
        add_days = [60, 120, 180, 240, 300]

    event_bals = []
    b = initial_balance
    for d in add_days:
        b += add_amount
        event_bals.append(b)

    U_fn_s1, _ = scenario1_paths.__wrapped__ if hasattr(scenario1_paths, '__wrapped__') else (None, None)
    days_u  = [0, 90, 180, 270, 365]
    utils_u = [0.80, 0.95, 0.995, 0.85, 0.80]
    U_fn = build_utilization_path(days_u, utils_u)
    b_fn = build_balance_path(add_days, event_bals, start_balance=initial_balance)
    return b_fn, U_fn
