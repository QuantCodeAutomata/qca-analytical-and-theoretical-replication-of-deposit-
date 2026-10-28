# Repository: QCA Deposit Monotonicity Replication

## Purpose
This repo replicates analytical, theoretical, and empirical results from a DeFi lending-market
paper on deposit-monotonicity rate-curve design.  No external market data is used for exp_1 and
exp_3; exp_2 requires Drift onchain data (synthetic approximation included).

## Key Formulas
- Utilization: U = B / T
- Supply APR: R_s(U) = U * R_b(U)
- Supplier income: I = b * U * R_b(U), where b = supplier balance, α = b/T
- Elasticity: η(U) = (U / R_b(U)) * dR_b/dU = d ln R_b / d ln U
- Income derivative: dI/dδ = U * R_b(U) * [1 - α(1 + η(U))]
- Protected share: α_safe(U) = 1 / (1 + η(U))
- Endpoint bound (fixed-share): K* = ln(R_max/R_o) / ln(1/O)
- Maximum protectable share: α̂_max = 1 / (1 + K*)

## Canonical Parameters (Section 2)
- O = 0.80, R_o = 0.05, R_max = 1.00
- m = (1.00 - 0.05) / (1 - 0.80) = 4.75
- K* ≈ 13.43, α̂_max ≈ 6.93%
- U_c ≈ 90.92% for K=19 power repair

## Module Map
- src/curves.py           → PiecewiseLinearCurve, SingleKinkCurve
- src/monotonicity.py     → income formula, derivative identity, Section 2 sweep
- src/admissibility.py    → fixed-share and fresh-depositor admissibility, combined
- src/smoothing.py        → Appendix E η_max(w), Table 5 replication
- src/single_kink_repair.py → power repair construction
- src/multikink_caps.py   → segment-cap admissibility checks
- src/fixed_kink_repairs.py → floor/raise/lower/mixed repairs
- src/holding_period.py   → accumulated supplier-interest integration
- src/exposure_metrics.py → α_safe computation and exposure flagging
- src/drift_market.py     → Drift market state representation

## Testing
All key numerical targets are unit-tested in tests/:
- test_formulas.py        → derivative identity, Section 2 values, elasticity
- test_admissibility.py   → K*, α̂_max, boundary curves, piecewise checks
- test_repairs.py         → U_c, fixed-kink requirements, holding-period totals

## Install
pip install numpy scipy matplotlib pandas sympy cvxpy
