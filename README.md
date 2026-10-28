# QCA: Analytical and Theoretical Replication of Deposit Monotonicity

This repository replicates the analytical, theoretical, and empirical results from a paper on
DeFi lending-market rate-curve design and deposit-monotonicity protection.

## Repository Structure

```
├── data/                  # Data loading scripts
├── src/
│   ├── curves.py          # Rate curve classes and core formulas
│   ├── monotonicity.py    # Deposit monotonicity analysis
│   ├── admissibility.py   # Admissible rate-curve families
│   ├── smoothing.py       # Appendix E C1-smoothing formulas
│   ├── single_kink_repair.py  # Exp 3: single-kink power repair
│   ├── multikink_caps.py      # Exp 3: multiple-kink segment-cap admissibility
│   ├── fixed_kink_repairs.py  # Exp 3: floor/raise/lower/mixed repairs
│   ├── holding_period.py      # Exp 3: holding-period income comparisons
│   ├── exposure_metrics.py    # Exp 2: protected-share & exposure logic
│   └── drift_market.py        # Exp 2: Drift market state representation
├── exp/
│   ├── exp1_theory.py         # Exp 1: full theory replication
│   ├── exp2_drift.py          # Exp 2: Drift empirical replication (synthetic)
│   └── exp3_repair.py         # Exp 3: repair & income comparison
├── tests/
│   ├── test_formulas.py       # Unit tests for core formulas
│   ├── test_admissibility.py  # Unit tests for admissibility theory
│   └── test_repairs.py        # Unit tests for repair constructions
├── results/
│   └── RESULTS.md             # Summary of experiment findings
└── AGENTS.md                  # Repository context for AI agents
```

## Experiments

### Exp 1 — Theory Replication
Verifies deposit-monotonicity derivative identity, elasticity-based monotonicity condition,
admissible rate-curve families, and Appendix E smoothing formulas using only
paper-specified synthetic parameters.

### Exp 2 — Drift Empirical Replication
Reconstructs hourly Drift SOL/USDC market states, computes protected share α_safe(U),
and measures supplier exposure (requires real Drift onchain data; synthetic
approximation provided).

### Exp 3 — Repair & Income Comparison
Constructs compliant rate-curve repairs (single-kink power repair, multiple-kink
fixed-kink repairs) and compares holding-period supplier income under identical scenarios.

## Setup

```bash
pip install numpy scipy matplotlib pandas sympy cvxpy
```

## Running Experiments

```bash
python exp/exp1_theory.py
python exp/exp2_drift.py
python exp/exp3_repair.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Parameters (Section 2 canonical example)
- O = 0.80 (optimal utilization / kink)
- R_o = 0.05 (borrow APR at kink)
- R_max = 1.00 (maximum borrow APR)
- m = (R_max - R_o)/(1 - O) = 4.75 (above-kink slope)
- K* = ln(R_max/R_o) / ln(1/O) ≈ 13.43
- α̂_max = 1/(1 + K*) ≈ 6.93%
