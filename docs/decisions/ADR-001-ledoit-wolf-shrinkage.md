# ADR-001: Ledoit-Wolf shrinkage as the default covariance estimator

## Status
Accepted

## Context

Portfolio optimisers (min-variance, risk-parity, max-Sharpe) need a covariance
matrix Σ over the selected universe. With ~20–30 selected stocks and a
36-month (≈756-day) training window, sample covariance is nominally estimable
(T > N), but its condition number is still large — the ratio between the
largest and smallest eigenvalue routinely exceeds 300–500x on this dataset
(see `results/tables/covariance_conditioning.json`). The smallest eigenvalues
are dominated by estimation noise rather than genuine low-risk directions, and
a mean-variance optimiser actively seeks out exactly those directions because
they appear to offer free risk reduction. This is the standard explanation for
why naive Markowitz portfolios are unstable and turnover-heavy out of sample.

## Decision

Use **Ledoit-Wolf shrinkage** (`sklearn.covariance.LedoitWolf`) toward a
scaled identity target as the default covariance estimator
(`risk/covariance.ledoit_wolf_cov`), with sample covariance, OAS shrinkage,
constant-correlation, and EWMA covariance implemented as alternatives for
comparison (`risk/covariance.ESTIMATORS`).

Ledoit-Wolf was chosen over the alternatives because:
- The shrinkage intensity is derived analytically to minimise expected
  squared Frobenius-norm error — there is no hyperparameter to tune or
  cross-validate, which matters given the training windows here are already
  short.
- It is a well-established, citable method (Ledoit & Wolf, 2004) rather than
  a heuristic, which matters for defending the choice in an interview.

## Alternatives considered

- **Sample covariance** — kept as an explicit ablation
  (`min_variance (sample cov)` in `results/tables/strategy_comparison.csv`)
  to demonstrate the problem shrinkage solves, not just assert it.
- **OAS shrinkage** — usually shrinks harder than Ledoit-Wolf; available but
  not the default, since it can over-shrink with the training window sizes
  used here.
- **Constant-correlation (Elton-Gruber)** — a much cruder shrinkage target
  (replace every pairwise correlation with the average) that is
  surprisingly hard to beat empirically; kept as a baseline.
- **Factor models (e.g., single-index / Fama-French)** — would likely
  outperform pure shrinkage but require a defensible factor set for Indian
  equities, which was out of scope for this iteration.

## Consequences

- Optimised portfolios (min-variance, risk-parity) are less concentrated and
  less turnover-heavy than they would be under sample covariance — visible in
  `results/tables/strategy_comparison.csv` (`AvgTurnover` column).
- Shrinkage biases the estimate toward the target (scaled identity), which
  is a real cost when the true covariance structure is far from that target
  — e.g., during periods of unusually high average pairwise correlation
  (crisis periods), shrinkage may understate true systemic co-movement.
- The volatility-forecast integration (`risk/covariance.cov_from_vols`)
  replaces only the diagonal of the Ledoit-Wolf matrix, inheriting this
  trade-off for the correlation structure it keeps.
