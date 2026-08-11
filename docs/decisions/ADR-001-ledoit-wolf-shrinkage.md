# ADR-001: Ledoit-Wolf shrinkage as the default covariance estimator

## Status
Accepted

## Context

The portfolio optimizers (min-variance, risk-parity, max-Sharpe) all need a covariance matrix Σ for whatever stocks got selected that round. With around 20-30 stocks and a 36-month (~756 day) training window, sample covariance is technically estimable — there's more data than parameters, barely — but its condition number is still pretty bad. On this dataset the ratio between the largest and smallest eigenvalue routinely goes above 300-500x (see `results/tables/covariance_conditioning.json`). Those smallest eigenvalues are basically estimation noise, not real low-risk directions, and a mean-variance optimizer will happily chase exactly those directions because they look like free risk reduction. This is the usual explanation for why plain Markowitz portfolios end up unstable and trade a lot more than you'd want out of sample.

## Decision

I went with Ledoit-Wolf shrinkage (`sklearn.covariance.LedoitWolf`), shrinking toward a scaled identity matrix, as the default (`risk/covariance.ledoit_wolf_cov`). Sample covariance, OAS shrinkage, constant-correlation, and EWMA covariance are all implemented too so I could compare them (`risk/covariance.ESTIMATORS`).

Why Ledoit-Wolf specifically:
- The shrinkage amount is derived analytically to minimize expected squared error — there's no hyperparameter I'd need to tune or cross-validate, which matters a lot given how short these training windows already are.
- It's a well-established method with an actual citation behind it (Ledoit & Wolf, 2004) rather than something I made up, which matters if I ever need to defend the choice out loud.

## Alternatives I considered

- **Sample covariance** — I kept this in as an explicit comparison (`min_variance (sample cov)` in `results/tables/strategy_comparison.csv`) so I could actually show the problem shrinkage fixes instead of just asserting it exists.
- **OAS shrinkage** — tends to shrink harder than Ledoit-Wolf. It's available, just not the default, since it seemed to over-shrink given how short my training windows are.
- **Constant-correlation (Elton-Gruber)** — a much blunter approach, where you just replace every pairwise correlation with the average one. Surprisingly hard to beat in practice, so I kept it around as another baseline.
- **Factor models** (single-index, Fama-French style) — would probably do better than pure shrinkage, but building a defensible factor set for Indian equities felt like its own project. Left it out of scope for now.

## Consequences

- Min-variance and risk-parity portfolios end up less concentrated and trade less than they would under sample covariance — you can see this directly in the `AvgTurnover` column of `results/tables/strategy_comparison.csv`.
- Shrinkage pulls the estimate toward the target (scaled identity), and that's a real cost when the true covariance structure looks nothing like that target. During crisis periods when average correlation spikes, shrinkage can end up understating how much everything is actually moving together.
- The volatility-forecast integration (`risk/covariance.cov_from_vols`) only swaps out the diagonal of the Ledoit-Wolf matrix, so it inherits this same trade-off on the correlation side.
