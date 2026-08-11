# Methodology

Every number this project reports traces back to a formula and an assumption
documented here. If a claim in the README seems too good, this is the file
that either justifies it or admits the limitation.

## 1. Universe and survivorship bias {#survivorship}

The universe (`src/riskengine/data/universe.py`) is 124 NSE large/mid-cap
tickers spanning 12 sectors, built as a **current-membership snapshot**, not a
point-in-time reconstruction. Free point-in-time NSE constituent history does
not exist, so this is a real, unfixed limitation. Two names failed to download
(`LTIM.NS`, `TATAMOTORS.NS` — both retired/renamed on Yahoo Finance around
2025 corporate actions) and are excluded; see
`results/tables/data_quality_report.csv`.

**Why this matters:** any stock that delisted, went bankrupt, or was demoted
out of the index before today does not appear here, so the backtest never
gets a chance to lose money on it the way a real 2015 investor could have.
Absolute returns (CAGR, Sharpe) are therefore overstated versus what an
investor building this portfolio in 2015 would have actually earned.

**What mitigates it, partially:**
- The universe deliberately retains names that performed badly over the
  sample (IDEA, BHEL, SAIL, ZEEL, PAYTM, RBLBANK, IDFCFIRSTB) rather than
  curating only winners in hindsight.
- Every strategy is compared against an **equal-weight portfolio of the same
  universe**, not just the Nifty 50. Survivorship inflates both sides of that
  comparison roughly equally, so the *relative* ranking between strategies is
  far less contaminated than any single strategy's absolute return.
- The headline claim in the README is the relative comparison
  (strategy vs. equal-weight vs. Nifty), not the absolute CAGR.

## 2. Data pipeline

- Source: Yahoo Finance via `yfinance`, daily OHLCV, `auto_adjust=True`
  (split- and dividend-adjusted).
- Range: 2015-01-01 to 2025-06-30 (`config.START_DATE`, `config.END_DATE`).
- A ticker is dropped from the clean panel unless it has ≥500 trading days of
  history and <5% missing observations within its active window
  (`config.MIN_HISTORY_DAYS`, `config.MAX_MISSING_FRAC`). See
  `data.quality.data_quality_report`.
- Gaps up to 5 trading days are forward-filled (weekend/holiday noise). Longer
  gaps are left as `NaN` — a suspended stock should not silently show
  zero volatility.
- Single-day moves beyond 50% are flagged (`EXTREME_RETURN`) as likely
  unadjusted corporate actions and reported, though not automatically removed
  from the P&L stream (only from feature computation — see
  `quality.winsorise_returns`).

## 3. Return and rate conventions

- **Simple returns**, not log returns, are the default throughout
  (`data.quality.to_returns`), because portfolio return is a weight-weighted
  sum of simple asset returns — log returns do not aggregate linearly across
  assets, which is a common source of silent error in backtests that use them
  for portfolio-level P&L.
- **Risk-free rate**: a constant 6.5% annual (`config.RISK_FREE_ANNUAL`),
  approximating the 2015–2025 average 10-year G-Sec yield. This is a
  simplification — the real rate moved between roughly 5.8% and 7.5% over the
  sample — and every Sharpe/Sortino/Treynor number inherits that
  simplification.
- **Trading day convention**: 252 days/year throughout (`config.TRADING_DAYS`).

## 4. Risk metrics (`risk/metrics.py`)

Standard formulas (Sharpe, Sortino, Calmar, Treynor, Information Ratio,
Ulcer Index) computed on **excess** daily returns, annualised by
√252 for volatility-type quantities and ×252 for mean-type quantities. Beta
and alpha use a rolling 252-day OLS window (`features/market.py`) rather than
a single full-sample estimate, because beta is demonstrably non-stationary in
Indian equities (bank betas in 2020–21 do not resemble 2017).

## 5. Value-at-Risk

Four estimators are implemented (`risk/var.py`): parametric (Gaussian),
historical (empirical quantile), Cornish-Fisher (skew/kurtosis-adjusted
Gaussian), and Monte Carlo (historical bootstrap by default). **Sign
convention: VaR is always a positive number representing a loss.**

VaR is only useful if it is validated, so every VaR number reported is
backtested (`risk/var_backtest.py`) with:
- **Kupiec's unconditional coverage test** — did the realised breach rate
  match the target rate (e.g., 5% at 95% confidence)?
- **Christoffersen's independence test** — were breaches spread out, or
  clustered together (e.g., all in March 2020)? A model can pass Kupiec and
  fail Christoffersen — the right number of breaches, all in one bad week —
  which is a real failure mode plain VaR reporting hides.

Results in `results/tables/var_backtest.csv` are reported as-is, including
rejections. Historical VaR at 95% was rejected by Christoffersen in the
sample tested (breaches clustered around the 2020 and 2022 drawdowns) — this
is exactly the well-known weakness of a rolling-window historical VaR under
volatility clustering, not a bug.

## 6. Covariance estimation (`risk/covariance.py`)

With ~120 assets and a 36-month (≈756-day) training window, sample covariance
has thousands of free parameters estimated from a comparable number of
observations — the matrix is technically invertible but its smallest
eigenvalues are dominated by estimation noise. A mean-variance optimiser
exploits exactly those noisy directions, which is the textbook explanation for
why naive Markowitz portfolios are unstable out of sample.

**Ledoit-Wolf shrinkage** toward a scaled identity matrix is the default
estimator. The shrinkage intensity is chosen analytically (no hyperparameter
tuning). `results/tables/covariance_conditioning.json` shows the condition
number reduction empirically on this dataset.

## 7. Allocators (`optimize/allocators.py`)

Six allocators share one contract (long-only, fully invested, capped at
`profile.max_weight`) so the comparison between them is fair:

| Allocator | Idea | Known weakness |
|---|---|---|
| Equal-weight | 1/N | Ignores risk and correlation entirely — and is the baseline every other strategy has to beat (DeMiguel, Garlappi & Uppal 2009 found this is harder than it sounds) |
| Inverse-volatility | Weight ∝ 1/σ | Ignores correlation |
| Min-variance | Minimise wᵀΣw | Maximally sensitive to Σ estimation error without shrinkage |
| Risk-parity | Equalise each asset's contribution to portfolio variance | No return information at all |
| Max-Sharpe (tangency) | Classic Markowitz | Maximally sensitive to expected-return estimates, the noisiest inputs in finance |
| Score-based | This project's original scoring model, corrected (z-scores + softmax instead of min-max) | Depends on the risk-profile weight choices being reasonable |

## 8. The score-based model, and what changed from the original notebook

The original notebook min-max normalised each metric and summed with fixed
weights. Two problems: (1) min-max is destroyed by a single outlier stock —
one freak Sharpe compresses everyone else toward zero; (2) weights on a
min-max scale are not comparable to each other, so "0.3 on Sharpe, 0.3 on
volatility" does not mean equal influence.

The rebuilt version (`optimize/scoring.py`) uses cross-sectional z-scores
(winsorised at ±3σ) and converts the weighted z-score sum into portfolio
weights via a softmax rather than direct normalisation, so a below-average
stock gets a small weight instead of being clipped to exactly zero.

## 9. Walk-forward backtest (`backtest/engine.py`)

- Train on a rolling 36-month window, hold for 3 months, roll forward
  (`config.TRAIN_MONTHS`, `config.HOLD_MONTHS`).
- At rebalance date *t*, the allocator sees returns strictly through *t*.
  Positions are applied at *t*'s close and start earning returns from *t+1* —
  trading on the same day's close using that day's own data is the most
  common one-day look-ahead bug in retail backtests, and shifting by one day
  removes it structurally rather than by convention.
- Between rebalances, weights **drift** with prices rather than being reset
  daily — a daily free rebalance is not something any real investor gets.
- Transaction costs (`config.COST_BPS` = 15bps, covering STT + exchange +
  SEBI + stamp + slippage on a discount-broker delivery trade) are charged on
  the traded value at each rebalance (drifted weight → target weight
  difference), so a strategy that churns pays for churning.
- Every look-ahead claim above is enforced by
  [`tests/test_leakage.py`](../tests/test_leakage.py), which corrupts only
  the future portion of the data and asserts that every quantity decided
  before the corruption point is bit-for-bit unchanged.

## 10. Statistical significance (`backtest/stats.py`)

A backtest reporting "Sharpe 0.90 vs 0.39" invites the obvious question: is
that difference distinguishable from noise over ~7.5 years of data? Usually
it is not, so this project reports the answer rather than the point estimate
alone.

- **Stationary block bootstrap** (Politis-Romano) for the 95% CI on
  (strategy Sharpe − benchmark Sharpe), resampling both series with the same
  indices to preserve their contemporaneous correlation, and resampling in
  random-length blocks to preserve short-range autocorrelation that a naive
  i.i.d. bootstrap would understate.
- **Probabilistic Sharpe Ratio (PSR)** and **Deflated Sharpe Ratio (DSR)**
  (Bailey & López de Prado): PSR corrects the Sharpe standard error for
  skewness/kurtosis; DSR further corrects for having tried multiple
  strategies and reporting the best one, which is exactly what a strategy
  horse race does.
- **Newey-West HAC t-statistic** on active returns (5-lag) as a second,
  independent check.

Result in this project: of six strategies vs. equal-weight, only
**max-Sharpe** shows a bootstrap 95% CI on the Sharpe difference that
excludes zero. The rest are directionally positive but not statistically
distinguishable from equal-weight over this sample — see
`results/tables/significance_tests.csv`. That is reported as the finding,
not smoothed over.

## 11. Volatility forecasting (`models/`)

**Why volatility, not price or return direction:** volatility clusters and is
autocorrelated (the empirical fact behind every GARCH-family paper since
1986), which makes it genuinely forecastable. Daily/monthly price *levels* or
*returns* for liquid large-caps are close to a random walk at this horizon —
a model "predicting" them is almost always relearning yesterday's price, not
finding signal.

- **Target**: 21-trading-day-forward realised volatility, strictly aligned
  (row *t* uses returns from *t+1* to *t+21* only —
  see `test_leakage.py::test_forward_vol_target_is_aligned_to_the_future`).
- **Features**: multi-horizon realised vol, range-based estimators
  (Parkinson, Garman-Klass, Rogers-Satchell — up to ~7x more statistically
  efficient than close-to-close for the same number of days), EWMA vol,
  vol-of-vol, leverage-effect proxies, drawdown state, liquidity/turnover,
  India VIX level and change, rolling beta.
- **Sampling**: month-end only, not daily — daily sampling of a 21-day
  forward target creates 95%-overlapping windows between adjacent rows, which
  inflates the apparent sample size without adding information.
- **Model**: LightGBM, log-target, expanding-window walk-forward (train on
  everything before year *Y*, predict year *Y*, roll forward from 2018).
- **Baselines**: EWMA(0.94) (RiskMetrics standard — a genuinely strong
  baseline, not a strawman), naive random-walk (today's realised vol),
  simple hybrid average, and GARCH(1,1)-t fit per stock.
- **Evaluation**: RMSE, QLIKE (the loss function the volatility literature
  actually uses, because it is robust to noise in the realised-vol proxy and
  penalises under-forecasting more than over-forecasting), and the
  **Diebold-Mariano test** for whether an improvement is statistically
  significant rather than assumed.
- **Integration**: forecasts replace only the *diagonal* of the covariance
  matrix used by the optimiser (`risk.covariance.cov_from_vols`), keeping
  historical correlations. This isolates exactly what the vol forecast
  contributes — any change in strategy performance is attributable to the
  volatility forecast and nothing else. See `results/tables/vol_forecast_impact.csv`
  for whether it actually helped (spoiler: it did not uniformly help — see
  the README for the honest result).

## 12. What would make this materially better

Documented rather than hidden, since a real point-in-time universe, a
time-varying risk-free rate, level-2 order-book-based cost modelling, and a
factor-attribution layer would each meaningfully improve on the current
version — none of them were in scope for this iteration.
