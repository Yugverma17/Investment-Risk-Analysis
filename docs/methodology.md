# Methodology

This is where I explain every formula and assumption behind the numbers in the README. If something in there looks too good to be true, this file either justifies it or admits it's a limitation.

## 1. Universe and survivorship bias {#survivorship}

The universe (`src/riskengine/data/universe.py`) is 124 NSE large/mid-cap tickers across 12 sectors. I built it as a snapshot of what these sectors look like *today*, not a reconstruction of what the index actually contained back in 2015. I looked for free point-in-time constituent data and couldn't find any, so this is a real limitation I haven't solved. Two tickers (`LTIM.NS`, `TATAMOTORS.NS`) failed to download entirely — both got renamed/restructured on Yahoo Finance around 2025 corporate actions — and I excluded them rather than faking data for them. See `results/tables/data_quality_report.csv`.

Why this actually matters: any stock that got delisted, went bankrupt, or dropped out of the index sometime between 2015 and now simply isn't in this dataset. That means the backtest never had a chance to lose money on it the way a real investor building this portfolio in 2015 would have. So the absolute return numbers (CAGR, Sharpe) are inflated relative to what someone actually investing back then would have experienced.

What I did to soften this, even though it's not a full fix:
- I kept stocks that performed badly over the sample instead of only listing winners in hindsight — Vodafone Idea, BHEL, SAIL, Zee, Paytm, RBL Bank, IDFC First are all still in there.
- Every strategy gets compared against an equal-weight portfolio built from the *same* universe, not just Nifty. Since survivorship bias inflates both sides of that comparison roughly equally, the relative ranking between strategies is a lot more trustworthy than any single strategy's absolute number.
- Because of that, the headline claims in the README are relative comparisons (strategy vs. equal-weight vs. Nifty), not a standalone CAGR figure.

## 2. Data pipeline

- I pull daily OHLCV from Yahoo Finance through `yfinance`, with `auto_adjust=True` so splits and dividends are handled automatically.
- Date range is 2015-01-01 to 2025-06-30 (`config.START_DATE`, `config.END_DATE`).
- A stock gets dropped from the clean panel if it has fewer than 500 trading days of history, or more than 5% missing observations in its active window (`config.MIN_HISTORY_DAYS`, `config.MAX_MISSING_FRAC` — see `data.quality.data_quality_report`).
- Gaps of up to 5 trading days get forward-filled (mostly weekend/holiday noise in the calendar alignment). Anything longer stays as `NaN` on purpose — if I forward-filled a long suspension, it would look like the stock had zero volatility during that stretch, which isn't true.
- Any single-day move bigger than 50% gets flagged as a likely unadjusted corporate action (`EXTREME_RETURN`) and shows up in the quality report. I don't strip these from the actual P&L calculation though, only from feature computation (`quality.winsorise_returns`) — I didn't want to quietly delete a real crash day just because it looked extreme.

## 3. Return and rate conventions

- I use simple returns everywhere, not log returns (`data.quality.to_returns`). The reason: portfolio return is a weighted sum of the individual asset returns, and that only works cleanly with simple returns — log returns don't add up across assets the same way, which trips people up more often than you'd expect.
- Risk-free rate is a flat 6.5% annual (`config.RISK_FREE_ANNUAL`), which is roughly the average 10-year G-Sec yield over 2015-2025. It's a simplification — the actual rate moved between about 5.8% and 7.5% during this period — and every Sharpe/Sortino/Treynor number in this project inherits that simplification.
- 252 trading days a year, consistently (`config.TRADING_DAYS`).

## 4. Risk metrics (`risk/metrics.py`)

Standard stuff here — Sharpe, Sortino, Calmar, Treynor, Information Ratio, Ulcer Index — computed on excess daily returns and annualized (×√252 for volatility, ×252 for mean returns). Beta and alpha use a rolling 252-day window instead of one full-sample estimate, because beta genuinely isn't stable over time for Indian stocks. Bank betas in 2020-21 don't look anything like they did in 2017, and using a single full-sample number would hide that.

## 5. Value-at-Risk

I implemented four ways to compute VaR (`risk/var.py`): parametric/Gaussian, historical (just the empirical quantile), Cornish-Fisher (Gaussian adjusted for skew and kurtosis), and Monte Carlo (bootstrapped from history by default). One convention to keep straight: VaR is always reported as a positive number representing a loss.

Computing a VaR number is easy. Knowing whether it's actually any good is the hard part, so I backtested every one (`risk/var_backtest.py`) with two tests:
- **Kupiec's test** — did the number of breaches match what we'd expect (e.g. roughly 5% of days at 95% confidence)?
- **Christoffersen's test** — were those breaches spread out over time, or did they all happen in the same bad week? A model can pass Kupiec and still fail Christoffersen — right number of breaches overall, but they all landed during one crash — which is a real failure mode that just counting breaches would completely miss.

I left the results in `results/tables/var_backtest.csv` exactly as they came out, rejections included. Historical VaR at 95% actually fails Christoffersen in this sample — the breaches clustered around the 2020 and 2022 drawdowns instead of being spread out. That's a known weakness of rolling-window historical VaR when volatility clusters like this, not a bug in my code.

## 6. Covariance estimation (`risk/covariance.py`)

With about 120 stocks and a 36-month training window (~756 days), sample covariance ends up with thousands of parameters estimated from a similar number of data points. The matrix is technically invertible, but its smallest eigenvalues are basically noise. And a mean-variance optimizer will happily exploit exactly those noisy directions because they look like free risk reduction — this is the standard explanation for why plain Markowitz portfolios tend to fall apart out of sample.

My default fix is Ledoit-Wolf shrinkage, pulling the covariance matrix toward a scaled identity matrix. The nice thing about it is the shrinkage amount is chosen analytically — no hyperparameter to tune. `results/tables/covariance_conditioning.json` shows how much this actually helps on this specific dataset.

## 7. Allocators (`optimize/allocators.py`)

All six allocators follow the same rules — long-only, fully invested, capped at whatever `profile.max_weight` is — so comparing them is actually fair:

| Allocator | Idea | Where it falls short |
|---|---|---|
| Equal-weight | 1/N | Ignores risk and correlation completely — and it's still the bar every other strategy has to clear (DeMiguel, Garlappi & Uppal showed back in 2009 that this is harder than people think) |
| Inverse-volatility | Weight ∝ 1/σ | Ignores correlation |
| Min-variance | Minimize wᵀΣw | Very sensitive to estimation error in Σ without shrinkage |
| Risk-parity | Every asset contributes equally to portfolio variance | Doesn't use any return information at all |
| Max-Sharpe (tangency) | Classic Markowitz | Extremely sensitive to expected-return estimates, which are the noisiest inputs you can feed a finance model |
| Score-based | My original scoring idea from the first version of this project, fixed up (z-scores + softmax instead of min-max) | Only as good as the risk-profile weight choices |

## 8. The score-based model, and what I changed from the original notebook

My original notebook used min-max normalization on each metric and summed them with fixed weights. Two problems with that: one outlier stock with a freak Sharpe ratio compresses everything else toward zero, and the weights aren't really comparable to each other on a min-max scale — "0.3 on Sharpe, 0.3 on volatility" doesn't actually mean those two get equal say.

The rebuilt version (`optimize/scoring.py`) uses cross-sectional z-scores instead (winsorized at ±3σ so one outlier can't dominate), and turns the weighted z-score sum into portfolio weights through a softmax rather than direct normalization. That way a below-average stock still gets a small weight instead of getting clipped straight to zero.

## 9. Walk-forward backtest (`backtest/engine.py`)

- Trains on a rolling 36-month window, holds for 3 months, then rolls forward (`config.TRAIN_MONTHS`, `config.HOLD_MONTHS`).
- At each rebalance date *t*, the allocator only sees returns up through *t*. Positions get applied at *t*'s close and start earning returns from *t+1* onward. Trading on the same day's close using that same day's data is probably the single most common look-ahead bug in DIY backtests, and shifting everything by a day removes it structurally instead of just being a rule I promise I'm following.
- Between rebalances, weights drift with prices instead of getting reset daily — nobody gets a free daily rebalance in real life.
- Transaction costs (`config.COST_BPS` = 15bps — covers STT, exchange charges, SEBI fees, stamp duty, and slippage on a discount-broker delivery trade) get charged on however much actually gets traded at each rebalance, so a strategy that churns its holdings pays for that churn.
- I didn't just claim the look-ahead protection works — [`tests/test_leakage.py`](../tests/test_leakage.py) actually checks it, by corrupting only the future part of the price data and confirming every decision made before that point comes out bit-for-bit identical either way.

## 10. Statistical significance (`backtest/stats.py`)

Reporting "Sharpe 0.90 vs 0.39" on its own kind of dodges the real question: over ~7.5 years of data, is that difference actually distinguishable from noise? Usually the answer is no, so I built this to check rather than just quoting the point estimate and moving on.

- **Stationary block bootstrap** (Politis-Romano) for a 95% confidence interval on (strategy Sharpe − benchmark Sharpe). I resample both series using the same indices each time so their correlation with each other is preserved, and I resample in random-length blocks rather than single points so short-range autocorrelation doesn't get understated the way a naive bootstrap would understate it.
- **Probabilistic Sharpe Ratio (PSR)** and **Deflated Sharpe Ratio (DSR)**, from Bailey & López de Prado. PSR adjusts the Sharpe standard error for skew and kurtosis. DSR goes a step further and corrects for the fact that I tried six strategies and I'm reporting the best one — which is exactly the kind of thing that inflates results if you don't account for it.
- A **Newey-West HAC t-statistic** on active returns (5 lags) as a second, independent sanity check.

What actually came out of this: of the six strategies tested against equal-weight, only max-Sharpe has a bootstrap 95% CI on the Sharpe difference that doesn't touch zero. The rest look better on paper but I can't say they're statistically distinguishable from equal-weight over this sample. See `results/tables/significance_tests.csv`. I'm treating that as the actual finding rather than picking whichever strategy looked best and running with it.

## 11. Volatility forecasting (`models/`)

Why I went with volatility instead of trying to predict price or direction: volatility clusters and is autocorrelated (this is basically the whole premise behind every GARCH paper since 1986), so it's genuinely something you can forecast. Daily or monthly price levels — or returns — for liquid large-caps are close enough to a random walk at this horizon that a model "predicting" them is almost always just relearning yesterday's price rather than finding real signal.

- **Target**: realised volatility over the next 21 trading days, carefully aligned so row *t* only uses returns from *t+1* through *t+21* (checked directly in `test_leakage.py::test_forward_vol_target_is_aligned_to_the_future`).
- **Features**: realised vol at several horizons, range-based estimators (Parkinson, Garman-Klass, Rogers-Satchell — these are roughly 5-7x more statistically efficient than plain close-to-close vol for the same number of days), EWMA vol, vol-of-vol, some leverage-effect proxies, drawdown state, liquidity/turnover, India VIX level and its change, rolling beta.
- **Sampling**: I only sample at month-end, not daily. Sampling daily against a 21-day-forward target creates windows that overlap 95% with their neighbors, which inflates how big the sample looks without actually adding new information.
- **Model**: LightGBM, trained on the log of the target, using an expanding walk-forward window (train on everything before year Y, predict year Y, roll forward starting 2018).
- **Baselines**: EWMA(0.94) — the RiskMetrics standard, and a genuinely strong baseline, not something I set up to lose — plus a naive random-walk (just today's realised vol), a simple hybrid average, and a GARCH(1,1)-t fit separately for each stock.
- **Evaluation**: RMSE, and QLIKE (which is what the volatility-forecasting literature actually uses, because it's more robust to noise in the realised-vol proxy and penalizes under-forecasting harder than over-forecasting). I also ran Diebold-Mariano tests so I could say whether an improvement was statistically real instead of just assuming it.
- **How it plugs into the portfolio**: the forecast only replaces the diagonal of the covariance matrix (`risk.covariance.cov_from_vols`), keeping the historical correlation structure untouched. I did it this way on purpose so I could isolate exactly what the volatility forecast was contributing — any change in strategy performance is attributable to the vol forecast and nothing else. Check `results/tables/vol_forecast_impact.csv` to see whether it actually helped. Short version: not consistently — see the README for the full result.

## 12. What would make this meaningfully better

I'm listing these instead of pretending they don't matter: real point-in-time index membership, a time-varying risk-free rate, more realistic order-book-based cost modeling, and a proper factor-attribution layer would each be a real improvement. None of them made it into this version — mostly a time/scope call, not something I forgot about.
