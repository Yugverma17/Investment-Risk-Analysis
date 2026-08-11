# RiskLens — Portfolio Risk & Allocation Engine for Indian Equities

I started this as a single Jupyter notebook that picked stocks based on Sharpe ratio and beta, using monthly data and one train/test split. It "worked" in the sense that the numbers looked good, but I couldn't actually defend it if someone asked how I knew it wasn't just luck. So I rebuilt it properly: daily data going back to 2015, six different portfolio strategies tested against each other the way a real investor would experience them (no peeking at the future), realistic trading costs, and actual statistics to check whether any of it means anything.

It covers ~220 large and mid-cap NSE stocks, 2015 through mid-2025.

**[Live dashboard →](https://investment-risk-analysis.streamlit.app/)**
&nbsp;·&nbsp; [Methodology](docs/methodology.md) &nbsp;·&nbsp; [PRD](docs/PRD.md) &nbsp;·&nbsp; [Interview prep](docs/interview_prep.md)

---

## The main thing I found

I tested 6 allocation strategies walk-forward against the Nifty 50 benchmark, on a ~220-stock universe. Only **one of them** actually beat Nifty in a way that holds up statistically (95% confidence interval on the Sharpe difference doesn't touch zero) — and it's the simplest one: **plain equal-weight**. The fancier optimizers (max-Sharpe, min-variance, risk-parity) all looked good on paper but I can't say with confidence they weren't just lucky over this particular 7-year stretch. That's not a disappointing result — it's consistent with a well-known finding in the literature (DeMiguel, Garlappi & Uppal, 2009): the more assets an optimizer has to estimate expected returns and correlations for, the more room there is for that estimation error to get exploited, and a ~220-stock universe gives it plenty of room.

I also built a volatility-forecasting model with LightGBM that clearly beats EWMA and GARCH on prediction accuracy (statistically significant, not just a slightly better number). But when I actually plugged that better forecast into the portfolio optimizer, it didn't make the portfolio perform better. Two genuinely different things — being good at predicting volatility, and that prediction actually helping you make more money — and I think that's worth showing rather than hiding.

<p align="center">
  <img src="results/figures/equity_curves.png" width="90%" alt="Equity curves: six strategies vs Nifty 50">
</p>

## Results

Everything below comes straight out of `python scripts/run_backtests.py` — I didn't touch any of these numbers by hand.

### Strategy comparison (walk-forward, net of 15bps transaction costs)

|                           | CAGR   | Ann.Vol   |   Sharpe |   Sortino | MaxDD   |   Calmar |   Beta | Alpha   | InfoRatio   | HitRate   | AvgTurnover   | CostDrag   | Ann.Return   | Ulcer   |   Skew |   Kurtosis |   Treynor |
|:--------------------------|:-------|:----------|---------:|----------:|:--------|---------:|-------:|:--------|:------------|:----------|:--------------|:-----------|:-------------|:--------|-------:|-----------:|----------:|
| equal_weight              | 26.4%  | 20.0%     |     0.96 |      1.3  | -34.0%  |     0.78 |   0.9  | 13.0%   | 0.98        | 60.7%     | 26.0%         | 0.4%       | 25.5%        | 9.1%    |  -1.16 |      11.02 |      0.21 |
| inverse_vol               | 24.6%  | 19.3%     |     0.91 |      1.23 | -32.9%  |     0.75 |   0.88 | 11.5%   | 0.90        | 60.7%     | 26.7%         | 0.4%       | 23.9%        | 8.8%    |  -1.2  |      12.37 |      0.2  |
| min_variance              | 18.5%  | 17.3%     |     0.7  |      0.96 | -30.1%  |     0.61 |   0.77 | 6.9%    | 0.45        | 51.7%     | 34.1%         | 0.5%       | 18.5%        | 9.3%    |  -0.93 |      11.38 |      0.16 |
| risk_parity               | 24.2%  | 18.9%     |     0.91 |      1.23 | -32.6%  |     0.74 |   0.86 | 11.3%   | 0.88        | 58.4%     | 27.1%         | 0.4%       | 23.5%        | 8.8%    |  -1.16 |      11.46 |      0.2  |
| max_sharpe                | 20.9%  | 20.5%     |     0.72 |      0.99 | -33.9%  |     0.62 |   0.85 | 9.0%    | 0.55        | 48.3%     | 30.5%         | 0.5%       | 21.1%        | 10.8%   |  -0.86 |       7.83 |      0.17 |
| score_based               | 22.7%  | 22.4%     |     0.74 |      1.01 | -34.8%  |     0.65 |   0.92 | 10.3%   | 0.63        | 53.9%     | 29.9%         | 0.5%       | 23.0%        | 11.6%   |  -1    |      10.21 |      0.18 |
| min_variance (sample cov) | 18.3%  | 17.4%     |     0.69 |      0.95 | -30.1%  |     0.61 |   0.76 | 6.7%    | 0.43        | 50.6%     | 34.1%         | 0.5%       | 18.3%        | 9.4%    |  -0.9  |      11.24 |      0.15 |
| Nifty 50 (buy & hold)     | 12.4%  | 17.6%     |     0.39 |      0.53 | -38.4%  |     0.32 |   1    | 0.0%    | —           | 0.0%      | —             | —          | 13.2%        | 7.7%    |  -1.13 |      18.48 |      0.07 |

### Is any of this actually significant, or did I just get lucky?

I ran a stationary block bootstrap (2000 iterations) on the Sharpe difference between each strategy and Nifty 50. If the 95% confidence interval doesn't cross zero, I'm calling it real. More on this in [Methodology §10](docs/methodology.md#10-statistical-significance-backteststatspy).

| strategy                  |   sharpe_diff |   ci_95_low |   ci_95_high |   bootstrap_p |   active_t_stat |   active_p |   PSR |   DSR | significant_5pct   |
|:--------------------------|--------------:|------------:|-------------:|--------------:|----------------:|-----------:|------:|------:|:-------------------|
| equal_weight              |         0.565 |       0.022 |        1.063 |         0.03  |            2.52 |      0.012 | 0.993 | 0.874 | True               |
| inverse_vol               |         0.515 |      -0.001 |        1.012 |         0.046 |            2.34 |      0.019 | 0.991 | 0.845 | False              |
| min_variance              |         0.309 |      -0.198 |        0.812 |         0.242 |            1.18 |      0.236 | 0.968 | 0.689 | False              |
| risk_parity               |         0.516 |      -0.003 |        1.017 |         0.046 |            2.27 |      0.023 | 0.991 | 0.846 | False              |
| max_sharpe                |         0.33  |      -0.264 |        0.876 |         0.272 |            1.41 |      0.16  | 0.971 | 0.707 | False              |
| score_based               |         0.35  |      -0.263 |        0.919 |         0.248 |            1.61 |      0.107 | 0.974 | 0.725 | False              |
| min_variance (sample cov) |         0.298 |      -0.209 |        0.799 |         0.262 |            1.14 |      0.256 | 0.966 | 0.678 | False              |

### How did these hold up during actual crashes?

|                                    | equal_weight   | min_variance   | score_based   | Nifty 50   |
|:-----------------------------------|:---------------|:---------------|:--------------|:-----------|
| COVID crash (Feb-Mar 2020)         | -33.0%         | -29.6%         | -33.8%        | -36.5%     |
| COVID recovery (Apr-Dec 2020)      | 95.6%          | 66.5%          | 82.4%         | 83.7%      |
| 2022 rate shock (Jan-Jun 2022)     | -17.4%         | -25.4%         | -11.7%        | -11.9%     |
| Oct 2021 - Mar 2023 (flat market)  | -7.9%          | -11.1%         | -10.9%        | -6.0%      |
| 2024-25 correction (Sep 24-Feb 25) | -24.9%         | -24.8%         | -26.0%        | -15.6%     |

<p align="center">
  <img src="results/figures/drawdowns.png" width="90%" alt="Drawdown comparison">
</p>

Worth pointing out: during the 2022 rate-hike selloff and the late-2024 correction, my strategies actually lost *more* than plain Nifty did. Makes sense once you think about it — I'm holding 20-25 stocks, Nifty holds 50, so there's less diversification cushioning a broad, sector-agnostic drop.

### VaR — and whether it actually held up

Anyone can compute a Value-at-Risk number. The question is whether it's calibrated correctly, so I backtested it with Kupiec's test (did the number of breaches match what was expected?) and Christoffersen's test (were the breaches spread out, or all clumped together in one bad stretch?).

|                |   n_obs |   n_breaches |   expected |   breach_rate |   kupiec_p |   christoffersen_p |   cc_p | verdict   |
|:---------------|--------:|-------------:|-----------:|--------------:|-----------:|-------------------:|-------:|:----------|
| historical@95% |    1572 |           92 |       78.6 |        0.0585 |     0.1306 |             0.022  | 0.0232 | REJECT    |
| historical@99% |    1572 |           22 |       15.7 |        0.014  |     0.1333 |             0.4292 | 0.237  | PASS      |
| parametric@95% |    1572 |           86 |       78.6 |        0.0547 |     0.3986 |             0.0007 | 0.0021 | REJECT    |
| parametric@99% |    1572 |           37 |       15.7 |        0.0235 |     0      |             0.2861 | 0      | REJECT    |

<p align="center">
  <img src="results/figures/var_breaches.png" width="90%" alt="VaR breach chart">
</p>

Historical VaR at 95% fails the Christoffersen test — the breaches weren't random, they clustered around the 2020 and 2022 crashes, which is a known weakness of this method during volatile periods. I'm leaving that REJECT in the table instead of quietly picking a different method that passes.

### Volatility forecasting: LightGBM vs. EWMA vs. GARCH(1,1)

|                       |   RMSE |    MAE |   QLIKE |      R2 |     n |   RMSE_vs_EWMA_% |   QLIKE_vs_EWMA_% |
|:----------------------|-------:|-------:|--------:|--------:|------:|-----------------:|------------------:|
| Random walk (RV21)    | 0.1671 | 0.1119 |  0.4534 | -0.0729 | 18339 |           8.0118 |           30.322  |
| EWMA(0.94)            | 0.1547 | 0.103  |  0.348  |  0.0804 | 18339 |           0      |            0      |
| Hybrid (RV21+RV252)/2 | 0.1435 | 0.0957 |  0.2938 |  0.209  | 18339 |          -7.2568 |          -15.5465 |
| LightGBM              | 0.1358 | 0.0869 |  0.2929 |  0.2915 | 18339 |         -12.23   |          -15.8165 |
| GARCH(1,1)-t          | 0.1695 | 0.1146 |  0.3147 | -0.104  | 18339 |           9.565  |           -9.5504 |

<p align="center">
  <img src="results/figures/vol_pred_vs_actual.png" width="90%" alt="Forecast vs realised volatility">
</p>

And the significance tests (Diebold-Mariano) behind those numbers:


| comparison                     | loss   |   DM_stat |   p_value |     n | verdict              |
|:-------------------------------|:-------|----------:|----------:|------:|:---------------------|
| LightGBM vs Random walk (RV21) | QLIKE  |   -16.276 |    0      | 18660 | LightGBM better (5%) |
| LightGBM vs Random walk (RV21) | MSE    |   -10.825 |    0      | 18660 | LightGBM better (5%) |
| LightGBM vs EWMA(0.94)         | QLIKE  |    -7.328 |    0      | 18660 | LightGBM better (5%) |
| LightGBM vs EWMA(0.94)         | MSE    |    -9.549 |    0      | 18660 | LightGBM better (5%) |
| LightGBM vs GARCH(1,1)-t       | QLIKE  |    -2.98  |    0.0029 | 18339 | LightGBM better (5%) |
| LightGBM vs GARCH(1,1)-t       | MSE    |   -12.483 |    0      | 18339 | LightGBM better (5%) |


LightGBM clearly beats every baseline here — random-walk, EWMA, and GARCH — on both RMSE and QLIKE, all with Diebold-Mariano p-values under 0.01. Worth saying: this flipped when I expanded the universe from ~120 to ~220 stocks — at the smaller scale, LightGBM's edge over GARCH specifically on QLIKE wasn't statistically significant (p ≈ 0.16). More data made the result cleaner, not messier, which is a reassuring sign it's a real effect and not noise.

### Does the better forecast actually improve the portfolio, though?

|                               | CAGR   | Ann.Vol   |   Sharpe |   Sortino | MaxDD   |   Calmar |   Beta | Alpha   |   InfoRatio | HitRate   | Ann.Return   | Ulcer   |   Skew |   Kurtosis |   Treynor |
|:------------------------------|:-------|:----------|---------:|----------:|:--------|---------:|-------:|:--------|------------:|:----------|:-------------|:--------|-------:|-----------:|----------:|
| min_variance + vol forecast   | 16.2%  | 17.2%     |     0.59 |      0.81 | -30.0%  |     0.54 |   0.77 | 4.9%    |        0.29 | 49.4%     | 16.5%        | 9.6%    |  -0.84 |      11.45 |      0.13 |
| risk_parity + vol forecast    | 24.1%  | 18.9%     |     0.91 |      1.23 | -33.0%  |     0.73 |   0.86 | 11.1%   |        0.88 | 60.7%     | 23.4%        | 8.9%    |  -1.15 |      11.54 |      0.2  |
| min_variance (historical vol) | 18.5%  | 17.3%     |     0.7  |      0.96 | -30.1%  |     0.61 |   0.77 | 6.9%    |        0.45 | 51.7%     | 18.5%        | 9.3%    |  -0.93 |      11.38 |      0.16 |
| risk_parity (historical vol)  | 24.2%  | 18.9%     |     0.91 |      1.23 | -32.6%  |     0.74 |   0.86 | 11.3%   |        0.88 | 58.4%     | 23.5%        | 8.8%    |  -1.16 |      11.46 |      0.2  |

Short answer: not really, or at least not consistently. See [Methodology §11](docs/methodology.md#11-volatility-forecasting-models) for my best guess at why — a better volatility number doesn't necessarily make a better input to an optimizer that's sensitive to a different kind of estimation error.

---

## What I'm *not* claiming

- **This isn't investment advice.** Treat it as a technical/analytical project, not a recommendation for anyone to put real money into.
- **Survivorship bias is real here and I haven't fixed it.** The stock list is basically "large/mid-caps as they exist today," not what the index actually looked like in 2015. Free point-in-time constituent data doesn't really exist, so I mitigated it where I could (kept known underperformers like Vodafone Idea, BHEL, SAIL in the list) but the absolute return numbers are still inflated. That's why I lean on the relative comparisons — strategy vs. equal-weight vs. Nifty — rather than any single CAGR number. Details in [Methodology §1](docs/methodology.md#1-universe-and-survivorship-bias).
- **The risk-free rate is a flat 6.5%** the whole way through, not the actual G-Sec yield which moved around a fair bit over this period.
- **15bps transaction cost** is a reasonable estimate for a discount broker, not a live number.

## How it's organized

```
src/riskengine/
├── data/        which stocks to use, downloading + cleaning prices, data quality checks
├── features/    volatility estimators, rolling beta, momentum
├── risk/        Sharpe/Sortino/Calmar/Treynor, 4 VaR methods + backtesting, covariance shrinkage
├── optimize/    the 6 portfolio strategies, position limits, stock scoring
├── backtest/    the walk-forward simulator + statistical significance testing
├── models/      the LightGBM volatility model and its features, GARCH baseline
└── report/      turns results into charts and tables

scripts/         run these to reproduce everything: fetch_data.py, run_vol_model.py, run_backtests.py
notebooks/       00 is my original notebook, 01-03 walk through the rebuilt analysis
app/             the Streamlit dashboard
tests/           110 tests — the important ones check the code isn't cheating by peeking at future data
docs/            why I made each decision, every formula, the PRD, interview prep notes
```

## How to run it

```bash
# 1. set up the environment (Python 3.12)
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev,app]"

# 2. download the stock data (~1 min, needs internet)
python scripts/fetch_data.py

# 3. train the volatility model (~11 min — includes fitting GARCH per stock)
python scripts/run_vol_model.py

# 4. run the full backtest + stats (~15 min)
python scripts/run_backtests.py

# 5. run the tests (fast, works offline)
pytest

# 6. open the dashboard
streamlit run app/streamlit_app.py
```

## Built with

Python 3.12, pandas/numpy/scipy, scikit-learn (for covariance shrinkage), LightGBM, the `arch` package for GARCH, statsmodels, scipy.optimize, Streamlit + Plotly for the app, pytest, ruff, and GitHub Actions for CI.

## Why I built it this way

This started as a resume project for placement season, aimed at data analyst / data scientist roles. I could've optimized for "biggest number in the README" but decided to optimize for "can I actually answer follow-up questions about this in an interview." [`docs/interview_prep.md`](docs/interview_prep.md) has the questions I expect this to raise and how I'd answer each one from the actual numbers above.

## License

MIT — see [LICENSE](LICENSE).
