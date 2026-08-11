# RiskLens — Portfolio Risk & Allocation Engine for Indian Equities

I started this as a single Jupyter notebook that picked stocks based on Sharpe ratio and beta, using monthly data and one train/test split. It "worked" in the sense that the numbers looked good, but I couldn't actually defend it if someone asked how I knew it wasn't just luck. So I rebuilt it properly: daily data going back to 2015, six different portfolio strategies tested against each other the way a real investor would experience them (no peeking at the future), realistic trading costs, and actual statistics to check whether any of it means anything.

It covers ~120 large and mid-cap NSE stocks, 2015 through mid-2025.

**[Live dashboard →](https://investment-risk-analysis.streamlit.app/)**
&nbsp;·&nbsp; [Methodology](docs/methodology.md) &nbsp;·&nbsp; [PRD](docs/PRD.md) &nbsp;·&nbsp; [Interview prep](docs/interview_prep.md)

---

## The main thing I found

I tested 6 allocation strategies walk-forward against a simple equal-weight portfolio, on the same 120 stocks. Only **one of them** — the max-Sharpe optimizer — actually beat equal-weight in a way that holds up statistically (95% confidence interval on the Sharpe difference doesn't touch zero). The other five looked better on paper but I can't say with confidence they weren't just lucky over this particular 7-year stretch.

I also built a volatility-forecasting model with LightGBM that clearly beats EWMA and GARCH on prediction accuracy (statistically significant, not just a slightly better number). But when I actually plugged that better forecast into the portfolio optimizer, it didn't make the portfolio perform better. Two genuinely different things — being good at predicting volatility, and that prediction actually helping you make more money — and I think that's worth showing rather than hiding.

<p align="center">
  <img src="results/figures/equity_curves.png" width="90%" alt="Equity curves: six strategies vs Nifty 50">
</p>

## Results

Everything below comes straight out of `python scripts/run_backtests.py` — I didn't touch any of these numbers by hand.

### Strategy comparison (walk-forward, net of 15bps transaction costs)

|                           | CAGR   | Ann.Vol   |   Sharpe |   Sortino | MaxDD   |   Calmar |   Beta | Alpha   | InfoRatio   | HitRate   | AvgTurnover   | CostDrag   | Ann.Return   | Ulcer   |   Skew |   Kurtosis |   Treynor |
|:--------------------------|:-------|:----------|---------:|----------:|:--------|---------:|-------:|:--------|:------------|:----------|:--------------|:-----------|:-------------|:--------|-------:|-----------:|----------:|
| equal_weight              | 22.0%  | 19.2%     |     0.81 |      1.09 | -34.3%  |     0.64 |   0.92 | 9.1%    | 0.82        | 53.9%     | 24.5%         | 0.4%       | 21.8%        | 9.6%    |  -1.19 |      15.43 |      0.17 |
| inverse_vol               | 20.7%  | 18.5%     |     0.77 |      1.05 | -33.1%  |     0.62 |   0.89 | 8.1%    | 0.73        | 52.8%     | 24.8%         | 0.4%       | 20.5%        | 9.5%    |  -1.13 |      15.97 |      0.16 |
| min_variance              | 18.6%  | 16.5%     |     0.74 |      1.02 | -30.8%  |     0.6  |   0.75 | 6.9%    | 0.49        | 64.0%     | 32.5%         | 0.5%       | 18.4%        | 9.4%    |  -0.71 |      16.73 |      0.16 |
| risk_parity               | 21.2%  | 18.2%     |     0.8  |      1.09 | -33.3%  |     0.64 |   0.87 | 8.5%    | 0.77        | 57.3%     | 25.2%         | 0.4%       | 20.9%        | 9.5%    |  -1.09 |      15.66 |      0.17 |
| max_sharpe                | 25.9%  | 21.1%     |     0.9  |      1.25 | -35.4%  |     0.73 |   0.95 | 12.5%   | 0.92        | 60.7%     | 28.0%         | 0.4%       | 25.3%        | 10.3%   |  -0.77 |      11.56 |      0.2  |
| score_based               | 25.1%  | 22.2%     |     0.84 |      1.15 | -34.8%  |     0.72 |   0.99 | 11.8%   | 0.85        | 58.4%     | 27.2%         | 0.4%       | 24.9%        | 10.0%   |  -1.03 |      14.3  |      0.19 |
| min_variance (sample cov) | 18.6%  | 16.5%     |     0.73 |      1.02 | -30.7%  |     0.6  |   0.75 | 6.9%    | 0.48        | 64.0%     | 33.0%         | 0.5%       | 18.4%        | 9.4%    |  -0.69 |      16.64 |      0.16 |
| Nifty 50 (buy & hold)     | 12.4%  | 17.6%     |     0.39 |      0.53 | -38.4%  |     0.32 |   1    | 0.0%    | —           | 0.0%      | —             | —          | 13.2%        | 7.7%    |  -1.13 |      18.48 |      0.07 |

### Is any of this actually significant, or did I just get lucky?

I ran a stationary block bootstrap (2000 iterations) on the Sharpe difference between each strategy and Nifty 50. If the 95% confidence interval doesn't cross zero, I'm calling it real. More on this in [Methodology §10](docs/methodology.md#10-statistical-significance-backteststatspy).

| strategy                  |   sharpe_diff |   ci_95_low |   ci_95_high |   bootstrap_p |   active_t_stat |   active_p |   PSR |   DSR | significant_5pct   |
|:--------------------------|--------------:|------------:|-------------:|--------------:|----------------:|-----------:|------:|------:|:-------------------|
| equal_weight              |         0.412 |      -0.026 |        0.843 |         0.064 |            2.22 |      0.026 | 0.982 | 0.774 | False              |
| inverse_vol               |         0.377 |      -0.05  |        0.806 |         0.091 |            1.98 |      0.047 | 0.978 | 0.747 | False              |
| min_variance              |         0.344 |      -0.102 |        0.814 |         0.171 |            1.3  |      0.195 | 0.974 | 0.72  | False              |
| risk_parity               |         0.408 |      -0.02  |        0.857 |         0.065 |            2.07 |      0.038 | 0.982 | 0.772 | False              |
| max_sharpe                |         0.505 |       0.014 |        0.986 |         0.042 |            2.48 |      0.013 | 0.991 | 0.842 | True               |
| score_based               |         0.445 |      -0.044 |        0.914 |         0.07  |            2.37 |      0.018 | 0.986 | 0.8   | False              |
| min_variance (sample cov) |         0.34  |      -0.108 |        0.807 |         0.177 |            1.27 |      0.203 | 0.973 | 0.717 | False              |

### How did these hold up during actual crashes?

|                                    | equal_weight   | min_variance   | score_based   | Nifty 50   |
|:-----------------------------------|:---------------|:---------------|:--------------|:-----------|
| COVID crash (Feb-Mar 2020)         | -33.3%         | -29.9%         | -33.6%        | -36.5%     |
| COVID recovery (Apr-Dec 2020)      | 86.7%          | 72.7%          | 78.3%         | 83.7%      |
| 2022 rate shock (Jan-Jun 2022)     | -20.2%         | -20.6%         | -17.5%        | -11.9%     |
| Oct 2021 - Mar 2023 (flat market)  | -16.8%         | -14.4%         | -11.1%        | -6.0%      |
| 2024-25 correction (Sep 24-Feb 25) | -25.8%         | -23.2%         | -27.9%        | -15.6%     |

<p align="center">
  <img src="results/figures/drawdowns.png" width="90%" alt="Drawdown comparison">
</p>

Worth pointing out: during the 2022 rate-hike selloff and the late-2024 correction, my strategies actually lost *more* than plain Nifty did. Makes sense once you think about it — I'm holding 20-25 stocks, Nifty holds 50, so there's less diversification cushioning a broad, sector-agnostic drop.

### VaR — and whether it actually held up

Anyone can compute a Value-at-Risk number. The question is whether it's calibrated correctly, so I backtested it with Kupiec's test (did the number of breaches match what was expected?) and Christoffersen's test (were the breaches spread out, or all clumped together in one bad stretch?).

|                |   n_obs |   n_breaches |   expected |   breach_rate |   kupiec_p |   christoffersen_p |   cc_p | verdict   |
|:---------------|--------:|-------------:|-----------:|--------------:|-----------:|-------------------:|-------:|:----------|
| historical@95% |    1572 |           88 |       78.6 |        0.056  |     0.2854 |             0.0111 | 0.0224 | REJECT    |
| historical@99% |    1572 |           22 |       15.7 |        0.014  |     0.1333 |             0.4292 | 0.237  | PASS      |
| parametric@95% |    1572 |           69 |       78.6 |        0.0439 |     0.257  |             0.1114 | 0.1481 | PASS      |
| parametric@99% |    1572 |           31 |       15.7 |        0.0197 |     0.0006 |             0.6417 | 0.0026 | REJECT    |

<p align="center">
  <img src="results/figures/var_breaches.png" width="90%" alt="VaR breach chart">
</p>

Historical VaR at 95% fails the Christoffersen test — the breaches weren't random, they clustered around the 2020 and 2022 crashes, which is a known weakness of this method during volatile periods. I'm leaving that REJECT in the table instead of quietly picking a different method that passes.

### Volatility forecasting: LightGBM vs. EWMA vs. GARCH(1,1)

|                       |   RMSE |    MAE |   QLIKE |     R2 |     n |   RMSE_vs_EWMA_% |   QLIKE_vs_EWMA_% |
|:----------------------|-------:|-------:|--------:|-------:|------:|-----------------:|------------------:|
| Random walk (RV21)    | 0.1538 | 0.1025 |  0.4126 | 0.0149 | 10590 |           7.3342 |           24.1868 |
| EWMA(0.94)            | 0.1433 | 0.0945 |  0.3322 | 0.1449 | 10590 |           0      |            0      |
| Hybrid (RV21+RV252)/2 | 0.1335 | 0.0881 |  0.2838 | 0.2573 | 10590 |          -6.8022 |          -14.5868 |
| LightGBM              | 0.1281 | 0.0796 |  0.2833 | 0.3163 | 10590 |         -10.5786 |          -14.7226 |
| GARCH(1,1)-t          | 0.1495 | 0.0994 |  0.2988 | 0.0684 | 10590 |           4.3803 |          -10.0475 |

<p align="center">
  <img src="results/figures/vol_pred_vs_actual.png" width="90%" alt="Forecast vs realised volatility">
</p>

And the significance tests (Diebold-Mariano) behind those numbers:



| comparison                     | loss   |   DM_stat |   p_value |     n | verdict                   |
|:-------------------------------|:-------|----------:|----------:|------:|:--------------------------|
| LightGBM vs Random walk (RV21) | QLIKE  |   -10.678 |    0      | 10658 | LightGBM better (5%)      |
| LightGBM vs Random walk (RV21) | MSE    |    -9.065 |    0      | 10658 | LightGBM better (5%)      |
| LightGBM vs EWMA(0.94)         | QLIKE  |    -4.46  |    0      | 10658 | LightGBM better (5%)      |
| LightGBM vs EWMA(0.94)         | MSE    |    -6.94  |    0      | 10658 | LightGBM better (5%)      |
| LightGBM vs GARCH(1,1)-t       | QLIKE  |    -1.402 |    0.1609 | 10590 | no significant difference |
| LightGBM vs GARCH(1,1)-t       | MSE    |    -8.608 |    0      | 10590 | LightGBM better (5%)      |


LightGBM clearly beats the naive random-walk and EWMA baselines. Against GARCH it wins on plain squared error but the difference isn't statistically significant on QLIKE (p ≈ 0.16), which is the metric that actually matters more for volatility forecasts. I could've just reported the MSE number and called it a clean win — didn't feel right.

### Does the better forecast actually improve the portfolio, though?

|                               | CAGR   | Ann.Vol   |   Sharpe |   Sortino | MaxDD   |   Calmar |   Beta | Alpha   |   InfoRatio | HitRate   | Ann.Return   | Ulcer   |   Skew |   Kurtosis |   Treynor |
|:------------------------------|:-------|:----------|---------:|----------:|:--------|---------:|-------:|:--------|------------:|:----------|:-------------|:--------|-------:|-----------:|----------:|
| min_variance + vol forecast   | 16.7%  | 16.2%     |     0.64 |      0.89 | -30.5%  |     0.55 |   0.75 | 5.3%    |        0.34 | 56.2%     | 16.8%        | 9.4%    |  -0.77 |      16.56 |      0.14 |
| risk_parity + vol forecast    | 20.8%  | 18.1%     |     0.79 |      1.07 | -33.4%  |     0.62 |   0.88 | 8.2%    |        0.74 | 56.2%     | 20.5%        | 9.5%    |  -1.11 |      15.76 |      0.16 |
| min_variance (historical vol) | 18.6%  | 16.5%     |     0.74 |      1.02 | -30.8%  |     0.6  |   0.75 | 6.9%    |        0.49 | 64.0%     | 18.4%        | 9.4%    |  -0.71 |      16.73 |      0.16 |
| risk_parity (historical vol)  | 21.2%  | 18.2%     |     0.8  |      1.09 | -33.3%  |     0.64 |   0.87 | 8.5%    |        0.77 | 57.3%     | 20.9%        | 9.5%    |  -1.09 |      15.66 |      0.17 |

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

# 2. download the stock data (~2 min, needs internet)
python scripts/fetch_data.py

# 3. train the volatility model (~8 min — includes fitting GARCH per stock)
python scripts/run_vol_model.py

# 4. run the full backtest + stats (~9 min)
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
