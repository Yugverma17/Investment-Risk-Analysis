# Resume bullets

Picking 2-3 of these depending on how much space I have. Rule I set for myself: every number in here has to be something I can pull straight from `results/tables/` — nothing I'd have to fudge or explain away if someone asked "how'd you get that."

## Primary (data analyst / data scientist roles)

> Built a walk-forward-backtested portfolio allocation engine over 10 years of NSE equity data (120 stocks, 6 strategies), with transaction-cost modelling and bootstrap significance testing; found only 1 of 6 strategies beat equal-weight with a 95% CI excluding zero, and reported that as the headline result instead of the best-looking raw number.

> Designed and backtested a Value-at-Risk framework (parametric, historical, Monte Carlo, Cornish-Fisher) validated with Kupiec and Christoffersen tests against realised returns; surfaced a Christoffersen rejection (breaches clustered around the 2020/2022 drawdowns) that a simple breach-count check would have missed entirely.

## Secondary (ML/quant-leaning variant)

> Built a LightGBM volatility-forecasting model (21-day horizon, range-based features) that beat EWMA and GARCH(1,1) baselines on RMSE/QLIKE with Diebold-Mariano significance (p<0.001 vs. EWMA); found the improved forecast did NOT uniformly improve downstream portfolio performance when wired into a mean-variance optimizer, and reported that negative result instead of quietly leaving it out.

## Secondary (engineering-rigor variant)

> Wrote a 110+ test suite for a quant backtesting engine, including look-ahead-bias tests that corrupt only future data and assert every historical decision comes out bit-for-bit unchanged — the single highest-value test category in a domain where one silent leakage bug invalidates everything downstream.

## One-liner (LinkedIn headline / summary style)

> Rebuilt a single-notebook stock-picking backtest into a statistically validated, walk-forward portfolio risk engine — where the actual deliverable is knowing which of its own claims hold up and which don't.

---

**Things I'm deliberately not using, even though they'd sound more impressive:**
- "Achieved 25% CAGR" on its own, without the equal-weight/Nifty comparison next to it. That number's inflated by survivorship bias (`docs/methodology.md §1`), and quoting it bare is basically inviting a question I don't want to get asked cold.
- "Predicted stock prices with LightGBM" — the model predicts *volatility*, not price. Blurring that line in a resume bullet either misrepresents what I actually built or gets caught the second someone asks a follow-up.
