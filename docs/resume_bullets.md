# Resume bullets

Pick 2–3 depending on space. Each is written so every number in it is
reproducible from `results/tables/` by running the scripts in the README —
don't use a number here you can't defend if asked "how did you get that."

## Primary (data analyst / data scientist roles)

> Built a walk-forward-backtested portfolio allocation engine over 10 years
> of NSE equity data (120 stocks, 6 strategies), with transaction-cost
> modelling and bootstrap significance testing; found only 1 of 6 strategies
> beat equal-weight with a 95% CI excluding zero, and reported it as the
> headline result instead of the best raw number.

> Designed and backtested a Value-at-Risk framework (parametric, historical,
> Monte Carlo, Cornish-Fisher) validated with Kupiec and Christoffersen tests
> against realised returns — surfaced a Christoffersen rejection (clustered
> breaches around 2020/2022 drawdowns) that a naive breach-count check would
> have missed.

## Secondary (ML/quant-leaning variant)

> Built a LightGBM volatility-forecasting model (21-day horizon, range-based
> features) that beat EWMA and GARCH(1,1) baselines on RMSE/QLIKE with
> Diebold-Mariano significance (p<0.001 vs. EWMA); found the improved
> forecast did NOT uniformly improve downstream portfolio performance when
> wired into a mean-variance optimizer — an honest negative result, not
> hidden to preserve the narrative.

## Secondary (engineering-rigor variant)

> Wrote a 110+ test suite for a quant backtesting engine, including
> look-ahead-bias tests that corrupt only future data and assert every
> historical decision is bit-for-bit unchanged — the single highest-value
> test category in a domain where a silent leakage bug invalidates every
> other result.

## One-liner (LinkedIn headline / summary style)

> Rebuilt a single-notebook stock-picking backtest into a statistically
> validated, walk-forward portfolio risk engine — the project's main
> deliverable is knowing which of its own claims survive scrutiny.

---

**Do not use, even though it would look more impressive:**
- "Achieved 25% CAGR" alone, without the equal-weight/Nifty comparison — the
  absolute number is inflated by survivorship bias (see
  `docs/methodology.md §1`) and citing it bare invites a question you don't
  want asked cold.
- "Predicted stock prices with LightGBM" — the model predicts *volatility*,
  not price; conflating them in a resume bullet either misrepresents the
  work or gets caught in the first follow-up question.
