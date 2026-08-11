# Interview prep — questions I expect, and how I'd answer them

I put this together by imagining what someone with an actual finance background would ask if they looked closely at this project, and writing down honest answers using the real numbers instead of vague hand-waving.

## "Walk me through this project."

The short version: I rebuilt a stock-picking backtest that originally tested one strategy on one lucky path with monthly data. I turned it into a walk-forward comparison of six allocation strategies over ten years of daily NSE data, with real transaction costs, statistical significance testing, and a volatility-forecasting model I benchmarked against GARCH. The most useful thing I found wasn't "my strategy wins" — it was that only one of six strategies significantly beat equal-weight, and a genuinely better volatility forecast didn't actually translate into a better portfolio. I'd rather show up with a project that found that than one claiming some inflated CAGR number.

## "What's your Sharpe ratio / CAGR?"

I'd point to the comparison table instead of quoting one number. Max-Sharpe came out on top at 25.9% CAGR and 0.90 Sharpe vs. Nifty's 12.4% and 0.39 — but the more honest answer is in `results/tables/significance_tests.csv`: only max-Sharpe has a bootstrap 95% confidence interval that actually excludes zero versus equal-weight. Everything else is directionally positive but I can't say it's distinguishable from luck over this particular sample. I'd rather say that out loud than round up.

## "Isn't 20%+ CAGR suspiciously high?"

Yeah, and I'd bring it up myself before getting asked. It's inflated by survivorship bias in the universe (`docs/methodology.md#1`). The stock list is a current snapshot, not a point-in-time reconstruction, so it never got the chance to lose money on a stock that actually got delisted along the way. That's a real limitation I haven't solved — what I did instead was keep known underperformers in the universe on purpose (Vodafone Idea, BHEL, SAIL, Zee, Paytm) and lean on relative comparisons (strategy vs. equal-weight vs. Nifty, same universe) instead of resting anything on the raw absolute number.

## "How do you know your backtest isn't leaking future information?"

Two parts to this. Structurally: every rebalance decision on date *t* only ever sees `returns.loc[:t]`. Positions get applied at *t*'s close and start earning P&L from *t+1*, never from *t* itself. But I didn't just want to claim that — `tests/test_leakage.py` actually proves it, by corrupting only the future portion of a price series and checking that every decision made before that point comes out bit-for-bit identical either way. That's a stronger claim than "I read the code carefully."

## "Why volatility forecasting instead of price prediction?"

Volatility clusters — it's autocorrelated, which is basically the entire premise behind GARCH models going back to 1986 — so it's genuinely something you can forecast at a 21-day horizon. Daily or monthly price *levels* for liquid large-caps are close enough to a random walk that a model claiming high accuracy on that is almost always accidentally leaking yesterday's price into its own features, not finding real signal. I wanted something I could actually defend, not something that looks impressive right up until someone asks what my train/test split was.

## "Your vol model beat EWMA — so what?"

The interesting part isn't really that it beat EWMA on RMSE and QLIKE, though it did, and it's statistically significant (Diebold-Mariano p < 0.001 against both random-walk and EWMA). The interesting part is that plugging the better forecast into the portfolio's covariance matrix didn't uniformly improve performance — `results/tables/vol_forecast_impact.csv` shows min-variance actually got worse with it. A better forecast isn't automatically a better input for a downstream optimizer that's sensitive to a totally different kind of error. That's a real thing that happens in applied quant work and doesn't get talked about much, so I'd rather report it honestly than quietly pick whichever config looks best.

## "What's the difference between VaR and CVaR, and why report both?"

VaR at 95% is saying "on the worst 5% of days, you lose at least this much." CVaR (expected shortfall) is saying "on those worst 5% of days, here's what you lose on average" — it's the mean of the tail beyond VaR, so it's always at least as large as VaR. CVaR is also sub-additive, meaning diversifying can never make it worse, which isn't technically true for VaR. Regulators have been leaning toward CVaR for exactly that reason. I report both because VaR is the number people actually ask for and CVaR is the one that's mathematically better behaved.

## "How do you know your VaR model is any good?"

I backtested it instead of just trusting a number I computed once. Kupiec's test checks whether the count of breaches matches the target rate. Christoffersen's test checks whether those breaches were spread out or clustered together — a model can pass the first and completely fail the second (right number of breaches overall, but they all happened in the same bad week), and just counting breaches wouldn't catch that. In this project, historical VaR at 95% actually fails Christoffersen — the breaches clustered around the 2020 and 2022 drawdowns, which is a textbook failure mode for rolling-window historical VaR under volatility clustering. I'd explain that as a real finding about the method's limits, not hide the REJECT.

## "Why Ledoit-Wolf shrinkage instead of just sample covariance?"

With around 20-30 selected stocks and a 36-month window, sample covariance ends up with a condition number in the hundreds — the smallest eigenvalues are basically pure estimation noise, and a mean-variance optimizer will chase exactly those noisy directions because they look like free risk reduction. That's the standard explanation for why naive Markowitz portfolios fall apart out of sample. Shrinking toward a scaled identity fixes the conditioning with no hyperparameter to tune — I wrote this up in `docs/decisions/ADR-001` and showed it directly in `notebooks/02_risk_engine.ipynb`.

## "Why not use cvxpy for the optimization?"

At this scale — 20-30 assets, one objective, a few linear constraints — scipy's SLSQP converges reliably, and I didn't want to add a heavier solver dependency for a problem that doesn't actually need it. I wrote the trade-off down in `docs/decisions/ADR-002` rather than just picking silently. If this scaled up to hundreds of assets with turnover or factor-neutrality constraints, I'd switch to a real convex solver for the optimality guarantees.

## "What would you do differently with more time or a real budget?"

Get point-in-time index constituents from a paid data vendor, since that kills the survivorship problem at the root. Use a risk-free rate that actually moves over time instead of a flat 6.5%. Add a proper factor-attribution layer (Fama-French style) so I could explain *why* a strategy outperformed, not just *that* it did. All three are written up in `docs/methodology.md#12`.

## "What was the hardest bug?"

The forward-volatility target for the ML model. It's really easy to write `returns.rolling(h).std().shift(-h)`, get the shift direction backwards, and end up training on *past* volatility mislabeled as the future — which silently produces a model that looks great in-sample and is worthless everywhere else. I caught it by writing a test that checks the exact alignment against a hand-computed slice (`test_leakage.py::test_forward_vol_target_is_aligned_to_the_future`) instead of just trusting that a visual scan of the code would catch it.
