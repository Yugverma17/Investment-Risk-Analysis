# Interview prep — questions this project will invite, and how to answer them

Numbers below are placeholders (`[X]`) filled from `results/tables/*.csv` once
the pipeline has run — see the README for the live figures. The point of this
file is the *shape* of the answer, not memorising a number.

## "Walk me through this project."

Thirty-second version: "I rebuilt a stock-picking backtest that originally
tested one strategy on one lucky path with monthly data. I turned it into a
walk-forward comparison of six allocation strategies over ten years of daily
NSE data, with transaction costs, statistical significance testing, and a
volatility-forecasting model I benchmarked against GARCH. The most useful
result wasn't 'my strategy wins' — it was that only one of six strategies
significantly beat equal-weight, and a genuinely better volatility forecast
did *not* translate into a better portfolio. I'd rather show you a project
that found that than one that claims 40% CAGR."

## "What's your Sharpe ratio / CAGR?"

Answer with the comparison table, not a single number: "[best strategy] hit
[X] CAGR and [X] Sharpe vs. Nifty's [X] and [X] — but the honest answer is in
`results/tables/significance_tests.csv`: only [max_sharpe] shows a bootstrap
95% CI that excludes zero versus equal-weight. The rest are directionally
positive but not distinguishable from luck over this sample. I'd rather say
that than round up."

## "Isn't 20%+ CAGR suspiciously high?"

Yes, and I'd flag it before you ask: it's inflated by survivorship bias in
the universe (see `docs/methodology.md#1`). The universe is a current
snapshot, not point-in-time, so it never had the chance to lose money on a
stock that got delisted along the way. That's a real, unresolved limitation
— I mitigated it by deliberately keeping known underperformers (IDEA, BHEL,
SAIL, ZEEL, PAYTM) in the universe and by making every headline comparison
relative (strategy vs. equal-weight vs. Nifty, same universe) rather than
resting the claim on the absolute number.

## "How do you know your backtest isn't leaking future information?"

Two answers. First, structurally: every rebalance decision at date *t* only
sees `returns.loc[:t]`; positions are applied at *t*'s close and start
earning P&L from *t+1*, never *t* itself. Second, I don't just assert that —
`tests/test_leakage.py` proves it by corrupting only the *future* portion of
a price series and asserting every quantity decided before the corruption
point is bit-for-bit identical between the corrupted and uncorrupted runs.
That's a stronger claim than a code review can make by inspection alone.

## "Why volatility forecasting and not price prediction?"

Volatility clusters — it's autocorrelated, which is the empirical fact
behind every GARCH paper since 1986 — so it's genuinely forecastable at a
21-day horizon. Daily or monthly price *levels* for liquid large-caps are
close to a random walk at that horizon; a model that claims to predict them
with high accuracy is almost always leaking the previous price into the
feature set, not finding real signal. I wanted to build something I could
defend, not something that looks impressive until someone asks "what's your
train/test split."

## "Your vol model beat EWMA — so what?"

The interesting part isn't that it beat EWMA on RMSE/QLIKE (it did,
significantly — Diebold-Mariano p < 0.001 against both the random-walk and
EWMA baselines). It's that plugging the better forecast into the portfolio's
covariance matrix did **not** uniformly improve performance —
`results/tables/vol_forecast_impact.csv` shows min-variance actually got
worse. A better forecast isn't automatically a better input to a downstream
optimizer that's sensitive to a different kind of error. That's a real,
underreported finding in applied quant work, and I'd rather report it than
cherry-pick the config that looks best.

## "What's the difference between VaR and CVaR, and why do you report both?"

VaR at 95% says "on the worst 5% of days, you lose *at least* this much."
CVaR (expected shortfall) says "on those worst 5% of days, you lose *on
average* this much" — it's the mean of the tail beyond VaR, so it's always
≥ VaR and, unlike VaR, it's sub-additive (diversifying can never make CVaR
worse, whereas VaR technically can). Regulators increasingly prefer CVaR for
exactly that reason. I report both because VaR is the number everyone asks
for and CVaR is the number that's actually coherent.

## "How do you know your VaR model is any good?"

I backtested it rather than trusting a single computed number. Kupiec's test
checks whether the *count* of breaches matches the target rate; Christoffersen's
test checks whether breaches were *independent* rather than clustered — a
model can pass the first and fail the second (right number of breaches, all
in one bad week), which plain breach-counting hides. In this project,
historical VaR at 95% was rejected by Christoffersen — breaches clustered
around 2020/2022 drawdowns, which is the textbook failure mode of a
rolling-window historical VaR under volatility clustering. I'd explain that
as a finding about the method, not hide the REJECT verdict.

## "Why Ledoit-Wolf shrinkage instead of just using sample covariance?"

With ~20-30 selected assets and a 36-month window, sample covariance has a
condition number in the hundreds — its smallest eigenvalues are almost pure
estimation noise, and a mean-variance optimizer actively exploits those noisy
directions because they look like free risk reduction. That's the textbook
explanation for why naive Markowitz portfolios are unstable out of sample.
Shrinkage toward a scaled identity fixes the conditioning with no
hyperparameter to tune — see `docs/decisions/ADR-001` and the direct
demonstration in `notebooks/02_risk_engine.ipynb`.

## "Why not use cvxpy for the optimization?"

At this scale — 20-30 assets, one quadratic or ratio objective, a few linear
constraints — scipy's SLSQP converges reliably and I didn't want a
heavyweight solver dependency for a problem that doesn't need it. I wrote
that trade-off down explicitly in `docs/decisions/ADR-002` rather than just
picking one silently — if this were scaled to hundreds of assets with
turnover or factor-neutrality constraints, I'd revisit that and move to a
real convex solver for the optimality guarantees.

## "What would you do differently with more time / a real budget?"

Point-in-time index constituents from a paid vendor (kills the survivorship
problem at the root), a time-varying risk-free rate instead of a constant
6.5%, and a proper factor-attribution layer (Fama-French style) to explain
*why* a strategy outperformed rather than just *that* it did. All three are
in `docs/methodology.md#12`.

## "What was the hardest bug?"

The forward-volatility target for the ML model. It's easy to write
`returns.rolling(h).std().shift(-h)` and get the shift direction backwards —
off by one and you're training on the *past* volatility labeled as the
future, which silently produces a model that looks great in-sample and is
worthless. I caught it by writing a test that checks the exact alignment
against a hand-computed slice
(`test_leakage.py::test_forward_vol_target_is_aligned_to_the_future`) rather
than trusting a visual scan of the code.
