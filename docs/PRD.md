# Product Requirements — RiskLens

## Problem

Retail investors in India who want a diversified equity portfolio face three
bad options: pay 1–2% AUM fees to an actively-managed mutual fund whose
manager risk they can't evaluate; do it themselves with no systematic way to
size positions or measure risk beyond "how much did I make"; or use a
robo-advisor / smallcase that gives them a black-box allocation with no
visibility into *why* it picked those weights or *how risky* the result
actually is in rupee terms.

The specific gap: existing retail tools show past returns prominently and
downside risk barely at all. A user rarely sees "if a bad month like March
2020 happens again, you could lose ₹X" stated plainly before they invest.

## Users

**Primary persona — "Rohan," 27, software engineer, Bangalore.**
₹40k/month surplus, has read about SIPs and index funds, mildly convinced
individual stock-picking beats an index fund but has no systematic way to do
it. Currently either buys stocks his colleagues mention or defaults to a
Nifty index fund out of decision paralysis. Wants to understand *why* a
portfolio looks the way it does, not just receive a black-box number.

**Secondary persona — "Priya," 34, working parent, Mumbai.**
Has ₹15L sitting in a savings account earmarked for a 7-year goal (child's
education). Extremely loss-averse; the single number she cares about most is
"how much could I lose in a bad year," not the expected return. Currently
uses a mix of FDs and one mutual fund SIP because nothing gives her a direct
answer to that question.

## Jobs to be done

1. When I have a lump sum or monthly surplus to invest, help me build a
   diversified equity portfolio suited to how much loss I can tolerate, so I
   don't have to research 100+ stocks myself.
2. When I'm holding a portfolio, tell me in rupee terms — not just a
   percentage — how much I could plausibly lose in a bad week/month, so I can
   decide if that matches what I signed up for.
3. When someone claims a strategy "beats the market," let me check whether
   that claim would have survived costs and bad luck, so I don't get talked
   into a strategy that only looks good in a cherry-picked backtest.

## Scope

### In scope (v1 — this project)
- Risk-profile-driven portfolio construction (Conservative / Balanced /
  Aggressive) over a fixed NSE large/mid-cap universe.
- Walk-forward backtested comparison against 5 alternative strategies and the
  Nifty 50, with transaction costs and statistical significance testing.
- VaR/CVaR in rupee terms, backtested for calibration (Kupiec/Christoffersen).
- 21-day volatility forecasting to demonstrate the risk model can be
  data-driven rather than purely backward-looking.
- A dashboard (Streamlit) surfacing all of the above for a chosen profile and
  capital amount.

### Explicitly out of scope (v1)
- Live brokerage integration or order execution (this is an analysis tool,
  not a trading system) — see Prohibited-action policy; no financial
  transaction is ever executed by this software.
- Personalized investment advice — outputs are educational/illustrative, not
  a recommendation tailored to any individual's full financial situation.
  See `docs/methodology.md` for every simplifying assumption.
- Options, derivatives, debt instruments, or international equities.
- Point-in-time historical index membership (see methodology §1 —
  unavailable at zero cost; explicitly flagged as a limitation rather than
  silently assumed away).
- Real-time/intraday data — daily granularity only.

## Success metrics (how v1 would be evaluated if it had real users)

| Metric | Target | Why it's the right metric |
|---|---|---|
| Backtested Sharpe improvement vs. Nifty 50, with 95% CI excluding zero | ≥1 of 6 strategies | A single strategy passing genuine significance testing is a stronger claim than six strategies each "beating the market" on a raw point estimate |
| VaR calibration (Kupiec pass rate across confidence levels tested) | Reported transparently, not gamed | A risk product's core promise is honest risk disclosure — a "PASS" achieved by loosening the test would defeat the entire point |
| Time from "pick a risk profile" to "see an allocation + risk report" | <5 seconds (cached data) | The dashboard's core loop must be fast enough to explore 3 profiles in one sitting |
| Volatility forecast improvement vs. EWMA baseline (QLIKE), with DM-test significance | Statistically significant improvement | An ML feature that isn't demonstrably better than a 3-line baseline doesn't earn its complexity budget |

## Competitive landscape (brief)

- **Smallcase** — thematic pre-built baskets, good UX, no visible risk
  methodology or backtested significance testing; the user trusts the brand,
  not a shown methodology.
- **INDmoney / Groww "smart" portfolios** — similar black-box positioning;
  strong on fund/stock discovery, weak on transparent risk quantification.
- **Zerodha Varsity** — excellent risk *education* content, but no
  personalized, backtested portfolio construction tool attached to it.
- **RiskLens' differentiation**: every allocation decision is traceable to a
  documented formula (`docs/methodology.md`), every backtest claim carries a
  significance test, and every risk number is validated against what
  actually happened historically rather than asserted once and left
  unchecked.

## Roadmap (beyond this iteration, not built)

- **Now**: ship v1 as scoped above.
- **Next**: point-in-time universe via a paid data vendor; time-varying
  risk-free rate; sector/factor exposure reporting.
- **Later**: multi-goal planning (map several goals with different horizons
  to different risk profiles simultaneously); paper-trading mode to track
  live forward performance against the backtest, closing the loop between
  claimed and realised results.
