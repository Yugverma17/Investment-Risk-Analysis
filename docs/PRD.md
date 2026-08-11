# Product Requirements — RiskLens

## The problem

If you're a retail investor in India and you want a diversified stock portfolio, your options aren't great. You can pay 1-2% AUM to an actively managed mutual fund and just trust the manager, with no real way to evaluate whether they're any good. You can DIY it, but then you're on your own for position sizing and risk measurement beyond "did I make money." Or you can use a robo-advisor or smallcase, which gives you an allocation without telling you *why* it picked those weights or *how risky* it actually is in terms you can understand.

The specific gap I noticed: most of these tools show you past returns front and center, and downside risk almost as an afterthought. You rarely see something like "if a month like March 2020 happens again, you could lose ₹X" stated plainly before you put money in.

## Who this is for

**Rohan, 27, software engineer in Bangalore.** Has about ₹40k/month he could invest, has read up on SIPs and index funds, is halfway convinced picking stocks himself could beat an index fund but has no real system for doing it. Right now he either buys whatever his colleagues are talking about or just defaults to a Nifty index fund because he can't decide. What he actually wants is to understand *why* a portfolio looks the way it does, not just get handed a number.

**Priya, 34, working parent in Mumbai.** Has around ₹15L sitting in savings, earmarked for her kid's education in about 7 years. Very loss-averse — the number she actually cares about is "how much could I lose in a bad year," not the expected return. Right now she's stuck with FDs and one mutual fund SIP because nothing gives her a straight answer to that question.

## What people actually need this to do

1. When I have money to invest, help me build a diversified portfolio that matches how much loss I can actually stomach, without me having to research a hundred stocks myself.
2. When I'm holding a portfolio, tell me in rupees — not a vague percentage — how much I could realistically lose in a bad week or month, so I can decide if that's something I signed up for.
3. When someone claims a strategy "beats the market," give me a way to check if that claim would actually survive real costs and bad luck, so I don't get talked into something that only looks good because of a cherry-picked backtest.

## Scope

### What's in this version
- Risk-profile-driven portfolio construction (Conservative / Balanced / Aggressive) over a fixed set of NSE large/mid-cap stocks.
- Walk-forward backtested comparison against 5 other strategies plus Nifty 50, with real transaction costs and statistical significance testing baked in.
- VaR and CVaR shown in rupees, and backtested for calibration (Kupiec/Christoffersen), not just computed once and trusted.
- A 21-day volatility forecasting model, mainly to show the risk model can actually be data-driven instead of purely backward-looking.
- A Streamlit dashboard that ties all of this together for whatever profile and capital amount someone picks.

### What's deliberately not in this version
- No live brokerage integration or order execution — this stays an analysis tool, not something that actually trades. No financial transaction ever gets executed by any of this code.
- No personalized investment advice. Everything here is educational, not a recommendation tailored to anyone's actual financial situation — every simplifying assumption is written out in `docs/methodology.md`.
- No options, derivatives, debt instruments, or international equities.
- No point-in-time historical index membership — I looked and couldn't find a free source for this, so I flagged it as a real limitation (see methodology §1) instead of quietly pretending the universe is historically accurate.
- No real-time or intraday data. Daily granularity only.

## How I'd actually measure if this worked

| Metric | Target | Why this one |
|---|---|---|
| Backtested Sharpe improvement vs. Nifty 50, 95% CI excluding zero | At least 1 of 6 strategies | One strategy that actually passes a real significance test says more than six strategies each claiming to "beat the market" on a raw point estimate |
| VaR calibration (Kupiec pass rate across confidence levels) | Reported honestly, not gamed | The whole point of a risk tool is honest disclosure — a "PASS" I got by loosening the test until it passes would defeat the purpose |
| Time from picking a risk profile to seeing an allocation + risk report | Under 5 seconds with cached data | Someone should be able to flip through all 3 profiles in one sitting without getting bored waiting |
| Volatility forecast improvement over EWMA (QLIKE), with DM-test significance | Statistically significant | If the ML model can't beat a 3-line baseline with actual statistical backing, it's not worth the added complexity |

## Who else is doing this, briefly

- **Smallcase** — nice pre-built thematic baskets, good UX, but no visible risk methodology or backtest significance testing. You're trusting the brand, not seeing the actual method.
- **INDmoney / Groww's "smart" portfolios** — similar black-box feel. Good at fund/stock discovery, weak on actually showing you the risk math.
- **Zerodha Varsity** — genuinely great risk *education* content, but it's not attached to any personalized, backtested portfolio tool.
- **What RiskLens does differently**: every allocation decision traces back to a documented formula (`docs/methodology.md`), every backtest claim comes with a significance test attached, and every risk number gets checked against what actually happened historically instead of just being computed once and left alone.

## If I kept building this

- **Right now**: ship what's described above.
- **Next**: a real point-in-time universe (would need a paid data vendor), a risk-free rate that actually moves over time instead of staying flat, sector/factor exposure reporting.
- **Later**: handling multiple goals with different horizons and risk profiles at once, and a paper-trading mode that tracks live forward performance against what the backtest predicted — closing the loop between what I claimed and what actually happened.
