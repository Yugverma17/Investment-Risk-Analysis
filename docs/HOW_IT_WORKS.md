# How RiskLens Works — Explained From Zero

This document explains the *entire* project, from why it exists to what every important piece of code does, in the simplest language possible. If you've never coded before, never touched the stock market, and don't know what any of these fancy words mean — this is written for you. Nothing is assumed.

Grab a snack. This is long, but by the end you will understand the whole thing.

---

## Table of contents

1. [The one-sentence version](#1-the-one-sentence-version)
2. [Where this started, and what was wrong with it](#2-where-this-started-and-what-was-wrong-with-it)
3. [The big picture: what does the new project actually do?](#3-the-big-picture-what-does-the-new-project-actually-do)
4. [Chapter 1 — Picking which stocks to look at](#4-chapter-1--picking-which-stocks-to-look-at)
5. [Chapter 2 — Getting the price data](#5-chapter-2--getting-the-price-data)
6. [Chapter 3 — What is "risk," actually?](#6-chapter-3--what-is-risk-actually)
7. [Chapter 4 — The six ways to split your money](#7-chapter-4--the-six-ways-to-split-your-money)
8. [Chapter 5 — The time machine (backtesting)](#8-chapter-5--the-time-machine-backtesting)
9. [Chapter 6 — Teaching a computer to guess how bumpy the ride will be](#9-chapter-6--teaching-a-computer-to-guess-how-bumpy-the-ride-will-be)
10. [Chapter 7 — Making sure the code isn't cheating](#10-chapter-7--making-sure-the-code-isnt-cheating)
11. [Chapter 8 — The dashboard (the part you actually click on)](#11-chapter-8--the-dashboard-the-part-you-actually-click-on)
12. [Chapter 9 — What we actually found out](#12-chapter-9--what-we-actually-found-out)
13. [The folder tour](#13-the-folder-tour)
14. [Line-by-line code walkthroughs](#14-line-by-line-code-walkthroughs)
15. [How to run all of this yourself](#15-how-to-run-all-of-this-yourself)
16. [Glossary — every fancy word, defined simply](#16-glossary--every-fancy-word-defined-simply)

---

## 1. The one-sentence version

**RiskLens is a computer program that looks at 10 years of real stock prices from India, tries six different ways of splitting money across ~220 stocks, and then honestly reports which of those ways actually worked better than just doing nothing fancy at all — instead of pretending they all worked.**

That's it. Everything below is just explaining how it does that, piece by piece.

---

## 2. Where this started, and what was wrong with it

Imagine you have ₹1,000 and 20 different types of candy to choose from. You want to buy a mix of candy that you'll enjoy, but you also don't want to accidentally buy 20 candies that all taste like the same flavor and go stale on the same day (that would be "risky" — if one goes bad, they all go bad).

The very first version of this project was **one single notebook** — think of a notebook as one long page of instructions and results, like a recipe written on a single sheet of paper. That notebook:

- Looked at stock prices only **once a month** (imagine trying to understand traffic patterns by only looking outside once a month — you'd miss almost everything)
- Picked stocks using one formula, then checked "did this work?" using the exact same time period it used to pick the stocks — like a student who writes an answer, then grades their own answer using the answer key they already peeked at. Of course they get 100%. That doesn't mean they actually understood the material.
- Never asked "was this actually good, or did I just get lucky?"

That's not a useless first draft — it's a completely normal way to start. But it can't be trusted, and a good engineer's job is to notice that and fix it, not to just make the chart look pretty and stop there.

So the project got rebuilt from scratch, properly.

---

## 3. The big picture: what does the new project actually do?

Think of it like a factory assembly line with distinct stations. A box (data) enters at Station 1 and comes out the other end as a finished, tested product (a portfolio + an honest report card).

```
Station 1: DATA        →  Download 10 years of daily prices for ~220 stocks
Station 2: CLEANING    →  Check nothing is broken or missing, throw out junk
Station 3: RISK MATH   →  Turn prices into numbers like "how bumpy is this stock"
Station 4: STRATEGIES  →  6 different recipes for splitting money across stocks
Station 5: TIME MACHINE→  Pretend to travel back in time and test each recipe honestly
Station 6: AI MODEL    →  Teach a computer to predict future bumpiness
Station 7: TESTS       →  Hundreds of automatic checks that catch mistakes
Station 8: DASHBOARD   →  A website where you can actually play with the results
```

Every one of those stations is its own folder of code. We'll walk through each one.

---

## 4. Chapter 1 — Picking which stocks to look at

**File:** `src/riskengine/data/universe.py`

Before you can do ANYTHING with stocks, you have to decide: which stocks? You can't realistically track every single company on the Indian stock market (there are thousands), so this project picks **222 well-known, easy-to-trade companies** — things like Reliance, TCS, HDFC Bank, Titan, Maruti Suzuki. Companies you'd probably recognize even if you don't follow the stock market.

These 222 companies are organized into groups called **sectors** — like sorting a toy box into "cars," "dolls," "blocks" instead of one giant pile. Here the sectors are things like Financials (banks), IT (software companies), Pharma (medicine companies), Auto (car companies), and 17 others.

### Why does the *specific list* matter so much?

Here's a subtle trap this project deliberately walks into with its eyes open, then explains honestly: if you only pick companies that are **still successful today**, you're cheating without realizing it. Imagine picking your class's "top 10 smartest kids" — but you pick them by looking at who did well on the *final* exam, and then bragging "look how smart my picks were on the final exam!" That's circular. Of course they did well — you picked them because they did well.

The real fix would be "for every single day in the last 10 years, know exactly which companies were considered big and important **on that day**, including companies that later went bankrupt or got tiny." That data costs money and isn't freely available, so this project does the next best thing: it keeps some companies on the list that actually did **badly** over the years (a phone company that struggled, a bank that ran into trouble) instead of only picking winners in hindsight. It's not a perfect fix, but it's an honest, documented one — the project's README says this out loud instead of hiding it.

---

## 5. Chapter 2 — Getting the price data

**File:** `src/riskengine/data/loaders.py`

Once we know *which* 222 stocks we want, we need their actual price history. This is like calling up a library and asking "give me the daily temperature reading for these 222 cities, every single day, for the last 10 years."

The code does this using a free tool called `yfinance`, which talks to Yahoo Finance's stock database. A few important, simple ideas here:

- **We download the "adjusted" price, not the raw price.** Here's why that matters: imagine a company does something called a "stock split" — like taking a ₹100 candy bar and cutting it into two ₹50 candy bars. The total amount of candy didn't change, but if you only looked at the raw price, it would look like the price CRASHED by 50% overnight, which is fake — nothing actually went wrong. "Adjusted" prices fix this automatically so the computer doesn't get confused by fake crashes.
- **We save the data to a file after downloading it once.** Downloading 220 stocks' worth of 10-year history takes a couple of minutes. Nobody wants to wait 2 minutes every single time they want to run a calculation, so the code saves ("caches") the download to a file on the computer, and next time it just reads that file instantly instead of downloading again.
- **Two of our 222 stocks failed to download.** One company (LTIMindtree) and one (Tata Motors) had their ticker symbols change due to corporate restructuring in 2025, so Yahoo Finance's records under the old name came back empty. Instead of hiding this or crashing, the code notices, writes it down in a report, and just moves on with the 220 that worked. This is a small but important idea that repeats throughout this whole project: **when something goes wrong, write it down honestly instead of pretending it didn't happen.**

---

## 6. Chapter 3 — What is "risk," actually?

This is the heart of the whole project, so let's slow down here.

### Volatility = how bumpy the ride is

Imagine two toy cars driving down a street.

- **Car A** moves at a steady, predictable speed the whole time.
- **Car B** sometimes zooms fast, sometimes stops completely, sometimes swerves — it's the same *average* speed as Car A, but the ride feels completely different.

Car B is "more volatile." **Volatility is just a number that measures how much something bounces around**, even if the average stays the same. A stock that moves up 1% then down 1% then up 1% every day is calm (low volatility). A stock that jumps +10% one day and -8% the next is wild (high volatility).

**File:** `src/riskengine/features/volatility.py`

The code has several different ways of measuring "how bumpy," because it turns out some ways are smarter than others. The simplest one just looks at the closing price each day and measures how spread out those daily changes are. A smarter one (called "Parkinson," named after the person who invented it) also looks at the **highest** and **lowest** price the stock touched during the day, not just where it ended up — like judging how wild a rollercoaster is by looking at the whole track, not just where the cart parked at the end. That gives a more accurate bumpiness score using the exact same number of days of data.

### Sharpe ratio = "how much reward did I get for how much stomach-churning I had to endure?"

**File:** `src/riskengine/risk/metrics.py`

Imagine two ways to earn ₹100:
- **Way A:** You do one calm hour of chores, and you get ₹100.
- **Way B:** You do one wild, stressful hour of chores where you might get ₹100 OR you might get nothing, and it's genuinely nerve-wracking not knowing which — but on average across many tries, you also get ₹100.

Even though both pay ₹100 on average, Way A is clearly better — same reward, way less stress. The **Sharpe ratio** is a single number that captures exactly this idea for investments: it's (your extra reward above a "safe" baseline) divided by (how bumpy/stressful the ride was). A HIGHER Sharpe ratio means you got more reward for the amount of stress you took on. A stock or strategy with Sharpe ratio 2.0 is giving you way better "reward per unit of stress" than one with Sharpe ratio 0.4.

### Value-at-Risk (VaR) = "on a bad day, how much could I realistically lose?"

**File:** `src/riskengine/risk/var.py`

This is like a weather forecast, but for money instead of rain. A weather forecaster doesn't say "it will definitely rain 2 inches tomorrow" — they say "there's a 5% chance it rains more than 2 inches." VaR works the same way for your money: **"On the worst 5% of days, I'd expect to lose more than ₹X."**

How do you actually calculate this? One simple way (called "historical VaR," which is what most of the code uses) is genuinely simple: look at the last several hundred days of actual returns, sort them from worst to best, and find the number that only the worst 5% of days were beyond. That's it — no fancy math, just "look at what actually happened before, and find the bad-day cutoff."

### CVaR = "and on that bad day, how bad on average?"

VaR tells you the *cutoff* for a bad day. CVaR (Conditional VaR, also called "Expected Shortfall") goes one step further and says "OK, and once we're already in that worst 5% of days, what's the *average* loss on those days specifically?" It's always a bigger number than VaR, because it's describing the average of the worst outcomes, not just the boundary.

### Why do we check the risk-checker itself?

**File:** `src/riskengine/risk/var_backtest.py`

Here's something clever the project does: it doesn't just trust its own VaR number. It goes back and checks: "I predicted the worst 5% of days would lose more than X. Did that ACTUALLY happen roughly 5% of the time in real life, or was I wrong?"

There are two checks:
1. **Did the count match?** (Called the "Kupiec test.") If VaR said "5% of days will be bad," did roughly 5% of days actually turn out bad — not 15%, not 0.5%?
2. **Were the bad days spread out, or clumped together?** (Called the "Christoffersen test.") Imagine a weather forecaster who correctly predicts "5% of days will be rainy" — but ALL of those rainy days happen to fall in the same single week, and the rest of the year is bone dry. Technically the *count* was right, but the forecaster clearly didn't actually understand rain patterns; they got lucky on the count. This test catches that specific kind of fake-correctness.

In this project, the risk-checker actually **fails** this second test in a couple of places — the bad days did clump together around real crashes (like the COVID crash in 2020). The project reports this honestly instead of hiding it, because that's a genuinely useful thing to know: VaR is less reliable exactly during the times you'd want it to be most reliable (a real crash).

---

## 7. Chapter 4 — The six ways to split your money

**File:** `src/riskengine/optimize/allocators.py`

OK, now imagine you've picked, say, 20 stocks out of the 220 that look promising this quarter. How much of your money do you put into each one? There isn't one obviously correct answer, so this project tries **six different recipes** and honestly compares them.

1. **Equal-weight** — the simplest possible idea. If you picked 20 stocks, put exactly 1/20th of your money in each one. No cleverness at all.
2. **Inverse-volatility** — give MORE money to calmer stocks and LESS money to bumpier stocks. Makes intuitive sense: don't put too many eggs in the shakiest basket.
3. **Minimum-variance** — use real math (matrix algebra) to find the exact mix of money across all 20 stocks that makes the WHOLE portfolio as calm as mathematically possible, taking into account which stocks tend to move together and which move in opposite directions.
4. **Risk-parity** — instead of minimizing total bumpiness, make sure every stock contributes an EQUAL SHARE of the portfolio's total bumpiness. Two banks that always move together only "count" as one risky bet, not two.
5. **Max-Sharpe** — the "textbook perfect" answer: mathematically find the exact mix that gives you the best possible reward-per-stress ratio. Sounds like it should always win... but it's also the most sensitive to guessing wrong about the future, which turns out to matter a lot (more on this later).
6. **Score-based** — this is the original notebook's idea, rebuilt properly. Score every stock on things like "how good is its reward-per-stress ratio" and "how bumpy has it been," combine those scores, and give more money to higher-scoring stocks.

None of these six gets to cheat. They all follow the exact same rules: you must invest ALL your money (no keeping cash under the mattress), you can't bet against a stock (no "negative" holdings), and no single stock can be more than a certain percentage of your money (so you're never "all in" on one company).

---

## 8. Chapter 5 — The time machine (backtesting)

**File:** `src/riskengine/backtest/engine.py`

This is the single most important idea in the whole project, so let's really slow down.

Imagine you built a machine that claims it can predict the winning lottery numbers. How would you actually test if it works? You definitely would NOT show it last week's winning numbers and ask it to predict... last week's winning numbers. That's not a test, that's just copying the answer.

A real test would be: freeze the machine's knowledge at some point BEFORE the draw, have it make a prediction, THEN reveal what actually happened, and see if it was right — for many different draws over time, not just one.

**Backtesting is exactly this idea, applied to picking stocks.** Here's how the code does it, step by step:

1. Pick a date — let's say January 1st, 2018.
2. Only show the computer stock prices **up to and including** that date. It is not allowed to see anything that happens after — not even one extra day.
3. Using ONLY that past information, the computer picks its 20 stocks and decides how much money to put in each.
4. Now "fast forward" 3 months in the actual historical record and see what really happened to those stocks. Record the real result.
5. Move the freeze-date forward by 3 months (to April 1st, 2018) and repeat the entire process — pick new stocks, using only information available as of April 1st.
6. Keep doing this, one 3-month period at a time, all the way from 2018 to 2025.

At the end, you have a long chain of real, honest decisions, each one made with only the information that would have genuinely been available at the time. That's the closest thing to "actually investing for real, 7 years ago, and seeing what happens" that you can do without an actual time machine.

### Two sneaky details that are easy to get wrong

**Detail 1 — the one-day shift.** If the computer decides on January 1st's closing price what to buy, it should NOT get credit for January 1st's own price movement — it didn't know the final closing price early enough that day to act on it. It should only start earning (or losing) money from January 2nd onward. This sounds like a tiny detail, but getting this wrong is one of the most common ways beginners accidentally cheat without realizing it, because it makes results look slightly better than they really would be.

**Detail 2 — trading isn't free.** Every time you buy or sell a stock, in real life you pay a small fee (things like broker charges and government taxes). This project charges **0.15%** of whatever gets traded, every single time the portfolio changes its mix. A strategy that changes its mind a lot and trades constantly gets punished for that realistically, exactly like it would in real life.

### How do we KNOW the time machine isn't secretly cheating?

**File:** `tests/test_leakage.py`

This is genuinely clever, and it's my favorite part of the whole project to explain. Instead of just promising "I definitely didn't let the computer peek at the future, I swear" — the code actually **proves** it, automatically, every time.

Here's the trick: take the real price history, and secretly multiply every price **after** a certain cutoff date by 1.5 (pretend everything got 50% more expensive, but only in the future portion). Then run the whole time-machine test again. **If the code decided everything correctly, every single decision made BEFORE that cutoff date should come out byte-for-byte identical, whether or not you messed with the future.** Why? Because a decision made on, say, June 2020, should have had zero way of knowing that prices in 2023 were about to get artificially inflated in this test. If even one number changes, it means the code accidentally let information leak backward in time — which would be a serious, invisible bug that could make every single result in the whole project fake and misleading.

This test genuinely runs and genuinely passes. That's not something you can fake by just writing a comment saying "trust me."

---

## 9. Chapter 6 — Teaching a computer to guess how bumpy the ride will be

**File:** `src/riskengine/models/vol_forecast.py`

You might expect this project to try to predict *stock prices* — "will TCS go up or down tomorrow?" On purpose, it doesn't do that, and there's a genuinely important reason why.

### Why not just predict the price?

Predicting tomorrow's exact stock price is famously close to impossible, and here's the honest reason: stock prices already reflect almost everything that's publicly known about a company. If it were easy to reliably predict "up" or "down" from public information, everyone would do it, and that predictable pattern would disappear the moment enough people tried to exploit it. Anyone claiming "my AI predicts stock prices with 95% accuracy" is almost always accidentally cheating (often by letting the model see a hint of the answer without realizing it) or is simply wrong.

**But volatility (bumpiness) is a different story.** It turns out bumpy periods tend to stay bumpy for a while, and calm periods tend to stay calm for a while — bumpiness has *momentum*, in a way that price direction just doesn't. This has been well known and studied since the 1980s. So instead of the impossible task ("will the price go up?"), this project does the genuinely achievable one: **"how bumpy will this stock be over the next month?"**

### Meet the four competitors

Think of this like a cooking competition with four chefs, all trying to predict next month's bumpiness for every stock:

1. **"Just look at last month"** (naive baseline) — the laziest possible guess: assume next month will be exactly as bumpy as last month was.
2. **EWMA** — a slightly smarter recipe used across the finance industry for decades. It looks at recent history but gives more weight to the MOST recent days, gradually caring less about older days. A genuinely strong, respected baseline — not a strawman set up to lose.
3. **GARCH** — an even more mathematically sophisticated, classic statistical formula, also decades old and extremely well-trusted by professional risk managers.
4. **LightGBM** — a modern machine-learning model (a specific, well-known "gradient boosting" algorithm). Instead of one fixed formula, it's shown dozens of clues about each stock (how bumpy has it been at different time-scales, how far has the price fallen from its recent peak, what's the overall stock market's mood, and more) and it learns, from thousands of real past examples, its own pattern for combining those clues into a bumpiness guess.

### And the result?

LightGBM, the AI model, genuinely won — it beat all three other competitors, and the code proves the win wasn't just luck using something called a "Diebold-Mariano test" (a statistical test specifically designed to check "is Model A actually reliably better than Model B, or could this be random noise?"). That's a real, defensible result.

**But here's the honest twist**, and it's maybe the single most important lesson in the whole project: when that better bumpiness-prediction was actually plugged into the money-splitting recipes (feeding the AI's better guesses into the min-variance and risk-parity strategies), it **did not reliably make the portfolio perform better.** Being good at one specific sub-task (predicting bumpiness accurately) doesn't automatically mean it helps with the bigger task (making more money safely) — because the bigger task depends on lots of OTHER things too, like which specific stocks got picked and how correlations between stocks behaved. This project reports that honestly instead of quietly hiding the disappointing part and only bragging about the win.

---

## 10. Chapter 7 — Making sure the code isn't cheating

**Folder:** `tests/`

Imagine you write a math homework assignment, and before turning it in, you have a friend who ONLY checks answers — never writes anything themselves, just verifies "is this actually right?" That's exactly what a **test** is in programming.

This project has **110 separate tests**, and every single one runs automatically. A few examples of what they check:

- "If I feed in a stock that goes up exactly 0.1% every single day for a year, does the growth-rate formula produce EXACTLY the number a hand calculator would give?" (This is called a "known-answer test" — you compute the right answer by hand first, then check the code agrees.)
- "Does every money-splitting recipe always add up to exactly 100% of the money, never more, never less?"
- "If I secretly change the future, do all past decisions stay exactly the same?" (the time-machine-cheating test from Chapter 5)
- "Does a Value-at-Risk calculation trained on random, genuinely bell-curve-shaped data come out close to the textbook mathematical answer?"

Why does this matter so much for a project like this? Because financial calculations have a nasty property: **a broken formula usually doesn't crash the program — it just quietly produces a wrong number that looks completely normal.** Nobody gets an error message. You just get a beautiful, confident-looking chart that's secretly lying to you. Tests are the thing standing between "this code runs" and "this code is actually correct," and in a project whose entire point is to be honest about what works and what doesn't, that distinction matters enormously.

---

## 11. Chapter 8 — The dashboard (the part you actually click on)

**File:** `app/streamlit_app.py`

Everything explained so far happens by running Python scripts and staring at spreadsheets of numbers — not exactly friendly for a normal person. So there's also a proper website (built using a tool called Streamlit) where you can just click around.

It has four tabs, like tabs in a binder:

1. **"Build my portfolio"** — pick how much risk you're comfortable with (Conservative / Balanced / Aggressive), how much money you have, and how many years you plan to invest. The dashboard then shows you an actual suggested list of stocks and how much of your money goes into each one, drawn as a pie chart, plus what your worst-case-day loss might realistically look like in real rupees.
2. **"Backtest vs Nifty"** — shows all six strategies' time-machine test results, compared against the Nifty 50 (India's most famous stock market index), as line charts that go up and down over the actual 7+ years of history.
3. **"Risk report card"** — shows whether the Value-at-Risk predictions actually held up in reality (the Kupiec/Christoffersen checks from Chapter 3), and the AI bumpiness-predictor's results from Chapter 6.
4. **"Methodology"** — plain-English explanation of every simplifying assumption the project makes, written so nobody can accuse it of hiding its limitations.

One design detail worth calling out: the "Build my portfolio" tab deliberately shows a warning next to its headline numbers explaining that they look unusually good **because** the stocks were picked specifically for having recently performed well — and points you toward the (more honest, lower) numbers in the Backtest tab instead. A dashboard's job isn't just to show numbers, it's to help you not fool yourself with them.

---

## 12. Chapter 9 — What we actually found out

This is the payoff — after all that machinery, what did we actually learn?

**Finding 1: The fancy strategies didn't clearly beat the simplest one.** Out of six different money-splitting recipes, tested honestly with the time machine across ~220 stocks, only **plain equal-weight** (the simplest possible idea — just split money evenly) showed a statistically real advantage over the Nifty 50 market index. The fancier, more mathematically sophisticated recipes (max-Sharpe, min-variance) looked good on paper but couldn't be proven better than luck. This actually matches a famous, well-respected finding from real academic research (a 2009 paper by DeMiguel, Garlappi & Uppal) — the more things a smart formula has to guess about (like expected future returns for 220 different stocks), the more chances it has to guess wrong, and those wrong guesses can cancel out its cleverness. Simple isn't a failure here — it's the actual finding.

**Finding 2: The AI bumpiness-predictor genuinely worked, but that alone wasn't enough.** As explained in Chapter 6 — a real, statistically proven win at its specific job, that still didn't reliably translate into a better end result.

**Finding 3: The risk-checker (VaR) has real, honest limitations.** It sometimes failed its own "did the bad days clump together unfairly" test, especially around real market crashes — which is a known, textbook weakness of this style of risk measurement, and the project reports it rather than hiding it.

**Finding 4: This kind of portfolio underperforms during certain kinds of crashes.** Because the strategies concentrate money into ~20 stocks (instead of the market index's 50), they had *less* diversification cushion during some broad, sector-wide downturns in 2022 and late 2024 — losing slightly more than the plain market index during those specific stretches.

None of these four findings are the kind of thing a hype-driven, "look how much money my AI made" project would choose to report. That's exactly the point: **the actual deliverable of this project isn't a stock-picking machine — it's a demonstrated ability to test claims honestly and report what's actually true, even when that's less flattering than what was hoped for.**

---

## 13. The folder tour

Here's every folder in the project, in one sentence each:

| Folder | What lives here, in one sentence |
|---|---|
| `src/riskengine/data/` | Decides which 222 stocks to use, downloads their prices, checks the data isn't broken |
| `src/riskengine/features/` | Turns raw prices into "bumpiness" and "market comparison" numbers |
| `src/riskengine/risk/` | Sharpe ratio, VaR, CVaR, and the math that checks a covariance matrix isn't garbage |
| `src/riskengine/optimize/` | The six money-splitting recipes and the rules they all have to follow |
| `src/riskengine/backtest/` | The time machine — runs every recipe honestly through history, plus the statistics that check "was this real, or luck?" |
| `src/riskengine/models/` | The AI bumpiness-predictor (LightGBM) and its GARCH/EWMA competitors |
| `src/riskengine/report/` | Turns all the raw numbers into the charts and tables you actually see |
| `scripts/` | The "push this button to run everything" files |
| `notebooks/` | A step-by-step lab notebook version of the analysis, with charts already showing |
| `app/` | The clickable website (Streamlit dashboard) |
| `tests/` | The 110 automatic homework-checkers |
| `docs/` | Every explanation, including this one |

---

## 14. Line-by-line code walkthroughs

This section takes a handful of the most important, most representative pieces of code and explains **every single line** in plain English. These aren't randomly chosen — they're the pieces that best show off how the project thinks.

### Walkthrough A: the Sharpe ratio formula

File: `src/riskengine/risk/metrics.py`

```python
def sharpe_ratio(returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    ex = returns.dropna() - daily_rf(rf_annual)
    sd = ex.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-10:
        return np.nan
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS))
```

- **`def sharpe_ratio(returns, rf_annual=RISK_FREE_ANNUAL):`** — this defines a reusable recipe named `sharpe_ratio` that needs two ingredients: a list of daily returns, and a "safe baseline" interest rate (which defaults to 6.5% a year if you don't specify one — like the interest you'd get just leaving money in a very safe government bond).
- **`ex = returns.dropna() - daily_rf(rf_annual)`** — `returns.dropna()` throws away any missing/blank days first (you can't do math on a blank spot). Then it subtracts the "safe baseline," turned into a daily number, from every single day's return. `ex` now means "excess return" — how much better (or worse) than the safe, boring option each day was.
- **`sd = ex.std(ddof=1)`** — this calculates the "standard deviation," which is just the technical name for "how spread out / bumpy" these excess returns are. Bigger number = bumpier.
- **`if not np.isfinite(sd) or sd < 1e-10: return np.nan`** — this is a safety check. If the bumpiness comes out as basically zero (meaning literally every day had the exact same return, which happens in test scenarios, or could happen from a data glitch), then dividing by that tiny number next would create a nonsense, enormous, meaningless result. So instead the code says "I don't have a sensible answer for this" (`NaN` means "Not a Number," programming's way of saying "undefined").
- **`return float(ex.mean() / sd * np.sqrt(TRADING_DAYS))`** — the actual formula: (average excess return) divided by (bumpiness), which gives you "reward per unit of stress" for one single day. Multiplying by the square root of 252 (the number of trading days in a year) scales that single-day number up into a yearly number, because that's how the math for combining daily randomness into yearly randomness works out.

### Walkthrough B: historical Value-at-Risk

File: `src/riskengine/risk/var.py`

```python
def historical_var(returns: pd.Series, confidence: float = 0.95, horizon: int = 1) -> float:
    r = _as_array(returns)
    q = np.quantile(r, 1 - confidence)
    return float(-q * np.sqrt(horizon))
```

- **`def historical_var(returns, confidence=0.95, horizon=1):`** — a recipe that needs a list of returns, how confident you want to be (95% is the default — meaning "tell me the cutoff for the worst 5% of days"), and how many days ahead you're asking about.
- **`r = _as_array(returns)`** — converts the data into a plain, simple list of numbers that's easy and fast for the computer to sort and search through.
- **`q = np.quantile(r, 1 - confidence)`** — this is the actual heart of the whole function. `1 - confidence` with the default 0.95 gives `0.05`. "Quantile" means "find the value at this percentile point when everything is sorted." So this line says: *sort every single day's return from worst to best, and find the point where exactly 5% of days were worse than this.* That's your bad-day cutoff, straight from real history — no guessing, no assumed bell curve, just "what actually happened."
- **`return float(-q * np.sqrt(horizon))`** — two things happen here. First, the minus sign: `q` comes out as a negative number (like -0.023, meaning "-2.3%"), but VaR is always reported as a POSITIVE number representing a loss (so "2.3%," not "-2.3%") — that's just a convention this whole project sticks to consistently, so numbers never get confusingly mixed up. Second, multiplying by the square root of the number of days scales a one-day risk estimate up to a multi-day one (this is a standard, if imperfect, shortcut used across the finance industry).

### Walkthrough C: the simplest money-splitting recipe

File: `src/riskengine/optimize/allocators.py`

```python
def equal_weight(returns: pd.DataFrame, cons: Constraints, **_) -> pd.Series:
    assets = list(returns.columns)
    return apply_caps(pd.Series(1.0, index=assets), cons.effective_max_weight(len(assets)))
```

- **`def equal_weight(returns, cons, **_):`** — needs a table of stock returns (each column is one stock), and a set of rules (`cons`, short for "constraints" — things like "no stock can be more than 15% of the money"). The `**_` just means "and ignore any other extra ingredients other recipes might need, since this one doesn't need them."
- **`assets = list(returns.columns)`** — grabs the names of every stock in the table (the column headers), turning them into a simple list, like reading off the labels on a row of jars.
- **`pd.Series(1.0, index=assets)`** — creates a simple list where EVERY stock starts out getting the exact same number: `1.0`. If you have 20 stocks, that's twenty 1.0's in a row — literally "give everyone an identical starting amount."
- **`cons.effective_max_weight(len(assets))`** — checks: given how many stocks we have and the "no more than X% per stock" rule, is that rule even mathematically possible? (For example, if the rule says "max 10% per stock" but you only have 5 stocks, you CAN'T possibly reach 100% total — 5 times 10% is only 50%! So this automatically loosens the rule just enough to make it possible, rather than crashing.)
- **`apply_caps(...)`** — takes those equal starting amounts, turns them into percentages that add up to exactly 100%, and enforces the "no stock over the limit" rule, giving any trimmed-off extra to the other stocks proportionally.

Notice how short and simple this one is compared to the more "clever" ones — that's not accidental. It's the whole point of including it: the simplest possible idea, with almost nothing that could go subtly wrong, ended up being the one that actually held up.

### Walkthrough D: the time-machine loop, in plain English (not literal code — the concept)

File: `src/riskengine/backtest/engine.py`

The real code here is longer and handles a lot of edge cases, but the *heart* of it, stripped down to plain English, is:

```
for each 3-month checkpoint in history, starting from 2018:
    look ONLY at price history up to and including today's checkpoint
    (nothing after this date is allowed to be seen)

    score every stock using only that visible history
    pick the best ~20 stocks according to whichever recipe we're testing
    decide how much money goes into each one

    remember these decisions

    fast-forward 3 real months and record what ACTUALLY happened
    to a portfolio built exactly this way

    charge a small trading fee for whatever changed since last time

move the checkpoint forward 3 months, repeat
```

Every single one of those checkpoints becomes one honest data point. String them all together from 2018 to 2025, and you get a real track record — not a guess, not a single lucky story, but the actual result of making that same disciplined decision, over and over, for seven and a half years.

---

## 15. How to run all of this yourself

If you (or anyone reading this) wants to actually run the project on their own computer:

```bash
# Set up a clean Python environment (like a fresh, empty toolbox)
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev,app]"

# Download the real stock data (takes about a minute)
python scripts/fetch_data.py

# Train the AI bumpiness-predictor and its competitors (about 11 minutes)
python scripts/run_vol_model.py

# Run the full time-machine test on all six strategies (about 15 minutes)
python scripts/run_backtests.py

# Run all 110 automatic checks
pytest

# Open the clickable website
streamlit run app/streamlit_app.py
```

Nothing here needs to be run to *see* the results, though — every chart and table is already saved in the `results/` folder and shown in the README, so you can understand everything just by reading.

---

## 16. Glossary — every fancy word, defined simply

| Word | What it actually means |
|---|---|
| **Portfolio** | A basket of different stocks you own, each in a certain amount |
| **Volatility** | How much a price bounces around, regardless of its average |
| **Sharpe ratio** | Reward earned, divided by how bumpy/stressful the ride was to earn it |
| **Value-at-Risk (VaR)** | "On a bad day, I'd expect to lose at least this much" |
| **CVaR / Expected Shortfall** | "And once it IS a bad day, here's the average loss on days like that" |
| **Backtest** | Honestly testing a strategy using only information that would have been available at each point in the past — a financial "time machine" |
| **Look-ahead bias / leakage** | The bug where a test accidentally lets the computer peek at the future — the single most common way DIY investing backtests secretly lie |
| **Rebalancing** | Periodically re-deciding how much money goes into each stock (this project does it every 3 months) |
| **Covariance / correlation** | A measure of whether two stocks tend to move together (correlated) or independently |
| **Shrinkage** | A math trick that makes a noisy, unreliable estimate (like a covariance matrix built from too little data) more trustworthy by gently pulling it toward a simpler, safer guess |
| **Sector** | A category of similar companies (banks, tech companies, car makers, etc.) |
| **Survivorship bias** | The mistake of only studying things that succeeded, which makes the past look better than it really was |
| **Statistical significance** | A test for whether a result is real and repeatable, or could easily just be random luck |
| **GARCH / EWMA** | Two well-established, decades-old statistical formulas for guessing how bumpy something will be next |
| **LightGBM** | A modern machine-learning algorithm that learns patterns from lots of past examples instead of following one fixed formula |
| **Diebold-Mariano test** | A statistical test that answers "is Model A really more accurate than Model B, or is this just noise?" |
| **Kupiec test / Christoffersen test** | Two checks for whether a risk forecast (like VaR) is actually well-calibrated to what really happens |
| **Nifty 50** | India's most well-known stock market index — 50 of the biggest companies, used as "the market" benchmark to compare everything against |
| **CAGR** | Compound Annual Growth Rate — the single steady yearly growth rate that would have gotten you from the starting amount to the ending amount |
| **Test (in code)** | An automatic check that verifies a piece of code produces the correct answer, so mistakes get caught immediately instead of silently poisoning every result downstream |

---

That's the whole project, top to bottom. If any single piece of this still doesn't make sense, that's a documentation bug, not a "you're not smart enough" problem — the whole point of this file is that it shouldn't be.
