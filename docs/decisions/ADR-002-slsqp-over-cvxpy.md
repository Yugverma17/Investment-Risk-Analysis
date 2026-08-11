# ADR-002: scipy SLSQP over a dedicated convex solver (cvxpy)

## Status
Accepted

## Context

Three of the six allocators — min-variance, risk-parity, max-Sharpe — need actual constrained optimization: minimize or maximize some objective over the portfolio weights, subject to weights summing to 1, per-stock weight caps, and linear sector-cap constraints. Min-variance and max-Sharpe are convex problems (quadratic and quadratic-over-linear respectively). Risk-parity technically isn't convex in general, though it behaves well enough in practice for long-only portfolios like these.

## Decision

I used `scipy.optimize.minimize(method="SLSQP")` for all three (`optimize/allocators.py`) instead of pulling in a dedicated convex-optimization library like cvxpy with a QP backend.

Reasoning:
- The problems here are small — 20-30 assets after filtering, one objective, a handful of linear constraints. SLSQP handles this fine, and most of the actual runtime goes to covariance estimation and metric computation anyway, not the optimizer itself.
- Didn't want to add a heavier dependency (cvxpy plus something like OSQP or ECOS underneath it) for a problem that genuinely doesn't need it.
- SLSQP takes an analytic gradient easily for the quadratic objectives (min-variance), so it converges fast without me having to rewrite the objective in cvxpy's disciplined-convex-programming syntax.
- Risk-parity's objective (minimizing the spread of each asset's risk contribution) isn't convex without some reformulation work. SLSQP just handles it directly as-is.

## Consequences

- If this ever scaled up to hundreds of assets, or needed more exotic constraints like turnover limits or factor-neutrality, cvxpy would genuinely be the better call — a real convex solver gives you global-optimality guarantees that SLSQP's local search can't. At the scale this project actually runs at (≤30 names per rebalance), that guarantee isn't worth much in practice.
- SLSQP does occasionally fail to converge on an awkward constraint combination. When that happens, the allocator just falls back to equal-weight (`optimize/allocators.py::_clean`, gated on `res.success`) instead of blowing up the backtest — I wanted the 8-fold walk-forward loop to survive one bad fold rather than crash the whole run over it.
- If I ever extend this to a bigger universe or add turnover/factor constraints, I'd want to revisit this — SLSQP's risk of landing in a local optimum only gets worse as the problem gets more complex.
