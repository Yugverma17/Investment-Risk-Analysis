# ADR-002: scipy SLSQP over a dedicated convex solver (cvxpy)

## Status
Accepted

## Context

Three of the six allocators (min-variance, risk-parity, max-Sharpe) require
constrained numerical optimisation: minimise or maximise a scalar objective
over portfolio weights subject to a simplex constraint (weights sum to 1),
box constraints (per-asset weight caps), and linear sector-cap constraints.
Min-variance and max-Sharpe are convex (quadratic and quadratic-over-linear
respectively); risk-parity is not convex in general, though it is well-behaved
in practice for long-only portfolios.

## Decision

Use `scipy.optimize.minimize(method="SLSQP")` for all three optimised
allocators (`optimize/allocators.py`) rather than a dedicated convex-optimisation
library such as `cvxpy` + a QP backend.

Reasoning:
- Problem sizes here are small (20–30 assets after filtering, one quadratic
  or ratio objective, a handful of linear constraints). SLSQP converges
  reliably at this scale and the runtime is dominated by covariance
  estimation and metric computation, not the optimiser itself.
- Avoids adding a heavyweight dependency (cvxpy plus a QP/SOCP backend such as
  OSQP or ECOS) for a problem that does not need it.
- SLSQP's analytic gradient is trivial to supply for the quadratic objectives
  (`min_variance`), which keeps convergence fast without needing the
  objective to be expressed in a solver-specific disciplined-convex-programming
  form.
- The one non-convex objective (risk-parity's sum-of-squared risk-contribution
  deviations) is not directly expressible as a convex program without a
  reformulation; SLSQP handles it as-is.

## Consequences

- **cvxpy would be the better choice at larger scale** (hundreds of assets,
  more exotic constraints like turnover limits or factor-neutrality) because
  a proper convex solver gives global-optimality guarantees that SLSQP's
  local search does not. This project's universe (≤30 selected names per
  rebalance) does not need that.
- SLSQP can occasionally fail to converge on a difficult constraint
  combination; every allocator falls back to equal-weight
  (`optimize/allocators.py::_clean`, guarded by `res.success`) rather than
  propagating a solver failure into the backtest, which is a deliberate
  robustness choice for an 8-fold walk-forward loop that must not crash on
  one bad fold.
- If the project is extended to a larger universe or turnover/factor
  constraints, this decision should be revisited — SLSQP's local-optimum risk
  grows with problem complexity.
