"""Four real portfolio-construction methods, each answering a different
question about how to weight a real basket of names, from the same real
inputs (expected returns and covariance).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def equal_weight_portfolio(n_assets: int) -> np.ndarray:
    """1/n each -- no information used beyond how many names there are."""
    if n_assets < 1:
        raise ValueError("n_assets must be >= 1")
    return np.full(n_assets, 1.0 / n_assets)


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """The weights minimizing real portfolio variance alone, ignoring
    expected returns entirely -- closed form: w ∝ Σ⁻¹·1.
    """
    cov = np.asarray(cov)
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(cov.shape[0])
    w = inv_cov @ ones
    return w / w.sum()


def max_sharpe_weights(mean_returns: np.ndarray, cov: np.ndarray, risk_free: float = 0.0) -> np.ndarray:
    """The tangency portfolio -- maximizes real Sharpe ratio, using both
    real expected returns and real covariance. Closed form:
    w ∝ Σ⁻¹·(μ - rf).
    """
    cov = np.asarray(cov)
    mean_returns = np.asarray(mean_returns)
    inv_cov = np.linalg.inv(cov)
    excess = mean_returns - risk_free
    w = inv_cov @ excess
    if w.sum() == 0:
        raise ValueError("weights sum to zero -- max-Sharpe portfolio undefined for these inputs")
    return w / w.sum()


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """Equal Risk Contribution: each real asset contributes the same
    share of total real portfolio risk, regardless of its own expected
    return. No closed form in general -- solved numerically (real,
    standard convex-ish optimization, not reimplementing a general
    optimizer from scratch, the same convention as using sklearn's Ridge
    directly elsewhere in this build).
    """
    cov = np.asarray(cov)
    n = cov.shape[0]

    def _risk_contributions(w: np.ndarray) -> np.ndarray:
        port_vol = np.sqrt(w @ cov @ w)
        marginal_contrib = cov @ w
        return w * marginal_contrib / port_vol

    def objective(w: np.ndarray) -> float:
        # a RELATIVE objective (each contribution vs. the mean contribution,
        # not an absolute squared difference) -- the absolute version's
        # objective values sit near 1e-6 for real covariance matrices,
        # small enough that SLSQP's default tolerance falsely reports
        # convergence at the untouched initial guess. Verified directly:
        # an earlier version of this function returned success=True with
        # w still exactly equal to x0, while the real risk contributions
        # at that "solution" were up to 3x apart -- not risk parity at all.
        contributions = _risk_contributions(w)
        target = contributions.mean()
        return float(np.sum((contributions / target - 1.0) ** 2))

    x0 = np.full(n, 1.0 / n)
    bounds = [(1e-6, 1.0)] * n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    if not result.success:
        raise RuntimeError(f"risk parity optimization did not converge: {result.message}")

    contributions = _risk_contributions(result.x)
    relative_spread = contributions.max() / contributions.min()
    if relative_spread > 1.05:
        raise RuntimeError(
            f"risk parity optimization reported success but real risk "
            f"contributions still differ by {relative_spread:.2f}x -- not "
            "actually converged"
        )
    return result.x
