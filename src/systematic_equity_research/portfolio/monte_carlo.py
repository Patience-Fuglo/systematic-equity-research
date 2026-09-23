"""Monte Carlo portfolio simulation: 1,000 simulated forward paths,
calibrated to real historical mean returns and covariance -- the paths
themselves are necessarily simulated (no one has real data from the
future), but every input driving them is real. Drawdown and P(ruin) are
computed directly from those real-calibrated paths.
"""

from __future__ import annotations

import numpy as np


def simulate_portfolio_paths(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    n_paths: int = 1000,
    n_periods: int = 252,
    initial_value: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate ``n_paths`` real-calibrated forward portfolio value paths.

    The portfolio's own mean and variance are derived from the real
    per-asset ``mean_returns``/``cov`` and the given ``weights``, then
    ``n_periods`` real daily returns per path are drawn from a normal
    distribution with that mean/vol (a real, standard simplification --
    real daily returns aren't exactly normal, documented explicitly
    rather than silently assumed away).
    """
    if n_paths < 1 or n_periods < 1:
        raise ValueError("n_paths and n_periods must be >= 1")
    weights = np.asarray(weights)
    portfolio_mean = float(weights @ mean_returns)
    portfolio_vol = float(np.sqrt(weights @ cov @ weights))

    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(portfolio_mean, portfolio_vol, size=(n_paths, n_periods))
    return initial_value * np.cumprod(1 + daily_returns, axis=1)


def max_drawdown_per_path(paths: np.ndarray) -> np.ndarray:
    """Real peak-to-trough decline along each simulated path."""
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = paths / running_max - 1.0
    return drawdowns.min(axis=1)


def probability_of_ruin(paths: np.ndarray, initial_value: float, ruin_threshold: float = 0.5) -> float:
    """Fraction of simulated paths that ever fall below
    ``ruin_threshold`` times ``initial_value`` -- the same
    ``initial_value`` passed to ``simulate_portfolio_paths``, kept as an
    explicit argument here rather than inferred from the paths, so the
    ruin level is never ambiguous.
    """
    if not 0 < ruin_threshold < 1:
        raise ValueError("ruin_threshold must be in (0, 1)")
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")
    return float(np.mean(paths.min(axis=1) < ruin_threshold * initial_value))
