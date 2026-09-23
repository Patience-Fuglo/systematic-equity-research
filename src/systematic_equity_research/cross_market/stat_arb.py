"""Statistical arbitrage pairs: hedge-ratio estimation and mean-reversion
half-life, both from scratch via ``numpy.linalg.lstsq`` (the same
no-statsmodels convention used in ``alpha-validation-toolkit``'s own
factor-neutralization module).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """OLS hedge ratio beta: price_a ~ alpha + beta * price_b. The real
    spread (price_a - beta * price_b) is what a pairs trade actually
    holds -- long one leg, short beta units of the other.
    """
    aligned = pd.concat([price_a.rename("a"), price_b.rename("b")], axis=1).dropna()
    if len(aligned) < 3:
        raise ValueError("need at least 3 real paired observations")
    design = np.column_stack([np.ones(len(aligned)), aligned["b"].to_numpy()])
    coeffs, _, _, _ = np.linalg.lstsq(design, aligned["a"].to_numpy(), rcond=None)
    return float(coeffs[1])


def compute_spread(price_a: pd.Series, price_b: pd.Series, hedge_ratio: float) -> pd.Series:
    aligned = pd.concat([price_a.rename("a"), price_b.rename("b")], axis=1).dropna()
    return (aligned["a"] - hedge_ratio * aligned["b"]).rename("spread")


def mean_reversion_half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck-style half-life: regress the real change in
    spread on the real lagged spread level (``d(spread)_t = theta *
    spread_{t-1} + noise``). A negative real theta means the spread
    mean-reverts; half-life = ln(2) / -theta (in the same real time
    units as ``spread``'s index spacing). Returns ``inf`` if theta >= 0
    (real, non-mean-reverting, no defined half-life).
    """
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    aligned = pd.concat([delta.rename("delta"), lagged.rename("lagged")], axis=1, join="inner").dropna()
    if len(aligned) < 3:
        raise ValueError("need at least 3 real paired observations")
    design = aligned["lagged"].to_numpy().reshape(-1, 1)
    theta, _, _, _ = np.linalg.lstsq(design, aligned["delta"].to_numpy(), rcond=None)
    theta = float(theta[0])
    if theta >= 0:
        return float("inf")
    return float(np.log(2) / -theta)
