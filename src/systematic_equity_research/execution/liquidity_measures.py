"""Real, from-scratch liquidity/illiquidity estimators, computed from
observable price and volume data alone -- none require a real limit
order book, all three are standard, published techniques for inferring
microstructure quantities from bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def roll_spread(prices: pd.Series) -> float:
    """Roll (1984): infers the effective bid-ask spread from the serial
    covariance of consecutive real price changes, under the assumption
    that trades bounce between the real bid and ask.

    ``spread = 2 * sqrt(-cov(dP_t, dP_{t-1}))``. Returns NaN if the real
    serial covariance is non-negative -- the model is only defined when
    it's negative (bid-ask bounce), a real, known limitation of Roll's
    estimator, not a bug here.
    """
    price_changes = prices.diff().dropna()
    if len(price_changes) < 2:
        return float("nan")
    cov = np.cov(price_changes.iloc[1:], price_changes.iloc[:-1])[0, 1]
    if cov >= 0:
        return float("nan")
    return float(2 * np.sqrt(-cov))


def amihud_illiquidity(returns: pd.Series, dollar_volume: pd.Series, scale: float = 1e6) -> float:
    """Amihud (2002): average absolute real return per real dollar
    traded -- a real, simple proxy for price impact. Higher = a given
    real dollar volume moves the price more, i.e. less liquid.
    """
    aligned = pd.concat([returns.abs(), dollar_volume], axis=1, join="inner").dropna()
    aligned.columns = ["abs_return", "dollar_volume"]
    aligned = aligned[aligned["dollar_volume"] > 0]
    if len(aligned) == 0:
        return float("nan")
    return float((aligned["abs_return"] / aligned["dollar_volume"]).mean() * scale)


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Corwin-Schultz (2012): infers the real effective spread from
    consecutive real daily high-low ranges alone, without needing trade
    or quote data -- real high/low ranges widen with both volatility and
    the real spread, and this estimator separates the two using
    overlapping 2-day windows.
    """
    if len(high) != len(low):
        raise ValueError("high and low must be the same length")
    k = (3 - 2 * np.sqrt(2))

    log_hl = np.log(high / low) ** 2
    beta = log_hl + log_hl.shift(1)

    two_day_high = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    two_day_low = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = np.log(two_day_high / two_day_low) ** 2

    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0).rename("corwin_schultz_spread")
