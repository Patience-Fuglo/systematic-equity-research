"""TWAP and VWAP optimal execution schedules, from scratch.

Both answer the same question -- how to split a real order across a
real trading session -- with different real weighting: TWAP splits
evenly across time; VWAP splits proportionally to each real bar's share
of the session's real total volume, matching activity to when the real
market itself is most liquid.
"""

from __future__ import annotations

import pandas as pd


def twap_schedule(total_shares: float, n_bars: int) -> pd.Series:
    """Equal-sized slices across ``n_bars`` -- a flat, time-based schedule."""
    if total_shares <= 0:
        raise ValueError("total_shares must be positive")
    if n_bars < 1:
        raise ValueError("n_bars must be >= 1")
    per_bar = total_shares / n_bars
    return pd.Series([per_bar] * n_bars, name="twap_shares")


def vwap_schedule(total_shares: float, real_bar_volumes: pd.Series) -> pd.Series:
    """Slices sized proportionally to each real bar's share of the
    session's real total volume -- more shares traded when the real
    market itself is most active, less when it's thin.
    """
    if total_shares <= 0:
        raise ValueError("total_shares must be positive")
    total_volume = real_bar_volumes.sum()
    if total_volume <= 0:
        raise ValueError("real_bar_volumes must sum to a positive total")
    weights = real_bar_volumes / total_volume
    return (weights * total_shares).rename("vwap_shares")
