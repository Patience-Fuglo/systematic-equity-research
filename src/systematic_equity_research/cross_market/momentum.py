"""Cross-sectional momentum: rank real names by trailing real return and
go long the real winners, short the real losers -- the classic
Jegadeesh-Titman style momentum construction, applied here to the same
real multi-ticker universe used throughout this repo.
"""

from __future__ import annotations

import pandas as pd


def cross_sectional_momentum_signal(price_panel: pd.DataFrame, lookback: int) -> pd.Series:
    """Real trailing ``lookback``-period return for each column
    (ticker) of ``price_panel`` (a wide DataFrame: real dates as rows,
    real tickers as columns, real close prices as values), evaluated at
    the panel's final real date.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if len(price_panel) <= lookback:
        raise ValueError(f"need more than {lookback} real rows, got {len(price_panel)}")
    momentum = price_panel.iloc[-1] / price_panel.iloc[-1 - lookback] - 1.0
    return momentum.rename(f"momentum_{lookback}d")


def momentum_long_short_weights(momentum_signal: pd.Series, top_pct: float = 0.3) -> pd.Series:
    """Equal-weight long the top ``top_pct`` real names by momentum,
    equal-weight short the bottom ``top_pct``, flat in between -- a real,
    standard long-short momentum construction. Long and short legs each
    sum to +1/-1, so the whole portfolio is real dollar-neutral.
    """
    if not 0 < top_pct <= 0.5:
        raise ValueError("top_pct must be in (0, 0.5]")
    n = len(momentum_signal)
    n_per_leg = max(1, int(round(n * top_pct)))
    ranked = momentum_signal.sort_values(ascending=False)

    weights = pd.Series(0.0, index=momentum_signal.index)
    long_names = ranked.index[:n_per_leg]
    short_names = ranked.index[-n_per_leg:]
    weights.loc[long_names] = 1.0 / n_per_leg
    weights.loc[short_names] = -1.0 / n_per_leg
    return weights
