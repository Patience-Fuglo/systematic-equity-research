"""Real cross-sectional feature panel: momentum, realized volatility,
and Amihud illiquidity (same real formula as
``systematic_equity_research.execution.amihud_illiquidity``, computed
here as a rolling per-day series rather than one summary value) across
multiple real tickers, indexed by (date, ticker).
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from quant_toolkit.data import load_ohlcv


def build_cross_sectional_panel(
    tickers: Sequence[str],
    start: str,
    end: str,
    momentum_windows: Sequence[int] = (5, 20),
    vol_window: int = 20,
    label_horizon: int = 5,
) -> pd.DataFrame:
    """One real row per (date, ticker), with a unique MultiIndex -- real
    momentum over each window in ``momentum_windows``, real trailing
    realized volatility, real rolling Amihud illiquidity, and the real
    forward return target. Rows with any NaN feature (real warm-up
    period) are dropped.
    """
    frames = []
    for ticker in tickers:
        bars = load_ohlcv(ticker, start, end)
        returns = bars["close"].pct_change()
        dollar_volume = bars["close"] * bars["volume"]

        feat = pd.DataFrame(index=bars.index)
        for w in momentum_windows:
            feat[f"momentum_{w}d"] = bars["close"].pct_change(w)
        feat[f"volatility_{vol_window}d"] = returns.rolling(vol_window).std()
        feat[f"amihud_{vol_window}d"] = (
            (returns.abs() / dollar_volume).rolling(vol_window).mean() * 1e6
        )
        feat["forward_return"] = bars["close"].shift(-label_horizon) / bars["close"] - 1.0
        feat["ticker"] = ticker
        frames.append(feat)

    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.set_index("ticker", append=True)
    return panel.dropna()
