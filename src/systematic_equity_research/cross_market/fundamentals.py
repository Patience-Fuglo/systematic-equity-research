"""Real fundamental snapshot features: current market cap, trailing P/E,
and dividend yield, via yfinance's own real company-info endpoint.

Honest scope: this is a real, current point-in-time snapshot, not a real
historical fundamentals panel (that requires paid data this build
doesn't have access to) -- documented explicitly rather than implied to
be more than it is.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import yfinance as yf


def real_fundamental_snapshot(tickers: Sequence[str]) -> pd.DataFrame:
    """Real, current-moment fundamental data for each ticker: market cap,
    trailing P/E, dividend yield. One row per real ticker; a field that
    yfinance doesn't have for a given real name comes back as NaN rather
    than a fabricated placeholder.
    """
    rows = []
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        rows.append({
            "ticker": ticker,
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
        })
    return pd.DataFrame(rows).set_index("ticker")
