"""Factor attribution: is the cross-sectional signal's real OOS return
stream genuine alpha, or a known Fama-French factor relabeled? Reuses
``alpha_validation_toolkit.metrics.fama_french_alpha`` directly -- the
same real OLS-based neutralization already built and tested there, not
rebuilt a second time.
"""

from __future__ import annotations

import pandas as pd
from quant_toolkit.metrics import OLSResult, fama_french_alpha


def aggregate_daily_portfolio_returns(oos_returns: pd.Series) -> pd.Series:
    """The cross-sectional signal's OOS returns are indexed by real
    (date, ticker) pairs -- Fama-French factors are daily. This collapses
    each real date's per-ticker returns into one equal-weighted real
    daily portfolio return, aligning the two.
    """
    return oos_returns.groupby(level="date").mean()


def run_factor_attribution(oos_returns: pd.Series, factors: pd.DataFrame) -> OLSResult:
    """Regress the real daily portfolio return stream on Mkt-RF/SMB/HML."""
    daily_returns = aggregate_daily_portfolio_returns(oos_returns)
    return fama_french_alpha(daily_returns, factors)
