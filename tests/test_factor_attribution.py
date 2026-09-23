import numpy as np
import pandas as pd
import pytest

from systematic_equity_research.alpha import aggregate_daily_portfolio_returns, run_factor_attribution


def test_aggregate_daily_portfolio_returns_averages_within_date():
    dates = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"])
    tickers = ["A", "B", "A", "B"]
    index = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
    oos_returns = pd.Series([0.02, 0.04, -0.01, 0.03], index=index)
    daily = aggregate_daily_portfolio_returns(oos_returns)
    assert daily.loc[pd.Timestamp("2024-01-02")] == pytest.approx(0.03)
    assert daily.loc[pd.Timestamp("2024-01-03")] == pytest.approx(0.01)


def test_run_factor_attribution_returns_real_ols_result():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2024-01-02", periods=60, freq="B")
    tickers = ["A", "B"]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    oos_returns = pd.Series(rng.normal(scale=0.01, size=len(index)), index=index)

    factors = pd.DataFrame(
        {
            "mkt_rf": rng.normal(scale=0.01, size=len(dates)),
            "smb": rng.normal(scale=0.005, size=len(dates)),
            "hml": rng.normal(scale=0.005, size=len(dates)),
            "rf": np.full(len(dates), 0.0001),
        },
        index=dates,
    )

    result = run_factor_attribution(oos_returns, factors)
    assert hasattr(result, "alpha")
    assert hasattr(result, "betas")
    assert set(result.betas.keys()) == {"mkt_rf", "smb", "hml"}
    assert result.n_obs > 0


# ---- real end-to-end integration -------------------------------------------


def test_real_factor_attribution_on_the_real_cross_sectional_signal():
    from quant_toolkit.data import load_fama_french_factors
    from systematic_equity_research.alpha import build_cross_sectional_panel, run_cross_sectional_validation

    tickers = ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"]
    panel = build_cross_sectional_panel(tickers, "2025-08-01", "2026-09-15", label_horizon=5)
    feature_cols = [c for c in panel.columns if c != "forward_return"]
    results = run_cross_sectional_validation(panel, feature_cols=feature_cols, n_splits=3, label_horizon=5)
    oos_returns = results.attrs["oos_returns"]

    factors = load_fama_french_factors("2025-08-01", "2026-09-15")
    result = run_factor_attribution(oos_returns, factors)

    assert result.n_obs > 0
    assert set(result.betas.keys()) == {"mkt_rf", "smb", "hml"}
