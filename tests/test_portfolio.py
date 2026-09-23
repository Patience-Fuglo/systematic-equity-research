import numpy as np
import pytest

from systematic_equity_research.portfolio import (
    equal_weight_portfolio,
    max_drawdown_per_path,
    max_sharpe_weights,
    min_variance_weights,
    probability_of_ruin,
    risk_parity_weights,
    simulate_portfolio_paths,
)


# ---- construction -----------------------------------------------------


def test_equal_weight_portfolio_sums_to_one_and_is_uniform():
    w = equal_weight_portfolio(4)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w, 0.25)


def test_equal_weight_rejects_zero_assets():
    with pytest.raises(ValueError):
        equal_weight_portfolio(0)


def test_min_variance_weights_sum_to_one():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = min_variance_weights(cov)
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_favors_the_lower_variance_asset():
    # asset 0 has much lower variance, no covariance between them
    cov = np.array([[0.01, 0.0], [0.0, 0.25]])
    w = min_variance_weights(cov)
    assert w[0] > w[1]


def test_max_sharpe_weights_sum_to_one():
    mean_returns = np.array([0.10, 0.05])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = max_sharpe_weights(mean_returns, cov)
    assert w.sum() == pytest.approx(1.0)


def test_risk_parity_weights_sum_to_one_and_equalize_risk_contribution():
    cov = np.array([[0.01, 0.0], [0.0, 0.04]])
    w = risk_parity_weights(cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-4)
    port_vol = np.sqrt(w @ cov @ w)
    risk_contrib = w * (cov @ w) / port_vol
    # both assets should contribute roughly equally to total real risk
    assert risk_contrib[0] == pytest.approx(risk_contrib[1], abs=1e-3)


def test_risk_parity_gives_lower_vol_asset_more_weight():
    cov = np.array([[0.01, 0.0], [0.0, 0.04]])  # asset 0 is 2x less volatile
    w = risk_parity_weights(cov)
    assert w[0] > w[1]


# ---- Monte Carlo --------------------------------------------------------


def test_simulate_portfolio_paths_shape_and_start_value():
    mean_returns = np.array([0.0003, 0.0002])
    cov = np.array([[0.0002, 0.0], [0.0, 0.0003]])
    weights = np.array([0.5, 0.5])
    paths = simulate_portfolio_paths(mean_returns, cov, weights, n_paths=50, n_periods=30, initial_value=1.0, seed=0)
    assert paths.shape == (50, 30)
    assert (paths > 0).all()  # geometric compounding never goes negative


def test_simulate_portfolio_paths_rejects_non_positive_dims():
    mean_returns = np.array([0.0])
    cov = np.array([[0.0001]])
    with pytest.raises(ValueError):
        simulate_portfolio_paths(mean_returns, cov, np.array([1.0]), n_paths=0)


def test_max_drawdown_per_path_matches_hand_computed_case():
    # a path that goes 1.0 -> 1.2 -> 0.9 -> 1.1: drawdown = 0.9/1.2 - 1 = -0.25
    paths = np.array([[1.0, 1.2, 0.9, 1.1]])
    dd = max_drawdown_per_path(paths)
    assert dd[0] == pytest.approx(-0.25)


def test_probability_of_ruin_matches_hand_computed_case():
    # 2 paths: one dips to 0.4 (below 0.5 threshold), one stays above
    paths = np.array([[1.0, 0.4, 0.6], [1.0, 0.9, 1.1]])
    p_ruin = probability_of_ruin(paths, initial_value=1.0, ruin_threshold=0.5)
    assert p_ruin == pytest.approx(0.5)


def test_probability_of_ruin_rejects_invalid_threshold():
    paths = np.array([[1.0, 1.1]])
    with pytest.raises(ValueError):
        probability_of_ruin(paths, initial_value=1.0, ruin_threshold=1.5)


# ---- real end-to-end integration -------------------------------------------


def test_real_portfolio_comparison_across_all_four_methods():
    from quant_toolkit.data import load_ohlcv

    tickers = ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"]
    returns = {}
    for t in tickers:
        bars = load_ohlcv(t, "2025-08-01", "2026-09-15")
        returns[t] = bars["close"].pct_change().dropna()
    import pandas as pd

    returns_df = pd.DataFrame(returns).dropna()
    mean_returns = returns_df.mean().to_numpy()
    cov = returns_df.cov().to_numpy()

    ew = equal_weight_portfolio(len(tickers))
    mv = min_variance_weights(cov)
    ms = max_sharpe_weights(mean_returns, cov)
    rp = risk_parity_weights(cov)

    for w in (ew, mv, ms, rp):
        assert w.sum() == pytest.approx(1.0, abs=1e-3)

    paths = simulate_portfolio_paths(mean_returns, cov, rp, n_paths=1000, n_periods=252, seed=42)
    assert paths.shape == (1000, 252)
    p_ruin = probability_of_ruin(paths, initial_value=1.0, ruin_threshold=0.5)
    assert 0.0 <= p_ruin <= 1.0
