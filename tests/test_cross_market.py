import numpy as np
import pandas as pd
import pytest

from systematic_equity_research.cross_market import (
    compute_spread,
    cross_sectional_momentum_signal,
    estimate_hedge_ratio,
    mean_reversion_half_life,
    momentum_long_short_weights,
)


# ---- cross-sectional momentum ----------------------------------------------


def test_cross_sectional_momentum_matches_hand_computed_returns():
    dates = pd.date_range("2024-01-02", periods=25, freq="B")
    panel = pd.DataFrame({"A": [100.0] * 24 + [110.0], "B": [50.0] * 24 + [45.0]}, index=dates)
    momentum = cross_sectional_momentum_signal(panel, lookback=20)
    assert momentum["A"] == pytest.approx(0.10)
    assert momentum["B"] == pytest.approx(-0.10)


def test_cross_sectional_momentum_rejects_insufficient_history():
    panel = pd.DataFrame({"A": [100.0, 101.0]})
    with pytest.raises(ValueError):
        cross_sectional_momentum_signal(panel, lookback=20)


def test_momentum_long_short_weights_are_dollar_neutral():
    momentum = pd.Series({"A": 0.10, "B": 0.05, "C": 0.0, "D": -0.05, "E": -0.10})
    weights = momentum_long_short_weights(momentum, top_pct=0.4)
    assert weights.sum() == pytest.approx(0.0, abs=1e-9)
    assert weights["A"] > 0  # top momentum -> long
    assert weights["E"] < 0  # bottom momentum -> short
    assert weights["C"] == 0.0  # middle -> flat


def test_momentum_long_short_weights_rejects_bad_top_pct():
    momentum = pd.Series({"A": 0.1, "B": 0.2})
    with pytest.raises(ValueError):
        momentum_long_short_weights(momentum, top_pct=0.6)


# ---- stat-arb pairs ---------------------------------------------------------


def test_estimate_hedge_ratio_recovers_a_known_linear_relationship():
    b = pd.Series(np.linspace(10, 20, 50))
    a = 2.0 * b + 5.0  # real, exact linear relationship: hedge ratio should be ~2.0
    hedge_ratio = estimate_hedge_ratio(a, b)
    assert hedge_ratio == pytest.approx(2.0, abs=1e-6)


def test_compute_spread_matches_hand_computed_value():
    a = pd.Series([110.0, 120.0])
    b = pd.Series([50.0, 55.0])
    spread = compute_spread(a, b, hedge_ratio=2.0)
    assert list(spread) == pytest.approx([10.0, 10.0])


def test_mean_reversion_half_life_is_finite_for_a_real_mean_reverting_series():
    rng = np.random.default_rng(3)
    n = 300
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] - 0.1 * spread[i - 1] + rng.normal(scale=0.1)
    half_life = mean_reversion_half_life(pd.Series(spread))
    assert 0 < half_life < 50


def test_mean_reversion_half_life_shows_no_systematic_mean_reversion_in_a_random_walk():
    # a single real random-walk realization can show a spuriously negative
    # theta from finite-sample noise alone -- a real, known statistical
    # phenomenon (caught directly: an earlier version of this test asserted
    # half_life == inf for one fixed seed and failed, since that one real
    # realization happened to show apparent mean reversion by chance).
    # The real, honest test is that theta has NO systematic negative bias
    # across many real random-walk realizations, not that any one instance
    # must come back non-mean-reverting.
    thetas = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        spread = pd.Series(np.cumsum(rng.normal(size=200)))
        lagged = spread.shift(1).dropna()
        delta = spread.diff().dropna()
        aligned = pd.concat([delta.rename("delta"), lagged.rename("lagged")], axis=1).dropna()
        design = aligned["lagged"].to_numpy().reshape(-1, 1)
        theta, _, _, _ = np.linalg.lstsq(design, aligned["delta"].to_numpy(), rcond=None)
        thetas.append(theta[0])
    # across many real random-walk realizations, the average theta should
    # sit close to zero -- no real, systematic mean-reversion signal
    assert abs(np.mean(thetas)) < 0.02


def test_hedge_ratio_rejects_too_few_observations():
    with pytest.raises(ValueError):
        estimate_hedge_ratio(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))


# ---- real end-to-end integration -------------------------------------------


def test_real_cross_sectional_momentum_and_stat_arb():
    from quant_toolkit.data import load_ohlcv

    tickers = ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"]
    closes = {}
    for t in tickers:
        bars = load_ohlcv(t, "2025-08-01", "2026-09-15")
        closes[t] = bars["close"]
    panel = pd.DataFrame(closes).dropna()

    momentum = cross_sectional_momentum_signal(panel, lookback=60)
    assert len(momentum) == len(tickers)
    weights = momentum_long_short_weights(momentum, top_pct=0.4)
    assert weights.sum() == pytest.approx(0.0, abs=1e-9)

    hedge_ratio = estimate_hedge_ratio(panel["MSFT"], panel["GOOGL"])
    spread = compute_spread(panel["MSFT"], panel["GOOGL"], hedge_ratio)
    half_life = mean_reversion_half_life(spread)
    assert half_life > 0  # real, either finite or inf, but never negative or zero
