import numpy as np
import pandas as pd
import pytest

from systematic_equity_research.execution import (
    PretradeCheck,
    amihud_illiquidity,
    corwin_schultz_spread,
    pretrade_checklist,
    roll_spread,
    twap_schedule,
    vwap_schedule,
)


# ---- Roll spread -------------------------------------------------------


def test_roll_spread_matches_hand_computed_bid_ask_bounce():
    # a clean, textbook bid-ask bounce: price alternates +0.5/-0.5 around
    # a fair value -- classic negative serial covariance setup
    prices = pd.Series([100.0, 100.5, 100.0, 100.5, 100.0, 100.5, 100.0, 100.5])
    spread = roll_spread(prices)
    changes = prices.diff().dropna()
    expected_cov = np.cov(changes.iloc[1:], changes.iloc[:-1])[0, 1]
    assert expected_cov < 0
    assert spread == pytest.approx(2 * np.sqrt(-expected_cov))


def test_roll_spread_is_nan_for_a_trending_series():
    # a monotonically rising series has positive serial covariance --
    # Roll's model is undefined here, by construction
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    assert np.isnan(roll_spread(prices))


# ---- Amihud illiquidity --------------------------------------------------


def test_amihud_illiquidity_matches_hand_computed_value():
    returns = pd.Series([0.01, -0.02, 0.005])
    dollar_volume = pd.Series([1_000_000.0, 2_000_000.0, 500_000.0])
    result = amihud_illiquidity(returns, dollar_volume, scale=1.0)
    expected = ((0.01 / 1_000_000) + (0.02 / 2_000_000) + (0.005 / 500_000)) / 3
    assert result == pytest.approx(expected)


def test_amihud_illiquidity_is_higher_for_thinner_real_volume():
    returns = pd.Series([0.02, 0.02])
    thick_volume = pd.Series([10_000_000.0, 10_000_000.0])
    thin_volume = pd.Series([100_000.0, 100_000.0])
    assert amihud_illiquidity(returns, thin_volume) > amihud_illiquidity(returns, thick_volume)


# ---- Corwin-Schultz -------------------------------------------------------


def test_corwin_schultz_spread_is_nonnegative_and_zero_for_constant_range():
    high = pd.Series([101.0] * 10)
    low = pd.Series([99.0] * 10)
    spread = corwin_schultz_spread(high, low)
    assert (spread.dropna() >= 0).all()


def test_corwin_schultz_spread_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        corwin_schultz_spread(pd.Series([101.0, 102.0]), pd.Series([99.0]))


def test_corwin_schultz_widens_with_a_real_wider_high_low_range():
    narrow_high, narrow_low = pd.Series([100.5] * 5), pd.Series([99.5] * 5)
    wide_high, wide_low = pd.Series([103.0] * 5), pd.Series([97.0] * 5)
    narrow_spread = corwin_schultz_spread(narrow_high, narrow_low).dropna().mean()
    wide_spread = corwin_schultz_spread(wide_high, wide_low).dropna().mean()
    assert wide_spread > narrow_spread


# ---- TWAP / VWAP schedules -------------------------------------------------


def test_twap_schedule_splits_evenly():
    schedule = twap_schedule(1000, 4)
    assert list(schedule) == [250.0, 250.0, 250.0, 250.0]


def test_twap_schedule_sums_to_total_shares():
    schedule = twap_schedule(777, 5)
    assert schedule.sum() == pytest.approx(777)


def test_vwap_schedule_weights_by_real_volume():
    volumes = pd.Series([100.0, 300.0, 600.0])  # total 1000
    schedule = vwap_schedule(1000, volumes)
    assert list(schedule) == pytest.approx([100.0, 300.0, 600.0])


def test_vwap_schedule_sums_to_total_shares():
    volumes = pd.Series([50.0, 25.0, 125.0])
    schedule = vwap_schedule(500, volumes)
    assert schedule.sum() == pytest.approx(500)


def test_twap_rejects_non_positive_total_shares():
    with pytest.raises(ValueError):
        twap_schedule(0, 5)


def test_vwap_rejects_zero_total_volume():
    with pytest.raises(ValueError):
        vwap_schedule(100, pd.Series([0.0, 0.0]))


# ---- Pre-trade checklist ---------------------------------------------------


def test_pretrade_checklist_passes_for_a_small_safe_order():
    result = pretrade_checklist(
        order_size=1000, real_avg_daily_volume=1_000_000, real_amihud_illiquidity=0.1,
    )
    assert isinstance(result, PretradeCheck)
    assert result.passed is True
    assert result.warnings == []


def test_pretrade_checklist_flags_high_participation_rate():
    result = pretrade_checklist(
        order_size=200_000, real_avg_daily_volume=1_000_000, real_amihud_illiquidity=0.1,
        max_participation_rate=0.10,
    )
    assert result.passed is False
    assert any("participation_rate" in w for w in result.warnings)


def test_pretrade_checklist_flags_high_illiquidity_even_at_small_size():
    result = pretrade_checklist(
        order_size=100, real_avg_daily_volume=1_000_000, real_amihud_illiquidity=5.0,
        amihud_warning_threshold=1.0,
    )
    assert result.passed is False
    assert any("illiquidity" in w for w in result.warnings)


def test_pretrade_checklist_rejects_non_positive_order_size():
    with pytest.raises(ValueError):
        pretrade_checklist(order_size=0, real_avg_daily_volume=1_000_000, real_amihud_illiquidity=0.1)
