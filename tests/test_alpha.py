import numpy as np
import pandas as pd
import pytest

from systematic_equity_research.alpha import (
    build_cross_sectional_panel,
    fit_cross_sectional_ridge,
    run_cross_sectional_validation,
    summarize_cross_sectional_validation,
)


# ---- fit_cross_sectional_ridge ---------------------------------------------


def _synthetic_xy(n=50):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(0.5 * X["f1"] + rng.normal(scale=0.1, size=n))
    return X, y


def test_fit_cross_sectional_ridge_fits_a_real_model():
    X, y = _synthetic_xy()
    model = fit_cross_sectional_ridge(X, y)
    assert hasattr(model, "predict")
    predictions = model.predict(X)
    assert len(predictions) == len(X)


def test_fit_cross_sectional_ridge_rejects_too_few_rows():
    X, y = _synthetic_xy(n=5)
    with pytest.raises(ValueError):
        fit_cross_sectional_ridge(X, y)


# ---- run_cross_sectional_validation (synthetic multi-ticker panel) --------


def _synthetic_panel(n_days=120, tickers=("A", "B", "C")):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rows = []
    for ticker in tickers:
        f1 = rng.normal(size=n_days)
        forward_return = 0.02 * f1 + rng.normal(scale=0.01, size=n_days)
        for i, date in enumerate(dates):
            rows.append({"date": date, "ticker": ticker, "f1": f1[i], "forward_return": forward_return[i]})
    panel = pd.DataFrame(rows).set_index(["date", "ticker"])
    return panel


def test_run_cross_sectional_validation_returns_one_row_per_completed_fold():
    panel = _synthetic_panel()
    results = run_cross_sectional_validation(panel, feature_cols=["f1"], n_splits=3, label_horizon=5)
    assert len(results) == 3
    assert list(results.columns) == ["fold", "n_train", "n_test", "ic"]


def test_run_cross_sectional_validation_attaches_oos_series_indexed_by_multiindex():
    panel = _synthetic_panel()
    results = run_cross_sectional_validation(panel, feature_cols=["f1"], n_splits=3, label_horizon=5)
    oos_returns = results.attrs["oos_returns"]
    assert isinstance(oos_returns.index, pd.MultiIndex)
    assert len(oos_returns) == results["n_test"].sum()


def test_summarize_cross_sectional_validation_computes_real_stats():
    panel = _synthetic_panel()
    results = run_cross_sectional_validation(panel, feature_cols=["f1"], n_splits=3, label_horizon=5)
    summary = summarize_cross_sectional_validation(results, n_trials=15)
    assert summary["n_folds_completed"] == 3
    assert 0.0 <= summary["psr"] <= 1.0
    assert 0.0 <= summary["dsr"] <= 1.0
    assert summary["dsr"] <= summary["psr"] + 1e-9


def test_summarize_handles_zero_completed_folds():
    empty = pd.DataFrame(columns=["fold", "n_train", "n_test", "ic"])
    empty.attrs["oos_returns"] = pd.Series(dtype=float)
    summary = summarize_cross_sectional_validation(empty)
    assert summary["n_folds_completed"] == 0
    assert np.isnan(summary["psr"])


# ---- real end-to-end integration -------------------------------------------


def test_real_cross_sectional_panel_and_validation():
    tickers = ["AAPL", "TSLA", "MSFT"]
    panel = build_cross_sectional_panel(tickers, "2025-08-01", "2026-09-15")
    assert len(panel) > 0
    assert set(panel.index.get_level_values("ticker")) <= set(tickers)

    feature_cols = [c for c in panel.columns if c != "forward_return"]
    results = run_cross_sectional_validation(panel, feature_cols=feature_cols, n_splits=3, label_horizon=5)
    assert len(results) >= 1  # real data -- at least one real fold should complete

    summary = summarize_cross_sectional_validation(results, n_trials=15)
    assert summary["n_oos_observations"] > 0
