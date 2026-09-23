"""Cross-sectional Ridge alpha signal, validated with
alpha-validation-toolkit's own purged walk-forward + PSR/DSR -- reused
directly, not rebuilt a second time.

The panel is indexed by (date, ticker) -- a real MultiIndex, so
``information_coefficient`` and friends align rows correctly even
though many rows share the same date. Fold splitting itself still has
to operate on real calendar dates alone (purging/embargo are time-based
concepts), so this splits on the panel's unique dates and then expands
each date-range back out to every real ticker's row on those dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from quant_toolkit.metrics import (
    deflated_sharpe_ratio,
    icir,
    information_coefficient,
    probabilistic_sharpe_ratio,
)
from quant_toolkit.validation import PurgedWalkForward
from sklearn.linear_model import Ridge


def fit_cross_sectional_ridge(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> Ridge:
    if len(X) < 10:
        raise ValueError(f"need at least 10 real rows to fit, got {len(X)}")
    return Ridge(alpha=alpha).fit(X, y)


def run_cross_sectional_validation(
    panel: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int,
    label_horizon: int,
    embargo_pct: float = 0.02,
) -> pd.DataFrame:
    """Purged walk-forward over the panel's real unique dates, expanded
    to every real ticker's row within each date range. Returns one row
    per completed fold (``fold, n_train, n_test, ic``), with real
    concatenated OOS predictions/targets/returns attached via
    ``.attrs`` -- same contract as market-making-sim/alt-data-alpha-signal's
    own validation modules.
    """
    unique_dates = pd.DatetimeIndex(sorted(panel.index.get_level_values("date").unique()))
    splitter = PurgedWalkForward(n_splits=n_splits, label_horizon=label_horizon, embargo_pct=embargo_pct)

    fold_rows = []
    oos_returns, oos_predictions, oos_targets = [], [], []

    for fold_num, (train_dates, test_dates) in enumerate(splitter.split(unique_dates), start=1):
        train_mask = panel.index.get_level_values("date").isin(train_dates)
        test_mask = panel.index.get_level_values("date").isin(test_dates)
        X_train, y_train = panel.loc[train_mask, feature_cols], panel.loc[train_mask, "forward_return"]
        X_test, y_test = panel.loc[test_mask, feature_cols], panel.loc[test_mask, "forward_return"]

        try:
            model = fit_cross_sectional_ridge(X_train, y_train)
        except ValueError:
            continue

        predictions = pd.Series(model.predict(X_test), index=X_test.index)
        fold_ic = information_coefficient(predictions, y_test)

        position = np.sign(predictions)
        strategy_return = position * y_test

        oos_returns.append(strategy_return)
        oos_predictions.append(predictions)
        oos_targets.append(y_test)
        fold_rows.append({"fold": fold_num, "n_train": len(X_train), "n_test": len(X_test), "ic": fold_ic})

    result = pd.DataFrame(fold_rows)
    result.attrs["oos_returns"] = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    result.attrs["oos_predictions"] = pd.concat(oos_predictions).sort_index() if oos_predictions else pd.Series(dtype=float)
    result.attrs["oos_targets"] = pd.concat(oos_targets).sort_index() if oos_targets else pd.Series(dtype=float)
    return result


def summarize_cross_sectional_validation(fold_results: pd.DataFrame, n_trials: int = 15) -> dict:
    oos_returns = fold_results.attrs.get("oos_returns", pd.Series(dtype=float))
    return {
        "n_folds_completed": len(fold_results),
        "mean_ic": fold_results["ic"].mean() if len(fold_results) else float("nan"),
        "icir": icir(fold_results["ic"]) if len(fold_results) else float("nan"),
        "n_oos_observations": len(oos_returns),
        "psr": probabilistic_sharpe_ratio(oos_returns, benchmark_sr=0.0) if len(oos_returns) else float("nan"),
        "dsr": deflated_sharpe_ratio(oos_returns, n_trials=n_trials) if len(oos_returns) else float("nan"),
    }
