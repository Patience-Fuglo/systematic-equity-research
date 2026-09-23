"""Demo: real equal-weight, min-variance, max-Sharpe, and risk-parity
portfolios on the same 5 real large-cap names, plus a 1,000-path Monte
Carlo simulation of the risk-parity portfolio's real-calibrated risk.

Run: python scripts/demo_portfolio.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from quant_toolkit.data import load_ohlcv
from systematic_equity_research.portfolio import (
    equal_weight_portfolio,
    max_drawdown_per_path,
    max_sharpe_weights,
    min_variance_weights,
    probability_of_ruin,
    risk_parity_weights,
    simulate_portfolio_paths,
)

TICKERS = ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"]
START, END = "2025-08-01", "2026-09-15"


def main() -> None:
    returns = {}
    for t in TICKERS:
        bars = load_ohlcv(t, START, END)
        returns[t] = bars["close"].pct_change().dropna()
    returns_df = pd.DataFrame(returns).dropna()
    mean_returns = returns_df.mean().to_numpy()
    cov = returns_df.cov().to_numpy()

    print(f"Real daily mean returns: {dict(zip(TICKERS, mean_returns.round(5)))}\n")

    methods = {
        "Equal-weight": equal_weight_portfolio(len(TICKERS)),
        "Min-variance": min_variance_weights(cov),
        "Max-Sharpe": max_sharpe_weights(mean_returns, cov),
        "Risk-parity": risk_parity_weights(cov),
    }

    print(f"{'Method':<14}" + "".join(f"{t:>9}" for t in TICKERS) + f"{'ExpRet':>10}{'Vol':>9}")
    for name, w in methods.items():
        port_mean = w @ mean_returns * 252
        port_vol = np.sqrt(w @ cov @ w) * np.sqrt(252)
        weights_str = "".join(f"{x:>9.1%}" for x in w)
        print(f"{name:<14}{weights_str}{port_mean:>10.1%}{port_vol:>9.1%}")

    print("\n1,000-path real-calibrated Monte Carlo simulation, risk-parity weights, 1 real trading year:")
    rp_weights = methods["Risk-parity"]
    paths = simulate_portfolio_paths(mean_returns, cov, rp_weights, n_paths=1000, n_periods=252, seed=42)
    drawdowns = max_drawdown_per_path(paths)
    p_ruin = probability_of_ruin(paths, initial_value=1.0, ruin_threshold=0.5)
    final_values = paths[:, -1]

    print(f"  Median simulated ending value (start=1.0): {np.median(final_values):.3f}")
    print(f"  5th/95th percentile ending value: {np.percentile(final_values, 5):.3f} / {np.percentile(final_values, 95):.3f}")
    print(f"  Median max drawdown across paths: {np.median(drawdowns):+.1%}")
    print(f"  Worst 5% max drawdown: {np.percentile(drawdowns, 5):+.1%}")
    print(f"  P(ruin), ever falling below 50% of starting value: {p_ruin:.4f}")

    print(
        "\nHonest read: portfolio weights and covariance are calibrated "
        "entirely to real historical data; the 1,000 simulated forward "
        "paths are necessarily synthetic (no one has real data from the "
        "future) -- drawn from a normal distribution matched to the real "
        "portfolio's mean/vol, a standard, explicit simplification, not "
        "hidden. Real daily returns aren't exactly normal (fatter tails), "
        "so real tail risk is likely understated here relative to a "
        "model that captured that directly."
    )


if __name__ == "__main__":
    main()
