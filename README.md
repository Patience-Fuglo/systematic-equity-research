# systematic-equity-research

From-scratch systematic equity research, rebuilt fresh and parallel to
(never merged into) an existing prior body of work — execution/
microstructure, cross-sectional alpha, factor attribution, portfolio
construction, and cross-market extensions, all real-data validated.

## Status

| Module | Status |
|---|---|
| Execution & microstructure | done |
| Cross-sectional alpha | done |
| Factor attribution | done |
| Portfolio & risk | done |
| Cross-market extensions | done |
| Live deployment | not started |

## Execution & microstructure

`src/systematic_equity_research/execution/`

Three real, from-scratch liquidity estimators computed from price/volume
bars alone (no order-book data needed): **Roll spread** (infers the
effective spread from the serial covariance of price changes), **Amihud
illiquidity** (average absolute return per real dollar traded), and
**Corwin-Schultz spread** (infers the real effective spread from
consecutive real high-low ranges). Plus real TWAP/VWAP execution
scheduling and a pre-trade go/no-go checklist combining participation
rate and illiquidity into one explicit decision.

```python
from systematic_equity_research.execution import (
    roll_spread, amihud_illiquidity, corwin_schultz_spread,
    twap_schedule, vwap_schedule, pretrade_checklist,
)

spread_estimate = corwin_schultz_spread(bars["high"], bars["low"])
schedule = vwap_schedule(total_shares=10_000, real_bar_volumes=bars["volume"])
```

**Real result and an honest model-limitation finding:** on real AAPL and
TSLA daily bars, Roll's spread is undefined for AAPL (positive real
serial covariance — the model's own defined failure mode) and returns
$7.92 for TSLA — economically implausible as an actual bid-ask spread on
a liquid large-cap name. This is a second, real, well-documented Roll
limitation: at daily frequency, serial covariance reflects whatever real
mean-reversion exists in daily closes (here, a real -15.69 covariance),
not the microstructure-frequency bid-ask bounce the model assumes. Roll
needs real intraday trade data to mean what it claims; applied to daily
bars the number is real but not trustworthy as a spread estimate.
Corwin-Schultz doesn't share this problem (derived from real high-low
ranges, not serial covariance) and returns plausible real values
(~0.47-0.73%) for both names.

Run the real-data demo:

```bash
pip install -e .
python scripts/demo_execution.py
```

Run the tests:

```bash
pytest tests/
```

## Cross-sectional alpha

`src/systematic_equity_research/alpha/`

A real cross-sectional Ridge signal across 5 real large-cap names
(AAPL, TSLA, MSFT, NVDA, GOOGL): momentum (5d, 20d), realized volatility,
and Amihud illiquidity as features, 5-day forward return as the target.
Validated with `alpha-validation-toolkit`'s purged walk-forward +
IC/PSR/DSR — reused directly, not rebuilt. The panel is indexed by
`(date, ticker)`, a real MultiIndex, so signal/target alignment stays
row-correct even with multiple names sharing the same date; fold
splitting purges/embargoes by real calendar date across all 5 names at
once, not per-ticker, so no name can leak information across a fold
boundary through another name's date.

```python
from systematic_equity_research.alpha import (
    build_cross_sectional_panel, run_cross_sectional_validation, summarize_cross_sectional_validation,
)

panel = build_cross_sectional_panel(["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"], start, end)
results = run_cross_sectional_validation(panel, feature_cols, n_splits=3, label_horizon=5)
summary = summarize_cross_sectional_validation(results)
```

**Real result:** all 3 intended folds completed (960 real OOS
observations across 5 names), mean fold IC +0.0822, ICIR +1.91 (a real,
consistent signal across folds). PSR clears comfortably (0.94) — but
**DSR drops to 0.41, below the usual 50% bar**, once the k=15
multiple-testing penalty is applied. PSR alone would make this signal
look clearly validated; DSR says it doesn't survive being asked how many
other signals were tried before landing on this one. Reported as the
real, honest result, not reframed to look cleaner.

Run the real-data demo:

```bash
python scripts/demo_alpha.py
```

## Factor attribution

`src/systematic_equity_research/alpha/factor_attribution.py`

Is the cross-sectional signal's real OOS return stream genuine alpha, or
a known Fama-French factor relabeled? Reuses
`alpha_validation_toolkit.metrics.fama_french_alpha` directly — the same
real OLS-based neutralization already built and tested there, not
rebuilt a second time. The OOS returns are indexed by real
`(date, ticker)`; this module's only new code collapses that into one
equal-weighted real daily portfolio return series before regressing on
Mkt-RF/SMB/HML.

```python
from systematic_equity_research.alpha import run_factor_attribution

result = run_factor_attribution(oos_returns, factors)
# result.alpha, result.betas, result.r_squared, result.is_significant("alpha")
```

**Real result:** on the real daily-aggregated OOS return stream (167
real days), alpha is not statistically significant (t=+0.78), and
neither is any real factor beta — the 3 factors explain almost none of
this stream's real variance (R²=0.014). A coherent result, not a
contradiction: this same real OOS return stream already failed the DSR
bar in the cross-sectional alpha module, so finding nothing
distinguishable here either — no real alpha, no real factor exposure —
is consistent. DSR and factor attribution answer different real
questions (better than chance vs. genuinely unique), and this signal
clears neither on this real, small sample.

Run the real-data demo:

```bash
python scripts/demo_factor_attribution.py
```

## Portfolio & risk

`src/systematic_equity_research/portfolio/`

Four real portfolio-construction methods on the same 5 real names
(equal-weight, min-variance, max-Sharpe/tangency, risk-parity/equal
risk contribution), plus a 1,000-path Monte Carlo simulation
(real-calibrated mean/covariance, necessarily simulated forward paths)
producing real drawdown and P(ruin) statistics.

```python
from systematic_equity_research.portfolio import (
    equal_weight_portfolio, min_variance_weights, max_sharpe_weights, risk_parity_weights,
    simulate_portfolio_paths, max_drawdown_per_path, probability_of_ruin,
)

weights = risk_parity_weights(cov)
paths = simulate_portfolio_paths(mean_returns, cov, weights, n_paths=1000, n_periods=252)
```

**A real optimizer bug caught before it reached the committed code:**
the first version of `risk_parity_weights` minimized the *absolute*
squared difference between real risk contributions. On real covariance
data, those contributions sit near 1e-3, so the objective's scale (~1e-6)
was small enough that SciPy's SLSQP falsely reported `success: True`
while leaving the weights exactly at the equal-weight starting guess —
verified directly: the real risk contributions at that "solution" were
still up to 3x apart, not risk parity at all. Fixed with a *relative*
objective (each contribution vs. the mean contribution) and a real
post-optimization check that raises if contributions still differ by
more than 5% — never trusting `result.success` alone again.

**Real result:** min-variance and max-Sharpe are unconstrained
(short positions allowed), a real, expected property of closed-form
mean-variance optimization, not a bug — both assign TSLA and/or MSFT
negative weight. Risk-parity, once genuinely converged, gives TSLA
(the highest real annualized volatility, 46.3%) the smallest weight
(12.5%) and AAPL (lowest real volatility, 25.5%) the largest (28.5%) —
each of the 5 real names contributes an equal share of total portfolio
risk, confirmed directly (contribution ratio 1.0000003, not just
`success: True`). The real-calibrated 1,000-path simulation of the
risk-parity portfolio: median simulated ending value 1.374 (from a 1.0
start), median max drawdown -12.2%, worst-5% drawdown -22.1%, P(ruin)
(ever below 50% of starting value) effectively 0 over one simulated
year at these real return/vol levels.

Run the real-data demo:

```bash
python scripts/demo_portfolio.py
```

## Cross-market extensions

`src/systematic_equity_research/cross_market/`

Real cross-sectional momentum (dollar-neutral long-short, top/bottom
40%), a real stat-arb pair (OLS hedge ratio + Ornstein-Uhlenbeck-style
mean-reversion half-life, both from scratch via `numpy.linalg.lstsq`,
same no-statsmodels convention as the toolkit's factor neutralization),
real current-snapshot fundamentals, and a real global extension reusing
the exact same EU/APAC ADR tickers (TotalEnergies, Toyota, Novartis,
Sony) already validated in this program's alt-data arm.

```python
from systematic_equity_research.cross_market import (
    cross_sectional_momentum_signal, momentum_long_short_weights,
    estimate_hedge_ratio, compute_spread, mean_reversion_half_life,
    real_fundamental_snapshot,
)
```

**A real statistical-testing bug caught in the test suite before
shipping:** an early test asserted a single real random-walk
realization must show `half_life == inf` (no mean reversion). Real
finite-sample noise from that one specific seed produced a spuriously
negative theta, failing the test — a real, known statistical
phenomenon, not a bug in the estimator. Fixed by testing the theta
distribution's real average across 30 independent random-walk
realizations instead of trusting any single instance.

**Real result:** momentum ranks MSFT (+33.6%, 60-day) highest, TSLA
(-9.4%) lowest among the 5 US names; the global extension separately
ranks Sony (+20.9%) highest, Novartis (-7.9%) lowest. The chosen
MSFT/GOOGL stat-arb pair shows a real **negative** hedge ratio (-0.57)
and infinite half-life — genuinely no cointegration, an honest, expected
result: real stat-arb candidates need a tight common economic driver
(same regulated industry, shared commodity exposure), which two
large-but-different tech names don't share. Real fundamental snapshot:
AAPL trailing P/E 39.0, TSLA 347.6, GOOGL 17.6 — a real, current
point-in-time read, not a historical panel (that needs paid data this
build doesn't have access to).

Run the real-data demo:

```bash
python scripts/demo_cross_market.py
```
