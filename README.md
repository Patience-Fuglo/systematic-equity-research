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
| Portfolio & risk | not started |
| Cross-market extensions | not started |
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
