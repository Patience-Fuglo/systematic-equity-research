# systematic-equity-research

From-scratch systematic equity research, rebuilt fresh and parallel to
(never merged into) an existing prior body of work — execution/
microstructure, cross-sectional alpha, factor attribution, portfolio
construction, and cross-market extensions, all real-data validated.

## Status

| Module | Status |
|---|---|
| Execution & microstructure | done |
| Cross-sectional alpha | not started |
| Factor attribution | not started |
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
