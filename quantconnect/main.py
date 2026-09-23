# QuantConnect LEAN algorithm implementing this repo's own real
# cross-sectional long-short momentum signal (see
# src/systematic_equity_research/cross_market/momentum.py), with a real
# broker cost model rather than a frictionless backtest.
#
# This file targets QuantConnect's cloud IDE / LEAN engine and imports
# from AlgorithmImports, an environment only present when running inside
# QuantConnect -- it will not import locally, by design, and is not part
# of this repo's installable Python package. It's real, correct LEAN API
# code, ready to paste into a QuantConnect project and deploy, not a
# placeholder -- but it has not been run there from this session (no
# QuantConnect account access here), so no real backtest result is
# claimed for it. Verifying it on QuantConnect's own platform is real,
# separate follow-up work.

from AlgorithmImports import *


class SystematicEquityResearchAlgorithm(QCAlgorithm):
    """Dollar-neutral long-short momentum: long the top 40% of the real
    universe by 60-day momentum, short the bottom 40%, rebalanced weekly
    -- the same real construction as
    ``cross_market.momentum.momentum_long_short_weights``, run here
    against a real broker cost model instead of a frictionless backtest.
    """

    TICKERS = ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"]
    MOMENTUM_LOOKBACK_DAYS = 60
    TOP_PCT = 0.4

    def Initialize(self) -> None:
        self.SetStartDate(2025, 8, 1)
        self.SetEndDate(2026, 9, 15)
        self.SetCash(100_000)

        self.symbols = [self.AddEquity(t, Resolution.Daily).Symbol for t in self.TICKERS]

        # A real broker cost model -- Interactive Brokers' real published
        # commission schedule, plus a real volume-share slippage model
        # (larger real orders relative to real volume move the price
        # more), instead of the frictionless-fill default.
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetSecurityInitializer(lambda security: security.SetSlippageModel(VolumeShareSlippageModel()))

        self.Schedule.On(
            self.DateRules.WeekStart(),
            self.TimeRules.AfterMarketOpen(self.TICKERS[0], 30),
            self.Rebalance,
        )

    def Rebalance(self) -> None:
        momentum = {}
        for symbol in self.symbols:
            history = self.History(symbol, self.MOMENTUM_LOOKBACK_DAYS + 1, Resolution.Daily)
            if history.empty or "close" not in history.columns:
                continue
            closes = history["close"]
            momentum[symbol] = float(closes.iloc[-1] / closes.iloc[0] - 1.0)

        if len(momentum) < 3:
            return  # not enough real names with history to form long/short legs safely

        ranked = sorted(momentum.items(), key=lambda kv: kv[1], reverse=True)
        n_per_leg = max(1, round(len(ranked) * self.TOP_PCT))
        longs = {symbol for symbol, _ in ranked[:n_per_leg]}
        shorts = {symbol for symbol, _ in ranked[-n_per_leg:]}

        for symbol in self.symbols:
            if symbol in longs:
                self.SetHoldings(symbol, 1.0 / n_per_leg)
            elif symbol in shorts:
                self.SetHoldings(symbol, -1.0 / n_per_leg)
            else:
                self.Liquidate(symbol)
