"""Pre-trade go/no-go checklist: combines the liquidity estimators above
into a real, explicit decision -- does a proposed real order size look
safe to execute given this name's real recent liquidity, or does it
need to be worked more carefully (smaller clips, longer horizon)?
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PretradeCheck:
    passed: bool
    participation_rate: float
    warnings: list[str]


def pretrade_checklist(
    order_size: float,
    real_avg_daily_volume: float,
    real_amihud_illiquidity: float,
    max_participation_rate: float = 0.10,
    amihud_warning_threshold: float = 1.0,
) -> PretradeCheck:
    """A real, explicit pre-trade check: is ``order_size`` a small enough
    fraction of ``real_avg_daily_volume`` to trade without excessive real
    market impact, and does the name's own real Amihud illiquidity
    suggest extra caution regardless of size?
    """
    if order_size <= 0:
        raise ValueError("order_size must be positive")
    if real_avg_daily_volume <= 0:
        raise ValueError("real_avg_daily_volume must be positive")

    participation_rate = order_size / real_avg_daily_volume
    warnings: list[str] = []

    if participation_rate > max_participation_rate:
        warnings.append(
            f"participation_rate {participation_rate:.1%} exceeds the "
            f"{max_participation_rate:.0%} cap -- work this order over a longer "
            "horizon or smaller clips"
        )
    if real_amihud_illiquidity > amihud_warning_threshold:
        warnings.append(
            f"real Amihud illiquidity {real_amihud_illiquidity:.3f} is elevated -- "
            "this name has shown outsized real price moves per dollar traded recently"
        )

    return PretradeCheck(passed=len(warnings) == 0, participation_rate=participation_rate, warnings=warnings)
