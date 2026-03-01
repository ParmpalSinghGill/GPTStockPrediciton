from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    lookback_bars: int = 260
    level_lookback_bars: int = 900
    pivot_window: int = 5
    level_tolerance_pct: float = 0.01
    min_psy_touches: int = 2
    psychological_step: float = 0.0
    min_trend_touches: int = 3
    max_cross_bars: int = 2
    max_cross_streak: int = 2
    trend_max_age_bars: int = 220
    rectangle_min_bars: int = 20
    rectangle_max_bars: int = 220
    rectangle_min_touches: int = 2
    rectangle_max_outside_bars: int = 2
