from __future__ import annotations

import numpy as np
import pandas as pd


class CandlestickPatternDetector:
    """Detects single, two, and three-candle candlestick patterns."""

    def detect(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        o = ohlc["Open"].astype(float)
        h = ohlc["High"].astype(float)
        l = ohlc["Low"].astype(float)
        c = ohlc["Close"].astype(float)

        eps = 1e-12
        body = (c - o).abs()
        rng = (h - l).abs() + eps
        upper = h - pd.concat([o, c], axis=1).max(axis=1)
        lower = pd.concat([o, c], axis=1).min(axis=1) - l
        body_ratio = body / rng

        po, ph, pl, pc = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
        p2o, p2h, p2l, p2c = o.shift(2), h.shift(2), l.shift(2), c.shift(2)

        bull = c > o
        bear = c < o
        pbull = pc > po
        pbear = pc < po
        p2bull = p2c > p2o
        p2bear = p2c < p2o

        small = body_ratio <= 0.30
        long_body = body_ratio >= 0.60

        doji = body_ratio <= 0.08
        dragonfly_doji = doji & (lower >= 0.6 * rng) & (upper <= 0.1 * rng)
        gravestone_doji = doji & (upper >= 0.6 * rng) & (lower <= 0.1 * rng)
        long_legged_doji = doji & (upper >= 0.35 * rng) & (lower >= 0.35 * rng)

        hammer = (lower >= 2.0 * body) & (upper <= body)
        inverted_hammer = (upper >= 2.0 * body) & (lower <= body)
        shooting_star = inverted_hammer & bear
        hanging_man = hammer & bear

        marubozu_bull = bull & (upper <= 0.05 * rng) & (lower <= 0.05 * rng) & long_body
        marubozu_bear = bear & (upper <= 0.05 * rng) & (lower <= 0.05 * rng) & long_body

        bullish_engulfing = pbear & bull & (o <= pc) & (c >= po)
        bearish_engulfing = pbull & bear & (o >= pc) & (c <= po)

        bullish_harami = pbear & bull & (o >= pc) & (c <= po)
        bearish_harami = pbull & bear & (o <= pc) & (c >= po)

        piercing_line = pbear & bull & (o < pl) & (c > (po + pc) / 2.0) & (c < po)
        dark_cloud_cover = pbull & bear & (o > ph) & (c < (po + pc) / 2.0) & (c > po)

        morning_star = (
            p2bear
            & small.shift(1, fill_value=False)
            & bull
            & (c > (p2o + p2c) / 2.0)
        )
        evening_star = (
            p2bull
            & small.shift(1, fill_value=False)
            & bear
            & (c < (p2o + p2c) / 2.0)
        )

        tweezer_bottom = pbear & bull & (np.abs(pl - l) / np.maximum(pl.abs(), 1e-12) <= 0.002)
        tweezer_top = pbull & bear & (np.abs(ph - h) / np.maximum(ph.abs(), 1e-12) <= 0.002)

        label = pd.Series("", index=ohlc.index, dtype="object")

        # Priority: multi-candle reversals first, then single-candle morphologies.
        label = label.mask(evening_star, "EveningStar")
        label = label.mask(morning_star & (label == ""), "MorningStar")
        label = label.mask(dark_cloud_cover & (label == ""), "DarkCloudCover")
        label = label.mask(piercing_line & (label == ""), "PiercingLine")
        label = label.mask(bearish_engulfing & (label == ""), "BearishEngulfing")
        label = label.mask(bullish_engulfing & (label == ""), "BullishEngulfing")
        label = label.mask(bearish_harami & (label == ""), "BearishHarami")
        label = label.mask(bullish_harami & (label == ""), "BullishHarami")
        label = label.mask(tweezer_top & (label == ""), "TweezerTop")
        label = label.mask(tweezer_bottom & (label == ""), "TweezerBottom")

        label = label.mask(shooting_star & (label == ""), "ShootingStar")
        label = label.mask(hanging_man & (label == ""), "HangingMan")
        label = label.mask(inverted_hammer & (label == ""), "InvertedHammer")
        label = label.mask(hammer & (label == ""), "Hammer")
        label = label.mask(marubozu_bear & (label == ""), "MarubozuBearish")
        label = label.mask(marubozu_bull & (label == ""), "MarubozuBullish")
        label = label.mask(long_legged_doji & (label == ""), "LongLeggedDoji")
        label = label.mask(gravestone_doji & (label == ""), "GravestoneDoji")
        label = label.mask(dragonfly_doji & (label == ""), "DragonflyDoji")
        label = label.mask(doji & (label == ""), "Doji")

        out = ohlc.copy()
        out["pattern"] = label
        return out[out["pattern"] != ""]
