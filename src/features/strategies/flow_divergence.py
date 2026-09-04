import numpy as np
import pandas as pd
from ..base import BaseFeatureStrategy


class VolumePriceDivergenceStrategy(BaseFeatureStrategy):

    def __init__(self, window: int = 20):
        self.window = window

    def construct(
        self, df: pd.DataFrame, eps: float = 1e-8
    ) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c = df['high_adj'], df['low_adj'], df['close_adj']
        v = df['volume']
        r = df['log_return'].fillna(0)

        clv = ((c - l) - (h - c)) / ((h - l) + eps)
        vol_money_flow = clv * v
        feat['cmf_20'] = vol_money_flow.rolling(self.window).sum() / (
            v.rolling(self.window).sum() + eps
        )
        feat['cmf_slope_5'] = feat['cmf_20'] - feat['cmf_20'].shift(5)

        obv = (np.sign(r) * v).cumsum()
        feat['obv_zscore_20'] = (
            obv - obv.rolling(self.window).mean()
        ) / (obv.rolling(self.window).std(ddof=1) + eps)
        feat['obv_slope_5'] = (obv - obv.shift(5)) / (
            v.rolling(self.window).mean() + eps
        )

        vpt = (r * v).cumsum()
        feat['vpt_zscore_20'] = (
            vpt - vpt.rolling(self.window).mean()
        ) / (vpt.rolling(self.window).std(ddof=1) + eps)
        norm_price_ret = c / c.shift(self.window) - 1.0
        feat['cmf_price_divergence'] = feat['cmf_20'] - norm_price_ret
        return feat


class DirectionalPressureStrategy(BaseFeatureStrategy):

    def __init__(self, window: int = 14):
        self.window = window

    def construct(
        self, df: pd.DataFrame, eps: float = 1e-8
    ) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c, o = (
            df['high_adj'],
            df['low_adj'],
            df['close_adj'],
            df['open_adj'],
        )
        v = df['volume']
        candle_range = (h - l) + eps
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l

        buy_strength = (c - l) / candle_range
        sell_strength = (h - c) / candle_range
        feat['net_pressure_ratio'] = buy_strength - sell_strength
        feat['net_pressure_volume'] = feat['net_pressure_ratio'] * np.log(
            v + 1.0
        )
        feat['net_pressure_vol_roll'] = (
            feat['net_pressure_volume'].rolling(self.window).mean()
        )
        feat['wick_absorption_direction'] = (
            lower_wick - upper_wick
        ) / candle_range
        feat['wick_absorption_thrust'] = feat[
            'wick_absorption_direction'
        ] * (v / (v.rolling(self.window).mean() + eps))

        body_ratio = (c - o) / candle_range
        feat['body_direction_momentum'] = body_ratio.rolling(5).mean()
        feat['body_direction_accel'] = (
            feat['body_direction_momentum']
            - feat['body_direction_momentum'].shift(2)
        )
        return feat