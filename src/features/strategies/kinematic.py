import numpy as np
import pandas as pd
from ..base import BaseFeatureStrategy


class BaselineStrategy(BaseFeatureStrategy):

    def __init__(self, dynamic_lags: list[int]):
        self.lags = dynamic_lags

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
        feat['geo_body_ratio'] = (c - o) / candle_range
        feat['geo_upper_shadow'] = (h - np.maximum(o, c)) / candle_range
        feat['geo_lower_shadow'] = (np.minimum(o, c) - l) / candle_range
        feat['overnight_gap'] = (o / c.shift(1)) - 1.0

        log_v = np.log(v + eps)
        for k in self.lags:
            if k >= 3:
                roll_vol_mean = v.shift(1).rolling(k).mean()
                feat[f'rvol_{k}'] = v / (roll_vol_mean + eps)
                mu_lv = log_v.shift(1).rolling(k).mean()
                std_lv = log_v.shift(1).rolling(k).std(ddof=1)
                feat[f'vol_shock_{k}'] = (log_v - mu_lv) / (std_lv + eps)

        if isinstance(df.index, pd.DatetimeIndex):
            dow = df.index.dayofweek
            month = df.index.month
            feat['sin_dow'] = np.sin(2 * np.pi * dow / 5.0)
            feat['cos_dow'] = np.cos(2 * np.pi * dow / 5.0)
            feat['sin_month'] = np.sin(2 * np.pi * month / 12.0)
            feat['cos_month'] = np.cos(2 * np.pi * month / 12.0)
        return feat


class TrendMomentumStrategy(BaseFeatureStrategy):

    def __init__(self, dynamic_lags: list[int]):
        self.lags = dynamic_lags

    def construct(
        self, df: pd.DataFrame, eps: float = 1e-8
    ) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df['close_adj']
        for k in self.lags:
            ret_k = np.log(p / p.shift(k))
            feat[f'ret_{k}'] = ret_k
            feat[f'tsmom_sign_{k}'] = np.sign(ret_k)
            if k >= 3:
                roll_max = p.shift(1).rolling(k).max()
                roll_min = p.shift(1).rolling(k).min()
                feat[f'breakout_ratio_{k}'] = (p - roll_max) / (
                    (roll_max - roll_min) + eps
                )
        ema_12 = p.ewm(span=12, adjust=False).mean()
        ema_26 = p.ewm(span=26, adjust=False).mean()
        feat['macd_dist'] = (ema_12 / ema_26) - 1.0
        feat['macd_slope_5'] = feat['macd_dist'] - feat['macd_dist'].shift(5)
        return feat


class OscillatorMeanReversionStrategy(BaseFeatureStrategy):

    def __init__(self, dynamic_lags: list[int]):
        self.lags = [k for k in dynamic_lags if k <= 20] or [5, 10, 20]

    def construct(
        self, df: pd.DataFrame, eps: float = 1e-8
    ) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df['close_adj']
        for k in self.lags:
            if k >= 3:
                roll_mean = p.shift(1).rolling(k).mean()
                roll_std = p.shift(1).rolling(k).std(ddof=1)
                feat[f'zscore_{k}'] = (p - roll_mean) / (roll_std + eps)
                upper_band = roll_mean + (2.0 * roll_std)
                lower_band = roll_mean - (2.0 * roll_std)
                feat[f'bollinger_pctB_{k}'] = (p - lower_band) / (
                    (upper_band - lower_band) + eps
                )
        delta = p.diff()
        gain = delta.where(delta > 0, 0.0).shift(1).rolling(14).mean()
        loss = -delta.where(delta < 0, 0.0).shift(1).rolling(14).mean()
        rs = gain / (loss + eps)
        feat['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))
        return feat


class FractionalMemoryStrategy(BaseFeatureStrategy):

    def __init__(
        self, optimal_d: float, threshold: float = 1e-4, max_window: int = 100
    ):
        self.d = optimal_d
        self.threshold = threshold
        self.max_window = max_window
        w = [1.0]
        for k in range(1, max_window):
            w_k = -w[-1] / k * (self.d - k + 1)
            if abs(w_k) < self.threshold:
                break
            w.append(w_k)
        self.weights_forward = np.array(w)
        self.actual_max_len = len(self.weights_forward)

    def construct(
        self, df: pd.DataFrame, eps: float = 1e-8
    ) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df['close_adj']

        def apply_frac_diff(x):
            x_rev = x[::-1]
            valid_len = min(len(x_rev), self.actual_max_len)
            return np.dot(self.weights_forward[:valid_len], x_rev[:valid_len])

        feat[f'frac_diff_d{self.d:.2f}'] = p.rolling(
            self.max_window, min_periods=10
        ).apply(apply_frac_diff, raw=True)
        return feat


class WaveletMultiResolutionStrategy(BaseFeatureStrategy):

    def construct(
        self, df: pd.DataFrame, eps: float = 1e-8
    ) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df['close_adj']
        p_s1 = p.shift(1)
        feat['wavelet_detail_d1'] = (p - p_s1) / np.sqrt(2.0)
        feat['wavelet_detail_d2'] = (
            (p + p_s1) - (p.shift(2) + p.shift(3))
        ) / 2.0
        feat['wavelet_approx_a3'] = p.rolling(8).mean()
        feat['wavelet_energy_ratio'] = (feat['wavelet_detail_d1'] ** 2) / (
            (feat['wavelet_detail_d2'] ** 2) + eps
        )
        return feat


class KinematicDynamicsStrategy(BaseFeatureStrategy):

    def construct(
        self, df: pd.DataFrame, eps: float = 1e-8
    ) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        p = df['close_adj']
        h, l = df['high_adj'], df['low_adj']
        v = df['volume']
        r = df['log_return'].fillna(0)
        feat['kinematic_velocity'] = r
        feat['kinematic_acceleration'] = r - r.shift(1)
        feat['kinematic_jerk'] = (
            feat['kinematic_acceleration']
            - feat['kinematic_acceleration'].shift(1)
        )
        roll_std = p.shift(1).rolling(20).std(ddof=1)
        bb_width = 4.0 * roll_std
        p_prev = p.shift(1)
        tr = pd.concat(
            [h - l, (h - p_prev).abs(), (l - p_prev).abs()], axis=1
        ).max(axis=1)
        atr = tr.shift(1).rolling(20).mean()
        feat['kinematic_squeeze_ratio'] = bb_width / (3.0 * atr + eps)
        clv = ((p - l) - (h - p)) / (h - l + eps)
        feat['kinematic_clv_vol'] = clv * v
        feat['kinematic_clv_vol_roll20'] = (
            feat['kinematic_clv_vol'].rolling(20).mean()
        )
        return feat