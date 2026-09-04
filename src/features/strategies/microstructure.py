import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from ..base import BaseFeatureStrategy


class LiquidityMicrostructureStrategy(BaseFeatureStrategy):

    def construct(
        self, df: pd.DataFrame, eps: float = 1e-8
    ) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        h, l, c = df['high'], df['low'], df['close']
        v = df['volume']
        r = df['log_return']

        dollar_volume = c * v
        amihud_raw = r.abs() / dollar_volume.replace(0, np.nan)
        feat['liq_amihud_raw'] = amihud_raw
        feat['liq_amihud_z'] = (
            amihud_raw - amihud_raw.rolling(20).mean()
        ) / (amihud_raw.rolling(20).std() + eps)

        def calc_autocorr(x):
            return pd.Series(x).autocorr(lag=1) if len(x) > 2 else 0.0

        feat['liq_roll_measure_20'] = r.rolling(20).apply(
            calc_autocorr, raw=False
        )
        feat['liq_log_turnover'] = np.log(dollar_volume + eps)
        feat['liq_turnover_ratio_20'] = dollar_volume / (
            dollar_volume.shift(1).rolling(20).mean() + eps
        )
        return feat


class OrderFlowToxicityStrategy(BaseFeatureStrategy):

    def construct(
        self, df: pd.DataFrame, eps: float = 1e-8
    ) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        if all(col in df.columns for col in ['bid_size_1', 'ask_size_1']):
            order_imbalance = df['bid_size_1'] - df['ask_size_1']
        elif all(col in df.columns for col in ['bid_size', 'ask_size']):
            order_imbalance = df['bid_size'] - df['ask_size']
        else:
            c, h, l = df['close_adj'], df['high_adj'], df['low_adj']
            v = df['volume']
            buy_pressure = (c - l) / (h - l + eps)
            sell_pressure = (h - c) / (h - l + eps)
            order_imbalance = (buy_pressure - sell_pressure) * v

        feat['flow_imbalance_proxy'] = order_imbalance
        feat['flow_imbalance_zscore'] = (
            order_imbalance - order_imbalance.rolling(20).mean()
        ) / (order_imbalance.rolling(20).std() + eps)
        v_series = (
            df['volume']
            if 'volume' in df.columns
            else pd.Series(1, index=df.index)
        )
        feat['flow_vpin_10'] = (
            order_imbalance.abs().rolling(10).sum()
            / (v_series.rolling(10).sum() + eps)
        )
        return feat


class GMMRegimeStrategy(BaseFeatureStrategy):

    def __init__(self, window: int = 120, update_freq: int = 20):
        self.window = window
        self.update_freq = update_freq

    def construct(
        self, df: pd.DataFrame, eps: float = 1e-8
    ) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df['log_return'].fillna(0)
        vol = r.rolling(20).std(ddof=1).fillna(0)
        n = len(df)
        gmm_bull, gmm_bear = np.zeros(n), np.zeros(n)
        X = np.column_stack((r.values, vol.values))

        for i in range(self.window, n, self.update_freq):
            start_idx = max(0, i - self.window)
            X_train = np.nan_to_num(X[start_idx:i])
            pred_end = min(n, i + self.update_freq)
            X_test = np.nan_to_num(X[i:pred_end])
            try:
                if np.var(X_train[:, 0]) > 1e-8:
                    gmm = GaussianMixture(
                        n_components=2, random_state=42, n_init=1
                    )
                    gmm.fit(X_train)
                    probs = gmm.predict_proba(X_test)
                    bull_idx = np.argmax(gmm.means_[:, 0])
                    gmm_bull[i:pred_end] = probs[:, bull_idx]
                    gmm_bear[i:pred_end] = probs[:, 1 - bull_idx]
            except Exception:
                pass

        feat['gmm_prob_bull'] = gmm_bull
        feat['gmm_prob_bear'] = gmm_bear
        feat.iloc[: self.window, :] = np.nan
        return feat