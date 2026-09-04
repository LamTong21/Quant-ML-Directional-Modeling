import numpy as np
import pandas as pd
from ..base import BaseFeatureStrategy


class MarketContextStrategy(BaseFeatureStrategy):
    """Khai thác tín hiệu ngoại sinh từ chỉ số thị trường chung (VN-Index)."""

    def __init__(self, window_short: int = 20, window_long: int = 60):
        self.ws = window_short
        self.wl = window_long

    def construct(
        self, df: pd.DataFrame, eps: float = 1e-8
    ) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        if 'mkt_return' not in df.columns:
            return feat

        r_asset = df['log_return'].fillna(0)
        r_mkt = df['mkt_return'].fillna(0)

        feat['mkt_relative_return'] = r_asset - r_mkt
        feat['mkt_rs_cum_20'] = (
            r_asset.rolling(self.ws).sum() - r_mkt.rolling(self.ws).sum()
        )
        feat['mkt_rs_momentum'] = feat['mkt_rs_cum_20'] - feat[
            'mkt_rs_cum_20'
        ].shift(5)

        cov_60 = r_asset.rolling(self.wl).cov(r_mkt)
        var_mkt_60 = r_mkt.rolling(self.wl).var()
        rolling_beta = cov_60 / (var_mkt_60 + eps)
        feat['mkt_rolling_beta_60'] = rolling_beta.clip(lower=-1.0, upper=3.5)
        feat['mkt_beta_shock'] = (
            feat['mkt_rolling_beta_60']
            - feat['mkt_rolling_beta_60'].shift(5)
        )

        vol_asset_20 = r_asset.rolling(self.ws).std(ddof=1)
        vol_mkt_20 = r_mkt.rolling(self.ws).std(ddof=1)
        feat['mkt_vol_ratio_20'] = vol_asset_20 / (vol_mkt_20 + eps)
        feat['mkt_correlation_20'] = (
            r_asset.rolling(self.ws).corr(r_mkt).fillna(0)
        )
        feat['mkt_divergence_signed'] = (
            np.sign(r_asset) * np.sign(r_mkt) * (r_asset - r_mkt)
        )
        return feat