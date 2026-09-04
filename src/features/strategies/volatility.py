import numpy as np
import pandas as pd
from ..base import BaseFeatureStrategy


class VolatilityDynamicsStrategy(BaseFeatureStrategy):

    def __init__(self, dynamic_lags: list[int], has_asymmetric_vol: bool):
        self.lags = dynamic_lags
        self.has_asymmetric = has_asymmetric_vol

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
        r = df['log_return']
        feat['vol_parkinson'] = np.sqrt(
            (1.0 / (4.0 * np.log(2.0))) * (np.log(h / l) ** 2)
        )
        log_hl = np.log(h / l)
        log_co = np.log(c / o)
        gk_core = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
        feat['vol_garman_klass'] = np.sqrt(np.maximum(0.0, gk_core))
        rs_core = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
        feat['vol_rogers_satchell'] = np.sqrt(np.maximum(0.0, rs_core))

        window = 20
        var_o = np.log(o / c.shift(1)).rolling(window).var()
        var_c = np.log(c / o).rolling(window).var()
        var_rs = (
            pd.Series(rs_core, index=df.index).rolling(window).mean()
        )
        k_val = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
        feat['vol_yang_zhang'] = np.sqrt(
            np.maximum(0.0, var_o + k_val * var_c + (1.0 - k_val) * var_rs)
        )

        if self.has_asymmetric:
            downside_ret = r.where(r < 0, 0.0)
            feat['vol_downside_dev_20'] = downside_ret.rolling(20).std(ddof=1)
            feat['vol_upside_dev_20'] = (
                r.where(r > 0, 0.0).rolling(20).std(ddof=1)
            )
            feat['vol_semi_variance_ratio_raw'] = feat[
                'vol_downside_dev_20'
            ] / (feat['vol_upside_dev_20'] + eps)

        feat['skewness_20'] = r.rolling(20).skew()
        feat['kurtosis_20'] = r.rolling(20).kurt()
        return feat


class VolatilityJumpDiffusionStrategy(BaseFeatureStrategy):

    def __init__(self, window: int = 20):
        self.window = window

    def construct(
        self, df: pd.DataFrame, eps: float = 1e-8
    ) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df['log_return'].fillna(0)
        rv = (r**2).rolling(self.window).sum()
        abs_r = r.abs()
        bv = (np.pi / 2.0) * (abs_r * abs_r.shift(1)).rolling(
            self.window
        ).sum()
        feat['jump_diffusion_component'] = np.maximum(rv - bv, 0.0) / (rv + eps)
        feat['continuous_vol_ratio'] = bv / (rv + eps)
        feat['jump_signed_shock'] = feat['jump_diffusion_component'] * np.sign(
            r
        )
        return feat


class SignedVolatilityDivergenceStrategy(BaseFeatureStrategy):

    def __init__(self, window: int = 20):
        self.window = window

    def construct(
        self, df: pd.DataFrame, eps: float = 1e-8
    ) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        r = df['log_return'].fillna(0)
        downside_ret = r.where(r < 0, 0.0)
        upside_ret = r.where(r > 0, 0.0)
        downside_var = (downside_ret**2).rolling(self.window).mean()
        upside_var = (upside_ret**2).rolling(self.window).mean()

        feat['vol_directional_bias'] = (upside_var - downside_var) / (
            upside_var + downside_var + eps
        )
        feat['vol_directional_accel'] = feat['vol_directional_bias'] - feat[
            'vol_directional_bias'
        ].shift(3)
        signed_body = np.sign(df['close_adj'] - df['open_adj'])
        daily_range = np.log(df['high_adj'] / df['low_adj'])
        feat['signed_expansion_thrust'] = signed_body * daily_range
        return feat