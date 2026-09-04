import math
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, adfuller, bds, kpss


class Layer4MemoryDependence:

    @staticmethod
    def compute_hurst_dfa(series: pd.Series) -> float:
        s = series.dropna().values
        y = np.cumsum(s - np.mean(s))
        n = len(y)
        if n < 40:
            return 0.5
        scales = np.floor(
            np.logspace(np.log10(10), np.log10(n // 4), num=15)
        ).astype(int)
        scales = np.unique(scales)
        fluctuations = []
        for s_len in scales:
            num_segments = n // s_len
            segment_fluct = []
            for i in range(num_segments):
                seg = y[i * s_len : (i + 1) * s_len]
                x = np.arange(s_len)
                poly = np.polyfit(x, seg, 1)
                trend = np.polyval(poly, x)
                segment_fluct.append(np.sqrt(np.mean((seg - trend) ** 2)))
            fluctuations.append(np.mean(segment_fluct))
        return float(np.polyfit(np.log(scales), np.log(fluctuations), 1)[0])

    @staticmethod
    def lo_mackinlay_vr(prices: pd.Series, k: int) -> dict:
        p = np.log(prices.dropna().values)
        t = len(p)
        if t <= k + 2:
            return {'vr': 1.0, 'z_stat': 0.0, 'p_value': 1.0}
        r1 = p[1:] - p[:-1]
        mu = (p[-1] - p[0]) / (t - 1)
        var_1 = np.sum((r1 - mu) ** 2) / (t - 2)
        rk = p[k:] - p[:-k]
        m = k * (t - k) * (1 - (k / (t - 1)))
        var_k = np.sum((rk - k * mu) ** 2) / m
        vr = var_k / var_1
        delta = np.zeros(k - 1)
        denom = (np.sum((r1 - mu) ** 2)) ** 2
        for j in range(1, k):
            num = np.sum(((r1[j:] - mu) ** 2) * ((r1[:-j] - mu) ** 2))
            delta[j - 1] = ((2.0 * (k - j) / k) ** 2) * (num / denom)
        phi_k = np.sum(delta)
        z_star = (vr - 1.0) / np.sqrt(phi_k) if phi_k > 0 else 0.0
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_star)))
        return {
            'vr': float(vr),
            'z_stat': float(z_star),
            'p_value': float(p_val),
        }

    @classmethod
    def multi_scale_memory(
        cls, prices: pd.Series, returns: pd.Series
    ) -> dict:
        r = returns.dropna()
        if len(r) < 50:
            return {
                'dynamic_lags': [1, 3, 5],
                'hurst': 0.5,
                'short_term_mean_reverting': False,
                'long_term_trending': False,
            }
        max_lag = min(30, len(r) // 3)
        acf_vals = acf(r, nlags=max_lag, fft=True)
        conf_interval = 1.96 / np.sqrt(len(r))
        sig_lags = [
            i
            for i, val in enumerate(acf_vals[1:], 1)
            if abs(val) > conf_interval
        ]
        if not sig_lags:
            sig_lags = [1, 3, 5]
        vr_short = cls.lo_mackinlay_vr(prices, k=3)
        vr_long = cls.lo_mackinlay_vr(prices, k=20)
        return {
            'dynamic_lags': sorted(list(set(sig_lags[:5] + [1, 5, 20]))),
            'hurst': cls.compute_hurst_dfa(returns),
            'short_term_mean_reverting': bool(
                vr_short['vr'] < 1.0 and vr_short['p_value'] < 0.05
            ),
            'long_term_trending': bool(
                vr_long['vr'] > 1.0 and vr_long['p_value'] < 0.05
            ),
        }


class Layer4bSpectralCycleDiagnostics:

    @staticmethod
    def run(prices: pd.Series) -> dict:
        p = prices.dropna().values
        n = len(p)
        if n < 40:
            return {'dominant_cycle_len': 10}
        detrended = p - np.polyval(np.polyfit(np.arange(n), p, 1), np.arange(n))
        fft_vals = np.abs(rfft(detrended))
        freqs = rfftfreq(n, d=1.0)
        fft_vals[0] = 0
        peak_idx = np.argmax(fft_vals)
        dominant_freq = freqs[peak_idx] if freqs[peak_idx] > 0 else 0.1
        dominant_cycle = int(np.clip(1.0 / dominant_freq, 3, 30))
        return {'dominant_cycle_len': dominant_cycle}


class Layer5FractionalIntegration:

    @staticmethod
    def get_weights(
        d: float, size: int, threshold: float = 1e-4
    ) -> np.ndarray:
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < threshold:
                break
            w.append(w_k)
        return np.array(w[::-1])

    @classmethod
    def fractionally_diff(
        cls, series: pd.Series, d: float, threshold: float = 1e-4
    ) -> pd.Series:
        weights = cls.get_weights(d, len(series), threshold)
        width = len(weights)
        vals = series.values
        res = [
            np.dot(weights, vals[i - width : i])
            for i in range(width, len(vals))
        ]
        return pd.Series(
            res, index=series.index[width:], name=f'frac_diff_{d:.2f}'
        )

    @classmethod
    def find_optimal_d(cls, series: pd.Series, d_step: float = 0.05) -> dict:
        s = series.dropna()
        if len(s) < 50:
            return {
                'optimal_d': 1.0,
                'price_stationary': False,
                'memory_retention_corr': 0.0,
            }
        adf_raw_p = adfuller(s, autolag='AIC')[1]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=InterpolationWarning)
            _, kpss_raw_p, _, _ = kpss(s, regression='c', nlags='auto')

        best_d, best_corr = 1.0, 0.0
        for d in np.arange(0.0, 1.05, d_step):
            fd = cls.fractionally_diff(s, d)
            if len(fd) < 50:
                continue
            adf_p = adfuller(fd.dropna(), autolag='AIC')[1]
            if adf_p < 0.05:
                best_d = float(round(d, 2))
                aligned = pd.concat([s, fd], axis=1).dropna()
                best_corr = float(
                    np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
                )
                break
        return {
            'price_stationary': bool(adf_raw_p < 0.05 and kpss_raw_p > 0.05),
            'optimal_d': best_d,
            'memory_retention_corr': best_corr,
        }


class Layer6VolatilityDynamics:

    @staticmethod
    def estimate(df: pd.DataFrame) -> dict:
        c = df['close_adj']
        ret = np.log(c / c.shift(1)).dropna()
        if len(ret) < 30:
            return {
                'has_arch_effect': False,
                'has_asymmetric_vol': False,
                'leverage_effect_corr': 0.0,
            }
        lm_stat, lm_p, _, _ = het_arch(ret, nlags=5)
        ret_lag = ret.shift(1).dropna()
        vol_proxy = (ret**2).iloc[1:]
        aligned_df = pd.concat([ret_lag, vol_proxy], axis=1).dropna()
        leverage_corr = aligned_df.iloc[:, 0].corr(aligned_df.iloc[:, 1])
        return {
            'arch_lm_stat': float(lm_stat),
            'arch_lm_pvalue': float(lm_p),
            'has_arch_effect': bool(lm_p < 0.05),
            'leverage_effect_corr': float(leverage_corr),
            'has_asymmetric_vol': bool(leverage_corr < -0.1),
        }


class Layer6bVolatilityJumpDiagnostics:

    @staticmethod
    def run(df: pd.DataFrame, window: int = 20) -> dict:
        r = df['log_return'].dropna()
        if len(r) < window * 2:
            return {'has_volatility_jumps': False, 'mean_jump_ratio': 0.0}
        rv = (r**2).rolling(window).sum()
        abs_r = r.abs()
        bv = (np.pi / 2.0) * (abs_r * abs_r.shift(1)).rolling(window).sum()
        jump_ratio = np.maximum(rv - bv, 0) / (rv + 1e-8)
        significant_jumps = (jump_ratio > 0.25).sum()
        return {
            'has_volatility_jumps': bool(significant_jumps > (len(r) * 0.05)),
            'mean_jump_ratio': float(jump_ratio.mean()),
        }


class Layer7bVolumeDiagnostics:

    @staticmethod
    def run(df: pd.DataFrame) -> dict:
        v = df['volume'].dropna()
        r_abs = (
            np.log(df['close_adj'] / df['close_adj'].shift(1)).abs().dropna()
        )
        aligned = pd.concat([v, r_abs], axis=1).dropna()
        if aligned.empty or len(aligned) < 30:
            return {'is_volume_significant': False, 'vol_ret_corr': 0.0}
        corr, p_val = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return {
            'vol_ret_corr': float(corr),
            'is_volume_significant': bool(p_val < 0.05 and corr > 0.1),
        }


class Layer7cMicrostructureDiagnostics:

    @staticmethod
    def run(df: pd.DataFrame) -> dict:
        r = df['log_return'].fillna(0)
        if (
            'micro_mid_deviation' not in df.columns
            or 'l2_spread' not in df.columns
        ):
            return {
                'has_l2_orderbook': False,
                'micro_deviation_sig': False,
                'l2_spread_stationary': False,
                'micro_dev_predictive_power': 0.0,
            }
        micro_dev = df['micro_mid_deviation'].fillna(0)
        l2_spread = df['l2_spread'].fillna(0)
        dev_corr, dev_pval = stats.spearmanr(
            micro_dev.shift(1).fillna(0), r
        )
        try:
            spread_adf_stat = adfuller(l2_spread, autolag='AIC')[1]
            is_spread_stat = bool(spread_adf_stat < 0.05)
        except Exception:
            is_spread_stat = False
        return {
            'has_l2_orderbook': True,
            'micro_deviation_sig': bool(dev_pval < 0.05 and abs(dev_corr) > 0.05),
            'micro_dev_predictive_power': float(dev_corr),
            'l2_spread_stationary': is_spread_stat,
        }


class Layer8NonlinearDependence:

    @staticmethod
    def run(returns: pd.Series) -> dict:
        r = returns.dropna()
        if len(r) < 50:
            return {
                'is_nonlinear_dependent': False,
                'mutual_information_lag1': 0.0,
            }
        resid = AutoReg(r.values, lags=1).fit().resid
        resid_std = (resid - np.nanmean(resid)) / (np.nanstd(resid) + 1e-8)
        bds_stat, p_val = bds(resid_std, max_dim=2, epsilon=None)
        return {
            'bds_stat_dim2': float(bds_stat),
            'bds_pvalue_dim2': float(p_val),
            'is_nonlinear_dependent': bool(p_val < 0.05),
        }


class Layer8bComplexityDiagnostics:

    @staticmethod
    def run(returns: pd.Series) -> dict:
        r = returns.dropna().values
        if len(r) < 30:
            return {'is_high_complexity': False, 'permutation_entropy': 1.0}
        m, tau = 3, 1
        n = len(r) - (m - 1) * tau
        patterns = np.array([r[i : i + m * tau : tau] for i in range(n)])
        ranks = np.argsort(patterns, axis=1)
        _, counts = np.unique(ranks, axis=0, return_counts=True)
        probs = counts / counts.sum()
        pe = -np.sum(probs * np.log2(probs + 1e-8)) / np.log2(
            math.factorial(m)
        )
        return {'permutation_entropy': float(pe), 'is_high_complexity': bool(pe > 0.85)}


class Stage2SingleAssetDGPScanner:

    @staticmethod
    def execute(df: pd.DataFrame) -> dict:
        returns = df['log_return'].dropna()
        prices = df['close_adj'].dropna()

        mem_stats = Layer4MemoryDependence.multi_scale_memory(prices, returns)
        cycle_stats = Layer4bSpectralCycleDiagnostics.run(prices)
        frac_stats = Layer5FractionalIntegration.find_optimal_d(prices)
        vol_stats = Layer6VolatilityDynamics.estimate(df)
        jump_stats = Layer6bVolatilityJumpDiagnostics.run(df)
        volu_stats = Layer7bVolumeDiagnostics.run(df)
        nonlin_stats = Layer8NonlinearDependence.run(returns)
        complex_stats = Layer8bComplexityDiagnostics.run(returns)
        micro_stats = Layer7cMicrostructureDiagnostics.run(df)

        return {
            'dynamic_lags': mem_stats['dynamic_lags'],
            'optimal_d': frac_stats['optimal_d'],
            'dominant_cycle': cycle_stats['dominant_cycle_len'],
            'flags': {
                'has_long_trend': mem_stats['long_term_trending'],
                'has_short_reversion': mem_stats['short_term_mean_reverting'],
                'has_vol_clustering': vol_stats['has_arch_effect'],
                'has_asymmetric_vol': vol_stats['has_asymmetric_vol'],
                'is_volume_significant': volu_stats['is_volume_significant'],
                'is_nonlinear': nonlin_stats['is_nonlinear_dependent'],
                'has_vol_jumps': jump_stats['has_volatility_jumps'],
                'is_high_complexity': complex_stats['is_high_complexity'],
                'has_l2_orderbook': micro_stats['has_l2_orderbook'],
                'micro_deviation_sig': micro_stats['micro_deviation_sig'],
            },
        }