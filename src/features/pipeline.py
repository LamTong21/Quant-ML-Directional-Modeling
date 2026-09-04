import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .diagnostics import Stage2SingleAssetDGPScanner
from .selection import Stage4SelectionRouter
from .strategies.flow_divergence import (
    DirectionalPressureStrategy,
    VolumePriceDivergenceStrategy,
)
from .strategies.kinematic import (
    BaselineStrategy,
    FractionalMemoryStrategy,
    KinematicDynamicsStrategy,
    OscillatorMeanReversionStrategy,
    TrendMomentumStrategy,
    WaveletMultiResolutionStrategy,
)
from .strategies.market_context import MarketContextStrategy
from .strategies.microstructure import (
    GMMRegimeStrategy,
    LiquidityMicrostructureStrategy,
    OrderFlowToxicityStrategy,
)
from .strategies.volatility import (
    SignedVolatilityDivergenceStrategy,
    VolatilityDynamicsStrategy,
    VolatilityJumpDiffusionStrategy,
)
from .transforms import Layer14LagTransformEngine


class Layer10FeatureRouter:

    def __init__(self, payload: dict):
        self.payload = payload
        self.registry = {}
        self._dispatch()

    def _dispatch(self):
        flags = self.payload.get('flags', {})
        lags = self.payload.get('dynamic_lags', [1, 3, 5, 10, 20])
        opt_d = self.payload.get('optimal_d', 1.0)

        self.registry['Baseline & Geometry'] = BaselineStrategy(
            dynamic_lags=lags
        )
        self.registry['Liquidity Microstructure'] = (
            LiquidityMicrostructureStrategy()
        )
        self.registry['Kinematics & Flow'] = KinematicDynamicsStrategy()
        self.registry['GMM State Clustering'] = GMMRegimeStrategy()
        self.registry['Wavelet Multi-Resolution'] = (
            WaveletMultiResolutionStrategy()
        )
        self.registry['Volume-Price Flow Divergence'] = (
            VolumePriceDivergenceStrategy(window=20)
        )
        self.registry['Directional Candle Pressure'] = (
            DirectionalPressureStrategy(window=14)
        )
        self.registry['Signed Volatility Bias'] = (
            SignedVolatilityDivergenceStrategy(window=20)
        )
        self.registry['Market Exogenous Context (VN-INDEX)'] = (
            MarketContextStrategy(window_short=20, window_long=60)
        )

        if flags.get('has_long_trend', True):
            self.registry['Trend & Momentum'] = TrendMomentumStrategy(
                dynamic_lags=lags
            )
        if flags.get('has_short_reversion', True):
            self.registry['Oscillators & Reversion'] = (
                OscillatorMeanReversionStrategy(dynamic_lags=lags)
            )
        if 0.0 < opt_d < 1.0:
            self.registry['Fractional Memory'] = FractionalMemoryStrategy(
                optimal_d=opt_d
            )
        if flags.get('has_vol_clustering', True):
            self.registry['Volatility Dynamics'] = VolatilityDynamicsStrategy(
                dynamic_lags=lags,
                has_asymmetric_vol=flags.get('has_asymmetric_vol', False),
            )
        if flags.get('has_vol_jumps', False):
            self.registry['Volatility Jump Diffusion'] = (
                VolatilityJumpDiffusionStrategy(window=20)
            )
        if flags.get('is_volume_significant', True):
            self.registry['Order Flow Toxicity (Proxy)'] = (
                OrderFlowToxicityStrategy()
            )

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_frames = [strat.construct(df) for strat in self.registry.values()]
        return pd.concat(feature_frames, axis=1)


class EconometricsFeaturePipeline(BaseEstimator, TransformerMixin):
    """Pipeline đầu cuối, kết hợp tự động Stage 2, 3 và 4."""

    def __init__(self, target_horizon: int = 5):
        self.target_horizon = target_horizon
        self.is_fitted = False
        self.optimal_lags_ = None
        self.final_features_list_ = None

    def _prepare_target(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        y = df['target_label'].copy().rename('target')
        valid_idx = y.dropna().index
        return df.loc[valid_idx], y.loc[valid_idx]

    def fit(self, X: pd.DataFrame, y=None):
        X_data, y_data = self._prepare_target(X)
        aligned = pd.concat([X_data, y_data], axis=1).dropna()
        X_train_aligned = aligned.drop(columns=['target'])
        y_train_aligned = aligned['target']

        self.routing_payload_ = Stage2SingleAssetDGPScanner.execute(
            X_train_aligned
        )
        self.router_ = Layer10FeatureRouter(self.routing_payload_)
        X_candidates = self.router_.execute(X_train_aligned)

        aligned_s3 = pd.concat(
            [X_candidates, y_train_aligned], axis=1
        ).dropna()
        X_c_clean = aligned_s3.drop(columns=['target'])
        y_c_clean = aligned_s3['target']

        self.s4_router_ = Stage4SelectionRouter(self.routing_payload_)
        _, self.optimal_lags_, self.final_features_list_ = (
            self.s4_router_.execute(X_c_clean, y_c_clean)
        )
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError('Pipeline chưa được fit. Hãy gọi fit() trước.')
        X_candidates = self.router_.execute(X)
        X_causal_subset = X_candidates[
            [c for c in self.optimal_lags_.keys() if c in X_candidates.columns]
        ]
        X_transformed = (
            Layer14LagTransformEngine.apply_volatility_scaled_lags(
                X_causal_subset, self.optimal_lags_
            )
        )
        available_cols = [
            c for c in self.final_features_list_ if c in X_transformed.columns
        ]
        return X_transformed[available_cols]

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)