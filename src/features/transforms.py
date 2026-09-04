import pandas as pd


class Layer14LagTransformEngine:

    @staticmethod
    def apply_volatility_scaled_lags(
        X: pd.DataFrame, optimal_lags: dict, eps: float = 1e-8
    ) -> pd.DataFrame:
        X_transformed = pd.DataFrame(index=X.index)
        for col, lag in optimal_lags.items():
            if col not in X.columns:
                continue
            feature_series = X[col]
            feature_lagged = feature_series.shift(lag)
            roll_mean = feature_lagged.rolling(20).mean()
            roll_std = feature_lagged.rolling(20).std(ddof=1)

            X_transformed[f'{col}_zscaled_lag{lag}'] = (
                feature_lagged - roll_mean
            ) / (roll_std + eps)
            momentum = (feature_series - feature_lagged) / (
                feature_lagged.abs() + eps
            )
            X_transformed[f'{col}_momentum'] = momentum
            X_transformed[f'{col}_acceleration'] = (
                momentum - momentum.shift(1)
            )
        return X_transformed