import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif
from statsmodels.tsa.stattools import grangercausalitytests
from .transforms import Layer14LagTransformEngine


class Layer12RedundancyControl:

    @staticmethod
    def linear_vif_prune(
        X: pd.DataFrame, threshold: float = 0.85, max_vif: float = 5.0
    ) -> pd.DataFrame:
        X_curr = X.copy()
        corr_matrix = X_curr.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop_corr = [
            col for col in upper.columns if any(upper[col] > threshold)
        ]
        X_curr = X_curr.drop(columns=to_drop_corr)

        while X_curr.shape[1] > 1:
            vifs = []
            cols = X_curr.columns
            X_vals = X_curr.values
            for j in range(len(cols)):
                y_target = X_vals[:, j]
                X_other = np.delete(X_vals, j, axis=1)
                X_mat = np.column_stack([np.ones(len(X_other)), X_other])
                try:
                    beta, _, _, _ = np.linalg.lstsq(
                        X_mat, y_target, rcond=None
                    )
                    preds = X_mat @ beta
                    ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
                    ss_res = np.sum((y_target - preds) ** 2)
                    r2 = 1.0 - (ss_res / (ss_tot + 1e-9))
                    vif = 1.0 / (1.0 - np.clip(r2, 0, 0.9999))
                except np.linalg.LinAlgError:
                    vif = max_vif + 1.0
                vifs.append(vif)

            if max(vifs) > max_vif:
                X_curr = X_curr.drop(columns=[cols[np.argmax(vifs)]])
            else:
                break
        return X_curr

    @staticmethod
    def nonlinear_hrp_prune(
        X: pd.DataFrame, max_clusters: int = 15
    ) -> pd.DataFrame:
        if X.shape[1] <= max_clusters:
            return X
        corr = X.corr(method='spearman').clip(-1.0, 1.0).fillna(0)
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0, None))
        dist_sym = (dist + dist.T) / 2.0
        np.fill_diagonal(dist_sym.values, 0)

        condensed_dist = squareform(dist_sym, checks=False)
        link = linkage(condensed_dist, method='ward')
        clusters = fcluster(link, t=max_clusters, criterion='maxclust')

        selected_medoids = []
        for c_id in np.unique(clusters):
            cluster_cols = X.columns[clusters == c_id]
            if len(cluster_cols) == 1:
                selected_medoids.append(cluster_cols[0])
            else:
                sub_dist = dist_sym.loc[cluster_cols, cluster_cols]
                selected_medoids.append(sub_dist.sum(axis=1).idxmin())
        return X[selected_medoids]


class Layer13PredictiveDiagnostics:

    @staticmethod
    def causality_screen(
        X: pd.DataFrame,
        y: pd.Series,
        max_lag: int = 5,
        p_threshold: float = 0.05,
    ) -> tuple[list, dict]:
        valid_features, optimal_lags = [], {}
        df_test = pd.concat([y, X], axis=1).dropna()
        if df_test.empty:
            return valid_features, optimal_lags

        y_col = df_test.columns[0]
        y_vals = df_test[y_col].values

        for col in X.columns:
            test_data = df_test[[y_col, col]]
            best_lag, is_causal, best_p = 1, False, 1.0
            try:
                gc_res = grangercausalitytests(
                    test_data, maxlag=max_lag, verbose=False
                )
                for lag in range(1, max_lag + 1):
                    p_val = gc_res[lag][0]['ssr_ftest'][1]
                    if p_val < best_p:
                        best_p = p_val
                        best_lag = lag
                if best_p < p_threshold:
                    is_causal = True
            except Exception:
                pass

            if not is_causal:
                best_mi = 0.0
                for lag in range(1, max_lag + 1):
                    x_lagged = df_test[col].shift(lag).fillna(0).values
                    mi_score = mutual_info_classif(
                        x_lagged.reshape(-1, 1), y_vals, random_state=42
                    )[0]
                    if mi_score > best_mi:
                        best_mi = mi_score
                        best_lag = lag
                if best_mi > 0.01:
                    is_causal = True

            if is_causal:
                valid_features.append(col)
                optimal_lags[col] = best_lag

        if len(valid_features) < 3:
            corr_with_y = (
                df_test.drop(columns=[y_col])
                .corrwith(df_test[y_col], method='spearman')
                .abs()
            )
            for f in (
                corr_with_y.sort_values(ascending=False).head(5).index.tolist()
            ):
                if f not in valid_features:
                    valid_features.append(f)
                    optimal_lags[f] = 1
        return valid_features, optimal_lags


class Stage4SelectionRouter:

    def __init__(self, payload: dict):
        self.payload = payload
        self.max_clusters = payload.get('max_clusters', 14)
        self.mi_threshold = payload.get('mi_threshold', 0.01)
        self.min_features = 6
        self.max_features = 16
        self.directional_priority_keywords = [
            'mkt_relative',
            'mkt_rs',
            'mkt_rolling_beta',
            'mkt_divergence',
            'cmf',
            'obv',
            'vpt',
            'net_pressure',
            'vol_directional',
            'signed_expansion',
            'liq_amihud',
            'kurtosis_20',
        ]

    def execute(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> tuple[pd.DataFrame, dict, list]:
        aligned = pd.concat([X_train, y_train], axis=1).dropna()
        if aligned.empty:
            return X_train, {}, list(X_train.columns)

        X_mat = aligned.iloc[:, :-1]
        y_mat = aligned.iloc[:, -1]

        zero_cols = [c for c in X_mat.columns if X_mat[c].std() < 1e-8]
        if zero_cols:
            X_mat = X_mat.drop(columns=zero_cols)

        X_pruned = Layer12RedundancyControl.linear_vif_prune(
            X_mat, threshold=0.85, max_vif=6.0
        )
        X_hrp = Layer12RedundancyControl.nonlinear_hrp_prune(
            X_pruned, max_clusters=self.max_clusters
        )

        valid_causal_feats, optimal_lags = (
            Layer13PredictiveDiagnostics.causality_screen(X_hrp, y_mat)
        )
        X_causal_subset = X_hrp[
            [col for col in valid_causal_feats if col in X_hrp.columns]
        ]
        X_transformed = (
            Layer14LagTransformEngine.apply_volatility_scaled_lags(
                X_causal_subset, optimal_lags
            )
        )

        X_clean = X_transformed.dropna()
        y_clean = y_mat.loc[X_clean.index]
        valid_trans_cols = [
            c for c in X_clean.columns if X_clean[c].std() > 1e-8
        ]
        X_clean = X_clean[valid_trans_cols]

        mi_scores = mutual_info_classif(X_clean, y_clean, random_state=42)
        mi_series = pd.Series(mi_scores, index=X_clean.columns).sort_values(
            ascending=False
        )

        is_priority = mi_series.index.map(
            lambda name: any(
                k in name for k in self.directional_priority_keywords
            )
        )
        priority_features = mi_series[
            is_priority & (mi_series > (self.mi_threshold * 0.5))
        ].index.tolist()
        general_candidates = mi_series[
            mi_series > self.mi_threshold
        ].index.tolist()

        combined = []
        for feat in priority_features + general_candidates:
            if feat not in combined:
                combined.append(feat)

        if len(combined) < self.min_features:
            valid_features = mi_series.head(self.min_features).index.tolist()
        elif len(combined) > self.max_features:
            valid_features = combined[: self.max_features]
        else:
            valid_features = combined

        return X_transformed[valid_features], optimal_lags, valid_features