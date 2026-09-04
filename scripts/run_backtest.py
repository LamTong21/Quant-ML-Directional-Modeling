import argparse
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from src.features.pipeline import EconometricsFeaturePipeline
from src.models.validation import StandardPurgedCV
from src.models.optimization import train_optuna_xgboost
from src.models.thresholding import calculate_zero_anchored_thresholds
from src.backtest.execution import DirectionalMTMSimulator
from src.backtest.metrics import evaluate_directional_metrics, analyze_feature_importance


def main():
    parser = argparse.ArgumentParser(description="Run nested Purged CV backtest pipeline.")
    parser.add_argument("--data_path", type=str, default="data/processed/df_final.csv", help="Path to labeled data")
    parser.add_argument("--outer_splits", type=int, default=5, help="Number of outer CV splits")
    parser.add_argument("--inner_splits", type=int, default=3, help="Number of inner CV splits")
    parser.add_argument("--max_train_size", type=int, default=750, help="Maximum rolling training bars")
    parser.add_argument("--buffer_days", type=int, default=60, help="Feature lookback buffer for OOS evaluation")
    parser.add_argument("--horizon", type=int, default=5, help="Target horizon barrier gap")
    args = parser.parse_args()

    df_final = pd.read_csv(args.data_path)
    df_final["time"] = pd.to_datetime(df_final["time"])
    df_final.set_index("time", inplace=True)
    df_final.sort_index(inplace=True)

    outer_cv = StandardPurgedCV(
        n_splits=args.outer_splits, purge_gap=args.horizon, max_train_size=args.max_train_size
    )

    oos_predictions = []
    trained_models = []

    print("=" * 80)
    print("STARTING PURGED TIME-SERIES DIRECTIONAL MODELING")
    print("=" * 80)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(df_final)):
        print(f"\n>>> [FOLD {fold + 1}/{args.outer_splits}]")

        df_train_raw = df_final.iloc[train_idx].copy()
        df_test_raw = df_final.iloc[test_idx].copy()

        # Zero-variance check trước khi đưa vào pipeline
        zero_cols = [
            c for c in df_train_raw.columns
            if pd.api.types.is_numeric_dtype(df_train_raw[c]) and df_train_raw[c].std() < 1e-8
        ]
        if zero_cols:
            df_train_raw.drop(columns=zero_cols, inplace=True)
            df_test_raw.drop(columns=zero_cols, inplace=True)

        df_test_buffer = pd.concat([df_train_raw.iloc[-args.buffer_days:].copy(), df_test_raw])

        # 1. Feature pipeline fitting on train set only (chống rò rỉ tương lai)
        pipeline = EconometricsFeaturePipeline(target_horizon=args.horizon)
        pipeline.fit(df_train_raw)

        X_train = pipeline.transform(df_train_raw)
        _, y_train_raw = pipeline._prepare_target(df_train_raw)
        train_data = pd.concat([X_train, y_train_raw], axis=1).dropna()

        X_tr = train_data.drop(columns=["target"])
        y_tr = train_data["target"]
        y_tr_mapped = (y_tr + 1).astype(int).values  # Map: -1 -> 0, 0 -> 1, 1 -> 2

        # 2. Dynamic sample weighting
        fold_sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr_mapped)

        # 3. Inner CV: Hyperparameter Optimization
        best_params = train_optuna_xgboost(
            X_tr,
            y_tr_mapped,
            fold_sample_weights,
            n_inner_splits=args.inner_splits,
            purge_gap=args.horizon,
            n_trials=12,
            timeout=120,
        )

        # 4. Out-of-fold probability generation
        inner_cv = StandardPurgedCV(n_splits=args.inner_splits, purge_gap=args.horizon)
        oof_probs = np.zeros((len(X_tr), 3))
        oof_counts = np.zeros(len(X_tr))

        for in_tr_idx, in_val_idx in inner_cv.split(X_tr):
            w_in_tr = fold_sample_weights[in_tr_idx]
            fold_model = xgb.XGBClassifier(**best_params)
            fold_model.fit(X_tr.iloc[in_tr_idx], y_tr_mapped[in_tr_idx], sample_weight=w_in_tr)
            oof_probs[in_val_idx] += fold_model.predict_proba(X_tr.iloc[in_val_idx])
            oof_counts[in_val_idx] += 1

        valid_mask = oof_counts > 0
        oof_probs = oof_probs[valid_mask] / oof_counts[valid_mask, None]

        up_cut, down_cut = calculate_zero_anchored_thresholds(oof_probs, min_margin=0.04)
        print(f"   [+] Zero-Anchored Thresholds -> Up Cut (>): {up_cut:+.3f} | Down Cut (<): {down_cut:+.3f}")

        # 5. Huấn luyện mô hình toàn phần trên Fold Train
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_tr, y_tr_mapped, sample_weight=fold_sample_weights)
        trained_models.append(final_model)

        # 6. Suy luận trên tập Out-of-sample thông qua lookback buffer
        X_test_all = pipeline.transform(df_test_buffer)
        X_test_raw = X_test_all.loc[X_test_all.index >= df_test_raw.index.min()]
        _, y_test_raw = pipeline._prepare_target(df_test_raw)
        common_idx = X_test_raw.index.intersection(y_test_raw.index)

        X_te = X_test_raw.loc[common_idx].reindex(columns=X_tr.columns, fill_value=0.0)
        y_te = y_test_raw.loc[common_idx]

        test_probs = final_model.predict_proba(X_te)
        test_delta = test_probs[:, 2] - test_probs[:, 0]

        y_pred = np.zeros(len(X_te))
        y_pred[test_delta > up_cut] = 1.0
        y_pred[test_delta < down_cut] = -1.0

        res_df = pd.DataFrame(
            {
                "fold": fold + 1,
                "model_raw_target": y_pred,
                "prob_short": test_probs[:, 0],
                "prob_neutral": test_probs[:, 1],
                "prob_long": test_probs[:, 2],
                "prob_delta": test_delta,
                "true_target": y_te.values,
            },
            index=y_te.index,
        )
        oos_predictions.append(res_df)

    df_oos = pd.concat(oos_predictions).sort_index()
    print(f"\n[+] HOÀN TẤT OOS INFERENCE: {len(df_oos)} PHIÊN ĐÃ ĐƯỢC DỰ BÁO.")

    # Đánh giá Metrics phân loại
    metrics_result = evaluate_directional_metrics(df_oos)
    print("\n" + "=" * 65)
    print("BÁO CÁO HIỆU NĂNG PHÂN LOẠI HƯỚNG GIÁ OOS (DIRECTIONAL ACCURACY)")
    print("=" * 65)
    print(metrics_result["report"])
    print("\nConfusion Matrix:")
    print(metrics_result["confusion_matrix"])

    # Mô phỏng thực thi Mark-to-Market trễ T+1
    sim = DirectionalMTMSimulator()
    df_sim = sim.run_simulation(df_oos, df_final)
    total_ret = (df_sim["equity"].iloc[-1] / 100000.0) - 1.0
    ann_ret = (1.0 + total_ret) ** (252 / len(df_sim)) - 1.0
    sharpe = (df_sim["net_return"].mean() / (df_sim["net_return"].std() + 1e-9)) * np.sqrt(252)
    max_dd = df_sim["drawdown"].min()

    print("\n" + "=" * 65)
    print("HIỆU NĂNG MÔ PHỎNG MARK-TO-MARKET TRỄ T+1 (NET OF FRICTION)")
    print("=" * 65)
    print(f"Tổng số ngày quan sát : {len(df_sim)} phiên")
    print(f"Lợi nhuận lũy kế (Net) : {total_ret:.2%}")
    print(f"Lợi nhuận hàng năm     : {ann_ret:.2%}")
    print(f"Sharpe Ratio (Net)     : {sharpe:.2f}")
    print(f"Max Drawdown (MDD)     : {max_dd:.2%}")

    # Phân tích Feature Importance
    df_imp = analyze_feature_importance(trained_models, top_n=15)
    print("\nTop 15 Features có Gain đóng góp cao nhất:")
    print(df_imp.to_string(index=False))


if __name__ == "__main__":
    main()