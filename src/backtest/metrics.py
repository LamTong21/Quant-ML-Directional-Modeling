import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_directional_metrics(df_oos: pd.DataFrame) -> dict:
    true_target = df_oos['true_target'].astype(int)
    pred_target = df_oos['model_raw_target'].astype(int)

    report_str = classification_report(
        true_target,
        pred_target,
        target_names=['Down (-1)', 'Neutral (0)', 'Up (1)'],
        digits=4,
    )
    cm = confusion_matrix(true_target, pred_target, labels=[-1, 0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=['True Down (-1)', 'True Neutral (0)', 'True Up (1)'],
        columns=['Pred Down (-1)', 'Pred Neutral (0)', 'Pred Up (1)'],
    )
    return {'report': report_str, 'confusion_matrix': cm_df}


def analyze_feature_importance(trained_models: list, top_n: int = 15) -> pd.DataFrame:
    if not trained_models:
        return pd.DataFrame()

    records = []
    for fold_idx, model in enumerate(trained_models):
        booster = getattr(model, 'get_booster', lambda: None)()
        if booster:
            score_gain = booster.get_score(importance_type='gain')
        elif hasattr(model, 'feature_importances_'):
            f_names = getattr(
                model, 'feature_names_in_', [f'f_{i}' for i in range(len(model.feature_importances_))]
            )
            score_gain = dict(zip(f_names, model.feature_importances_))
        else:
            continue

        total_gain = sum(score_gain.values()) if score_gain else 1.0
        for f_name, gain in score_gain.items():
            records.append({
                'Fold': f'Fold_{fold_idx + 1}',
                'Feature': f_name,
                'Gain': gain,
                'Relative_Importance': gain / total_gain,
            })

    df_imp = pd.DataFrame(records)
    if df_imp.empty:
        return pd.DataFrame()

    summary_imp = (
        df_imp.groupby('Feature')
        .agg(
            Mean_Importance=('Relative_Importance', 'mean'),
            Std_Importance=('Relative_Importance', 'std'),
            Fold_Count=('Fold', 'nunique'),
        )
        .reset_index()
    )
    summary_imp['Std_Importance'] = summary_imp['Std_Importance'].fillna(0.0)
    summary_imp = summary_imp.sort_values(
        by='Mean_Importance', ascending=False
    ).reset_index(drop=True)
    return summary_imp.head(top_n)