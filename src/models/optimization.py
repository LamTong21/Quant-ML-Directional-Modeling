import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss
import xgboost as xgb
from .validation import StandardPurgedCV


def train_optuna_xgboost(
    X_tr: pd.DataFrame,
    y_tr_mapped: np.ndarray,
    sample_weights: np.ndarray,
    n_inner_splits: int = 3,
    purge_gap: int = 5,
    n_trials: int = 12,
    timeout: int = 120,
) -> dict:
    inner_cv = StandardPurgedCV(n_splits=n_inner_splits, purge_gap=purge_gap)

    def objective(trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            'n_estimators': trial.suggest_int('n_estimators', 40, 120, step=20),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'learning_rate': trial.suggest_float(
                'learning_rate', 0.02, 0.07, log=True
            ),
            'subsample': trial.suggest_float('subsample', 0.65, 0.85),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree', 0.65, 0.85
            ),
            'reg_lambda': trial.suggest_float('reg_lambda', 3.0, 20.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 4, 12),
        }
        val_losses = []
        for in_tr_idx, in_val_idx in inner_cv.split(X_tr):
            w_inner_train = sample_weights[in_tr_idx]
            w_inner_val = sample_weights[in_val_idx]
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr.iloc[in_tr_idx],
                y_tr_mapped[in_tr_idx],
                sample_weight=w_inner_train,
            )
            probs = model.predict_proba(X_tr.iloc[in_val_idx])
            val_losses.append(
                log_loss(
                    y_tr_mapped[in_val_idx],
                    probs,
                    sample_weight=w_inner_val,
                    labels=[0, 1, 2],
                )
            )
        return np.mean(val_losses)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best_params = study.best_params
    best_params.update(
        {'objective': 'multi:softprob', 'num_class': 3, 'random_state': 42, 'n_jobs': -1}
    )
    return best_params