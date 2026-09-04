import numpy as np
import pandas as pd


class Layer1DataIntegrity:
    @staticmethod
    def run_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        df = df.copy()
        initial_len = len(df)

        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

        max_oc = df[['open', 'close']].max(axis=1)
        min_oc = df[['open', 'close']].min(axis=1)
        df['high'] = np.maximum(df['high'], max_oc)
        df['low'] = np.minimum(df['low'], min_oc)
        df = df[(df['low'] > 0) & (df['volume'] >= 0)]

        if 'split_factor' not in df.columns:
            df['split_factor'] = 1.0
        if 'dividend' not in df.columns:
            df['dividend'] = 0.0

        daily_ret = (
            (df['close'] + df['dividend'])
            / (df['close'].shift(1) * df['split_factor'])
        ) - 1.0
        df['close_adj'] = df['close'].iloc[0] * (1.0 + daily_ret).cumprod()
        df['close_adj'] = df['close_adj'].fillna(df['close'])

        ratio = df['close_adj'] / df['close']
        for col in ['open', 'high', 'low']:
            df[f'{col}_adj'] = df[col] * ratio

        # Loại bỏ Zero-Variance columns
        dropped_zero_var = []
        for col in df.columns:
            if col not in ['split_factor', 'dividend', 'target_label']:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].std() < 1e-8 or df[col].nunique() <= 1:
                        dropped_zero_var.append(col)

        if dropped_zero_var:
            df.drop(columns=dropped_zero_var, inplace=True)
            print(
                f'  [Layer1 Alert] Đã loại bỏ các cột Zero-Variance rỗng:'
                f' {dropped_zero_var}'
            )

        return df, {
            'initial_bars': initial_len,
            'clean_bars': len(df),
            'dropped_flat_cols': dropped_zero_var,
        }