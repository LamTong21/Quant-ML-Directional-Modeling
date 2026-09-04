import numpy as np
import pandas as pd


class DirectionalMTMSimulator:

    def __init__(
        self,
        initial_capital: float = 100000.0,
        base_fee_bps: float = 15.0,
        slippage_bps: float = 5.0,
        target_vol: float = 0.15,
    ):
        self.initial_capital = initial_capital
        self.friction = (base_fee_bps + slippage_bps) / 10000.0
        self.daily_vol_target = target_vol / np.sqrt(252)

    def run_simulation(
        self, df_oos: pd.DataFrame, df_mkt: pd.DataFrame
    ) -> pd.DataFrame:
        aligned = (
            df_oos[['model_raw_target']]
            .join(df_mkt[['close_adj', 'local_volatility']])
            .dropna()
        )

        aligned['daily_asset_return'] = (
            aligned['close_adj'].pct_change().fillna(0.0)
        )
        aligned['active_position'] = (
            aligned['model_raw_target'].shift(1).fillna(0.0)
        )

        sigma = aligned['local_volatility'].shift(1).replace(0, np.nan).bfill()
        raw_weight = self.daily_vol_target / sigma
        aligned['weight'] = (
            np.clip(raw_weight, 0.0, 1.0) * aligned['active_position']
        )

        aligned['turnover'] = aligned['weight'].diff().abs().fillna(0.0)
        aligned['costs'] = aligned['turnover'] * self.friction
        aligned['net_return'] = (
            aligned['weight'] * aligned['daily_asset_return']
        ) - aligned['costs']
        aligned['equity'] = self.initial_capital * (
            1.0 + aligned['net_return']
        ).cumprod()

        peak = aligned['equity'].cummax()
        aligned['drawdown'] = (aligned['equity'] - peak) / peak
        return aligned