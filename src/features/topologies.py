import numpy as np
import pandas as pd


class Layer2MicrostructureTopologies:
    """Khai phá Topology: Mở rộng không gian trạng thái với dữ liệu vi cấu trúc

    (L1/L2 Order Book và OHLCV).
    """

    @staticmethod
    def compute(df: pd.DataFrame, max_depth_levels: int = 10) -> pd.DataFrame:
        df = df.copy()

        df['log_return'] = np.log(df['close_adj'] / df['close_adj'].shift(1))
        df['overnight_return'] = np.log(
            df['open_adj'] / df['close_adj'].shift(1)
        )
        df['intraday_return'] = np.log(df['close_adj'] / df['open_adj'])
        df['hl_log_range'] = np.log(df['high_adj'] / df['low_adj'])

        available_levels = []
        for i in range(1, max_depth_levels + 1):
            if f'bid_price_{i}' in df.columns and f'ask_price_{i}' in df.columns:
                available_levels.append(i)

        if len(available_levels) > 1:
            df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2.0
            df['l2_spread'] = df['ask_price_1'] - df['bid_price_1']

            total_weighted_bid_size = np.zeros(len(df))
            total_weighted_ask_size = np.zeros(len(df))
            cross_weighted_bid_price = np.zeros(len(df))
            cross_weighted_ask_price = np.zeros(len(df))

            for i in available_levels:
                decay_weight = np.exp(-(i - 1) * 0.5)
                b_size_w = df[f'bid_size_{i}'] * decay_weight
                a_size_w = df[f'ask_size_{i}'] * decay_weight

                total_weighted_bid_size += b_size_w
                total_weighted_ask_size += a_size_w
                cross_weighted_bid_price += df[f'bid_price_{i}'] * b_size_w
                cross_weighted_ask_price += df[f'ask_price_{i}'] * a_size_w

            imbalance_denominator = (
                total_weighted_bid_size + total_weighted_ask_size + 1e-8
            )
            df['order_imbalance_depth'] = (
                total_weighted_bid_size - total_weighted_ask_size
            ) / imbalance_denominator
            df['micro_price_depth'] = (
                (
                    cross_weighted_bid_price
                    / (total_weighted_bid_size + 1e-8)
                )
                * total_weighted_ask_size
                + (
                    cross_weighted_ask_price
                    / (total_weighted_ask_size + 1e-8)
                )
                * total_weighted_bid_size
            ) / imbalance_denominator
            df['micro_mid_deviation'] = (
                df['micro_price_depth'] - df['mid_price']
            )

        elif all(
            col in df.columns
            for col in ['best_bid', 'best_ask', 'bid_size', 'ask_size']
        ):
            df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2.0
            imbalance_denominator = df['bid_size'] + df['ask_size'] + 1e-8
            df['micro_price'] = (
                df['best_bid'] * df['ask_size']
                + df['best_ask'] * df['bid_size']
            ) / imbalance_denominator
            df['l2_spread'] = df['best_ask'] - df['best_bid']
            df['micro_mid_deviation'] = df['micro_price'] - df['mid_price']
            df['order_imbalance_depth'] = (
                df['bid_size'] - df['ask_size']
            ) / imbalance_denominator

        return df