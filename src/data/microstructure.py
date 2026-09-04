import numpy as np
import pandas as pd


def extract_daily_microstructure(df_intraday: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp các chỉ số vi cấu trúc từ dữ liệu nội phiên (Intraday)

    thành các biến EOD Epsilon-Causal phục vụ Model.
    """
    if df_intraday.empty:
        return pd.DataFrame()

    df = df_intraday.copy()
    df['date'] = pd.to_datetime(df['time']).dt.date

    # Giả lập / Ước tính Order Flow Imbalance (OFI) từ nến nội phiên
    price_diff = df['close'].diff().fillna(0)
    buy_vol = np.where(price_diff >= 0, df['volume'], 0)
    sell_vol = np.where(price_diff < 0, df['volume'], 0)
    df['order_imbalance'] = buy_vol - sell_vol

    # Tổng hợp theo từng ngày
    daily_agg = (
        df.groupby('date')
        .agg(
            intraday_ofi_mean=('order_imbalance', 'mean'),
            intraday_ofi_std=('order_imbalance', 'std'),
            intraday_ofi_sum=('order_imbalance', 'sum'),
            intraday_vol_sum=('volume', 'sum'),
            intraday_tick_count=('volume', 'count'),
            intraday_high=('high', 'max'),
            intraday_low=('low', 'min'),
        )
        .reset_index()
    )

    # VPIN Proxy
    daily_agg['daily_vpin'] = daily_agg['intraday_ofi_sum'].abs() / (
        daily_agg['intraday_vol_sum'] + 1e-8
    )
    daily_agg['intraday_ofi_std'] = daily_agg['intraday_ofi_std'].fillna(0)

    daily_agg.rename(columns={'date': 'time'}, inplace=True)
    daily_agg['time'] = pd.to_datetime(daily_agg['time'])
    return daily_agg