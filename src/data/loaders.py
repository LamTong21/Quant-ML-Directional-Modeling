import numpy as np
import pandas as pd
from vnstock import Quote


class AssetAndMarketLoader:
    def __init__(
        self,
        symbol: str,
        benchmark_symbol: str = 'VNINDEX',
        start_date: str = '2018-01-01',
        end_date: str = '2026-06-01',
        source: str = 'KBS',
    ):
        self.symbol = symbol
        self.benchmark_symbol = benchmark_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.source = source

    def fetch_data(self) -> pd.DataFrame:
        print(
            f'[*] Đang tải dữ liệu cổ phiếu cơ sở ({self.symbol}) và thị'
            f' trường ({self.benchmark_symbol})...'
        )
        quote_asset = Quote(symbol=self.symbol, source=self.source)
        quote_mkt = Quote(symbol=self.benchmark_symbol, source=self.source)

        df_asset = quote_asset.history(
            start=self.start_date, end=self.end_date, interval='1D'
        )
        if df_asset is None or df_asset.empty:
            raise ValueError(f'Không có dữ liệu Daily cho mã {self.symbol}')
        df_asset['time'] = pd.to_datetime(df_asset['time'])
        df_asset.sort_values(by='time', inplace=True)

        print(f'  -> Đang tải dữ liệu Benchmark: {self.benchmark_symbol}...')
        df_mkt = quote_mkt.history(
            start=self.start_date, end=self.end_date, interval='1D'
        )
        if df_mkt is None or df_mkt.empty:
            print('  [!] Thử nguồn VCI cho chỉ số VNINDEX...')
            quote_mkt_fallback = Quote(symbol=self.benchmark_symbol, source='VCI')
            df_mkt = quote_mkt_fallback.history(
                start=self.start_date, end=self.end_date, interval='1D'
            )

        df_mkt['time'] = pd.to_datetime(df_mkt['time'])
        df_mkt.sort_values(by='time', inplace=True)

        df_mkt = df_mkt[['time', 'close', 'volume', 'high', 'low']].rename(
            columns={
                'close': 'mkt_close',
                'volume': 'mkt_volume',
                'high': 'mkt_high',
                'low': 'mkt_low',
            }
        )
        df_mkt['mkt_return'] = np.log(
            df_mkt['mkt_close'] / df_mkt['mkt_close'].shift(1)
        )

        df_merged = pd.merge(df_asset, df_mkt, on='time', how='inner')
        df_merged.sort_values(by='time', inplace=True)

        micro_cols = [
            c for c in df_merged.columns if 'intraday' in c or 'vpin' in c
        ]
        if micro_cols:
            df_merged[micro_cols] = (
                df_merged[micro_cols].ffill().fillna(0)
            )

        df_merged.dropna(subset=['mkt_return'], inplace=True)
        print(
            f'  ✓ Đã đồng bộ thành công: {len(df_merged)} phiên chung giữa'
            f' {self.symbol} và {self.benchmark_symbol}.'
        )
        return df_merged