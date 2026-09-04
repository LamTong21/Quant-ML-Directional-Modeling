import argparse
import os
import pandas as pd
from src.data.loaders import AssetAndMarketLoader
from src.data.integrity import Layer1DataIntegrity
from src.data.topologies import Layer2MicrostructureTopologies
from src.data.labeling import Layer0TripleBarrier


def main():
    parser = argparse.ArgumentParser(description="Ingest, audit and label stock data.")
    parser.add_argument("--symbol", type=str, default="DIG", help="Stock ticker symbol")
    parser.add_argument("--benchmark", type=str, default="VNINDEX", help="Benchmark index ticker")
    parser.add_argument("--start_date", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2026-06-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--source", type=str, default="KBS", help="Data source (KBS, VCI, TCBS)")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    print(f"=== 1. FETCHING DATA: {args.symbol} vs {args.benchmark} ===")
    loader = AssetAndMarketLoader(
        symbol=args.symbol,
        benchmark_symbol=args.benchmark,
        start_date=args.start_date,
        end_date=args.end_date,
        source=args.source,
    )
    df_raw = loader.fetch_data()
    df_raw.to_csv("data/raw/df_raw.csv", index=False)

    print("\n=== 2. DATA INTEGRITY & AUDITING ===")
    df_raw["time"] = pd.to_datetime(df_raw["time"])
    df_raw.set_index("time", inplace=True)
    df_raw.sort_index(inplace=True)

    df_clean, audit_stats = Layer1DataIntegrity.run_audit(df_raw)
    for k, v in audit_stats.items():
        print(f"  - {k}: {v}")

    print("\n=== 3. COMPUTING TOPOLOGIES & TRIPLE BARRIER LABELING ===")
    df_topo = Layer2MicrostructureTopologies.compute(df_clean)
    df_final = Layer0TripleBarrier.label(df_topo, h=5, pt=1.0, sl=1.0, vol_span=20)

    output_path = os.path.join(args.output_dir, "df_final.csv")
    df_final.to_csv(output_path)
    print(f"\n[+] Ingestion complete! Saved {len(df_final)} rows to {output_path}")


if __name__ == "__main__":
    main()