import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_feature_importance(summary_df: pd.DataFrame, top_n: int = 15, save_path: str = "feature_importance.png"):
    plt.figure(figsize=(12, 7))
    top_features = summary_df.head(top_n)
    sns.barplot(
        data=top_features,
        x="Mean_Importance",
        y="Feature",
        palette="mako",
    )
    plt.title(f"Top {top_n} Features Quan Trọng Nhất (Trung bình hóa qua các Folds)", fontsize=13, fontweight="bold")
    plt.xlabel("Tỷ trọng đóng góp trung bình (Relative Gain)", fontsize=11)
    plt.ylabel("Tên Đặc trưng (Feature)", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[+] Biểu đồ Feature Importance đã được lưu tại: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize feature importance report.")
    parser.add_argument("--csv_file", type=str, default="", help="Path to feature importance summary CSV")
    parser.add_argument("--output_img", type=str, default="feature_importance.png", help="Output path for plot")
    parser.add_argument("--top_n", type=int, default=15, help="Number of top features to plot")
    args = parser.parse_args()

    if args.csv_file:
        df = pd.read_csv(args.csv_file)
        plot_feature_importance(df, top_n=args.top_n, save_path=args.output_img)
    else:
        print("[!] Không có file input CSV, vui lòng truyền tham số --csv_file.")


if __name__ == "__main__":
    main()