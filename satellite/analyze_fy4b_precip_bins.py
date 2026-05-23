from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INVALID_VALUES = {-999, 65531, 65532}
PHYSICAL_COLUMNS = ["COT", "CTH", "CER", "CTT"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="筛选 FY4B 降水样本并统计云参量分档占比。"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip.csv"),
        help="read_FY4B.py 生成的 test_allprecip.csv 路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认与输入 CSV 同目录。",
    )
    return parser


def load_and_filter(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    for column in PHYSICAL_COLUMNS + ["precipitation"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    valid_mask = df["precipitation"] > 0
    for column in PHYSICAL_COLUMNS:
        valid_mask &= df[column].notna()
        valid_mask &= ~df[column].isin(INVALID_VALUES)

    filtered = df.loc[valid_mask].copy()

    # FY4B 的 CTH 常见为米，这里转为 km 以匹配分档。
    filtered["CTH_km"] = filtered["CTH"] / 1000.0
    filtered["CTT_C"] = filtered["CTT"] - 273.15
    return filtered


def classify_series(
    series: pd.Series,
    conditions: list[pd.Series],
    labels: list[str],
) -> pd.Categorical:
    values = np.select(conditions, labels, default="Unclassified")
    category_order = labels + ["Unclassified"]
    return pd.Categorical(values, categories=category_order, ordered=True)


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["CTH_bin"] = classify_series(
        df["CTH_km"],
        [
            (df["CTH_km"] >= 0.0) & (df["CTH_km"] <= 2.5),
            (df["CTH_km"] > 2.5) & (df["CTH_km"] <= 5.0),
            (df["CTH_km"] > 5.0) & (df["CTH_km"] <= 7.5),
            (df["CTH_km"] > 7.5) & (df["CTH_km"] <= 10.0),
            df["CTH_km"] > 10.0,
        ],
        ["0-2.5 km", "2.6-5.0 km", "5.1-7.5 km", "7.6-10.0 km", ">10.0 km"],
    )

    df["CTT_bin"] = classify_series(
        df["CTT_C"],
        [
            (df["CTT_C"] >= 0.0) & (df["CTT_C"] <= 5.0),
            (df["CTT_C"] > 5.0) & (df["CTT_C"] <= 10.0),
            (df["CTT_C"] > 10.0) & (df["CTT_C"] <= 15.0),
            (df["CTT_C"] > 15.0) & (df["CTT_C"] <= 20.0),
            df["CTT_C"] > 20.0,
        ],
        ["0-5 C", "5-10 C", "10-15 C", "15-20 C", ">20 C"],
    )

    df["COT_bin"] = classify_series(
        df["COT"],
        [
            (df["COT"] >= 0.0) & (df["COT"] <= 10.0),
            (df["COT"] > 10.0) & (df["COT"] <= 20.0),
            (df["COT"] > 20.0) & (df["COT"] <= 30.0),
            (df["COT"] > 30.0) & (df["COT"] <= 40.0),
            df["COT"] > 40.0,
        ],
        ["0-10", "11-20", "21-30", "31-40", ">40"],
    )

    df["CER_bin"] = classify_series(
        df["CER"],
        [
            (df["CER"] >= 0.0) & (df["CER"] <= 10.0),
            (df["CER"] > 10.0) & (df["CER"] <= 20.0),
            (df["CER"] > 20.0) & (df["CER"] <= 30.0),
            (df["CER"] > 30.0) & (df["CER"] <= 40.0),
            df["CER"] > 40.0,
        ],
        ["0-10", "11-20", "21-30", "31-40", ">40"],
    )
    return df


def summarize_bins(df: pd.DataFrame) -> pd.DataFrame:
    summary_frames = []
    for variable, bin_col in [
        ("CTH", "CTH_bin"),
        ("CTT", "CTT_bin"),
        ("COT", "COT_bin"),
        ("CER", "CER_bin"),
    ]:
        counts = df[bin_col].value_counts(sort=False, dropna=False)
        counts = counts[counts.index != "Unclassified"]
        ratios = counts / counts.sum()
        summary = pd.DataFrame(
            {
                "variable": variable,
                "bin": counts.index.astype(str),
                "count": counts.values,
                "ratio": ratios.values,
            }
        )
        summary_frames.append(summary)

    return pd.concat(summary_frames, ignore_index=True)


def plot_bin_ratios(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    colors = ["#2E86AB", "#F18F01", "#C73E1D", "#6A994E", "#7B2CBF", "#577590"]

    for ax, variable in zip(axes.flat, ["CTH", "CTT", "COT", "CER"]):
        subset = summary_df[summary_df["variable"] == variable].copy()
        ax.bar(
            subset["bin"],
            subset["ratio"] * 100.0,
            color=colors[: len(subset)],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_title(f"{variable} bin ratio")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, max(5, subset["ratio"].max() * 120))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        for x, (_, row) in enumerate(subset.iterrows()):
            ax.text(
                x,
                row["ratio"] * 100.0 + 0.5,
                f"{row['ratio'] * 100.0:.1f}%\n(n={int(row['count'])})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("FY4B precipitating samples: bin proportion by variable", fontsize=16)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_csv = args.input_csv
    output_dir = args.output_dir if args.output_dir is not None else input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_df = load_and_filter(input_csv)
    if filtered_df.empty:
        raise ValueError("筛选后无有效样本，请检查输入数据或筛选条件。")

    filtered_with_bins = add_bins(filtered_df)
    summary_df = summarize_bins(filtered_with_bins)

    filtered_output = output_dir / "test_allprecip_filtered_positive_binned.csv"
    summary_output = output_dir / "test_allprecip_bin_summary.csv"
    figure_output = output_dir / "test_allprecip_bin_ratio.png"

    filtered_with_bins.to_csv(filtered_output, index=False)
    summary_df.to_csv(summary_output, index=False)
    plot_bin_ratios(summary_df, figure_output)

    print(f"输入文件: {input_csv}")
    print(f"筛选后样本数: {len(filtered_with_bins)}")
    print(f"筛选后数据已保存: {filtered_output}")
    print(f"分档统计已保存: {summary_output}")
    print(f"组合图已保存: {figure_output}")


if __name__ == "__main__":
    main()
