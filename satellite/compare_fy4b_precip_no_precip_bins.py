from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_NO_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_no_precip_date_hour_station_cloud_stats.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou")
VARIABLES = ["CTH", "CTT", "COT", "CER"]
BIN_LABELS = {
    "CTH": ["0-1.5 km", "1.5-2.5 km", "2.5-3.5 km", "3.5-4.5 km", ">4.5 km"],
    "CTT": ["0-5 C", "5-10 C", "10-15 C", "15-20 C", ">20 C"],
    "COT": ["0-10", "11-20", "21-30", "31-40", ">40"],
    "CER": ["0-10", "11-20", "21-30", "31-40", ">40"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare precip and no-precip FY4B cloud-parameter bin ratios.",
    )
    parser.add_argument(
        "--precip-csv",
        type=Path,
        default=DEFAULT_PRECIP_CSV,
        help="Date-hour-station cloud stats CSV for precipitation samples.",
    )
    parser.add_argument(
        "--no-precip-csv",
        type=Path,
        default=DEFAULT_NO_PRECIP_CSV,
        help="Date-hour-station cloud stats CSV for no-precipitation samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary CSV and figure.",
    )
    parser.add_argument(
        "--min-file-count",
        type=int,
        default=3,
        help="Minimum file_count required for a row to be included.",
    )
    return parser


def load_samples(csv_path: Path, sample_type: str, stat: str, min_file_count: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_columns = {"file_count", *[f"{variable}_{stat}" for variable in VARIABLES]}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

    df["file_count"] = pd.to_numeric(df["file_count"], errors="coerce")
    df = df.loc[df["file_count"] >= min_file_count].copy()
    df["sample_type"] = sample_type

    for variable in VARIABLES:
        df[variable] = pd.to_numeric(df[f"{variable}_{stat}"], errors="coerce")

    df["CTH_km"] = df["CTH"] / 1000.0
    df["CTT_C"] = df["CTT"] - 273.15
    return df


def classify_series(series: pd.Series, conditions: list[pd.Series], labels: list[str]) -> pd.Categorical:
    values = np.select(conditions, labels, default="Unclassified")
    return pd.Categorical(values, categories=labels + ["Unclassified"], ordered=True)


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["CTH_bin"] = classify_series(
        df["CTH_km"],
        [
            (df["CTH_km"] >= 0.0) & (df["CTH_km"] <= 1.5),
            (df["CTH_km"] > 1.5) & (df["CTH_km"] <= 2.5),
            (df["CTH_km"] > 2.5) & (df["CTH_km"] <= 3.5),
            (df["CTH_km"] > 3.5) & (df["CTH_km"] <= 4.5),
            df["CTH_km"] > 4.5,
        ],
        BIN_LABELS["CTH"],
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
        BIN_LABELS["CTT"],
    )

    for variable in ["COT", "CER"]:
        df[f"{variable}_bin"] = classify_series(
            df[variable],
            [
                (df[variable] >= 0.0) & (df[variable] <= 10.0),
                (df[variable] > 10.0) & (df[variable] <= 20.0),
                (df[variable] > 20.0) & (df[variable] <= 30.0),
                (df[variable] > 30.0) & (df[variable] <= 40.0),
                df[variable] > 40.0,
            ],
            BIN_LABELS[variable],
        )

    return df


def summarize_bins(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for variable in VARIABLES:
        bin_col = f"{variable}_bin"
        valid_df = df.loc[df[bin_col] != "Unclassified"].copy()
        for bin_label in BIN_LABELS[variable]:
            precip_count = int(((valid_df[bin_col] == bin_label) & (valid_df["sample_type"] == "precip")).sum())
            no_precip_count = int(((valid_df[bin_col] == bin_label) & (valid_df["sample_type"] == "no_precip")).sum())
            total_count = precip_count + no_precip_count
            precip_ratio = precip_count / total_count if total_count else 0.0
            no_precip_ratio = no_precip_count / total_count if total_count else 0.0
            summary_rows.append(
                {
                    "variable": variable,
                    "bin": bin_label,
                    "precip_count": precip_count,
                    "no_precip_count": no_precip_count,
                    "total_count": total_count,
                    "precip_ratio": precip_ratio,
                    "no_precip_ratio": no_precip_ratio,
                }
            )
    return pd.DataFrame(summary_rows)


def plot_stacked_ratios(summary_df: pd.DataFrame, output_path: Path, stat: str, min_file_count: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    for ax, variable in zip(axes.flat, VARIABLES):
        subset = summary_df.loc[summary_df["variable"] == variable].copy()
        x = np.arange(len(subset))
        precip_percent = subset["precip_ratio"].to_numpy() * 100.0
        no_precip_percent = subset["no_precip_ratio"].to_numpy() * 100.0
        ax.bar(
            x,
            no_precip_percent,
            bottom=precip_percent,
            color="white",
            edgecolor="black",
            linewidth=0.8,
            label="No precip",
        )
        ax.bar(
            x,
            precip_percent,
            color="white",
            edgecolor="black",
            linewidth=0.8,
            hatch="///",
            label="Precip",
        )
        ax.set_title(variable)
        ax.set_xlabel("Bin", fontsize=13)
        ax.set_ylabel("Ratio (%)", fontsize=13)
        ax.set_ylim(0, 112)
        ax.set_xticks(x)
        ax.set_xticklabels(subset["bin"], rotation=30, ha="right", fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        for idx, row in enumerate(subset.itertuples(index=False)):
            ax.text(
                idx,
                102,
                f"n={int(row.total_count)}",
                ha="center",
                va="bottom",
                fontsize=10,
                clip_on=False,
            )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=13)
    fig.suptitle(
        f"FY4B cloud-parameter {stat} bins: precip vs no precip (file_count >= {min_file_count})",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_stat(args: argparse.Namespace, stat: str) -> tuple[Path, Path, int, int]:
    precip_df = load_samples(args.precip_csv, "precip", stat, args.min_file_count)
    no_precip_df = load_samples(args.no_precip_csv, "no_precip", stat, args.min_file_count)
    combined_df = add_bins(pd.concat([precip_df, no_precip_df], ignore_index=True))
    summary_df = summarize_bins(combined_df)

    summary_output = args.output_dir / f"fy4b_precip_no_precip_bin_summary_{stat}_filecount{args.min_file_count}.csv"
    figure_output = args.output_dir / f"fy4b_precip_no_precip_bin_ratio_{stat}_filecount{args.min_file_count}.png"
    summary_df.to_csv(summary_output, index=False)
    plot_stacked_ratios(summary_df, figure_output, stat, args.min_file_count)
    return summary_output, figure_output, len(precip_df), len(no_precip_df)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stat in ["mean", "max"]:
        summary_output, figure_output, precip_rows, no_precip_rows = process_stat(args, stat)
        print(f"[{stat}] Precip rows after file_count filter: {precip_rows}")
        print(f"[{stat}] No-precip rows after file_count filter: {no_precip_rows}")
        print(f"[{stat}] Summary CSV: {summary_output}")
        print(f"[{stat}] Figure: {figure_output}")


if __name__ == "__main__":
    main()
