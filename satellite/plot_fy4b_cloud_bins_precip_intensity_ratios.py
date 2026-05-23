from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output")
VARIABLES = ["CTH", "CTT", "COT", "CER"]
BIN_LABELS = {
    "CTH": ["0-1.5 km", "1.5-2.5 km", "2.5-3.5 km", "3.5-4.5 km", ">4.5 km"],
    "CTT": ["0-5 C", "5-10 C", "10-15 C", "15-20 C", ">20 C"],
    "COT": ["0-10", "11-20", "21-30", "31-40", ">40"],
    "CER": ["0-10", "11-20", "21-30", "31-40", ">40"],
}
INTENSITY_LABELS = ["0-0.5", "0.5-2", "2-5", ">5"]
HATCHES = ["", "///", "\\\\\\", "xx"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot grouped precipitation-intensity ratios for each cloud-parameter bin.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Date-hour-station warm-cloud precipitation statistics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary CSV and figure outputs.",
    )
    parser.add_argument(
        "--min-file-count",
        type=int,
        default=3,
        help="Minimum file_count required for a row to be included.",
    )
    return parser


def load_samples(input_csv: Path, stat: str, min_file_count: int) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_columns = {"file_count", "precipitation", *[f"{variable}_{stat}" for variable in VARIABLES]}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Input CSV is missing required columns: {sorted(missing_columns)}")

    df["file_count"] = pd.to_numeric(df["file_count"], errors="coerce")
    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
    df = df.loc[(df["file_count"] >= min_file_count) & (df["precipitation"] > 0)].copy()

    for variable in VARIABLES:
        df[variable] = pd.to_numeric(df[f"{variable}_{stat}"], errors="coerce")

    df["CTH_km"] = df["CTH"] / 1000.0
    df["CTT_C"] = df["CTT"] - 273.15
    return df


def classify_series(series: pd.Series, conditions: list[pd.Series], labels: list[str]) -> pd.Categorical:
    values = np.select(conditions, labels, default="Unclassified")
    return pd.Categorical(values, categories=labels + ["Unclassified"], ordered=True)


def add_cloud_bins(df: pd.DataFrame) -> pd.DataFrame:
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


def add_intensity_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["intensity_bin"] = classify_series(
        df["precipitation"],
        [
            (df["precipitation"] >= 0.0) & (df["precipitation"] <= 0.5),
            (df["precipitation"] > 0.5) & (df["precipitation"] <= 2.0),
            (df["precipitation"] > 2.0) & (df["precipitation"] <= 5.0),
            df["precipitation"] > 5.0,
        ],
        INTENSITY_LABELS,
    )
    return df


def summarize_intensity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in VARIABLES:
        cloud_bin_col = f"{variable}_bin"
        valid_df = df.loc[
            (df[cloud_bin_col] != "Unclassified") & (df["intensity_bin"] != "Unclassified")
        ].copy()
        for cloud_bin in BIN_LABELS[variable]:
            cloud_subset = valid_df.loc[valid_df[cloud_bin_col] == cloud_bin]
            total_count = len(cloud_subset)
            for intensity_bin in INTENSITY_LABELS:
                count = int((cloud_subset["intensity_bin"] == intensity_bin).sum())
                rows.append(
                    {
                        "variable": variable,
                        "cloud_bin": cloud_bin,
                        "intensity_bin": intensity_bin,
                        "count": count,
                        "total_count": total_count,
                        "ratio": count / total_count if total_count else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def plot_grouped_bars(summary_df: pd.DataFrame, output_path: Path, stat: str, min_file_count: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    width = 0.18
    offsets = (np.arange(len(INTENSITY_LABELS)) - (len(INTENSITY_LABELS) - 1) / 2) * width

    for ax, variable in zip(axes.flat, VARIABLES):
        subset = summary_df.loc[summary_df["variable"] == variable].copy()
        x = np.arange(len(BIN_LABELS[variable]))

        for idx, intensity_bin in enumerate(INTENSITY_LABELS):
            intensity_subset = subset.loc[subset["intensity_bin"] == intensity_bin]
            ratios = (
                intensity_subset.set_index("cloud_bin")
                .reindex(BIN_LABELS[variable])["ratio"]
                .fillna(0.0)
                .to_numpy()
                * 100.0
            )
            ax.bar(
                x + offsets[idx],
                ratios,
                width=width,
                color="white",
                edgecolor="black",
                linewidth=0.8,
                hatch=HATCHES[idx],
                label=intensity_bin,
            )

        total_counts = (
            subset.drop_duplicates(subset=["cloud_bin"])
            .set_index("cloud_bin")
            .reindex(BIN_LABELS[variable])["total_count"]
            .fillna(0)
            .astype(int)
        )
        for x_idx, total_count in enumerate(total_counts):
            ax.text(
                x_idx,
                102,
                f"n={total_count}",
                ha="center",
                va="bottom",
                fontsize=10,
                clip_on=False,
            )

        ax.set_title(variable, fontsize=14)
        ax.set_xlabel("Cloud-parameter bin", fontsize=13)
        ax.set_ylabel("Rain-intensity ratio (%)", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS[variable], rotation=30, ha="right", fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(0, 112)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=12)
    fig.suptitle(
        f"Warm-cloud rain-intensity ratios by cloud-parameter bins ({stat}, file_count >= {min_file_count})",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_stat(args: argparse.Namespace, stat: str) -> tuple[Path, Path, int]:
    df = load_samples(args.input_csv, stat, args.min_file_count)
    df = add_intensity_bins(add_cloud_bins(df))
    summary_df = summarize_intensity_ratios(df)

    summary_output = args.output_dir / f"fy4b_cloud_bin_rain_intensity_ratio_{stat}_filecount{args.min_file_count}.csv"
    figure_output = args.output_dir / f"fy4b_cloud_bin_rain_intensity_ratio_{stat}_filecount{args.min_file_count}.png"
    summary_df.to_csv(summary_output, index=False)
    plot_grouped_bars(summary_df, figure_output, stat, args.min_file_count)
    return summary_output, figure_output, len(df)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stat in ["mean", "max"]:
        summary_output, figure_output, row_count = process_stat(args, stat)
        print(f"[{stat}] Rows after filter: {row_count}")
        print(f"[{stat}] Summary CSV: {summary_output}")
        print(f"[{stat}] Figure: {figure_output}")


if __name__ == "__main__":
    main()
