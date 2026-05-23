from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output")
INTENSITY_LABELS = ["0-0.5", "0.5-2", "2-5", ">5"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot warm-cloud precipitation intensity histogram.",
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
        help="Directory for histogram CSV and PNG outputs.",
    )
    parser.add_argument(
        "--min-file-count",
        type=int,
        default=3,
        help="Minimum file_count required for a row to be included.",
    )
    return parser


def load_precipitation(input_csv: Path, min_file_count: int) -> pd.Series:
    df = pd.read_csv(input_csv)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_columns = {"file_count", "precipitation"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Input CSV is missing required columns: {sorted(missing_columns)}")

    df["file_count"] = pd.to_numeric(df["file_count"], errors="coerce")
    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
    filtered = df.loc[(df["file_count"] >= min_file_count) & (df["precipitation"] > 0), "precipitation"]
    return filtered.dropna().reset_index(drop=True)


def classify_intensity(precipitation: pd.Series) -> pd.Categorical:
    values = np.select(
        [
            (precipitation >= 0.0) & (precipitation <= 0.5),
            (precipitation > 0.5) & (precipitation <= 2.0),
            (precipitation > 2.0) & (precipitation <= 5.0),
            precipitation > 5.0,
        ],
        INTENSITY_LABELS,
        default="Unclassified",
    )
    return pd.Categorical(values, categories=INTENSITY_LABELS + ["Unclassified"], ordered=True)


def build_histogram_table(precipitation: pd.Series) -> pd.DataFrame:
    intensity_bin = classify_intensity(precipitation)
    counts = pd.Series(intensity_bin).value_counts(sort=False)
    counts = counts[counts.index != "Unclassified"]
    total = counts.sum()
    return pd.DataFrame(
        {
            "intensity_bin": counts.index.astype(str),
            "count": counts.values,
            "ratio": counts.values / total if total else np.zeros(len(counts), dtype=float),
        }
    )


def plot_histogram(histogram_df: pd.DataFrame, output_path: Path, min_file_count: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.bar(
        histogram_df["intensity_bin"],
        histogram_df["count"],
        color="white",
        edgecolor="black",
        linewidth=0.9,
        hatch="///",
    )
    ax.set_xlabel("Precipitation intensity (mm/h)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_title(
        f"Warm-cloud precipitation intensity (file_count >= {min_file_count}, n={int(histogram_df['count'].sum())})",
        fontsize=15,
    )
    ax.bar_label(bars, labels=[str(int(value)) for value in histogram_df["count"]], padding=3, fontsize=11)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    precipitation = load_precipitation(args.input_csv, args.min_file_count)
    if precipitation.empty:
        raise ValueError("No precipitation rows found after filtering.")

    histogram_df = build_histogram_table(precipitation)
    csv_output = args.output_dir / f"warmcloud_precip_intensity_histogram_filecount{args.min_file_count}.csv"
    figure_output = args.output_dir / f"warmcloud_precip_intensity_histogram_filecount{args.min_file_count}.png"
    histogram_df.to_csv(csv_output, index=False)
    plot_histogram(histogram_df, figure_output, args.min_file_count)

    print(f"Input CSV: {args.input_csv}")
    print(f"Rows after filter: {len(precipitation)}")
    print(f"Min precipitation: {precipitation.min()}")
    print(f"Max precipitation: {precipitation.max()}")
    print(f"Histogram CSV: {csv_output}")
    print(f"Figure: {figure_output}")


if __name__ == "__main__":
    main()
