from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output")
VARIABLES = ["COT", "CER", "CTH", "CTT"]
AXIS_LABELS = {
    "COT": "Cloud optical thickness",
    "CER": "Cloud effective radius",
    "CTH": "Cloud top height (km)",
    "CTT": "Cloud top temperature (C)",
}
INTENSITY_LABELS = ["0-0.5", "0.5-2", "2-5", ">5"]
INTENSITY_COLORS = {
    "0-0.5": "#4C78A8",
    "0.5-2": "#59A14F",
    "2-5": "#F28E2B",
    ">5": "#E15759",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot pairwise FY4B cloud-parameter scatter plots colored by precipitation intensity.",
    )
    parser.add_argument(
        "--precip-csv",
        type=Path,
        default=DEFAULT_PRECIP_CSV,
        help="Date-hour-station warm-cloud precipitation statistics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for scatter plot outputs.",
    )
    parser.add_argument(
        "--min-file-count",
        type=int,
        default=3,
        help="Minimum file_count required for a row to be included.",
    )
    return parser


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


def load_precip_samples(csv_path: Path, stat: str, min_file_count: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_columns = {"file_count", "precipitation", *[f"{variable}_{stat}" for variable in VARIABLES]}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

    df["file_count"] = pd.to_numeric(df["file_count"], errors="coerce")
    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
    df = df.loc[(df["file_count"] >= min_file_count) & (df["precipitation"] > 0)].copy()

    for variable in VARIABLES:
        df[variable] = pd.to_numeric(df[f"{variable}_{stat}"], errors="coerce")

    df["CTH"] = df["CTH"] / 1000.0
    df["CTT"] = df["CTT"] - 273.15
    df["intensity_bin"] = classify_intensity(df["precipitation"])
    return df.loc[df["intensity_bin"] != "Unclassified"].dropna(subset=VARIABLES).reset_index(drop=True)


def plot_pair_scatter_by_intensity(
    precip_df: pd.DataFrame,
    output_path: Path,
    stat: str,
    min_file_count: int,
) -> None:
    pairs = list(combinations(VARIABLES, 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for ax, (x_var, y_var) in zip(axes.flat, pairs):
        for intensity_label in INTENSITY_LABELS:
            subset = precip_df.loc[precip_df["intensity_bin"] == intensity_label]
            ax.scatter(
                subset[x_var],
                subset[y_var],
                s=20,
                marker="o",
                color=INTENSITY_COLORS[intensity_label],
                edgecolors="none",
                alpha=0.72,
                label=intensity_label,
            )
        ax.set_xlabel(AXIS_LABELS[x_var], fontsize=12)
        ax.set_ylabel(AXIS_LABELS[y_var], fontsize=12)
        ax.set_title(f"{y_var} vs {x_var}", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(linestyle="--", alpha=0.25)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=intensity_label,
            markerfacecolor=INTENSITY_COLORS[intensity_label],
            markersize=8,
        )
        for intensity_label in INTENSITY_LABELS
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(INTENSITY_LABELS),
        title="Precipitation intensity (mm/h)",
        fontsize=12,
        title_fontsize=12,
    )
    fig.suptitle(
        (
            f"FY4B pairwise cloud-parameter scatter by rain intensity "
            f"({stat}, file_count >= {min_file_count}; precip n={len(precip_df)})"
        ),
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_stat(args: argparse.Namespace, stat: str) -> Path:
    precip_df = load_precip_samples(args.precip_csv, stat, args.min_file_count)
    output_path = args.output_dir / f"fy4b_cloud_pair_scatter_rain_intensity_{stat}_filecount{args.min_file_count}.png"
    plot_pair_scatter_by_intensity(precip_df, output_path, stat, args.min_file_count)
    return output_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stat in ["mean", "max"]:
        output_path = process_stat(args, stat)
        print(f"[{stat}] Figure: {output_path}")


if __name__ == "__main__":
    main()
