from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_NO_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_no_precip_date_hour_station_cloud_stats.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output")
VARIABLES = ["CTH", "CTT", "COT", "CER"]
BIN_LABELS = {
    "CTH": ["0-1.5 km", "1.5-2.5 km", "2.5-3.5 km", "3.5-4.5 km", ">4.5 km"],
    "CTT": ["0-5 C", "5-10 C", "10-15 C", "15-20 C", ">20 C"],
    "COT": ["0-10", "11-20", "21-30", "31-40", ">40"],
    "CER": ["0-10", "11-20", "21-30", "31-40", ">40"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot 5x5 FY4B precipitation-probability heatmaps for cloud-parameter pairs.",
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
        help="Directory for heatmap CSV and PNG outputs.",
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


def summarize_pair_probability(df: pd.DataFrame, x_var: str, y_var: str) -> pd.DataFrame:
    x_bin = f"{x_var}_bin"
    y_bin = f"{y_var}_bin"
    valid_df = df.loc[(df[x_bin] != "Unclassified") & (df[y_bin] != "Unclassified")].copy()
    rows = []

    for y_label in BIN_LABELS[y_var]:
        for x_label in BIN_LABELS[x_var]:
            cell = valid_df.loc[(valid_df[x_bin] == x_label) & (valid_df[y_bin] == y_label)]
            precip_count = int((cell["sample_type"] == "precip").sum())
            no_precip_count = int((cell["sample_type"] == "no_precip").sum())
            total_count = precip_count + no_precip_count
            precip_probability = precip_count / total_count if total_count else np.nan
            rows.append(
                {
                    "x_variable": x_var,
                    "y_variable": y_var,
                    "x_bin": x_label,
                    "y_bin": y_label,
                    "precip_count": precip_count,
                    "no_precip_count": no_precip_count,
                    "total_count": total_count,
                    "precip_probability": precip_probability,
                }
            )

    return pd.DataFrame(rows)


def plot_heatmap(ax: plt.Axes, pair_df: pd.DataFrame, x_var: str, y_var: str, panel_label: str) -> None:
    matrix = pair_df.pivot(index="y_bin", columns="x_bin", values="precip_probability")
    matrix = matrix.reindex(index=BIN_LABELS[y_var], columns=BIN_LABELS[x_var])
    total_matrix = pair_df.pivot(index="y_bin", columns="x_bin", values="total_count")
    total_matrix = total_matrix.reindex(index=BIN_LABELS[y_var], columns=BIN_LABELS[x_var])

    image = ax.imshow(matrix.to_numpy(dtype=float) * 100.0, vmin=0, vmax=100, cmap="YlGnBu", origin="lower")
    ax.set_title(panel_label, fontsize=24, loc="left")
    ax.set_xlabel(x_var, fontsize=17)
    ax.set_ylabel(y_var, fontsize=17)
    ax.set_xticks(np.arange(len(BIN_LABELS[x_var])))
    ax.set_yticks(np.arange(len(BIN_LABELS[y_var])))
    ax.set_xticklabels(BIN_LABELS[x_var], rotation=35, ha="right", fontsize=14)
    ax.set_yticklabels(BIN_LABELS[y_var], fontsize=14)

    for row_idx, y_label in enumerate(BIN_LABELS[y_var]):
        for col_idx, x_label in enumerate(BIN_LABELS[x_var]):
            probability = matrix.loc[y_label, x_label]
            total_count = total_matrix.loc[y_label, x_label]
            if pd.isna(probability):
                label = "NA\nn=0"
            else:
                label = f"{probability * 100.0:.1f}%\nn={int(total_count)}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=11)

    return image


def process_stat(args: argparse.Namespace, stat: str) -> tuple[Path, Path, int, int]:
    precip_df = load_samples(args.precip_csv, "precip", stat, args.min_file_count)
    no_precip_df = load_samples(args.no_precip_csv, "no_precip", stat, args.min_file_count)
    combined_df = add_bins(pd.concat([precip_df, no_precip_df], ignore_index=True))

    pair_frames = []
    pairs = list(combinations(VARIABLES, 2))
    fig, axes = plt.subplots(2, 3, figsize=(22, 14), constrained_layout=True)
    image = None

    for panel_idx, (ax, (x_var, y_var)) in enumerate(zip(axes.flat, pairs)):
        pair_df = summarize_pair_probability(combined_df, x_var, y_var)
        pair_frames.append(pair_df)
        panel_label = f"({chr(ord('a') + panel_idx)})"
        image = plot_heatmap(ax, pair_df, x_var, y_var, panel_label)

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, label="Precipitation probability (%)")
        colorbar.ax.tick_params(labelsize=14)
        colorbar.set_label("Precipitation probability (%)", fontsize=16)

    summary_df = pd.concat(pair_frames, ignore_index=True)
    summary_output = args.output_dir / f"fy4b_precip_probability_pair_heatmap_summary_{stat}_filecount{args.min_file_count}.csv"
    figure_output = args.output_dir / f"fy4b_precip_probability_pair_heatmaps_{stat}_filecount{args.min_file_count}.png"
    summary_df.to_csv(summary_output, index=False)
    fig.savefig(figure_output, dpi=300, bbox_inches="tight")
    plt.close(fig)

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
