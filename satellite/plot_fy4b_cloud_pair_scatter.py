from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_NO_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_no_precip_date_hour_station_cloud_stats.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output")
VARIABLES = ["COT", "CER", "CTH", "CTT"]
AXIS_LABELS = {
    "COT": "Cloud optical thickness",
    "CER": "Cloud effective radius",
    "CTH": "Cloud top height (km)",
    "CTT": "Cloud top temperature (C)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot pairwise cloud-parameter scatter plots for precip and sampled no-precip samples.",
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
        help="Directory for scatter plot outputs.",
    )
    parser.add_argument(
        "--min-file-count",
        type=int,
        default=3,
        help="Minimum file_count required for a row to be included.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling no-precipitation rows.",
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

    df["CTH"] = df["CTH"] / 1000.0
    df["CTT"] = df["CTT"] - 273.15
    return df.dropna(subset=VARIABLES).reset_index(drop=True)


def sample_no_precip(no_precip_df: pd.DataFrame, precip_count: int, random_state: int) -> pd.DataFrame:
    sample_count = min(len(no_precip_df), precip_count * 2)
    return no_precip_df.sample(n=sample_count, random_state=random_state).reset_index(drop=True)


def plot_pair_scatter(
    precip_df: pd.DataFrame,
    no_precip_df: pd.DataFrame,
    output_path: Path,
    stat: str,
    min_file_count: int,
) -> None:
    pairs = list(combinations(VARIABLES, 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for ax, (x_var, y_var) in zip(axes.flat, pairs):
        ax.scatter(
            no_precip_df[x_var],
            no_precip_df[y_var],
            s=12,
            marker="o",
            facecolors="none",
            edgecolors="0.55",
            linewidths=0.5,
            alpha=0.6,
            label="No precip sample",
        )
        ax.scatter(
            precip_df[x_var],
            precip_df[y_var],
            s=18,
            marker="x",
            color="black",
            linewidths=0.7,
            alpha=0.75,
            label="Precip",
        )
        ax.set_xlabel(AXIS_LABELS[x_var], fontsize=12)
        ax.set_ylabel(AXIS_LABELS[y_var], fontsize=12)
        ax.set_title(f"{y_var} vs {x_var}", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(linestyle="--", alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)
    fig.suptitle(
        (
            f"FY4B pairwise cloud-parameter scatter ({stat}, file_count >= {min_file_count}; "
            f"precip n={len(precip_df)}, no-precip sampled n={len(no_precip_df)})"
        ),
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_no_precip_pair_scatter(
    no_precip_df: pd.DataFrame,
    output_path: Path,
    stat: str,
    min_file_count: int,
) -> None:
    pairs = list(combinations(VARIABLES, 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for ax, (x_var, y_var) in zip(axes.flat, pairs):
        ax.scatter(
            no_precip_df[x_var],
            no_precip_df[y_var],
            s=14,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=0.55,
            alpha=0.65,
        )
        ax.set_xlabel(AXIS_LABELS[x_var], fontsize=12)
        ax.set_ylabel(AXIS_LABELS[y_var], fontsize=12)
        ax.set_title(f"{y_var} vs {x_var}", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(linestyle="--", alpha=0.25)

    fig.suptitle(
        (
            f"FY4B no-precip pairwise cloud-parameter scatter "
            f"({stat}, file_count >= {min_file_count}; sampled n={len(no_precip_df)})"
        ),
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_stat(args: argparse.Namespace, stat: str) -> tuple[Path, Path]:
    precip_df = load_samples(args.precip_csv, "precip", stat, args.min_file_count)
    no_precip_df = load_samples(args.no_precip_csv, "no_precip", stat, args.min_file_count)
    sampled_no_precip_df = sample_no_precip(no_precip_df, len(precip_df), args.random_state)

    output_path = args.output_dir / f"fy4b_cloud_pair_scatter_{stat}_filecount{args.min_file_count}.png"
    no_precip_output_path = args.output_dir / f"fy4b_cloud_pair_scatter_no_precip_{stat}_filecount{args.min_file_count}.png"
    plot_pair_scatter(precip_df, sampled_no_precip_df, output_path, stat, args.min_file_count)
    plot_no_precip_pair_scatter(sampled_no_precip_df, no_precip_output_path, stat, args.min_file_count)
    return output_path, no_precip_output_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stat in ["mean", "max"]:
        output_path, no_precip_output_path = process_stat(args, stat)
        print(f"[{stat}] Figure: {output_path}")
        print(f"[{stat}] No-precip figure: {no_precip_output_path}")


if __name__ == "__main__":
    main()
