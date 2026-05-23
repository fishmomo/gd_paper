from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


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
        description="Plot 3D cloud-parameter scatter plots for precip and sampled no-precip samples.",
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


def plot_triplet_scatter(
    precip_df: pd.DataFrame,
    no_precip_df: pd.DataFrame,
    output_path: Path,
    stat: str,
    min_file_count: int,
) -> None:
    triplets = list(combinations(VARIABLES, 3))
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    axes = [fig.add_subplot(2, 2, idx + 1, projection="3d") for idx in range(len(triplets))]

    for ax, (x_var, y_var, z_var) in zip(axes, triplets):
        ax.scatter(
            no_precip_df[x_var],
            no_precip_df[y_var],
            no_precip_df[z_var],
            s=16,
            marker="o",
            color="red",
            alpha=0.35,
            edgecolors="none",
        )
        ax.scatter(
            precip_df[x_var],
            precip_df[y_var],
            precip_df[z_var],
            s=22,
            marker="o",
            color="blue",
            alpha=0.75,
            edgecolors="none",
        )
        ax.set_xlabel(AXIS_LABELS[x_var], fontsize=11, labelpad=8)
        ax.set_ylabel(AXIS_LABELS[y_var], fontsize=11, labelpad=8)
        ax.set_zlabel(AXIS_LABELS[z_var], fontsize=11, labelpad=8)
        ax.set_title(f"{x_var} - {y_var} - {z_var}", fontsize=13)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.view_init(elev=22, azim=38)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Precip", markerfacecolor="blue", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="No precip sample", markerfacecolor="red", markersize=9),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=2, fontsize=12)
    fig.suptitle(
        (
            f"FY4B 3D cloud-parameter scatter ({stat}, file_count >= {min_file_count}; "
            f"precip n={len(precip_df)}, no-precip sampled n={len(no_precip_df)})"
        ),
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_stat(args: argparse.Namespace, stat: str) -> Path:
    precip_df = load_samples(args.precip_csv, "precip", stat, args.min_file_count)
    no_precip_df = load_samples(args.no_precip_csv, "no_precip", stat, args.min_file_count)
    sampled_no_precip_df = sample_no_precip(no_precip_df, len(precip_df), args.random_state)

    output_path = args.output_dir / f"fy4b_cloud_3d_triplet_scatter_{stat}_filecount{args.min_file_count}.png"
    plot_triplet_scatter(precip_df, sampled_no_precip_df, output_path, stat, args.min_file_count)
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
