from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
        description="Plot pairwise KDE distributions for FY4B cloud parameters.",
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
        help="Directory for KDE plot outputs.",
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
    parser.add_argument(
        "--levels",
        type=int,
        default=6,
        help="Number of KDE contour levels.",
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


def plot_pair_kde(
    df: pd.DataFrame,
    output_path: Path,
    stat: str,
    min_file_count: int,
    levels: int,
    sample_label: str,
    fill_cmap: str,
    line_color: str,
) -> None:
    pairs = list(combinations(VARIABLES, 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for ax, (x_var, y_var) in zip(axes.flat, pairs):
        sns.kdeplot(
            data=df,
            x=x_var,
            y=y_var,
            fill=True,
            levels=levels,
            thresh=0.05,
            cmap=fill_cmap,
            alpha=0.55,
            ax=ax,
        )
        sns.kdeplot(
            data=df,
            x=x_var,
            y=y_var,
            levels=levels,
            thresh=0.05,
            color=line_color,
            linewidths=1.0,
            ax=ax,
        )
        ax.set_xlabel(AXIS_LABELS[x_var], fontsize=12)
        ax.set_ylabel(AXIS_LABELS[y_var], fontsize=12)
        ax.set_title(f"{y_var} vs {x_var}", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(linestyle="--", alpha=0.2)

    fig.suptitle(
        (
            f"FY4B pairwise cloud-parameter KDE ({sample_label}, {stat}, "
            f"file_count >= {min_file_count}; n={len(df)})"
        ),
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_stat(args: argparse.Namespace, stat: str) -> tuple[Path, Path]:
    precip_df = load_samples(args.precip_csv, "precip", stat, args.min_file_count)
    no_precip_df = load_samples(args.no_precip_csv, "no_precip", stat, args.min_file_count)
    sampled_no_precip_df = sample_no_precip(no_precip_df, len(precip_df), args.random_state)

    precip_output_path = args.output_dir / f"fy4b_cloud_pair_kde_precip_{stat}_filecount{args.min_file_count}.png"
    no_precip_output_path = args.output_dir / f"fy4b_cloud_pair_kde_no_precip_{stat}_filecount{args.min_file_count}.png"
    plot_pair_kde(
        precip_df,
        precip_output_path,
        stat,
        args.min_file_count,
        args.levels,
        sample_label="precip",
        fill_cmap="Blues",
        line_color="#0B4F8A",
    )
    plot_pair_kde(
        sampled_no_precip_df,
        no_precip_output_path,
        stat,
        args.min_file_count,
        args.levels,
        sample_label="no precip sample",
        fill_cmap="Reds",
        line_color="#9B2226",
    )
    return precip_output_path, no_precip_output_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stat in ["mean", "max"]:
        output_path, no_precip_output_path = process_stat(args, stat)
        print(f"[{stat}] Precip KDE figure: {output_path}")
        print(f"[{stat}] No-precip KDE figure: {no_precip_output_path}")


if __name__ == "__main__":
    main()
