from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_NO_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_no_precip_date_hour_station_cloud_stats.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output\equal_no_precip_pair_combined_scatter_kde")
VARIABLES = ["COT", "CER", "CTH", "CTT"]
AXIS_LABELS = {
    "COT": "Cloud optical thickness",
    "CER": "Cloud effective radius",
    "CTH": "Cloud top height (km)",
    "CTT": "Cloud top temperature (C)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repeatedly sample no-precipitation rows without replacement, using the "
            "same sample size as precipitation rows, then plot no-precipitation-only "
            "2x3 combined pair scatter and KDE figures."
        ),
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
        help="Directory for repeated sampling scatter/KDE outputs.",
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
        help="Random seed used to shuffle no-precipitation rows.",
    )
    parser.add_argument(
        "--stats",
        nargs="+",
        choices=["mean", "max"],
        default=["mean", "max"],
        help="Cloud-parameter statistic columns to plot.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=7,
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


def build_no_precip_draws(no_precip_df: pd.DataFrame, sample_size: int, random_state: int) -> list[pd.DataFrame]:
    shuffled = no_precip_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    draw_count = len(shuffled) // sample_size
    return [
        shuffled.iloc[draw_index * sample_size : (draw_index + 1) * sample_size].reset_index(drop=True)
        for draw_index in range(draw_count)
    ]


def plot_combined_scatter(
    no_precip_df: pd.DataFrame,
    output_path: Path,
    draw_number: int,
    stat: str,
    min_file_count: int,
) -> None:
    pairs = list(combinations(VARIABLES, 2))
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)

    for panel_idx, (ax, (x_var, y_var)) in enumerate(zip(axes.flat, pairs)):
        ax.scatter(
            no_precip_df[x_var],
            no_precip_df[y_var],
            s=16,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=0.55,
            alpha=0.62,
        )
        ax.set_xlabel(AXIS_LABELS[x_var], fontsize=16)
        ax.set_ylabel(AXIS_LABELS[y_var], fontsize=16)
        ax.set_title(f"({chr(ord('a') + panel_idx)})", fontsize=22, loc="left")
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(linestyle="--", alpha=0.25)

    fig.suptitle(
        f"No-precipitation sample draw {draw_number:02d} ({stat}, file_count >= {min_file_count}; n={len(no_precip_df)})",
        fontsize=18,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_kde(
    no_precip_df: pd.DataFrame,
    output_path: Path,
    draw_number: int,
    stat: str,
    min_file_count: int,
    levels: int,
) -> None:
    pairs = list(combinations(VARIABLES, 2))
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)

    for panel_idx, (ax, (x_var, y_var)) in enumerate(zip(axes.flat, pairs)):
        sns.kdeplot(
            data=no_precip_df,
            x=x_var,
            y=y_var,
            fill=True,
            levels=levels,
            thresh=0.05,
            cmap="Blues",
            alpha=0.55,
            ax=ax,
        )
        sns.kdeplot(
            data=no_precip_df,
            x=x_var,
            y=y_var,
            levels=levels,
            thresh=0.05,
            color="#1F77B4",
            linewidths=1.0,
            ax=ax,
        )
        ax.set_xlabel(AXIS_LABELS[x_var], fontsize=16)
        ax.set_ylabel(AXIS_LABELS[y_var], fontsize=16)
        ax.set_title(f"({chr(ord('a') + panel_idx)})", fontsize=22, loc="left")
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(linestyle="--", alpha=0.2)

    fig.suptitle(
        f"No-precipitation sample draw {draw_number:02d} KDE ({stat}, file_count >= {min_file_count}; n={len(no_precip_df)})",
        fontsize=18,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_stat(args: argparse.Namespace, stat: str) -> pd.DataFrame:
    precip_df = load_samples(args.precip_csv, "precip", stat, args.min_file_count)
    no_precip_df = load_samples(args.no_precip_csv, "no_precip", stat, args.min_file_count)
    draws = build_no_precip_draws(no_precip_df, len(precip_df), args.random_state)

    manifest_rows = []
    stat_dir = args.output_dir / stat

    for draw_index, draw_df in enumerate(draws, start=1):
        draw_dir = stat_dir / f"draw_{draw_index:02d}"
        draw_dir.mkdir(parents=True, exist_ok=True)
        scatter_path = draw_dir / f"draw_{draw_index:02d}_no_precip_pair_scatter_{stat}.png"
        kde_path = draw_dir / f"draw_{draw_index:02d}_no_precip_pair_kde_{stat}.png"
        plot_combined_scatter(draw_df, scatter_path, draw_index, stat, args.min_file_count)
        plot_combined_kde(draw_df, kde_path, draw_index, stat, args.min_file_count, args.levels)
        manifest_rows.append(
            {
                "stat": stat,
                "draw": draw_index,
                "precip_rows_used_for_sample_size": len(precip_df),
                "no_precip_rows": len(draw_df),
                "scatter_path": str(scatter_path),
                "kde_path": str(kde_path),
            }
        )

    stat_dir.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(stat_dir / f"manifest_{stat}_filecount{args.min_file_count}.csv", index=False)
    return manifest_df


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_manifests = []
    for stat in args.stats:
        manifest_df = process_stat(args, stat)
        all_manifests.append(manifest_df)
        draw_count = manifest_df["draw"].nunique() if not manifest_df.empty else 0
        print(f"[{stat}] Draws: {draw_count}")
        print(f"[{stat}] Combined figures: {len(manifest_df) * 2}")

    combined_manifest = pd.concat(all_manifests, ignore_index=True) if all_manifests else pd.DataFrame()
    combined_manifest.to_csv(args.output_dir / f"manifest_filecount{args.min_file_count}.csv", index=False)
    print(f"Output directory: {args.output_dir}")
    print(f"Manifest: {args.output_dir / f'manifest_filecount{args.min_file_count}.csv'}")


if __name__ == "__main__":
    main()
