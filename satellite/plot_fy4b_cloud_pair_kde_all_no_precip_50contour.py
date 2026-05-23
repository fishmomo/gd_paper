from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


DEFAULT_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_NO_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_no_precip_date_hour_station_cloud_stats.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output\kde_all_no_precip_50contour")
VARIABLES = ["COT", "CER", "CTH", "CTT"]
AXIS_LABELS = {
    "COT": "Cloud optical thickness",
    "CER": "Cloud effective radius",
    "CTH": "Cloud top height (km)",
    "CTT": "Cloud top temperature (C)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot pairwise KDE distributions for precip and all no-precip samples with dedicated 50% contours.",
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
        "--levels",
        type=int,
        default=6,
        help="Number of filled KDE contour levels.",
    )
    parser.add_argument(
        "--gridsize",
        type=int,
        default=120,
        help="Grid size for bivariate KDE estimation.",
    )
    parser.add_argument(
        "--bw-adjust",
        type=float,
        default=1.0,
        help="Bandwidth adjustment for KDE.",
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


def draw_kde_layer(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_var: str,
    y_var: str,
    fill_cmap: str,
    contour_color: str,
    half_contour_color: str,
    levels: int,
    gridsize: int,
    bw_adjust: float,
    ) -> None:
    sns.kdeplot(
        data=df,
        x=x_var,
        y=y_var,
        fill=True,
        levels=levels,
        thresh=0.05,
        cmap=fill_cmap,
        alpha=0.45,
        gridsize=gridsize,
        bw_adjust=bw_adjust,
        ax=ax,
    )
    sns.kdeplot(
        data=df,
        x=x_var,
        y=y_var,
        levels=levels,
        thresh=0.05,
        color=contour_color,
        linewidths=0.9,
        gridsize=gridsize,
        bw_adjust=bw_adjust,
        ax=ax,
    )
    sns.kdeplot(
        data=df,
        x=x_var,
        y=y_var,
        levels=[0.5],
        thresh=0.0,
        color=half_contour_color,
        linewidths=0.0,
        gridsize=gridsize,
        bw_adjust=bw_adjust,
        ax=ax,
    )


def compute_probability_boundary(
    precip_df: pd.DataFrame,
    no_precip_df: pd.DataFrame,
    x_var: str,
    y_var: str,
    gridsize: int,
    bw_adjust: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined_x = pd.concat([precip_df[x_var], no_precip_df[x_var]], ignore_index=True)
    combined_y = pd.concat([precip_df[y_var], no_precip_df[y_var]], ignore_index=True)

    x_padding = (combined_x.max() - combined_x.min()) * 0.05 or 1.0
    y_padding = (combined_y.max() - combined_y.min()) * 0.05 or 1.0
    x_grid = np.linspace(combined_x.min() - x_padding, combined_x.max() + x_padding, gridsize)
    y_grid = np.linspace(combined_y.min() - y_padding, combined_y.max() + y_padding, gridsize)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_positions = np.vstack([xx.ravel(), yy.ravel()])

    precip_values = np.vstack([precip_df[x_var].to_numpy(), precip_df[y_var].to_numpy()])
    no_precip_values = np.vstack([no_precip_df[x_var].to_numpy(), no_precip_df[y_var].to_numpy()])

    precip_kde = gaussian_kde(precip_values, bw_method="scott")
    no_precip_kde = gaussian_kde(no_precip_values, bw_method="scott")
    precip_kde.set_bandwidth(precip_kde.factor * bw_adjust)
    no_precip_kde.set_bandwidth(no_precip_kde.factor * bw_adjust)

    precip_density = precip_kde(grid_positions).reshape(xx.shape)
    no_precip_density = no_precip_kde(grid_positions).reshape(xx.shape)

    precip_prior = len(precip_df) / (len(precip_df) + len(no_precip_df))
    no_precip_prior = len(no_precip_df) / (len(precip_df) + len(no_precip_df))
    posterior_precip = (precip_prior * precip_density) / (
        precip_prior * precip_density + no_precip_prior * no_precip_density + 1e-12
    )
    return xx, yy, posterior_precip


def plot_pair_kde_overlay(
    precip_df: pd.DataFrame,
    no_precip_df: pd.DataFrame,
    output_path: Path,
    stat: str,
    min_file_count: int,
    levels: int,
    gridsize: int,
    bw_adjust: float,
) -> None:
    pairs = list(combinations(VARIABLES, 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for ax, (x_var, y_var) in zip(axes.flat, pairs):
        draw_kde_layer(
            ax,
            no_precip_df,
            x_var,
            y_var,
            fill_cmap="Reds",
            contour_color="#9B2226",
            half_contour_color="#5C0011",
            levels=levels,
            gridsize=gridsize,
            bw_adjust=bw_adjust,
        )
        draw_kde_layer(
            ax,
            precip_df,
            x_var,
            y_var,
            fill_cmap="Blues",
            contour_color="#0B4F8A",
            half_contour_color="#001F54",
            levels=levels,
            gridsize=gridsize,
            bw_adjust=bw_adjust,
        )
        xx, yy, posterior_precip = compute_probability_boundary(
            precip_df,
            no_precip_df,
            x_var,
            y_var,
            gridsize,
            bw_adjust,
        )
        ax.contour(
            xx,
            yy,
            posterior_precip,
            levels=[0.5],
            colors=["#6A00F4"],
            linewidths=2.4,
        )
        ax.set_xlabel(AXIS_LABELS[x_var], fontsize=12)
        ax.set_ylabel(AXIS_LABELS[y_var], fontsize=12)
        ax.set_title(f"{y_var} vs {x_var}", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(linestyle="--", alpha=0.2)

    legend_handles = [
        Line2D([0], [0], color="#0B4F8A", linewidth=1.2, label="Precip KDE contour"),
        Line2D([0], [0], color="#9B2226", linewidth=1.2, label="No-precip KDE contour"),
        Line2D([0], [0], color="#6A00F4", linewidth=2.4, label="Precip / no-precip 50% boundary"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=11)
    fig.suptitle(
        (
            f"FY4B pairwise cloud-parameter KDE with all no-precip samples ({stat}, "
            f"file_count >= {min_file_count}; precip n={len(precip_df)}, no-precip n={len(no_precip_df)})"
        ),
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_stat(args: argparse.Namespace, stat: str) -> Path:
    precip_df = load_samples(args.precip_csv, "precip", stat, args.min_file_count)
    no_precip_df = load_samples(args.no_precip_csv, "no_precip", stat, args.min_file_count)

    output_path = args.output_dir / f"fy4b_cloud_pair_kde_all_no_precip_50contour_{stat}_filecount{args.min_file_count}.png"
    plot_pair_kde_overlay(
        precip_df,
        no_precip_df,
        output_path,
        stat,
        args.min_file_count,
        args.levels,
        args.gridsize,
        args.bw_adjust,
    )
    return output_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stat in ["mean", "max"]:
        output_path = process_stat(args, stat)
        print(f"[{stat}] KDE figure: {output_path}")


if __name__ == "__main__":
    main()
