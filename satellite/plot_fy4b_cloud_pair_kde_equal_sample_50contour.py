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
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output\kde_equal_no_precip_50contour")
VARIABLES = ["COT", "CER", "CTH", "CTT"]
AXIS_LABELS = {
    "COT": "Cloud optical thickness",
    "CER": "Cloud effective radius",
    "CTH": "Cloud top height (km)",
    "CTT": "Cloud top temperature (C)",
}
EPSILON = 1e-12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot 1:1 precip/no-precip pairwise KDE overlays with 50% posterior-probability contours.",
    )
    parser.add_argument("--precip-csv", type=Path, default=DEFAULT_PRECIP_CSV)
    parser.add_argument("--no-precip-csv", type=Path, default=DEFAULT_NO_PRECIP_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-file-count", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--stats", nargs="+", choices=["mean", "max"], default=["mean"])
    parser.add_argument("--levels", type=int, default=7)
    parser.add_argument("--gridsize", type=int, default=140)
    parser.add_argument("--bw-adjust", type=float, default=1.0)
    return parser


def load_samples(csv_path: Path, sample_type: str, stat: str, min_file_count: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["_source_row"] = np.arange(len(df))
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


def sample_no_precip_equal(no_precip_df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if len(no_precip_df) < sample_size:
        raise ValueError(f"No-precip rows ({len(no_precip_df)}) are fewer than precip rows ({sample_size}).")
    return no_precip_df.sample(n=sample_size, replace=False, random_state=random_state).reset_index(drop=True)


def build_grid(
    precip_df: pd.DataFrame,
    no_precip_df: pd.DataFrame,
    x_var: str,
    y_var: str,
    gridsize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    combined_x = pd.concat([precip_df[x_var], no_precip_df[x_var]], ignore_index=True)
    combined_y = pd.concat([precip_df[y_var], no_precip_df[y_var]], ignore_index=True)
    x_padding = (combined_x.max() - combined_x.min()) * 0.05 or 1.0
    y_padding = (combined_y.max() - combined_y.min()) * 0.05 or 1.0
    x_grid = np.linspace(combined_x.min() - x_padding, combined_x.max() + x_padding, gridsize)
    y_grid = np.linspace(combined_y.min() - y_padding, combined_y.max() + y_padding, gridsize)
    xx, yy = np.meshgrid(x_grid, y_grid)
    return x_grid, y_grid, xx, yy


def evaluate_kde(
    df: pd.DataFrame,
    x_var: str,
    y_var: str,
    xx: np.ndarray,
    yy: np.ndarray,
    bw_adjust: float,
) -> np.ndarray:
    values = np.vstack([df[x_var].to_numpy(), df[y_var].to_numpy()])
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kde = gaussian_kde(values, bw_method="scott")
    kde.set_bandwidth(kde.factor * bw_adjust)
    density = kde(positions).reshape(xx.shape)
    return np.clip(density, 0.0, None)


def compute_kde_products(
    precip_df: pd.DataFrame,
    no_precip_df: pd.DataFrame,
    x_var: str,
    y_var: str,
    gridsize: int,
    bw_adjust: float,
) -> dict[str, np.ndarray]:
    x_grid, y_grid, xx, yy = build_grid(precip_df, no_precip_df, x_var, y_var, gridsize)
    precip_density = evaluate_kde(precip_df, x_var, y_var, xx, yy, bw_adjust)
    no_precip_density = evaluate_kde(no_precip_df, x_var, y_var, xx, yy, bw_adjust)

    # Equal 1:1 sampling implies equal priors for the 50% posterior boundary.
    posterior_precip = precip_density / (precip_density + no_precip_density + EPSILON)
    precip_probability_grid = precip_density / (precip_density.sum() + EPSILON)
    no_precip_probability_grid = no_precip_density / (no_precip_density.sum() + EPSILON)
    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "xx": xx,
        "yy": yy,
        "precip_density": precip_density,
        "no_precip_density": no_precip_density,
        "posterior_precip": posterior_precip,
        "precip_probability_grid": precip_probability_grid,
        "no_precip_probability_grid": no_precip_probability_grid,
    }


def draw_kde_layer(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_var: str,
    y_var: str,
    fill_cmap: str,
    contour_color: str,
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
        alpha=0.42,
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
        linewidths=0.95,
        gridsize=gridsize,
        bw_adjust=bw_adjust,
        ax=ax,
    )


def plot_pair_kde_overlay(
    precip_df: pd.DataFrame,
    no_precip_df: pd.DataFrame,
    output_path: Path,
    grids_dir: Path,
    stat: str,
    min_file_count: int,
    levels: int,
    gridsize: int,
    bw_adjust: float,
) -> pd.DataFrame:
    pairs = list(combinations(VARIABLES, 2))
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    grid_rows = []

    for panel_idx, (ax, (x_var, y_var)) in enumerate(zip(axes.flat, pairs)):
        pair_name = f"{x_var}_{y_var}"
        products = compute_kde_products(precip_df, no_precip_df, x_var, y_var, gridsize, bw_adjust)
        grid_path = grids_dir / f"kde_grid_{pair_name}_{stat}_filecount{min_file_count}.npz"
        np.savez_compressed(
            grid_path,
            x_grid=products["x_grid"],
            y_grid=products["y_grid"],
            precip_density=products["precip_density"],
            no_precip_density=products["no_precip_density"],
            posterior_precip=products["posterior_precip"],
            precip_probability_grid=products["precip_probability_grid"],
            no_precip_probability_grid=products["no_precip_probability_grid"],
            stat=np.array(stat),
            x_var=np.array(x_var),
            y_var=np.array(y_var),
            precip_rows=np.array(len(precip_df)),
            no_precip_rows=np.array(len(no_precip_df)),
            min_file_count=np.array(min_file_count),
            random_state_note=np.array("stored in run manifest"),
        )
        grid_rows.append(
            {
                "stat": stat,
                "pair_order": panel_idx + 1,
                "x_var": x_var,
                "y_var": y_var,
                "pair_name": pair_name,
                "grid_path": str(grid_path),
            }
        )

        draw_kde_layer(ax, no_precip_df, x_var, y_var, "Reds", "#9B2226", levels, gridsize, bw_adjust)
        draw_kde_layer(ax, precip_df, x_var, y_var, "Blues", "#0B4F8A", levels, gridsize, bw_adjust)
        ax.contour(
            products["xx"],
            products["yy"],
            products["posterior_precip"],
            levels=[0.5],
            colors=["#6A00F4"],
            linewidths=2.5,
        )
        ax.set_xlabel(AXIS_LABELS[x_var], fontsize=16)
        ax.set_ylabel(AXIS_LABELS[y_var], fontsize=16)
        ax.set_title(f"({chr(ord('a') + panel_idx)})", fontsize=22, loc="left")
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(linestyle="--", alpha=0.2)

    legend_handles = [
        Line2D([0], [0], color="#0B4F8A", linewidth=1.4, label="Precip KDE"),
        Line2D([0], [0], color="#9B2226", linewidth=1.4, label="No-precip KDE"),
        Line2D([0], [0], color="#6A00F4", linewidth=2.5, label="50% posterior boundary"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=14)
    fig.suptitle(
        (
            f"FY4B pairwise cloud-parameter KDE, 1:1 samples "
            f"({stat}, file_count >= {min_file_count}; n={len(precip_df)} each)"
        ),
        fontsize=18,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(grid_rows)


def process_stat(args: argparse.Namespace, stat: str) -> dict[str, Path | int]:
    precip_df = load_samples(args.precip_csv, "precip", stat, args.min_file_count)
    no_precip_all_df = load_samples(args.no_precip_csv, "no_precip", stat, args.min_file_count)
    sampled_no_precip_df = sample_no_precip_equal(no_precip_all_df, len(precip_df), args.random_state)

    stat_dir = args.output_dir / stat
    grids_dir = stat_dir / "kde_grids"
    samples_dir = stat_dir / "samples"
    grids_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    precip_sample_path = samples_dir / f"precip_sample_{stat}_filecount{args.min_file_count}.csv"
    no_precip_sample_path = samples_dir / f"no_precip_equal_sample_{stat}_filecount{args.min_file_count}.csv"
    precip_df.to_csv(precip_sample_path, index=False)
    sampled_no_precip_df.to_csv(no_precip_sample_path, index=False)

    figure_path = stat_dir / f"fy4b_cloud_pair_kde_equal_no_precip_50contour_{stat}_filecount{args.min_file_count}.png"
    grid_manifest_df = plot_pair_kde_overlay(
        precip_df,
        sampled_no_precip_df,
        figure_path,
        grids_dir,
        stat,
        args.min_file_count,
        args.levels,
        args.gridsize,
        args.bw_adjust,
    )
    grid_manifest_path = stat_dir / f"kde_grid_manifest_{stat}_filecount{args.min_file_count}.csv"
    grid_manifest_df.to_csv(grid_manifest_path, index=False)

    return {
        "figure_path": figure_path,
        "grid_manifest_path": grid_manifest_path,
        "precip_sample_path": precip_sample_path,
        "no_precip_sample_path": no_precip_sample_path,
        "precip_rows": len(precip_df),
        "no_precip_rows": len(sampled_no_precip_df),
        "no_precip_pool_rows": len(no_precip_all_df),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for stat in args.stats:
        result = process_stat(args, stat)
        manifest_rows.append(
            {
                "stat": stat,
                "min_file_count": args.min_file_count,
                "random_state": args.random_state,
                "precip_rows": result["precip_rows"],
                "no_precip_rows": result["no_precip_rows"],
                "no_precip_pool_rows": result["no_precip_pool_rows"],
                "figure_path": str(result["figure_path"]),
                "grid_manifest_path": str(result["grid_manifest_path"]),
                "precip_sample_path": str(result["precip_sample_path"]),
                "no_precip_sample_path": str(result["no_precip_sample_path"]),
            }
        )
        print(f"[{stat}] Figure: {result['figure_path']}")
        print(f"[{stat}] KDE grid manifest: {result['grid_manifest_path']}")
        print(f"[{stat}] Sample rows: precip={result['precip_rows']}, no_precip={result['no_precip_rows']}")

    manifest_path = args.output_dir / f"manifest_filecount{args.min_file_count}.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Output directory: {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
