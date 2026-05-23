from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter


DEFAULT_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_NO_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_no_precip_date_hour_station_cloud_stats.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output\triplet_rotation")
VARIABLES = ["COT", "CER", "CTH", "CTT"]
TRIPLETS = list(combinations(VARIABLES, 3))
AXIS_LABELS = {
    "COT": "Cloud optical thickness",
    "CER": "Cloud effective radius",
    "CTH": "Cloud top height (km)",
    "CTT": "Cloud top temperature (C)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export rotating 3D FY4B cloud-parameter scatter plots to inspect precip/no-precip separation.",
    )
    parser.add_argument("--precip-csv", type=Path, default=DEFAULT_PRECIP_CSV)
    parser.add_argument("--no-precip-csv", type=Path, default=DEFAULT_NO_PRECIP_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-file-count", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--elev", type=float, default=22.0, help="Elevation angle for 3D view.")
    parser.add_argument("--frames", type=int, default=36, help="Number of azimuth frames per GIF.")
    parser.add_argument("--fps", type=int, default=6, help="Frames per second for GIF.")
    return parser


def load_samples(csv_path: Path, stat: str, min_file_count: int) -> pd.DataFrame:
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

    for variable in VARIABLES:
        df[variable] = pd.to_numeric(df[f"{variable}_{stat}"], errors="coerce")

    df["CTH"] = df["CTH"] / 1000.0
    df["CTT"] = df["CTT"] - 273.15
    return df.dropna(subset=VARIABLES).reset_index(drop=True)


def sample_no_precip(no_precip_df: pd.DataFrame, precip_count: int, random_state: int) -> pd.DataFrame:
    sample_count = min(len(no_precip_df), precip_count * 2)
    return no_precip_df.sample(n=sample_count, random_state=random_state).reset_index(drop=True)


def add_scatter(ax, precip_df: pd.DataFrame, no_precip_df: pd.DataFrame, triplet: tuple[str, str, str]) -> None:
    x_var, y_var, z_var = triplet
    ax.scatter(
        no_precip_df[x_var],
        no_precip_df[y_var],
        no_precip_df[z_var],
        s=18,
        marker="o",
        color="red",
        alpha=0.32,
        edgecolors="none",
        label="No precip",
    )
    ax.scatter(
        precip_df[x_var],
        precip_df[y_var],
        precip_df[z_var],
        s=28,
        marker="o",
        color="blue",
        alpha=0.8,
        edgecolors="none",
        label="Precip",
    )
    ax.set_xlabel(AXIS_LABELS[x_var], fontsize=11, labelpad=8)
    ax.set_ylabel(AXIS_LABELS[y_var], fontsize=11, labelpad=8)
    ax.set_zlabel(AXIS_LABELS[z_var], fontsize=11, labelpad=8)
    ax.set_title(f"{x_var} - {y_var} - {z_var}", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.25)


def save_rotation_gif(
    precip_df: pd.DataFrame,
    no_precip_df: pd.DataFrame,
    triplet: tuple[str, str, str],
    output_path: Path,
    stat: str,
    min_file_count: int,
    elev: float,
    frames: int,
    fps: int,
) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    add_scatter(ax, precip_df, no_precip_df, triplet)
    ax.view_init(elev=elev, azim=0)
    ax.legend(loc="upper right", fontsize=10)
    fig.suptitle(
        (
            f"{triplet[0]}-{triplet[1]}-{triplet[2]} | {stat} | "
            f"file_count >= {min_file_count} | precip n={len(precip_df)} | no-precip n={len(no_precip_df)}"
        ),
        fontsize=12,
    )

    def update(frame_idx: int):
        azim = frame_idx * (360.0 / frames)
        ax.view_init(elev=elev, azim=azim)
        return (ax,)

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_angle_panel(
    precip_df: pd.DataFrame,
    no_precip_df: pd.DataFrame,
    triplet: tuple[str, str, str],
    output_path: Path,
    stat: str,
    min_file_count: int,
    elev: float,
) -> None:
    azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    axes = [fig.add_subplot(2, 4, idx + 1, projection="3d") for idx in range(len(azimuths))]

    for ax, azim in zip(axes, azimuths):
        add_scatter(ax, precip_df, no_precip_df, triplet)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"azim={azim}", fontsize=11)
        ax.tick_params(axis="both", labelsize=8)

    fig.suptitle(
        (
            f"Rotating views for {triplet[0]}-{triplet[1]}-{triplet[2]} "
            f"({stat}, file_count >= {min_file_count})"
        ),
        fontsize=15,
    )
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def process_stat(args: argparse.Namespace, stat: str) -> list[tuple[Path, Path]]:
    precip_df = load_samples(args.precip_csv, stat, args.min_file_count)
    no_precip_df = load_samples(args.no_precip_csv, stat, args.min_file_count)
    sampled_no_precip_df = sample_no_precip(no_precip_df, len(precip_df), args.random_state)

    stat_output_dir = args.output_dir / stat
    stat_output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[tuple[Path, Path]] = []
    for triplet in TRIPLETS:
        triplet_name = "_".join(triplet).lower()
        gif_path = stat_output_dir / f"fy4b_{triplet_name}_{stat}_rotation.gif"
        panel_path = stat_output_dir / f"fy4b_{triplet_name}_{stat}_angles.png"
        save_rotation_gif(
            precip_df,
            sampled_no_precip_df,
            triplet,
            gif_path,
            stat,
            args.min_file_count,
            args.elev,
            args.frames,
            args.fps,
        )
        save_angle_panel(
            precip_df,
            sampled_no_precip_df,
            triplet,
            panel_path,
            stat,
            args.min_file_count,
            args.elev,
        )
        output_paths.append((gif_path, panel_path))
    return output_paths


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for stat in ["mean", "max"]:
        outputs = process_stat(args, stat)
        for gif_path, panel_path in outputs:
            print(f"[{stat}] GIF: {gif_path}")
            print(f"[{stat}] Panel: {panel_path}")


if __name__ == "__main__":
    main()
