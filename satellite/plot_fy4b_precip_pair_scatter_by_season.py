from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output\seasonal_precip_pair_scatter")
VARIABLES = ["COT", "CER", "CTH", "CTT"]
AXIS_LABELS = {
    "COT": "Cloud optical thickness",
    "CER": "Cloud effective radius",
    "CTH": "Cloud top height (km)",
    "CTT": "Cloud top temperature (C)",
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
SEASON_COLORS = {
    "Winter": "#3B82F6",
    "Spring": "#2A9D8F",
    "Summer": "#F4A261",
    "Autumn": "#C0392B",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot pairwise precip-sample cloud-parameter scatter by season.",
    )
    parser.add_argument(
        "--precip-csv",
        type=Path,
        default=DEFAULT_PRECIP_CSV,
        help="Date-hour-station cloud stats CSV for precipitation samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for seasonal scatter outputs.",
    )
    parser.add_argument(
        "--min-file-count",
        type=int,
        default=3,
        help="Minimum file_count required for a row to be included.",
    )
    return parser


def month_to_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


def load_precip_samples(csv_path: Path, stat: str, min_file_count: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_columns = {"date", "file_count", "precipitation", *[f"{variable}_{stat}" for variable in VARIABLES]}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["file_count"] = pd.to_numeric(df["file_count"], errors="coerce")
    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
    df = df.loc[(df["file_count"] >= min_file_count) & (df["precipitation"] > 0)].copy()

    for variable in VARIABLES:
        df[variable] = pd.to_numeric(df[f"{variable}_{stat}"], errors="coerce")

    df["CTH"] = df["CTH"] / 1000.0
    df["CTT"] = df["CTT"] - 273.15
    df = df.dropna(subset=["date", *VARIABLES]).reset_index(drop=True)
    df["season"] = df["date"].dt.month.astype(int).map(month_to_season)
    return df


def build_season_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("season", observed=False)
        .size()
        .reindex(SEASON_ORDER, fill_value=0)
        .reset_index(name="sample_count")
    )


def plot_pair_scatter_by_season(
    precip_df: pd.DataFrame,
    output_path: Path,
    stat: str,
    min_file_count: int,
) -> None:
    pairs = list(combinations(VARIABLES, 2))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for ax, (x_var, y_var) in zip(axes.flat, pairs):
        for season in SEASON_ORDER:
            season_df = precip_df.loc[precip_df["season"] == season]
            ax.scatter(
                season_df[x_var],
                season_df[y_var],
                s=16,
                color=SEASON_COLORS[season],
                alpha=0.65,
                edgecolors="none",
                label=season,
            )
        ax.set_xlabel(AXIS_LABELS[x_var], fontsize=12)
        ax.set_ylabel(AXIS_LABELS[y_var], fontsize=12)
        ax.set_title(f"{y_var} vs {x_var}", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(linestyle="--", alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, fontsize=12)
    fig.suptitle(
        f"FY4B precip pair scatter by season ({stat}, file_count >= {min_file_count}, n={len(precip_df)})",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_stat(args: argparse.Namespace, stat: str) -> tuple[Path, Path]:
    precip_df = load_precip_samples(args.precip_csv, stat, args.min_file_count)
    summary_df = build_season_summary(precip_df)

    figure_output = args.output_dir / f"fy4b_precip_pair_scatter_by_season_{stat}_filecount{args.min_file_count}.png"
    summary_output = args.output_dir / f"fy4b_precip_pair_scatter_by_season_{stat}_filecount{args.min_file_count}.csv"
    summary_df.to_csv(summary_output, index=False)
    plot_pair_scatter_by_season(precip_df, figure_output, stat, args.min_file_count)
    return summary_output, figure_output


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stat in ["mean", "max"]:
        summary_output, figure_output = process_stat(args, stat)
        print(f"[{stat}] Season summary: {summary_output}")
        print(f"[{stat}] Figure: {figure_output}")


if __name__ == "__main__":
    main()
