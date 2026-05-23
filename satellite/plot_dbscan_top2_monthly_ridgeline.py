from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path(r"H:\YeWu\Zhou\guangzhou\output\fy4b_precip_pair_dbscan_top2_details_mean_filecount3.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output")
PAIR_ORDER = ["cth_cot", "cth_cer", "ctt_cth"]
PAIR_TITLES = {
    "cth_cot": "CTH-COT",
    "cth_cer": "CTH-CER",
    "ctt_cth": "CTT-CTH",
}
MONTH_LABELS = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}
CLUSTER_STYLES = {
    1: {"fill": "#4C78A8", "line": "#1D3557", "label": "Cluster 1"},
    2: {"fill": "#E07A5F", "line": "#9C2F2F", "label": "Cluster 2"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot month-hour ridgeline distributions for DBSCAN top-2 precip clusters.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input top-2 detail CSV, such as fy4b_precip_pair_dbscan_top2_details_mean_filecount3.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for ridgeline outputs.",
    )
    return parser


def load_samples(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_columns = {"date", "hour", "pair_name", "cluster_rank", "sample_index"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["cluster_rank"] = pd.to_numeric(df["cluster_rank"], errors="coerce")
    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce")
    df = df.dropna(subset=["date", "hour", "pair_name", "cluster_rank", "sample_index"]).copy()
    df["hour"] = df["hour"].astype(int)
    df["cluster_rank"] = df["cluster_rank"].astype(int)
    df["sample_index"] = df["sample_index"].astype(int)
    df["month"] = df["date"].dt.month
    return df


def summarize_hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["pair_name", "cluster_rank", "month", "hour"])["sample_index"]
        .nunique()
        .reset_index(name="sample_count")
    )

    full_index = pd.MultiIndex.from_product(
        [PAIR_ORDER, [1, 2], list(range(1, 13)), list(range(24))],
        names=["pair_name", "cluster_rank", "month", "hour"],
    )
    summary = (
        summary.set_index(["pair_name", "cluster_rank", "month", "hour"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    return summary


def plot_ridgeline(summary_df: pd.DataFrame, output_path: Path, title_suffix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 11), sharey=True)

    for ax, pair_name in zip(axes, PAIR_ORDER):
        pair_df = summary_df.loc[summary_df["pair_name"] == pair_name].copy()
        max_count = pair_df["sample_count"].max()
        scale = 0.82 / max(max_count, 1)

        for month in range(1, 13):
            baseline = month
            ax.hlines(baseline, 0, 23, color="0.7", linewidth=0.6, zorder=1)

            for cluster_rank in [1, 2]:
                subset = (
                    pair_df.loc[(pair_df["month"] == month) & (pair_df["cluster_rank"] == cluster_rank)]
                    .sort_values("hour")
                )
                hours = subset["hour"].to_numpy(dtype=float)
                counts = subset["sample_count"].to_numpy(dtype=float)
                ridge = baseline + counts * scale
                style = CLUSTER_STYLES[cluster_rank]

                ax.fill_between(
                    hours,
                    baseline,
                    ridge,
                    color=style["fill"],
                    alpha=0.5,
                    linewidth=0,
                    zorder=2 if cluster_rank == 1 else 3,
                )
                ax.plot(
                    hours,
                    ridge,
                    color=style["line"],
                    linewidth=1.4,
                    zorder=4 if cluster_rank == 1 else 5,
                )

        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Hour", fontsize=12)
        ax.set_title(PAIR_TITLES[pair_name], fontsize=14)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.18)

    axes[0].set_yticks(range(1, 13))
    axes[0].set_yticklabels([MONTH_LABELS[m] for m in range(1, 13)], fontsize=11)
    axes[0].set_ylabel("Month", fontsize=12)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CLUSTER_STYLES[1]["fill"], alpha=0.5, edgecolor="none", label="Cluster 1"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CLUSTER_STYLES[2]["fill"], alpha=0.5, edgecolor="none", label="Cluster 2"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=12)
    fig.suptitle(f"Monthly ridgeline of hourly precip-sample counts ({title_suffix})", fontsize=16)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.9, wspace=0.08)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_samples(args.input_csv)
    summary_df = summarize_hourly_counts(df)

    stem = args.input_csv.stem
    summary_output = args.output_dir / f"{stem}_month_hour_counts.csv"
    figure_output = args.output_dir / f"{stem}_hourly_ridgeline.png"

    summary_df.to_csv(summary_output, index=False)
    plot_ridgeline(summary_df, figure_output, stem)

    print(f"Input rows: {len(df)}")
    print(f"Month-hour summary: {summary_output}")
    print(f"Figure: {figure_output}")


if __name__ == "__main__":
    main()
