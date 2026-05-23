from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path(r"H:\YeWu\Zhou\guangzhou\output\fy4b_precip_pair_dbscan_top2_details_mean_filecount3.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output")
PRECIP_INTENSITY_LABELS = ["0-0.5", "0.5-2", "2-5", ">5"]
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
    1: {"fill": "#4C78A8", "line": "#1D3557", "label": "Cluster 1 union"},
    2: {"fill": "#E07A5F", "line": "#9C2F2F", "label": "Cluster 2 union"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot month-hour ridgeline for the union of DBSCAN top-2 clusters across all pair combinations.",
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
        help="Directory for union ridgeline outputs.",
    )
    return parser


def load_samples(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_columns = {"date", "hour", "cluster_rank", "sample_index"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["cluster_rank"] = pd.to_numeric(df["cluster_rank"], errors="coerce")
    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce")
    df = df.dropna(subset=["date", "hour", "cluster_rank", "sample_index"]).copy()
    df["hour"] = df["hour"].astype(int)
    df["cluster_rank"] = df["cluster_rank"].astype(int)
    df["sample_index"] = df["sample_index"].astype(int)
    df["month"] = df["date"].dt.month
    return df


def build_union_cluster_rows(df: pd.DataFrame) -> pd.DataFrame:
    base_df = (
        df.sort_values(["sample_index", "date", "hour"])
        .drop_duplicates(subset=["sample_index"])
        .loc[:, ["sample_index", "date", "hour", "month", "COT", "CER", "CTH", "CTT", "precipitation", "file_count"]]
        .reset_index(drop=True)
    )

    union_frames = []
    for cluster_rank in [1, 2]:
        member_indices = (
            df.loc[df["cluster_rank"] == cluster_rank, "sample_index"]
            .drop_duplicates()
            .to_list()
        )
        subset = base_df.loc[base_df["sample_index"].isin(member_indices)].copy()
        subset["cluster_rank"] = cluster_rank
        union_frames.append(subset)

    return pd.concat(union_frames, ignore_index=True)


def summarize_cloud_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_rank in [1, 2]:
        subset = df.loc[df["cluster_rank"] == cluster_rank]
        for variable in ["COT", "CER", "CTH", "CTT"]:
            values = pd.to_numeric(subset[variable], errors="coerce").dropna()
            rows.append(
                {
                    "cluster_rank": cluster_rank,
                    "variable": variable,
                    "sample_count": len(values),
                    "min_value": float(values.min()) if not values.empty else np.nan,
                    "max_value": float(values.max()) if not values.empty else np.nan,
                    "median_value": float(values.median()) if not values.empty else np.nan,
                    "mean_value": float(values.mean()) if not values.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def classify_precip_intensity(precipitation: pd.Series) -> pd.Categorical:
    values = np.select(
        [
            (precipitation >= 0.0) & (precipitation <= 0.5),
            (precipitation > 0.5) & (precipitation <= 2.0),
            (precipitation > 2.0) & (precipitation <= 5.0),
            precipitation > 5.0,
        ],
        PRECIP_INTENSITY_LABELS,
        default="Unclassified",
    )
    return pd.Categorical(values, categories=PRECIP_INTENSITY_LABELS + ["Unclassified"], ordered=True)


def summarize_precip_intensity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    work_df["intensity_bin"] = classify_precip_intensity(pd.to_numeric(work_df["precipitation"], errors="coerce"))
    work_df = work_df.loc[work_df["intensity_bin"] != "Unclassified"].copy()

    rows = []
    for cluster_rank in [1, 2]:
        cluster_df = work_df.loc[work_df["cluster_rank"] == cluster_rank]
        total_count = len(cluster_df)
        for intensity_bin in PRECIP_INTENSITY_LABELS:
            count = int((cluster_df["intensity_bin"] == intensity_bin).sum())
            rows.append(
                {
                    "cluster_rank": cluster_rank,
                    "intensity_bin": intensity_bin,
                    "count": count,
                    "total_count": total_count,
                    "ratio": count / total_count if total_count else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["cluster_rank", "month", "hour"])["sample_index"]
        .nunique()
        .reset_index(name="sample_count")
    )

    full_index = pd.MultiIndex.from_product(
        [[1, 2], list(range(1, 13)), list(range(24))],
        names=["cluster_rank", "month", "hour"],
    )
    summary = (
        summary.set_index(["cluster_rank", "month", "hour"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    return summary


def plot_ridgeline(summary_df: pd.DataFrame, output_path: Path, title_suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    max_count = summary_df["sample_count"].max()
    scale = 0.82 / max(max_count, 1)

    for month in range(1, 13):
        baseline = month
        ax.hlines(baseline, 0, 23, color="0.7", linewidth=0.6, zorder=1)

        for cluster_rank in [1, 2]:
            subset = (
                summary_df.loc[(summary_df["month"] == month) & (summary_df["cluster_rank"] == cluster_rank)]
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
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels([MONTH_LABELS[m] for m in range(1, 13)], fontsize=11)
    ax.set_ylabel("Month", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.18)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CLUSTER_STYLES[1]["fill"], alpha=0.5, edgecolor="none", label=CLUSTER_STYLES[1]["label"]),
        plt.Rectangle((0, 0), 1, 1, facecolor=CLUSTER_STYLES[2]["fill"], alpha=0.5, edgecolor="none", label=CLUSTER_STYLES[2]["label"]),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=12)
    # fig.suptitle(f"Hourly ridgeline for union of top-2 clusters ({title_suffix})", fontsize=16)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.08, top=0.9)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_samples(args.input_csv)
    union_df = build_union_cluster_rows(df)
    summary_df = summarize_hourly_counts(union_df)
    stats_df = summarize_cloud_statistics(union_df)
    precip_ratio_df = summarize_precip_intensity_ratios(union_df)

    stem = args.input_csv.stem
    union_csv = args.output_dir / f"{stem}_top2_union_samples.csv"
    summary_output = args.output_dir / f"{stem}_top2_union_month_hour_counts.csv"
    stats_output = args.output_dir / f"{stem}_top2_union_cloud_stats.csv"
    precip_ratio_output = args.output_dir / f"{stem}_top2_union_precip_intensity_ratios.csv"
    figure_output = args.output_dir / f"{stem}_top2_union_hourly_ridgeline.png"

    union_df.to_csv(union_csv, index=False)
    summary_df.to_csv(summary_output, index=False)
    stats_df.to_csv(stats_output, index=False)
    precip_ratio_df.to_csv(precip_ratio_output, index=False)
    plot_ridgeline(summary_df, figure_output, stem)

    print(f"Input rows: {len(df)}")
    print(f"Union sample table: {union_csv}")
    print(f"Month-hour summary: {summary_output}")
    print(f"Cloud statistics: {stats_output}")
    print(f"Precip intensity ratios: {precip_ratio_output}")
    print(f"Figure: {figure_output}")


if __name__ == "__main__":
    main()
