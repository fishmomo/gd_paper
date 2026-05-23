from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_CSV = Path(r"H:\YeWu\Zhou\guangzhou\output\fy4b_precip_pair_dbscan_top2_details_mean_filecount3.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize month-hour frequencies by x_var/y_var pair and cluster_label for DBSCAN top-2 detail samples.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input CSV such as fy4b_precip_pair_dbscan_top2_details_mean_filecount3.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for frequency summary outputs.",
    )
    return parser


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_columns = {"date", "hour", "x_var", "y_var", "cluster_label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["cluster_label"] = pd.to_numeric(df["cluster_label"], errors="coerce")
    df = df.dropna(subset=["date", "hour", "x_var", "y_var", "cluster_label"]).copy()

    df["hour"] = df["hour"].astype(int)
    df["cluster_label"] = df["cluster_label"].astype(int)
    df = df[df["cluster_label"].isin([0, 1])].copy()
    df["month"] = df["date"].dt.month
    df["pair_name"] = df["x_var"].astype(str) + "_" + df["y_var"].astype(str)
    return df


def summarize_long_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["pair_name", "x_var", "y_var", "cluster_label", "month", "hour"])
        .size()
        .reset_index(name="frequency")
        .sort_values(["pair_name", "cluster_label", "month", "hour"])
        .reset_index(drop=True)
    )
    return summary


def summarize_wide_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    wide = summary_df.pivot_table(
        index=["pair_name", "x_var", "y_var", "cluster_label", "month"],
        columns="hour",
        values="frequency",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    wide.columns = [
        f"hour_{int(col):02d}" if isinstance(col, (int, float)) else str(col)
        for col in wide.columns
    ]
    return wide.sort_values(["pair_name", "cluster_label", "month"]).reset_index(drop=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(args.input_csv)
    summary_long = summarize_long_table(df)
    summary_wide = summarize_wide_table(summary_long)

    input_stem = args.input_csv.stem
    long_csv = args.output_dir / f"{input_stem}_month_hour_frequency_long.csv"
    wide_csv = args.output_dir / f"{input_stem}_month_hour_frequency_wide.csv"

    summary_long.to_csv(long_csv, index=False)
    summary_wide.to_csv(wide_csv, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Long frequency table: {long_csv}")
    print(f"Wide frequency table: {wide_csv}")


if __name__ == "__main__":
    main()
