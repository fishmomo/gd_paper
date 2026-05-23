from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427.csv")
DEFAULT_OUTPUT_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_positive_by_station.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter FY4B CSV rows with precipitation > 0 and sort by station within each file/hour.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input CSV produced by read_FY4B.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Filtered output CSV path.",
    )
    return parser


def get_filename_column(df: pd.DataFrame) -> str:
    for column in ("file_name", "filename"):
        if column in df.columns:
            return column
    raise KeyError("Input CSV must contain either 'file_name' or 'filename'.")


def extract_hour_from_filename(filename: object) -> str | None:
    match = re.search(r"\d{8}(\d{2})\d{4}", str(filename))
    if match is None:
        return None
    return match.group(1)


def build_hour_key(df: pd.DataFrame, filename_column: str) -> pd.Series:
    if "time" in df.columns:
        time_hour = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce").dt.strftime("%H")
        filename_hour = df[filename_column].map(extract_hour_from_filename)
        return time_hour.fillna(filename_hour)
    return df[filename_column].map(extract_hour_from_filename)


def filter_and_sort(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    filename_column = get_filename_column(df)
    required_columns = {filename_column, "station_id", "precipitation"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Input CSV is missing required columns: {sorted(missing_columns)}")

    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
    df["_hour"] = build_hour_key(df, filename_column)

    filtered = df.loc[df["precipitation"] > 0].copy()
    filtered = filtered.sort_values(
        by=[filename_column, "_hour", "station_id"],
        kind="mergesort",
    )
    return filtered.drop(columns=["_hour"]).reset_index(drop=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_df = filter_and_sort(args.input_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Filtered rows: {len(output_df)}")


if __name__ == "__main__":
    main()
