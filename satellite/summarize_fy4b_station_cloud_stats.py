from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_positive_by_station.csv")
DEFAULT_OUTPUT_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
CLOUD_COLUMNS = ["COT", "CTH", "CER", "CTT"]
INVALID_VALUES = {-999, 65531, 65532}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize FY4B cloud parameters by station for positive-precipitation rows.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input CSV produced by filter_fy4b_positive_precip.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Station cloud-parameter statistics output CSV path.",
    )
    parser.add_argument(
        "--precip-mode",
        choices=["positive", "zero"],
        default="positive",
        help="Use positive precipitation rows or zero-precipitation rows.",
    )
    parser.add_argument(
        "--min-file-count",
        type=int,
        default=3,
        help="Minimum file_count required when summarizing zero-precipitation rows.",
    )
    return parser


def get_filename_column(df: pd.DataFrame) -> str:
    for column in ("file_name", "filename"):
        if column in df.columns:
            return column
    raise KeyError("Input CSV must contain either 'file_name' or 'filename'.")


def extract_timestamp_from_filename(filename: object) -> str | None:
    match = re.search(r"\d{14}", str(filename))
    if match is None:
        return None
    return match.group(0)


def extract_date_from_filename(filename: object) -> str | None:
    timestamp = extract_timestamp_from_filename(filename)
    if timestamp is None:
        return None
    return f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"


def extract_hour_from_filename(filename: object) -> str | None:
    timestamp = extract_timestamp_from_filename(filename)
    if timestamp is None:
        return None
    return timestamp[8:10]


def build_hour_key(df: pd.DataFrame, filename_column: str) -> pd.Series:
    if "time" in df.columns:
        time_hour = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce").dt.strftime("%H")
        filename_hour = df[filename_column].map(extract_hour_from_filename)
        return time_hour.fillna(filename_hour)
    return df[filename_column].map(extract_hour_from_filename)


def load_precip_rows(input_csv: Path, precip_mode: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(input_csv)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    filename_column = get_filename_column(df)
    required_columns = {filename_column, "station_id", "precipitation", *CLOUD_COLUMNS}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Input CSV is missing required columns: {sorted(missing_columns)}")

    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
    if precip_mode == "positive":
        df = df.loc[df["precipitation"] > 0].copy()
    elif precip_mode == "zero":
        df = df.loc[df["precipitation"] == 0].copy()
    else:
        raise ValueError(f"Unsupported precip mode: {precip_mode}")

    df["_date"] = df[filename_column].map(extract_date_from_filename)
    df["_hour"] = build_hour_key(df, filename_column)
    df["_file"] = df[filename_column].astype(str)

    for column in CLOUD_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df, filename_column


def build_valid_stat_rows(df: pd.DataFrame, filename_column: str) -> pd.DataFrame:
    group_columns = ["_date", "_hour", "station_id"]
    grouped = df.groupby(group_columns, dropna=False, sort=False)
    group_size = grouped["station_id"].transform("size")
    invalid_mask = df[CLOUD_COLUMNS].isin(INVALID_VALUES) | df[CLOUD_COLUMNS].isna()
    group_all_invalid = invalid_mask.groupby(
        [df[column] for column in group_columns],
        dropna=False,
        sort=False,
    ).transform("all")

    single_row_skip = (group_size == 1) & invalid_mask.any(axis=1)
    multi_row_skip = (group_size > 1) & group_all_invalid.any(axis=1)
    keep_mask = ~(single_row_skip | multi_row_skip)

    valid_df = df.loc[keep_mask].copy()
    valid_df[CLOUD_COLUMNS] = valid_df[CLOUD_COLUMNS].mask(invalid_mask.loc[keep_mask])
    return valid_df.reset_index(drop=True)


def summarize_by_date_hour_station(input_csv: Path, precip_mode: str, min_file_count: int) -> pd.DataFrame:
    df, filename_column = load_precip_rows(input_csv, precip_mode)
    df = build_valid_stat_rows(df, filename_column)

    agg_dict = {
        "file_count": pd.NamedAgg(column="_file", aggfunc="nunique"),
        "precipitation": pd.NamedAgg(column="precipitation", aggfunc="mean"),
    }
    for column in CLOUD_COLUMNS:
        agg_dict[f"{column}_max"] = pd.NamedAgg(column=column, aggfunc="max")
        agg_dict[f"{column}_mean"] = pd.NamedAgg(column=column, aggfunc="mean")

    summary_df = df.groupby(["_date", "_hour", "station_id"], as_index=False).agg(**agg_dict)
    summary_df = summary_df.rename(columns={"_date": "date", "_hour": "hour"})
    if precip_mode == "zero":
        summary_df = summary_df.loc[summary_df["file_count"] >= min_file_count].copy()
    return summary_df.sort_values(by=["date", "hour", "station_id"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary_df = summarize_by_date_hour_station(args.input_csv, args.precip_mode, args.min_file_count)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_csv, index=False)

    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Precip mode: {args.precip_mode}")
    if args.precip_mode == "zero":
        print(f"Minimum file_count: {args.min_file_count}")
    print(f"Date-hour-station rows: {len(summary_df)}")


if __name__ == "__main__":
    main()
