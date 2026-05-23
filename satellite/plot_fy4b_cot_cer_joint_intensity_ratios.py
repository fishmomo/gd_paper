from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\邢台观测站\CWR_project\guangzhou\satellite\outputs\cer_cot_joint_intensity_ratio")
COT_BINS = ["0-10", "11-20", "21-30", "31-40", ">40"]
CER_BINS = ["0-10", "11-20", "21-30", "31-40", ">40"]
INTENSITY_BINS = ["0-0.5", "0.5-2", "2-5", ">5"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize conditional rain-intensity ratios for joint COT-CER bins.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Date-hour-station warm-cloud precipitation statistics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for joint-bin summary outputs.",
    )
    parser.add_argument(
        "--min-file-count",
        type=int,
        default=3,
        help="Minimum file_count required for a row to be included.",
    )
    return parser


def classify_series(series: pd.Series, conditions: list[pd.Series], labels: list[str]) -> pd.Categorical:
    values = np.select(conditions, labels, default="Unclassified")
    return pd.Categorical(values, categories=labels + ["Unclassified"], ordered=True)


def load_samples(input_csv: Path, stat: str, min_file_count: int) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_columns = {"file_count", "precipitation", f"COT_{stat}", f"CER_{stat}"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Input CSV is missing required columns: {sorted(missing_columns)}")

    df["file_count"] = pd.to_numeric(df["file_count"], errors="coerce")
    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
    df["COT"] = pd.to_numeric(df[f"COT_{stat}"], errors="coerce")
    df["CER"] = pd.to_numeric(df[f"CER_{stat}"], errors="coerce")

    df = df.loc[(df["file_count"] >= min_file_count) & (df["precipitation"] > 0)].copy()
    return df.dropna(subset=["precipitation", "COT", "CER"]).reset_index(drop=True)


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["COT_bin"] = classify_series(
        df["COT"],
        [
            (df["COT"] >= 0.0) & (df["COT"] <= 10.0),
            (df["COT"] > 10.0) & (df["COT"] <= 20.0),
            (df["COT"] > 20.0) & (df["COT"] <= 30.0),
            (df["COT"] > 30.0) & (df["COT"] <= 40.0),
            df["COT"] > 40.0,
        ],
        COT_BINS,
    )
    df["CER_bin"] = classify_series(
        df["CER"],
        [
            (df["CER"] >= 0.0) & (df["CER"] <= 10.0),
            (df["CER"] > 10.0) & (df["CER"] <= 20.0),
            (df["CER"] > 20.0) & (df["CER"] <= 30.0),
            (df["CER"] > 30.0) & (df["CER"] <= 40.0),
            df["CER"] > 40.0,
        ],
        CER_BINS,
    )
    df["intensity_bin"] = classify_series(
        df["precipitation"],
        [
            (df["precipitation"] >= 0.0) & (df["precipitation"] <= 0.5),
            (df["precipitation"] > 0.5) & (df["precipitation"] <= 2.0),
            (df["precipitation"] > 2.0) & (df["precipitation"] <= 5.0),
            df["precipitation"] > 5.0,
        ],
        INTENSITY_BINS,
    )
    return df.loc[
        (df["COT_bin"] != "Unclassified")
        & (df["CER_bin"] != "Unclassified")
        & (df["intensity_bin"] != "Unclassified")
    ].copy()


def summarize_joint_ratios(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cot_bin in COT_BINS:
        for cer_bin in CER_BINS:
            subset = df.loc[(df["COT_bin"] == cot_bin) & (df["CER_bin"] == cer_bin)]
            joint_bin_count = len(subset)
            for intensity_bin in INTENSITY_BINS:
                count = int((subset["intensity_bin"] == intensity_bin).sum())
                rows.append(
                    {
                        "COT_bin": cot_bin,
                        "CER_bin": cer_bin,
                        "intensity_bin": intensity_bin,
                        "count": count,
                        "joint_bin_count": joint_bin_count,
                        "ratio": count / joint_bin_count if joint_bin_count else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def process_stat(args: argparse.Namespace, stat: str) -> Path:
    df = add_bins(load_samples(args.input_csv, stat, args.min_file_count))
    summary_df = summarize_joint_ratios(df)
    output_path = args.output_dir / f"fy4b_cot_cer_joint_rain_intensity_ratio_{stat}_filecount{args.min_file_count}.csv"
    summary_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for stat in ["mean", "max"]:
        output_path = process_stat(args, stat)
        print(f"[{stat}] Joint summary: {output_path}")


if __name__ == "__main__":
    main()
