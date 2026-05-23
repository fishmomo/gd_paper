from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_STATION_CSV = Path(r"H:\YeWu\Zhou\guangzhou\points_in_gd_shape.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output")
INTENSITY_LABELS = ["0-0.5", "0.5-2", "2-5", ">5"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot station maps of warm-cloud precipitation-intensity frequency.",
    )
    parser.add_argument(
        "--precip-csv",
        type=Path,
        default=DEFAULT_PRECIP_CSV,
        help="Date-hour-station warm-cloud precipitation statistics CSV.",
    )
    parser.add_argument(
        "--station-csv",
        type=Path,
        default=DEFAULT_STATION_CSV,
        help="Station metadata CSV with station, lon, lat, Alti columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for station frequency CSV and map PNG outputs.",
    )
    parser.add_argument(
        "--min-file-count",
        type=int,
        default=3,
        help="Minimum file_count required for a row to be included.",
    )
    return parser


def classify_intensity(precipitation: pd.Series) -> pd.Categorical:
    values = np.select(
        [
            (precipitation >= 0.0) & (precipitation <= 0.5),
            (precipitation > 0.5) & (precipitation <= 2.0),
            (precipitation > 2.0) & (precipitation <= 5.0),
            precipitation > 5.0,
        ],
        INTENSITY_LABELS,
        default="Unclassified",
    )
    return pd.Categorical(values, categories=INTENSITY_LABELS + ["Unclassified"], ordered=True)


def load_station_meta(station_csv: Path) -> pd.DataFrame:
    station_df = pd.read_csv(station_csv)
    required_columns = {"station", "lon", "lat", "Alti"}
    missing_columns = required_columns - set(station_df.columns)
    if missing_columns:
        raise KeyError(f"Station CSV is missing required columns: {sorted(missing_columns)}")

    station_df = station_df.rename(columns={"station": "station_id"})
    station_df["station_id"] = station_df["station_id"].astype(str)
    station_df["lon"] = pd.to_numeric(station_df["lon"], errors="coerce")
    station_df["lat"] = pd.to_numeric(station_df["lat"], errors="coerce")
    station_df["Alti"] = pd.to_numeric(station_df["Alti"], errors="coerce")
    return station_df.dropna(subset=["lon", "lat"]).copy()


def load_precip_rows(precip_csv: Path, min_file_count: int) -> pd.DataFrame:
    df = pd.read_csv(precip_csv)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_columns = {"station_id", "file_count", "precipitation"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Precip CSV is missing required columns: {sorted(missing_columns)}")

    df["station_id"] = df["station_id"].astype(str)
    df["file_count"] = pd.to_numeric(df["file_count"], errors="coerce")
    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
    df = df.loc[(df["file_count"] >= min_file_count) & (df["precipitation"] > 0)].copy()
    df["intensity_bin"] = classify_intensity(df["precipitation"])
    return df.loc[df["intensity_bin"] != "Unclassified"].copy()


def summarize_station_frequency(precip_df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        precip_df.groupby(["station_id", "intensity_bin"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=INTENSITY_LABELS, fill_value=0)
    )
    counts.columns = [f"{label}_count" for label in counts.columns]
    counts = counts.reset_index()
    counts["total_count"] = counts[[f"{label}_count" for label in INTENSITY_LABELS]].sum(axis=1)

    merged = station_df.merge(counts, on="station_id", how="left")
    for label in INTENSITY_LABELS:
        merged[f"{label}_count"] = merged[f"{label}_count"].fillna(0).astype(int)
    merged["total_count"] = merged["total_count"].fillna(0).astype(int)
    return merged.sort_values("station_id", kind="mergesort").reset_index(drop=True)


def scale_marker_size(counts: pd.Series) -> pd.Series:
    counts = counts.astype(float)
    positive = counts[counts > 0]
    if positive.empty:
        return pd.Series(np.zeros(len(counts)), index=counts.index)
    max_count = positive.max()
    return 25.0 + 475.0 * np.sqrt(counts / max_count)


def scale_altitude_marker_size(altitude: pd.Series) -> pd.Series:
    altitude = altitude.astype(float).clip(lower=0)
    positive = altitude[altitude > 0]
    if positive.empty:
        return pd.Series(np.full(len(altitude), 50.0), index=altitude.index)
    max_altitude = positive.max()
    return 40.0 + 460.0 * np.sqrt(altitude / max_altitude)


def plot_maps(summary_df: pd.DataFrame, output_path: Path, min_file_count: int) -> None:
    try:
        import cartopy.crs as ccrs
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("cartopy is required to draw the station maps.") from exc

    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )

    lon_min, lon_max = summary_df["lon"].min() - 0.5, summary_df["lon"].max() + 0.5
    lat_min, lat_max = summary_df["lat"].min() - 0.5, summary_df["lat"].max() + 0.5

    for ax, label in zip(axes.flat, INTENSITY_LABELS):
        count_col = f"{label}_count"
        sizes = scale_marker_size(summary_df[count_col])
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)
        ax.scatter(
            summary_df["lon"],
            summary_df["lat"],
            s=sizes,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            transform=projection,
        )
        ax.set_title(f"{label} mm/h", fontsize=14)
        ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.5)
        ax.text(
            0.02,
            0.02,
            f"stations={int((summary_df[count_col] > 0).sum())}\nmax n={int(summary_df[count_col].max())}",
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
            ha="left",
        )

    fig.suptitle(
        f"Station frequency by precipitation-intensity level (file_count >= {min_file_count})",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_altitude_map(station_df: pd.DataFrame, output_path: Path) -> None:
    try:
        import cartopy.crs as ccrs
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("cartopy is required to draw the station altitude map.") from exc

    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        figsize=(9, 7),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )

    lon_min, lon_max = station_df["lon"].min() - 0.5, station_df["lon"].max() + 0.5
    lat_min, lat_max = station_df["lat"].min() - 0.5, station_df["lat"].max() + 0.5
    sizes = scale_altitude_marker_size(station_df["Alti"])

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)
    ax.scatter(
        station_df["lon"],
        station_df["lat"],
        s=sizes,
        marker="^",
        facecolors="none",
        edgecolors="black",
        linewidths=0.9,
        transform=projection,
    )
    ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.5)
    ax.set_title("Station altitude distribution", fontsize=15)
    ax.text(
        0.02,
        0.02,
        f"stations={len(station_df)}\nmax Alti={station_df['Alti'].max():.1f} m",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_altitude_correlations(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in INTENSITY_LABELS:
        count_col = f"{label}_count"
        valid_df = summary_df[["Alti", count_col]].dropna()
        if len(valid_df) < 2 or valid_df["Alti"].nunique() < 2 or valid_df[count_col].nunique() < 2:
            correlation = np.nan
        else:
            correlation = valid_df["Alti"].corr(valid_df[count_col])
        rows.append(
            {
                "intensity_bin": label,
                "altitude_frequency_correlation": correlation,
                "abs_correlation": abs(correlation) if pd.notna(correlation) else np.nan,
                "station_count": len(valid_df),
            }
        )
    return pd.DataFrame(rows)


def plot_altitude_correlation(correlation_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    x = np.arange(len(correlation_df))
    correlations = correlation_df["altitude_frequency_correlation"].to_numpy(dtype=float)
    abs_correlations = correlation_df["abs_correlation"].fillna(0.0).to_numpy(dtype=float)
    sizes = 200.0 + 1800.0 * abs_correlations

    ax.scatter(
        x,
        np.zeros_like(x),
        s=sizes,
        marker="s",
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
    )
    for idx, corr in enumerate(correlations):
        label = "r=NA" if np.isnan(corr) else f"r={corr:.2f}"
        ax.text(idx, 0.18, label, ha="center", va="bottom", fontsize=12)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(correlation_df["intensity_bin"], fontsize=12)
    ax.set_yticks([])
    ax.set_ylim(-0.35, 0.45)
    ax.set_xlabel("Precipitation intensity (mm/h)", fontsize=13)
    ax.set_title("Correlation between station altitude and precipitation-intensity frequency", fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def scale_ratio_marker_size(ratios: pd.Series) -> pd.Series:
    ratios = ratios.astype(float).clip(lower=0.0, upper=1.0)
    return 20.0 + 480.0 * np.sqrt(ratios)


def plot_altitude_intensity_distribution_maps(
    summary_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    output_path: Path,
    min_file_count: int,
) -> None:
    try:
        import cartopy.crs as ccrs
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("cartopy is required to draw the altitude-intensity distribution maps.") from exc

    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )

    lon_min, lon_max = summary_df["lon"].min() - 0.5, summary_df["lon"].max() + 0.5
    lat_min, lat_max = summary_df["lat"].min() - 0.5, summary_df["lat"].max() + 0.5

    for ax, label in zip(axes.flat, INTENSITY_LABELS):
        count_col = f"{label}_count"
        ratios = summary_df[count_col] / summary_df["total_count"].replace(0, np.nan)
        sizes = scale_ratio_marker_size(ratios.fillna(0.0))
        corr_row = correlation_df.loc[correlation_df["intensity_bin"] == label]
        correlation = corr_row["altitude_frequency_correlation"].iloc[0] if not corr_row.empty else np.nan
        corr_label = "r=NA" if pd.isna(correlation) else f"r={correlation:.2f}"

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)
        ax.scatter(
            summary_df["lon"],
            summary_df["lat"],
            s=sizes,
            marker="s",
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            transform=projection,
        )
        ax.set_title(f"{label} mm/h ({corr_label})", fontsize=14)
        ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.5)
        ax.text(
            0.02,
            0.02,
            f"stations={int((summary_df[count_col] > 0).sum())}\nmax ratio={ratios.max(skipna=True):.2f}",
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
            ha="left",
        )

    fig.suptitle(
        f"Station distribution of altitude-related rain-intensity tendency (file_count >= {min_file_count})",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    station_df = load_station_meta(args.station_csv)
    precip_df = load_precip_rows(args.precip_csv, args.min_file_count)
    summary_df = summarize_station_frequency(precip_df, station_df)

    csv_output = args.output_dir / f"station_precip_intensity_frequency_filecount{args.min_file_count}.csv"
    figure_output = args.output_dir / f"station_precip_intensity_frequency_maps_filecount{args.min_file_count}.png"
    altitude_figure_output = args.output_dir / "station_altitude_distribution_map.png"
    correlation_csv_output = args.output_dir / f"station_altitude_precip_intensity_correlation_filecount{args.min_file_count}.csv"
    correlation_figure_output = args.output_dir / f"station_altitude_precip_intensity_correlation_filecount{args.min_file_count}.png"
    altitude_intensity_map_output = args.output_dir / f"station_altitude_precip_intensity_distribution_maps_filecount{args.min_file_count}.png"
    summary_df.to_csv(csv_output, index=False)

    correlation_df = summarize_altitude_correlations(summary_df)
    correlation_df.to_csv(correlation_csv_output, index=False)
    plot_altitude_correlation(correlation_df, correlation_figure_output)

    try:
        plot_maps(summary_df, figure_output, args.min_file_count)
        print(f"Map figure: {figure_output}")
        plot_altitude_map(station_df, altitude_figure_output)
        print(f"Altitude map figure: {altitude_figure_output}")
        plot_altitude_intensity_distribution_maps(
            summary_df,
            correlation_df,
            altitude_intensity_map_output,
            args.min_file_count,
        )
        print(f"Altitude-intensity distribution map figure: {altitude_intensity_map_output}")
    except ModuleNotFoundError as exc:
        print(f"[Skip map] {exc}")

    print(f"Rows after precipitation filter: {len(precip_df)}")
    print(f"Station frequency CSV: {csv_output}")
    print(f"Altitude correlation CSV: {correlation_csv_output}")
    print(f"Altitude correlation figure: {correlation_figure_output}")


if __name__ == "__main__":
    main()
