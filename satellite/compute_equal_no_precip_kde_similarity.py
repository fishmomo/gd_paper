from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


DEFAULT_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_NO_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_no_precip_date_hour_station_cloud_stats.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output\equal_no_precip_kde_similarity")
VARIABLES = ["COT", "CER", "CTH", "CTT"]
EPSILON = 1e-12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute pairwise similarity metrics between repeated no-precipitation KDE samples.",
    )
    parser.add_argument("--precip-csv", type=Path, default=DEFAULT_PRECIP_CSV)
    parser.add_argument("--no-precip-csv", type=Path, default=DEFAULT_NO_PRECIP_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-file-count", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--stats", nargs="+", choices=["mean", "max"], default=["mean", "max"])
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Fractional padding added to each variable range before KDE evaluation.",
    )
    return parser


def load_samples(csv_path: Path, sample_type: str, stat: str, min_file_count: int) -> pd.DataFrame:
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
    df["sample_type"] = sample_type

    for variable in VARIABLES:
        df[variable] = pd.to_numeric(df[f"{variable}_{stat}"], errors="coerce")

    df["CTH"] = df["CTH"] / 1000.0
    df["CTT"] = df["CTT"] - 273.15
    return df.dropna(subset=VARIABLES).reset_index(drop=True)


def build_no_precip_draws(no_precip_df: pd.DataFrame, sample_size: int, random_state: int) -> list[pd.DataFrame]:
    shuffled = no_precip_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    draw_count = len(shuffled) // sample_size
    return [
        shuffled.iloc[draw_index * sample_size : (draw_index + 1) * sample_size].reset_index(drop=True)
        for draw_index in range(draw_count)
    ]


def build_grid(full_df: pd.DataFrame, x_var: str, y_var: str, grid_size: int, padding: float) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = full_df[x_var].min(), full_df[x_var].max()
    y_min, y_max = full_df[y_var].min(), full_df[y_var].max()
    x_pad = (x_max - x_min) * padding or 1.0
    y_pad = (y_max - y_min) * padding or 1.0
    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, grid_size)
    y_grid = np.linspace(y_min - y_pad, y_max + y_pad, grid_size)
    return x_grid, y_grid


def kde_probability_grid(df: pd.DataFrame, x_var: str, y_var: str, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x_grid, y_grid)
    values = np.vstack([df[x_var].to_numpy(), df[y_var].to_numpy()])
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = gaussian_kde(values)(positions).reshape(xx.shape)
    density = np.clip(density, 0.0, None)
    total = density.sum()
    if total <= 0:
        raise ValueError(f"KDE density sum is not positive for {x_var}-{y_var}.")
    return density / total


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.ravel(p).astype(float) + EPSILON
    q = np.ravel(q).astype(float) + EPSILON
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.ravel(p).astype(float)
    q = np.ravel(q).astype(float)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def cosine_similarity(p: np.ndarray, q: np.ndarray) -> float:
    p = np.ravel(p).astype(float)
    q = np.ravel(q).astype(float)
    denominator = np.linalg.norm(p) * np.linalg.norm(q)
    return float(np.dot(p, q) / denominator) if denominator else np.nan


def plot_metric_matrix(matrix: np.ndarray, metric_name: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Draw", fontsize=13)
    ax.set_ylabel("Draw", fontsize=13)
    ticks = np.arange(matrix.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(i + 1) for i in ticks], fontsize=8)
    ax.set_yticklabels([str(i + 1) for i in ticks], fontsize=8)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(metric_name, fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_metric(values: pd.Series, lower_is_more_similar: bool) -> dict[str, float]:
    result = {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    result["most_similar"] = result["min"] if lower_is_more_similar else result["max"]
    result["least_similar"] = result["max"] if lower_is_more_similar else result["min"]
    return result


def process_stat(args: argparse.Namespace, stat: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    precip_df = load_samples(args.precip_csv, "precip", stat, args.min_file_count)
    no_precip_df = load_samples(args.no_precip_csv, "no_precip", stat, args.min_file_count)
    draws = build_no_precip_draws(no_precip_df, len(precip_df), args.random_state)

    stat_dir = args.output_dir / stat
    grids_dir = stat_dir / "kde_grids"
    matrices_dir = stat_dir / "metric_matrices"
    figures_dir = stat_dir / "metric_heatmaps"
    grids_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pairwise_rows = []
    summary_rows = []

    for x_var, y_var in combinations(VARIABLES, 2):
        pair_name = f"{x_var}_{y_var}"
        x_grid, y_grid = build_grid(no_precip_df, x_var, y_var, args.grid_size, args.padding)
        kde_grids = np.stack([
            kde_probability_grid(draw_df, x_var, y_var, x_grid, y_grid)
            for draw_df in draws
        ])
        np.savez_compressed(
            grids_dir / f"kde_grids_{pair_name}_{stat}_filecount{args.min_file_count}.npz",
            kde_grids=kde_grids,
            x_grid=x_grid,
            y_grid=y_grid,
            stat=np.array(stat),
            x_var=np.array(x_var),
            y_var=np.array(y_var),
            sample_size=np.array(len(precip_df)),
            draw_count=np.array(len(draws)),
        )

        draw_count = len(draws)
        jsd_matrix = np.zeros((draw_count, draw_count), dtype=float)
        hellinger_matrix = np.zeros((draw_count, draw_count), dtype=float)
        cosine_matrix = np.ones((draw_count, draw_count), dtype=float)

        for i in range(draw_count):
            for j in range(i + 1, draw_count):
                p = kde_grids[i]
                q = kde_grids[j]
                jsd = jensen_shannon_divergence(p, q)
                hellinger = hellinger_distance(p, q)
                cosine = cosine_similarity(p, q)
                jsd_matrix[i, j] = jsd_matrix[j, i] = jsd
                hellinger_matrix[i, j] = hellinger_matrix[j, i] = hellinger
                cosine_matrix[i, j] = cosine_matrix[j, i] = cosine
                pairwise_rows.append(
                    {
                        "stat": stat,
                        "x_var": x_var,
                        "y_var": y_var,
                        "pair_name": pair_name,
                        "draw_i": i + 1,
                        "draw_j": j + 1,
                        "jsd": jsd,
                        "hellinger": hellinger,
                        "cosine": cosine,
                    }
                )

        np.savez_compressed(
            matrices_dir / f"metric_matrices_{pair_name}_{stat}_filecount{args.min_file_count}.npz",
            jsd=jsd_matrix,
            hellinger=hellinger_matrix,
            cosine=cosine_matrix,
        )
        plot_metric_matrix(jsd_matrix, "JSD", f"{pair_name} JSD ({stat})", figures_dir / f"jsd_{pair_name}_{stat}.png")
        plot_metric_matrix(
            hellinger_matrix,
            "Hellinger distance",
            f"{pair_name} Hellinger ({stat})",
            figures_dir / f"hellinger_{pair_name}_{stat}.png",
        )
        plot_metric_matrix(cosine_matrix, "Cosine similarity", f"{pair_name} Cosine ({stat})", figures_dir / f"cosine_{pair_name}_{stat}.png")

        pairwise_df = pd.DataFrame([row for row in pairwise_rows if row["stat"] == stat and row["pair_name"] == pair_name])
        for metric, lower_is_more_similar in [("jsd", True), ("hellinger", True), ("cosine", False)]:
            metric_summary = summarize_metric(pairwise_df[metric], lower_is_more_similar)
            if lower_is_more_similar:
                most_idx = pairwise_df[metric].idxmin()
                least_idx = pairwise_df[metric].idxmax()
            else:
                most_idx = pairwise_df[metric].idxmax()
                least_idx = pairwise_df[metric].idxmin()
            summary_rows.append(
                {
                    "stat": stat,
                    "x_var": x_var,
                    "y_var": y_var,
                    "pair_name": pair_name,
                    "metric": metric,
                    "draw_count": draw_count,
                    "sample_size_per_draw": len(precip_df),
                    "pairwise_count": len(pairwise_df),
                    "mean": metric_summary["mean"],
                    "median": metric_summary["median"],
                    "std": metric_summary["std"],
                    "min": metric_summary["min"],
                    "max": metric_summary["max"],
                    "most_similar_value": metric_summary["most_similar"],
                    "least_similar_value": metric_summary["least_similar"],
                    "most_similar_draw_i": int(pairwise_df.loc[most_idx, "draw_i"]),
                    "most_similar_draw_j": int(pairwise_df.loc[most_idx, "draw_j"]),
                    "least_similar_draw_i": int(pairwise_df.loc[least_idx, "draw_i"]),
                    "least_similar_draw_j": int(pairwise_df.loc[least_idx, "draw_j"]),
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    summary_df = pd.DataFrame(summary_rows)
    pairwise_df.to_csv(stat_dir / f"kde_similarity_pairwise_{stat}_filecount{args.min_file_count}.csv", index=False)
    summary_df.to_csv(stat_dir / f"kde_similarity_summary_{stat}_filecount{args.min_file_count}.csv", index=False)
    return pairwise_df, summary_df


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_pairwise = []
    all_summary = []
    for stat in args.stats:
        pairwise_df, summary_df = process_stat(args, stat)
        all_pairwise.append(pairwise_df)
        all_summary.append(summary_df)
        print(f"[{stat}] Pairwise rows: {len(pairwise_df)}")
        print(f"[{stat}] Summary rows: {len(summary_df)}")

    combined_pairwise = pd.concat(all_pairwise, ignore_index=True) if all_pairwise else pd.DataFrame()
    combined_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    combined_pairwise.to_csv(args.output_dir / f"kde_similarity_pairwise_filecount{args.min_file_count}.csv", index=False)
    combined_summary.to_csv(args.output_dir / f"kde_similarity_summary_filecount{args.min_file_count}.csv", index=False)
    print(f"Output directory: {args.output_dir}")
    print(f"Summary CSV: {args.output_dir / f'kde_similarity_summary_filecount{args.min_file_count}.csv'}")
    print(f"Pairwise CSV: {args.output_dir / f'kde_similarity_pairwise_filecount{args.min_file_count}.csv'}")


if __name__ == "__main__":
    main()
