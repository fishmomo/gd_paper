from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


DEFAULT_PRECIP_CSV = Path(r"H:\YeWu\Zhou\guangzhou\warmcloud_allprecip_0427_date_hour_station_cloud_stats.csv")
DEFAULT_OUTPUT_DIR = Path(r"H:\YeWu\Zhou\guangzhou\output")
PAIR_CONFIGS = [
    ("CTH", "COT"),
    ("CTH", "CER"),
    ("CTT", "CTH"),
]
AXIS_LABELS = {
    "COT": "Cloud optical thickness",
    "CER": "Cloud effective radius",
    "CTH": "Cloud top height (km)",
    "CTT": "Cloud top temperature (C)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index precip samples, cluster selected FY4B cloud-parameter pairs with DBSCAN, and compare label consistency.",
    )
    parser.add_argument(
        "--precip-csv",
        type=Path,
        default=DEFAULT_PRECIP_CSV,
        help="Precipitation sample CSV with date/hour/station cloud statistics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for cluster outputs.",
    )
    parser.add_argument(
        "--min-file-count",
        type=int,
        default=3,
        help="Minimum file_count required for a row to be included.",
    )
    parser.add_argument(
        "--stat",
        choices=["mean", "max"],
        default="mean",
        help="Cloud-parameter statistic to cluster.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.16,
        help="DBSCAN eps on standardized feature space.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=15,
        help="DBSCAN min_samples.",
    )
    parser.add_argument(
        "--annotate-indices",
        action="store_true",
        help="Annotate scatter points with sample_index on the figure.",
    )
    return parser


def load_precip_samples(csv_path: Path, stat: str, min_file_count: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_columns = {"date", "hour", "station_id", "precipitation", "file_count"}
    required_columns |= {f"{var}_{stat}" for var in ["COT", "CER", "CTH", "CTT"]}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

    df["file_count"] = pd.to_numeric(df["file_count"], errors="coerce")
    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
    df = df.loc[(df["file_count"] >= min_file_count) & df["precipitation"].notna()].copy()

    for variable in ["COT", "CER", "CTH", "CTT"]:
        df[variable] = pd.to_numeric(df[f"{variable}_{stat}"], errors="coerce")

    df["CTH"] = df["CTH"] / 1000.0
    df["CTT"] = df["CTT"] - 273.15
    df = df.dropna(subset=["COT", "CER", "CTH", "CTT"]).reset_index(drop=True)
    df["sample_index"] = np.arange(len(df))
    return df


def run_pair_dbscan(df: pd.DataFrame, x_var: str, y_var: str, eps: float, min_samples: int) -> np.ndarray:
    features = df[[x_var, y_var]].to_numpy()
    scaled = StandardScaler().fit_transform(features)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(scaled)
    return labels


def align_labels_to_reference(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    ref_unique = np.unique(reference)
    tgt_unique = np.unique(target)
    confusion = np.zeros((len(ref_unique), len(tgt_unique)), dtype=int)

    for i, ref_label in enumerate(ref_unique):
        for j, tgt_label in enumerate(tgt_unique):
            confusion[i, j] = np.sum((reference == ref_label) & (target == tgt_label))

    row_ind, col_ind = linear_sum_assignment(confusion.max() - confusion)
    label_map = {tgt_unique[col]: ref_unique[row] for row, col in zip(row_ind, col_ind)}
    next_new_label = int(max(ref_unique.max(initial=-1), tgt_unique.max(initial=-1)) + 1)
    aligned = np.empty_like(target)
    for idx, label in enumerate(target):
        if label in label_map:
            aligned[idx] = label_map[label]
        else:
            aligned[idx] = next_new_label
            next_new_label += 1
    return aligned


def cluster_overlap_metrics(reference: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    aligned = align_labels_to_reference(reference, target)
    same_ratio = float(np.mean(reference == aligned))
    ari = float(adjusted_rand_score(reference, target))
    ref_cluster_count = int(len(set(reference)) - (1 if -1 in reference else 0))
    tgt_cluster_count = int(len(set(target)) - (1 if -1 in target else 0))
    ref_noise_ratio = float(np.mean(reference == -1))
    tgt_noise_ratio = float(np.mean(target == -1))
    return {
        "same_label_ratio": same_ratio,
        "adjusted_rand_score": ari,
        "pair_a_cluster_count": ref_cluster_count,
        "pair_b_cluster_count": tgt_cluster_count,
        "pair_a_noise_ratio": ref_noise_ratio,
        "pair_b_noise_ratio": tgt_noise_ratio,
    }


def build_cluster_table(df: pd.DataFrame, pair_labels: dict[str, np.ndarray]) -> pd.DataFrame:
    table = df[
        ["sample_index", "date", "hour", "station_id", "precipitation", "file_count", "COT", "CER", "CTH", "CTT"]
    ].copy()
    for pair_name, labels in pair_labels.items():
        table[f"{pair_name}_cluster"] = labels
    return table


def get_top_two_clusters(labels: np.ndarray) -> list[int]:
    valid_labels = [label for label in np.unique(labels) if label != -1]
    ranked = sorted(valid_labels, key=lambda label: np.sum(labels == label), reverse=True)
    return ranked[:2]


def calculate_set_overlap_metrics(indices_a: set[int], indices_b: set[int]) -> dict[str, float | int]:
    intersection = indices_a & indices_b
    union = indices_a | indices_b
    overlap_base = min(len(indices_a), len(indices_b))
    return {
        "size_a": len(indices_a),
        "size_b": len(indices_b),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "repeat_ratio": float(len(intersection) / overlap_base) if overlap_base > 0 else np.nan,
        "jaccard_ratio": float(len(intersection) / len(union)) if len(union) > 0 else np.nan,
    }


def build_top_cluster_detail_table(df: pd.DataFrame, pair_labels: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for x_var, y_var in PAIR_CONFIGS:
        pair_name = f"{x_var}_{y_var}".lower()
        labels = pair_labels[pair_name]
        top_labels = get_top_two_clusters(labels)
        for rank, cluster_label in enumerate(top_labels, start=1):
            mask = labels == cluster_label
            subset = df.loc[
                mask,
                ["sample_index", "date", "hour", "station_id", "precipitation", "file_count", "COT", "CER", "CTH", "CTT"],
            ].copy()
            subset["pair_name"] = pair_name
            subset["x_var"] = x_var
            subset["y_var"] = y_var
            subset["x_value"] = df.loc[mask, x_var].values
            subset["y_value"] = df.loc[mask, y_var].values
            subset["cluster_label"] = cluster_label
            subset["cluster_rank"] = rank
            subset["cluster_size"] = int(mask.sum())
            rows.append(subset)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_top_cluster_similarity_table(pair_labels: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    top_cluster_members: dict[str, dict[int, tuple[int, set[int]]]] = {}

    for pair_name, labels in pair_labels.items():
        top_labels = get_top_two_clusters(labels)
        rank_map: dict[int, tuple[int, set[int]]] = {}
        for rank, cluster_label in enumerate(top_labels, start=1):
            member_indices = set(np.where(labels == cluster_label)[0].tolist())
            rank_map[rank] = (cluster_label, member_indices)
        top_cluster_members[pair_name] = rank_map

    for rank in [1, 2]:
        for pair_a, pair_b in combinations(pair_labels.keys(), 2):
            if rank not in top_cluster_members[pair_a] or rank not in top_cluster_members[pair_b]:
                continue
            label_a, indices_a = top_cluster_members[pair_a][rank]
            label_b, indices_b = top_cluster_members[pair_b][rank]
            metrics = calculate_set_overlap_metrics(indices_a, indices_b)
            rows.append(
                {
                    "cluster_rank": rank,
                    "pair_a": pair_a,
                    "pair_b": pair_b,
                    "pair_a_label": label_a,
                    "pair_b_label": label_b,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def plot_cluster_pairs(
    df: pd.DataFrame,
    pair_labels: dict[str, np.ndarray],
    output_path: Path,
    stat: str,
    min_file_count: int,
    annotate_indices: bool,
    eps: float,
    min_samples: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    for ax, (x_var, y_var) in zip(axes, PAIR_CONFIGS):
        pair_name = f"{x_var}_{y_var}".lower()
        labels = pair_labels[pair_name]
        colors = ["#bdbdbd" if label == -1 else to_hex(cmap(int(label) % 10)) for label in labels]
        ax.scatter(
            df[x_var],
            df[y_var],
            color=colors,
            s=28,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.25,
        )
        if annotate_indices:
            for _, row in df.iterrows():
                ax.text(
                    row[x_var],
                    row[y_var],
                    str(int(row["sample_index"])),
                    fontsize=5.5,
                    alpha=0.75,
                )
        ax.set_xlabel(AXIS_LABELS[x_var], fontsize=11)
        ax.set_ylabel(AXIS_LABELS[y_var], fontsize=11)
        cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.mean(labels == -1)
        top_labels = get_top_two_clusters(labels)
        top_sizes = [int(np.sum(labels == label)) for label in top_labels]
        top_summary = ", ".join([f"L{label}:{size}" for label, size in zip(top_labels, top_sizes)]) if top_labels else "no cluster"
        ax.set_title(
            f"{y_var} vs {x_var}\nclusters={cluster_count}, noise={noise_ratio:.2f}, top2={top_summary}",
            fontsize=12,
        )
        ax.grid(linestyle="--", alpha=0.25)

    fig.suptitle(
        (
            f"Precip pair DBSCAN clustering ({stat}, file_count >= {min_file_count}, n={len(df)}, "
            f"eps={eps}, min_samples={min_samples})"
        ),
        fontsize=15,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    precip_df = load_precip_samples(args.precip_csv, args.stat, args.min_file_count)

    pair_labels: dict[str, np.ndarray] = {}
    for x_var, y_var in PAIR_CONFIGS:
        pair_name = f"{x_var}_{y_var}".lower()
        labels = run_pair_dbscan(precip_df, x_var, y_var, args.eps, args.min_samples)
        pair_labels[pair_name] = labels

    cluster_table = build_cluster_table(precip_df, pair_labels)
    similarity_table = build_top_cluster_similarity_table(pair_labels)
    top_cluster_detail_table = build_top_cluster_detail_table(precip_df, pair_labels)

    cluster_csv = args.output_dir / f"fy4b_precip_pair_dbscan_{args.stat}_filecount{args.min_file_count}.csv"
    similarity_csv = args.output_dir / f"fy4b_precip_pair_dbscan_top2_similarity_{args.stat}_filecount{args.min_file_count}.csv"
    top2_detail_csv = args.output_dir / f"fy4b_precip_pair_dbscan_top2_details_{args.stat}_filecount{args.min_file_count}.csv"
    figure_path = args.output_dir / f"fy4b_precip_pair_dbscan_{args.stat}_filecount{args.min_file_count}.png"

    cluster_table.to_csv(cluster_csv, index=False)
    similarity_table.to_csv(similarity_csv, index=False)
    top_cluster_detail_table.to_csv(top2_detail_csv, index=False)
    plot_cluster_pairs(
        precip_df,
        pair_labels,
        figure_path,
        args.stat,
        args.min_file_count,
        args.annotate_indices,
        args.eps,
        args.min_samples,
    )

    print(f"Samples: {len(precip_df)}")
    print(f"Cluster labels: {cluster_csv}")
    print(f"Similarity table: {similarity_csv}")
    print(f"Top-2 cluster details: {top2_detail_csv}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
