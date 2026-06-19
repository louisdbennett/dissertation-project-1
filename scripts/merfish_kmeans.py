import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from merfish_utils import FEATURES, prepare_filtered_data

MIN_CLASS_COUNT = 500

OUTPUT_DIR = Path("analysis_outputs/merfish_kmeans")
DEFAULT_K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9]
DEFAULT_N_GENE_PCS = 150
DEFAULT_PC_SENSITIVITY_VALUES = [0, 3, 5, 10, 20, 30, 50, 100, 150, 200]
MIN_CLUSTER_SUPERTYPES = 3
KMEANS_RANDOM_STATE = 0


def cluster_supertype_centroids(
    min_count: int = MIN_CLASS_COUNT,
    k_values=None,
    n_gene_pcs: int = DEFAULT_N_GENE_PCS,
) -> dict[str, object]:
    """Cluster supertype centroids on the MERFISH dataset."""
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    prepared = prepare_filtered_data(
        target="supertype",
        min_count=min_count,
        class_name=None,
        filter_to_snr_z_range=True,
    )
    df = prepared["data"].copy()
    pc_columns = [f"expr_pc_{i + 1}" for i in range(n_gene_pcs) if f"expr_pc_{i + 1}" in df.columns]
    df = df[["supertype", "x_ccf", "y_ccf", "z_ccf", *pc_columns]].copy()

    centroids = (
        df.groupby("supertype", as_index=False)
        .agg(
            n_cells=("supertype", "size"),
            x_ccf=("x_ccf", "mean"),
            y_ccf=("y_ccf", "mean"),
            z_ccf=("z_ccf", "mean"),
        )
        .sort_values("n_cells", ascending=False)
        .reset_index(drop=True)
    )
    if pc_columns:
        pc_means = df.groupby("supertype", as_index=False)[pc_columns].mean()
        centroids = centroids.merge(pc_means, on="supertype", how="left")

    scaled_centroids = StandardScaler().fit_transform(centroids[FEATURES + pc_columns])
    valid_k_values = [k for k in k_values if 2 <= k < len(centroids)]

    rows = []
    best_score = None
    best_labels = None

    for k in valid_k_values:
        # cluster the supertype centroids and score how well separated the groups look
        labels = KMeans(n_clusters=k, n_init=20, random_state=KMEANS_RANDOM_STATE).fit_predict(scaled_centroids)
        cluster_sizes = pd.Series(labels).value_counts()
        min_cluster_size = int(cluster_sizes.min())
        score = silhouette_score(scaled_centroids, labels)
        is_valid = min_cluster_size >= MIN_CLUSTER_SUPERTYPES
        rows.append({"k": k, "silhouette_score": score, "min_cluster_size": min_cluster_size, "is_valid": is_valid})

        if is_valid and (best_score is None or score > best_score):
            best_score = score
            best_labels = labels

    if best_labels is None:
        best_row = pd.DataFrame(rows).sort_values("silhouette_score", ascending=False).iloc[0]
        best_k = int(best_row["k"])
        best_labels = KMeans(
            n_clusters=best_k,
            n_init=20,
            random_state=KMEANS_RANDOM_STATE,
        ).fit_predict(scaled_centroids)
    else:
        best_k = int(pd.DataFrame(rows).query("is_valid").sort_values("silhouette_score", ascending=False).iloc[0]["k"])

    centroids["centroid_cluster"] = best_labels
    k_summary = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    k_summary.to_csv(OUTPUT_DIR / "k_summary.csv", index=False)
    centroids.to_csv(OUTPUT_DIR / "centroid_clusters.csv", index=False)

    return {
        "best_k": best_k,
    }


def run_pc_sensitivity(
    min_count: int = MIN_CLASS_COUNT,
    pc_values=None,
) -> pd.DataFrame:
    """Compare the centroid clustering result across different numbers of expression PCs."""
    if pc_values is None:
        pc_values = DEFAULT_PC_SENSITIVITY_VALUES

    rows = []
    for n_gene_pcs in pc_values:
        prepared = prepare_filtered_data(
            target="supertype",
            min_count=min_count,
            class_name=None,
            filter_to_snr_z_range=True,
        )

        df = prepared["data"].copy()
        
        pc_columns = [
            f"expr_pc_{i + 1}"
            for i in range(n_gene_pcs)
            if f"expr_pc_{i + 1}" in df.columns
        ]
        
        df = df[["supertype", "x_ccf", "y_ccf", "z_ccf", *pc_columns]].copy()

        centroids = (
            df.groupby("supertype", as_index=False)
            .agg(
                n_cells=("supertype", "size"),
                x_ccf=("x_ccf", "mean"),
                y_ccf=("y_ccf", "mean"),
                z_ccf=("z_ccf", "mean"),
            )
            .sort_values("n_cells", ascending=False)
            .reset_index(drop=True)
        )

        if pc_columns:
            pc_means = df.groupby("supertype", as_index=False)[pc_columns].mean()
            centroids = centroids.merge(pc_means, on="supertype", how="left")

        scaled_centroids = StandardScaler().fit_transform(centroids[FEATURES + pc_columns])
        valid_k_values = [k for k in DEFAULT_K_VALUES if 2 <= k < len(centroids)]

        k_scores = []
        for k in valid_k_values:
            labels = KMeans(n_clusters=k, n_init=20, random_state=KMEANS_RANDOM_STATE).fit_predict(scaled_centroids)
            cluster_sizes = pd.Series(labels).value_counts()
            min_cluster_size = int(cluster_sizes.min())
            k_scores.append(
                {
                    "n_gene_pcs": n_gene_pcs,
                    "k": k,
                    "silhouette_score": silhouette_score(scaled_centroids, labels),
                    "min_cluster_size": min_cluster_size,
                    "is_valid": min_cluster_size >= MIN_CLUSTER_SUPERTYPES,
                }
            )

        k_summary = pd.DataFrame(k_scores)
        valid_summary = k_summary[k_summary["is_valid"]]
        best_row = (valid_summary if not valid_summary.empty else k_summary).sort_values("silhouette_score", ascending=False).iloc[0]
        rows.append(
            {
                "n_gene_pcs": n_gene_pcs,
                "best_k": int(best_row["k"]),
                "best_silhouette_score": best_row["silhouette_score"],
                "min_cluster_size": int(best_row["min_cluster_size"]),
            }
        )

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "pc_sensitivity_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()

    if args.sensitivity:
        print(run_pc_sensitivity())
    else:
        print(cluster_supertype_centroids())