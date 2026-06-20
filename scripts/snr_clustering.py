from argparse import ArgumentParser
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score

from snr_utils import load_snr_data


OUTPUT_DIR = Path("analysis_outputs/snr_clustering")

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")


def save_dendrogram(linkage_matrix) -> None:
    """Save the dendrogram panel."""
    fig, ax = plt.subplots(figsize=(13, 4))
    from scipy.cluster.hierarchy import dendrogram

    dendrogram(linkage_matrix, ax=ax, no_labels=True, color_threshold=0, above_threshold_color="black")
    ax.set_xticks([])
    ax.set_ylabel("distance")
    ax.set_title("Hierarchical clustering of SNR projection patterns")
    fig.savefig(OUTPUT_DIR / "dendrogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(x_binary: pd.DataFrame, cluster_labels: pd.Series, cluster_order: list[int]) -> None:
    """Save the mean projection heatmap by cluster."""
    heatmap = (
        x_binary.assign(projection_cluster=cluster_labels.to_numpy())
        .groupby("projection_cluster")
        .mean()
        .reindex(cluster_order)
        .T
    )
    heatmap.index = [col.replace("_endpoint", "") for col in heatmap.index]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        heatmap,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "fraction of neurons with endpoint"},
    )
    ax.set_xlabel("projection cluster")
    ax.set_ylabel("target area")
    ax.set_xticklabels([str(cluster) for cluster in cluster_order], rotation=0)
    fig.savefig(OUTPUT_DIR / "cluster_mean_endpoints.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_analysis(n_clusters: int = 3) -> None:
    """Cluster SNR neurons by binary endpoint pattern and save the main summaries."""
    df, x_binary, endpoint_cols, linkage_matrix, cluster_order = load_snr_data(
        n_clusters=n_clusters,
        with_details=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"label_set": "proj", "metric": "euclidean", "score": silhouette_score(x_binary, df["proj"], metric="euclidean")},
            {
                "label_set": "projection_cluster",
                "metric": "euclidean",
                "score": silhouette_score(x_binary, df["projection_cluster"], metric="euclidean"),
            },
        ]
    ).to_csv(OUTPUT_DIR / "silhouette_comparison.csv", index=False)

    save_dendrogram(linkage_matrix)
    save_heatmap(x_binary, df["projection_cluster"], cluster_order)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=3)
    args = parser.parse_args()
    run_analysis(n_clusters=args.n_clusters)
