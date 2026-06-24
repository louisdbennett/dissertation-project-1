from argparse import ArgumentParser
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score

from snr_utils import CLUSTER_TRANSFORM, DEFAULT_CLUSTER_COLUMN, get_binary_endpoint_matrix, load_snr_data


OUTPUT_DIR = Path("analysis_outputs/snr_clustering")
GROUP_ORDERS = {
    "proj": ["rsp", "rsp_orb", "orb"],
}
X_LABELS = {
    "proj": "Projection group",
    "projection_cluster_binary": "Binary projection cluster",
    "projection_cluster_log": "Log projection cluster",
    "projection_cluster_xyz": "XYZ projection cluster",
}

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")


def save_dendrogram(linkage_matrix, transform: str) -> None:
    """Save the dendrogram panel."""
    fig, ax = plt.subplots(figsize=(13, 4))
    from scipy.cluster.hierarchy import dendrogram

    dendrogram(linkage_matrix, ax=ax, no_labels=True, color_threshold=0, above_threshold_color="black")
    ax.set_xticks([])
    ax.set_ylabel("Distance")
    fig.savefig(OUTPUT_DIR / f"dendrogram_{transform}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(x_binary: pd.DataFrame, labels: pd.Series, group_col: str, transform: str) -> None:
    """Save the mean projection heatmap for one SNR grouping column."""
    if group_col == "proj":
        group_order = [label for label in GROUP_ORDERS[group_col] if label in labels.dropna().unique()]
    else:
        group_order = sorted(labels.dropna().unique().tolist())
    heatmap = (
        x_binary.assign(group=labels.to_numpy())
        .groupby("group")
        .mean()
        .reindex(group_order)
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
        cbar_kws={"label": "Fraction of neurons with endpoint"},
    )
    ax.set_xlabel(X_LABELS[group_col])
    ax.set_xticklabels([str(label) for label in group_order], rotation=0)
    fig.savefig(OUTPUT_DIR / f"{group_col}_mean_endpoints_{transform}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    n_clusters: int = 3,
    group_col: str = DEFAULT_CLUSTER_COLUMN,
    transform: str = CLUSTER_TRANSFORM,
) -> pd.DataFrame:
    """Cluster SNR neurons and save the main summaries for one endpoint transform."""
    df, x_features, endpoint_cols, linkage_matrix, cluster_order = load_snr_data(
        n_clusters=n_clusters,
        transform=transform,
        with_details=True,
    )
    x_binary, _ = get_binary_endpoint_matrix(df)

    keep = df["proj"].notna()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "transform": transform,
                "label_set": "proj",
                "score": silhouette_score(
                    x_features.loc[keep],
                    df.loc[keep, "proj"],
                    metric="euclidean",
                ),
            },
            {
                "transform": transform,
                "label_set": f"projection_cluster_{transform}",
                "score": silhouette_score(
                    x_features, df[f"projection_cluster_{transform}"], metric="euclidean"
                ),
            },
        ]
    )
    summary.to_csv(OUTPUT_DIR / f"silhouette_comparison_{transform}.csv", index=False)

    save_dendrogram(linkage_matrix, transform)
    save_heatmap(x_binary, df[group_col], group_col, transform)
    return df[["neuron_ID", "proj", "projection_cluster_binary", "projection_cluster_log", "projection_cluster_xyz"]]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--group-col", choices=["proj", "projection_cluster_binary", "projection_cluster_log", "projection_cluster_xyz"], default=DEFAULT_CLUSTER_COLUMN)
    parser.add_argument("--transform", choices=["binary", "log", "xyz"], default=CLUSTER_TRANSFORM)
    args = parser.parse_args()
    run_analysis(n_clusters=args.n_clusters, group_col=args.group_col, transform=args.transform)
