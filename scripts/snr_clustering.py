from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from snr_permanova import clr_transform, run_permanova


SNR_PATH = Path("analysis_tables/snr_proj_location_table.csv")
RAW_SNR_PATH = Path("master_detailed_comment.csv")
PROBABILITY_PATH = Path("analysis_outputs/snr_classification/full_probabilities.csv")
OUTPUT_DIR = Path("analysis_outputs/snr_clustering")
K_VALUES = [2, 3, 4, 5]
RANDOM_STATE = 0
MIN_CLUSTER_SIZE = 10
META_COLS = {"neuron_ID", "mouseID", "injection", "x", "y", "z", "proj", "predicted_label"}


def load_snr_feature_table() -> pd.DataFrame:
    """Load the SNR analysis rows with the projection-feature columns added."""
    snr = pd.read_csv(SNR_PATH)
    raw = pd.read_csv(RAW_SNR_PATH, low_memory=False)
    feature_cols = [col for col in raw.columns if col.endswith("_length") or col.endswith("_endpoint")]
    return snr.merge(raw[["neuron_ID", *feature_cols]], on="neuron_ID", how="left")


if __name__ == "__main__":
    snr = load_snr_feature_table()
    transferred = pd.read_csv(PROBABILITY_PATH)

    feature_cols = [col for col in snr.columns if col.endswith("_length") or col.endswith("_endpoint")]
    x = StandardScaler().fit_transform((snr[feature_cols] + 1).transform("log"))
    prob_cols = [col for col in transferred.columns if col not in META_COLS]
    composition = clr_transform(transferred[prob_cols].to_numpy(dtype=float))

    rows = []
    best_labels = None
    best_k = None
    best_r2 = None

    baseline = run_permanova(composition, snr["proj"].to_numpy())
    rows.append(
        {
            "method": "proj",
            "k": snr["proj"].nunique(),
            "min_cluster_size": snr["proj"].value_counts().min(),
            "silhouette_projection_features": None,
            "pseudo_f": baseline["pseudo_f"],
            "r2": baseline["r2"],
            "p_value": baseline["p_value"],
        }
    )

    for k in K_VALUES:
        labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(x)
        label_counts = pd.Series(labels).value_counts()
        min_cluster_size = int(label_counts.min())
        result = run_permanova(composition, labels)
        rows.append(
            {
                "method": "kmeans_projection_features",
                "k": k,
                "min_cluster_size": min_cluster_size,
                "silhouette_projection_features": silhouette_score(x, labels),
                "pseudo_f": result["pseudo_f"],
                "r2": result["r2"],
                "p_value": result["p_value"],
            }
        )

        if min_cluster_size >= MIN_CLUSTER_SIZE and (best_r2 is None or result["r2"] > best_r2):
            best_labels = labels
            best_k = k
            best_r2 = result["r2"]

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "cluster_search_summary.csv", index=False)

    if best_labels is not None:
        assignment = snr.copy()
        assignment["projection_feature_cluster"] = best_labels
        assignment.to_csv(OUTPUT_DIR / "best_projection_feature_clusters.csv", index=False)
