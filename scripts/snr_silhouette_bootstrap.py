import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from snr_utils import CLUSTER_TRANSFORM, DEFAULT_CLUSTER_COLUMN, load_snr_data


OUTPUT_DIR = Path("analysis_outputs/snr_clustering")
N_BOOTSTRAP = 2000
RANDOM_STATE = 0
BASELINE_LABEL = "proj"

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")


def run_bootstrap(
    x: np.ndarray,
    proj_labels: np.ndarray,
    cluster_labels: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Bootstrap the silhouette-score gain of the cluster labels over proj."""
    rng = np.random.default_rng(random_state)
    proj_score = silhouette_score(x, proj_labels, metric="euclidean")
    cluster_score = silhouette_score(x, cluster_labels, metric="euclidean")
    observed_delta = cluster_score - proj_score

    deltas = []
    while len(deltas) < n_bootstrap:
        n_samples = len(proj_labels)
        sample_idx = rng.integers(0, n_samples, size=n_samples)
        proj_boot = proj_labels[sample_idx]
        cluster_boot = cluster_labels[sample_idx]

        if pd.Series(proj_boot).value_counts().min() < 2:
            continue
        if pd.Series(cluster_boot).value_counts().min() < 2:
            continue

        x_boot = x[sample_idx]
        deltas.append(
            silhouette_score(x_boot, cluster_boot, metric="euclidean")
            - silhouette_score(x_boot, proj_boot, metric="euclidean")
        )

    deltas = np.asarray(deltas)
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    p_value = (1 + np.sum(deltas <= 0)) / (len(deltas) + 1)

    return pd.DataFrame(
        [
            {
                "proj_score": proj_score,
                "cluster_score": cluster_score,
                "delta_cluster_minus_proj": observed_delta,
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "p_boot": p_value,
                "n_bootstrap": len(deltas),
            }
        ]
    )


def run_analysis(
    transform: str = CLUSTER_TRANSFORM,
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """Compare proj against the current cluster labels with a bootstrap."""
    df, x_features, _, _, _ = load_snr_data(transform=transform, with_details=True)
    keep = df[BASELINE_LABEL].notna() & df[DEFAULT_CLUSTER_COLUMN].notna()

    summary = run_bootstrap(
        x_features.loc[keep].to_numpy(),
        df.loc[keep, BASELINE_LABEL].to_numpy(),
        df.loc[keep, DEFAULT_CLUSTER_COLUMN].to_numpy(),
        n_bootstrap=n_bootstrap,
    )
    summary.insert(0, "transform", transform)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"silhouette_bootstrap_{transform}_{BASELINE_LABEL}_vs_{DEFAULT_CLUSTER_COLUMN}.csv"
    summary.to_csv(out_path, index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transform", choices=["binary", "log", "xyz"], default=CLUSTER_TRANSFORM)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    summary = run_analysis(
        transform=args.transform,
        n_bootstrap=args.n_bootstrap,
    )
    print(summary.round(4).to_string(index=False))
