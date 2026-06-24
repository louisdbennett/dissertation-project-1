from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.stats.composition import clr
from skbio.stats.distance import permanova


INPUT_PATH = Path("analysis_outputs/snr_classification/full_probabilities.csv")
OUTPUT_DIR = Path("analysis_outputs/snr_classification")
DEFAULT_GROUP_COLUMN = "projection_cluster_log"
N_PERMUTATIONS = 4999
RANDOM_STATE = 0
EPSILON = 1e-12

META_COLS = {
    "neuron_ID",
    "mouseID",
    "injection",
    "comment",
    "x",
    "y",
    "z",
    "proj",
    "projection_cluster_binary",
    "projection_cluster_log",
    "projection_cluster_xyz",
    "predicted_label",
}


def get_probability_columns(df: pd.DataFrame) -> list[str]:
    """Keep only transferred supertype probability columns."""
    return [
        col
        for col in df.columns
        if col not in META_COLS and not col.endswith("_endpoint")
    ]


def prepare_probability_matrix(df: pd.DataFrame, probability_columns: list[str]) -> np.ndarray:
    """Convert the transferred probabilities into CLR-transformed coordinates."""
    x = df[probability_columns].to_numpy(dtype=float)
    x = x + EPSILON
    x = x / x.sum(axis=1, keepdims=True)
    return clr(x)


def build_distance_matrix(x: np.ndarray, ids: pd.Index) -> DistanceMatrix:
    """Build the Euclidean distance matrix used by scikit-bio PERMANOVA."""
    return DistanceMatrix(squareform(pdist(x, metric="euclidean")), ids=ids.astype(str).tolist())


def get_r2(x: np.ndarray, groups: np.ndarray) -> float:
    """Calculate the PERMANOVA-style R2 on the CLR coordinates."""
    grand_centroid = x.mean(axis=0)
    ss_total = ((x - grand_centroid) ** 2).sum()
    ss_within = 0.0
    for group in np.unique(groups):
        x_group = x[groups == group]
        group_centroid = x_group.mean(axis=0)
        ss_within += ((x_group - group_centroid) ** 2).sum()
    return (ss_total - ss_within) / ss_total


def run_one_permanova(
    x: np.ndarray,
    groups: pd.Series,
    n_permutations: int = N_PERMUTATIONS,
) -> pd.Series:
    """Run one scikit-bio PERMANOVA and add a simple R2 effect size."""
    distance_matrix = build_distance_matrix(x, groups.index)
    result = permanova(
        distance_matrix,
        grouping=groups.astype(str).to_numpy(),
        permutations=n_permutations,
        seed=RANDOM_STATE,
    )
    result = result.copy()
    result["r2"] = get_r2(x, groups.to_numpy())
    return result


def run_pairwise_permanova(
    x: np.ndarray,
    groups: pd.Series,
    n_permutations: int = N_PERMUTATIONS,
) -> pd.DataFrame:
    """Run pairwise scikit-bio PERMANOVA across all group pairs."""
    rows = []
    unique_groups = pd.Index(sorted(groups.dropna().unique().tolist()))

    for i, group_a in enumerate(unique_groups):
        for group_b in unique_groups[i + 1 :]:
            keep = groups.isin([group_a, group_b])
            result = run_one_permanova(
                x[keep.to_numpy()],
                groups.loc[keep],
                n_permutations=n_permutations,
            )
            rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    **result.to_dict(),
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-col", default=DEFAULT_GROUP_COLUMN)
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()

    df = pd.read_csv(INPUT_PATH)
    probability_columns = get_probability_columns(df)
    keep = df[args.group_col].notna()
    x = prepare_probability_matrix(df.loc[keep], probability_columns)
    groups = df.loc[keep, args.group_col]

    global_result = run_one_permanova(x, groups, n_permutations=args.permutations).to_frame().T
    pairwise_result = run_pairwise_permanova(x, groups, n_permutations=args.permutations)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_result.to_csv(OUTPUT_DIR / f"{args.group_col}_permanova.csv", index=False)
    pairwise_result.to_csv(OUTPUT_DIR / f"{args.group_col}_permanova_pairwise.csv", index=False)
