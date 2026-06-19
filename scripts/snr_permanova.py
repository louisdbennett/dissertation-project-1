from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("analysis_outputs/snr_classification/full_probabilities.csv")
OUTPUT_DIR = Path("analysis_outputs/snr_classification")
N_PERMUTATIONS = 4999
RANDOM_STATE = 0
EPSILON = 1e-12


def clr_transform(x: np.ndarray) -> np.ndarray:
    x = x + EPSILON
    x = x / x.sum(axis=1, keepdims=True)
    log_x = np.log(x)
    return log_x - log_x.mean(axis=1, keepdims=True)


def permanova_statistic(x: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    n_samples = len(groups)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    grand_centroid = x.mean(axis=0)
    ss_total = ((x - grand_centroid) ** 2).sum()

    ss_within = 0.0
    for group in unique_groups:
        x_group = x[groups == group]
        group_centroid = x_group.mean(axis=0)
        ss_within += ((x_group - group_centroid) ** 2).sum()

    ss_between = ss_total - ss_within
    df_between = n_groups - 1
    df_within = n_samples - n_groups
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    pseudo_f = ms_between / ms_within
    r2 = ss_between / ss_total
    return pseudo_f, r2


def run_permanova(x: np.ndarray, groups: np.ndarray, n_permutations: int = N_PERMUTATIONS) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    observed_f, r2 = permanova_statistic(x, groups)

    permuted_f = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        permuted_f[i], _ = permanova_statistic(x, rng.permutation(groups))

    p_value = (1 + np.sum(permuted_f >= observed_f)) / (n_permutations + 1)

    return {
        "pseudo_f": observed_f,
        "r2": r2,
        "p_value": p_value,
    }


def run_pairwise_permanova(x: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    unique_groups = pd.Index(np.unique(groups))
    rows = []

    for i, group_a in enumerate(unique_groups):
        for group_b in unique_groups[i + 1 :]:
            keep = np.isin(groups, [group_a, group_b])
            result = run_permanova(x[keep], groups[keep])
            rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    **result,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    meta_cols = {"neuron_ID", "mouseID", "injection", "x", "y", "z", "proj", "predicted_label"}
    prob_cols = [col for col in df.columns if col not in meta_cols]

    x = clr_transform(df[prob_cols].to_numpy(dtype=float))
    groups = df["proj"].to_numpy()

    global_result = pd.DataFrame([run_permanova(x, groups)])
    pairwise_result = run_pairwise_permanova(x, groups)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_path = OUTPUT_DIR / "permanova_global.csv"
    pairwise_path = OUTPUT_DIR / "permanova_pairwise.csv"
    global_result.to_csv(global_path, index=False)
    pairwise_result.to_csv(pairwise_path, index=False)
