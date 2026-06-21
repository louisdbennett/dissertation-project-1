from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("master_detailed_comment.csv")
OUTPUT_DIR = Path("analysis_outputs/snr_bar_tracing")
BAD_LABEL = "bad tracing"
N_BOOT = 5000
SUMMARY_COLUMNS = [
    "total_endpoints",
    "total_length",
    "n_endpoint_areas",
    "n_length_areas",
    "x",
    "y",
    "z",
]


def bootstrap_ci(bad: pd.Series, other: pd.Series) -> tuple[float, float]:
    """Return a bootstrap 95% interval for the mean difference."""
    rng = np.random.default_rng(0)
    bad_values = bad.dropna().to_numpy()
    other_values = other.dropna().to_numpy()

    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        diffs[i] = (
            rng.choice(bad_values, size=len(bad_values), replace=True).mean()
            - rng.choice(other_values, size=len(other_values), replace=True).mean()
        )

    return tuple(np.quantile(diffs, [0.025, 0.975]))


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise how bad tracing rows differ from the rest of the SNR table."""
    endpoint_cols = [col for col in df.columns if col.endswith("_endpoint")]
    length_cols = [col for col in df.columns if col.endswith("_length")]

    work = df.copy()
    work["comment"] = work["comment"].fillna("NA")
    work["group"] = np.where(work["comment"] == BAD_LABEL, "bad", "other")
    work["total_endpoints"] = work[endpoint_cols].sum(axis=1)
    work["total_length"] = work[length_cols].sum(axis=1)
    work["n_endpoint_areas"] = (work[endpoint_cols] > 0).sum(axis=1)
    work["n_length_areas"] = (work[length_cols] > 0).sum(axis=1)

    bad = work.loc[work["group"] == "bad"]
    other = work.loc[work["group"] == "other"]

    rows = []
    for column in SUMMARY_COLUMNS:
        ci_low, ci_high = bootstrap_ci(bad[column], other[column])
        rows.append(
            {
                "measure": column,
                "mean_bad": bad[column].mean(),
                "mean_other": other[column].mean(),
                "diff_bad_minus_other": bad[column].mean() - other[column].mean(),
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    summary = build_summary(pd.read_csv(DATA_PATH))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
