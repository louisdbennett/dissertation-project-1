import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_PATH = Path("analysis_outputs/snr_classification/full_probabilities.csv")
OUTPUT_DIR = Path("analysis_outputs/snr_classification")
DEFAULT_PROBABILITY_COLUMN = "0036 L2/3 IT ENT Glut_4"
DEFAULT_GROUP_COLUMN = "proj"
GROUP_ORDER = ["orb", "rsp_orb", "rsp"]
AXIS_LABELS = {
    "proj": "Projection group",
}


def format_label(label: str) -> str:
    if label == "other":
        return "Other"
    parts = label.split(" ", 1)
    clean = parts[1] if len(parts) == 2 and parts[0].isdigit() else label
    return clean.replace("_", " ")


def make_output_path(probability_column: str) -> Path:
    safe_name = probability_column.lower().replace("/", "_").replace(" ", "_")
    return OUTPUT_DIR / f"{safe_name}_probability_distribution.png"


def plot_probability_distribution(
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
    group_column: str = DEFAULT_GROUP_COLUMN,
) -> Path:
    """Plot per-neuron transferred probabilities by SNR group."""
    df = pd.read_csv(INPUT_PATH)
    title_group_label = AXIS_LABELS.get(group_column, group_column).lower()
    group_order = [group for group in GROUP_ORDER if group in df[group_column].unique()]
    if not group_order:
        group_order = df[group_column].dropna().unique().tolist()

    grouped_values = [
        df.loc[df[group_column] == group, probability_column].to_numpy(dtype=float)
        for group in group_order
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    violin = ax.violinplot(grouped_values, positions=np.arange(1, len(group_order) + 1), showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("steelblue")
        body.set_edgecolor("none")
        body.set_alpha(0.35)

    for idx, values in enumerate(grouped_values, start=1):
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.vlines(idx, q1, q3, color="black", linewidth=4, alpha=0.8)
        ax.hlines(median, idx - 0.18, idx + 0.18, color="black", linewidth=2)

        jitter = np.linspace(-0.14, 0.14, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=18,
            color="black",
            alpha=0.35,
            linewidths=0,
        )

    ax.set_xticks(np.arange(1, len(group_order) + 1))
    ax.set_xticklabels(group_order)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted probability")
    ax.set_xlabel(AXIS_LABELS.get(group_column, group_column))
    ax.set_title(f"{format_label(probability_column)} probability by {title_group_label}")

    output_path = make_output_path(probability_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probability-column", default=DEFAULT_PROBABILITY_COLUMN)
    parser.add_argument("--group-column", default=DEFAULT_GROUP_COLUMN)
    args = parser.parse_args()

    plot_probability_distribution(
        probability_column=args.probability_column,
        group_column=args.group_column,
    )
