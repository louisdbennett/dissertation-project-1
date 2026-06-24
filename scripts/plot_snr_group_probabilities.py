import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from snr_utils import DEFAULT_CLUSTER_COLUMN


INPUT_PATH = Path("analysis_outputs/snr_classification/full_probabilities.csv")
OUTPUT_PATH = Path("analysis_outputs/snr_classification/group_mean_probabilities.png")
AXIS_LABELS = {
    "proj": "Projection group",
    "projection_cluster_binary": "Binary projection cluster",
    "projection_cluster_log": "Log projection cluster",
}


def format_label(label: str) -> str:
    if label == "other":
        return "Other"
    parts = label.split(" ", 1)
    clean = parts[1] if len(parts) == 2 and parts[0].isdigit() else label
    return clean.replace("_", " ")


def plot_group_mean_probabilities(group_column: str = DEFAULT_CLUSTER_COLUMN) -> Path:
    df = pd.read_csv(INPUT_PATH)
    group_label = AXIS_LABELS.get(group_column, group_column)
    title_group_label = group_label.lower()

    non_probability_cols = {
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
        "predicted_label",
    }
    value_cols = [
        col
        for col in df.columns
        if col not in non_probability_cols
        and not col.endswith("_endpoint")
    ]
    summary = df.groupby(group_column)[value_cols].mean().reset_index()
    order = summary[value_cols].mean().sort_values(ascending=False).index.tolist()
    group_order = summary[group_column].tolist()

    fig, axes = plt.subplots(1, len(group_order), figsize=(5 * len(group_order), 5), sharex=True)
    if len(group_order) == 1:
        axes = [axes]

    for ax, group in zip(axes, group_order):
        row = summary[summary[group_column] == group]
        values = row[order].iloc[0]
        labels = [format_label(label) for label in order]

        ax.barh(labels, values, color="steelblue", alpha=0.85)
        ax.invert_yaxis()
        ax.set_title(str(group))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean predicted probability")

    axes[0].set_ylabel("Supertype")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-column", default=DEFAULT_CLUSTER_COLUMN)
    args = parser.parse_args()
    plot_group_mean_probabilities(group_column=args.group_column)
