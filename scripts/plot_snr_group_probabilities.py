from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_PATH = Path("analysis_outputs/snr_classification/summary.csv")
OUTPUT_PATH = Path("analysis_outputs/snr_classification/proj_mean_probabilities.png")


def format_label(label: str) -> str:
    if label == "other":
        return "Other"
    parts = label.split(" ", 1)
    clean = parts[1] if len(parts) == 2 and parts[0].isdigit() else label
    return clean.replace("_", " ")


def plot_group_mean_probabilities(group_column: str = "proj") -> Path:
    summary = pd.read_csv(SUMMARY_PATH)

    value_cols = [col for col in summary.columns if col != group_column]
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
    fig.suptitle(f"Mean supertype probabilities by {group_column}", y=1.02)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-column", default="proj")
    args = parser.parse_args()

    print(plot_group_mean_probabilities(group_column=args.group_column))
