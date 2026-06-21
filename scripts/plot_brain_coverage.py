import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from snr_utils import load_snr_data

MERFISH_DATA_PATH = Path("analysis_tables/merfish_supertype_location_table.csv")
SCATTER_OUTPUT_PATH = Path("analysis_outputs/soma_scatter_plots.png")


def make_scatter_plot() -> Path:
    merfish = pd.read_csv(MERFISH_DATA_PATH, dtype={"cell_id": "string"})
    snr = load_snr_data()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [
        ("x_ccf", "y_ccf", "x", "y", "X vs Y"),
        ("x_ccf", "z_ccf", "x", "z", "X vs Z"),
        ("y_ccf", "z_ccf", "y", "z", "Y vs Z"),
    ]
    axis_labels = {
        "x_ccf": "x",
        "y_ccf": "y",
        "z_ccf": "z",
        "x": "x",
        "y": "y",
        "z": "z",
    }

    for ax, (mx, my, sx, sy, title) in zip(axes, pairs):
        ax.scatter(
            merfish[mx],
            merfish[my],
            s=6,
            alpha=0.08,
            color="seagreen",
            label="MERFISH",
            linewidths=0,
            zorder=1,
        )
        ax.scatter(
            snr[sx],
            snr[sy],
            s=32,
            alpha=0.95,
            color="black",
            label="SNR",
            edgecolors="white",
            linewidths=0,
            zorder=3,
        )
        ax.set_xlabel(axis_labels[mx])
        ax.set_ylabel(axis_labels[my])
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            if hasattr(handle, "set_alpha"):
                handle.set_alpha(0.9 if label != "Other" else 0.5)
            unique_handles.append(handle)
            unique_labels.append(label)
            seen.add(label)
    fig.legend(
        unique_handles,
        unique_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=max(2, len(unique_labels)),
        markerscale=2.2,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    output_path = SCATTER_OUTPUT_PATH
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    make_scatter_plot()
