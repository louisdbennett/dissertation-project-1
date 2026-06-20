import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import vedo
from brainrender import Scene, settings
from brainrender.actors import Points
from merfish_utils import group_supertypes
from snr_utils import load_snr_data

MERFISH_DATA_PATH = Path("analysis_tables/merfish_supertype_location_table.csv")
SCATTER_OUTPUT_PATH = Path("analysis_outputs/soma_scatter_plots.png")
BRAINRENDER_OUTPUT_PATH = Path("analysis_outputs/brainrender_map.png")


def format_group_label(label: str) -> str:
    if label == "other":
        return "Other"
    parts = label.split(" ", 1)
    clean = parts[1] if len(parts) == 2 and parts[0].isdigit() else label
    return clean.replace("_", " ")


def make_scatter_plot(
    z_filtered: bool = False,
    show_grouped_supertypes: bool = False,
) -> Path:
    merfish = pd.read_csv(MERFISH_DATA_PATH, dtype={"cell_id": "string"})
    snr = load_snr_data()

    if z_filtered:
        merfish = merfish[
            (merfish["z_ccf"] >= snr["z"].min()) & (merfish["z_ccf"] <= snr["z"].max())
        ]
    if show_grouped_supertypes:
        merfish, _, _ = group_supertypes(merfish)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [
        ("x_ccf", "y_ccf", "x", "y", "X vs Y"),
        ("x_ccf", "z_ccf", "x", "z", "X vs Z"),
        ("y_ccf", "z_ccf", "y", "z", "Y vs Z"),
    ]
    grouped_colors = plt.get_cmap("tab10" if merfish.get("supertype_grouped", pd.Series()).nunique() <= 10 else "tab20", merfish["supertype_grouped"].nunique()) if show_grouped_supertypes else None
    axis_labels = {
        "x_ccf": "x",
        "y_ccf": "y",
        "z_ccf": "z",
        "x": "x",
        "y": "y",
        "z": "z",
    }

    for ax, (mx, my, sx, sy, title) in zip(axes, pairs):
        if show_grouped_supertypes:
            label_order = [label for label in merfish["supertype_grouped"].unique() if label != "other"]
            if "other" in merfish["supertype_grouped"].values:
                label_order.append("other")

            for color_idx, label in enumerate(label_order):
                group_df = merfish[merfish["supertype_grouped"] == label]
                color = "#d9d9d9" if label == "other" else grouped_colors(color_idx)
                ax.scatter(
                    group_df[mx],
                    group_df[my],
                    s=8 if label != "other" else 6,
                    alpha=0.18 if label != "other" else 0.05,
                    color=color,
                    label=format_group_label(label),
                    linewidths=0,
                    zorder=1 if label == "other" else 2,
                )
        else:
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
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        markerscale=2.2,
    )
    title = "SOMA locations across SNR and MERFISH datasets"

    fig.suptitle(title, y=1.02)
    fig.tight_layout(rect=(0, 0, 0.82, 1))

    output_path = SCATTER_OUTPUT_PATH
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--z-filtered", action="store_true")
    parser.add_argument("--show-grouped-supertypes", action="store_true")
    args = parser.parse_args()

    make_scatter_plot(
        z_filtered=args.z_filtered,
        show_grouped_supertypes=args.show_grouped_supertypes,
    )
