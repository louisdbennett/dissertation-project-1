import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import vedo
from brainrender import Scene, settings
from brainrender.actors import Points
from merfish_utils import SUPERTYPE_COVERAGE, group_supertypes

MERFISH_DATA_PATH = Path("analysis_tables/merfish_supertype_location_table.csv")
SNR_DATA_PATH = Path("analysis_tables/snr_proj_location_table.csv")
SCATTER_OUTPUT_PATH = Path("analysis_outputs/scatter_plots.png")
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
    snr = pd.read_csv(SNR_DATA_PATH)

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
        "x_ccf": "x (CCF)",
        "y_ccf": "y (CCF)",
        "z_ccf": "z (CCF)",
        "x": "x (CCF)",
        "y": "y (CCF)",
        "z": "z (CCF)",
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
            linewidths=0.35,
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
    title = "Soma coverage: SNR vs MERFISH"
    if z_filtered:
        title += " (SNR z range)"
    if show_grouped_supertypes:
        title += f" + grouped supertypes ({SUPERTYPE_COVERAGE:.0%} coverage)"
    fig.suptitle(title, y=1.02)
    fig.tight_layout(rect=(0, 0, 0.82, 1))

    output_path = SCATTER_OUTPUT_PATH
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def make_brainrender_plot() -> Path:
    merfish = pd.read_csv(MERFISH_DATA_PATH)
    snr = pd.read_csv(SNR_DATA_PATH)

    settings.OFFSCREEN = True
    vedo.settings.default_backend = "vtk"

    scene = Scene(atlas_name="allen_mouse_25um", title="MERFISH EC cells + SNR soma")

    for structure, color in [("ENTm", "orange"), ("ENTl", "seagreen")]:
        coords = merfish.loc[
            merfish["structure"] == structure, ["x_ccf", "y_ccf", "z_ccf"]
        ].to_numpy()
        scene.add(Points(coords, radius=10, colors=color, alpha=0.35))

    snr_coords = snr[["x", "y", "z"]].to_numpy()
    scene.add(Points(snr_coords, radius=40, colors="black", alpha=1.0))

    scene.add_brain_region("ENTl", alpha=0.05, color="lightblue")
    scene.add_brain_region("ENTm", alpha=0.05, color="lightblue")

    scene.render(interactive=False, camera="three_quarters", zoom=1.35)
    BRAINRENDER_OUTPUT_PATH.parent.mkdir(exist_ok=True)
    scene.screenshot(name=str(BRAINRENDER_OUTPUT_PATH))

    return BRAINRENDER_OUTPUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scatter", "brainrender"], default="scatter")
    parser.add_argument("--z-filtered", action="store_true")
    parser.add_argument("--show-grouped-supertypes", action="store_true")
    args = parser.parse_args()

    if args.mode == "brainrender":
        make_brainrender_plot()
    else:
        make_scatter_plot(
            z_filtered=args.z_filtered,
            show_grouped_supertypes=args.show_grouped_supertypes,
        )
