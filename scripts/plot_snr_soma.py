from argparse import ArgumentParser
from pathlib import Path

from snr_utils import (
    CLUSTER_COLORS,
    CLUSTER_LABELS,
    DEFAULT_CLUSTER_COLUMN,
    PROJ_COLORS,
    PROJ_LABELS,
    load_snr_data,
    save_brainrender_plot,
)

PLOT_CONFIG = {
    "proj": {
        "output_path": Path("analysis_outputs/snr_soma_by_proj.png"),
        "order": ["orb", "rsp_orb", "rsp"],
        "colors": PROJ_COLORS,
        "legend_labels": PROJ_LABELS,
    },
    "projection_cluster_binary": {
        "output_path": Path("analysis_outputs/snr_soma_by_cluster_binary.png"),
        "order": [1, 2, 3],
        "colors": CLUSTER_COLORS,
        "legend_labels": CLUSTER_LABELS,
    },
    "projection_cluster_log": {
        "output_path": Path("analysis_outputs/snr_soma_by_cluster_log.png"),
        "order": [1, 2, 3],
        "colors": CLUSTER_COLORS,
        "legend_labels": CLUSTER_LABELS,
    },
}


def make_plot(group_col: str = DEFAULT_CLUSTER_COLUMN) -> Path:
    """Render SNR soma locations in brainrender for one grouping."""
    df = load_snr_data()
    config = PLOT_CONFIG[group_col]
    return save_brainrender_plot(
        df,
        group_col=group_col,
        output_path=config["output_path"],
        order=config["order"],
        colors=config["colors"],
        legend_labels=config["legend_labels"],
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--group-col", choices=["proj", "projection_cluster_binary", "projection_cluster_log"], default=DEFAULT_CLUSTER_COLUMN)
    args = parser.parse_args()
    make_plot(group_col=args.group_col)
