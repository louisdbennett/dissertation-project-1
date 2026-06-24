import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_PATH = Path("analysis_outputs/merfish_prediction/raw_supertype_frequency.png")
MERFISH_PATH = Path("analysis_tables/merfish_supertype_location_table.csv")
SNR_PATH = Path("analysis_tables/snr_proj_location_table.csv")


def plot_supertype_frequency(filter_to_snr_z_range: bool = True) -> Path:
    """Plot the raw MERFISH supertype frequency distribution."""
    df = pd.read_csv(MERFISH_PATH, dtype={"cell_id": "string"})
    if filter_to_snr_z_range:
        snr = pd.read_csv(SNR_PATH)
        z_min = snr["z"].min()
        z_max = snr["z"].max()
        df = df[(df["z_ccf"] >= z_min) & (df["z_ccf"] <= z_max)]

    counts = df["supertype"].value_counts().sort_values(ascending=False).reset_index()
    counts.columns = ["supertype", "n_cells"]
    counts["rank"] = np.arange(1, len(counts) + 1)
    counts["coverage"] = counts["n_cells"].cumsum() / counts["n_cells"].sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(counts["rank"], counts["n_cells"], color="#3a6ea5", linewidth=2)
    axes[0].scatter(counts["rank"], counts["n_cells"], color="#3a6ea5", s=14)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Raw supertype rank")
    axes[0].set_ylabel("Cells per supertype")

    for _, row in counts.head(5).iterrows():
        short_label = row["supertype"].split(" ", 1)[-1].replace("_", " ")
        axes[0].annotate(
            short_label,
            (row["rank"], row["n_cells"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    axes[1].plot(counts["rank"], counts["coverage"], color="#b55d60", linewidth=2)
    axes[1].set_xlabel("Raw supertype rank")
    axes[1].set_ylabel("Cumulative coverage")
    axes[1].set_ylim(0, 1.02)

    for level in [0.5, 0.75, 0.9]:
        rank = int(np.searchsorted(counts["coverage"].to_numpy(), level) + 1)
        axes[1].axhline(level, color="#cfcfcf", linestyle="--", linewidth=1)
        axes[1].axvline(rank, color="#e0e0e0", linestyle=":", linewidth=1)
        axes[1].text(rank + 0.5, level + 0.015, f"{int(level * 100)}% by rank {rank}", fontsize=8)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-z", action="store_true")
    args = parser.parse_args()
    plot_supertype_frequency(filter_to_snr_z_range=not args.full_z)
