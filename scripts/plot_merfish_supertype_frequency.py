import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

OUTPUT_PATH = Path("analysis_outputs/merfish_prediction/raw_supertype_frequency.png")
MERFISH_PATH = Path("analysis_tables/merfish_supertype_location_table.csv")
SNR_PATH = Path("analysis_tables/snr_proj_location_table.csv")


def plot_supertype_frequency(filter_to_snr_z_range: bool = True) -> Path:
    """Plot the distribution of raw MERFISH supertype counts."""
    df = pd.read_csv(MERFISH_PATH, dtype={"cell_id": "string"})
    if filter_to_snr_z_range:
        snr = pd.read_csv(SNR_PATH)
        z_min = snr["z"].min()
        z_max = snr["z"].max()
        df = df[(df["z_ccf"] >= z_min) & (df["z_ccf"] <= z_max)]

    counts = df["supertype"].value_counts().sort_values(ascending=False).to_numpy()
    kde = gaussian_kde(counts)
    x_grid = np.linspace(counts.min(), counts.max(), 400)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=30, density=True, color="#c7d7eb", edgecolor="white")
    ax.plot(x_grid, kde(x_grid), color="#2f5d8a", linewidth=2.5)
    ax.set_xlabel("Cells per supertype")
    ax.set_ylabel("Density")
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
