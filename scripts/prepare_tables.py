from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "analysis_tables"
ANALYSIS_OUTPUT_DIR = REPO_ROOT / "analysis_outputs"
N_GENE_PCS = 200
PCA_PLOT_COMPONENTS = 200
PCA_VARIANCE_PLOT_PATH = ANALYSIS_OUTPUT_DIR / "variance_explained.png"


def prepare_merfish() -> pd.DataFrame:
    """Build the clean MERFISH table with location and a few expression PCs."""
    adata = ad.read_h5ad(REPO_ROOT / "ec_obj_imputed_log2.h5ad")

    keep_mask = (
        adata.obs["x_ccf"].notna()
        & adata.obs["y_ccf"].notna()
        & adata.obs["z_ccf"].notna()
        & adata.obs["subclass"].notna()
        & adata.obs["supertype"].notna()
    )
    keep_adata = adata[keep_mask].copy()

    n_pcs = min(PCA_PLOT_COMPONENTS, keep_adata.n_obs - 1, keep_adata.n_vars)
    sc.pp.pca(keep_adata, n_comps=n_pcs)

    sc.pl.pca_variance_ratio(keep_adata, n_pcs=n_pcs, log=True, show=False)
    ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)
    fig = plt.gcf()
    fig.savefig(PCA_VARIANCE_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    keep = keep_adata.obs.copy().reset_index(names="cell_id")
    pc_df = pd.DataFrame(
        keep_adata.obsm["X_pca"][:, :N_GENE_PCS],
        columns=[f"expr_pc_{i + 1}" for i in range(N_GENE_PCS)],
        index=keep.index,
    )
    keep = pd.concat([keep, pc_df], axis=1)

    keep = keep[
        [
            "cell_id",
            "brain_section_label",
            "structure",
            "substructure",
            "class",
            "subclass",
            "x_ccf",
            "y_ccf",
            "z_ccf",
            "supertype",
            *[f"expr_pc_{i + 1}" for i in range(N_GENE_PCS)],
        ]
    ].sort_values("cell_id")

    return keep


def prepare_snr() -> pd.DataFrame:
    """Build the clean SNR table used by the analysis scripts."""
    df = pd.read_csv(REPO_ROOT / "master_detailed_comment.csv")
    endpoint_cols = [col for col in df.columns if col.endswith("_endpoint")]

    keep = df[
        df["proj"].notna()
        & df["x"].notna()
        & df["y"].notna()
        & df["z"].notna()
    ].copy()

    keep = keep[
        [
            "neuron_ID",
            "mouseID",
            "injection",
            "x",
            "y",
            "z",
            "proj",
            *endpoint_cols,
        ]
    ].sort_values("neuron_ID")

    return keep


if __name__ == "__main__":
    OUTDIR.mkdir(exist_ok=True)

    merfish = prepare_merfish()
    snr = prepare_snr()

    merfish.to_csv(OUTDIR / "merfish_supertype_location_table.csv", index=False)
    snr.to_csv(OUTDIR / "snr_proj_location_table.csv", index=False)
