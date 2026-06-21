from pathlib import Path

import anndata as ad
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "analysis_tables"


def prepare_merfish() -> pd.DataFrame:
    """Build the clean MERFISH table with the location and label columns used in the analysis."""
    adata = ad.read_h5ad(REPO_ROOT / "ec_obj_imputed_log2.h5ad")

    keep_mask = (
        adata.obs["x_ccf"].notna()
        & adata.obs["y_ccf"].notna()
        & adata.obs["z_ccf"].notna()
        & adata.obs["subclass"].notna()
        & adata.obs["supertype"].notna()
    )
    keep_adata = adata[keep_mask].copy()

    keep = keep_adata.obs.copy().reset_index(names="cell_id")

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
        ]
    ].sort_values("cell_id")

    return keep


def prepare_snr() -> pd.DataFrame:
    """Build the clean SNR table used by the analysis scripts."""
    df = pd.read_csv(REPO_ROOT / "master_detailed_comment.csv")
    endpoint_cols = [col for col in df.columns if col.endswith("_endpoint")]

    keep = df[
        df["x"].notna()
        & df["y"].notna()
        & df["z"].notna()
    ].copy()

    keep = keep[
        [
            "neuron_ID",
            "mouseID",
            "injection",
            "comment",
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
