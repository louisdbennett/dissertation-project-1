from pathlib import Path
import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "ec_obj_imputed_log2.h5ad"
CELL_TABLE_PATH = REPO_ROOT / "analysis_tables" / "merfish_de_cell_table.csv"
OUTPUT_DIR = REPO_ROOT / "analysis_outputs" / "merfish_differential_expression"
TARGET_SUPERTYPE = "0036 L2/3 IT ENT Glut_4"
DE_METHOD = "wilcoxon"


def make_output_stem(target_supertype: str) -> str:
    """Build a simple output stem from the target supertype."""
    safe_name = (
        target_supertype.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )
    return f"{safe_name}_vs_class"


def extract_rank_genes_groups(adata: ad.AnnData, group: str) -> pd.DataFrame:
    """Extract one Scanpy rank_genes_groups result table."""
    result = sc.get.rank_genes_groups_df(adata, group=group)
    return result.rename(
        columns={
            "names": "gene_id",
            "scores": "score",
            "logfoldchanges": "logfoldchange",
            "pvals": "p_value",
            "pvals_adj": "p_value_adj",
        }
    )


def run_differential_expression(
    target_supertype: str = TARGET_SUPERTYPE,
) -> pd.DataFrame:
    """Run one-vs-rest differential expression for one MERFISH supertype within its class."""
    cells = pd.read_csv(CELL_TABLE_PATH, dtype={"cell_id": "string"})
    target_rows = cells[cells["supertype"] == target_supertype]
    if target_rows.empty:
        raise ValueError(f"Target supertype not found: {target_supertype}")

    target_class = target_rows["class"].iloc[0]
    cells = cells[cells["class"] == target_class].copy()

    adata = ad.read_h5ad(DATA_PATH)
    subset = adata[cells["cell_id"].tolist()].copy()

    cell_meta = cells.set_index("cell_id").loc[subset.obs_names].copy()
    subset.obs["de_group"] = np.where(
        cell_meta["supertype"].eq(target_supertype),
        target_supertype,
        "rest",
    )

    sc.tl.rank_genes_groups(
        subset,
        groupby="de_group",
        groups=[target_supertype],
        reference="rest",
        method=DE_METHOD,
        use_raw=False,
        pts=True,
    )

    results = extract_rank_genes_groups(subset, group=target_supertype)
    gene_map = subset.var["gene_symbol"].astype("string")
    gene_map = gene_map.where(gene_map.notna(), pd.Series(subset.var_names, index=subset.var_names))
    gene_map = gene_map.reset_index()
    gene_map.columns = ["gene_id", "gene_symbol"]
    results = results.merge(gene_map, on="gene_id", how="left")

    keep_cols = [
        "gene_symbol",
        "logfoldchange",
        "p_value_adj",
        "pct_nz_group",
        "pct_nz_reference",
    ]

    results = results[keep_cols].sort_values(
        by=["p_value_adj", "logfoldchange"],
        ascending=[True, False],
    ).reset_index(drop=True)

    stem = make_output_stem(target_supertype)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-supertype", default=TARGET_SUPERTYPE)
    args = parser.parse_args()
    results = run_differential_expression(
        target_supertype=args.target_supertype,
    )
    print(results.head(20).round(4).to_string(index=False))
