# AGENTS.md

## Project Summary

This repository contains a dissertation project on linking:
- **SNR projection group**
- **MERFISH transcriptomic identity**

The broad aim is to understand genetic makeup by projection region.

## Working Style

Keep the project simple and readable.

Preferred style:
- take small steps
- use plain English
- avoid unnecessary abstraction
- avoid large refactors unless they clearly improve clarity
- prefer conservative interpretation over ambitious claims
- do not use location to justify target design choices

## Data And Sharing Constraints

This repository should be shareable without datasets.

Treat these as local-only:
- `ec_obj_imputed_log2.h5ad`
- `master_detailed_comment.csv`
- `analysis_tables/*.csv`
- `analysis_outputs/`

If code depends on those files, make that clear in the script or notes.

## Main Files

- `README.md`
  - short overview and suggested run order
- `docs/running_notes.md`
  - current project memory and conclusions
- `scripts/prepare_tables.py`
  - builds the local analysis tables
- `scripts/snr_prediction.py`
  - grouped SNR baseline
- `scripts/merfish_prediction.py`
  - MERFISH location-only prediction
- `scripts/merfish_model_comparison.py`
  - MERFISH model comparison
- `scripts/snr_classify_classes.py`
  - broad MERFISH class transfer check
- `scripts/snr_classification_by_group.py`
  - grouped MERFISH transfer to SNR
- `scripts/snr_permanova.py`
  - composition test across SNR groups
