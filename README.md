# Dissertation Project

This repository contains the code for a dissertation project on linking neuron's projections with their genetic makeup.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Order

Add the raw data files to the root of the repository and run the scripts in the following order.

### 1. Prepare the local tables

```bash
./.venv/bin/python scripts/prepare_tables.py
```

### 2. Fit the MERFISH model

```bash
./.venv/bin/python scripts/merfish_model_comparison.py
```

### 4. Run the MERFISH model on the SNR data

```bash
./.venv/bin/python scripts/snr_classification_by_group.py --model xgboost --n-estimators 500 --max-depth 4 --learning-rate 0.1
```

### 5. Test whether the predicted supertypes differ by SNR group

```bash
./.venv/bin/python scripts/snr_permanova.py
```

### 6. Create plots

```bash
./.venv/bin/python scripts/plot_snr_group_probabilities.py
./.venv/bin/python scripts/plot_brain_coverage.py --z-filtered --show-grouped-supertypes
./.venv/bin/python scripts/plot_merfish_xgboost.py
```
