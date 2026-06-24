# Dissertation Project

This repository contains the code for a dissertation project on linking neuron's projections with their genetic makeup.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Add the raw data files to the repository root before running anything.

## Run Order

### 1. Prepare the local tables

```bash
./.venv/bin/python scripts/prepare_tables.py
```

### 2. Build the SNR projection clusters

```bash
./.venv/bin/python scripts/snr_clustering.py
```

### 3. Fit the MERFISH model

```bash
./.venv/bin/python scripts/merfish_model_comparison.py
```

### 4. Run the best MERFISH model on the SNR data

```bash
./.venv/bin/python scripts/snr_classification_by_group.py --model xgboost --n-estimators 500 --max-depth 4 --learning-rate 0.1
```

### 5. Test whether the predicted supertypes differ by SNR group

```bash
./.venv/bin/python scripts/snr_permanova.py
```

### 6. Produce the main plots

```bash
./.venv/bin/python scripts/plot_snr_soma.py
./.venv/bin/python scripts/plot_snr_group_probabilities.py
./.venv/bin/python scripts/plot_snr_probability_distributions.py
./.venv/bin/python scripts/plot_brain_coverage.py --z-filtered --show-grouped-supertypes
./.venv/bin/python scripts/plot_merfish_xgboost.py
./.venv/bin/python scripts/plot_merfish_supertype_frequency.py
```

### 7. Perform DE 

```bash
./.venv/bin/python scripts/merfish_differential_expression.py
```