import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON_BIN = ROOT_DIR / ".venv" / "bin" / "python"
OUTPUT_DIR = ROOT_DIR / "analysis_outputs" / "snr_classification"
RESULTS_PATH = OUTPUT_DIR / "coverage_sensitivity.csv"
COVERAGE_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for coverage in COVERAGE_VALUES:
        env = os.environ.copy()
        env["SUPERTYPE_COVERAGE"] = str(coverage)

        subprocess.run(
            [str(PYTHON_BIN), str(ROOT_DIR / "scripts" / "snr_classification_by_group.py")],
            check=True,
            stdout=subprocess.DEVNULL,
            env=env,
        )
        subprocess.run(
            [str(PYTHON_BIN), str(ROOT_DIR / "scripts" / "snr_permanova.py")],
            check=True,
            stdout=subprocess.DEVNULL,
            env=env,
        )

        n_labels = len(pd.read_csv(OUTPUT_DIR / "summary.csv").columns) - 1
        result = pd.read_csv(OUTPUT_DIR / "permanova_global.csv").iloc[0]
        rows.append(
            {
                "coverage": coverage,
                "n_labels": n_labels,
                "pseudo_f": result["pseudo_f"],
                "r2": result["r2"],
                "p_value": result["p_value"],
                "significant_0_05": result["p_value"] < 0.05,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
