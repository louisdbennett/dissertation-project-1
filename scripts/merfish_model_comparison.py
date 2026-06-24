from pathlib import Path

import pandas as pd
import warnings
from scipy.linalg import LinAlgWarning

from merfish_utils import (
    GRID_MODELS,
    prepare_filtered_data,
    screen_model_grid,
)
from merfish_prediction import run_analysis

MODEL_COMPARISON_OUTPUT = Path("analysis_outputs/model_comparison")

warnings.filterwarnings("ignore", category=LinAlgWarning, module="sklearn.discriminant_analysis")


def run_model_comparison() -> pd.DataFrame:
    """
    Run a test/train split on each hyperparameter to pick the best
    performing of each of the models then compare this on a cross
    validation dataset.
    """
    prepared = prepare_filtered_data()
    final_results = []

    for model_name in GRID_MODELS:
        screen_results = screen_model_grid(
            model_name=model_name,
            x=prepared["x"],
            y=prepared["y"],
            class_labels=prepared["class_labels"],
            num_classes=len(prepared["class_order"]),
        )
        best_setting = screen_results.iloc[0].to_dict()

        best_params = {
            key: value
            for key, value in best_setting.items()
            if key not in {"model", "holdout_accuracy", "holdout_log_loss"}
            and not pd.isna(value)
        }
        cv_results = run_analysis(model_name=model_name, **best_params)
        final_results.append(
            {
                "model": model_name,
                **best_params,
                "pooled_accuracy": cv_results["metrics"]["pooled_accuracy"],
                "pooled_log_loss": cv_results["metrics"]["pooled_log_loss"],
            }
        )

    return pd.DataFrame(final_results).sort_values(
        by=["pooled_log_loss", "pooled_accuracy"],
        ascending=[True, False],
    ).reset_index(drop=True)


if __name__ == "__main__":
    final_results = run_model_comparison()
    MODEL_COMPARISON_OUTPUT.mkdir(parents=True, exist_ok=True)
    final_results.to_csv(MODEL_COMPARISON_OUTPUT / "merfish_model_comparison.csv", index=False)
    print(final_results)
