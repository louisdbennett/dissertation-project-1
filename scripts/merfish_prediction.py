import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from merfish_utils import add_model_cli_args, build_selected_model, evaluate_model, get_model_cli_params, prepare_filtered_data

MODEL_OUTPUT_DIR = Path("analysis_outputs/merfish_prediction")


def run_analysis(
    model_name: str = "logistic",
    save_model_path: Path = None,
    **model_params,
) -> dict[str, object]:
    """Run the main MERFISH location-only classification analysis.

    This helper prepares the standard filtered MERFISH dataset, fits the chosen
    classifier under cross-validation, and returns the pooled metrics plus a few
    core summaries of class-level performance.

    If ``save_model_path`` is given, the chosen model is also fit on the full
    filtered MERFISH dataset and saved there.
    """
    prepared = prepare_filtered_data()
    model = build_selected_model(model_name, len(prepared["class_order"]), **model_params)

    model_result = evaluate_model(
        model,
        prepared["x"],
        prepared["y"],
        prepared["class_labels"],
        prepared["cv_splits"],
    )

    class_order = prepared["class_order"]
    y = prepared["y"]
    y_pred = model_result["y_pred"]

    conf_mat = confusion_matrix(y, y_pred, labels=np.arange(len(class_order)))
    per_class_recall = np.divide(
        np.diag(conf_mat),
        conf_mat.sum(axis=1),
        out=np.zeros(len(class_order), dtype=float),
        where=conf_mat.sum(axis=1) != 0,
    )

    metrics = {
        "pooled_accuracy": model_result["metrics"]["pooled_accuracy"],
        "pooled_log_loss": model_result["metrics"]["pooled_log_loss"],
        "rows_used": len(prepared["data"]),
        "retained_labels": len(class_order),
    }

    if save_model_path is not None:
        fitted_model = build_selected_model(model_name, len(prepared["class_order"]), **model_params)
        fitted_model.fit(prepared["x"], prepared["y"])
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted_model, save_model_path)

    return {
        "metrics": metrics,
        "cv_scores": model_result["cv_scores"],
        "per_class_recall": pd.Series(per_class_recall, index=class_order, name="recall"),
        "confusion_matrix": pd.DataFrame(conf_mat, index=class_order, columns=class_order),
        "model_path": save_model_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_cli_args(parser, default_model="logistic")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    model_params = get_model_cli_params(args)
    save_model_path = None
    if args.save_model:
        save_model_path = MODEL_OUTPUT_DIR / f"{args.model}_model.joblib"

    results = run_analysis(
        model_name=args.model,
        save_model_path=save_model_path,
        **model_params,
    )
    print(results["metrics"])
