import argparse
from pathlib import Path

import pandas as pd

from merfish_utils import (
    FEATURES,
    SNR_DATA_PATH,
    add_model_cli_args,
    build_selected_model,
    evaluate_model,
    get_model_cli_params,
    prepare_filtered_data,
    summarise_snr_predictions,
)


OUTPUT_DIR = Path("analysis_outputs/snr_classification")
CLASS_MIN_COUNT = 50


def classify_snr_classes(
    model_name: str = "logistic",
    snr_group_col: str = "proj",
    **model_params,
) -> dict[str, object]:
    """Fit a MERFISH class model and apply it to SNR coordinates."""
    prepared = prepare_filtered_data(target="class", min_count=CLASS_MIN_COUNT, class_name=None)
    snr = pd.read_csv(SNR_DATA_PATH)
    model = build_selected_model(model_name, len(prepared["class_order"]), **model_params)
    fitted_model = build_selected_model(model_name, len(prepared["class_order"]), **model_params)

    evaluated = evaluate_model(
        model,
        prepared["x"],
        prepared["y"],
        prepared["class_labels"],
        prepared["cv_splits"],
    )

    fitted_model.fit(prepared["x"], prepared["y"])

    x_snr = snr.rename(columns={"x": "x_ccf", "y": "y_ccf", "z": "z_ccf"})[FEATURES]
    prob_df = pd.DataFrame(fitted_model.predict_proba(x_snr), columns=prepared["class_order"])
    pred_idx = fitted_model.predict(x_snr)
    pred_labels = pd.Series(pred_idx).map(dict(enumerate(prepared["class_order"]))).rename("predicted_label")

    group_summary = summarise_snr_predictions(snr, pred_labels, prob_df, group_col=snr_group_col)
    dominant_class = prob_df.mean().reindex(prepared["class_order"]).idxmax()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"broad_class_summary_by_{snr_group_col}.csv"
    group_summary.to_csv(output_path, index=False)

    return {
        "metrics": evaluated["metrics"],
        "dominant_class": dominant_class,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_cli_args(parser, default_model="logistic")
    parser.add_argument("--snr-group-col", default="proj")
    args = parser.parse_args()

    model_params = get_model_cli_params(args)
    results = classify_snr_classes(
        model_name=args.model,
        snr_group_col=args.snr_group_col,
        **model_params,
    )
    print(
        {
            "pooled_accuracy": results["metrics"]["pooled_accuracy"],
            "pooled_log_loss": results["metrics"]["pooled_log_loss"],
            "dominant_class": results["dominant_class"],
        }
    )
