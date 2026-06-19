import argparse
from pathlib import Path

import pandas as pd

from merfish_utils import (
    FEATURES,
    SNR_DATA_PATH,
    add_model_cli_args,
    build_selected_model,
    get_model_cli_params,
    prepare_filtered_data
)

OUTPUT_DIR = Path("analysis_outputs/snr_classification")

def fit_grouped_supertypes(
    model_name: str = "logistic",
    snr_group_col: str = "proj",
    **model_params,
) -> dict[str, object]:
    """Fit the grouped MERFISH supertype model and apply it to SNR coordinates."""
    prepared = prepare_filtered_data()
    snr = pd.read_csv(SNR_DATA_PATH)

    fitted_model = build_selected_model(model_name, len(prepared["class_order"]), **model_params)

    fitted_model.fit(prepared["x"], prepared["y"])

    x_snr = snr.rename(columns={"x": "x_ccf", "y": "y_ccf", "z": "z_ccf"})[FEATURES]
    prob_df = pd.DataFrame(
        fitted_model.predict_proba(x_snr),
        columns=prepared["class_order"],
    )
    pred_idx = fitted_model.predict(x_snr)
    pred_labels = pd.Series(pred_idx).map(dict(enumerate(prepared["class_order"]))).rename("predicted_label")

    summary_by_group = prob_df.assign(**{snr_group_col: snr[snr_group_col]}).groupby(snr_group_col).mean().reset_index()
    full_output = pd.concat(
        [
            snr.reset_index(drop=True),
            pred_labels.reset_index(drop=True),
            prob_df.reset_index(drop=True),
        ],
        axis=1,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.csv"
    full_output_path = OUTPUT_DIR / "full_probabilities.csv"
    summary_by_group.to_csv(summary_path, index=False)
    full_output.to_csv(full_output_path, index=False)

    return {"n_labels": len(prepared["class_order"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_cli_args(parser, default_model="logistic")
    parser.add_argument("--snr-group-col", default="proj")
    args = parser.parse_args()

    model_params = get_model_cli_params(args)
    results = fit_grouped_supertypes(
        model_name=args.model,
        snr_group_col=args.snr_group_col,
        **model_params,
    )
