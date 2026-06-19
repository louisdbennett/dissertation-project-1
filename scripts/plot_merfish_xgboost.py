import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from merfish_utils import build_selected_model, prepare_filtered_data


OUTPUT_DIR = Path("analysis_outputs/merfish_prediction")
PD_OUTPUT_PATH = OUTPUT_DIR / "xgboost_partial_dependence.png"
SCATTER_OUTPUT_PATH = OUTPUT_DIR / "xgboost_scatter_plots.png"
DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.1,
}
N_GRID_POINTS = 80
N_PD_ROWS = 2000

FEATURE_LABELS = {
    "x_ccf": "x",
    "y_ccf": "y",
    "z_ccf": "z",
}
SCATTER_PAIRS = [
    ("x_ccf", "y_ccf", "X-Y"),
    ("x_ccf", "z_ccf", "X-Z"),
    ("y_ccf", "z_ccf", "Y-Z"),
]


def format_label(label: str) -> str:
    if label == "other":
        return "Other"
    parts = label.split(" ", 1)
    clean = parts[1] if len(parts) == 2 and parts[0].isdigit() else label
    return clean.replace("_", " ")


def average_partial_dependence(
    model,
    x: pd.DataFrame,
    class_order: list[str],
    feature: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Average the fitted class probabilities over a grid of one feature."""
    feature_grid = np.linspace(x[feature].min(), x[feature].max(), N_GRID_POINTS)
    if len(x) > N_PD_ROWS:
        keep_idx = np.linspace(0, len(x) - 1, N_PD_ROWS, dtype=int)
        x_base = x.iloc[keep_idx].copy()
    else:
        x_base = x.copy()
    rows = []

    for feature_value in feature_grid:
        x_grid = x_base.copy()
        x_grid[feature] = feature_value
        mean_prob = model.predict_proba(x_grid).mean(axis=0)
        rows.append(mean_prob)

    return feature_grid, pd.DataFrame(rows, columns=class_order)


def make_plot(
    n_estimators: int = DEFAULT_PARAMS["n_estimators"],
    max_depth: int = DEFAULT_PARAMS["max_depth"],
    learning_rate: float = DEFAULT_PARAMS["learning_rate"],
) -> tuple[Path, Path]:
    """Fit the current MERFISH XGBoost model and save simple location plots."""
    prepared = prepare_filtered_data()
    model = build_selected_model(
        "xgboost",
        len(prepared["class_order"]),
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    model.fit(prepared["x"], prepared["y"])

    x = prepared["x"].copy()
    class_order = prepared["class_order"]
    label_counts = prepared["label_counts"]

    pd_results = {
        feature: average_partial_dependence(model, x, class_order, feature)
        for feature in FEATURE_LABELS
    }

    predicted_idx = model.predict(x)
    predicted_labels = pd.Series(predicted_idx).map(dict(enumerate(class_order)))
    plot_df = prepared["data"][["x_ccf", "y_ccf", "z_ccf"]].reset_index(drop=True).copy()
    plot_df["predicted_label"] = predicted_labels

    label_order = label_counts.index.tolist()
    cmap = plt.get_cmap("tab10" if len(label_order) <= 10 else "tab20", len(label_order))
    color_map = {
        label: ("#d9d9d9" if label == "other" else cmap(i))
        for i, label in enumerate(label_order)
    }

    pd_fig, pd_axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, feature in zip(pd_axes, FEATURE_LABELS):
        feature_grid, pd_df = pd_results[feature]
        for label in label_order:
            ax.plot(
                feature_grid,
                pd_df[label],
                label=format_label(label),
                color=color_map[label],
                linewidth=2.2 if label != "other" else 1.8,
                alpha=0.95 if label != "other" else 0.85,
            )
        ax.set_xlabel(FEATURE_LABELS[feature])
        ax.set_ylabel("Mean predicted probability")
        ax.set_title(f"partial dependence of {feature[0]}")

    pd_handles, pd_labels = pd_axes[0].get_legend_handles_labels()
    pd_fig.legend(
        pd_handles,
        pd_labels,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
    )
    pd_fig.suptitle(
        "MERFISH supertype classifier partial dependence",
        y=1.02,
    )
    pd_fig.tight_layout(rect=(0, 0, 0.88, 1))

    scatter_fig, scatter_axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (x_col, y_col, title) in zip(scatter_axes, SCATTER_PAIRS):
        for label in label_order:
            group_df = plot_df[plot_df["predicted_label"] == label]
            ax.scatter(
                group_df[x_col],
                group_df[y_col],
                s=10,
                alpha=0.35 if label != "other" else 0.18,
                color=color_map[label],
                linewidths=0,
                label=format_label(label),
            )
        ax.set_xlabel(FEATURE_LABELS[x_col])
        ax.set_ylabel(FEATURE_LABELS[y_col])
        ax.set_title(title)

    scatter_handles, scatter_labels = scatter_axes[0].get_legend_handles_labels()
    scatter_fig.legend(
        scatter_handles,
        scatter_labels,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        markerscale=2,
    )
    scatter_fig.suptitle(
        "MERFISH predicted supertypes",
        y=1.02,
    )
    scatter_fig.tight_layout(rect=(0, 0, 0.88, 1))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd_fig.savefig(PD_OUTPUT_PATH, dpi=200, bbox_inches="tight")
    scatter_fig.savefig(SCATTER_OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(pd_fig)
    plt.close(scatter_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_PARAMS["n_estimators"])
    parser.add_argument("--max-depth", type=int, default=DEFAULT_PARAMS["max_depth"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_PARAMS["learning_rate"])
    args = parser.parse_args()

    make_plot(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
