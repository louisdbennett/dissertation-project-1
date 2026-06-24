import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from merfish_utils import build_selected_model, prepare_filtered_data


OUTPUT_DIR = Path("analysis_outputs/merfish_prediction")
PD_OUTPUT_PATH = OUTPUT_DIR / "xgboost_partial_dependence.png"
SCATTER_OUTPUT_PATH = OUTPUT_DIR / "xgboost_scatter_plots.png"
ACTUAL_SCATTER_OUTPUT_PATH = OUTPUT_DIR / "merfish_actual_scatter_plots.png"
CONFUSION_MATRIX_OUTPUT_PATH = OUTPUT_DIR / "xgboost_confusion_matrix.png"
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

def save_scatter_plot(
    plot_df: pd.DataFrame,
    label_col: str,
    label_order: list[str],
    color_map: dict[str, object],
    output_path: Path,
) -> None:
    """Save one 3-panel MERFISH scatter plot for a given label column."""
    scatter_fig, scatter_axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (x_col, y_col, title) in zip(scatter_axes, SCATTER_PAIRS):
        for label in label_order:
            group_df = plot_df[plot_df[label_col] == label]
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
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=max(2, len(scatter_labels)),
        markerscale=2,
    )
    scatter_fig.tight_layout(rect=(0, 0.08, 1, 1))
    scatter_fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(scatter_fig)


def collect_cv_predictions(
    x: pd.DataFrame,
    y: pd.Series,
    cv_splits,
    num_classes: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> np.ndarray:
    """Collect out-of-fold XGBoost predictions across the configured CV splits."""
    y_pred = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in cv_splits:
        model = build_selected_model(
            "xgboost",
            num_classes,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        y_pred[test_idx] = model.predict(x.iloc[test_idx]).astype(int)

    return y_pred


def save_confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_order: list[str],
    output_path: Path,
) -> None:
    """Save a simple row-normalized confusion matrix plot."""
    conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_order)))
    row_totals = conf_mat.sum(axis=1)
    conf_mat_norm = np.divide(
        conf_mat.astype(float),
        row_totals[:, None],
        out=np.zeros_like(conf_mat, dtype=float),
        where=row_totals[:, None] != 0,
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(conf_mat_norm, cmap="Greens", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = [format_label(label).replace(" ENT Glut ", "\nENT Glut ") for label in class_order]
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for tick in ax.get_xticklabels():
        tick.set_ha("center")

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            value = conf_mat_norm[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_plot(
    n_estimators: int = DEFAULT_PARAMS["n_estimators"],
    max_depth: int = DEFAULT_PARAMS["max_depth"],
    learning_rate: float = DEFAULT_PARAMS["learning_rate"],
) -> tuple[Path, Path, Path, Path]:
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
    plot_df["actual_label"] = prepared["data"]["supertype_grouped"].reset_index(drop=True)
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
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=max(2, len(pd_labels)),
    )
    pd_fig.tight_layout(rect=(0, 0.08, 1, 1))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd_fig.savefig(PD_OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(pd_fig)
    save_scatter_plot(plot_df, "predicted_label", label_order, color_map, SCATTER_OUTPUT_PATH)
    save_scatter_plot(plot_df, "actual_label", label_order, color_map, ACTUAL_SCATTER_OUTPUT_PATH)
    cv_pred = collect_cv_predictions(
        prepared["x"],
        prepared["y"],
        prepared["cv_splits"],
        len(class_order),
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    save_confusion_matrix_plot(prepared["y"], cv_pred, class_order, CONFUSION_MATRIX_OUTPUT_PATH)

    return (
        PD_OUTPUT_PATH,
        SCATTER_OUTPUT_PATH,
        ACTUAL_SCATTER_OUTPUT_PATH,
        CONFUSION_MATRIX_OUTPUT_PATH,
    )


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
