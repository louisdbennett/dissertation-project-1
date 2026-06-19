"""
Shared MERFISH analysis utilities.
"""

import argparse
import os
from itertools import product
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, make_scorer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "analysis_tables" / "merfish_supertype_location_table.csv"
SNR_DATA_PATH = REPO_ROOT / "analysis_tables" / "snr_proj_location_table.csv"

FEATURES = ["x_ccf", "y_ccf", "z_ccf"]
TARGET = "supertype_grouped"

# lets us do some sensitivity analysis on this later
SUPERTYPE_COVERAGE = float(os.environ.get("SUPERTYPE_COVERAGE", "0.5"))

N_SPLITS = 5
SCREEN_TEST_SIZE = 0.2
SCREEN_RANDOM_STATE = 0

# define the optimisation grids
GRID_MODELS = ["logistic", "lda", "qda", "knn", "xgboost"]
MODEL_GRIDS = {
    "logistic": {"C": [0.03, 0.1, 0.3, 1.0]},
    "lda": {},
    "qda": {},
    "knn": {
        "n_neighbors": [5, 15, 31, 61],
        "weights": ["uniform", "distance"],
    },
    "xgboost": {
        "n_estimators": [100, 300, 500],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.1],
    },
}

FILTER_TO_SNR_Z_RANGE = True

def build_model_logistic(C: float = 0.1) -> Pipeline:
    """Build the default scaled multinomial logistic regression pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=1000, solver="saga")),
        ]
    )


def build_model_lda() -> Pipeline:
    """Build a scaled linear discriminant analysis pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearDiscriminantAnalysis()),
        ]
    )


def build_model_qda() -> Pipeline:
    """Build a scaled quadratic discriminant analysis pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", QuadraticDiscriminantAnalysis()),
        ]
    )


def build_model_knn(
    n_neighbors: int = 15,
    weights: str = "uniform",
) -> Pipeline:
    """Build a scaled k-nearest neighbours pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
        ]
    )


def build_model_xgboost(
    num_classes: int,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
) -> XGBClassifier:
    """Build an XGBoost classifier for binary or multiclass MERFISH comparisons."""
    common_params = {
        "eval_metric": "mlogloss",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    if num_classes <= 2:
        return XGBClassifier(
            objective="binary:logistic",
            **common_params,
        )
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        **common_params,
    )
    
def build_selected_model(model_name: str, num_classes: int, **model_params):
    """Build one supported model family from its short name and parameters."""
    if model_name == "logistic":
        return build_model_logistic(**model_params)
    if model_name == "lda":
        return build_model_lda()
    if model_name == "qda":
        return build_model_qda()
    if model_name == "knn":
        return build_model_knn(**model_params)
    if model_name == "xgboost":
        return build_model_xgboost(num_classes, **model_params)
    raise ValueError(f"Unknown model_name: {model_name}")


def add_model_cli_args(parser: argparse.ArgumentParser, default_model: str = "logistic") -> None:
    """Add the shared model-selection CLI arguments to a parser."""
    parser.add_argument("--model", choices=GRID_MODELS, default=default_model)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)
    parser.add_argument("--weights", choices=["uniform", "distance"], default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)


def get_model_cli_params(args: argparse.Namespace) -> dict[str, object]:
    """Collect the shared model hyperparameters from parsed CLI arguments."""
    return {
        key: value
        for key, value in {
            "C": args.c,
            "n_neighbors": args.n_neighbors,
            "weights": args.weights,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        }.items()
        if value is not None
    }


def prepare_filtered_data(
    target: str = TARGET,
    min_count: Optional[int] = None,
    class_name: Optional[str] = None,
    filter_to_snr_z_range: bool = FILTER_TO_SNR_Z_RANGE,
) -> dict[str, object]:
    """
    Load the MERFISH table and optionally filter by class and SNR z range
    Creates the chosen target for modelling, if its the supertype groups
    then will collapse these. 
    """
    df = pd.read_csv(DATA_PATH, dtype={"cell_id": "string"})

    if class_name is not None:
        df = df[df["class"] == class_name]

    z_min = None
    z_max = None
    if filter_to_snr_z_range:
        snr = pd.read_csv(SNR_DATA_PATH)
        z_min = snr["z"].min()
        z_max = snr["z"].max()
        df = df[(df["z_ccf"] >= z_min) & (df["z_ccf"] <= z_max)]

    if target == 'supertype_grouped':
        df, class_order, label_counts = group_supertypes(df)
    else:
        class_counts = df[target].value_counts()
        if min_count is not None:
            keep_classes = class_counts[class_counts >= min_count].index
            df = df[df[target].isin(keep_classes)].copy()
            class_counts = class_counts[class_counts >= min_count]
        class_order = class_counts.index.tolist()
        label_counts = class_counts

    class_to_idx = {label: idx for idx, label in enumerate(class_order)}
    class_labels = np.arange(len(class_order))

    x = df[FEATURES].reset_index(drop=True)
    y = df[target].map(class_to_idx).astype(int).reset_index(drop=True)
    cv_splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True).split(x, y))

    return {
        "data": df,
        "x": x,
        "y": y,
        "cv_splits": cv_splits,
        "class_order": class_order,
        "class_labels": class_labels,
        "class_to_idx": class_to_idx,
        "label_counts": label_counts,
        "z_range": (z_min, z_max),
    }

def evaluate_model(model, x: pd.DataFrame, y: pd.Series, class_labels: np.ndarray, cv_splits):
    """
    This helper fits the model across the supplied cross-validation splits,
    collects out-of-fold predictions and probabilities, and then calculates
    pooled accuracy and log loss across the full dataset.

    Returns:
        cv_scores: dataframe with one row per CV split and columns for accuracy and log loss
        y_pred: out-of-fold hard class predictions
        y_prob: out-of-fold class probabilities
        metrics: pooled accuracy and pooled log loss across all rows
    """
    cv_scores = cross_validate(
        model,
        x,
        y,
        cv=cv_splits,
        scoring={
            "accuracy": "accuracy",
            "neg_log_loss": make_scorer(
                log_loss,
                response_method="predict_proba",
                greater_is_better=False,
                labels=class_labels,
            ),
        },
        return_train_score=False,
    )
    y_pred = cross_val_predict(model, x, y, cv=cv_splits)
    y_prob = cross_val_predict(model, x, y, cv=cv_splits, method="predict_proba")

    return {
        "cv_scores": pd.DataFrame(
            {
                "accuracy": cv_scores["test_accuracy"],
                "log_loss": -cv_scores["test_neg_log_loss"],
            }
        ),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": {
            "pooled_accuracy": accuracy_score(y, y_pred),
            "pooled_log_loss": log_loss(y, y_prob, labels=class_labels),
        },
    }

def screen_model_grid(
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    class_labels: np.ndarray,
    num_classes: int,
) -> pd.DataFrame:
    """
    This helper creates one stratified train-test split, fits each parameter
    setting for the chosen model family once, and returns the holdout results.
    It is used for the faster screening step before the slower full
    cross-validated evaluation.

    Returns:
        dataframe sorted by holdout log loss and holdout accuracy
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=SCREEN_TEST_SIZE,
        random_state=SCREEN_RANDOM_STATE,
    )

    # use one quick stratified split for screening before the slower full CV step
    train_idx, test_idx = next(splitter.split(x, y))
    x_train = x.iloc[train_idx]
    x_test = x.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    grid = MODEL_GRIDS[model_name]
    keys = list(grid.keys())
    
    # expand the configured parameter grid into one dict per setting
    param_sets = [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))] if keys else [{}]

    rows = []
    for params in param_sets:
        # fit each setting once on the train split and score it on the holdout split
        model = build_selected_model(model_name, num_classes, **params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)
        rows.append(
            {
                "model": model_name,
                **params,
                "holdout_accuracy": accuracy_score(y_test, y_pred),
                "holdout_log_loss": log_loss(y_test, y_prob, labels=class_labels),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["holdout_log_loss", "holdout_accuracy"],
        ascending=[True, False],
    ).reset_index(drop=True)


def group_supertypes(
    df: pd.DataFrame,
    coverage: float = SUPERTYPE_COVERAGE,
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Collapse MERFISH supertypes by frequency coverage into `supertype_grouped`."""
    label_counts = df["supertype"].value_counts()
    cumulative_coverage = (label_counts / label_counts.sum()).cumsum()
    keep_labels = cumulative_coverage[cumulative_coverage < coverage].index.tolist()
    if len(keep_labels) < len(label_counts):
        keep_labels.append(cumulative_coverage[cumulative_coverage >= coverage].index[0])
    collapsed = df["supertype"].where(df["supertype"].isin(keep_labels), "other")
    collapsed_order = keep_labels + ["other"]
    grouped_df = df.copy()
    grouped_df[TARGET] = collapsed
    grouped_counts = grouped_df[TARGET].value_counts().reindex(collapsed_order)
    return grouped_df, collapsed_order, grouped_counts
