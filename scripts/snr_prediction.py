from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, make_scorer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_PATH = Path("analysis_tables/snr_proj_location_table.csv")
FEATURES = ["x", "y", "z"]
TARGET = "proj"
CLASS_ORDER = ["rsp", "rsp_orb", "orb"]
CLASS_LABELS = list(range(len(CLASS_ORDER)))


def build_model_logistic() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(solver="lbfgs", max_iter=2000)),
        ]
    )


def build_model_lda() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearDiscriminantAnalysis()),
        ]
    )


def build_model_qda() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", QuadraticDiscriminantAnalysis()),
        ]
    )


def build_model_knn() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]
    )


def build_model_xgboost() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(CLASS_ORDER),
    )


def get_model(model_name: str) -> Pipeline:
    if model_name == "logistic":
        return build_model_logistic()
    if model_name == "lda":
        return build_model_lda()
    if model_name == "qda":
        return build_model_qda()
    if model_name == "knn":
        return build_model_knn()
    if model_name == "xgboost":
        return build_model_xgboost()
    raise ValueError(f"Unknown model_name: {model_name}")


def run_analysis(model_name: str = "logistic") -> dict[str, object]:
    """Run the grouped SNR location-only classification analysis.

    This helper loads the SNR table, encodes projection labels, evaluates the
    chosen classifier with grouped cross-validation by ``mouseID``, and returns
    the pooled metrics plus the main confusion summaries.

    Returns:
        metrics: pooled accuracy and pooled log loss
        cv_scores: one row per CV split with accuracy and log loss
        confusion_matrix: confusion matrix across projection labels
        confusion_matrix_row_normalized: row-normalized confusion matrix
    """
    df = pd.read_csv(DATA_PATH)
    x = df[FEATURES]
    y = df[TARGET].map({label: idx for idx, label in enumerate(CLASS_ORDER)}).astype(int)
    groups = df["mouseID"]

    model = get_model(model_name)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True)

    cv_scores = cross_validate(
        model,
        x,
        y,
        cv=cv,
        groups=groups,
        scoring={
            "accuracy": "accuracy",
            "neg_log_loss": make_scorer(
                log_loss,
                response_method="predict_proba",
                greater_is_better=False,
                labels=CLASS_LABELS,
            ),
        },
        return_train_score=False,
    )
    y_pred = cross_val_predict(model, x, y, cv=cv, groups=groups)
    y_prob = cross_val_predict(model, x, y, cv=cv, groups=groups, method="predict_proba")

    conf_mat = confusion_matrix(y, y_pred, labels=CLASS_LABELS)
    conf_mat_norm = confusion_matrix(y, y_pred, labels=CLASS_LABELS, normalize="true")
    metrics = {
        "pooled_accuracy": accuracy_score(y, y_pred),
        "pooled_log_loss": log_loss(y, y_prob, labels=CLASS_LABELS),
    }

    return {
        "metrics": metrics,
        "cv_scores": pd.DataFrame(
            {
                "accuracy": cv_scores["test_accuracy"],
                "log_loss": -cv_scores["test_neg_log_loss"],
            }
        ),
        "confusion_matrix": pd.DataFrame(conf_mat, index=CLASS_ORDER, columns=CLASS_ORDER),
        "confusion_matrix_row_normalized": pd.DataFrame(
            conf_mat_norm, index=CLASS_ORDER, columns=CLASS_ORDER
        ),
    }

if __name__ == "__main__":
    run_analysis()
