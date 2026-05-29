import os
import json
import argparse
import shutil
from pathlib import Path
from contextlib import nullcontext

import joblib
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def get_run_context(run_name: str):
    """
    Safe for:
    - mlflow run .
    - python modelling.py

    Avoid nested MLflow run conflicts.
    """

    active_run = mlflow.active_run()

    if active_run is not None:
        print(f"Using existing MLflow run: {active_run.info.run_id}")
        return nullcontext()

    print("Starting new MLflow run")
    return mlflow.start_run(run_name=run_name)


def make_confusion_matrix_plot(cm, labels, out_path: Path, title: str):
    fig = plt.figure(figsize=(6, 5))

    plt.imshow(cm, interpolation="nearest")

    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))

    plt.xticks(
        tick_marks,
        labels,
        rotation=45,
        ha="right"
    )

    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.tight_layout()

    fig.savefig(out_path, dpi=180)

    plt.close(fig)


def infer_problem_type(y: pd.Series):
    return "binary" if y.nunique(dropna=True) == 2 else "multiclass"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV dataset"
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target column name"
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory for artifacts"
    )

    parser.add_argument(
        "--run_name",
        default="logreg-autolog-ci"
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42
    )

    args = parser.parse_args()

    # =========================================================
    # MLflow Tracking
    # =========================================================

    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # =========================================================
    # Output directory
    # =========================================================

    data_path = Path(args.data)

    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Load dataset
    # =========================================================

    df = pd.read_csv(data_path)

    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found"
        )

    df = df.dropna(subset=[args.target]).copy()

    # =========================================================
    # Convert bool columns to string
    # Avoid sklearn SimpleImputer bool issue
    # =========================================================

    bool_columns = df.select_dtypes(
        include=["bool"]
    ).columns

    for col in bool_columns:
        df[col] = df[col].astype(str)

    X = df.drop(columns=[args.target])

    y = df[args.target]

    # =========================================================
    # Train test split
    # =========================================================

    problem_type = infer_problem_type(y)

    stratify = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    # =========================================================
    # Feature type detection
    # =========================================================

    cat_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    num_cols = X_train.select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    ).columns.tolist()

    print("Categorical columns:", cat_cols)
    print("Numeric columns:", num_cols)

    # =========================================================
    # Preprocessing pipelines
    # =========================================================

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                num_cols
            ),
            (
                "cat",
                categorical_transformer,
                cat_cols
            ),
        ],
        remainder="drop",
    )

    # =========================================================
    # Model
    # =========================================================

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    # =========================================================
    # MLflow autolog
    # =========================================================

    mlflow.autolog()

    # =========================================================
    # Training
    # =========================================================

    with get_run_context(run_name=args.run_name):

        print("Training model...")

        model.fit(X_train, y_train)

        # =====================================================
        # Prediction
        # =====================================================

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        f1 = f1_score(
            y_test,
            y_pred,
            average="binary"
            if problem_type == "binary"
            else "weighted",
        )

        prec = precision_score(
            y_test,
            y_pred,
            average="binary"
            if problem_type == "binary"
            else "weighted",
            zero_division=0,
        )

        rec = recall_score(
            y_test,
            y_pred,
            average="binary"
            if problem_type == "binary"
            else "weighted",
            zero_division=0,
        )

        roc_auc = None

        try:

            if hasattr(model, "predict_proba"):

                proba = model.predict_proba(X_test)

                if problem_type == "binary":

                    roc_auc = roc_auc_score(
                        y_test,
                        proba[:, 1]
                    )

                else:

                    roc_auc = roc_auc_score(
                        y_test,
                        proba,
                        multi_class="ovr"
                    )

        except Exception as e:

            print(f"ROC AUC skipped: {e}")

        # =====================================================
        # Save local artifacts
        # =====================================================

        # 1. Simpan format bawaan Anda (Joblib)
        model_path = out_dir / "model.joblib"
        joblib.dump(model, model_path)

        # 2. TAMBAHAN BARU: Simpan model format MLflow ke direktori lokal
        # Ini akan membuat folder artifacts/latest/model
        mlflow_model_dir = out_dir / "model"
        if mlflow_model_dir.exists():
            shutil.rmtree(mlflow_model_dir) # Hapus folder lama agar tidak error
        mlflow.sklearn.save_model(model, str(mlflow_model_dir))

        # 3. Simpan metrik & log
        metrics = {
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
        }

        if roc_auc is not None:
            metrics["roc_auc"] = float(roc_auc)

        metrics_path = out_dir / "metrics.json"

        metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        report = classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )

        report_path = (
            out_dir / "classification_report.txt"
        )

        report_path.write_text(
            report,
            encoding="utf-8",
        )

        labels = sorted(
            pd.unique(
                pd.concat([
                    pd.Series(y_test),
                    pd.Series(y_pred),
                ])
            )
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=labels,
        )

        cm_path = (
            out_dir / "confusion_matrix.png"
        )

        make_confusion_matrix_plot(
            cm,
            labels,
            cm_path,
            title="Confusion Matrix",
        )

        # =====================================================
        # Explicit MLflow artifact logging
        # =====================================================

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(cm_path))

        print("Training completed successfully")
        print(f"Artifacts saved to: {out_dir}")
        print("Metrics:", metrics)


if __name__ == "__main__":
    main()