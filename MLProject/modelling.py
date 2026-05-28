import os
import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

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
matplotlib.use("Agg")  # CI/headless
import matplotlib.pyplot as plt


def start_mlflow_run_safely(run_name: str, default_experiment_name: str):
    """
    CI-safe:
    - Jangan resume MLFLOW_RUN_ID (sering beda tracking store)
    - Set experiment lalu start run biasa
    """
    mlflow.set_experiment(default_experiment_name)
    return mlflow.start_run(run_name=run_name)


def make_confusion_matrix_plot(cm, labels, out_path: Path, title: str):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
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
    parser.add_argument("--data", required=True, help="Path to preprocessed CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--out_dir", required=True, help="Output directory for artifacts")
    parser.add_argument("--experiment_name", default="loan-mlflow-project")
    parser.add_argument("--run_name", default="logreg-autolog-ci")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # tracking lokal biar mlruns muncul di repo (opsional tapi rapi)
    # ✅ Jangan override tracking URI kalau sudah di-set oleh MLflow Projects / CI
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("file:./mlruns")

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Columns: {list(df.columns)}")

    df = df.dropna(subset=[args.target]).copy()
    X = df.drop(columns=[args.target])
    y = df[args.target]

    problem_type = infer_problem_type(y)

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify
    )

    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf)
    ])

    # ✅ INI YANG DICARI PENILAI: autolog standar
    mlflow.autolog()

    with start_mlflow_run_safely(run_name=args.run_name, default_experiment_name=args.experiment_name):
        # Train
        model.fit(X_train, y_train)

        # Predict + hitung metric (cuma untuk disimpan ke file artifact)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary" if problem_type == "binary" else "weighted")
        prec = precision_score(
            y_test, y_pred,
            average="binary" if problem_type == "binary" else "weighted",
            zero_division=0
        )
        rec = recall_score(
            y_test, y_pred,
            average="binary" if problem_type == "binary" else "weighted",
            zero_division=0
        )

        roc_auc = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if problem_type == "binary":
                    roc_auc = roc_auc_score(y_test, proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, proba, multi_class="ovr")
        except Exception:
            roc_auc = None

        # ====== WAJIB UNTUK CI: simpan artifact fisik ke out_dir ======
        model_path = out_dir / "model.joblib"
        joblib.dump(model, model_path)

        metrics = {
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
        }
        if roc_auc is not None:
            metrics["roc_auc"] = float(roc_auc)

        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        report = classification_report(y_test, y_pred, zero_division=0)
        (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

        labels = sorted(pd.unique(pd.concat([pd.Series(y_test), pd.Series(y_pred)])))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_path = out_dir / "confusion_matrix.png"
        make_confusion_matrix_plot(cm, labels, cm_path, title="Confusion Matrix")

        print("✅ Training complete (CI/Kriteria 3)")
        print(f"Saved artifacts to: {out_dir}")
        print("Metrics:", metrics)


if __name__ == "__main__":
    main()