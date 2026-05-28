# Customer Churn Prediction with MLflow

## Overview

This project implements a machine learning pipeline for customer churn prediction using:

* Python
* Scikit-Learn
* MLflow Tracking
* MLflow Projects
* Docker
* GitHub Actions CI/CD

The workflow automatically:

1. Retrains the model
2. Logs metrics and artifacts to MLflow
3. Saves model artifacts
4. Builds Docker image from MLflow model
5. Pushes Docker image to Docker Hub

---

# Project Structure

```bash
.
├── MLProject/
│   ├── modelling.py
│   ├── MLproject
│   └── conda.yaml
│
├── artifacts/
│   └── latest/
│       ├── model.joblib
│       ├── metrics.json
│       ├── classification_report.txt
│       └── confusion_matrix.png
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── customer_churn_preprocessing.csv
├── mlflow.db
├── README.md
└── requirements.txt
```

---

# Features

* MLflow autologging
* Logistic Regression model
* Automatic preprocessing pipeline
* Confusion matrix generation
* Classification report export
* Docker image build from MLflow artifact
* GitHub Actions CI/CD automation

---

# Tech Stack

* Python 3.11
* Scikit-Learn
* MLflow
* Pandas
* NumPy
* Matplotlib
* Docker
* GitHub Actions

---

# Dataset

The dataset used:

```text
customer_churn_preprocessing.csv
```

Target column:

```text
Churn
```

---

# MLflow Tracking

This project uses MLflow Tracking for:

* Metrics logging
* Parameters logging
* Model artifact storage
* Experiment tracking

Tracking backend:

```text
sqlite:///mlflow.db
```

---

# Run Locally

## 1. Clone repository

```bash
git clone https://github.com/your-username/your-repository.git

cd your-repository
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run training manually

```bash
python MLProject/modelling.py \
  --data customer_churn_preprocessing.csv \
  --target Churn \
  --out_dir artifacts/latest
```

---

# Run with MLflow Projects

```bash
mlflow run ./MLProject \
  --experiment-name customer-churn-project \
  -P data=customer_churn_preprocessing.csv \
  -P target=Churn \
  -P out_dir=../artifacts/latest \
  --env-manager=local
```

---

# Start MLflow UI

```bash
mlflow ui
```

Open browser:

```text
http://127.0.0.1:5000
```

---

# GitHub Actions CI/CD

The CI/CD pipeline automatically:

* trains the model
* stores MLflow artifacts
* builds Docker image
* pushes image to Docker Hub

Workflow file:

```text
.github/workflows/ci.yml
```

---

# Docker Build

Docker image is generated directly from MLflow model artifact:

```bash
mlflow models build-docker \
  --model-uri runs:/RUN_ID/model \
  --name customer-churn-model
```

---

# Docker Run

```bash
docker run -p 5000:8080 customer-churn-model
```

---

# Docker Hub

Before pushing image, configure GitHub Secrets:

| Secret Name        | Description             |
| ------------------ | ----------------------- |
| DOCKERHUB_USERNAME | Docker Hub username     |
| DOCKERHUB_TOKEN    | Docker Hub access token |

---

# MLflow Artifacts

Generated artifacts:

| Artifact                  | Description                    |
| ------------------------- | ------------------------------ |
| model.joblib              | trained model                  |
| metrics.json              | evaluation metrics             |
| classification_report.txt | classification report          |
| confusion_matrix.png      | confusion matrix visualization |

---

# Example Metrics

```json
{
  "accuracy": 0.84,
  "f1": 0.81,
  "precision": 0.79,
  "recall": 0.83,
  "roc_auc": 0.88
}
```

---

# Example Confusion Matrix

Generated automatically after training:

```text
artifacts/latest/confusion_matrix.png
```

---

# Author

Developed for:

* MLflow Tracking
* CI/CD Automation
* Dockerized ML Deployment
* MLOps Practice

---

# License

This project is for educational and portfolio purposes.
