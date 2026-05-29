from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI(title="Customer Churn Prediction API")

# Setup Prometheus Instrumentator untuk mengekspos endpoint /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Setup konfigurasi lokasi tracking MLflow lokal di dalam container
os.environ["MLFLOW_TRACKING_URI"] = "file:///app/mlruns"
run_id = os.environ.get("RUN_ID")

# Load Model MLflow berdasarkan RUN_ID dari GitHub Actions
if run_id:
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
else:
    model = None
    print("Warning: RUN_ID tidak ditemukan.")

@app.get("/")
def ping():
    return {"status": "Healthy"}

@app.post("/invocations")
async def predict(request: Request):
    if not model:
        return {"error": "Model tidak berhasil dimuat."}
    
    data = await request.json()
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}