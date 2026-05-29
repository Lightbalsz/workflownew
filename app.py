from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator
import mlflow.pyfunc
import pandas as pd
import os

app = FastAPI(title="Customer Churn Prediction API")

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Path langsung ke folder model yang sudah disalin ke dalam container
MODEL_PATH = "/app/artifacts/latest/model"

print(f"Mencoba memuat model dari: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    print("Model berhasil dimuat!")
else:
    model = None
    print(f"Error: Folder model tidak ditemukan di {MODEL_PATH}")

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